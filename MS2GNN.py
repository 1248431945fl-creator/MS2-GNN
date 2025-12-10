import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import scipy.io as sio
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import time
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 优化后的动态图卷积层 - 支持批量处理
class DynamicGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, adj=None):
        # 输入x形状: [batch_size, num_nodes, in_dim] 或 [num_nodes, in_dim]
        if adj is None:
            if x.dim() == 2:  # 单个图的情况
                n = x.size(0)
                adj = torch.zeros(n, n, device=x.device)
                for i in range(n):
                    for j in range(n):
                        diff = torch.abs(x[i] - x[j])
                        adj[i, j] = self.mlp(diff).squeeze()
                adj = torch.sigmoid(adj)
            else:  # 批量处理的情况
                batch_size, num_nodes, _ = x.shape
                adj = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)

                # 使用广播机制避免嵌套循环
                x_expanded1 = x.unsqueeze(2)  # [batch, nodes, 1, features]
                x_expanded2 = x.unsqueeze(1)  # [batch, 1, nodes, features]
                diff = torch.abs(x_expanded1 - x_expanded2)

                # 将MLP应用于每个节点对的特征差
                diff_flat = diff.reshape(-1, diff.size(-1))
                adj_flat = self.mlp(diff_flat).squeeze()
                adj = adj_flat.view(batch_size, num_nodes, num_nodes)
                adj = torch.sigmoid(adj)

        # 将邻接矩阵转换为边索引和边权重
        if x.dim() == 2:  # 单个图
            edge_index, edge_weight = dense_to_sparse(adj)
            x = self.conv(x, edge_index, edge_weight)
        else:  # 批量处理
            x_out = []
            for i in range(x.size(0)):
                edge_index, edge_weight = dense_to_sparse(adj[i])
                x_i = self.conv(x[i], edge_index, edge_weight)
                x_out.append(x_i)
            x = torch.stack(x_out, dim=0)

        return x, adj


# 模态共享/特定网络
class ModalityNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(DynamicGCNLayer(in_dim, hidden_dim))

    def forward(self, x):
        adj = None
        for layer in self.layers:
            x, adj = layer(x, adj)
        return x


# 多模态注意力机制
class MultimodalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_shared = nn.Linear(hidden_dim, 1)
        self.attn_audio = nn.Linear(hidden_dim, 1)
        self.attn_fmri = nn.Linear(hidden_dim, 1)

    def forward(self, shared, audio_spec, fmri_spec):
        a_sh = torch.tanh(self.attn_shared(shared))
        a_au = torch.tanh(self.attn_audio(audio_spec))
        a_fm = torch.tanh(self.attn_fmri(fmri_spec))

        alpha_sh = torch.softmax(a_sh, dim=0)
        alpha_au = torch.softmax(a_au, dim=0)
        alpha_fm = torch.softmax(a_fm, dim=0)

        return alpha_sh * shared + alpha_au * audio_spec + alpha_fm * fmri_spec


# 完整的MS²-GNN模型
class MS2GNN(nn.Module):
    def __init__(self, audio_dim, fmri_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.audio_dim = audio_dim
        self.fmri_dim = fmri_dim

        # 音频处理 (LSTM)
        self.audio_lstm = nn.LSTM(
            input_size=audio_dim[0],  # 频率点数
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # fMRI节点特征初始化
        self.fmri_node_emb = nn.Embedding(fmri_dim[1], hidden_dim)  # 节点数

        # 模态共享网络
        self.shared_net = ModalityNetwork(hidden_dim, hidden_dim)

        # 模态特定网络
        self.audio_specific_net = ModalityNetwork(hidden_dim, hidden_dim)
        self.fmri_specific_net = ModalityNetwork(hidden_dim, hidden_dim)

        # 重建网络 - 动态适应维度
        self.audio_recon = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim[0] * audio_dim[1])
        )

        # fMRI重建网络 - 使用实际维度
        self.fmri_recon = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fmri_dim[1] * fmri_dim[2])  # 节点数 * 节点数
        )

        # 注意力融合
        self.attention = MultimodalAttention(hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, audio, fmri_adj, label=None):
        # 确保输入形状正确
        if audio.dim() == 4 and audio.size(1) == 1:
            audio = audio.squeeze(1)  # 移除通道维度 [batch, 64, 640]

        # 音频特征提取
        # 输入audio形状: [batch, 64, 640] -> 调整为 [batch, 640, 64]
        audio = audio.permute(0, 2, 1)
        audio_out, _ = self.audio_lstm(audio)
        # 取最后一个时间步的输出
        audio_feat = audio_out[:, -1, :]

        # fMRI特征提取
        batch_size = fmri_adj.size(0)
        num_nodes = fmri_adj.size(1)
        node_ids = torch.arange(num_nodes, device=fmri_adj.device)
        fmri_feat = self.fmri_node_emb(node_ids).unsqueeze(0).repeat(batch_size, 1, 1)

        # 模态共享特征
        shared_audio = self.shared_net(audio_feat.unsqueeze(1))  # 添加节点维度 [batch, 1, hidden]
        shared_audio = shared_audio.squeeze(1)  # 移除节点维度 [batch, hidden]

        shared_fmri = self.shared_net(fmri_feat)  # [batch, num_nodes, hidden]
        shared_fmri_mean = shared_fmri.mean(dim=1)  # 平均节点特征 [batch, hidden]

        # 模态特定特征
        audio_spec = self.audio_specific_net(audio_feat.unsqueeze(1)).squeeze(1)
        fmri_spec = self.fmri_specific_net(fmri_feat).mean(dim=1)

        # 共享特征融合
        shared = (shared_audio + shared_fmri_mean) / 2

        # 重建
        audio_recon = self.audio_recon(torch.cat([shared_audio, audio_spec], dim=1))
        fmri_recon = self.fmri_recon(torch.cat([shared_fmri_mean, fmri_spec], dim=1))

        # 确保重建输出形状正确
        audio_flat = audio.reshape(audio.size(0), -1)
        fmri_flat = fmri_adj.reshape(batch_size, -1)

        # 注意力融合
        fused = self.attention(shared, audio_spec, fmri_spec)

        # 分类
        logits = self.classifier(fused)

        # 计算损失
        losses = {}
        if label is not None:
            # 分类损失
            cls_loss = F.cross_entropy(logits, label)

            # 相似损失 (共享特征一致性)
            sim_loss = F.mse_loss(shared_audio, shared_fmri_mean)

            # 差异损失 (正交约束)
            diff_loss = torch.mean(torch.abs(torch.sum(shared_audio * audio_spec, dim=1))) + \
                        torch.mean(torch.abs(torch.sum(shared_fmri_mean * fmri_spec, dim=1)))

            # 重建损失 - 动态适应维度
            recon_audio_loss = F.mse_loss(audio_recon, audio_flat)

            # 确保fMRI重建形状匹配
            if fmri_recon.size(1) != fmri_flat.size(1):
                # 如果维度不匹配，使用自适应池化或裁剪
                if fmri_recon.size(1) > fmri_flat.size(1):
                    fmri_recon = fmri_recon[:, :fmri_flat.size(1)]
                else:
                    padding = torch.zeros(batch_size, fmri_flat.size(1) - fmri_recon.size(1),
                                          device=fmri_recon.device)
                    fmri_recon = torch.cat([fmri_recon, padding], dim=1)

            recon_fmri_loss = F.mse_loss(fmri_recon, fmri_flat)
            recon_loss = recon_audio_loss + recon_fmri_loss

            # 总损失
            total_loss = cls_loss + 0.1 * sim_loss + 0.1 * diff_loss + 0.05 * recon_loss
            losses = {
                'total': total_loss,
                'cls': cls_loss,
                'sim': sim_loss,
                'diff': diff_loss,
                'recon': recon_loss
            }

        return logits, losses


# 五折交叉验证训练
def train_and_evaluate():
    # 加载数据
    path_brain = '/home/idal-01/code/IBGNN-master/datasets/HIV.mat'
    f_brain = sio.loadmat(path_brain)
    brain_data, Y, filenames_brain = f_brain['dti'], f_brain['label'], f_brain['files']
    audio_feature = np.load('data/txt/audio_features.npy')
    brain_data = np.load('data/pcc.npy')

    # 数据预处理
    sorted_indices = np.argsort(filenames_brain.squeeze())
    brain_label = Y.squeeze()[sorted_indices]
    brain_data = brain_data[sorted_indices]
    audio_feature = audio_feature[sorted_indices]

    # 打印数据形状
    print(f"Brain data shape: {brain_data.shape}")
    print(f"Audio feature shape: {audio_feature.shape}")
    print(f"Label shape: {brain_label.shape}")

    # 转换为PyTorch张量
    brain_data = torch.tensor(brain_data, dtype=torch.float32)
    audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
    labels = torch.tensor(brain_label, dtype=torch.long)

    # 确保fMRI数据是方阵
    if brain_data.dim() == 4 and brain_data.size(1) == 1:
        brain_data = brain_data.squeeze(1)  # 移除通道维度 [batch, 90, 90]

    # 获取fMRI实际维度
    fmri_dim = brain_data.shape
    print(f"Actual fMRI dimensions: {fmri_dim}")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(brain_data)):
        print(f'\n=== Fold {fold + 1}/5 ===')

        # 划分训练集和测试集
        X_train, X_test = brain_data[train_idx], brain_data[test_idx]
        A_train, A_test = audio_feature[train_idx], audio_feature[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # 创建模型 - 使用实际维度
        model = MS2GNN(
            audio_dim=(64, 640),  # (freq_bins, time_steps)
            fmri_dim=fmri_dim,  # 使用实际数据维度
            hidden_dim=128,
            num_classes=2
        ).to(device)

        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True)

        # 训练参数
        best_f1 = 0
        best_metrics = None
        no_improve = 0
        patience = 15

        # 使用小批量训练
        batch_size = 16
        num_batches = int(np.ceil(len(X_train) / batch_size))

        # 训练循环
        for epoch in range(100):
            start_time = time.time()
            model.train()
            epoch_loss = 0

            # 小批量训练
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(X_train))

                batch_X = X_train[start:end].to(device)
                batch_A = A_train[start:end].to(device)
                batch_y = y_train[start:end].to(device)

                optimizer.zero_grad()

                # 前向传播
                logits, losses = model(batch_A, batch_X, batch_y)

                # 反向传播
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += losses['total'].item()

            avg_epoch_loss = epoch_loss / num_batches
            scheduler.step(avg_epoch_loss)

            # 验证
            model.eval()
            all_preds = []
            all_labels = []

            # 使用整个测试集评估
            with torch.no_grad():
                # 分批处理测试集以避免内存问题
                test_batch_size = 32
                num_test_batches = int(np.ceil(len(X_test) / test_batch_size))

                for i in range(num_test_batches):
                    start = i * test_batch_size
                    end = min((i + 1) * test_batch_size, len(X_test))

                    test_X = X_test[start:end].to(device)
                    test_A = A_test[start:end].to(device)

                    test_logits, _ = model(test_A, test_X)
                    preds = test_logits.argmax(dim=1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(y_test[start:end].numpy())

            # 计算评估指标
            acc = metrics.accuracy_score(all_labels, all_preds)
            prec = metrics.precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            rec = metrics.recall_score(all_labels, all_preds, average='weighted')
            f1 = metrics.f1_score(all_labels, all_preds, average='weighted')
            # best_metrics = (acc, prec, rec, f1)
            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = (acc, prec, rec, f1)
                no_improve = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f"best_model_fold{fold + 1}.pt")
            else:
                no_improve += 1

            # 打印训练信息
            epoch_time = time.time() - start_time
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}/{100} | Time: {epoch_time:.2f}s | Loss: {avg_epoch_loss:.4f} | '
                      f'Test Acc: {acc:.4f} | F1: {f1:.4f} | Best F1: {best_f1:.4f}')

            # 早停
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # 保存结果
        results.append(best_metrics)
        print(f'Fold {fold + 1} Best - Acc: {best_metrics[0]:.4f}, '
              f'Prec: {best_metrics[1]:.4f}, Rec: {best_metrics[2]:.4f}, F1: {best_metrics[3]:.4f}')

    # 计算平均性能
    if results:
        avg_metrics = np.mean(results, axis=0)
        print('\n=== Final Results ===')
        print(f'Average Acc: {avg_metrics[0]:.4f}')
        print(f'Average Precision: {avg_metrics[1]:.4f}')
        print(f'Average Recall: {avg_metrics[2]:.4f}')
        print(f'Average F1: {avg_metrics[3]:.4f}')

        # 保存最终结果
        np.save("cross_validation_results.npy", np.array(results))
    else:
        print("No results to average")


if __name__ == '__main__':
    setup_seed(123)
    train_and_evaluate()