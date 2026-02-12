import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.model_selection import KFold
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- 工具函数 ----------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------- 动态图卷积层（论文公式2、3，图4）-----------------------
class DynamicGCNLayer(nn.Module):
    """基于样本特征差动态构建全连接图，执行GCN卷积，并逐层更新邻接矩阵"""
    def __init__(self, in_dim, out_dim, hidden_dim=128, use_fixed_adj=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.conv = GCNConv(in_dim, out_dim)
        self.use_fixed_adj = use_fixed_adj  # 用于消融实验：是否使用固定图

    def forward(self, x, adj=None):
        """
        x: [N, in_dim]
        adj: 若为None，则基于当前x动态计算邻接矩阵；否则使用传入的固定邻接矩阵（消融实验）
        返回: [N, out_dim], 邻接矩阵[N, N]
        """
        if adj is None or not self.use_fixed_adj:
            # 论文公式(2)：动态计算邻接矩阵
            N = x.size(0)
            x1 = x.unsqueeze(1)  # [N,1,in_dim]
            x2 = x.unsqueeze(0)  # [1,N,in_dim]
            diff = torch.abs(x1 - x2)  # [N,N,in_dim]
            diff_flat = diff.view(-1, diff.size(-1))
            adj_flat = self.mlp(diff_flat).squeeze()  # [N*N]
            adj = adj_flat.view(N, N)
            adj = torch.sigmoid(adj)  # 边权归一化到(0,1)

        edge_index, edge_weight = dense_to_sparse(adj)
        x = self.conv(x, edge_index, edge_weight)
        # 论文公式(3)：使用Leaky ReLU非线性激活
        x = F.leaky_relu(x, negative_slope=0.2)
        return x, adj

# ---------------------- 模态共享/特定网络（堆叠动态GCN，每层独立计算邻接矩阵）-----------------------
class ModalityNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, num_layers=2, use_fixed_adj=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        cur_dim = in_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else self.out_dim
            self.layers.append(DynamicGCNLayer(cur_dim, next_dim, hidden_dim, use_fixed_adj))
            cur_dim = next_dim
        self.use_fixed_adj = use_fixed_adj

    def forward(self, x):
        """每层独立计算邻接矩阵（动态图迭代更新）"""
        for layer in self.layers:
            # 不传递上一层的邻接矩阵，每层都重新计算（论文图4）
            x, _ = layer(x, None)
        return x

# ---------------------- 多模态注意力（严格按论文公式8/9）-----------------------
class MultimodalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_sh = nn.Linear(hidden_dim, 1)   # 计算α_sh
        self.W_a  = nn.Linear(hidden_dim, 1)   # 计算α_sp^a，输入为Z_sp_a
        self.W_e  = nn.Linear(hidden_dim, 1)   # 计算α_sp^e，输入为Z_sp_e

    def forward(self, Z_sh, Z_sp_a, Z_sp_e):
        """
        输入:
            Z_sh : [N, hidden_dim]  共享特征（平均）
            Z_sp_a: [N, hidden_dim]  音频特定特征
            Z_sp_e: [N, hidden_dim]  EEG特定特征
        返回: 融合特征 [N, hidden_dim]  公式(9)
        """
        alpha_sh = torch.tanh(self.W_sh(Z_sh))        # [N,1]
        alpha_a  = torch.tanh(self.W_a(Z_sp_a))       # [N,1]  论文公式(8)：基于Z_sp_a
        alpha_e  = torch.tanh(self.W_e(Z_sp_e))       # [N,1]  基于Z_sp_e
        fused = alpha_sh * Z_sh + alpha_a * Z_sp_a + alpha_e * Z_sp_e
        return fused

# ---------------------- 完整MS²-GNN（论文对齐版）-----------------------
class MS2GNN(nn.Module):
    def __init__(self, audio_dim, eeg_dim, hidden_dim=128, num_classes=2, use_fixed_adj=False):
        """
        audio_dim: tuple (time_steps, freq_bins) 音频原始维度
        eeg_dim:   tuple (time_steps, electrodes) EEG原始维度
        hidden_dim: LSTM隐层维度及最终嵌入维度
        use_fixed_adj: 是否使用固定图（消融实验），默认False即动态图
        """
        super().__init__()
        # ---------- 1. LSTM特征提取（输出整个序列）----------
        self.audio_lstm = nn.LSTM(
            input_size=audio_dim[1],
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.eeg_lstm = nn.LSTM(
            input_size=eeg_dim[1],
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        audio_flat_dim = audio_dim[0] * hidden_dim   # T_a * hidden
        eeg_flat_dim   = eeg_dim[0]   * hidden_dim   # T_e * hidden

        # ---------- 2. 模态共享网络（GCN）----------
        self.shared_net_a = ModalityNetwork(audio_flat_dim, hidden_dim, out_dim=hidden_dim, use_fixed_adj=use_fixed_adj)
        self.shared_net_e = ModalityNetwork(eeg_flat_dim,   hidden_dim, out_dim=hidden_dim, use_fixed_adj=use_fixed_adj)

        # ---------- 3. 模态特定网络（GCN）----------
        self.specific_net_a = ModalityNetwork(audio_flat_dim, hidden_dim, out_dim=hidden_dim, use_fixed_adj=use_fixed_adj)
        self.specific_net_e = ModalityNetwork(eeg_flat_dim,   hidden_dim, out_dim=hidden_dim, use_fixed_adj=use_fixed_adj)

        # ---------- 4. 重建网络（GCN解码器）----------
        self.recon_net_a = ModalityNetwork(2 * hidden_dim, hidden_dim, out_dim=audio_flat_dim, use_fixed_adj=use_fixed_adj)
        self.recon_net_e = ModalityNetwork(2 * hidden_dim, hidden_dim, out_dim=eeg_flat_dim,   use_fixed_adj=use_fixed_adj)

        # ---------- 5. 注意力融合（严格论文公式8/9）----------
        self.attention = MultimodalAttention(hidden_dim)

        # ---------- 6. 分类器（最终融合特征）----------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # ---------- 7. 辅助分类器（LSTM输出特征）----------
        self.audio_cls = nn.Linear(audio_flat_dim, num_classes)
        self.eeg_cls   = nn.Linear(eeg_flat_dim,   num_classes)

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

    def forward(self, audio, eeg, label=None, lambda_=0.1, gamma=0.1, zeta=0.05):
        """
        audio: [N, T_a, F_a]
        eeg:   [N, T_e, F_e]
        返回: logits, losses (dict)
        """
        # ---------- LSTM特征提取 ----------
        audio_lstm_out, _ = self.audio_lstm(audio)   # [N, T_a, hidden]
        eeg_lstm_out, _   = self.eeg_lstm(eeg)       # [N, T_e, hidden]

        Z_a = audio_lstm_out.reshape(audio_lstm_out.size(0), -1)  # [N, T_a * hidden]
        Z_e = eeg_lstm_out.reshape(eeg_lstm_out.size(0), -1)      # [N, T_e * hidden]

        # ---------- 共享特征 ----------
        Z_sh_a = self.shared_net_a(Z_a)            # [N, hidden]
        Z_sh_e = self.shared_net_e(Z_e)            # [N, hidden]
        Z_sh   = (Z_sh_a + Z_sh_e) / 2            # 公式(5)

        # ---------- 特定特征 ----------
        Z_sp_a = self.specific_net_a(Z_a)          # [N, hidden]
        Z_sp_e = self.specific_net_e(Z_e)          # [N, hidden]

        # ---------- 重建（GCN解码）----------
        cat_a = torch.cat([Z_sh_a, Z_sp_a], dim=1)  # [N, 2*hidden]
        cat_e = torch.cat([Z_sh_e, Z_sp_e], dim=1)
        Z_hat_a = self.recon_net_a(cat_a)          # [N, T_a * hidden]
        Z_hat_e = self.recon_net_e(cat_e)          # [N, T_e * hidden]

        # ---------- 注意力融合（论文公式8/9）----------
        fused = self.attention(Z_sh, Z_sp_a, Z_sp_e)  # 注意：传入Z_sp_a/e，不是Z_sh_a/e

        # ---------- 分类 ----------
        logits_fused = self.classifier(fused)      # [N, num_classes]
        logits_audio = self.audio_cls(Z_a)         # [N, num_classes]
        logits_eeg   = self.eeg_cls(Z_e)           # [N, num_classes]

        # ---------- 损失计算（论文公式10-14）----------
        losses = {}
        if label is not None:
            # 1. 分类损失 + KL散度（公式11）
            ce_loss_a = F.cross_entropy(logits_audio, label)
            ce_loss_e = F.cross_entropy(logits_eeg, label)
            ce_loss_f = F.cross_entropy(logits_fused, label)

            log_probs_fused = F.log_softmax(logits_fused, dim=1)
            probs_audio = F.softmax(logits_audio, dim=1)
            probs_eeg   = F.softmax(logits_eeg, dim=1)
            kl_div_a = F.kl_div(log_probs_fused, probs_audio, reduction='batchmean')
            kl_div_e = F.kl_div(log_probs_fused, probs_eeg,   reduction='batchmean')

            cls_loss = ce_loss_a + ce_loss_e + ce_loss_f + kl_div_a + kl_div_e

            # 2. 相似损失（公式12）
            sim_loss = F.mse_loss(Z_sh_a, Z_sh_e)

            # 3. 差异损失（公式13）- 正交约束
            diff_loss_a = torch.norm(torch.mm(Z_sh_a.t(), Z_sp_a), p='fro') ** 2
            diff_loss_e = torch.norm(torch.mm(Z_sh_e.t(), Z_sp_e), p='fro') ** 2
            diff_loss = diff_loss_a + diff_loss_e

            # 4. 重建损失（公式14）
            recon_loss = F.mse_loss(Z_hat_a, Z_a) + F.mse_loss(Z_hat_e, Z_e)

            # 总损失（公式10）
            total_loss = cls_loss + lambda_ * sim_loss + gamma * diff_loss + zeta * recon_loss

            losses = {
                'total': total_loss,
                'cls': cls_loss,
                'sim': sim_loss,
                'diff': diff_loss,
                'recon': recon_loss
            }

        return logits_fused, losses

# ---------------------- 五折交叉验证训练（支持动态图/固定图开关）-----------------------
def train_and_evaluate(use_fixed_adj=False):   # 新增参数：是否使用固定图（消融实验）
    # ========== 数据加载（与原代码一致）==========
    fmri_flat = np.load('./fMri_pcc.npy')  # (1015, 90, 90)
    fmri_5 = fmri_flat.reshape(203, 5, 90, 90)
    fmri_10 = np.repeat(fmri_5, repeats=2, axis=1)
    brain_data = fmri_10.reshape(2030, 90, 90)

    audio_10 = np.load('stft_data_fixed_10/all_stfts_64x640.npy')
    audio_data = audio_10.reshape(2030, 64, 640)

    path_brain = '/home/idal-01/code/IBGNN-master/datasets/HIV.mat'
    f_brain = loadmat(path_brain)
    labels_203 = f_brain['label'].squeeze()
    labels = np.repeat(labels_203, repeats=10)

    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    brain_tensor = torch.tensor(brain_data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Fixed graph mode: {use_fixed_adj}")  # 打印图模式

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(audio_tensor)):
        print(f'\n=== Fold {fold + 1}/5 ===')

        X_train_audio, X_test_audio = audio_tensor[train_idx], audio_tensor[test_idx]
        X_train_brain, X_test_brain = brain_tensor[train_idx], brain_tensor[test_idx]
        y_train, y_test = label_tensor[train_idx], label_tensor[test_idx]

        model = MS2GNN(
            audio_dim=(64, 640),
            eeg_dim=(90, 90),
            hidden_dim=32,
            num_classes=2,
            use_fixed_adj=use_fixed_adj   # 传入固定图开关
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True)

        best_f1 = 0
        best_metrics = None
        no_improve = 0
        patience = 15
        batch_size = 16
        epochs = 100

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            epoch_loss = 0

            num_batches = int(np.ceil(len(X_train_audio) / batch_size))
            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, len(X_train_audio))

                batch_audio = X_train_audio[start:end].to(device)
                batch_eeg   = X_train_brain[start:end].to(device)
                batch_label = y_train[start:end].to(device)

                optimizer.zero_grad()
                logits, losses = model(batch_audio, batch_eeg, batch_label,
                                       lambda_=0.1, gamma=0.1, zeta=0.05)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += losses['total'].item()

            avg_loss = epoch_loss / num_batches
            scheduler.step(avg_loss)

            # 验证
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                num_test_batches = int(np.ceil(len(X_test_audio) / batch_size))
                for b in range(num_test_batches):
                    start = b * batch_size
                    end = min((b + 1) * batch_size, len(X_test_audio))
                    test_audio = X_test_audio[start:end].to(device)
                    test_eeg   = X_test_brain[start:end].to(device)
                    test_label = y_test[start:end]

                    logits, _ = model(test_audio, test_eeg)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(test_label.numpy())

            acc = metrics.accuracy_score(all_labels, all_preds)
            prec = metrics.precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            rec = metrics.recall_score(all_labels, all_preds, average='weighted')
            f1 = metrics.f1_score(all_labels, all_preds, average='weighted')

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = (acc, prec, rec, f1)
                no_improve = 0
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Best F1: {best_f1:.4f}')

            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        results.append(best_metrics)
        print(f'Fold {fold+1} Best - Acc: {best_metrics[0]:.4f}, Prec: {best_metrics[1]:.4f}, Rec: {best_metrics[2]:.4f}, F1: {best_metrics[3]:.4f}')

    if results:
        avg_metrics = np.mean(results, axis=0)
        print('\n=== Final Results ===')
        print(f'Average Acc: {avg_metrics[0]:.4f}')
        print(f'Average Precision: {avg_metrics[1]:.4f}')
        print(f'Average Recall: {avg_metrics[2]:.4f}')
        print(f'Average F1: {avg_metrics[3]:.4f}')
        np.save("cross_validation_results.npy", np.array(results))
    else:
        print("No results to average.")

if __name__ == '__main__':
    setup_seed(123)
    # 默认使用动态图（use_fixed_adj=False），如需固定图消融实验可改为True
    train_and_evaluate(use_fixed_adj=False)