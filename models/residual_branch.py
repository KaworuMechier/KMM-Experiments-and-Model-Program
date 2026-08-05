"""
Track B: 残差补偿分支 (Residual Track)
========================================
职责: 捕捉局部非线性突变、高频尖峰。
输入: 仅经过 RevIN 的原始数据 (保留全部高频信息)。
核心: Multi-Scale Gated Conv (3/7/15 分组卷积 + GLU 门控)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit: 一半通道做激活, 一半做门控."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * F.sigmoid(gate)


class MultiScaleGatedConv(nn.Module):
    """
    多尺度门控卷积: 3 个并行分支 (k=3,7,15) → GLU → 融合。

    TimesNet 用 2D Inception Conv 捕获多尺度。这里用 1D 分组卷积 + GLU——
    分组卷积保证 CI (通道独立), GLU 自动过滤纯随机白噪声。
    """
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.conv7 = nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model)
        self.conv15 = nn.Conv1d(d_model, d_model, 15, padding=7, groups=d_model)

        self.glu3 = GLU(d_model)
        self.glu7 = GLU(d_model)
        self.glu15 = GLU(d_model)

        self.fuse = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B*C, L, D) — 时间序列在潜空间
        # Conv1d in: (B*C, D, L)
        x_t = x.permute(0, 2, 1)  # (B*C, D, L)

        # 三个尺度并行
        c3 = self.conv3(x_t).permute(0, 2, 1)  # (B*C, L, D)
        c7 = self.conv7(x_t).permute(0, 2, 1)
        c15 = self.conv15(x_t).permute(0, 2, 1)

        # GLU 门控
        g3 = self.glu3(c3)
        g7 = self.glu7(c7)
        g15 = self.glu15(c15)

        # 融合
        out = torch.cat([g3, g7, g15], dim=-1)  # (B*C, L, 3D)
        out = self.fuse(out)                      # (B*C, L, D)
        out = self.dropout(out)
        return self.norm(out + x)  # 残差


class ResidualTrack(nn.Module):
    """
    残差补偿分支: RevIN → Conv1d TokenEmbed → MultiScaleGatedConv × 2 → Decoder → Y_local
    参数: ~15K
    """
    def __init__(self, d_model=64, pred_len=96, seq_len=96, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token Embedding (和 Track A 并行, 但只用 Conv 编码原始信号)
        self.token_embed = nn.Conv1d(1, d_model, kernel_size=7, padding=3)

        # 多尺度门控卷积层
        self.conv_blocks = nn.ModuleList([
            MultiScaleGatedConv(d_model, dropout) for _ in range(n_layers)
        ])

        # 解码器
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.to_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B*C, 1, L) — 仅 RevIN 后的原始信号, 无 DMD 去噪
        z = self.token_embed(x).permute(0, 2, 1)  # (B*C, L, d_model)

        for conv in self.conv_blocks:
            z = conv(z)

        # 解码
        y = z.permute(0, 2, 1)       # (B*C, d_model, L)
        y = self.temporal_proj(y)     # (B*C, d_model, H)
        y = y.permute(0, 2, 1)         # (B*C, H, d_model)
        y = self.to_out(y).squeeze(-1)  # (B*C, H)

        return y, z  # 返回预测 + 潜特征 (用于正交正则化)


class AdaptiveGate(nn.Module):
    """
    自适应上下文门控: 根据当前窗口的波动程度动态调节 α(t)。

    平稳期 → α→0: Y = Y_smooth (完全信任 KMM)
    剧烈波动 → α→1: Y = Y_smooth + Y_local (全力补偿)

    实现: 每通道的波动标准差 → MLP → sigmoid
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )

    def forward(self, h_smooth, h_local):
        # h_smooth, h_local: (B*C, L, D) — 可能不同 D
        smooth_std = h_smooth.std(dim=(1, -1))  # (B*C,)
        local_std = h_local.std(dim=(1, -1))
        feat = torch.stack([smooth_std, local_std], dim=-1)  # (B*C, 2)
        alpha = torch.sigmoid(self.proj(feat))  # (B*C, 1)
        return alpha
