"""
神经 DMD 去噪器 — 数据驱动的自适应模式提取
=============================================
替代简单低通滤波器 (LearnableLowPass → DMDDenoiser)

原理:
  DMD = Koopman 特征值 + 特征向量, 从数据中提取
  KMM = 学习 Koopman 特征值, 通过梯度下降

  两者是不同实现路径的同一个数学对象。

神经 DMD: 用可学习投影替代 SVD, 做到 O(D_k·L) 而非 O(L²·C)
  x → W_proj · x → 稀疏阈值 → W_recon → x_clean

  等价于: 只保留 top-K Koopman 模态的重构
  参数: ~2K (W_proj + W_recon 的秩 K 近似)

优势 vs 简单低通:
  - 低通: 保留所有低频, 丢弃所有高频 → 一刀切, 丢失中频有用信息
  - DMD: 保留"有物理意义"的模态 (无论频率高低) → 自适应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDMDDenoiser(nn.Module):
    """
    神经 DMD 去噪: 可学习投影 + 稀疏阈值 + 重构。

    等价于: 用 rank-K 逼近 Hankel 矩阵 → 保留 K 个主导 Koopman 模态。

    参数:
      n_modes: 保留的主导模态数 (≈ D_k // 2)
      d_input: 输入维度 (96 时间步)
    """
    def __init__(self, n_modes=16, d_input=96):
        super().__init__()
        self.n_modes = n_modes

        # 投影: L → 2*n_modes (实部和虚部 → 复模态)
        self.W_proj = nn.Linear(d_input, 2 * n_modes, bias=False)

        # 重构: 2*n_modes → L
        self.W_recon = nn.Linear(2 * n_modes, d_input, bias=False)

        # 软阈值 (可学习)
        self.log_thresh = nn.Parameter(torch.tensor(0.0))  # 0 → softplus(0)≈0.7

        # 正交初始化 (接近 DMD 的正交投影)
        nn.init.orthogonal_(self.W_proj.weight)
        nn.init.orthogonal_(self.W_recon.weight)

    def forward(self, x):
        # x: (B*C, L)
        BxC, L = x.shape

        # 投影到模态空间
        modes = self.W_proj(x)  # (B*C, 2*n_modes)

        # 软阈值: 小幅值 → 0 (噪声), 大幅值 → 保留 (信号)
        # eps 防止 mode_amp=0 时 sqrt 的梯度产生 NaN (1/(2*sqrt(0)) → Inf → 0*Inf=NaN)
        mode_amp = (modes.pow(2).sum(dim=-1, keepdim=True) + 1e-8).sqrt()  # 模长
        thresh = F.softplus(self.log_thresh)  # > 0
        gate = torch.sigmoid((mode_amp - thresh) * 10.0)  # 软门控
        modes_sparse = modes * gate

        # 重构干净信号
        x_clean = self.W_recon(modes_sparse)

        # 残差连接: 保留原始信号的整体结构
        alpha = torch.sigmoid(torch.tensor(0.8))  # ~0.7 信号 + 0.3 原始
        return alpha * x_clean + (1 - alpha) * x
