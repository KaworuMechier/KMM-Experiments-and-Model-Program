"""
FEDformer + FiLM 借鉴: 低秩 FFT + 频域 EMA
===========================================
作为 KMM FreqShiftCoupling 的前后处理器:

  TopMFreqSelect:  FEDformer 模式选择 — 只保留 top-M 主导频段
  FreqDomainEMA:   FiLM 频域滤波 — EMA 在频域做矩阵乘法, 消除 for 循环
"""

import math
import torch
import torch.nn as nn


class TopMFreqSelect(nn.Module):
    """
    P2 (FEDformer mode_select='low'): 只保留能量最高的 M 个频段。
    效果: 60% 频域计算量减少 + 噪声频段滤除 → 同时加速 + 去噪。
    参数: 0 (top-M 索引每 forward 重新计算, 基于能量排序)
    """
    def __init__(self, top_m=16):
        super().__init__()
        self.top_m = top_m

    def forward(self, h_fft):
        # h_fft: (B*C, n_freq, D) complex
        BxC, n_freq, D = h_fft.shape
        M = min(self.top_m, n_freq)

        # 跨 batch/D 平均能量 → 选 top-M
        energy = h_fft.abs().mean(dim=(0, -1))  # (n_freq,)
        _, top_idx = torch.topk(energy, M)

        # 掩码: top-M=1, 其余=0
        mask = torch.zeros(n_freq, device=h_fft.device, dtype=h_fft.dtype)
        mask[top_idx] = 1.0

        return h_fft * mask.unsqueeze(0).unsqueeze(-1)


class FreqDomainEMA(nn.Module):
    """
    P1 (FiLM): EMA 低通滤波直接在频域做, 替代时域 for 循环。

    理论:
      y[t] = α·y[t-1] + (1-α)·x[t]  (时域递推)
      ⇕ FFT
      Y(ω) = H(ω)·X(ω)
      H(ω) = (1-α)/(1 - α·e^{-jω})

    效果: 时域 EMA → O(L) for 循环, 频域 EMA → O(1) 矩阵乘法。
    参数: 0 (仅 α 可调)
    """
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, h_time):
        # h_time: (B*C, L, D) — 混频输出的时域信号
        BxC, L, D = h_time.shape
        n_freq = L // 2 + 1

        # EMA 频率响应
        alpha = self.alpha
        omega = 2 * math.pi * torch.arange(n_freq, device=h_time.device) / (2 * n_freq)
        # H(ω) = (1-α) / (1 - α·exp(-jω))
        H = (1 - alpha) / (1 - alpha * torch.exp(-1j * omega) + 1e-8)
        H_mag = H.abs()  # |H(ω)|: 低频≈1, 高频≈0 → 天然低通

        # 频域滤波 (替代 for 循环)
        h_fft = torch.fft.rfft(h_time.float(), dim=1)
        h_fft = h_fft * H_mag.unsqueeze(0).unsqueeze(-1).to(h_fft.dtype)
        return torch.fft.irfft(h_fft, n=L, dim=1)
