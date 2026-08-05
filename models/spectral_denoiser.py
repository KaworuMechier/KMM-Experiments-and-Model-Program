"""
可学习低通预滤波器 — 去噪骨架提取
=================================
在 RevIN 之后, Token Embed 之前:
  FFT → 保留低频(可学习 cutoff) → iFFT
  参数: 1 (cutoff ratio)
"""

import math
import torch
import torch.nn as nn


class LearnableLowPass(nn.Module):
    """
    可学习低通滤波器: 只保留前 r% 的低频成分。

    参数:
      logit_cutoff: sigmoid → [0.1, 0.9] 控制保留多少低频

    物理意义:
      ETT 数据的高频分量 ≈ 传感器噪声 + 不稳定的瞬时波动
      只让低频骨架 (日/周周期) 进入模型 → 混频在干净信号上操作
      → 拍频产生有意义的调制包络, 而非噪声的伪包络
    """
    def __init__(self, logit_cutoff=1.0):  # sigmoid(1.0) ≈ 0.73
        super().__init__()
        self.logit_cutoff = nn.Parameter(torch.tensor(logit_cutoff))

    def forward(self, x):
        # x: (B*C, L, 1) or (B*C, 1, L)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B*C, L)
            was_token = True
        else:
            was_token = False

        BxC, L = x.shape
        n_freq = L // 2 + 1

        # 频域滤波
        X = torch.fft.rfft(x.float(), dim=1)  # (B*C, n_freq)

        # 软截止: sigmoid → [0.1, 0.9]
        cutoff = torch.sigmoid(self.logit_cutoff) * 0.8 + 0.1
        cutoff_bin = int(cutoff.item() * n_freq)

        # 高斯衰减窗 (而不是硬截止)
        t = torch.arange(n_freq, device=x.device).float()
        sigma = cutoff_bin / 3.0
        window = torch.exp(-0.5 * ((t - cutoff_bin).clamp(min=0) / sigma) ** 2)
        window[:cutoff_bin] = 1.0  # 低频全保留

        X_filtered = X * window.unsqueeze(0)
        x_clean = torch.fft.irfft(X_filtered, n=L, dim=1)

        if was_token:
            x_clean = x_clean.unsqueeze(1)
        return x_clean
