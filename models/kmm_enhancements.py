"""
Hilbert 包络调制 + 自适应混频 + 卡尔曼门控
===========================================
三个零/低参数组件, 全部可独立消融:
  HilbertAmpMod:  0 params — 每模态的瞬时振幅包络 → AM 调制
  AdaptiveMixer:  ~200 params — 输入依赖的 α 权重替代固定值
  KalmanGate:     ~2K params — Kalman 增益替代固定 EMA α=0.85
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HilbertAmpMod(nn.Module):
    """
    Hilbert 变换 → 解析信号 → 瞬时振幅包络 → AM 调制。

    物理意义:
      模态 d 振荡在频率 θ_d。但振幅不是恒定的——早晚高峰大, 深夜小。
      Hilbert 包络 a_d(t) = |analytic(h_d(t))| 捕获这一刻的瞬时强度。
      h_d(t) = a_d(t) · h_d(t) → 混频时强振幅时刻产生更强的拍频。

    实现: 频域 Hilbert 滤波器 (无学习参数)
      H(ω) = -j·sign(ω)  → 解析信号 = x + j·H{x}
      包络 = |x + j·H{x}|
    """
    def forward(self, h):
        # h: (B*C, L, D_k)
        BxC, L, D = h.shape
        # 全频域 FFT
        X = torch.fft.fft(h.float(), dim=1)  # (B*C, L, D)
        n = X.shape[1]

        # Hilbert 滤波器: 正频率 ×2, 负频率 ×0, DC/Nyquist ×1
        mask = torch.ones(n, device=h.device)
        if n % 2 == 0:
            mask[0] = 0.0
            mask[n // 2] = 1.0
            mask[1:n // 2] = 2.0
        else:
            mask[0] = 0.0
            mask[1:(n + 1) // 2] = 2.0

        X_hilbert = X * mask.unsqueeze(0).unsqueeze(-1)
        analytic = torch.fft.ifft(X_hilbert, dim=1)  # 复解析信号

        # 瞬时振幅包络
        envelope = torch.abs(analytic)  # (B*C, L, D)

        # 归一化包络 (per-mode)
        envelope = envelope / (envelope.mean(dim=1, keepdim=True) + 1e-8)

        # AM 调制: h ← envelope · h  (残差形式, 保留原始信号)
        return h + 0.3 * (envelope * h - h)


class AdaptiveMixer(nn.Module):
    """
    自适应混频权重: 替代固定 α_freq, α_time, α_disp。
    输入依赖的软门控 → 不同样本、不同时间有不同的混频策略。

    控制论来源: MRAC (Model Reference Adaptive Control)
      — 增益由当前状态决定, 而非固定值。
    """
    def __init__(self, d_koopman, n_branches=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_koopman, d_koopman // 4),
            nn.GELU(),
            nn.Linear(d_koopman // 4, n_branches),
        )
        self.n_branches = n_branches

    def forward(self, h):
        # h: (B*C, L, D_k)
        # 全局上下文 → 分支权重
        h_pool = h.mean(dim=1)  # (B*C, D_k)
        alpha = torch.softmax(self.gate(h_pool), dim=-1)  # (B*C, 3)
        return alpha  # [α_freq, α_time, α_disp] per sample


class KalmanGate(nn.Module):
    """
    Kalman 自适应门控: 替代 FreqShift 中的固定 EMA (α=0.85)。

    控制论来源: Kalman 滤波器
      预测不确定时 → 降低增益 (信任观测少)
      预测置信时 → 提高增益 (信任观测多)

    每模态有独立的 Kalman 增益:
      K_d[t] = P_d[t] / (P_d[t] + R_d)
      P_d[t] = A_d²·P_d[t-1] + Q_d

    A_d (过程动态), Q_d (过程噪声), R_d (观测噪声) — 可学习
    """
    def __init__(self, d_koopman):
        super().__init__()
        # 每模态独立的 Kalman 参数
        self.A = nn.Parameter(torch.ones(d_koopman) * 0.9)       # 过程动态
        self.log_Q = nn.Parameter(torch.zeros(d_koopman) - 2.0)  # 过程噪声 (log)
        self.log_R = nn.Parameter(torch.zeros(d_koopman) - 2.0)  # 观测噪声 (log)

    def forward(self, h_raw, h_ema_prev):
        """
        h_raw:     (B*C, L, D) 当前时刻的混频输出
        h_ema_prev: (B*C, 1, D) 上一时刻的 EMA 状态
        返回: (B*C, L, D) Kalman 滤波后的输出
        """
        BxC, L, D = h_raw.shape
        Q = torch.exp(self.log_Q).clamp(1e-4, 1.0)  # (D,)
        R = torch.exp(self.log_R).clamp(1e-4, 1.0)  # (D,)
        A = self.A.clamp(0.5, 0.99)                  # (D,)

        P = torch.ones(BxC, D, device=h_raw.device)  # 初始不确定性
        h_kalman = torch.zeros_like(h_raw)

        for t in range(L):
            # 预测步
            P_pred = A.unsqueeze(0)**2 * P + Q.unsqueeze(0)
            # Kalman 增益
            K_gain = P_pred / (P_pred + R.unsqueeze(0) + 1e-8)
            # 更新步
            innovation = h_raw[:, t, :] - (h_ema_prev.squeeze(1) if t == 0 else h_kalman[:, t-1, :])
            h_kalman[:, t, :] = (h_kalman[:, t-1, :] if t > 0 else h_ema_prev.squeeze(1)) + K_gain * innovation
            # 不确定性更新
            P = (1 - K_gain) * P_pred

        return h_kalman
