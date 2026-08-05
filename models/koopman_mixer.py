"""
Koopman Mode Mixer (KMM) — 跨模态频率混频时序预测
=====================================================
核心创新: 在 Koopman 潜空间内做乘性频率混频。
  h_i(t)·h_j(t) → 和频(θ_i+θ_j) + 差频(|θ_i-θ_j|)
  
三个耦合模块:
  Frequency-Shift Coupling:  频域滚卷混频 → 产生新频率分量
  Time-Domain Coupling:      局部⊗全局签名 → 上下文调制
  Displacement Superposition: 多尺度时移 → 跨尺度特征融合

C≤10: D_k=32, blocks=2, ~55K params
C≤50: D_k=64, blocks=2, ~200K params  
C>50: D_k=128, blocks=2, K=1, ~350K params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .kmm_enhancements import HilbertAmpMod, AdaptiveMixer
from .fed_film_mixins import TopMFreqSelect, FreqDomainEMA
from .spectral_denoiser import LearnableLowPass
from .dmd_denoiser import NeuralDMDDenoiser
from .residual_branch import ResidualTrack, AdaptiveGate


# ============================================================================
# S0  基础模块
# ============================================================================

class RevIN(nn.Module):
    """可逆实例归一化"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_channels)) if affine else None
        self.affine_bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

    def forward(self, x, inverse=False):
        if not inverse:
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._stdev
            if self.affine_weight is not None:
                x = x * self.affine_weight[:x.shape[-1]].view(1, 1, -1)
                x = x + self.affine_bias[:x.shape[-1]].view(1, 1, -1)
            return x
        else:
            C = x.shape[-1]
            if self.affine_weight is not None:
                x = x - self.affine_bias[:C].view(1, 1, -1)
                x = x / (self.affine_weight[:C].view(1, 1, -1) + self.eps)
            return x * self._stdev[:, :, :C] + self._mean[:, :, :C]


class FreqAnalyzer(nn.Module):
    """轻量频谱分析器 → 频域条件注入 (n_freq → d_model, ~2K params)"""
    def __init__(self, d_model, n_freq=49):
        super().__init__()
        self.proj = nn.LazyLinear(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B*C, n_freq) — 振幅谱
        return self.act(self.proj(x))


# ============================================================================
# S1  Koopman 提升器
# ============================================================================

class KoopmanLifter(nn.Module):
    """线性投影 + 可学习周期编码 → 潜空间"""
    def __init__(self, d_model, d_koopman, seq_len=96):
        super().__init__()
        self.lift = nn.Linear(d_model, d_koopman)
        # 可学习周期编码 (替代随机傅里叶特征)
        self.theta_enc = nn.Parameter(torch.randn(d_koopman) * 0.1)
        self.register_buffer('t', torch.arange(seq_len).float())

    def forward(self, z):
        # z: (B*C, L, d_model)
        projected = self.lift(z)
        # 周期编码: cos(t·θ_d + φ_d) per mode
        t = self.t[:z.shape[1]].to(z.device)
        pe = torch.sin(t.unsqueeze(0).unsqueeze(-1) * self.theta_enc.unsqueeze(0).unsqueeze(0))
        # 投影到 d_koopman
        pe_proj = self._pe_proj(pe) if hasattr(self, '_pe_proj') else 0
        return projected + 0.1 * pe_proj


# ============================================================================
# S2  ModeMixer Block — 三个耦合模块
# ============================================================================

class FreqShiftCoupling(nn.Module):
    """
    局部频移耦合: 每模态仅与 ±K 邻域混频。
    物理意义: 相近频率的拍频 |θ_i-θ_j| ≈ 0 → 极低频调制包络。
    自由度: D_k × (2K+1) 替代 D_k² → 减少 16× (K=1 vs D_k=32)。

    和频(θ_i+θ_j) ≈ 2θ ≈ 高频 → 通常丢弃 (>Nyquist)
    差频(|θ_i-θ_j|) ≈ 微小量 → 低频拍频 → 捕获长期调制
    """
    def __init__(self, d_koopman, n_segments=3, kernel=4, top_m=16, large_c=False):
        super().__init__()
        self.d_koopman = d_koopman
        self.n_segments = n_segments
        self.kernel = 1 if large_c else kernel   # ECL: K=1 → 9×→3× 邻域
        self.shift = nn.Parameter(torch.zeros(n_segments, d_koopman))
        self.scale = nn.Parameter(torch.ones(d_koopman) * 0.1)
        # FEDformer + FiLM 优化
        self.top_m_sel = TopMFreqSelect(top_m=top_m)
        self.freq_ema = FreqDomainEMA(alpha=0.85)

    def forward(self, h, seg_idx=0):
        BxC, L, D = h.shape
        n_freq = L // 2 + 1
        h_fft = torch.fft.rfft(h.float(), dim=1)
        h_fft = torch.nan_to_num(h_fft.real, nan=0.0) + \
                1j * torch.nan_to_num(h_fft.imag, nan=0.0)

        # P2 (FEDformer): top-M 频段选择
        h_fft = self.top_m_sel(h_fft)

        # O(1) 相位脉冲
        shift_val = self.shift[seg_idx]  # (D,)
        shift_bin = shift_val % n_freq
        lo = shift_bin.floor().long().clamp(0, n_freq - 1)
        hi = (lo + 1).clamp(0, n_freq - 1)
        frac = (shift_bin - lo.float()).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        # 构建双频点脉冲 (scatter 替代 for loop)
        phase_fft = torch.zeros(BxC, n_freq, D, device=h.device, dtype=torch.complex64)
        idx_lo = lo.unsqueeze(0).expand(BxC, -1)  # (BxC, D)
        idx_hi = hi.unsqueeze(0).expand(BxC, -1)
        phase_fft.scatter_(1, idx_lo.unsqueeze(1),
                           (1.0 - frac).expand(BxC, 1, D).to(torch.complex64))
        phase_fft.scatter_(1, idx_hi.unsqueeze(1),
                           frac.expand(BxC, 1, D).to(torch.complex64))

        # 矢量化的邻域混频: gather 替代 for-roll, 1次计算全部 ±K 邻域
        K = self.kernel
        phase_h = h_fft * phase_fft.to(h_fft.dtype)  # (B*C, n_freq, D)

        # gather 邻域索引: indices[d, k] = d + offset
        offsets = torch.arange(-K, K + 1, device=h.device)
        indices = torch.arange(D, device=h.device).unsqueeze(0) + offsets.unsqueeze(1)
        indices = indices.clamp(0, D - 1).t()  # (D, 2K+1)
        neighbors = phase_h[:, :, indices].mean(dim=-1)  # (B*C, n_freq, D)
        mixed = h_fft * neighbors
        h_cross = torch.fft.irfft(mixed, n=L, dim=1)

        # P1 (FiLM): 频域 EMA — 替代 O(L) Python for 循环
        h_cross = self.freq_ema(h_cross)
        h_cross = torch.nan_to_num(h_cross, nan=0.0, posinf=1.0, neginf=-1.0)

        return h_cross * self.scale.unsqueeze(0).unsqueeze(0)


class TimeCoupling(nn.Module):
    """
    时域耦合: 局部模式 ⊙ 全局签名 → 上下文调制
    h_local = Conv1d(h) → 捕获局部时变
    h_global = mean(h) → 全局上下文
    h_time = h_local · σ(h_global) → 全局调制局部
    """
    def __init__(self, d_koopman):
        super().__init__()
        self.local_conv = nn.Conv1d(d_koopman, d_koopman, 5,
                                     padding=2, groups=d_koopman)
        self.global_proj = nn.Linear(d_koopman, d_koopman)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, h):
        # h: (B*C, L, D_k)
        # 局部
        h_local = self.local_conv(h.permute(0, 2, 1)).permute(0, 2, 1)
        h_local = F.gelu(h_local)

        # 全局签名
        h_global = h.mean(dim=1)  # (B*C, D_k)
        gate = torch.sigmoid(self.global_proj(h_global))  # (B*C, D_k)

        # 全局调制局部
        h_time = h_local * gate.unsqueeze(1)
        return h_time * self.scale


class DisplacementSuperposition(nn.Module):
    """
    位移叠加: 多尺度时移 → 跨尺度特征融合
    h_stack = [h, h左移1, h左移2, h右移1, h右移2]
    h_disp = Conv1d(Concat(h_stack)) → 5倍通道 → D_k
    """
    def __init__(self, d_koopman, large_c=False):
        super().__init__()
        n_shifts = 1 if large_c else 5   # ECL: 只保留 1 个位移
        self.fuse = nn.Conv1d(d_koopman * n_shifts, d_koopman, 1)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.large_c = large_c

    def forward(self, h):
        BxC, L, D = h.shape
        shifts = [0] if self.large_c else [0, -1, -2, 1, 2]
        parts = []
        for s in shifts:
            if s == 0:
                parts.append(h)
            elif s < 0:
                pad = torch.zeros(BxC, -s, D, device=h.device, dtype=h.dtype)
                parts.append(torch.cat([h[:, -s:, :], pad], dim=1))
            else:
                pad = torch.zeros(BxC, s, D, device=h.device, dtype=h.dtype)
                parts.append(torch.cat([pad, h[:, :-s, :]], dim=1))

        h_stack = torch.cat(parts, dim=-1)  # (B*C, L, 5*D)
        h_disp = self.fuse(h_stack.permute(0, 2, 1)).permute(0, 2, 1)
        h_disp = F.gelu(h_disp)
        return h_disp * self.scale


class ModeMixerBlock(nn.Module):
    """ModeMixer: 三个耦合模块并行 → 融合 → 残差"""
    def __init__(self, d_koopman, n_segments=3, dropout=0.1, large_c=False):
        super().__init__()
        self.d_koopman = d_koopman
        self.n_segments = n_segments

        # 三个耦合模块 (ECL: kernel=1, disp=1, skip Hilbert)
        self.large_c = large_c
        self.freq_couple = FreqShiftCoupling(d_koopman, n_segments, large_c=large_c)
        self.time_couple = TimeCoupling(d_koopman)
        self.disp_couple = DisplacementSuperposition(d_koopman, large_c=large_c)

        # Hilbert 包络调制 (0 params → AM)
        self.hilbert = HilbertAmpMod()

        # 自适应混频权重 (~200 params)
        self.adaptive_mixer = AdaptiveMixer(d_koopman)

        # 输出投影
        self.out_proj = nn.Linear(d_koopman, d_koopman)
        self.norm = nn.LayerNorm(d_koopman)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        BxC, L, D = h.shape

        # 自适应混频权重 (输入依赖)
        alpha = self.adaptive_mixer(h)  # (B*C, 3)

        # 三段并行耦合 (Hilbert AM 已在 block 外应用)
        h_freq = sum(self.freq_couple(h, k) for k in range(self.n_segments)) / self.n_segments
        h_time = self.time_couple(h)
        h_disp = self.disp_couple(h)

        # 自适应加权融合 (per-sample)
        h_coupled = (alpha[:, 0:1].unsqueeze(1) * h_freq +
                     alpha[:, 1:2].unsqueeze(1) * h_time +
                     alpha[:, 2:3].unsqueeze(1) * h_disp)

        # 残差连接
        h_out = h + h_coupled
        h_out = self.out_proj(h_out)
        h_out = F.gelu(h_out)
        return self.norm(self.dropout(h_out) + h)


# ============================================================================
# S3  Koopman 特征值基 (用于可解释性)
# ============================================================================

class KoopmanSpectrum(nn.Module):
    """
    可学习 Koopman 特征值谱。
    λ_d = exp(-exp(ν_d) + i·θ_d) → 每模态有独立的频率 θ_d 和衰减率 r_d。
    仅用于分析和正则化——θ_d 用于周期编码，不参与前向计算。
    """
    def __init__(self, d_koopman):
        super().__init__()
        # LRU init: |λ| ∈ [0.5, 0.99]
        nu_init = torch.log(-torch.log(torch.empty(d_koopman).uniform_(0.5, 0.99)))
        self.nu = nn.Parameter(nu_init)
        self.theta = nn.Parameter(torch.randn(d_koopman) * 0.1)

    def get_eigenvalues(self):
        r = torch.exp(-torch.exp(self.nu))
        return r * torch.exp(1j * self.theta)

    def spectral_regularizer(self):
        """防坍缩: 平均 |λ| 应接近 0.8"""
        r = torch.exp(-torch.exp(self.nu))
        return ((r.mean() - 0.8) ** 2).float()


# ============================================================================
# S4  主模型
# ============================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.enc_in = getattr(configs, 'enc_in', 7)

        d_model = getattr(configs, 'd_model', 128)
        self.d_koopman = getattr(configs, 'd_koopman', 128)
        n_blocks = getattr(configs, 'n_blocks', 2)
        dropout = getattr(configs, 'dropout', 0.1)

        # C-自适应
        if self.enc_in <= 10:
            d_model = 64
            self.d_koopman = 16                              # ETT: 32→16, 减半容量防过拟合
            n_blocks = 1                                      # ETT: 2→1, 减少深度
            self.n_segments = 3
            dropout = dropout if dropout > 0 else 0.45                      # ETT: 0.25→0.45, 强正则化
            self.use_coord_trans = False
        elif self.enc_in <= 50:
            d_model = 64                                     # 96→64, 与 ETT 对齐
            self.d_koopman = 48                              # Weather: ~120K
            n_blocks = 2
            self.n_segments = 3
            self.use_coord_trans = True
        elif self.enc_in <= 400:
            # Traffic/medium: METR-LA (207), PEMS-BAY (325)
            d_model = 128
            self.d_koopman = 64
            n_blocks = 2
            self.n_segments = 3                              # 恢复多段混频
            self.use_coord_trans = True
            self._large_c = False                            # 保留 Track B + dropout
            dropout = dropout if dropout > 0 else 0.15                     # ECL够用, 交通通过CLI加高
        else:
            # Extreme: ERA5 (C>400) — 纯压缩, 省显存
            d_model = 128
            self.d_koopman = 64
            n_blocks = 2
            self.n_segments = 1
            self.use_coord_trans = True
            self._large_c = True
            dropout = dropout if dropout > 0 else 0.25   # ERA5: 15x压缩, 强正则化防过拟合

        print(f"[KMM] D_k={self.d_koopman}, blocks={n_blocks}, "
              f"seg={self.n_segments}, coord={self.use_coord_trans}")

        # Token Embedding
        self.token_embed = nn.Conv1d(1, d_model, kernel_size=7, padding=3)

        # RevIN
        self.revin = RevIN(self.enc_in)

        # 神经 DMD 去噪 (Track A 输入预处理)
        self.denoiser = NeuralDMDDenoiser(n_modes=16, d_input=self.seq_len)

        # Track B: 残差补偿分支 (~15K params)
        self.track_b = ResidualTrack(d_model=d_model, pred_len=self.pred_len,
                                      seq_len=self.seq_len, dropout=dropout)

        # 自适应融合门控
        self.fusion_gate = AdaptiveGate()

        # Koopman 观测算子: C_obs ∈ R^{K×C} — 物理通道 → 潜空间模态
        # C≤50: K=C (全通道); C>50: K=16 (压缩, 40× 显存缩减)
        self.K_proj = self.enc_in if self.enc_in <= 50 else 64  # ECL: 16→64, 保留更多通道信息
        self.C_obs = nn.Parameter(torch.empty(self.K_proj, self.enc_in))
        nn.init.orthogonal_(self.C_obs)

        # Repeat-window shortcut: 对强自相关数据集(ECL)天然逼近基线
        self.repeat_weight = nn.Parameter(torch.tensor(1.0))  # init=1.0 → 从repeat基线开始

        # Koopman 提升器
        self.lifter = KoopmanLifter(d_model, self.d_koopman, self.seq_len)

        # FreqAnalyzer (条件注入)
        self.freq_analyzer = FreqAnalyzer(d_model)

        # Koopman 谱 (可解释性 + 正则化)
        self.spectrum = KoopmanSpectrum(self.d_koopman)

        # ModeMixer Blocks
        self.blocks = nn.ModuleList([
            ModeMixerBlock(self.d_koopman, self.n_segments, dropout,
                           large_c=getattr(self, '_large_c', False))
            for _ in range(n_blocks)
        ])

        # 解码器 — CI: 每通道输出 1 维
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.koopman_to_model = nn.Linear(self.d_koopman, d_model)
        self.model_to_out = nn.Linear(d_model, 1)

    def _channel_project(self, z, B, inverse=False, n_in=None):
        """物理通道 ⇄ 潜空间模态。n_in: 输入通道数 (前向=C, 逆向=K)."""
        BxC, L, D = z.shape
        C_phys = n_in if n_in is not None else self.enc_in
        K = self.K_proj
        z_r = z.reshape(B, C_phys, L, D)
        if not inverse:
            z_proj = torch.einsum('b c l d, k c -> b k l d', z_r, self.C_obs)
            return z_proj.reshape(B * K, L, D)
        else:
            z_recon = torch.einsum('b k l d, k c -> b c l d', z_r, self.C_obs)
            return z_recon.reshape(B * self.enc_in, L, D)

    def _revin_norm(self, x):
        B, L, C = x.shape
        mean = x.nanmean(dim=1, keepdim=True).detach()
        stdev = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-3).sqrt().clamp(min=1e-3).detach()
        self._revin_mean = torch.nan_to_num(mean, nan=0.0)
        self._revin_stdev = torch.nan_to_num(stdev, nan=1.0)
        x = (x - self._revin_mean) / self._revin_stdev
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if self.revin.affine_weight is not None:
            x = x * self.revin.affine_weight[:C].view(1, 1, -1)
            x = x + self.revin.affine_bias[:C].view(1, 1, -1)
        return x

    def _revin_denorm(self, x):
        C = x.shape[-1]
        if self.revin.affine_weight is not None:
            x = x - self.revin.affine_bias[:C].view(1, 1, -1)
            x = x / (self.revin.affine_weight[:C].view(1, 1, -1) + 1e-3)
        y = x * self._revin_stdev[:, :, :C] + self._revin_mean[:, :, :C]
        return torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)

    def forward(self, x):
        B, L, C_in = x.shape
        x_norm = self._revin_norm(x)

        # ═══════════════════════════════════════════════
        # Track A: KMM 主干 (DMD 去噪 → 平滑预测)
        # ═══════════════════════════════════════════════
        x_ci = x_norm.permute(0, 2, 1).reshape(B * C_in, 1, L)
        x_ci_denoised = self.denoiser(x_ci.squeeze(1)).unsqueeze(1)
        z = self.token_embed(x_ci_denoised).permute(0, 2, 1)

        z = self._channel_project(z, B, inverse=False)
        z = torch.nan_to_num(z, nan=0.0)  # sigma 溢出保护
        X_mag = torch.abs(torch.fft.rfft(z.float(), dim=1)).mean(dim=-1)
        freq_cond = self.freq_analyzer(X_mag)
        z = z + 0.1 * freq_cond.unsqueeze(1)

        h = self.lifter(z)
        self._h_before_latent = h  # 留存用于 manifold_loss, 避免二次前向

        # ModeMixer blocks
        mode_theta = self.spectrum.theta.detach()
        for block in self.blocks:
            block._mode_theta = mode_theta
        # Hilbert 包络 → AM (ECL: 跳过以省显存)
        large_c = getattr(self, '_large_c', False)
        if not large_c:
            h = self.blocks[0].hilbert(h) if hasattr(self.blocks[0], 'hilbert') else h
        for block in self.blocks:
            h = block(h)  # no checkpoint for ECL (complex ops incompatible)
        self._h_track_a = h  # 留存用于融合门控

        # Track A 解码
        y_smooth = h.permute(0, 2, 1)
        y_smooth = self.temporal_proj(y_smooth)
        y_smooth = y_smooth.permute(0, 2, 1)
        y_smooth = self.koopman_to_model(y_smooth)

        # 逆通道投影: K → C_phys
        y_smooth = self._channel_project(
            y_smooth.reshape(B, self.K_proj, self.pred_len, -1).reshape(B*self.K_proj, self.pred_len, -1),
            B, inverse=True, n_in=self.K_proj)
        y_smooth = torch.nan_to_num(y_smooth, nan=0.0)
        y_smooth = self.model_to_out(y_smooth).squeeze(-1)
        y_smooth = torch.nan_to_num(y_smooth, nan=0.0)
        y_smooth = y_smooth.reshape(B, C_in, self.pred_len).permute(0, 2, 1)

        # ═══════════════════════════════════════════════
        # Track B: 残差分支 (C>50 自动跳过 — 显存限制)
        # ═══════════════════════════════════════════════
        if C_in <= 50:
            y_local, h_local = self.track_b(x_ci)
            y_local = y_local.reshape(B, C_in, self.pred_len).permute(0, 2, 1)
        else:
            y_local = torch.zeros_like(y_smooth)
            h_local = self._h_track_a  # 占位, 不会用于 gate

        # ═══════════════════════════════════════════════
        # 自适应融合 (C>50 时 α=0 → Y=Y_smooth)
        # ═══════════════════════════════════════════════
        if C_in <= 50:
            alpha = self.fusion_gate(self._h_track_a, h_local)
            alpha = alpha.reshape(B, C_in, 1).unsqueeze(1).squeeze(-1)
        else:
            alpha = torch.zeros(B, 1, C_in, device=y_smooth.device)

        y = y_smooth + alpha * y_local
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        # RevIN denorm
        y = self._revin_denorm(y)

        # Repeat-window shortcut: 对 ECL 等强自相关数据集，最后 L 步
        # 是极好的基线预测，模型只需学习修正量
        # Handle pred_len > seq_len: take last min(pred_len, L) steps, pad rest
        n_shortcut = min(self.pred_len, L)
        shortcut = x[:, -n_shortcut:, :]
        if self.pred_len > L:
            pad_s = torch.zeros(B, self.pred_len - L, C_in, device=shortcut.device, dtype=shortcut.dtype)
            shortcut = torch.cat([shortcut, pad_s], dim=1)
        y = y + self.repeat_weight * shortcut

        full_out = torch.cat([
            torch.zeros(B, L, C_in, device=y.device, dtype=y.dtype), y
        ], dim=1)
        return full_out

    def orthogonal_loss(self):
        """正交正则化: 强制 Track A 和 Track B 的潜特征不拟合相同模式."""
        if not hasattr(self, '_h_track_a'):
            return torch.tensor(0.0)
        # 用 Track A 和 Track B 的时间平均特征做正交约束
        ha = F.normalize(self._h_track_a.mean(dim=1), dim=-1)
        # 需要再次前向取 Track B 特征 (开销大) → 仅在训练时从 forward 中取出
        return torch.tensor(0.0)  # 占位, orthogonal_loss 在 train.py 中直接计算

    def manifold_loss(self, out):
        """流形一致性: C>50 时跳过 (显存限制).
        Args:
            out: 模型前向输出 (B, L+H, C)，由 train loop 传入，避免二次前向
        """
        if getattr(self, '_large_c', False):
            return torch.tensor(0.0, device=out.device)
        if not hasattr(self, '_h_before_latent') or self._h_before_latent is None:
            return torch.tensor(0.0, device=out.device)
        B = out.shape[0]
        L = self.seq_len
        h_before = self._h_before_latent  # (B*K, L, D_k) — 已在 forward 中存储

        # 从已计算的输出中提取预测部分, 不再调用 self(x)
        pred = out[:, L:, :]  # (B, H, C)
        pred_ci = pred.permute(0, 2, 1).reshape(B * self.enc_in, 1, self.pred_len)
        z_pred = self.token_embed(pred_ci).permute(0, 2, 1)
        z_pred = self._channel_project(z_pred, B, inverse=False)
        h_after = self.lifter(z_pred)

        # 用平均潜状态的方向一致性 (scale-invariant)
        hb = F.normalize(h_before.mean(dim=1), dim=-1)
        ha = F.normalize(h_after.mean(dim=1), dim=-1)
        return (1.0 - (hb * ha).sum(dim=-1)).mean()

    def spectral_regularizer(self):
        return self.spectrum.spectral_regularizer()
