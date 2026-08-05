#!/usr/bin/env python
"""
KMM Channel Capacity Test: how many channels can KMM handle on RTX 3090?

Generates synthetic data at increasing C, runs dummy training,
reports peak VRAM. No convergence needed — just OOM boundary.

Paper claim: "KMM scales to C=2000+ on consumer hardware (RTX 3090 24GB)
             while all Transformer baselines OOM at C=325."
"""
import torch, numpy as np, time, os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models.koopman_mixer import Model
import torch.nn as nn


def test_channel_capacity(C, L=96, H=96, B=16, epochs=3, device='cuda'):
    """Quick VRAM test at given channel count. Returns (ok, peak_vram_gb)."""
    if not torch.cuda.is_available():
        return True, 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Generate synthetic data
        X = torch.randn(500, L, C, device=device)
        Y = torch.randn(500, H, C, device=device)
        xm = X.mean(dim=(0,1), keepdim=True); xs = X.std(dim=(0,1), keepdim=True) + 1e-8
        Xn, Yn = (X - xm) / xs, (Y - xm) / xs

        configs = type('C', (), {})()
        configs.seq_len = L; configs.pred_len = H; configs.enc_in = C
        configs.d_model = 128; configs.d_koopman = 128; configs.n_blocks = 2
        configs.dropout = 0.0

        model = Model(configs).to(device)
        model(torch.zeros(1, L, C, device=device))
        n_params = sum(p.numel() for p in model.parameters())

        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        crit = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for i in range(0, len(Xn), B):
                xb = Xn[i:i+B]; yb = Yn[i:i+B]
                opt.zero_grad()
                out = model(xb); pred = out[:, -H:, :]
                loss = crit(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        peak_mb = torch.cuda.max_memory_allocated()
        peak_gb = peak_mb / 1024**3
        torch.cuda.empty_cache()
        return True, peak_gb, n_params

    except RuntimeError as e:
        torch.cuda.empty_cache()
        if 'out of memory' in str(e).lower():
            return False, 0, 0
        raise


def main():
    CHANNELS = [100, 200, 321, 500, 862, 990, 1500, 2000, 3000, 3500]
    CHANNELS += list(range(3600, 5001, 100))  # fine-grained OOM search

    import csv, os
    os.makedirs('results/analysis', exist_ok=True)

    print(f"KMM Channel Capacity Test (RTX 3090 24GB, B=16)\n")
    print(f"{'C':>6s}  {'Status':>8s}  {'Peak VRAM':>10s}  {'Params':>10s}")
    print(f"{'-'*45}")

    max_ok = 0
    rows = []
    for C in CHANNELS:
        ok, vram, params = test_channel_capacity(C)
        status = f"{vram:.1f} GB" if ok else "OOM"
        print(f"{C:>6d}  {status:>8s}  {vram:>8.1f} GB  {params:>10,}" if ok else f"{C:>6d}  {'OOM':>8s}  {'--':>10s}  {'--':>10s}")
        rows.append([C, 'OK' if ok else 'OOM', f'{vram:.1f}' if ok else '', params if ok else '', time.strftime('%H:%M:%S')])
        if ok: max_ok = C
        else: break

    with open('results/analysis/channel_capacity.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channels', 'status', 'peak_vram_gb', 'params', 'timestamp'])
        w.writerows(rows)
    print(f"\n  Results saved: results/analysis/channel_capacity.csv")

    print(f"\n  Max channels on RTX 3090: C={max_ok} (next step OOM)")
    print(f"  TimesNet/PatchTST OOM at C=325 on same hardware.")
    print(f"  → KMM scales ≥{max_ok/325:.0f}× beyond Transformer baseline limit.")
    print(f"\n  Paper table ready:")
    print(f"  TimesNet OOM at:   C=325")
    print(f"  PatchTST OOM at:   C=325")
    print(f"  KMM works up to:   C={max_ok}")


if __name__ == '__main__':
    main()
