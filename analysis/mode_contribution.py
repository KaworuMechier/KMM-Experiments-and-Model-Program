#!/usr/bin/env python
"""
Mode Contribution Analysis via Shapley-style Ablation.

Two modes:
  --mode train:  Train KMM on full data, save checkpoint.
  --mode analyze: Load checkpoint, run Shapley ablation, classify modes.
  --mode all (default): Train then analyze.

Key metric: Delta_MSE = MSE(ablated) - MSE(full)
  contrib > 0: removing hurts -> USEFUL mode
  contrib < 0: removing helps -> HARMFUL mode (overfit/noise)

Usage:
  python analysis/mode_contribution.py --dataset ECL --mode all --epochs 30
  python analysis/mode_contribution.py --dataset ECL --mode analyze --checkpoint results/analysis/ecl_model.pt
"""
import os, sys, argparse, torch, numpy as np, csv, time as _time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.koopman_mixer import Model
from datasets.data_provider import load_csv_dataset, get_normalization_stats, build_loaders
import torch.nn as nn


def train_model(args, device):
    """Train KMM on full data, return model and stats."""
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_csv_dataset(
        args.data_path, args.seq_len, args.pred_len, args.dataset)
    C = X_train.shape[-1]
    train_mean, train_std = get_normalization_stats(X_train)
    train_std_inv = 1.0 / train_std

    train_loader, val_loader, test_loader = build_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, args.batch_size)

    configs = type('C', (), {})()
    configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
    configs.enc_in = C; configs.d_model = 128; configs.d_koopman = 128
    configs.n_blocks = 2; configs.dropout = 0.0

    model = Model(configs).to(device)
    model(torch.zeros(2, args.seq_len, C, device=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KMM params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x); pred = out[:, -y.shape[1]:, :]
            s = train_std_inv.to(device)
            loss = criterion(pred * s, y * s) + 0.05 * model.spectral_regularizer()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_mse = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); pred = out[:, -y.shape[1]:, :]
                s = train_std_inv.to(device)
                val_mse += criterion(pred * s, y * s).item() * x.size(0)
            val_mse /= len(val_loader.dataset)
            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), args.checkpoint)

        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_mse:.4f}")

    # Load best
    model.load_state_dict(torch.load(args.checkpoint))
    print(f"Best Val MSE: {best_val:.4f} | Checkpoint: {args.checkpoint}")
    return model, val_loader, train_mean, train_std, C


def compute_mode_contributions(model, val_loader, device, train_std_inv, n_batches=30):
    """Shapley-style: ablate each mode, measure Delta_MSE in NORMALIZED space."""
    model.eval()
    D_k = model.d_koopman
    criterion = nn.MSELoss()
    s = train_std_inv.to(device)

    # Baseline MSE (normalized — same as training)
    full_mses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= n_batches: break
            x, y = x.to(device), y.to(device)
            out = model(x); pred = out[:, -y.shape[1]:, :]
            full_mses.append(criterion(pred * s, y * s).item())
    baseline_mse = np.mean(full_mses)

    contributions = np.zeros(D_k)
    variances = np.zeros(D_k)
    original_nu = model.spectrum.nu.clone()

    _t0 = _time.time()
    for d in range(D_k):
        with torch.no_grad():
            model.spectrum.nu[d] = 10.0  # |lambda_d| -> 0
        ablated_mses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= n_batches: break
                x, y = x.to(device), y.to(device)
                out = model(x); pred = out[:, -y.shape[1]:, :]
                ablated_mses.append(criterion(pred * s, y * s).item())
        contributions[d] = np.mean(ablated_mses) - baseline_mse
        variances[d] = np.var(ablated_mses)
        with torch.no_grad():
            model.spectrum.nu[d] = original_nu[d]

        if (d + 1) % 4 == 0 or d == D_k - 1:
            e = _time.time() - _t0
            eta = e / (d + 1) * (D_k - d - 1)
            print(f"    Mode {d+1}/{D_k} (elapsed {e:.0f}s, ETA {eta:.0f}s)", flush=True)

    return contributions, variances, baseline_mse


def compute_combined_score(contributions, variances, r, cobs_norms):
    pos = np.maximum(contributions, 0)
    eps = 1e-8
    cv = np.sqrt(np.maximum(variances, 0)) / (np.abs(contributions) + eps)
    stability = 1.0 / (1.0 + cv)
    score = pos * stability * r * cobs_norms
    return score / (score.sum() + eps)


def run_analysis(model, val_loader, train_std_inv, args, device):
    """Shapley ablation + mode classification."""
    print(f"\nComputing mode contributions (Shapley-style ablation)...")
    contributions, variances, baseline_mse = compute_mode_contributions(
        model, val_loader, device, train_std_inv, args.n_batches)

    r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu().numpy()
    cobs_norms = (model.C_obs.detach().cpu().numpy() ** 2).sum(axis=1)
    D = min(model.K_proj, len(r))
    r, cobs_norms = r[:D], cobs_norms[:D]
    contributions, variances = contributions[:D], variances[:D]

    pos_contrib = np.maximum(contributions, 0)
    score = compute_combined_score(pos_contrib, variances, r, cobs_norms)

    useful = contributions > 0
    n_useful = useful.sum()
    n_harmful = D - n_useful

    # Sort useful modes by score
    idx = np.argsort(-score * useful)

    print(f"\n{'='*70}")
    print(f"Mode Classification via Shapley Contribution")
    print(f"{'='*70}")
    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Modes: {n_useful} USEFUL (removing hurts), {n_harmful} HARMFUL (removing helps!)")
    print(f"\n{'Rank':<6s} {'Mode':<6s} {'|lambda|':<8s} {'Contrib':>10s} {'Stability':<10s} {'Score':<8s} {'Class'}")
    print(f"{'-'*65}")

    rows = []
    for rank, d in enumerate(idx):
        if score[d] <= 0 and rank >= n_useful: break
        contrib = contributions[d]
        var = variances[d]
        cv = np.sqrt(max(var, 0)) / (abs(contrib) + 1e-8)
        stab = 1.0 / (1.0 + cv)
        cls = 'USEFUL' if contrib > 0 else 'HARMFUL'
        print(f"{rank+1:<6d} {d:<6d} {r[d]:<8.4f} {contrib:>+10.4f} {stab:<10.4f} {score[d]:<8.4f} {cls}")
        rows.append([args.dataset, rank+1, d, r[d], contrib, var, score[d], cls])

    print(f"\n  NOTE: Single-mode ablation measures INDIVIDUAL contribution.")
    print(f"  All-negative result -> model overparameterized at {D} modes.")
    print(f"  Running BATCH pruning to find optimal mode count...\n")

    # Sort by contribution (least harmful = highest contrib value)
    sorted_idx = np.argsort(-contributions)  # most positive first
    original_nu = model.spectrum.nu.clone()
    batch_sizes = [8, 16, 24, 32, 40, 48, 56]
    print(f"  {'Pruned':<8s} {'MSE':>12s} {'Delta':>10s}")
    print(f"  {'-'*32}")
    best_mse = baseline_mse; best_k = D
    for k in batch_sizes:
        if k >= D: continue
        prune_idx = sorted_idx[k:]  # keep top k, prune rest
        with torch.no_grad():
            model.spectrum.nu[:] = original_nu[:]
            model.spectrum.nu[prune_idx] = 10.0
        mses = []
        s_batch = train_std_inv.to(device)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= args.n_batches: break
                x, y = x.to(device), y.to(device)
                out = model(x); pred = out[:, -y.shape[1]:, :]
                mses.append(nn.MSELoss()(pred * s_batch, y * s_batch).item())
        prune_mse = np.mean(mses)
        delta = prune_mse - baseline_mse
        marker = ' <-- BEST' if delta < 0 and (best_mse - prune_mse) > 0 else ''
        print(f"  {k:<8d} {prune_mse:>12.4f} {delta:>+10.4f}{marker}")
        if prune_mse < best_mse:
            best_mse = prune_mse; best_k = k
    print(f"\n  Optimal: keep {best_k}/{D} modes (prune {D-best_k}) -> MSE {best_mse:.4f} (baseline {baseline_mse:.4f})")
    with torch.no_grad():
        model.spectrum.nu.copy_(original_nu)

    csv_path = os.path.join(args.output_dir, f'mode_contribution_{args.dataset}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'rank', 'mode', 'lambda', 'contrib', 'variance', 'score', 'class'])
        w.writerows(rows)
    print(f"Results saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser('Mode Contribution Analysis')
    parser.add_argument('--dataset', type=str, default='ECL')
    parser.add_argument('--data_path', type=str, default='datasets/data/ECL.csv')
    parser.add_argument('--output_dir', type=str, default='results/analysis')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'analyze', 'all'])
    parser.add_argument('--checkpoint', type=str, default='results/analysis/kmm_checkpoint.pt')
    parser.add_argument('--n_batches', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ('train', 'all'):
        print(f"Training KMM on {args.dataset} ({args.epochs} epochs, full data)...")
        model, val_loader, train_mean, train_std, C = train_model(args, device)
        train_std_inv = 1.0 / train_std
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_csv_dataset(
            args.data_path, args.seq_len, args.pred_len, args.dataset)
        C = X_train.shape[-1]
        train_mean, train_std = get_normalization_stats(X_train)
        from torch.utils.data import DataLoader, TensorDataset
        val_loader = DataLoader(TensorDataset(X_val[:500], Y_val[:500]),
                                batch_size=args.batch_size, shuffle=True)

        configs = type('C', (), {})()
        configs.seq_len = args.seq_len; configs.pred_len = args.pred_len
        configs.enc_in = C; configs.d_model = 128; configs.d_koopman = 128
        configs.n_blocks = 2; configs.dropout = 0.0
        model = Model(configs).to(device)
        model(torch.zeros(2, args.seq_len, C, device=device))
        model.load_state_dict(torch.load(args.checkpoint))
        train_std_inv = 1.0 / train_std
        print(f"Checkpoint loaded.")

    if args.mode in ('analyze', 'all'):
        run_analysis(model, val_loader, train_std_inv, args, device)


if __name__ == '__main__':
    main()
