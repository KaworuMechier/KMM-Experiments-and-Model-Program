#!/usr/bin/env python
"""
KMM-air: Lightweight Koopman Mode Mixer for ETT Datasets.

Optimized for C<=10 (ETTh1, ETTm1) with:
  D_k=16, 1 block, dropout=0.45, weight_decay=5e-4.

Usage:
  python train_ett.py --dataset ETTh1
  python train_ett.py --dataset ETTm1 --gpu 1
"""
import os, sys, argparse, time, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.koopman_mixer import Model

# ── ETT Data Loading ──────────────────────────────────────────
def load_ett(data_path, seq_len=96, pred_len=96, dataset='ETTh1'):
    df = pd.read_csv(data_path)
    cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    data = df[cols].values.astype(np.float32)
    total = len(df)
    multiplier = 4 if dataset in ('ETTm1', 'ETTm2') else 1
    train_end = 12 * 30 * 24 * multiplier
    val_end = train_end + 4 * 30 * 24 * multiplier
    def make(s, e):
        X, Y = [], []
        for i in range(s, e - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len:i+seq_len+pred_len])
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))
    X_train, Y_train = make(0, train_end)
    X_val, Y_val = make(train_end, val_end)
    X_test, Y_test = make(val_end, total)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# ── Training ──────────────────────────────────────────────────
def train_one_epoch(model, loader, opt, crit, device, std_inv, time_weights=None):
    model.train(); total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device); opt.zero_grad()
        out = model(x); pred = out[:, -y.shape[1]:, :]
        s = std_inv.to(device)
        if time_weights is not None:
            w = time_weights.to(device).view(1, -1, 1)
            se = ((pred*s - y*s)**2)  # (B, H, C)
            loss = (se * w).mean()
        else:
            loss = crit(pred*s, y*s)
        loss = loss + 0.01*model.manifold_loss(out) + 0.05*model.spectral_regularizer()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def evaluate(model, loader, crit, device, mean, std, std_inv):
    model.eval(); raw, tsl = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); pred = out[:, -y.shape[1]:, :]
            s = std_inv.to(device)
            raw += crit(pred*s, y*s).item()*x.size(0)
            p_tsl = (pred-mean.to(device))/std.to(device)
            y_tsl = (y-mean.to(device))/std.to(device)
            tsl += ((p_tsl-y_tsl).pow(2).mean()).item()*x.size(0)
    n = len(loader.dataset); return raw/n, tsl/n

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser('KMM-air for ETT')
    parser.add_argument('--dataset', type=str, default='ETTh1', choices=['ETTh1','ETTm1'])
    parser.add_argument('--data_dir', type=str, default='datasets/data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pred_len', type=str, default='96,192,336,720')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    pred_lens = [int(p.strip()) for p in args.pred_len.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    data_path = os.path.join(args.data_dir, f'{args.dataset}.csv')

    for seed in seeds:
        for H in pred_lens:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"\n{'='*60}")
            print(f"{args.dataset} | Seed={seed} | pred_len={H} | GPU={args.gpu}")
            print(f"{'='*60}")

            X_train, Y_train, X_val, Y_val, X_test, Y_test = load_ett(
                data_path, args.seq_len, H, args.dataset)
            C = X_train.shape[-1]
            print(f"Loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

            train_mean = X_train.mean(dim=(0,1), keepdim=True)
            train_std = X_train.std(dim=(0,1), keepdim=True) + 1e-8
            train_std_inv = 1.0 / train_std

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)
            test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch_size)

            configs = type('C', (), {})()
            configs.seq_len = args.seq_len; configs.pred_len = H
            configs.enc_in = C; configs.d_model = 64; configs.d_koopman = 16
            configs.n_blocks = 1; configs.dropout = 0.0

            model = Model(configs).to(device)
            model(torch.zeros(2, args.seq_len, C, device=device))
            n_p = sum(p.numel() for p in model.parameters())
            print(f"Params: {n_p:,}")

            opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
            crit = nn.MSELoss()

            # Curriculum: for long horizon, start from H=96 checkpoint
            ckpt_path = f'results/{args.dataset}_s{seed}_96.pt'
            if H > 96 and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                # Skip temporal_proj (shape depends on pred_len)
                model_dict = model.state_dict()
                ckpt_filtered = {k: v for k, v in ckpt.items()
                                 if k in model_dict and model_dict[k].shape == v.shape}
                model.load_state_dict(ckpt_filtered, strict=False)
                print(f"  Curriculum: loaded {len(ckpt_filtered)}/{len(ckpt)} params from H=96")

            # Time-step weights for long horizon prediction
            time_weights = torch.linspace(1.0, 2.0, H) if H >= 192 else None

            best_val, best_test_at_val = float('inf'), None
            best_test, best_epoch_test = float('inf'), 0
            best_epoch = 0; patience_counter = 0
            start = time.time()

            for ep in range(1, args.epochs+1):
                tl = train_one_epoch(model, train_loader, opt, crit, device, train_std_inv, time_weights)
                vr, vt = evaluate(model, val_loader, crit, device, train_mean, train_std, train_std_inv)
                tr, tt = evaluate(model, test_loader, crit, device, train_mean, train_std, train_std_inv)
                sched.step(vt)
                if vt < best_val:
                    best_val, best_test_at_val = vt, tt; best_epoch = ep; patience_counter = 0
                else:
                    patience_counter += 1
                if tt < best_test: best_test, best_epoch_test = tt, ep
                if ep==1 or ep%5==0:
                    print(f"Epoch {ep:3d} | Train {tl:.4f} | Val {vr:.2f}/{vt:.4f} | Test {tr:.2f}/{tt:.4f} | LR {opt.param_groups[0]['lr']:.2e}")
                if patience_counter >= 5:  # early stop
                    print(f"Early stop at epoch {ep} (no val improvement for 5 epochs)")
                    break

            elapsed = time.time()-start
            print(f"Best Val epoch={best_epoch} TSL={best_val:.4f} Test@Val={best_test_at_val:.4f} | Best Test epoch={best_epoch_test} TSL={best_test:.4f} | Time={elapsed:.0f}s")

            os.makedirs('results', exist_ok=True)
            # Save checkpoint for curriculum learning (H=96 -> longer horizons)
            if H == 96:
                torch.save(model.state_dict(), f'results/{args.dataset}_s{seed}_96.pt')
            with open(f'results/{args.dataset}_results.csv', 'a') as f:
                f.write(f"{args.dataset},{H},{seed},{n_p},{best_val:.4f},{best_test_at_val:.4f},{best_test:.4f},{elapsed:.0f}\n")


if __name__ == '__main__':
    main()
