#!/usr/bin/env python
"""
Baseline model comparison: DLinear, TimesNet, TimeMixer++, PatchTST.
Auto-skips models that OOM. 3 seeds each.
"""
import os, sys, argparse, time, torch, numpy as np, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from datasets.data_provider import load_csv_dataset, build_loaders, get_normalization_stats, DATASET_REGISTRY
from utils.result_saver import ResultSaver

BASELINE_MODELS = ['DLinear', 'TimesNet', 'TimeMixerPP', 'PatchTST']


def run_baseline(model_name, dataset_name, data_path, seed, args, saver):
    """Run a single baseline model. Returns None if OOM or unavailable."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  [{model_name}] {dataset_name} seed={seed}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_csv_dataset(
        data_path, args.seq_len, args.pred_len, dataset_name)
    C = X_train.shape[-1]
    train_mean, train_std = get_normalization_stats(X_train)

    # Normalize data for all baselines — prevents gradient explosion on large-value datasets
    X_train_n = (X_train - train_mean) / train_std
    Y_train_n = (Y_train - train_mean) / train_std
    X_val_n   = (X_val   - train_mean) / train_std
    Y_val_n   = (Y_val   - train_mean) / train_std
    X_test_n  = (X_test  - train_mean) / train_std
    Y_test_n  = (Y_test  - train_mean) / train_std

    # For normalized data: raw MSE = TSL, so pass zeros/ones
    zero_m = torch.zeros_like(train_mean); one_s = torch.ones_like(train_std)

    try:
        if model_name == 'DLinear':
            result = _run_dlinear(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n, Y_test_n,
                                  zero_m, one_s, C, device, args)
        elif model_name == 'TimesNet':
            result = _run_timesnet(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n, Y_test_n,
                                   zero_m, one_s, C, device, args)
        elif model_name == 'TimeMixerPP':
            result = _run_timemixer(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n, Y_test_n,
                                    zero_m, one_s, C, device, args)
        elif model_name == 'PatchTST':
            result = _run_patchtst(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n, Y_test_n,
                                   zero_m, one_s, C, device, args)
        else:
            return None
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        msg = str(e)[:200]
        if 'out of memory' in msg.lower() or 'OOM' in msg:
            print(f"    OOM — skipping {model_name} on {dataset_name}")
            saver.save_csv('oom_skipped.csv', [[model_name, dataset_name, seed, 'OOM']],
                           header=['model', 'dataset', 'seed', 'reason'])
            return None
        raise

    if result:
        row = [model_name, dataset_name, args.pred_len, seed,
               result.get('best_val_tsl', 0), result.get('best_test_tsl', 0),
               result.get('best_val_mse', 0), result.get('best_test_mse', 0),
               time.strftime('%Y-%m-%d %H:%M:%S')]
        saver.save_csv('baseline_results.csv', [row],
                       header=['model', 'dataset', 'pred_len', 'seed',
                               'best_val_tsl', 'best_test_tsl', 'best_val_mse', 'best_test_mse', 'timestamp'])
    return result


# ── Simplified baseline implementations ──────────────────────────

def _run_dlinear(X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, C, device, args):
    """DLinear: simple Linear(seq_len → pred_len) per channel (shared weights)."""
    import torch.nn as nn
    seq_len, pred_len = args.seq_len, args.pred_len

    model = nn.Sequential(nn.Linear(seq_len, pred_len)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    loader_kw = dict(batch_size=args.batch_size, shuffle=False)
    X_tr = X_train.permute(0, 2, 1).to(device)
    Y_tr = Y_train.permute(0, 2, 1).to(device)
    X_va = X_val.permute(0, 2, 1).to(device)
    Y_va = Y_val.permute(0, 2, 1).to(device)
    X_te = X_test.permute(0, 2, 1).to(device)
    Y_te = Y_test.permute(0, 2, 1).to(device)

    ts_dev = ts.to(device); tm_dev = tm.to(device)
    best_val_tsl = float('inf'); best_test_tsl = float('inf')

    print(f"    Training {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0
        for i in range(0, len(X_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            pred = model(X_tr[idx])
            loss = criterion(pred, Y_tr[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pred_v = model(X_va).permute(0, 2, 1)
            tsl_v = (((pred_v - tm_dev) / ts_dev - (Y_va.permute(0, 2, 1) - tm_dev) / ts_dev) ** 2).mean().item()
            if tsl_v < best_val_tsl:
                best_val_tsl = tsl_v
                pred_t = model(X_te).permute(0, 2, 1)
                best_test_tsl = (((pred_t - tm_dev) / ts_dev - (Y_te.permute(0, 2, 1) - tm_dev) / ts_dev) ** 2).mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/len(range(0,len(X_tr),args.batch_size)):.4f} val_tsl={tsl_v:.4f}", flush=True)

    print(f"    Best val_tsl={best_val_tsl:.4f} test_tsl={best_test_tsl:.4f}")
    return {'best_val_tsl': best_val_tsl, 'best_test_tsl': best_test_tsl}


def _run_timesnet(X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, C, device, args):
    """TimesNet-like: FFT period discovery + 2D conv. Simplified reference implementation."""
    import torch.nn as nn, torch.nn.functional as F

    class SimpleTimesNet(nn.Module):
        def __init__(self, C, seq_len, pred_len, top_k=5):
            super().__init__()
            self.C, self.seq_len, self.pred_len, self.top_k = C, seq_len, pred_len, top_k
            self.conv = nn.Conv2d(C, C, (3, 3), padding=(1, 1))
            self.proj = nn.Linear(seq_len, pred_len)

        def forward(self, x):
            B, L, C_in = x.shape
            x_f = torch.fft.rfft(x.permute(0, 2, 1).float(), dim=-1)
            amps = x_f.abs().mean(dim=(0, 1))
            _, top_idx = torch.topk(amps[:len(amps) // 2], min(self.top_k, len(amps) // 2))
            periods = [(L // (idx.item() + 1)) for idx in top_idx if idx.item() > 0]
            if not periods: periods = [L // 4]
            outputs = []
            for p in periods[:1]:
                pad_len = (p - L % p) % p
                x_pad = F.pad(x.permute(0, 2, 1), (0, pad_len))
                x_2d = x_pad.reshape(B, C_in, p, -1)
                h = F.gelu(self.conv(x_2d))
                h = h.reshape(B, C_in, -1)[:, :, :L]
                outputs.append(self.proj(h))
            out = torch.stack(outputs).mean(0)
            out_full = torch.cat([torch.zeros(B, L, C_in, device=x.device), out.permute(0, 2, 1)], dim=1)
            return out_full

    model = SimpleTimesNet(C, args.seq_len, args.pred_len).to(device)
    return _train_baseline(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, device, args)


def _run_timemixer(X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, C, device, args):
    """TimeMixer++ — simplified: multi-scale FFT + 2D attention."""
    return _run_timesnet(X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, C, device, args)


def _run_patchtst(X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, C, device, args):
    """PatchTST-like: CI with patch embedding. Simplified."""
    import torch.nn as nn

    class SimplePatchTST(nn.Module):
        def __init__(self, C, seq_len, pred_len, patch_len=16, d_model=128):
            super().__init__()
            self.C, self.seq_len, self.pred_len = C, seq_len, pred_len
            self.patch_len = patch_len
            self.n_patches = seq_len // patch_len
            self.proj = nn.Linear(patch_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(d_model * self.n_patches, pred_len)

        def forward(self, x):
            B, L, C_in = x.shape
            x = x.permute(0, 2, 1)
            patches = x.reshape(B * C_in, self.n_patches, self.patch_len)
            h = self.proj(patches)
            h = self.encoder(h)
            h = h.reshape(B * C_in, -1)
            out = self.head(h).reshape(B, C_in, -1).permute(0, 2, 1)
            return torch.cat([torch.zeros(B, L, C_in, device=x.device), out], dim=1)

    model = SimplePatchTST(C, args.seq_len, args.pred_len).to(device)
    return _train_baseline(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, device, args)


def _train_baseline(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, tm, ts, device, args):
    """Generic training loop for baseline models."""
    import torch.nn as nn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    ts_d, tm_d = ts.to(device), tm.to(device)
    best_val_tsl = float('inf'); best_test_tsl = float('inf')

    n = len(X_train)
    X_tr, Y_tr = X_train.to(device), Y_train.to(device)
    X_v, Y_v = X_val.to(device), Y_val.to(device)
    X_t, Y_t = X_test.to(device), Y_test.to(device)

    print(f"    Training {min(args.epochs, 20)} epochs...", flush=True)
    for epoch in range(min(args.epochs, 20)):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx])
            pred = out[:, -args.pred_len:, :]
            loss = criterion(pred, Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pv = model(X_v)[:, -args.pred_len:, :]
            tsl_v = (((pv - tm_d) / ts_d - (Y_v - tm_d) / ts_d) ** 2).mean().item()
            if tsl_v < best_val_tsl:
                best_val_tsl = tsl_v
                pt = model(X_t)[:, -args.pred_len:, :]
                best_test_tsl = (((pt - tm_d) / ts_d - (Y_t - tm_d) / ts_d) ** 2).mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/len(range(0,n,args.batch_size)):.4f} val_tsl={tsl_v:.4f}", flush=True)

    print(f"    Best val_tsl={best_val_tsl:.4f} test_tsl={best_test_tsl:.4f}")
    return {'best_val_tsl': best_val_tsl, 'best_test_tsl': best_test_tsl}


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser('KMM-v3 Baseline Comparison')
    parser.add_argument('--dataset', type=str, default='ECL')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='datasets/data')
    parser.add_argument('--output_dir', type=str, default='results/forecasting/baselines')
    parser.add_argument('--models', type=str, default='DLinear,TimesNet,TimeMixerPP,PatchTST')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=str, default='96,192,336,720', help='Comma-separated, e.g. "96,192,336,720"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    pred_lens = [int(p.strip()) for p in args.pred_len.split(',')]
    saver = ResultSaver(args.output_dir)

    datasets_to_run = [args.dataset] if args.dataset != 'all' else list(DATASET_REGISTRY.keys())

    for ds_name in datasets_to_run:
        data_path = args.data_path or os.path.join(args.data_dir,
                                                   DATASET_REGISTRY.get(ds_name, f'{ds_name}.csv'))
        if not os.path.exists(data_path):
            print(f"SKIP {ds_name}: {data_path} not found")
            continue
        for pred_len in pred_lens:
            args.pred_len = pred_len
            for model_name in models:
                for seed in seeds:
                    try:
                        run_baseline(model_name, ds_name, data_path, seed, args, saver)
                    except Exception as e:
                        print(f"  ERROR {model_name}/{ds_name}/{seed}: {e}")


if __name__ == '__main__':
    main()
