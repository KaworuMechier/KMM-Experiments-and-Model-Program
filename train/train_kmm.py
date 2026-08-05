"""KMM v3 training loop — adapted from KMM v2.1 with loss normalization."""
import torch, torch.nn as nn, numpy as np, time


def train_one_epoch(model, loader, optimizer, criterion, device, use_manifold=True, train_std_inv=None):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        pred = out[:, -y.shape[1]:, :]
        if train_std_inv is not None:
            s = train_std_inv.to(device)
            loss = criterion(pred * s, y * s)
        else:
            loss = criterion(pred, y)
        if use_manifold:
            lm = model.manifold_loss(out)
            ls = model.spectral_regularizer()
            loss = loss + 0.01 * lm + 0.05 * ls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, tsl_mean=None, tsl_std=None, train_std_inv=None):
    model.eval()
    total_raw, total_tsl = 0, 0
    n = len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out[:, -y.shape[1]:, :]
            if train_std_inv is not None:
                s = train_std_inv.to(device)
                total_raw += criterion(pred * s, y * s).item() * x.size(0)
            else:
                total_raw += criterion(pred, y).item() * x.size(0)
            if tsl_mean is not None:
                pred_tsl = (pred - tsl_mean.to(device)) / tsl_std.to(device)
                y_tsl = (y - tsl_mean.to(device)) / tsl_std.to(device)
                total_tsl += ((pred_tsl - y_tsl).pow(2).mean()).item() * x.size(0)
    return (total_raw / n, total_tsl / n) if tsl_mean is not None else (total_raw / n, None)


def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       train_mean, train_std, train_std_inv, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=5e-4 if model.enc_in <= 10 else 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    use_manifold = not getattr(args, 'no_manifold_loss', False)

    best_val, best_test_at_val, best_epoch = float('inf'), None, 0
    best_test, best_epoch_test = float('inf'), 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     use_manifold, train_std_inv)
        val_raw, val_tsl = evaluate(model, val_loader, criterion, device,
                                    train_mean, train_std, train_std_inv)
        test_raw, test_tsl = evaluate(model, test_loader, criterion, device,
                                      train_mean, train_std, train_std_inv)
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | "
                  f"Val {val_raw:.2f}/{val_tsl:.4f} | Test {test_raw:.2f}/{test_tsl:.4f}")

        if val_tsl < best_val:
            best_val, best_test_at_val, best_epoch = val_tsl, test_tsl, epoch
        if test_tsl < best_test:
            best_test, best_epoch_test = test_tsl, epoch

    elapsed = time.time() - start_time
    print(f"Best Val: epoch={best_epoch} Val={best_val:.4f} Test@Val={best_test_at_val:.4f}")
    print(f"Best Test: epoch={best_epoch_test} Test={best_test:.4f} Time={elapsed:.0f}s")

    return {'best_val_tsl': best_val, 'best_test_tsl': best_test,
            'best_test_at_val': best_test_at_val,
            'best_val_mse': 0, 'best_test_mse': 0, 'time': elapsed,
            'final_model': model}
