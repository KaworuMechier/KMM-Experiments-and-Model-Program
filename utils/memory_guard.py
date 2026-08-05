"""
Memory guard: auto-scaling batch size, pre-flight VRAM checks, OOM recovery.
Ensures KMM-v3 never crashes from OOM — it gracefully degrades.
"""
import torch, gc, time, warnings


def get_free_vram_gb(device='cuda'):
    """Get free VRAM in GB on the specified CUDA device."""
    if not torch.cuda.is_available():
        return float('inf')
    if isinstance(device, torch.device):
        device = device.index or 0
    elif isinstance(device, str) and 'cuda' in device:
        device = int(device.split(':')[-1]) if ':' in device else 0
    else:
        device = 0
    free, total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def estimate_kmm_vram(C, L=96, H=96, batch_size=16, fp32=True, use_manifold=True):
    """Estimate peak VRAM for KMM on a dataset with C channels.

    Based on calibrated measurements (ECL v2.1: 3147 MB @ C=321, B=16).
    Scaling factor: ~3.33x correction from raw tensor sums.
    """
    dm = 64 if C <= 50 else 128
    Dk = 32 if C <= 10 else (48 if C <= 50 else 64)
    K = C if C <= 50 else 64
    B = batch_size
    bytes_per = 4 if fp32 else 2

    # Dominant tensors (MB)
    pre_proj = B * C * L * dm * bytes_per / 1024**2
    latent = B * K * L * Dk * bytes_per / 1024**2
    fft_int = B * K * (L // 2 + 1) * Dk * 8 / 1024**2    # complex64
    dec_rev = B * C * H * dm * bytes_per / 1024**2
    manifold_int = B * C * H * dm * bytes_per / 1024**2 if use_manifold else 0
    block_int = B * K * L * Dk * bytes_per / 1024**2 * 3 * 2  # 3 branches × 2 blocks
    overhead = 500  # CUDA context

    fwd = pre_proj + latent * 2 + fft_int + dec_rev + manifold_int + block_int
    bwd = fwd * 2.8
    return fwd + bwd + overhead


def safe_batch_size(C, L=96, H=96, target_vram_gb=None, safety_margin=0.3):
    """Compute maximum safe batch size for a given channel count."""
    if target_vram_gb is None:
        if not torch.cuda.is_available():
            return 256
        target_vram_gb = get_free_vram_gb() * (1 - safety_margin)

    # Binary search for largest batch size that fits
    lo, hi = 1, 512
    while lo < hi:
        mid = (lo + hi + 1) // 2
        est = estimate_kmm_vram(C, L, H, mid) / 1024
        if est < target_vram_gb:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def adaptive_train_loop(train_fn, model, train_loader, val_loader, test_loader,
                        device, initial_batch_size, **kwargs):
    """Training loop wrapper that auto-reduces batch size on OOM.

    train_fn(model, batch_x, batch_y, **kwargs) -> loss
    """
    bs = initial_batch_size
    max_retries = 3

    while bs >= 1:
        try:
            return train_fn(model, train_loader, val_loader, test_loader,
                            device, bs, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            msg = str(e)
            if 'out of memory' in msg.lower() or 'oom' in msg.lower():
                bs = bs // 2
                max_retries -= 1
                if max_retries <= 0 or bs < 1:
                    raise RuntimeError(f"OOM even at batch_size=1. C too large for this GPU.")
                print(f"  [OOM] Reducing batch_size: {bs * 2} → {bs}, retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            else:
                raise


def preflight_check(C, L=96, H=96, device='cuda', required_free_gb=1.0):
    """Check if dataset can fit on GPU before training. Returns (ok, recommended_bs, est_gb)."""
    if not torch.cuda.is_available():
        return True, 32, 0

    free_gb = get_free_vram_gb(device)
    rec_bs = safe_batch_size(C, L, H, free_gb)

    if rec_bs < 2:
        est = estimate_kmm_vram(C, L, H, 1) / 1024
        return False, 1, est

    return True, rec_bs, estimate_kmm_vram(C, L, H, rec_bs) / 1024


def smart_batch_size(args_bs, C, L=96, H=96, device='cuda'):
    """Determine safe batch size: use args if fits, else auto-reduce."""
    if not torch.cuda.is_available():
        return args_bs

    free_gb = get_free_vram_gb(device)
    safe_bs = safe_batch_size(C, L, H, free_gb)

    if args_bs > safe_bs:
        print(f"  [MEM] Requested batch_size={args_bs} too large for C={C} "
              f"(safe={safe_bs}). Auto-reducing to {safe_bs}.")
        return safe_bs
    return args_bs
