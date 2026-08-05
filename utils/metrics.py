"""Metrics computation: MSE, MAE, TSL, and Lyapunov estimation."""
import torch, numpy as np


def compute_metrics(pred, target, train_mean=None, train_std=None):
    """Compute MSE, MAE, and TSL (normalized MSE)."""
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    result = {'mse': mse, 'mae': mae}
    if train_mean is not None and train_std is not None:
        pred_tsl = (pred - train_mean) / train_std
        target_tsl = (target - train_mean) / train_std
        result['tsl'] = ((pred_tsl - target_tsl) ** 2).mean().item()
    return result


def compute_lyapunov_from_spectrum(model, dt=0.01):
    """Estimate Lyapunov exponents from learned Koopman eigenvalues.

    L_d = log(|λ_d|) / dt

    For chaotic systems, the largest positive L_d ≈ max Lyapunov exponent.
    """
    r = torch.exp(-torch.exp(model.spectrum.nu)).detach().cpu()
    lyap = torch.log(r.clamp(min=1e-8)) / dt
    return lyap.numpy()
