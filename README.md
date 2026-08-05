# KMM: Koopman Mode Mixer

**Learning System Representations for Multivariate Time Series Forecasting, Diagnosis, and Control**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

KMM is a time series architecture that learns a task-agnostic Koopman representation
(eigenvalues, frequencies, observation matrix) from which **forecasting, system diagnosis,
sensor selection, and optimal control** all emerge as different readings of the same
learned parameters — without retraining.

> *"Stop forecasting. Start identifying."*

---

##  Paper

**Koopman Mode Mixer: Learning System Representations for Multivariate Time Series
Forecasting, Diagnosis, and Control**

Under review at IEEE TKDE.

---

##  Quick Start

```bash
git clone https://github.com/KaworuMechier/KMM-Experiments-and-Model-Program.git
cd KMM
pip install -r requirements.txt
```

### Train KMM

```bash
# Full benchmark — 4 prediction horizons × 3 seeds
python train/run_kmm.py --dataset ECL
python train/run_kmm.py --dataset Weather
python train/run_kmm.py --dataset all

# Lite configuration for small datasets (C <= 10)
python train/train_ett.py --dataset ETTh1 --pred_len 96
```

### Run Baselines

```bash
python train/run_baselines.py --dataset ECL --models DLinear,TimesNet,PatchTST
```

### Reproduce Experiments

```bash
# System identification — Lorenz-63 chaos diagnosis
python experiments/lorenz/run_lorenz.py --mode kmm

# Information sharing — synthetic latent-mode recovery
python experiments/latent_mode/run_latent_mode.py

# Convex MPC — real-time control on KS equation
python experiments/mpc/compare_mpc.py

# Sensor selection — dynamics-aware vs PCA
python experiments/sensor/sensor_selection_ks.py
```

---

##  Project Structure

```
KMM/
├── models/                    # KMM model architecture
│   ├── koopman_mixer.py       # Main model (diagonal Koopman operator)
│   ├── dmd_denoiser.py        # Neural DMD pre-processing
│   ├── kmm_enhancements.py    # Hilbert AM, AdaptiveMixer, Kalman gate
│   ├── residual_branch.py     # Optional local refinement (C <= 50)
│   └── spectral_denoiser.py   # Learnable low-pass filter
├── datasets/                  # Data loading
│   ├── data_provider.py       # CSV / Lorenz / noise / control data pipeline
│   └── data/                  # Sample datasets (large ones: see below)
├── train/                     # Training scripts
│   ├── run_kmm.py             # Main KMM training (all datasets)
│   ├── train_kmm.py           # Training loop + evaluation
│   ├── train_ett.py           # Lite training for C <= 10 (ETT datasets)
│   ├── run_baselines.py       # Baseline training (DLinear, TimesNet, PatchTST)
│   └── run_ablation.py        # Ablation study
├── experiments/               # Experiment scripts
│   ├── lorenz/                # Lorenz-63 chaos diagnosis
│   ├── latent_mode/           # Synthetic latent-mode recovery
│   ├── noise/                 # Cross-noise generalization
│   ├── mpc/                   # Convex MPC on KS equation
│   └── sensor/                # Sensor selection (KS + Lorenz-96)
├── analysis/                  # Analysis tools
│   ├── max_channels.py        # Channel capacity scaling test
│   └── mode_contribution.py   # Mode contribution analysis
├── utils/                     # Shared utilities
│   ├── metrics.py             # MSE, MAE, TSL
│   ├── memory_guard.py        # VRAM pre-flight check
│   ├── result_saver.py        # CSV/JSON/NPZ persistence
│   └── file_lock.py           # Multi-GPU safe concurrent writes
└── scripts/                   # Plotting and result management
```

---

##  Datasets

Small datasets (< 10 MB) are included in `datasets/data/`. Large datasets must be
downloaded separately:

| Dataset | Channels | Size | Source |
|:---|:---:|:---|:---|
| ETTh1, ETTm1 | 7 | ~3 MB | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | 21 | ~7 MB | [Weather Dataset](https://www.bgc-jena.mpg.de/wetter/) |
| PJM | 10 | ~4 MB | [PJM Interconnection](https://www.pjm.com/) |
| ECL | 321 | ~91 MB | [UCI Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |
| METR-LA | 207 | ~69 MB | [METR-LA](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY | 325 | ~82 MB | [PEMS-BAY](https://github.com/liyaguang/DCRNN) |
| ERA5-EC | 990 | ~249 MB | Proprietary (East China regional reanalysis) |

Place downloaded CSVs in `datasets/data/`.

---

##  Why KMM?

Traditional time series models (PatchTST, TimesNet, DLinear) are black-box predictors — they
output a number. KMM outputs a **reusable system model**. The same learned representation
enables five capabilities that no other architecture provides simultaneously:

| Capability | PatchTST | TimesNet | DLinear | KMM |
|:---|:---:|:---:|:---:|:---:|
| Forecasting | ✓ | ✓ | ✓ | ✓ |
| System fingerprint (chaos vs. stable) | ✗ | ✗ | ✗ | **✓** |
| Information sharing (C_obs) | ✗ | ✗ | ✗ | **✓** |
| Sensor selection (dynamics-aware) | ✗ | ✗ | ✗ | **✓** |
| Convex MPC (closed-form, <1ms) | ✗ | ✗ | ✗ | **✓** |
| Scales to C=990 on consumer GPU | ✗ | ✗ | ✗ | **✓** |

**Key results** (z-score normalized MSE, L=96, 3-seed mean):

| Dataset | C | DLinear | TimesNet | PatchTST | **KMM** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Weather | 21 | 0.196 | 0.172 | 0.175 | **0.164** |
| ECL | 321 | 0.197 | 0.168 | 0.196 | **0.175** |
| ERA5-EC | 990 | OOM | OOM | OOM | **0.359** |

Baseline results from DTSformer (Springer ML 2025) third-party benchmark. ERA5-EC: KMM is
the only model that fits on a consumer RTX 3090 24GB.

---

##  Key Architecture

| Parameter | Symbol | Shape | Meaning |
|:---|:---|:---|:---|
| Eigenvalue magnitude | $\nu_d$ | $D_k$ | Modal persistence ($|\lambda_d| = e^{-e^{\nu_d}}$) |
| Modal frequency | $\theta_d$ | $D_k$ | Intrinsic oscillation frequency |
| Observation matrix | $\mathbf{C}_{\text{obs}}$ | $K \times C$ | Channel-to-mode coupling |

The Koopman operator is **diagonal by construction** — each mode evolves independently:
$\mathbf{K} = \text{diag}(|\lambda_1|, \ldots, |\lambda_D|)$. This enables:

- **Stable long-range forecasting**: H-step prediction via $O(D)$ scalar powers
- **Closed-form convex MPC**: single $H \times H$ matrix inversion, independent of $D$
- **System fingerprinting**: eigenvalue spectrum reveals dynamical regime (chaotic vs. stable)
- **Hardware scalability**: $\mathbf{C}_{\text{obs}}$ compression from $C$ channels to $K$ modes

---

##  Citation

If you use KMM in your research, please cite:

```bibtex
@article{kmm2026,
  title   = {Koopman Mode Mixer: Learning System Representations for Multivariate
             Time Series Forecasting, Diagnosis, and Control},
  author  = {Yuansheng Xu},
  journal = {IEEE Trans. Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

---

##  License

MIT License — see [LICENSE](LICENSE) for details.