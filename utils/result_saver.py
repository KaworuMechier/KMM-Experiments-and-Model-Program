"""
ResultSaver: unified result persistence with file locking.
Ensures concurrent runs don't corrupt shared result files.
"""
import os, json, csv, torch, numpy as np
from .file_lock import safe_save


class ResultSaver:
    """Thread/process-safe result persistence."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_csv(self, filename, rows, header=None):
        """Append rows to CSV with file lock."""
        path = os.path.join(self.output_dir, filename)

        def _write(p):
            existed = os.path.exists(p) and os.path.getsize(p) > 0
            with open(p, 'a', newline='') as f:
                w = csv.writer(f)
                if not existed and header:
                    w.writerow(header)
                w.writerows(rows)

        safe_save(path, _write)

    def save_json(self, filename, data):
        path = os.path.join(self.output_dir, filename)

        def _write(p):
            existing = {}
            if os.path.exists(p):
                with open(p) as f:
                    existing = json.load(f)
            existing.update(data)
            with open(p, 'w') as f:
                json.dump(existing, f, indent=2, default=str)

        safe_save(path, _write)

    def save_model_outputs(self, filename, pred, target, modes, spectrum):
        """Save KMM predictions + extracted Koopman modes."""
        path = os.path.join(self.output_dir, filename)

        def _write(p):
            np.savez_compressed(p,
                                pred=pred.cpu().numpy() if torch.is_tensor(pred) else pred,
                                target=target.cpu().numpy() if torch.is_tensor(target) else target,
                                modes=modes.cpu().numpy() if torch.is_tensor(modes) else modes,
                                spectrum_nu=spectrum[0].cpu().numpy() if torch.is_tensor(spectrum[0]) else spectrum[0],
                                spectrum_theta=spectrum[1].cpu().numpy() if torch.is_tensor(spectrum[1]) else spectrum[1])

        safe_save(path, _write)

    def save_c_obs(self, filename, c_obs):
        """Save C_obs matrix for downstream analysis."""
        path = os.path.join(self.output_dir, filename)

        def _write(p):
            np.savez_compressed(p, c_obs=c_obs.cpu().numpy() if torch.is_tensor(c_obs) else c_obs)

        safe_save(path, _write)
