"""
mad_clean.filters
=================
FilterBank — holds learned dictionary atoms on a torch device and exposes
precomputed representations needed by both solvers.

Classes
-------
FilterBank
    Wraps a (K, F, F) atom array. Provides:
      - .atoms       : Tensor (K, F, F)  on device
      - .D           : Tensor (F², K)    flattened, for PatchSolver matmul
      - .D_fft       : Tensor            precomputed rfft2 of each atom,
                       shape (K, F, F//2+1) complex, for ConvSolver FISTA
      - .K, .F       : int
      - .device      : torch.device
    Serialisation: save(path) / FilterBank.load(path, device)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


__all__ = ["FilterBank"]


class FilterBank:
    """
    Learned dictionary atoms on a torch device.

    Parameters
    ----------
    atoms  : np.ndarray (K, F, F) float32
             Raw atom array from training. Normalised to unit L2 norm
             per atom on construction.
    device : str | torch.device  (default "cpu")
    """

    def __init__(self, atoms: np.ndarray, device: str | torch.device = "cpu"):
        if atoms.ndim != 3:
            raise ValueError(
                f"atoms must be (K, F, F), got shape {atoms.shape}"
            )
        self.device = torch.device(device)
        self.K, self.F, _ = atoms.shape

        # Normalise: unit L2 norm per atom
        atoms_f = atoms.astype(np.float32)
        norms   = np.linalg.norm(atoms_f.reshape(self.K, -1), axis=1,
                                  keepdims=True)          # (K, 1)
        atoms_f = atoms_f.reshape(self.K, -1) / np.maximum(norms, 1e-8)
        atoms_f = atoms_f.reshape(self.K, self.F, self.F)

        # ── core tensor ───────────────────────────────────────────────────────
        self.atoms: torch.Tensor = torch.from_numpy(atoms_f).to(self.device)
        # (K, F²) — for PatchSolver D @ z matmul
        self.D: torch.Tensor = self.atoms.reshape(self.K, -1).T.contiguous()
        # (F², K) layout: each column is a flattened atom

        # ── precomputed FFT of atoms for ConvSolver FISTA ─────────────────────
        # We precompute rfft2 at a canonical size F×F.
        # ConvSolver will zero-pad to island size at runtime.
        self.D_fft: torch.Tensor = torch.fft.rfft2(self.atoms)
        # shape: (K, F, F//2+1) complex

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save raw atoms (before normalisation) as .npy for portability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.atoms.cpu().numpy())

    @classmethod
    def load(cls, path: str | Path,
             device: str | torch.device = "cpu") -> "FilterBank":
        """Load a FilterBank from a .npy file saved by save()."""
        path  = Path(path)
        atoms = np.load(path).astype(np.float32)
        return cls(atoms, device=device)

    # ── helpers ───────────────────────────────────────────────────────────────

    def to(self, device: str | torch.device) -> "FilterBank":
        """Return a new FilterBank on a different device."""
        return FilterBank(self.atoms.cpu().numpy(), device=device)

    def __repr__(self) -> str:
        return (f"FilterBank(K={self.K}, F={self.F}, "
                f"device={self.device})")

    # ── dead atom diagnostics ─────────────────────────────────────────────────

    def dead_atom_report(self, threshold: float = 0.01) -> dict:
        norms = self.atoms.reshape(self.K, -1).norm(dim=1).cpu().numpy()
        dead  = int((norms < threshold).sum())
        return {
            "n_dead":   dead,
            "n_active": self.K - dead,
            "norm_min": float(norms.min()),
            "norm_mean":float(norms.mean()),
            "norm_max": float(norms.max()),
        }
