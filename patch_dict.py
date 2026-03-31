"""
mad_clean.train.patch_dict
==========================
PatchDictTrainer — pure PyTorch full-batch patch dictionary learning.

Trains on normalised patches via alternating minimisation:

    Z-step  : full-batch FISTA sparse coding  (all patches, GPU)
    D-step  : autograd + Adam + unit-ball projection

Returns a FilterBank with the same interface as ConvDictTrainer — the outer
code (MADClean, PatchSolver) requires no changes.

Classes
-------
PatchDictTrainer
    .fit(images, device) -> FilterBank
    .save(path, filter_bank)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.feature_extraction.image import extract_patches_2d

from mad_clean.filters import FilterBank


__all__ = ["PatchDictTrainer"]


class PatchDictTrainer:
    """
    Train a patch dictionary on CRUMB-style radio galaxy images.

    Algorithm (per epoch):
        Z-step : full-batch FISTA over all patches with fixed D
                 solves  min_Z  0.5 ||Z D - P||² + lmbda ||Z||_1
        D-step : backprop reconstruction loss, Adam step,
                 project each atom onto unit L2 ball

    Parameters
    ----------
    k               : int    number of atoms (default 128)
    atom_size       : int    patch / atom size in pixels (default 15)
    lmbda           : float  FISTA L1 sparsity penalty (default 0.1)
    n_epochs        : int    alternating minimisation epochs (default 50)
    fista_iter      : int    FISTA iterations per Z-step (default 100)
    lr_d            : float  Adam learning rate for D (default 1e-3)
    patches_per_img : int    random patches extracted per image (default 50)
    tol             : float  FISTA early-stop tolerance (default 1e-6)
    random_seed     : int    (default 42)
    """

    def __init__(
        self,
        k               : int   = 128,
        atom_size       : int   = 15,
        lmbda           : float = 0.1,
        n_epochs        : int   = 50,
        fista_iter      : int   = 100,
        lr_d            : float = 1e-3,
        patches_per_img : int   = 50,
        tol             : float = 1e-6,
        random_seed     : int   = 42,
    ):
        self.k               = k
        self.atom_size       = atom_size
        self.lmbda           = lmbda
        self.n_epochs        = n_epochs
        self.fista_iter      = fista_iter
        self.lr_d            = lr_d
        self.patches_per_img = patches_per_img
        self.tol             = tol
        self.random_seed     = random_seed
        self._D_final        = None   # (K, F²) after fit()

    def fit(
        self,
        images : np.ndarray,
        device : str = "cpu",
    ) -> FilterBank:
        """
        Train on (N, H, W) float32 images. Returns a FilterBank.

        Parameters
        ----------
        images : np.ndarray (N, H, W) float32
        device : torch device for computation and the returned FilterBank
        """
        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        K, F = self.k, self.atom_size

        # ── 80/20 train split ─────────────────────────────────────────────────
        n   = len(images)
        idx = rng.permutation(n)
        train = images[idx[: int(0.8 * n)]]

        # ── extract and normalise patches ─────────────────────────────────────
        print(f"Extracting patches  "
              f"(n_images={len(train)}, patches/img={self.patches_per_img}) …")
        patches_np = self._extract_patches(train, rng)
        print(f"  Total patches: {len(patches_np)}  shape: {F}×{F}")

        p_mean = patches_np.mean(axis=1, keepdims=True)
        p_std  = patches_np.std(axis=1, keepdims=True) + 1e-8
        patches_n = (patches_np - p_mean) / p_std          # (N_p, F²)

        P = torch.from_numpy(patches_n).float().to(dev)    # (N_p, F²)
        N_p = P.shape[0]

        print(f"PatchDictTrainer: K={K}  F={F}  patches={N_p}  "
              f"epochs={self.n_epochs}  fista_iter={self.fista_iter}  "
              f"lmbda={self.lmbda}  device={dev}")

        # ── initialise D: (K, F²), each row unit L2 norm ──────────────────────
        D_init = rng.standard_normal((K, F * F)).astype(np.float32)
        norms  = np.linalg.norm(D_init, axis=1, keepdims=True)
        D_init /= norms

        D         = torch.nn.Parameter(torch.from_numpy(D_init).to(dev))
        optimizer = torch.optim.Adam([D], lr=self.lr_d)

        # ── alternating minimisation ──────────────────────────────────────────
        for epoch in range(self.n_epochs):

            # ── Z-step: full-batch FISTA with D fixed ─────────────────────────
            with torch.no_grad():
                D_fixed = D.detach()                       # (K, F²)
                # Lipschitz constant: largest singular value² of D  (upper bound)
                L = float((D_fixed ** 2).sum())            # ||D||_F² ≥ σ_max²
                eta = 1.0 / (L + 1e-8)

                Z = torch.zeros(N_p, K, device=dev)
                Y = Z.clone()
                t = 1.0
                prev_obj = float("inf")

                for _ in range(self.fista_iter):
                    grad = (Y @ D_fixed - P) @ D_fixed.T   # (N_p, K)
                    Z_new = _soft_threshold(Y - eta * grad, eta * self.lmbda)

                    t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
                    Y     = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
                    Z     = Z_new
                    t     = t_new

                    obj = float(0.5 * ((Z @ D_fixed - P) ** 2).mean()
                                + self.lmbda * Z.abs().mean())
                    if abs(prev_obj - obj) / (abs(prev_obj) + 1e-8) < self.tol:
                        break
                    prev_obj = obj

            # ── D-step: backprop + Adam + unit-ball projection ─────────────────
            optimizer.zero_grad()
            recon = Z @ D                                  # (N_p, F²)
            loss  = 0.5 * ((recon - P) ** 2).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                norms_t = D.data.norm(dim=1, keepdim=True)  # (K, 1)
                D.data /= norms_t.clamp(min=1.0)

            sparsity = float((Z.abs() < 1e-6).float().mean())
            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={float(loss):.6f}  "
                  f"sparsity={sparsity:.3f}")

        # ── build FilterBank from final D ──────────────────────────────────────
        self._D_final = D.detach().cpu().numpy()           # (K, F²)
        atoms = self._D_final.reshape(K, F, F).astype(np.float32)

        fb     = FilterBank(atoms, device=device)
        report = fb.dead_atom_report()
        print(f"PatchDictTrainer complete  "
              f"active atoms: {report['n_active']}/{K}  "
              f"norm mean={report['norm_mean']:.3f}")
        if report["n_dead"] > 0:
            print(f"  WARNING: {report['n_dead']} dead atoms — "
                  f"consider lower lmbda or more epochs")
        return fb

    def save(self, path: str | Path, filter_bank: FilterBank) -> None:
        """Save the FilterBank atoms to a .npy file."""
        filter_bank.save(path)
        print(f"Saved PatchDictTrainer FilterBank → {path}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _extract_patches(
        self,
        images : np.ndarray,
        rng    : np.random.Generator,
    ) -> np.ndarray:
        """Extract random patches with random full-image rotation."""
        A = self.atom_size
        all_patches = []
        for img in images:
            angle   = rng.uniform(0, 360)
            img_rot = rotate(img, angle, reshape=False, order=1, mode="reflect")
            patches = extract_patches_2d(
                img_rot,
                patch_size=(A, A),
                max_patches=self.patches_per_img,
                random_state=int(rng.integers(0, 2**31)),
            )
            all_patches.append(patches.reshape(len(patches), -1))
        return np.vstack(all_patches).astype(np.float32)

    def __repr__(self) -> str:
        return (f"PatchDictTrainer(k={self.k}, atom_size={self.atom_size}, "
                f"lmbda={self.lmbda}, n_epochs={self.n_epochs}, "
                f"patches_per_img={self.patches_per_img})")


# ── helpers ────────────────────────────────────────────────────────────────────

def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return x.sign() * (x.abs() - threshold).clamp(min=0.0)
