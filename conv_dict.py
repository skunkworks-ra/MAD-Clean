"""
mad_clean.train.conv_dict
=========================
ConvDictTrainer — pure PyTorch minibatch convolutional dictionary learning (CDL).

Trains on full 150×150 images (not patches) via alternating minimisation:

    Z-step  : FISTA sparse coding per image in minibatch (reuses ConvSolver)
    D-step  : Fourier gradient via PyTorch autograd + Adam + unit-ball projection

Returns a FilterBank with the same interface as PatchDictTrainer — the outer
code (MADClean, ConvSolver) requires no changes.

Algorithm reference: DESIGN.md §7.

Classes
-------
ConvDictTrainer
    .fit(images, device) -> FilterBank
    .save(path, filter_bank)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

from mad_clean.filters import FilterBank


__all__ = ["ConvDictTrainer"]


class ConvDictTrainer:
    """
    Train a convolutional filter bank via minibatch alternating minimisation.

    Algorithm (per minibatch):
        Z-step : for each image, run FISTA (ConvSolver) with current D → Z_i (K,H,W)
        D-step : pad D to image size, compute recon = Σ_k d_k ⊛ z_k via FFT,
                 backprop loss, Adam step, project each atom onto unit L2 ball.

    Parameters
    ----------
    k                : int    number of filters (required — no default; choose after Variant A)
    atom_size        : int    filter size in pixels (required)
    batch_size       : int    images per minibatch (default 8)
    n_epochs         : int    training epochs (default 20)
    lr_d             : float  Adam learning rate for D (default 1e-3)
    lmbda            : float  FISTA L1 penalty — matches ConvSolver inference default (default 0.1)
    fista_iter_train : int    FISTA iterations per Z-step — fewer than inference (default 50)
    tol              : float  FISTA early-stopping tolerance (default 1e-4)
    random_seed      : int    (default 42)
    """

    def __init__(
        self,
        k                : int,
        atom_size        : int,
        batch_size       : int   = 8,
        n_epochs         : int   = 20,
        lr_d             : float = 1e-3,
        lmbda            : float = 0.1,
        fista_iter_train : int   = 50,
        tol              : float = 1e-4,
        random_seed      : int   = 42,
    ):
        self.k                = k
        self.atom_size        = atom_size
        self.batch_size       = batch_size
        self.n_epochs         = n_epochs
        self.lr_d             = lr_d
        self.lmbda            = lmbda
        self.fista_iter_train = fista_iter_train
        self.tol              = tol
        self.random_seed      = random_seed

    def fit(
        self,
        images : np.ndarray,
        device : str = "cpu",
    ) -> FilterBank:
        """
        Train on (N, H, W) float32 images. Returns a FilterBank.

        Parameters
        ----------
        images : np.ndarray (N, H, W) float32, values in [0, 1]
        device : torch device for computation and the returned FilterBank
        """
        # Import here to avoid circular import (ConvSolver imports FilterBank)
        from mad_clean.solvers import ConvSolver

        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        K, F = self.k, self.atom_size
        N, H, W = images.shape

        # Per-image normalise: zero mean, unit variance — same as PatchDictTrainer.
        # Without this, MSE is dominated by background zeros and atoms learn nothing.
        img_mean = images.mean(axis=(1, 2), keepdims=True)
        img_std  = images.std(axis=(1, 2), keepdims=True) + 1e-8
        images   = (images - img_mean) / img_std

        print(f"ConvDictTrainer: K={K}  F={F}  images={N}×{H}×{W}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")

        # ── initialise D: random (K, F, F), each atom unit L2 norm ───────────
        D_init = rng.standard_normal((K, F, F)).astype(np.float32)
        norms  = np.linalg.norm(D_init.reshape(K, -1), axis=1, keepdims=True)
        D_init /= norms.reshape(K, 1, 1)

        D         = torch.nn.Parameter(torch.from_numpy(D_init).to(dev))
        optimizer = torch.optim.Adam([D], lr=self.lr_d)

        # ── training loop ─────────────────────────────────────────────────────
        for epoch in range(self.n_epochs):
            idx          = rng.permutation(N)
            epoch_loss   = 0.0
            epoch_spar   = 0.0
            n_batches    = 0

            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]
                batch_np  = images[batch_idx]
                batch     = torch.from_numpy(batch_np).float().to(dev)
                B         = len(batch)

                # ── Z-step: FISTA per image, no gradient through D ────────────
                # Create a temporary FilterBank from current D so ConvSolver
                # can compute its atom FFTs and Lipschitz constant.
                # FilterBank normalises atoms — since we project onto unit ball
                # after each D-step, this is essentially a no-op (norms ≈ 1).
                with torch.no_grad():
                    fb_tmp = FilterBank(D.detach().cpu().numpy(), device=device)
                    solver = ConvSolver(
                        fb_tmp,
                        lmbda  = self.lmbda,
                        n_iter = self.fista_iter_train,
                        tol    = self.tol,
                    )
                    Z_list = [solver.encode_island(batch[i]) for i in range(B)]
                    # Z_list[i]: (K, H_w, W_w) — H_w, W_w >= H, W (may be padded)
                    # Crop to (K, H, W) so the D-step forward model is consistent.
                    Z = torch.stack([z[:, :H, :W] for z in Z_list]).detach()
                    # Z: (B, K, H, W) float32, no grad

                # ── D-step: forward model via FFT, backprop, Adam, project ─────
                optimizer.zero_grad()

                # Pad D from (K, F, F) to (K, H, W) for full-image convolution.
                D_padded  = F_.pad(D, (0, W - F, 0, H - F))          # (K, H, W)
                D_fft     = torch.fft.rfft2(D_padded)                 # (K, H, W//2+1)
                Z_fft     = torch.fft.rfft2(Z)                        # (B, K, H, W//2+1)
                recon_fft = (D_fft.unsqueeze(0) * Z_fft).sum(dim=1)  # (B, H, W//2+1)
                recon     = torch.fft.irfft2(recon_fft, s=(H, W))    # (B, H, W)

                loss = 0.5 * ((recon - batch) ** 2).mean()
                loss.backward()
                optimizer.step()

                # Project each atom onto unit L2 ball: d_k ← d_k / max(‖d_k‖, 1)
                with torch.no_grad():
                    norms_t = D.data.reshape(K, -1).norm(dim=1)   # (K,)
                    scale   = norms_t.clamp(min=1.0)
                    D.data /= scale.reshape(K, 1, 1)

                epoch_loss += float(loss)
                epoch_spar += float((Z.abs() < 1e-6).float().mean())
                n_batches  += 1

            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}  "
                  f"sparsity={epoch_spar / n_batches:.4f}")

        # ── build final FilterBank ────────────────────────────────────────────
        fb     = FilterBank(D.detach().cpu().numpy(), device=device)
        report = fb.dead_atom_report()
        print(f"ConvDictTrainer complete  "
              f"active atoms: {report['n_active']}/{K}  "
              f"norm mean={report['norm_mean']:.3f}")
        if report["n_dead"] > 0:
            print(f"  WARNING: {report['n_dead']} dead atoms — "
                  f"consider more epochs or lower lmbda")
        return fb

    def save(self, path: str | Path, filter_bank: FilterBank) -> None:
        """Save the FilterBank to a .npy file."""
        filter_bank.save(path)
        print(f"Saved ConvDictTrainer FilterBank → {path}")

    def __repr__(self) -> str:
        return (f"ConvDictTrainer(k={self.k}, atom_size={self.atom_size}, "
                f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
                f"lr_d={self.lr_d}, lmbda={self.lmbda})")
