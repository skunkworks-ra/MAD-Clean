"""
mad_clean.training.conv
=======================
ConvDictTrainer — pure PyTorch minibatch convolutional dictionary learning (CDL).

Trains on full 150×150 images (not patches) via alternating minimisation:

    Z-step  : FISTA sparse coding per image in minibatch (reuses ConvSolver)
    D-step  : Fourier gradient via PyTorch autograd + Adam + unit-ball projection

Returns a FilterBank with the same interface as PatchDictTrainer.

Classes
-------
ConvDictTrainer
    .fit(dirty, psf, device) -> FilterBank
    .save(path, filter_bank)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

from ..filters import FilterBank
from .._utils import soft_threshold


__all__ = ["ConvDictTrainer"]


def _z_step_psf(
    dirty_img   : torch.Tensor,   # (H, W)
    atoms_fft   : torch.Tensor,   # (K, H, W//2+1) complex
    psf_fft     : torch.Tensor,   # (H, W//2+1) complex
    lmbda       : float,
    n_iter      : int,
    tol         : float,
    K           : int,
    H           : int,
    W           : int,
    dev         : torch.device,
) -> torch.Tensor:
    """
    FISTA Z-step with PSF in the forward model (per image).

    Minimises:  0.5 * ||PSF * (Σ_k d_k * z_k) - dirty||²  +  λ||Z||₁

    The effective atoms seen by FISTA are PSF * d_k (in Fourier domain).
    Returns Z: (K, H, W) — activation maps in the pre-PSF (clean) domain.
    """
    eff_fft  = psf_fft.unsqueeze(0) * atoms_fft          # (K, H, W//2+1)
    L        = float((eff_fft.abs() ** 2).sum(0).max().real) + 1e-8
    step     = 1.0 / L

    dirty_fft = torch.fft.rfft2(dirty_img)

    Z = torch.zeros(K, H, W, device=dev)
    Y = Z.clone()
    t = 1.0
    prev_obj = float("inf")

    for _ in range(n_iter):
        Y_fft    = torch.fft.rfft2(Y)
        recon_fft = (eff_fft * Y_fft).sum(0)
        res_fft   = recon_fft - dirty_fft
        grad_fft  = eff_fft.conj() * res_fft
        grad      = torch.fft.irfft2(grad_fft, s=(H, W))

        Z_new  = soft_threshold(Y - step * grad, lmbda * step)
        t_new  = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        Y      = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
        Z      = Z_new
        t      = t_new

        with torch.no_grad():
            Z_fft_c    = torch.fft.rfft2(Z)
            r_fft_c    = (eff_fft * Z_fft_c).sum(0) - dirty_fft
            r_c        = torch.fft.irfft2(r_fft_c, s=(H, W))
            obj        = float(0.5 * (r_c ** 2).sum() + lmbda * Z.abs().sum())
            if abs(prev_obj - obj) / (abs(prev_obj) + 1e-8) < tol:
                break
            prev_obj = obj

    return Z


class ConvDictTrainer:
    """
    Train a convolutional filter bank via minibatch alternating minimisation.

    Algorithm (per minibatch):
        Z-step : for each image, run PSF-aware FISTA with current D → Z_i (K,H,W)
        D-step : pad D to image size, compute recon = PSF*(Σ_k d_k ⊛ z_k) via FFT,
                 backprop PSF-residual loss, Adam step, project atoms onto unit L2 ball.

    Parameters
    ----------
    k                : int    number of filters (required)
    atom_size        : int    filter size in pixels (required)
    batch_size       : int    images per minibatch (default 8)
    n_epochs         : int    training epochs (default 20)
    lr_d             : float  Adam learning rate for D (default 1e-3)
    lmbda            : float  FISTA L1 penalty (default 0.1)
    fista_iter_train : int    FISTA iterations per Z-step (default 50)
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
        dirty  : np.ndarray,
        psf    : np.ndarray,
        device : str = "cpu",
    ) -> FilterBank:
        """
        Train on (N, H, W) dirty images with PSF-residual loss.

        Minimises:  0.5 * ||PSF * (Σ_k d_k * z_k) - dirty||²  +  λ||Z||₁

        Parameters
        ----------
        dirty  : np.ndarray (N, H, W) float32 — PSF-convolved observations
        psf    : np.ndarray (H, W)    float32 — peak-normalised PSF (peak = 1)
        device : torch device for computation and the returned FilterBank
        """
        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        K, F = self.k, self.atom_size
        N, H, W = dirty.shape

        psf_t       = torch.from_numpy(psf).float().to(dev)
        psf_fft     = torch.fft.rfft2(torch.fft.ifftshift(psf_t), s=(H, W))

        print(f"ConvDictTrainer (PSF-residual): K={K}  F={F}  images={N}×{H}×{W}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")

        D_init = rng.standard_normal((K, F, F)).astype(np.float32)
        norms  = np.linalg.norm(D_init.reshape(K, -1), axis=1, keepdims=True)
        D_init /= norms.reshape(K, 1, 1)

        D         = torch.nn.Parameter(torch.from_numpy(D_init).to(dev))
        optimizer = torch.optim.Adam([D], lr=self.lr_d)

        for epoch in range(self.n_epochs):
            idx        = rng.permutation(N)
            epoch_loss = 0.0
            epoch_spar = 0.0
            n_batches  = 0

            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]
                batch_np  = dirty[batch_idx]
                batch     = torch.from_numpy(batch_np).float().to(dev)
                B         = len(batch)

                with torch.no_grad():
                    D_padded_z = F_.pad(D.detach(), (0, W - F, 0, H - F))
                    atoms_fft  = torch.fft.rfft2(D_padded_z)
                    Z_list = [
                        _z_step_psf(
                            batch[i], atoms_fft, psf_fft,
                            self.lmbda, self.fista_iter_train, self.tol,
                            K, H, W, dev,
                        )
                        for i in range(B)
                    ]
                    Z = torch.stack(Z_list).detach()

                optimizer.zero_grad()

                D_padded  = F_.pad(D, (0, W - F, 0, H - F))
                D_fft     = torch.fft.rfft2(D_padded)
                D_psf_fft = psf_fft.unsqueeze(0) * D_fft
                Z_fft     = torch.fft.rfft2(Z)
                recon_fft = (D_psf_fft.unsqueeze(0) * Z_fft).sum(dim=1)
                recon     = torch.fft.irfft2(recon_fft, s=(H, W))

                loss = 0.5 * ((recon - batch) ** 2).mean()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    norms_t = D.data.reshape(K, -1).norm(dim=1)
                    D.data /= norms_t.clamp(min=1.0).reshape(K, 1, 1)

                epoch_loss += float(loss)
                epoch_spar += float((Z.abs() < 1e-6).float().mean())
                n_batches  += 1

            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}  "
                  f"sparsity={epoch_spar / n_batches:.4f}", flush=True)

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
        """Save the FilterBank."""
        filter_bank.save(path)
        print(f"Saved ConvDictTrainer FilterBank → {path}")

    def __repr__(self) -> str:
        return (f"ConvDictTrainer(k={self.k}, atom_size={self.atom_size}, "
                f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, "
                f"lr_d={self.lr_d}, lmbda={self.lmbda})")
