"""
mad_clean.solvers
=================
Sparse coding solvers for MAD-CLEAN minor cycle.

Classes
-------
PatchSolver
    Variant A — patch dictionary. Tiles an island into overlapping patches,
    decodes each via OMP (sklearn) with PyTorch matmul for reconstruction,
    accumulates via torch.nn.functional.fold. No SPORCO dependency.

ConvSolver
    Variant B — convolutional dictionary. Decodes an island via FISTA
    (Fast Iterative Shrinkage-Thresholding Algorithm) entirely in PyTorch.
    Replaces SPORCO ConvBPDN. No external dependency beyond torch.

Both solvers accept and return torch.Tensor on the FilterBank's device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from mad_clean.filters import FilterBank


__all__ = ["PatchSolver", "ConvSolver"]


# ── Variant A — Patch Dictionary Solver ───────────────────────────────────────

class PatchSolver:
    """
    Patch dictionary sparse coding via OMP + PyTorch fold/unfold.

    Parameters
    ----------
    filter_bank : FilterBank
    n_nonzero   : int   OMP sparsity — max active atoms per patch (default 5)
    stride      : int   tiling stride in pixels (default 8)
    """

    def __init__(
        self,
        filter_bank : FilterBank,
        n_nonzero   : int = 5,
        stride      : int = 8,
    ):
        self.fb        = filter_bank
        self.n_nonzero = n_nonzero
        self.stride    = stride
        self.F         = filter_bank.F
        self.device    = filter_bank.device

        # D on CPU for sklearn OMP — we move recon back to GPU
        self._D_cpu = filter_bank.D.cpu().numpy()   # (F², K)

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """
        Decode a single source island.

        Parameters
        ----------
        island : Tensor (H_i, W_i) float32, on self.device

        Returns
        -------
        recon : Tensor (H_i, W_i) float32, on self.device
        """
        H, W = island.shape
        F    = self.F
        s    = self.stride

        # ── extract patches via unfold ─────────────────────────────────────
        # unfold expects (1, 1, H, W)
        island_4d = island.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        patches   = F_.unfold(island_4d, kernel_size=F, stride=s)
        # patches: (1, F², n_patches) — squeeze to (n_patches, F²)
        patches   = patches.squeeze(0).T.contiguous()  # (n_patches, F²)
        n_patches = patches.shape[0]

        # ── OMP on CPU (sklearn) — loop over patches ───────────────────────
        from sklearn.linear_model import orthogonal_mp

        patches_np = patches.cpu().numpy()   # (n_patches, F²)

        # Normalise each patch before OMP, denormalise after
        p_mean = patches_np.mean(axis=1, keepdims=True)   # (n_patches, 1)
        p_std  = patches_np.std (axis=1, keepdims=True) + 1e-8
        patches_n = (patches_np - p_mean) / p_std         # (n_patches, F²)

        # OMP: solve for all patches simultaneously (sklearn supports batching)
        # D_cpu is (F², K); patches_n.T is (F², n_patches)
        Z = orthogonal_mp(
            self._D_cpu,
            patches_n.T,
            n_nonzero_coefs=self.n_nonzero,
        )   # (K, n_patches)

        # ── reconstruct patches in PyTorch ────────────────────────────────
        Z_t     = torch.from_numpy(Z.T).float().to(self.device)  # (n_patches, K)
        D_t     = self.fb.D                                       # (F², K)
        # recon_patches: (n_patches, F²)
        recon_n = Z_t @ D_t.T
        # Denormalise
        p_mean_t = torch.from_numpy(p_mean).float().to(self.device)  # (n_patches,1)
        p_std_t  = torch.from_numpy(p_std ).float().to(self.device)
        recon_patches = recon_n * p_std_t + p_mean_t   # (n_patches, F²)

        # ── fold: overlap-average back to island shape ─────────────────────
        # fold expects (1, F², n_patches)
        recon_4d = recon_patches.T.unsqueeze(0)         # (1, F², n_patches)
        output_size = (H, W)

        folded = F_.fold(recon_4d, output_size=output_size,
                         kernel_size=F, stride=s)       # (1, 1, H, W)
        # Divisor: count how many patches contributed to each pixel
        ones    = torch.ones_like(recon_4d)
        divisor = F_.fold(ones, output_size=output_size,
                          kernel_size=F, stride=s)
        divisor = divisor.clamp(min=1.0)

        result = (folded / divisor).squeeze()           # (H, W)
        return result

    def __repr__(self) -> str:
        return (f"PatchSolver(K={self.fb.K}, F={self.F}, "
                f"n_nonzero={self.n_nonzero}, stride={self.stride}, "
                f"device={self.device})")


# ── Variant B — Convolutional FISTA Solver ────────────────────────────────────

class ConvSolver:
    """
    Convolutional sparse coding via FISTA (pure PyTorch, no SPORCO).

    Solves:
        min_{Z_k}  (1/2) ||Σ_k d_k ⊛ z_k  -  island||²  +  λ Σ_k ||z_k||_1

    where {d_k} are the K convolutional filters (atoms) and {z_k} are the
    2D activation maps (one per filter, same spatial size as island).

    Algorithm: FISTA with analytic Lipschitz constant from filter FFTs.
    All operations in PyTorch — runs on GPU if FilterBank is on GPU.

    Parameters
    ----------
    filter_bank : FilterBank
    lmbda       : float  L1 sparsity penalty (default 0.1)
    n_iter      : int    FISTA iterations (default 100)
    tol         : float  early stopping — relative change in objective (default 1e-4)
    """

    def __init__(
        self,
        filter_bank : FilterBank,
        lmbda       : float = 0.1,
        n_iter      : int   = 100,
        tol         : float = 1e-4,
    ):
        self.fb     = filter_bank
        self.lmbda  = lmbda
        self.n_iter = n_iter
        self.tol    = tol
        self.F      = filter_bank.F
        self.K      = filter_bank.K
        self.device = filter_bank.device

    def _lipschitz(self, atoms_fft: torch.Tensor, sig_shape: tuple) -> float:
        """
        Compute Lipschitz constant L of the data fidelity gradient.

        For convolutional forward model A: z → Σ_k d_k ⊛ z_k,
        L = max eigenvalue of A^T A = spectral norm of A.

        In the Fourier domain, A^T A is diagonal with entries
        Σ_k |D_k(ω)|² per frequency ω. L = max_ω Σ_k |D_k(ω)|².

        atoms_fft : (K, H, W//2+1) complex  — FFT of atoms zero-padded to sig_shape
        """
        power = (atoms_fft.abs() ** 2).sum(dim=0)   # (H, W//2+1) summed over K
        return float(power.max().real) + 1e-8

    def _prepare_atoms_fft(
        self,
        island: torch.Tensor,
    ) -> tuple:
        """
        Handle atom/island size mismatch; return atoms_fft and a working copy
        of the island padded to at least (F, F).

        Returns
        -------
        atoms_fft  : Tensor (K, H, W//2+1) complex
        island_w   : Tensor (H, W) — possibly zero-padded copy of island
        H, W       : int — working dimensions
        orig_H, orig_W : int — original island dimensions (for final crop)
        """
        F          = self.F
        orig_H, orig_W = island.shape
        island_w   = island

        # If island is smaller than atom, pad the island (copy — no mutation)
        if orig_H < F or orig_W < F:
            ph       = max(0, F - orig_H)
            pw       = max(0, F - orig_W)
            island_w = F_.pad(island, (0, pw, 0, ph))

        H, W   = island_w.shape
        pad_h  = H - F
        pad_w  = W - F

        atoms_padded = F_.pad(self.fb.atoms, (0, pad_w, 0, pad_h))   # (K, H, W)
        atoms_fft    = torch.fft.rfft2(atoms_padded)                  # (K, H, W//2+1)

        return atoms_fft, island_w, H, W, orig_H, orig_W

    def _run_fista(
        self,
        atoms_fft : torch.Tensor,   # (K, H, W//2+1) complex
        island    : torch.Tensor,   # (H, W) spatial — used for gradient and stopping
        H         : int,
        W         : int,
    ) -> torch.Tensor:
        """
        Run FISTA and return activation maps Z (K, H, W).

        This is the shared kernel used by both decode_island (inference) and
        encode_island (CDL Z-step during training).
        """
        island_fft = torch.fft.rfft2(island)
        step       = 1.0 / self._lipschitz(atoms_fft, (H, W))
        K          = self.K
        dev        = self.device

        Z        = torch.zeros(K, H, W, dtype=torch.float32, device=dev)
        Y        = Z.clone()
        t        = 1.0
        prev_obj = float("inf")

        for _ in range(self.n_iter):
            # ── gradient of data fidelity at Y ────────────────────────────
            Y_fft        = torch.fft.rfft2(Y)
            recon_fft    = (atoms_fft * Y_fft).sum(dim=0)
            residual_fft = recon_fft - island_fft
            grad_fft     = atoms_fft.conj() * residual_fft.unsqueeze(0)
            grad         = torch.fft.irfft2(grad_fft, s=(H, W))

            # ── proximal gradient + FISTA momentum ────────────────────────
            Z_new = self._soft_threshold(Y - step * grad, self.lmbda * step)
            t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            Y     = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
            Z     = Z_new
            t     = t_new

            # ── early stopping on relative objective change ───────────────
            with torch.no_grad():
                recon_fft_new = (atoms_fft * torch.fft.rfft2(Z)).sum(dim=0)
                recon_new     = torch.fft.irfft2(recon_fft_new, s=(H, W))
                data_fid      = 0.5 * ((recon_new - island) ** 2).sum()
                l1_term       = self.lmbda * Z.abs().sum()
                obj           = float(data_fid + l1_term)
                if abs(prev_obj - obj) / (abs(prev_obj) + 1e-8) < self.tol:
                    break
                prev_obj = obj

        return Z

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """
        Decode a single source island via FISTA; return reconstruction.

        Parameters
        ----------
        island : Tensor (H_i, W_i) float32, on self.device

        Returns
        -------
        recon : Tensor (H_i, W_i) float32 — Σ_k d_k ⊛ z_k*
        """
        atoms_fft, island_w, H, W, orig_H, orig_W = self._prepare_atoms_fft(island)
        Z      = self._run_fista(atoms_fft, island_w, H, W)
        Z_fft  = torch.fft.rfft2(Z)
        recon  = torch.fft.irfft2((atoms_fft * Z_fft).sum(dim=0), s=(H, W))
        return recon[:orig_H, :orig_W].float()

    def encode_island(self, island: torch.Tensor) -> torch.Tensor:
        """
        Encode a single island via FISTA; return activation maps Z.

        Used by ConvDictTrainer for the Z-step during CDL training.

        Parameters
        ----------
        island : Tensor (H_i, W_i) float32, on self.device

        Returns
        -------
        Z : Tensor (K, H, W) float32 — sparse activation maps
            H, W are the working dimensions (>= island size, >= F).
        """
        atoms_fft, island_w, H, W, _, _ = self._prepare_atoms_fft(island)
        return self._run_fista(atoms_fft, island_w, H, W)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Element-wise soft thresholding: sign(x) * max(|x| - threshold, 0)."""
        return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0.0)

    def __repr__(self) -> str:
        return (f"ConvSolver(K={self.K}, F={self.F}, "
                f"lmbda={self.lmbda}, n_iter={self.n_iter}, "
                f"device={self.device})")


# ── module-level alias to avoid shadowing torch.nn.functional ─────────────────
F_ = F
