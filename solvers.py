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

import numpy as np
import torch
import torch.nn.functional as F
from mad_clean.filters import FilterBank


__all__ = ["PatchSolver", "ConvSolver", "FlowSolver"]


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
        p_mean = patches_np.mean(axis=1, keepdims=True)
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
        p_mean_t = torch.from_numpy(p_mean).float().to(self.device)
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

    Without PSF:
        min_{Z_k}  (1/2) ||Σ_k d_k ⊛ z_k  -  island||²  +  λ Σ_k ||z_k||_1

    With PSF (psf != None):
        min_{Z_k}  (1/2) ||PSF * (Σ_k d_k ⊛ z_k)  -  island||²  +  λ Σ_k ||z_k||_1

        decode_island() returns D*Z (clean-sky estimate), not PSF*(D*Z).
        This matches ConvDictTrainer PSF-residual training.

    Parameters
    ----------
    filter_bank : FilterBank
    lmbda       : float             L1 sparsity penalty (default 0.1)
    n_iter      : int               FISTA iterations (default 100)
    tol         : float             early stopping tolerance (default 1e-4)
    psf         : np.ndarray | None (H_psf, W_psf) peak-normalised PSF.
                                    If given, PSF-residual FISTA is used.
    """

    def __init__(
        self,
        filter_bank : FilterBank,
        lmbda       : float                       = 0.1,
        n_iter      : int                         = 100,
        tol         : float                       = 1e-4,
        psf         : "np.ndarray | None"         = None,
    ):
        self.fb     = filter_bank
        self.lmbda  = lmbda
        self.n_iter = n_iter
        self.tol    = tol
        self.F      = filter_bank.F
        self.K      = filter_bank.K
        self.device = filter_bank.device
        self._psf   = psf  # (H_psf, W_psf) numpy float32, or None

    def _psf_fft(self, H: int, W: int) -> "torch.Tensor | None":
        """
        Centre-crop or zero-pad the stored PSF to (H, W) and return its rfft2.
        Returns None if no PSF was provided at construction.
        """
        if self._psf is None:
            return None
        psf = torch.from_numpy(self._psf).float().to(self.device)
        pH, pW = psf.shape
        # Centre-crop
        if pH > H:
            ch = (pH - H) // 2
            psf = psf[ch : ch + H, :]
        if pW > W:
            cw = (pW - W) // 2
            psf = psf[:, cw : cw + W]
        # Zero-pad
        h, w = psf.shape
        if h < H or w < W:
            out = torch.zeros(H, W, device=self.device)
            ph = (H - h) // 2
            pw = (W - w) // 2
            out[ph : ph + h, pw : pw + w] = psf
            psf = out
        return torch.fft.rfft2(torch.fft.ifftshift(psf))

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
        atoms_fft : torch.Tensor,                    # (K, H, W//2+1) complex
        island    : torch.Tensor,                    # (H, W)
        H         : int,
        W         : int,
        psf_fft   : "torch.Tensor | None" = None,   # (H, W//2+1) complex, or None
    ) -> torch.Tensor:
        """
        Run FISTA and return activation maps Z (K, H, W).

        Without psf_fft: minimises  0.5||D*Z - island||² + λ||Z||₁
        With    psf_fft: minimises  0.5||PSF*(D*Z) - island||² + λ||Z||₁
                         The effective atoms are PSF*d_k in the Fourier domain.

        Shared by decode_island (inference) and encode_island (CDL Z-step).
        """
        # Effective atoms: PSF * D  (or D if no PSF)
        eff_fft    = (psf_fft.unsqueeze(0) * atoms_fft) if psf_fft is not None \
                     else atoms_fft

        island_fft = torch.fft.rfft2(island)
        step       = 1.0 / self._lipschitz(eff_fft, (H, W))
        K          = self.K
        dev        = self.device

        Z        = torch.zeros(K, H, W, dtype=torch.float32, device=dev)
        Y        = Z.clone()
        t        = 1.0
        prev_obj = float("inf")

        for _ in range(self.n_iter):
            Y_fft        = torch.fft.rfft2(Y)
            recon_fft    = (eff_fft * Y_fft).sum(dim=0)
            residual_fft = recon_fft - island_fft
            grad_fft     = eff_fft.conj() * residual_fft
            grad         = torch.fft.irfft2(grad_fft, s=(H, W))

            Z_new = self._soft_threshold(Y - step * grad, self.lmbda * step)
            t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            Y     = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
            Z     = Z_new
            t     = t_new

            with torch.no_grad():
                Z_fft_c   = torch.fft.rfft2(Z)
                r_fft_c   = (eff_fft * Z_fft_c).sum(dim=0) - island_fft
                recon_c   = torch.fft.irfft2(r_fft_c + island_fft, s=(H, W))
                data_fid  = 0.5 * ((torch.fft.irfft2(r_fft_c, s=(H, W))) ** 2).sum()
                obj       = float(data_fid + self.lmbda * Z.abs().sum())
                if abs(prev_obj - obj) / (abs(prev_obj) + 1e-8) < self.tol:
                    break
                prev_obj = obj

        return Z

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """
        Decode a single source island via FISTA; return clean-sky reconstruction.

        Without PSF: minimises  ||D*Z - island||²  → returns D*Z
        With PSF:    minimises  ||PSF*(D*Z) - island||² → returns D*Z (clean sky)

        Parameters
        ----------
        island : Tensor (H_i, W_i) float32, on self.device

        Returns
        -------
        recon : Tensor (H_i, W_i) float32 — Σ_k d_k ⊛ z_k  (clean-domain)
        """
        atoms_fft, island_w, H, W, orig_H, orig_W = self._prepare_atoms_fft(island)
        psf_fft = self._psf_fft(H, W)   # None if no PSF stored
        Z       = self._run_fista(atoms_fft, island_w, H, W, psf_fft=psf_fft)
        # Reconstruct in the clean domain (D*Z, without PSF)
        Z_fft   = torch.fft.rfft2(Z)
        recon   = torch.fft.irfft2((atoms_fft * Z_fft).sum(dim=0), s=(H, W))
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


# ── Variant C — Flow Matching Solver ─────────────────────────────────────────

class FlowSolver:
    """
    Island deconvolution via conditional flow matching (Variant C).

    The FlowModel was trained dirty→clean: the Euler ODE starts from the
    dirty island and integrates to t=1 to produce the clean sky estimate.
    No PSF is required — the PSF is implicit in the training data.

    Uncertainty is estimated by running n_samples trajectories from slightly
    perturbed starting points (x0 = dirty + ε·N(0,I)) and computing std
    across the resulting clean estimates.

    Parameters
    ----------
    flow_model      : FlowModel — trained dirty→clean velocity field
    device          : str       — torch device string
    n_samples       : int       — trajectories for uncertainty (default 8)
    n_steps         : int       — Euler ODE steps (default 16)
    perturb_std     : float     — std of starting perturbation (default 0.05)
    """

    _CANVAS = 150
    _variant_label = "C/Flow"

    def __init__(
        self,
        flow_model  : "FlowModel",  # noqa: F821
        device      : str   = "cpu",
        n_samples   : int   = 8,
        n_steps     : int   = 16,
        perturb_std : float = 0.05,
    ):
        self.flow_model  = flow_model
        self.device      = torch.device(device)
        self.n_samples   = n_samples
        self.n_steps     = n_steps
        self.perturb_std = perturb_std

    # ------------------------------------------------------------------
    # CR-3: Euler ODE from dirty island

    def decode_island_with_uncertainty(
        self,
        island: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run n_samples Euler trajectories from perturbed dirty starting points.
        Returns (mean, std), both (H_i, W_i).

        Parameters
        ----------
        island : Tensor (H_i, W_i) float32 — dirty sub-image on self.device

        Returns
        -------
        mean : Tensor (H_i, W_i)
        std  : Tensor (H_i, W_i)
        """
        C       = self._CANVAS
        Hi, Wi  = island.shape
        dev     = self.device
        S       = self.n_samples

        # Zero-pad island to C×C; replicate for all samples: (S, 1, C, C)
        pad_h      = C - Hi
        pad_w      = C - Wi
        island_pad = F_.pad(island, (0, pad_w, 0, pad_h))
        x_t = island_pad.unsqueeze(0).unsqueeze(0).expand(S, 1, C, C).clone()

        # Perturb starting points slightly — spread gives uncertainty estimate
        if self.perturb_std > 0.0:
            x_t = x_t + self.perturb_std * torch.randn_like(x_t)

        # Euler ODE: t from 0 (dirty) → 1 (clean)
        dt  = 1.0 / self.n_steps
        net = self.flow_model._net.eval()

        with torch.no_grad():
            for step_i in range(self.n_steps):
                t_batch = torch.full((S,), step_i * dt, device=dev)
                v       = net(x_t, t_batch)
                x_t     = x_t + dt * v

        x_clean = x_t[:, 0, :Hi, :Wi]         # (S, Hi, Wi)
        mean    = x_clean.mean(dim=0)          # (Hi, Wi)
        std     = x_clean.std(dim=0)           # (Hi, Wi)
        return mean, std

    # ------------------------------------------------------------------

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """
        Decode dirty island → clean sky estimate. Returns (H_i, W_i).
        Backward-compatible with PatchSolver and ConvSolver interface.
        """
        mean, _ = self.decode_island_with_uncertainty(island)
        return mean

    def __repr__(self) -> str:
        return (f"FlowSolver(n_samples={self.n_samples}, "
                f"n_steps={self.n_steps}, "
                f"perturb_std={self.perturb_std}, "
                f"device={self.device})")


# ── module-level alias to avoid shadowing torch.nn.functional ─────────────────
F_ = F
