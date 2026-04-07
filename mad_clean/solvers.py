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
import torch.nn.functional as F_

from .filters import FilterBank
from ._utils import soft_threshold


__all__ = ["PatchSolver", "ConvSolver", "FlowSolver", "DPSSolver", "AmortisedSolver"]


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

    _variant_label = "A"

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
        orig_H, orig_W = island.shape
        F = self.F
        s = self.stride

        # Pad island to at least (F, F) so unfold never sees a spatial dim < kernel
        island_w = island
        if orig_H < F or orig_W < F:
            ph = max(0, F - orig_H)
            pw = max(0, F - orig_W)
            island_w = F_.pad(island, (0, pw, 0, ph))
        H, W = island_w.shape

        island_4d = island_w.unsqueeze(0).unsqueeze(0)
        patches   = F_.unfold(island_4d, kernel_size=F, stride=s)
        patches   = patches.squeeze(0).T.contiguous()  # (n_patches, F²)
        n_patches = patches.shape[0]

        from sklearn.linear_model import orthogonal_mp

        patches_np = patches.cpu().numpy()
        p_mean = patches_np.mean(axis=1, keepdims=True)
        p_std  = patches_np.std (axis=1, keepdims=True) + 1e-8
        patches_n = (patches_np - p_mean) / p_std

        Z = orthogonal_mp(
            self._D_cpu,
            patches_n.T,
            n_nonzero_coefs=self.n_nonzero,
        )   # (K, n_patches)

        Z_t     = torch.from_numpy(Z.T).float().to(self.device)  # (n_patches, K)
        D_t     = self.fb.D                                       # (F², K)
        recon_n = Z_t @ D_t.T
        p_mean_t = torch.from_numpy(p_mean).float().to(self.device)
        p_std_t  = torch.from_numpy(p_std ).float().to(self.device)
        recon_patches = recon_n * p_std_t + p_mean_t   # (n_patches, F²)

        recon_4d = recon_patches.T.unsqueeze(0)
        output_size = (H, W)

        folded = F_.fold(recon_4d, output_size=output_size,
                         kernel_size=F, stride=s)
        ones    = torch.ones_like(recon_4d)
        divisor = F_.fold(ones, output_size=output_size,
                          kernel_size=F, stride=s)
        divisor = divisor.clamp(min=1.0)

        result = (folded / divisor).squeeze()
        return result[:orig_H, :orig_W]

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

    _variant_label = "B"

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
        self._psf   = psf

    def _psf_fft(self, H: int, W: int) -> "torch.Tensor | None":
        """
        Centre-crop or zero-pad the stored PSF to (H, W) and return its rfft2.
        Returns None if no PSF was provided at construction.
        """
        if self._psf is None:
            return None
        psf = torch.from_numpy(self._psf).float().to(self.device)
        pH, pW = psf.shape
        if pH > H:
            ch = (pH - H) // 2
            psf = psf[ch : ch + H, :]
        if pW > W:
            cw = (pW - W) // 2
            psf = psf[:, cw : cw + W]
        h, w = psf.shape
        if h < H or w < W:
            out = torch.zeros(H, W, device=self.device)
            ph = (H - h) // 2
            pw = (W - w) // 2
            out[ph : ph + h, pw : pw + w] = psf
            psf = out
        return torch.fft.rfft2(torch.fft.ifftshift(psf))

    def _lipschitz(self, atoms_fft: torch.Tensor, sig_shape: tuple) -> float:
        power = (atoms_fft.abs() ** 2).sum(dim=0)
        return float(power.max().real) + 1e-8

    def _prepare_atoms_fft(self, island: torch.Tensor) -> tuple:
        F          = self.F
        orig_H, orig_W = island.shape
        island_w   = island

        if orig_H < F or orig_W < F:
            ph       = max(0, F - orig_H)
            pw       = max(0, F - orig_W)
            island_w = F_.pad(island, (0, pw, 0, ph))

        H, W   = island_w.shape
        pad_h  = H - F
        pad_w  = W - F

        atoms_padded = F_.pad(self.fb.atoms, (0, pad_w, 0, pad_h))
        atoms_fft    = torch.fft.rfft2(atoms_padded)

        return atoms_fft, island_w, H, W, orig_H, orig_W

    def _run_fista(
        self,
        atoms_fft : torch.Tensor,
        island    : torch.Tensor,
        H         : int,
        W         : int,
        psf_fft   : "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Run FISTA and return activation maps Z (K, H, W).

        Without psf_fft: minimises  0.5||D*Z - island||² + λ||Z||₁
        With    psf_fft: minimises  0.5||PSF*(D*Z) - island||² + λ||Z||₁
        """
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

            Z_new = soft_threshold(Y - step * grad, self.lmbda * step)
            t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            Y     = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
            Z     = Z_new
            t     = t_new

            with torch.no_grad():
                Z_fft_c   = torch.fft.rfft2(Z)
                r_fft_c   = (eff_fft * Z_fft_c).sum(dim=0) - island_fft
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
        """
        atoms_fft, island_w, H, W, orig_H, orig_W = self._prepare_atoms_fft(island)
        psf_fft = self._psf_fft(H, W)
        Z       = self._run_fista(atoms_fft, island_w, H, W, psf_fft=psf_fft)
        Z_fft   = torch.fft.rfft2(Z)
        recon   = torch.fft.irfft2((atoms_fft * Z_fft).sum(dim=0), s=(H, W))
        return recon[:orig_H, :orig_W].float()

    def encode_island(self, island: torch.Tensor) -> torch.Tensor:
        """Encode a single island via FISTA; return activation maps Z (K, H, W)."""
        atoms_fft, island_w, H, W, _, _ = self._prepare_atoms_fft(island)
        return self._run_fista(atoms_fft, island_w, H, W)

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

    def decode_island_with_uncertainty(
        self,
        island: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run n_samples Euler trajectories from perturbed dirty starting points.
        Returns (mean, std), both (H_i, W_i).
        """
        C       = self._CANVAS
        Hi, Wi  = island.shape
        dev     = self.device
        S       = self.n_samples

        pad_h      = C - Hi
        pad_w      = C - Wi
        island_pad = F_.pad(island, (0, pad_w, 0, pad_h))
        x_t = island_pad.unsqueeze(0).unsqueeze(0).expand(S, 1, C, C).clone()

        if self.perturb_std > 0.0:
            x_t = x_t + self.perturb_std * torch.randn_like(x_t)

        dt  = 1.0 / self.n_steps
        net = self.flow_model._net.eval()

        with torch.no_grad():
            for step_i in range(self.n_steps):
                t_batch = torch.full((S,), step_i * dt, device=dev)
                v, _    = net(x_t, t_batch)   # unpack (velocity, log_var)
                x_t     = x_t + dt * v

        x_clean = x_t[:, 0, :Hi, :Wi]
        mean    = x_clean.mean(dim=0)
        std     = x_clean.std(dim=0) if S > 1 else torch.zeros_like(mean)
        return mean, std

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        mean, _ = self.decode_island_with_uncertainty(island)
        return mean

    def __repr__(self) -> str:
        return (f"FlowSolver(n_samples={self.n_samples}, "
                f"n_steps={self.n_steps}, "
                f"perturb_std={self.perturb_std}, "
                f"device={self.device})")


# ── Variant C/DPS — Diffusion Posterior Sampling ──────────────────────────────

class DPSSolver:
    """
    Island deconvolution via Diffusion Posterior Sampling (FlowDPS).

    Samples from p(clean | dirty, PSF) by combining an unconditional flow prior
    p(clean) (from PriorTrainer) with an explicit Gaussian likelihood gradient
    injected at each Euler step (Kim et al., ICCV 2025 / Chung et al. 2022).

    Algorithm (per Euler step, t: 0 → 1)
    -------------------------------------
    1. Prior velocity:  v = prior_model.velocity(x_t, t)
    2. Tweedie estimate: x̂₁ = x_t + (1-t) · v
    3. Likelihood grad:  ∇ = PSF^T * (dirty - PSF * x̂₁) / σ²
                         [PSF^T = correlation = conjugate in FFT space]
    4. Corrected step:   x_{t+dt} = x_t + dt · (v + ζ · (1-t) · ∇)
                         [scale by (1-t): correction matters more near t=1]

    x_t starts from pure Gaussian noise (matching PriorTrainer's starting distribution).
    n_samples independent trajectories give a posterior mean and std.

    Key hyperparameter: dps_weight ζ (default 1.0).
    - Too high → reconstruction locked tightly to dirty (overfits PSF sidelobes)
    - Too low  → prior dominates, flux_rec drifts away from 1.0
    - Tune on validation: target flux_rec ≈ 1.0 ± 0.1

    Parameters
    ----------
    prior_model : FlowModel — trained on clean images via PriorTrainer
    psf_norm    : np.ndarray (H_p, W_p) — PSF with sum=1 (use data["psf_norm"])
    noise_std   : float — estimated noise level in dirty image (default 0.05)
    n_steps     : int   — Euler ODE steps (default 50)
    n_samples   : int   — independent posterior draws (default 8)
    dps_weight  : float — likelihood gradient scale ζ (default 1.0)
    device      : str   — torch device string
    """

    _CANVAS = 150
    _variant_label = "C/DPS"

    def __init__(
        self,
        prior_model : "FlowModel",   # noqa: F821
        psf_norm    : np.ndarray,
        noise_std   : float = 0.05,
        n_steps     : int   = 50,
        n_samples   : int   = 8,
        dps_weight  : float = 1.0,
        device      : str   = "cpu",
    ):
        self.prior_model = prior_model
        self.noise_std   = noise_std
        self.n_steps     = n_steps
        self.n_samples   = n_samples
        self.dps_weight  = dps_weight
        self.device      = torch.device(device)

        # Pre-compute PSF FFT at canvas size — done once at init, reused per call.
        C   = self._CANVAS
        psf = torch.from_numpy(psf_norm).float()
        pH, pW = psf.shape

        # Crop PSF if larger than canvas
        if pH > C:
            ch  = (pH - C) // 2
            psf = psf[ch : ch + C, :]
            pH  = psf.shape[0]
        if pW > C:
            cw  = (pW - C) // 2
            psf = psf[:, cw : cw + C]
            pW  = psf.shape[1]

        # Zero-pad to canvas, PSF centred
        canvas = torch.zeros(C, C)
        ph = (C - pH) // 2
        pw = (C - pW) // 2
        canvas[ph : ph + pH, pw : pw + pW] = psf

        # ifftshift centres the PSF at the DC component before FFT
        psf_fft = torch.fft.rfft2(torch.fft.ifftshift(canvas)).to(self.device)
        self._psf_fft      = psf_fft          # forward: PSF * x̂₁
        self._psf_fft_conj = psf_fft.conj()   # backward: PSF^T * r (correlation)

    def _run_euler(
        self,
        dirty    : torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Core Euler integration loop. Returns all posterior draws.

        Parameters
        ----------
        dirty     : (H_i, W_i) float32 Tensor — already on self.device
        n_samples : number of independent trajectories

        Returns
        -------
        x_clean : (S, H_i, W_i) — raw posterior samples (no averaging)
        """
        S   = n_samples
        C   = self._CANVAS
        dev = self.device
        Hi, Wi = dirty.shape

        dirty_pad = F_.pad(dirty, (0, C - Wi, 0, C - Hi))            # (C, C)
        dirty_fft = torch.fft.rfft2(dirty_pad).unsqueeze(0)           # (1, C, C//2+1)

        x_t    = torch.randn(S, 1, C, C, device=dev)
        dt     = 1.0 / self.n_steps
        sigma2 = self.noise_std ** 2
        net    = self.prior_model._net.eval().to(dev)

        with torch.no_grad():
            for step_i in range(self.n_steps):
                t_val   = step_i * dt
                t_batch = torch.full((S,), t_val, device=dev)

                # 1. Prior velocity
                v, _ = net(x_t, t_batch)                              # (S, 1, C, C)

                # 2. Tweedie estimate of x₁
                x_hat_1 = x_t + (1.0 - t_val) * v                    # (S, 1, C, C)

                # 3. Likelihood gradient: (1/σ²) · PSF^T * (dirty - PSF * x̂₁)
                xhat_fft  = torch.fft.rfft2(x_hat_1[:, 0])           # (S, C, C//2+1)
                pred_fft  = self._psf_fft * xhat_fft                  # (S, C, C//2+1)
                resid_fft = dirty_fft - pred_fft                      # (S, C, C//2+1)
                grad      = torch.fft.irfft2(
                    self._psf_fft_conj * resid_fft, s=(C, C)
                ).unsqueeze(1) / sigma2                                # (S, 1, C, C)

                # 4. Corrected Euler step
                x_t = x_t + dt * (v + self.dps_weight * (1.0 - t_val) * grad)

        return x_t[:, 0, :Hi, :Wi]                                    # (S, Hi, Wi)

    def sample(
        self,
        dirty    : torch.Tensor,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw n_samples posterior estimates; return (mean, std).

        Parameters
        ----------
        dirty     : (H_i, W_i) float32 Tensor — observed dirty island
        n_samples : override constructor n_samples if provided

        Returns
        -------
        mean : (H_i, W_i) — posterior mean
        std  : (H_i, W_i) — posterior std (zeros if n_samples == 1)
        """
        S       = n_samples if n_samples is not None else self.n_samples
        x_clean = self._run_euler(dirty.to(self.device), S)
        mean    = x_clean.mean(dim=0)
        std     = x_clean.std(dim=0) if S > 1 else torch.zeros_like(mean)
        return mean, std

    def sample_all(
        self,
        dirty    : torch.Tensor,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        """
        Draw n_samples from the posterior and return all individual draws.

        Use this for TARP calibration and morphology confidence — both need
        the full sample distribution, not just mean/std.

        Returns
        -------
        samples : (S, H_i, W_i) — individual posterior draws
        """
        S = n_samples if n_samples is not None else self.n_samples
        return self._run_euler(dirty.to(self.device), S)

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """FlowSolver-compatible interface: returns posterior mean."""
        mean, _ = self.sample(island, n_samples=self.n_samples)
        return mean

    def __repr__(self) -> str:
        return (f"DPSSolver(n_steps={self.n_steps}, "
                f"n_samples={self.n_samples}, "
                f"dps_weight={self.dps_weight}, "
                f"noise_std={self.noise_std}, "
                f"device={self.device})")


# ── Phase 2 — Amortised Posterior Solver ─────────────────────────────────────

class AmortisedSolver:
    """
    Island deconvolution via amortised posterior (Phase 2).

    Replaces DPS's expensive N×50-step per-island sampling with a single
    conditional flow q_φ(x_clean | dirty) trained to reproduce DPS posteriors.
    At inference: N samples cost N × n_steps forward passes (default 20),
    vs DPS's N × 50 — and with fewer steps because the dirty image is available
    as conditioning, not just as a likelihood gradient correction.

    Requires: ConditionalFlowModel trained via AmortisedPosteriorTrainer
    (scripts/collect_dps_samples.py + scripts/train.py --variant Q).

    Parameters
    ----------
    cond_model  : ConditionalFlowModel — trained on DPS posterior samples
    device      : str
    n_samples   : int   — posterior draws per island (default 8)
    n_steps     : int   — Euler ODE steps (default 20; fewer than DPS's 50)
    """

    _CANVAS = 150
    _variant_label = "C/Amortised"

    def __init__(
        self,
        cond_model  : "ConditionalFlowModel",   # noqa: F821
        device      : str = "cpu",
        n_samples   : int = 8,
        n_steps     : int = 20,
    ):
        self.cond_model = cond_model
        self.device     = torch.device(device)
        self.n_samples  = n_samples
        self.n_steps    = n_steps

    def _run_euler(
        self,
        dirty    : torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Conditional Euler loop. Returns (S, H_i, W_i) posterior samples."""
        S   = n_samples
        C   = self._CANVAS
        dev = self.device
        Hi, Wi = dirty.shape

        dirty_pad  = F_.pad(dirty.to(dev), (0, C - Wi, 0, C - Hi))   # (C, C)
        dirty_cond = dirty_pad.unsqueeze(0).unsqueeze(0).expand(S, 1, C, C)  # (S,1,C,C)

        x_t = torch.randn(S, 1, C, C, device=dev)
        dt  = 1.0 / self.n_steps
        net = self.cond_model._net.eval().to(dev)

        with torch.no_grad():
            for step_i in range(self.n_steps):
                t_batch = torch.full((S,), step_i * dt, device=dev)
                inp     = torch.cat([x_t, dirty_cond], dim=1)         # (S, 2, C, C)
                v, _    = net(inp, t_batch)
                x_t     = x_t + dt * v

        return x_t[:, 0, :Hi, :Wi]                                    # (S, Hi, Wi)

    def sample_all(
        self,
        dirty    : torch.Tensor,
        n_samples: int | None = None,
    ) -> torch.Tensor:
        """Return all (S, H_i, W_i) posterior draws."""
        S = n_samples if n_samples is not None else self.n_samples
        return self._run_euler(dirty, S)

    def sample(
        self,
        dirty    : torch.Tensor,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std), both (H_i, W_i)."""
        x_clean = self.sample_all(dirty, n_samples)
        S = x_clean.shape[0]
        mean = x_clean.mean(dim=0)
        std  = x_clean.std(dim=0) if S > 1 else torch.zeros_like(mean)
        return mean, std

    def decode_island(self, island: torch.Tensor) -> torch.Tensor:
        """FlowSolver-compatible interface."""
        mean, _ = self.sample(island)
        return mean

    def __repr__(self) -> str:
        return (f"AmortisedSolver(n_steps={self.n_steps}, "
                f"n_samples={self.n_samples}, "
                f"device={self.device})")
