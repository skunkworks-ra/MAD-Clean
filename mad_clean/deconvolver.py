"""
mad_clean.deconvolver
=====================
MADClean — the outer CLEAN-like major cycle.

Owns the PSF convolution (Fourier-plane, GPU), the residual update, the model
accumulation, and the convergence check. Delegates island detection to
IslandDetector and sparse decoding to PatchSolver or ConvSolver.

Classes
-------
MADClean
    .deconvolve(dirty, psf) -> dict
        Runs the full iterative loop. Returns model, residual, rms_curve,
        n_iter as numpy arrays — safe to hand back to CASA/LibRA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .io import load_image_data, save_fits
from .filters import FilterBank
from .detection import IslandDetector
from .solvers import PatchSolver, ConvSolver
from .psf_utils import compute_psf_patch
from ._utils import clip_box, guide


__all__ = ["MADClean"]

ArrayLike = Union[np.ndarray, str, Path]


class MADClean:
    """
    Morphological Atom Decomposition CLEAN.

    Implements the outer CLEAN-like major cycle with a learned sparse coding
    minor cycle. The PSF convolution is always explicit and outside the
    learned component. Operates entirely on GPU if device is "cuda".

    Parameters
    ----------
    filter_bank  : FilterBank
    solver       : PatchSolver | ConvSolver
    detector     : IslandDetector
    gamma        : float  loop gain (default 0.1)
    epsilon_frac : float  convergence as fraction of initial residual RMS
                          (default 0.01 = 1%)
    n_max        : int    maximum major cycle iterations (default 500)
    device       : str | torch.device

    Usage
    -----
    From Python (e.g. CASA/LibRA minor cycle hook):

        fb       = FilterBank.load("models/cdl_filters_patch.npz", device="cuda")
        solver   = PatchSolver(fb, n_nonzero=5, stride=8)
        detector = IslandDetector(sigma_thresh=3.0, device="cuda")
        mc       = MADClean(fb, solver, detector, gamma=0.1, device="cuda")

        result = mc.deconvolve(dirty_array, psf_array)
        model    = result["model"]     # np.ndarray (H, W) float32
        residual = result["residual"]  # np.ndarray (H, W) float32
    """

    def __init__(
        self,
        filter_bank   : FilterBank,
        solver        : Union[PatchSolver, ConvSolver],
        detector      : Union[IslandDetector, None] = None,
        gamma         : float = 0.1,
        epsilon_frac  : float = 0.01,
        n_max         : int   = 500,
        device        : Union[str, torch.device] = "cpu",
        verbose       : bool  = True,
        refresh_every : int   = 100,
        energy_frac   : float = 0.90,
    ):
        self.fb            = filter_bank
        self.solver        = solver
        self.detector      = detector
        self.gamma         = gamma
        self.epsilon_frac  = epsilon_frac
        self.n_max         = n_max
        self.device        = torch.device(device)
        self.verbose       = verbose
        self.refresh_every = refresh_every
        self.energy_frac   = energy_frac

        self._variant_label = getattr(solver, "_variant_label",
                                       solver.__class__.__name__)
        if self.verbose:
            print(f"MADClean ready  variant={self._variant_label}  "
                  f"γ={gamma}  ε_frac={epsilon_frac}  "
                  f"N_max={n_max}  refresh_every={refresh_every}  "
                  f"device={self.device}")

    # ── PSF convolution ───────────────────────────────────────────────────────

    def _convolve_psf(
        self,
        image    : torch.Tensor,
        psf_fft  : torch.Tensor,
    ) -> torch.Tensor:
        image_fft = torch.fft.rfft2(image)
        result    = torch.fft.irfft2(image_fft * psf_fft, s=image.shape)
        return result

    def _prepare_psf(self, psf: torch.Tensor) -> torch.Tensor:
        psf_shifted = torch.fft.ifftshift(psf)
        return torch.fft.rfft2(psf_shifted)

    # ── main loop ─────────────────────────────────────────────────────────────

    def deconvolve(
        self,
        dirty    : ArrayLike,
        psf      : ArrayLike,
        out_dir  : Union[str, Path, None] = None,
        psf_header = None,
    ) -> dict:
        """
        Run MAD-CLEAN deconvolution.

        Parameters
        ----------
        dirty     : (H, W) numpy array or FITS path
        psf       : (H, W) numpy array or FITS path. Peak at image centre.
        out_dir   : optional path — writes model.fits and residual.fits
        psf_header: optional astropy Header for WCS in output FITS

        Returns
        -------
        dict with keys: model, residual, rms_curve, n_iter, converged,
                        peak_flux, uncertainty
        """
        dirty_np = load_image_data(dirty)
        psf_np   = load_image_data(psf)

        if dirty_np.shape != psf_np.shape:
            raise ValueError(
                f"dirty {dirty_np.shape} and psf {psf_np.shape} must have "
                f"the same shape. Crop or pad the PSF to match."
            )

        H, W = dirty_np.shape[-2], dirty_np.shape[-1]

        dirty_t  = torch.from_numpy(dirty_np).float().to(self.device)
        psf_t    = torch.from_numpy(psf_np  ).float().to(self.device)

        psf_ref = psf_t if psf_t.ndim == 2 else psf_t.reshape(-1, H, W)[0]
        psf_fft = self._prepare_psf(psf_ref)

        psf_patch, (half_h, half_w) = compute_psf_patch(
            psf_ref, energy_frac=self.energy_frac
        )
        cy_p, cx_p = psf_patch.shape[0] // 2, psf_patch.shape[1] // 2

        residual    = dirty_t.clone()
        model       = torch.zeros_like(dirty_t)
        _has_uncert = hasattr(self.solver, "decode_island_with_uncertainty")
        uncertainty = torch.zeros_like(model) if _has_uncert else None

        dirty_peak = float(dirty_t.abs().max())
        rms_init   = float(guide(residual).std())

        epsilon = max(0.1 * dirty_peak, self.epsilon_frac * rms_init)

        rms_curve = [rms_init]

        if self.verbose:
            print(f"  dirty peak={dirty_peak:.4e}  "
                  f"initial RMS={rms_init:.4e}  "
                  f"ε={epsilon:.4e}  "
                  f"PSF patch={psf_patch.shape}")

        def _fft_refresh(m: torch.Tensor) -> torch.Tensor:
            if m.ndim == 2:
                return dirty_t - self._convolve_psf(m, psf_fft)
            n_slices = m.reshape(-1, H, W).shape[0]
            slices = [
                dirty_t.reshape(-1, H, W)[i] - self._convolve_psf(
                    m.reshape(-1, H, W)[i], psf_fft
                )
                for i in range(n_slices)
            ]
            return torch.stack(slices).reshape(m.shape)

        converged = False

        for it in range(self.n_max):
            guide_img = guide(residual)

            flat = int(guide_img.abs().argmax())
            py   = flat // W
            px   = flat % W
            peak_v_guide = float(guide_img[py, px])

            if abs(peak_v_guide) < epsilon:
                converged = True
                if self.verbose:
                    print(f"  Converged iter {it}  "
                          f"peak={peak_v_guide:.4e} < ε={epsilon:.4e}")
                break

            r0 = max(0, py - half_h)
            r1 = min(H, py + half_h + 1)
            c0 = max(0, px - half_w)
            c1 = min(W, px + half_w + 1)

            island_2d = residual[r0:r1, c0:c1] if residual.ndim == 2 \
                        else residual.reshape(-1, H, W)[0, r0:r1, c0:c1]

            if _has_uncert:
                model_patch, std = self.solver.decode_island_with_uncertainty(island_2d)
                uncertainty[..., r0:r1, c0:c1] += self.gamma * std
            else:
                model_patch = self.solver.decode_island(island_2d)

            model[..., r0:r1, c0:c1] += self.gamma * model_patch

            peak_m = float(model_patch[py - r0, px - c0])
            r0c, r1c, c0c, c1c, pr0, pr1, pc0, pc1 = clip_box(
                py, px, cy_p, cx_p, psf_patch.shape[0], psf_patch.shape[1], H, W
            )
            residual[..., r0c:r1c, c0c:c1c] -= (
                self.gamma * peak_m * psf_patch[pr0:pr1, pc0:pc1]
            )

            if (it + 1) % self.refresh_every == 0:
                residual = _fft_refresh(model)
                rms_new  = float(guide(residual).std())
                rms_curve.append(rms_new)
                if self.verbose:
                    print(f"  iter {it+1:4d}  peak={peak_v_guide:.4e}  "
                          f"RMS={rms_new:.4e}")

        else:
            if self.verbose:
                print(f"  N_max={self.n_max} reached  "
                      f"final peak={peak_v_guide:.4e}")

        residual = _fft_refresh(model)

        model_np    = model.cpu().numpy().astype(np.float32)
        residual_np = residual.cpu().numpy().astype(np.float32)
        rms_arr     = np.array(rms_curve, dtype=np.float32)
        uncert_np   = (uncertainty.cpu().numpy().astype(np.float32)
                       if uncertainty is not None else None)

        if out_dir is not None:
            out_dir = Path(out_dir)
            lbl = self._variant_label.replace("/", "_")
            save_fits(model_np,    out_dir / f"mad_clean_{lbl}_model.fits",
                      header=psf_header)
            save_fits(residual_np, out_dir / f"mad_clean_{lbl}_residual.fits",
                      header=psf_header)
            np.save(out_dir / f"mad_clean_{lbl}_rms_curve.npy", rms_arr)
            if uncert_np is not None:
                save_fits(uncert_np, out_dir / f"mad_clean_{lbl}_uncertainty.fits",
                          header=psf_header)
            if self.verbose:
                print(f"  Outputs written → {out_dir}/")

        return {
            "model"      : model_np,
            "residual"   : residual_np,
            "rms_curve"  : rms_arr,
            "n_iter"     : len(rms_curve) - 1,
            "converged"  : converged,
            "peak_flux"  : float(model.abs().max()),
            "uncertainty": uncert_np,
        }

    def __repr__(self) -> str:
        return (f"MADClean(solver={self.solver.__class__.__name__}, "
                f"γ={self.gamma}, ε_frac={self.epsilon_frac}, "
                f"N_max={self.n_max}, refresh_every={self.refresh_every}, "
                f"device={self.device})")
