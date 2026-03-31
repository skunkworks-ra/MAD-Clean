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

from mad_clean.io import load_image_data, save_fits
from mad_clean.filters import FilterBank
from mad_clean.detection import IslandDetector
from mad_clean.solvers import PatchSolver, ConvSolver


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

        fb       = FilterBank.load("models/cdl_filters_patch.npy", device="cuda")
        solver   = PatchSolver(fb, n_nonzero=5, stride=8)
        detector = IslandDetector(sigma_thresh=3.0, device="cuda")
        mc       = MADClean(fb, solver, detector, gamma=0.1, device="cuda")

        result = mc.deconvolve(dirty_array, psf_array)
        model    = result["model"]     # np.ndarray (H, W) float32
        residual = result["residual"]  # np.ndarray (H, W) float32

    From CLI:
        See __main__.py or run  python -m mad_clean --help
    """

    def __init__(
        self,
        filter_bank  : FilterBank,
        solver       : Union[PatchSolver, ConvSolver],
        detector     : IslandDetector,
        gamma        : float = 0.1,
        epsilon_frac : float = 0.01,
        n_max        : int   = 500,
        device       : Union[str, torch.device] = "cpu",
        verbose      : bool  = True,
    ):
        self.fb           = filter_bank
        self.solver       = solver
        self.detector     = detector
        self.gamma        = gamma
        self.epsilon_frac = epsilon_frac
        self.n_max        = n_max
        self.device       = torch.device(device)
        self.verbose      = verbose

        self._variant_label = getattr(solver, "_variant_label",
                                       solver.__class__.__name__)
        if self.verbose:
            print(f"MADClean ready  variant={self._variant_label}  "
                  f"γ={gamma}  ε_frac={epsilon_frac}  "
                  f"N_max={n_max}  device={self.device}")

    # ── PSF convolution ───────────────────────────────────────────────────────

    def _convolve_psf(
        self,
        image    : torch.Tensor,   # (H, W)
        psf_fft  : torch.Tensor,   # precomputed rfft2 of ifftshifted PSF
    ) -> torch.Tensor:
        """
        Convolve image with PSF via Fourier-plane multiplication.

        psf_fft is precomputed once per deconvolve() call from the
        ifftshift of the input PSF (peak moved from centre to (0,0)).
        """
        image_fft = torch.fft.rfft2(image)
        result    = torch.fft.irfft2(image_fft * psf_fft, s=image.shape)
        return result

    def _prepare_psf(self, psf: torch.Tensor) -> torch.Tensor:
        """
        Shift PSF peak from image centre to (0,0) and precompute rfft2.
        This is computed once per deconvolve() call.
        """
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
                    Dirty image from CASA/LibRA major cycle.
        psf       : (H, W) numpy array or FITS path
                    Real dirty beam. Peak must be at image centre (H//2, W//2),
                    matching CASA PSF output convention.
        out_dir   : optional path — if given, writes model.fits and residual.fits
        psf_header: optional astropy Header for WCS in output FITS

        Returns
        -------
        dict with keys:
            model     : np.ndarray (H, W) float32
            residual  : np.ndarray (H, W) float32
            rms_curve : np.ndarray (n_iter+1,) float32  — RMS per iteration
            n_iter    : int
        """
        # ── load inputs → GPU tensors ─────────────────────────────────────
        dirty_np = load_image_data(dirty)
        psf_np   = load_image_data(psf)

        if dirty_np.shape != psf_np.shape:
            raise ValueError(
                f"dirty {dirty_np.shape} and psf {psf_np.shape} must have "
                f"the same shape. Crop or pad the PSF to match."
            )

        dirty_t  = torch.from_numpy(dirty_np).float().to(self.device)
        psf_t    = torch.from_numpy(psf_np  ).float().to(self.device)
        psf_fft  = self._prepare_psf(psf_t)   # precomputed once

        residual    = dirty_t.clone()
        model       = torch.zeros_like(dirty_t)
        _has_uncert = hasattr(self.solver, "decode_island_with_uncertainty")
        uncertainty = torch.zeros_like(model) if _has_uncert else None

        rms_init  = float(residual.std())
        epsilon   = self.epsilon_frac * rms_init
        rms_curve = [rms_init]

        if self.verbose:
            print(f"  dirty peak={float(dirty_t.max()):.4e}  "
                  f"initial RMS={rms_init:.4e}  ε={epsilon:.4e}")

        # ── major cycle ───────────────────────────────────────────────────
        for it in range(self.n_max):

            bboxes, rms = self.detector.detect(residual)

            if not bboxes:
                if self.verbose:
                    print(f"  iter {it:3d}  no islands — stopping")
                break

            delta_m   = torch.zeros_like(model)
            delta_unc = torch.zeros_like(model) if _has_uncert else None
            for (r0, r1, c0, c1) in bboxes:
                island = residual[r0:r1, c0:c1]
                if _has_uncert:
                    recon, std = self.solver.decode_island_with_uncertainty(island)
                    delta_unc[r0:r1, c0:c1] += std
                else:
                    recon = self.solver.decode_island(island)
                delta_m[r0:r1, c0:c1] += recon

            if _has_uncert:
                uncertainty += self.gamma * delta_unc

            model    += self.gamma * delta_m
            residual  = dirty_t - self._convolve_psf(model, psf_fft)
            rms_new   = float(residual.std())
            rms_curve.append(rms_new)

            if self.verbose and (it + 1) % 50 == 0:
                print(f"  iter {it+1:3d}  RMS={rms_new:.4e}  "
                      f"islands={len(bboxes)}")

            if rms_new < epsilon:
                if self.verbose:
                    print(f"  Converged iter {it+1}  "
                          f"RMS={rms_new:.4e} < ε={epsilon:.4e}")
                break
        else:
            if self.verbose:
                print(f"  N_max={self.n_max} reached  "
                      f"final RMS={rms_curve[-1]:.4e}")

        # ── move outputs to CPU numpy ──────────────────────────────────────
        model_np    = model.cpu().numpy().astype(np.float32)
        residual_np = residual.cpu().numpy().astype(np.float32)
        rms_arr     = np.array(rms_curve, dtype=np.float32)
        uncert_np   = (uncertainty.cpu().numpy().astype(np.float32)
                       if uncertainty is not None else None)

        # ── optional FITS output ───────────────────────────────────────────
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
            "uncertainty": uncert_np,
        }

    def __repr__(self) -> str:
        return (f"MADClean(solver={self.solver.__class__.__name__}, "
                f"γ={self.gamma}, ε_frac={self.epsilon_frac}, "
                f"N_max={self.n_max}, device={self.device})")
