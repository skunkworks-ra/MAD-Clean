"""
mad_clean.deconvolver
=====================
Base class and derived deconvolvers for MAD-CLEAN.

Classes
-------
BaseDeconvolver
    Abstract base. Owns the iterative CLEAN loop: peak normalisation,
    minor/major cycle management, PSF subtraction, convergence.
    Subclasses implement _predict(island) → (component, uncertainty | None).

HogbomDeconvolver(BaseDeconvolver)
    Classical Hogbom CLEAN. _predict returns a delta at the peak pixel.

MADCleanDeconvolver(BaseDeconvolver)
    Learned minor cycle. Accepts any solver with a decode_island() method:
    PatchSolver, ConvSolver, FlowSolver, DPSSolver, LatentDPSSolver.
    If the solver also has decode_island_with_uncertainty(), per-pixel
    uncertainty is accumulated.

Loop design
-----------
Minor cycle (within one major cycle):
    1. Normalise residual by its peak → [0, 1]
    2. Extract island around peak (PSF-patch sized)
    3. _predict(island) → component
    4. Subtract gain × component × psf from residual (direct, not FFT)
    5. Accumulate in model (rescaled back to physical units)
    Trigger major cycle when: residual.max() < initial_peak × sidelobe_level

Major cycle:
    Recompute residual = dirty - psf ⊛ model  (exact, FFT)
    Repeat minor cycle from updated residual.

Stop when: residual.max() < noise_threshold
           OR n_major_cycles exhausted.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
import torch.nn.functional as F_

from .psf_utils import compute_psf_patch, psf_sidelobe_analysis
from ._utils import clip_box, guide

__all__ = ["BaseDeconvolver", "HogbomDeconvolver", "MADCleanDeconvolver"]


# ---------------------------------------------------------------------------
# Sidelobe level from PSF
# ---------------------------------------------------------------------------

def _compute_sidelobe_level(psf_norm: torch.Tensor) -> float:
    """
    Estimate the first sidelobe level from a sum-normalised PSF.

    Converts to peak-normalised, runs psf_sidelobe_analysis to find the
    first null radius, then returns max(|PSF|) between first and second nulls.
    If fewer than two nulls are found, falls back to 0.1.
    """
    psf_np   = psf_norm.cpu().numpy().astype(np.float64)
    peak     = psf_np.max()
    if peak < 1e-12:
        return 0.1
    psf_peak = psf_np / peak   # peak-normalise for sidelobe_analysis

    analysis = psf_sidelobe_analysis(psf_peak, n_sidelobes=3)
    nulls    = analysis["null_radii"]

    if len(nulls) < 2:
        return 0.1

    H, W  = psf_peak.shape
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist   = np.sqrt(yy ** 2 + xx ** 2)

    between = (dist > nulls[0]) & (dist <= nulls[1])
    if not between.any():
        return 0.1

    return float(np.abs(psf_peak[between]).max())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDeconvolver(abc.ABC):
    """
    Abstract base for iterative CLEAN deconvolvers.

    Parameters
    ----------
    psf             : (H, W) numpy array — PSF with peak=1 (CASA convention)
    gain            : float  loop gain (default 0.1)
    noise_threshold : float | None
                      Stop when residual.max() < noise_threshold.
                      If None: estimated as 3 × residual std after first major cycle.
    sidelobe_level  : float | None
                      Minor→major cycle trigger threshold (as fraction of current peak).
                      If None: computed from psf automatically.
    n_major_cycles  : int    maximum major cycles (default 10)
    max_minor_iter  : int    hard cap on minor cycle iterations per major cycle (default 100)
    device          : str | None  — 'cuda' if available, else 'cpu' with warning
    verbose         : bool
    """

    def __init__(
        self,
        psf             : np.ndarray,
        gain            : float        = 0.1,
        noise_threshold : Optional[float] = None,
        sidelobe_level  : Optional[float] = None,
        n_major_cycles  : int          = 10,
        max_minor_iter  : int          = 100,
        device          : Optional[str] = None,
        verbose         : bool         = True,
    ):
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                import warnings
                warnings.warn("CUDA not available — falling back to CPU.", stacklevel=2)
                device = "cpu"
        self.device    = torch.device(device)
        self.gain      = gain
        self.noise_threshold  = noise_threshold
        self.n_major_cycles   = n_major_cycles
        self.max_minor_iter   = max_minor_iter
        self.verbose          = verbose
        self._needs_peak_norm = True   # subclass may override

        # PSF setup — peak=1 (CASA convention: Jy/beam)
        psf_t = torch.from_numpy(psf.astype(np.float32)).to(self.device)
        self._psf = psf_t
        self._psf_fft  = torch.fft.rfft2(torch.fft.ifftshift(psf_t))

        psf_patch, (self._half_h, self._half_w) = compute_psf_patch(psf_t)
        self._psf_patch = psf_patch
        cy_p = psf_patch.shape[0] // 2
        cx_p = psf_patch.shape[1] // 2
        self._psf_cy = cy_p
        self._psf_cx = cx_p

        # Sidelobe level
        if sidelobe_level is not None:
            self._sidelobe_level = float(sidelobe_level)
        else:
            self._sidelobe_level = _compute_sidelobe_level(psf_t)

        if self.verbose:
            print(f"{self.__class__.__name__}  gain={gain}  "
                  f"sidelobe_level={self._sidelobe_level:.3f}  "
                  f"n_major_cycles={n_major_cycles}  device={self.device}")

    # ── abstract ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def _predict(
        self, island: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict the clean component for an island patch.

        Parameters
        ----------
        island : (H_i, W_i) float32 Tensor — peak-normalised to [0, 1]

        Returns
        -------
        component   : (H_i, W_i) float32 — clean component in [0, 1] space
        uncertainty : (H_i, W_i) float32 | None — per-pixel std, or None
        """

    # ── PSF convolution (FFT, full image) ─────────────────────────────────────

    def _major_cycle_residual(
        self, dirty: torch.Tensor, model: torch.Tensor
    ) -> torch.Tensor:
        """Exact residual: dirty - psf ⊛ model  (FFT convolution)."""
        H, W      = dirty.shape
        model_fft = torch.fft.rfft2(model)
        pred      = torch.fft.irfft2(self._psf_fft * model_fft, s=(H, W))
        return dirty - pred

    # ── main loop ─────────────────────────────────────────────────────────────

    def deconvolve(
        self,
        dirty   : np.ndarray,
        psf_raw : Optional[np.ndarray] = None,   # ignored — kept for API compat
    ) -> dict:
        """
        Run iterative deconvolution.

        Parameters
        ----------
        dirty   : (H, W) float32 numpy array — dirty image in physical units
        psf_raw : ignored (psf_norm provided at construction); kept for compat

        Returns
        -------
        dict with keys:
            model       : (H, W) float32 — accumulated clean model (physical units)
            residual    : (H, W) float32 — final residual
            uncertainty : (H, W) float32 | None
            n_major     : int
            converged   : bool
        """
        dev  = self.device
        H, W = dirty.shape[-2], dirty.shape[-1]

        dirty_t   = torch.from_numpy(dirty.astype(np.float32)).to(dev)
        residual  = dirty_t.clone()
        model     = torch.zeros_like(dirty_t)
        uncert    = torch.zeros_like(dirty_t)
        has_uncert = False

        initial_peak = float(guide(residual).abs().max())

        # Auto noise threshold: estimated after first major cycle
        noise_threshold = self.noise_threshold

        converged = False

        for major in range(self.n_major_cycles):
            major_peak = float(guide(residual).abs().max())
            minor_threshold = major_peak * self._sidelobe_level

            if noise_threshold is not None and major_peak < noise_threshold:
                converged = True
                break

            if self.verbose:
                print(f"  Major cycle {major + 1}/{self.n_major_cycles}  "
                      f"peak={major_peak:.4e}  "
                      f"minor_stop={minor_threshold:.4e}", flush=True)

            # ── minor cycle ────────────────────────────────────────────────
            n_minor = 0
            while True:
                guide_img  = guide(residual)
                peak_val   = float(guide_img.abs().max())

                if peak_val < minor_threshold:
                    break
                if noise_threshold is not None and peak_val < noise_threshold:
                    converged = True
                    break

                flat  = int(guide_img.abs().argmax())
                py, px = flat // W, flat % W

                # Extract island
                r0 = max(0, py - self._half_h)
                r1 = min(H, py + self._half_h + 1)
                c0 = max(0, px - self._half_w)
                c1 = min(W, px + self._half_w + 1)
                island = guide_img[r0:r1, c0:c1]

                if self._needs_peak_norm:
                    # Standard path: normalise to [0,1], predict, rescale back
                    island_peak = float(island.abs().max()) + 1e-12
                    island_norm = island / island_peak
                    component, unc = self._predict(island_norm)
                    contrib = self.gain * component * island_peak
                    if unc is not None:
                        unc_contrib = self.gain * unc * island_peak
                else:
                    # DPS path: solver has its own forward model, needs
                    # raw Jy/beam units — no peak normalisation
                    component, unc = self._predict(island)
                    contrib = self.gain * component
                    if unc is not None:
                        unc_contrib = self.gain * unc

                model[r0:r1, c0:c1] += contrib
                if unc is not None:
                    uncert[r0:r1, c0:c1] += unc_contrib
                    has_uncert = True

                # Subtract PSF ⊛ component from residual (FFT, exact).
                # Must match the major cycle: residual = dirty - PSF ⊛ model.
                contrib_img = torch.zeros_like(residual)
                contrib_img[r0:r1, c0:c1] = contrib
                contrib_fft = torch.fft.rfft2(contrib_img)
                residual -= torch.fft.irfft2(
                    self._psf_fft * contrib_fft, s=(H, W)
                )
                n_minor += 1

                if n_minor % 10 == 0:
                    print(f"    minor iter {n_minor}  peak={peak_val:.4e}", flush=True)

                if n_minor >= self.max_minor_iter:
                    print(f"    max_minor_iter={self.max_minor_iter} reached — stopping minor cycle.", flush=True)
                    break

            if converged:
                break

            # ── major cycle: exact FFT residual ───────────────────────────
            residual = self._major_cycle_residual(dirty_t, model)

            # Set noise threshold after first major cycle if not provided
            if noise_threshold is None and major == 0:
                noise_threshold = 3.0 * float(guide(residual).std())
                if self.verbose:
                    print(f"    noise_threshold set to {noise_threshold:.4e} "
                          f"(3σ of residual)", flush=True)

            if self.verbose:
                print(f"    minor steps={n_minor}  "
                      f"post-major peak={float(guide(residual).abs().max()):.4e}",
                      flush=True)

        return {
            "model"      : model.cpu().numpy().astype(np.float32),
            "residual"   : residual.cpu().numpy().astype(np.float32),
            "uncertainty": uncert.cpu().numpy().astype(np.float32) if has_uncert else None,
            "n_major"    : major + 1,
            "converged"  : converged,
        }


# ---------------------------------------------------------------------------
# Hogbom CLEAN
# ---------------------------------------------------------------------------

class HogbomDeconvolver(BaseDeconvolver):
    """
    Classical Hogbom CLEAN minor cycle.

    _predict returns a delta function at the peak pixel — the simplest possible
    component model. Equivalent to standard CLEAN minor cycle.
    """

    def _predict(
        self, island: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        component = torch.zeros_like(island)
        flat      = int(island.abs().argmax())
        H, W      = island.shape
        component[flat // W, flat % W] = float(island.abs().max())
        return component, None


# ---------------------------------------------------------------------------
# MADClean — learned minor cycle, model-agnostic
# ---------------------------------------------------------------------------

class MADCleanDeconvolver(BaseDeconvolver):
    """
    MAD-CLEAN deconvolver with a learned minor cycle.

    Accepts any solver with a decode_island(island) method:
        PatchSolver, ConvSolver, FlowSolver, DPSSolver, LatentDPSSolver

    If the solver also exposes decode_island_with_uncertainty(island),
    per-pixel uncertainty is accumulated across minor cycle iterations.

    Parameters
    ----------
    solver          : any solver with decode_island()
    psf             : (H, W) numpy array — PSF with peak=1 (CASA convention)
    gain            : float  (default 0.1)
    noise_threshold : float | None
    sidelobe_level  : float | None
    n_major_cycles  : int    (default 10)
    max_minor_iter  : int    (default 100)
    device          : str | None
    verbose         : bool
    """

    def __init__(
        self,
        solver          : object,
        psf             : np.ndarray,
        gain            : float        = 0.1,
        noise_threshold : Optional[float] = None,
        sidelobe_level  : Optional[float] = None,
        n_major_cycles  : int          = 10,
        max_minor_iter  : int          = 100,
        device          : Optional[str] = None,
        verbose         : bool         = True,
    ):
        super().__init__(
            psf             = psf,
            gain            = gain,
            noise_threshold = noise_threshold,
            sidelobe_level  = sidelobe_level,
            n_major_cycles  = n_major_cycles,
            max_minor_iter  = max_minor_iter,
            device          = device,
            verbose         = verbose,
        )
        self.solver      = solver
        self._has_uncert = hasattr(solver, "decode_island_with_uncertainty")
        self._needs_peak_norm = getattr(solver, "_needs_peak_norm", True)

    def _predict(
        self, island: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        island = island.to(self.device)
        if self._has_uncert:
            component, unc = self.solver.decode_island_with_uncertainty(island)
            return component, unc
        else:
            component = self.solver.decode_island(island)
            return component, None

    def __repr__(self) -> str:
        return (f"MADCleanDeconvolver(solver={self.solver.__class__.__name__}, "
                f"gain={self.gain}, sidelobe_level={self._sidelobe_level:.3f}, "
                f"n_major_cycles={self.n_major_cycles}, device={self.device})")
