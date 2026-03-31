"""
mad_clean.hogbom
================
Standalone PyTorch Hogbom CLEAN — Variant 0 classical baseline.

Supports:
  - Classic Hogbom  (use_psf_patch=False): full PSF, boundary-clipped direct subtract
  - PSF-patch mode  (use_psf_patch=True):  truncated PSF patch via compute_psf_patch()
  - Adaptive threshold ramp (cspeedup > 0)
  - CLEAN box to restrict peak search
  - Multi-dimensional input: (H, W), (nchan, H, W), (nstokes, H, W), (nchan, nstokes, H, W)
    Peak detection uses a 2-D guide image (first leading-dim slice).

Public API
----------
hogbom_clean(dirty, psf, ...) -> dict
"""

from __future__ import annotations

import torch

try:
    from mad_clean.psf_utils import compute_psf_patch
except (ImportError, ModuleNotFoundError):
    from psf_utils import compute_psf_patch  # type: ignore[no-redef]

__all__ = ["hogbom_clean"]


# ── index helpers ─────────────────────────────────────────────────────────────

def _clip_box(
    py: int, px: int,
    psf_cy: int, psf_cx: int,
    psf_h: int, psf_w: int,
    H: int, W: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Compute boundary-clipped residual and PSF regions for direct subtraction.

    At image peak (py, px) with PSF peak at (psf_cy, psf_cx) inside a
    (psf_h, psf_w) PSF tensor, returns indices (r0c, r1c, c0c, c1c, pr0, pr1,
    pc0, pc1) such that:

        residual[..., r0c:r1c, c0c:c1c] -= gain * peak_v * psf[pr0:pr1, pc0:pc1]

    Boundary clipping ensures no circular wrap and handles peaks at image edges
    exactly.
    """
    r0, r1 = py - psf_cy, py - psf_cy + psf_h
    c0, c1 = px - psf_cx, px - psf_cx + psf_w

    r0c, r1c = max(0, r0), min(H, r1)
    c0c, c1c = max(0, c0), min(W, c1)

    pr0 = r0c - r0
    pr1 = pr0 + (r1c - r0c)
    pc0 = c0c - c0
    pc1 = pc0 + (c1c - c0c)

    return r0c, r1c, c0c, c1c, pr0, pr1, pc0, pc1


# ── main function ─────────────────────────────────────────────────────────────

def hogbom_clean(
    dirty         : torch.Tensor,
    psf           : torch.Tensor,
    gain          : float            = 0.1,
    threshold     : float | None     = None,
    n_iter        : int              = 1000,
    clean_box     : tuple | None     = None,
    cspeedup      : float            = 0.0,
    use_psf_patch : bool             = True,
    energy_frac   : float            = 0.90,
    device        : str              = "cpu",
) -> dict:
    """
    Run Hogbom CLEAN deconvolution.

    Parameters
    ----------
    dirty         : (H, W) or (*, H, W) float32 tensor
    psf           : (H, W) or (*, H, W) float32 tensor, peak at centre (H//2, W//2)
    gain          : loop gain applied at each iteration (default 0.1)
    threshold     : stop when |guide peak| < threshold.
                    None (default) = auto-compute as 0.1 × dirty.abs().max()
                    (10% of initial dirty peak).
                    Pass 0.0 to disable early stopping.
    n_iter        : maximum iterations (default 1000)
    clean_box     : (r0, r1, c0, c1) restrict peak search to this box, or None
    cspeedup      : adaptive threshold ramp exponent — eff_thresh = threshold*2^(it/cspeedup).
                    Set to 0.0 to disable (fixed threshold).
    use_psf_patch : True  = truncated PSF patch (compute_psf_patch).
                    False = classic full PSF, boundary-clipped direct subtract.
    energy_frac   : PSF patch energy threshold [use_psf_patch=True only] (default 0.90)
    device        : "cpu" or "cuda"

    Returns
    -------
    dict with keys:
        model     : torch.Tensor — same shape as dirty
        residual  : torch.Tensor — same shape as dirty
        n_iter    : int — number of iterations executed
        converged : bool — True if stopped early by threshold
        peak_flux : float — model.abs().max() at end
    """
    dev      = torch.device(device)
    dirty    = dirty.float().to(dev)
    psf      = psf.float().to(dev)
    residual = dirty.clone()
    model    = torch.zeros_like(dirty)

    H, W = dirty.shape[-2], dirty.shape[-1]

    # ── PSF setup ─────────────────────────────────────────────────────────────
    # Reference PSF (always 2D): first slice for multi-channel input.
    psf_ref = psf if psf.ndim == 2 else psf.reshape(-1, H, W)[0]

    if use_psf_patch:
        psf_work, _ = compute_psf_patch(psf_ref, energy_frac=energy_frac)
    else:
        psf_work = psf_ref

    psf_h, psf_w = psf_work.shape
    cy_p, cx_p   = psf_h // 2, psf_w // 2

    # ── stopping threshold ────────────────────────────────────────────────────
    # Default: 10% of dirty peak.  Physical interpretation: stop when the
    # residual peak has dropped to 1/10th of the initial dirty image peak.
    if threshold is None:
        threshold = 0.1 * float(dirty.abs().max())

    # ── guide image: always 2D ────────────────────────────────────────────────
    def _guide(r: torch.Tensor) -> torch.Tensor:
        """First spatial slice for multi-dim tensors; identity for 2D."""
        return r if r.ndim == 2 else r.reshape(-1, H, W)[0]

    converged = False
    n_done    = 0

    for it in range(n_iter):
        guide = _guide(residual)

        # ── peak search ───────────────────────────────────────────────────────
        if clean_box is not None:
            r0b, r1b, c0b, c1b = clean_box
            search = guide[r0b:r1b, c0b:c1b]
            flat   = int(search.abs().argmax())
            sw     = c1b - c0b
            py     = r0b + flat // sw
            px     = c0b + flat % sw
        else:
            flat = int(guide.abs().argmax())
            py   = flat // W
            px   = flat % W

        peak_v_guide = float(guide[py, px])

        # ── convergence check ─────────────────────────────────────────────────
        eff_thresh = (threshold * float(2.0 ** (it / cspeedup))
                      if cspeedup > 0.0 else threshold)
        if abs(peak_v_guide) < eff_thresh:
            converged = True
            n_done    = it
            break

        # ── model update: accumulate peak_vals across all leading dims ────────
        # residual[..., py, px] returns a scalar for 2D input or (*,) for ND.
        peak_vals = residual[..., py, px].clone()
        model[..., py, px] = model[..., py, px] + gain * peak_vals

        # ── direct PSF subtract (boundary-clipped, no circular wrap) ─────────
        r0c, r1c, c0c, c1c, pr0, pr1, pc0, pc1 = _clip_box(
            py, px, cy_p, cx_p, psf_h, psf_w, H, W
        )
        psf_slice = psf_work[pr0:pr1, pc0:pc1]   # (sh, sw)

        # peak_vals[..., None, None] → (*,1,1), broadcasts with psf_slice (sh,sw)
        residual[..., r0c:r1c, c0c:c1c] -= gain * peak_vals[..., None, None] * psf_slice

        n_done = it + 1

    else:
        n_done = n_iter

    return {
        "model"     : model,
        "residual"  : residual,
        "n_iter"    : n_done,
        "converged" : converged,
        "peak_flux" : float(model.abs().max()),
    }
