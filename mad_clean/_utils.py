"""
mad_clean._utils
================
Shared low-level utilities — eliminates duplicated implementations across
hogbom.py, deconvolver.py, solvers.py, and training modules.

    soft_threshold  — 3 copies → 1
    clip_box        — 2 copies → 1
    guide           — 2 copies → 1
"""

from __future__ import annotations

import torch

__all__ = ["soft_threshold", "clip_box", "guide"]


def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Element-wise soft thresholding: sign(x) * max(|x| - threshold, 0)."""
    return x.sign() * (x.abs() - threshold).clamp(min=0.0)


def clip_box(
    py: int, px: int,
    psf_cy: int, psf_cx: int,
    psf_h: int, psf_w: int,
    H: int, W: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Boundary-clipped residual and PSF regions for direct PSF subtraction.

    At image peak (py, px) with PSF peak at (psf_cy, psf_cx) inside a
    (psf_h, psf_w) PSF tensor, returns indices (r0c, r1c, c0c, c1c, pr0, pr1,
    pc0, pc1) such that:

        residual[..., r0c:r1c, c0c:c1c] -= gain * peak_v * psf[pr0:pr1, pc0:pc1]

    Clips to image bounds — no circular wrap, safe at image edges.
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


def guide(r: torch.Tensor) -> torch.Tensor:
    """
    Return the 2D guide image from a potentially multi-dimensional tensor.

    For 2D input returns the tensor unchanged.
    For N-dimensional input (nchan, H, W) or (nchan, nstokes, H, W) returns
    the first spatial slice — the channel/Stokes used for peak detection.
    """
    if r.ndim == 2:
        return r
    H, W = r.shape[-2], r.shape[-1]
    return r.reshape(-1, H, W)[0]
