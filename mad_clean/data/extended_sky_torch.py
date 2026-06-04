"""Batched GPU sky generators for PatchFlow data generation.

All renderers operate on batches: parameters are (B,) tensors, outputs are
(B, H, W). This keeps the GPU fully occupied during data generation.

fft_convolve_batch convolves a (B, H, W) image batch against a single (H, W)
PSF using batched rfft2.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "render_gaussian_blob_batch",
    "render_shell_batch",
    "render_filament_batch",
    "fft_convolve_batch",
    "BEAM_SIGMA_PX",
    "SIGMA_MAX_PX",
]

BEAM_SIGMA_PX: float = 1.4
SIGMA_MAX_PX: float  = 128.0


# ---------------------------------------------------------------------------
# Shared grid (cached per size/device call)
# ---------------------------------------------------------------------------

def _xy_grids_batch(size: int, B: int, device: torch.device):
    """Return (X, Y) grids of shape (1, H, W) for broadcasting over batch."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")   # (H, W)
    return X.unsqueeze(0), Y.unsqueeze(0)                  # (1, H, W)


# ---------------------------------------------------------------------------
# Batched FFT convolution
# ---------------------------------------------------------------------------

def fft_convolve_batch(
    images:  torch.Tensor,
    psf:     torch.Tensor,
    psf_shift: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convolve (B, H, W) images with a single (H, W) PSF. Returns (B, H, W).

    psf_shift : (py, px) of PSF peak -- precompute once and pass in to avoid
                repeated nonzero() calls.
    """
    B, H, W = images.shape
    fi  = torch.fft.rfft2(images, s=(H, W))
    fp  = torch.fft.rfft2(psf.unsqueeze(0), s=(H, W))
    out = torch.fft.irfft2(fi * fp, s=(H, W))
    if psf_shift is None:
        py, px = (psf == psf.max()).nonzero(as_tuple=False)[0]
        psf_shift = (-int(py), -int(px))
    out = torch.roll(out, shifts=psf_shift, dims=(1, 2))
    return out


# ---------------------------------------------------------------------------
# Batched renderers
# ---------------------------------------------------------------------------

def render_gaussian_blob_batch(
    size:    int,
    cx:      torch.Tensor,   # (B,)
    cy:      torch.Tensor,   # (B,)
    flux:    torch.Tensor,   # (B,)
    sig_maj: torch.Tensor,   # (B,)
    sig_min: torch.Tensor,   # (B,)
    pa:      torch.Tensor,   # (B,)
    device:  torch.device,
) -> torch.Tensor:           # (B, H, W)
    B = cx.shape[0]
    X, Y = _xy_grids_batch(size, B, device)          # (1, H, W)
    cx = cx[:, None, None]; cy = cy[:, None, None]
    sig_maj = sig_maj[:, None, None]; sig_min = sig_min[:, None, None]
    cos_pa = torch.cos(pa)[:, None, None]
    sin_pa = torch.sin(pa)[:, None, None]

    dx = X - cx; dy = Y - cy
    u =  dx * cos_pa + dy * sin_pa
    v = -dx * sin_pa + dy * cos_pa
    g = torch.exp(-0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2))
    g = g / g.sum(dim=(1, 2), keepdim=True).clamp(min=1e-30)
    return g * flux[:, None, None]


def render_shell_batch(
    size:      int,
    cx:        torch.Tensor,   # (B,)
    cy:        torch.Tensor,   # (B,)
    flux:      torch.Tensor,   # (B,)
    radius:    torch.Tensor,   # (B,)
    thickness: torch.Tensor,   # (B,)
    device:    torch.device,
) -> torch.Tensor:             # (B, H, W)
    X, Y = _xy_grids_batch(size, cx.shape[0], device)
    cx = cx[:, None, None]; cy = cy[:, None, None]
    radius = radius[:, None, None]
    shell_sigma = (thickness / 2.0)[:, None, None]

    r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    profile = torch.exp(-0.5 * ((r - radius) / shell_sigma) ** 2)
    profile = profile / profile.sum(dim=(1, 2), keepdim=True).clamp(min=1e-30)
    return profile * flux[:, None, None]


def render_filament_batch(
    size:   int,
    cx:     torch.Tensor,   # (B,)
    cy:     torch.Tensor,   # (B,)
    flux:   torch.Tensor,   # (B,)
    length: torch.Tensor,   # (B,)
    width:  torch.Tensor,   # (B,)
    pa:     torch.Tensor,   # (B,)
    device: torch.device,
) -> torch.Tensor:          # (B, H, W)
    X, Y = _xy_grids_batch(size, cx.shape[0], device)
    cx = cx[:, None, None]; cy = cy[:, None, None]
    width_b = width[:, None, None]
    half_len = (length / 2.0)[:, None, None]
    cos_pa = torch.cos(pa)[:, None, None]
    sin_pa = torch.sin(pa)[:, None, None]

    dx = X - cx; dy = Y - cy
    along  =  dx * cos_pa + dy * sin_pa
    across = -dx * sin_pa + dy * cos_pa

    taper_sigma = width_b.clamp(min=BEAM_SIGMA_PX)
    sqrt2 = math.sqrt(2) * taper_sigma
    along_profile = (
        0.5 * (1.0 + torch.erf((along + half_len) / sqrt2))
        - 0.5 * (1.0 + torch.erf((along - half_len) / sqrt2))
    )
    across_profile = torch.exp(-0.5 * (across / width_b) ** 2)

    profile = along_profile * across_profile
    profile = profile / profile.sum(dim=(1, 2), keepdim=True).clamp(min=1e-30)
    return profile * flux[:, None, None]
