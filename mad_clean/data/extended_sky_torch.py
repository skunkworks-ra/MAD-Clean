"""PyTorch GPU sky generators for PatchFlow training.

Mirrors extended_sky.py but operates entirely in torch, enabling GPU-side
scene generation and FFT convolution. No numpy, no scipy.

All generators accept a torch.Generator for reproducibility and operate on
a specified device.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "render_gaussian_blob_t",
    "render_shell_t",
    "render_filament_t",
    "assemble_mixed_field_t",
    "fft_convolve_t",
]

BEAM_SIGMA_PX: float = 1.4
SIGMA_MAX_PX: float = 128.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xy_grids(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, Y) float32 meshgrids. X=col, Y=row."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    return X, Y


def fft_convolve_t(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """FFT-based 'same' convolution of (H, W) image with (H, W) psf."""
    H, W = image.shape
    # Pad to avoid wrap-around
    fh = torch.fft.rfft2(image, s=(H, W))
    fp = torch.fft.rfft2(psf,   s=(H, W))
    conv = torch.fft.irfft2(fh * fp, s=(H, W))
    # Circular shift to get 'same' output (PSF peak at [0,0] after rfft)
    py, px = (psf == psf.max()).nonzero(as_tuple=False)[0]
    conv = torch.roll(conv, shifts=(-int(py), -int(px)), dims=(0, 1))
    return conv


# ---------------------------------------------------------------------------
# Generator 1: Anisotropic Gaussian blob
# ---------------------------------------------------------------------------

def render_gaussian_blob_t(
    size: int,
    cx: float, cy: float,
    flux_jy: float,
    sig_maj: float, sig_min: float,
    pa: float,
    device: torch.device,
) -> torch.Tensor:
    """(size, size) float32 Gaussian blob, sums to flux_jy."""
    X, Y = _xy_grids(size, device)
    dx = X - cx
    dy = Y - cy
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    u =  dx * cos_pa + dy * sin_pa
    v = -dx * sin_pa + dy * cos_pa
    exponent = 0.5 * ((u / sig_maj) ** 2 + (v / sig_min) ** 2)
    g = torch.exp(-exponent)
    g = g / g.sum().clamp(min=1e-30)
    return (g * flux_jy).float()


# ---------------------------------------------------------------------------
# Generator 2: Limb-brightened shell
# ---------------------------------------------------------------------------

def render_shell_t(
    size: int,
    cx: float, cy: float,
    flux_jy: float,
    radius: float,
    thickness: float,
    device: torch.device,
) -> torch.Tensor:
    """(size, size) float32 limb-brightened shell, sums to flux_jy."""
    X, Y = _xy_grids(size, device)
    r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    shell_sigma = thickness / 2.0
    profile = torch.exp(-0.5 * ((r - radius) / shell_sigma) ** 2)
    profile = profile / profile.sum().clamp(min=1e-30)
    return (profile * flux_jy).float()


# ---------------------------------------------------------------------------
# Generator 3: Filament
# ---------------------------------------------------------------------------

def render_filament_t(
    size: int,
    cx: float, cy: float,
    flux_jy: float,
    length: float,
    width: float,
    pa: float,
    device: torch.device,
) -> torch.Tensor:
    """(size, size) float32 filament, sums to flux_jy."""
    X, Y = _xy_grids(size, device)
    dx = X - cx
    dy = Y - cy
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    along  =  dx * cos_pa + dy * sin_pa
    across = -dx * sin_pa + dy * cos_pa

    half_len = length / 2.0
    taper_sigma = max(width, BEAM_SIGMA_PX)
    sqrt2 = math.sqrt(2) * taper_sigma
    along_profile = (
        0.5 * (1.0 + torch.erf((along + half_len) / sqrt2))
        - 0.5 * (1.0 + torch.erf((along - half_len) / sqrt2))
    )
    across_profile = torch.exp(-0.5 * (across / width) ** 2)

    profile = along_profile * across_profile
    profile = profile / profile.sum().clamp(min=1e-30)
    return (profile * flux_jy).float()


# ---------------------------------------------------------------------------
# Mixed-field assembler
# ---------------------------------------------------------------------------

def assemble_mixed_field_t(
    size: int,
    n_sources: int,
    flux_range_jy: tuple[float, float],
    extended_rate: float,
    edge_margin: int,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Assemble a (size, size) distractor field on GPU.

    Returns sky tensor only -- targets not needed for PatchFlow distractors.
    """
    sky = torch.zeros(size, size, device=device)
    s_min, s_max = flux_range_jy
    lo = float(edge_margin)
    hi = float(size - edge_margin)

    def _uniform(lo: float, hi: float) -> float:
        return float(torch.empty(1, device=device).uniform_(lo, hi, generator=rng))

    for _ in range(n_sources):
        flux = _uniform(s_min, s_max)
        cx   = _uniform(lo, hi)
        cy   = _uniform(lo, hi)
        is_extended = _uniform(0.0, 1.0) < extended_rate

        if is_extended:
            kind = int(torch.randint(0, 3, (1,), device=device, generator=rng))
            if kind == 0:
                sig_maj = _uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)
                sig_min = _uniform(BEAM_SIGMA_PX, sig_maj)
                pa      = _uniform(0.0, math.pi)
                sky += render_gaussian_blob_t(size, cx, cy, flux, sig_maj, sig_min, pa, device)
            elif kind == 1:
                radius    = _uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)
                thickness = _uniform(BEAM_SIGMA_PX * 0.5, max(BEAM_SIGMA_PX, radius * 0.5))
                sky += render_shell_t(size, cx, cy, flux, radius, thickness, device)
            else:
                length = _uniform(2 * BEAM_SIGMA_PX, 2 * SIGMA_MAX_PX)
                width  = _uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX)
                pa     = _uniform(0.0, math.pi)
                sky += render_filament_t(size, cx, cy, flux, length, width, pa, device)
        else:
            r = max(int(edge_margin), min(size - edge_margin - 1, int(round(cy))))
            c = max(int(edge_margin), min(size - edge_margin - 1, int(round(cx))))
            sky[r, c] += flux

    return sky
