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
    "render_gaussian_blob_t",
    "render_shell_t",
    "render_filament_t",
    "fft_convolve_t",
    "assemble_mixed_field_t",
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
    images:    torch.Tensor,
    psf:       torch.Tensor | None = None,
    psf_shift: tuple[int, int] | None = None,
    psf_fft:   torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve (B, H, W) images with a single (H, W) PSF. Returns (B, H, W).

    Fast path: pass psf_fft = rfft2(roll(psf, shift)) precomputed once per PSF.
    Then no rfft2(psf) and no roll on the (B, H, W) output.
    """
    B, H, W = images.shape
    fi = torch.fft.rfft2(images, s=(H, W))
    if psf_fft is not None:
        return torch.fft.irfft2(fi * psf_fft, s=(H, W))
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


# ---------------------------------------------------------------------------
# Single-sample _t wrappers (used by PatchFlowDatasetTorch per-sample calls)
# ---------------------------------------------------------------------------

def _t1(v: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.float32, device=device)


def fft_convolve_t(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Convolve a single (H, W) image with a single (H, W) PSF."""
    return fft_convolve_batch(image.unsqueeze(0), psf=psf).squeeze(0)


def render_gaussian_blob_t(
    size: int, cx: float, cy: float,
    flux_jy: float, sig_maj: float, sig_min: float, pa: float,
    device: torch.device,
) -> torch.Tensor:
    d = device
    return render_gaussian_blob_batch(
        size,
        _t1(cx, d), _t1(cy, d), _t1(flux_jy, d),
        _t1(sig_maj, d), _t1(sig_min, d), _t1(pa, d),
        d,
    ).squeeze(0)


def render_shell_t(
    size: int, cx: float, cy: float,
    flux_jy: float, radius: float, thickness: float,
    device: torch.device,
) -> torch.Tensor:
    d = device
    return render_shell_batch(
        size,
        _t1(cx, d), _t1(cy, d), _t1(flux_jy, d),
        _t1(radius, d), _t1(thickness, d),
        d,
    ).squeeze(0)


def render_filament_t(
    size: int, cx: float, cy: float,
    flux_jy: float, length: float, width: float, pa: float,
    device: torch.device,
) -> torch.Tensor:
    d = device
    return render_filament_batch(
        size,
        _t1(cx, d), _t1(cy, d), _t1(flux_jy, d),
        _t1(length, d), _t1(width, d), _t1(pa, d),
        d,
    ).squeeze(0)


def assemble_mixed_field_t(
    size: int,
    n_sources: int,
    flux_range_jy: tuple[float, float] = (1e-4, 1e-1),
    extended_rate: float = 0.05,
    edge_margin: int = 16,
    rng: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Assemble a mixed point/extended source field on device. Returns (H, W)."""
    device = torch.device(device)
    lo = float(edge_margin)
    hi = float(size - edge_margin)
    s_min, s_max = flux_range_jy

    sky = torch.zeros(size, size, dtype=torch.float32, device=device)

    def _rand() -> float:
        return float(torch.rand(1, generator=rng, device=device).item())

    for _ in range(n_sources):
        cx = lo + _rand() * (hi - lo)
        cy = lo + _rand() * (hi - lo)
        flux = float(math.exp(math.log(s_min) + _rand() * math.log(s_max / s_min)))

        if _rand() < extended_rate:
            kind = int(_rand() * 3)
            if kind == 0:
                sig_maj = BEAM_SIGMA_PX + _rand() * (SIGMA_MAX_PX - BEAM_SIGMA_PX)
                sig_min = BEAM_SIGMA_PX + _rand() * (sig_maj - BEAM_SIGMA_PX)
                pa = _rand() * math.pi
                sky = sky + render_gaussian_blob_t(size, cx, cy, flux, sig_maj, sig_min, pa, device)
            elif kind == 1:
                radius    = BEAM_SIGMA_PX + _rand() * (SIGMA_MAX_PX - BEAM_SIGMA_PX)
                thickness = BEAM_SIGMA_PX * 0.5 + _rand() * max(BEAM_SIGMA_PX, radius * 0.5)
                sky = sky + render_shell_t(size, cx, cy, flux, radius, thickness, device)
            else:
                length = 2 * BEAM_SIGMA_PX + _rand() * (2 * SIGMA_MAX_PX - 2 * BEAM_SIGMA_PX)
                width  = BEAM_SIGMA_PX + _rand() * (SIGMA_MAX_PX - BEAM_SIGMA_PX)
                pa = _rand() * math.pi
                sky = sky + render_filament_t(size, cx, cy, flux, length, width, pa, device)
        else:
            r = max(0, min(size - 1, int(round(cy))))
            c = max(0, min(size - 1, int(round(cx))))
            sky[r, c] = sky[r, c] + flux

    return sky
