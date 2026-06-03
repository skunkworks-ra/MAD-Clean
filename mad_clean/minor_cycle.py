"""MDN-Asp minor cycle.

Public API
----------
render_aspen(aspen6d, shape, cx, cy) -> np.ndarray
    Place an Aspen Gaussian into a model image of given shape.

minor_cycle(residual, psf, sigma, config_idx, model, ...) -> (model_update, aspen_list)
    Run the iterative MDN-Asp minor cycle and return the model image update
    plus the list of committed (aspen6d, posterior) pairs.

The caller (CASA or a test harness) owns:
  - adding model_update into its running model image
  - major-cycle PSF convolution and vis-domain subtract
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from scipy.signal import fftconvolve

from mad_clean.data.cutout_dataset import CutoutDataset, standardise_log_flux
from mad_clean.data.extended_sky import BEAM_SIGMA_PX
from mad_clean.models.mdn_asp import MDNAsp, MixParams, make_cond

__all__ = ["render_aspen", "minor_cycle", "AspenCommit"]

_CUTOUT = 128
_HALF   = _CUTOUT // 2


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AspenCommit:
    """One committed Aspen component."""
    cx:       float        # absolute column position in full image
    cy:       float        # absolute row position in full image
    flux:     float        # committed flux (Jy), after loop_gain scaling
    sig_maj:  float        # major-axis sigma (px)
    sig_min:  float        # minor-axis sigma (px)
    pa:       float        # position angle (rad)
    params:   MixParams    # raw MDN mixture parameters (before mode selection)


# ---------------------------------------------------------------------------
# render_aspen
# ---------------------------------------------------------------------------

def render_aspen(
    cx: float, cy: float,
    flux: float,
    sig_maj: float, sig_min: float,
    pa: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Render one Aspen as a 2D Gaussian model image (Jy/pixel).

    Parameters
    ----------
    cx, cy   : centre position (col, row) in the full image
    flux     : total integrated flux (Jy)
    sig_maj  : major-axis sigma (px)
    sig_min  : minor-axis sigma (px)
    pa       : position angle of major axis (rad, CCW from +col)
    shape    : (H, W) of the output image

    Returns
    -------
    (H, W) float32 array, sums to `flux`.
    """
    H, W = shape
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    C, R = np.meshgrid(cols, rows)
    dc = C - cx
    dr = R - cy
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    u =  dc * cos_pa + dr * sin_pa   # along major axis
    v = -dc * sin_pa + dr * cos_pa   # along minor axis
    exponent = 0.5 * ((u / max(sig_maj, 1e-6)) ** 2
                    + (v / max(sig_min, 1e-6)) ** 2)
    g = np.exp(-exponent)
    g_sum = g.sum()
    if g_sum > 0:
        g *= flux / g_sum
    return np.clip(g, -3.4e38, 3.4e38).astype(np.float32)


# ---------------------------------------------------------------------------
# minor_cycle
# ---------------------------------------------------------------------------

def minor_cycle(
    residual:   np.ndarray,       # (H, W) float32, Jy/beam
    psf:        np.ndarray,       # (H, W) float32, peak = 1
    sigma:      float,            # noise estimate (Jy/beam)
    config_idx: int,              # 0=A 1=B 2=C 3=D
    model:      MDNAsp,
    *,
    loop_gain:   float = 0.1,
    n_sigma_stop: float = 3.0,
    sidelobe_level: float = 0.2,
    divergence_tol: float = 0.05,
    max_components: int = 1000,
    image_mask: np.ndarray | None = None,  # (H, W) bool; peak search restricted to True pixels
    sigma_max_px: float | None = None,     # override SIGMA_MAX_PX clip ceiling
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, list[AspenCommit]]:
    """Run the MDN-Asp minor cycle.

    Returns
    -------
    model_update : (H, W) float32 — accumulated Aspen model image (Jy/pixel).
                   Add this to CASA's running model image.
    commits      : list of AspenCommit, one per accepted component.
    stop_reason  : str — one of "noise_floor", "sidelobe_floor", "divergence", "max_components".
    """
    device = torch.device(device)
    model.eval()

    H, W = residual.shape
    residual = residual.copy().astype(np.float32)
    model_update = np.zeros((H, W), dtype=np.float32)
    commits: list[AspenCommit] = []

    noise_floor  = n_sigma_stop * sigma
    dirty_peak   = float(np.abs(residual).max())
    sidelobe_floor = sidelobe_level * dirty_peak
    prev_peak    = dirty_peak

    stop_reason = "max_components"
    while len(commits) < max_components:
        pos = np.where(residual > 0, residual, 0.0)
        search = pos if image_mask is None else np.where(image_mask, pos, 0.0)
        peak_idx = np.argmax(search)
        peak_row, peak_col = divmod(int(peak_idx), W)
        peak_val = float(residual[peak_row, peak_col])
        abs_peak = abs(peak_val)

        # Stopping rules
        if abs_peak < noise_floor:
            stop_reason = "noise_floor"
            break
        if abs_peak < sidelobe_floor:
            stop_reason = "sidelobe_floor"
            break
        if abs_peak > prev_peak * (1.0 + divergence_tol):
            stop_reason = "divergence"
            break
        prev_peak = abs_peak

        # --- Crop cutout ---
        r0 = peak_row - _HALF
        r1 = r0 + _CUTOUT
        c0 = peak_col - _HALF
        c1 = c0 + _CUTOUT
        res_cut = _safe_crop(residual, r0, r1, c0, c1)
        psf_cut = _crop_psf_centred(psf, _CUTOUT)

        # --- Conditioning ---
        sigma_local = float(1.4826 * np.median(np.abs(res_cut)))
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = sigma
        sig_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t = torch.tensor([config_idx],  dtype=torch.long)
        cond  = make_cond(sig_t, cfg_t).to(device)          # (1, 5)

        img_t = torch.from_numpy(
            np.stack([res_cut, psf_cut], axis=0)[None]       # (1, 2, H, W)
        ).to(device)

        # --- MDN forward pass ---
        with torch.no_grad():
            params = model(img_t, cond)

        mode6d = model.mode(params).squeeze(0).cpu().numpy()  # (6,)
        # mode6d: (x_off, y_off, log_flux_std, log_sig_maj, log_sig_min, pa)

        x_off      = float(mode6d[0])
        y_off      = float(mode6d[1])
        log_flux_std = float(mode6d[2])
        from mad_clean.data.extended_sky import BEAM_SIGMA_PX, SIGMA_MAX_PX
        _sigma_max = sigma_max_px if sigma_max_px is not None else SIGMA_MAX_PX
        sig_maj    = float(np.clip(np.exp(mode6d[3]), BEAM_SIGMA_PX, _sigma_max))
        sig_min    = float(np.clip(np.exp(mode6d[4]), BEAM_SIGMA_PX, sig_maj))
        pa         = float(mode6d[5])

        # Unstandardise flux; clamp to residual peak to guard against OOD extrapolation
        from mad_clean.data.cutout_dataset import unstandardise_log_flux
        log_flux = unstandardise_log_flux(log_flux_std)
        flux_full = float(np.clip(np.exp(log_flux), 0.0, dirty_peak))

        # Committed flux with loop gain
        flux_commit = loop_gain * flux_full

        # Absolute position in full image
        cx = float(peak_col) + x_off
        cy = float(peak_row) + y_off

        # --- Accumulate model image ---
        aspen_img = render_aspen(cx, cy, flux_commit, sig_maj, sig_min, pa, (H, W))
        model_update += aspen_img

        # --- Subtract PSF response from working residual ---
        psf_response = fftconvolve(aspen_img, psf, mode="same").astype(np.float32)
        residual -= psf_response

        commits.append(AspenCommit(
            cx=cx, cy=cy,
            flux=flux_commit,
            sig_maj=sig_maj, sig_min=sig_min, pa=pa,
            params=MixParams(
                logits=params.logits.cpu(),
                mu=params.mu.cpu(),
                log_std=params.log_std.cpu(),
            ),
        ))

    if commits:
        sig_majs = np.array([c.sig_maj for c in commits])
        sig_mins = np.array([c.sig_min for c in commits])
        fluxes   = np.array([c.flux    for c in commits])
        cxs      = np.array([c.cx      for c in commits])
        cys      = np.array([c.cy      for c in commits])
        print(f"  [minor] n={len(commits)}  "
              f"sig_maj med/max={np.median(sig_majs):.2f}/{sig_majs.max():.2f}px  "
              f"sig_min med/max={np.median(sig_mins):.2f}/{sig_mins.max():.2f}px  "
              f"flux med/max={np.median(fluxes):.4f}/{fluxes.max():.4f} Jy  "
              f"cx spread={cxs.std():.1f}px  cy spread={cys.std():.1f}px  "
              f"stop={stop_reason}")

    return model_update, commits, stop_reason


# ---------------------------------------------------------------------------
# Internal helpers (mirrors cutout_dataset.py)
# ---------------------------------------------------------------------------

def _safe_crop(arr, r0, r1, c0, c1):
    H, W = arr.shape
    out = np.zeros((r1 - r0, c1 - c0), dtype=arr.dtype)
    sr0 = max(0, r0); sr1 = min(H, r1)
    sc0 = max(0, c0); sc1 = min(W, c1)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    dr0 = sr0 - r0
    dc0 = sc0 - c0
    out[dr0:dr0+(sr1-sr0), dc0:dc0+(sc1-sc0)] = arr[sr0:sr1, sc0:sc1]
    return out


def _crop_psf_centred(psf, size):
    if psf.shape == (size, size):
        return psf.copy()
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = size // 2
    return _safe_crop(psf, py - half, py - half + size,
                           px - half, px - half + size)
