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

__all__ = ["render_aspen", "minor_cycle", "AspenCommit",
           "minor_cycle_flow", "FlowCommit",
           "minor_cycle_flow_greedy"]

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
    abs_noise_floor: float | None = None,  # absolute floor (Jy/beam); overrides n_sigma_stop*sigma when set
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

    noise_floor  = abs_noise_floor if abs_noise_floor is not None else n_sigma_stop * sigma
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
# Flow minor cycle (PSFCondFlow — pixel-window solver)
# ---------------------------------------------------------------------------

@dataclass
class FlowCommit:
    """One committed flow window."""
    cx:    float    # window-centre column in full image
    cy:    float    # window-centre row in full image
    flux:  float    # committed flux (Jy) = loop_gain * sum(thresholded window)


def _tile_origins(L: int, tile: int, stride: int) -> list[int]:
    """Tile top-left coords covering [0, L), last tile flush to the edge."""
    if L <= tile:
        return [0]
    os = list(range(0, L - tile + 1, stride))
    if os[-1] != L - tile:
        os.append(L - tile)
    return os


def minor_cycle_flow(
    residual:   np.ndarray,       # (H, W) float32, Jy/beam
    psf:        np.ndarray,       # (H, W) float32, peak = 1
    sigma:      float,            # noise estimate (Jy/beam)
    model,                        # PSFCondFlow
    *,
    threshold:    float | None = None,  # per-tile stop (this cycle's 3-sigma)
    loop_gain:    float = 0.1,
    n_sigma_stop: float = 3.0,
    speckle_frac:    float = 0.01,
    conf_k:          float = 3.0,    # keep pixels with median >= conf_k * MAD(draws)
    n_samples:       int   = 8,
    n_steps:         int   = 50,
    tile:            int   = 128,
    stride:          int   = 64,     # 50% overlap -> feathered overlap-add
    inner_max:       int   = 50,     # max flow passes per tile (clean-to-stop cap)
    max_components:  int   = 10000,  # total flow passes per minor cycle (safety)
    image_mask:  np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, list[FlowCommit], str]:
    """Run the PSFCondFlow minor cycle as a TILE SWEEP (not peak-greedy).

    Global quantities (noise, threshold) are computed on the full image by the
    caller, once per major cycle, and passed in.  Here we sweep a fixed,
    overlapping tile grid over the masked region and clean EACH tile down to the
    per-cycle stopping criterion (`threshold`), so diffuse emission that never
    forms a global peak is still covered.  Per tile, per pass:
      1. sample the flow on (residual tile, centred PSF), zero conditioning;
      2. confidence-gate the posterior (median >= conf_k * MAD across draws) so
         only real emission, not speckle, is kept;
      3. commit loop_gain * gated_window, FEATHERED (Hann) for seamless
         overlap-add into the model;
      4. clean the tile residual over that beam-area footprint by loop_gain
         (NO convolution — the major cycle re-images exactly);
    repeat on the tile until its in-mask peak < threshold (or the gate empties).
    """
    import torch as _torch
    from mad_clean.models.mdn_asp import COND_DIM

    device = _torch.device(device)
    model.eval()
    H, W = residual.shape
    residual = residual.copy().astype(np.float32)
    model_update = np.zeros((H, W), dtype=np.float32)
    commits: list[FlowCommit] = []

    stop = threshold if threshold is not None else n_sigma_stop * sigma
    zero_cond = _torch.zeros(1, COND_DIM, device=device)
    psf_cut = _crop_psf_centred(psf, tile)
    psf_t = _torch.from_numpy(psf_cut).float().to(device)

    # Hann feather (peak 1 centre, ->0 at tile edges) for overlap-add blending.
    w1 = np.hanning(tile).astype(np.float32)
    feather = np.outer(w1, w1); feather /= float(feather.max())

    total = 0
    for r0 in _tile_origins(H, tile, stride):
        for c0 in _tile_origins(W, tile, stride):
            r1 = r0 + tile; c1 = c0 + tile
            tmask = (np.ones((tile, tile), bool) if image_mask is None
                     else image_mask[r0:r1, c0:c1])
            if not tmask.any():
                continue
            sub = residual[r0:r1, c0:c1]            # view into residual
            for _ in range(inner_max):
                if total >= max_components:
                    break
                masked = np.where(tmask, sub, -np.inf)
                pr, pc = np.unravel_index(int(masked.argmax()), masked.shape)
                peak = float(sub[pr, pc])
                if peak < stop:
                    break
                img_t = _torch.stack([
                    _torch.from_numpy(sub.copy()).float().to(device), psf_t
                ])[None]                             # (1, 2, tile, tile)
                with _torch.no_grad():
                    draws = model.sample(img_t, zero_cond,
                                         n_samples=n_samples, n_steps=n_steps).squeeze(0)
                    med = draws.median(dim=0).values
                    mad = 1.4826 * (draws - med).abs().median(dim=0).values
                    wpk = float(med.max())
                    keep = (med >= conf_k * mad) & (med >= speckle_frac * wpk)
                    win = _torch.where(keep, med,
                                       _torch.zeros_like(med)).cpu().numpy().astype(np.float32)
                if wpk <= 0 or not win.any():
                    break
                commit = (loop_gain * win * feather).astype(np.float32)
                commit = np.where(tmask, commit, 0.0).astype(np.float32)
                model_update[r0:r1, c0:c1] += commit
                # Local beam-area residual clean (no convolution); feathered so
                # tile edges are left for the overlapping neighbour to clean.
                g = commit > 0
                sub[g] *= (1.0 - loop_gain * feather[g])
                sub[pr, pc] *= (1.0 - loop_gain)     # guarantee the tile peak drains
                commits.append(FlowCommit(cx=float(c0 + tile / 2),
                                          cy=float(r0 + tile / 2),
                                          flux=float(commit.sum())))
                total += 1
            if total >= max_components:
                break
        if total >= max_components:
            break

    stop_reason = "max_components" if total >= max_components else "tiles_below_threshold"
    if commits:
        fluxes = np.array([c.flux for c in commits])
        print(f"  [flow] passes={len(commits)}  stop={stop:.4g}  "
              f"flux/pass med/max={np.median(fluxes):.4g}/{fluxes.max():.4g} Jy  "
              f"total={fluxes.sum():.4g} Jy  stop_reason={stop_reason}")

    return model_update, commits, stop_reason


# ---------------------------------------------------------------------------
# Greedy flow minor cycle (à-la-CLEAN: peak -> flow window -> connected
# component -> commit -> gated PSF subtraction -> next peak)
# ---------------------------------------------------------------------------

def minor_cycle_flow_greedy(
    residual:   np.ndarray,       # (H, W) float32, Jy/beam
    psf:        np.ndarray,       # (H, W) float32, peak = 1
    sigma:      float,            # noise estimate (Jy/beam)
    model,                        # PSFCondFlow
    *,
    threshold:    float | None = None,  # peak stop (Jy/beam); default n_sigma_stop*sigma
    n_sigma_stop: float = 3.0,
    gain:         float = 0.6,    # fraction of the connected component committed per visit
    eps_frac:     float = 0.01,   # binarise the flow window at this fraction of its peak
    n_samples:    int   = 8,
    n_steps:      int   = 50,
    tile:         int   = 128,
    max_components: int = 2000,   # safety cap on greedy iterations
    image_mask:  np.ndarray | None = None,
    snapshot_every: int = 0,                 # write a model snapshot every N commits (0=off)
    snapshot_cb=None,                        # callable(model_update_copy, n_committed)
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, list[FlowCommit], str]:
    """Greedy CLEAN with the flow as the (extended) component model.

    Per iteration:
      1. find the residual peak inside the mask;
      2. crop a `tile`x`tile` window CENTRED on the peak;
      3. sample the flow (zero conditioning), take the posterior median window;
      4. DESPECKLE: keep only the connected component of the window that contains
         the central pixel (the real peak); every disconnected dot the flow
         painted elsewhere is dropped;
      5. commit `gain` * component (Jy/pixel) into the model;
      6. SUBTRACT `gain` * PSF (conv) committed_component from the residual, so the
         component AND its sidelobes leave and the next true peak emerges.
    Repeat until the in-mask peak < threshold.  The caller's major cycle re-images
    exactly (vis-domain), which is the regulariser; `gain` < 1 absorbs per-call
    flow error before that.
    """
    import torch as _torch
    from scipy.ndimage import label
    from mad_clean.models.mdn_asp import COND_DIM

    device = _torch.device(device)
    model.eval()
    H, W = residual.shape
    residual = residual.copy().astype(np.float32)
    model_update = np.zeros((H, W), dtype=np.float32)
    commits: list[FlowCommit] = []

    stop = threshold if threshold is not None else n_sigma_stop * sigma
    half = tile // 2
    zero_cond = _torch.zeros(1, COND_DIM, device=device)
    psf_cut = _crop_psf_centred(psf, tile)
    psf_t = _torch.from_numpy(psf_cut).float().to(device)

    from scipy.ndimage import binary_dilation
    valid = np.ones((H, W), dtype=bool) if image_mask is None else image_mask.astype(bool)
    done  = np.zeros((H, W), dtype=bool)   # sources already modelled THIS minor cycle

    stop_reason = "peak_below_threshold"
    while len(commits) < max_components:
        # peak search excludes the mask AND sources already committed this cycle
        search = np.where(valid & ~done, residual, -np.inf)
        pr, pc = np.unravel_index(int(np.argmax(search)), (H, W))
        peak = float(residual[pr, pc])
        if peak < stop:
            stop_reason = "peak_below_threshold"
            break

        # window centred on the peak
        r0 = pr - half; c0 = pc - half
        res_cut = _safe_crop(residual, r0, r0 + tile, c0, c0 + tile)

        img_t = _torch.stack([
            _torch.from_numpy(res_cut.copy()).float().to(device), psf_t
        ])[None]                                 # (1, 2, tile, tile)
        with _torch.no_grad():
            draws = model.sample(img_t, zero_cond,
                                 n_samples=n_samples, n_steps=n_steps).squeeze(0)
            med = draws.median(dim=0).values.cpu().numpy().astype(np.float32)

        wpk = float(med.max())
        if wpk <= 0:
            done[pr, pc] = True   # flow predicts nothing here; skip this peak
            continue

        # --- DESPECKLE: connected component containing the centre ---
        binary = med > eps_frac * wpk
        lab, n = label(binary)
        seed = lab[half, half]
        if seed == 0:
            # the real peak isn't where the flow put emission; take the component
            # nearest the centre (the flow's own brightest blob)
            seed = lab[np.unravel_index(int(med.argmax()), med.shape)]
        if seed == 0:
            done[pr, pc] = True
            continue
        component = np.where(lab == seed, med, 0.0).astype(np.float32)

        # --- place component into a full-image increment ---
        incr = np.zeros((H, W), dtype=np.float32)
        sr0 = max(0, r0); sr1 = min(H, r0 + tile)
        sc0 = max(0, c0); sc1 = min(W, c0 + tile)
        incr[sr0:sr1, sc0:sc1] = component[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0]
        incr = np.where(valid, incr, 0.0).astype(np.float32)

        commit = (gain * incr).astype(np.float32)
        model_update += commit

        # --- gated PSF subtraction (Cotton-Schwab / Clark) ---
        residual -= fftconvolve(commit, psf, mode="same").astype(np.float32)

        # --- A: one component per source. Exclude this source's footprint (plus a
        #        ~beam margin) from the rest of this minor cycle so we never
        #        revisit it and multiple-count the extended flux. The major cycle
        #        re-images exactly; the next minor cycle peels it again. ---
        done |= binary_dilation(incr > 0, iterations=3)

        commits.append(FlowCommit(cx=float(pc), cy=float(pr),
                                  flux=float(commit.sum())))

        if snapshot_every and snapshot_cb is not None and (len(commits) % snapshot_every == 0):
            snapshot_cb(model_update.copy(), len(commits))
    else:
        stop_reason = "max_components"

    if commits:
        print(f"  [greedy] components={len(commits)}  stop={stop:.4g}  "
              f"stop_reason={stop_reason}")

    return model_update, commits, stop_reason


# ---------------------------------------------------------------------------
# Refit pass
# ---------------------------------------------------------------------------

def refit_pass(
    commits:      list[AspenCommit],
    post_residual: np.ndarray,      # (H, W) residual after the major-cycle tclean
    psf:          np.ndarray,       # (H, W) float32, peak=1
    model_update: np.ndarray,       # (H, W) accumulated model from greedy pass
    model:        MDNAsp,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """One-pass refit of all committed components.

    For each component:
      1. Reconstruct the leave-one-out residual by adding back this component's
         PSF footprint to the post-minor-cycle residual.
      2. Re-run the MDN for shape (sig_maj, sig_min, PA, x_off, y_off) only --
         no SIGMA_MAX_PX ceiling so larger scales can be recovered.
      3. Read amplitude directly from the leave-one-out residual peak (signed --
         negative corrects overshoot from the greedy pass).
      4. Replace the old component in model_update with the new one.

    Returns the updated model_update, clipped to net non-negative pixelwise.
    """
    from mad_clean.data.extended_sky import BEAM_SIGMA_PX

    device = torch.device(device)
    model.eval()
    H, W = post_residual.shape

    new_model = model_update.copy()

    for commit in commits:
        # --- Reconstruct leave-one-out residual ---
        old_img   = render_aspen(commit.cx, commit.cy, commit.flux,
                                 commit.sig_maj, commit.sig_min, commit.pa, (H, W))
        psf_resp  = fftconvolve(old_img, psf, mode="same").astype(np.float32)
        loo_res   = post_residual + psf_resp          # add back this component

        peak_row  = int(round(commit.cy))
        peak_col  = int(round(commit.cx))
        peak_row  = max(0, min(H - 1, peak_row))
        peak_col  = max(0, min(W - 1, peak_col))

        # Keep greedy flux -- the LOO residual reflects remaining emission, not
        # this component's individual contribution, so reading amplitude from it
        # would inflate every component by ~45x.
        amplitude = commit.flux

        # --- MDN shape pass (no sigma_max clip) ---
        r0 = peak_row - _HALF;  r1 = r0 + _CUTOUT
        c0 = peak_col - _HALF;  c1 = c0 + _CUTOUT
        res_cut = _safe_crop(loo_res, r0, r1, c0, c1)
        psf_cut = _crop_psf_centred(psf, _CUTOUT)

        sigma_local = float(1.4826 * np.median(np.abs(res_cut)))
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = float(np.abs(loo_res).mean()) or 1e-6

        sig_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t = torch.tensor([0],           dtype=torch.long)   # config unused for shape
        cond  = make_cond(sig_t, cfg_t).to(device)

        img_t = torch.from_numpy(
            np.stack([res_cut, psf_cut], axis=0)[None]
        ).to(device)

        with torch.no_grad():
            params = model(img_t, cond)

        mode6d  = model.mode(params).squeeze(0).cpu().numpy()
        x_off   = float(mode6d[0])
        y_off   = float(mode6d[1])
        # No SIGMA_MAX_PX ceiling -- LOO residual gives cleaner view of true extent.
        sig_maj = float(np.clip(np.exp(mode6d[3]), BEAM_SIGMA_PX, _HALF))
        sig_min = float(np.clip(np.exp(mode6d[4]), BEAM_SIGMA_PX, sig_maj))
        pa      = float(mode6d[5])

        cx_new = float(peak_col) + x_off
        cy_new = float(peak_row) + y_off

        # --- Swap old component for new in model ---
        new_img = render_aspen(cx_new, cy_new, amplitude,
                               sig_maj, sig_min, pa, (H, W))
        new_model -= old_img
        new_model += new_img

    # Net positivity: individual components can be negative but the model cannot
    new_model = np.clip(new_model, 0.0, None)

    n_neg = int((new_model == 0).sum() - (model_update == 0).sum())
    print(f"  [refit] {len(commits)} components refit  "
          f"pixels zeroed by net-positivity clip: {max(n_neg, 0)}")

    return new_model


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
