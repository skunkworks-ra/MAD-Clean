"""Cutout dataset for MDN-Asp minor-cycle training.

Each sample is generated on-the-fly from a synthetic mixed-field sky scene
seeded deterministically from ``rng_seed + idx``.

Contract per sample
-------------------
- residual_cutout : float32 (H, W)   dirty image of the full field, cropped
- psf_cutout      : float32 (H, W)   PSF centred on its peak, cropped
- cond            : float32 (5,)     (sigma_local, config_one_hot)
- target          : float32 (6,)     (x_off, y_off, log_flux_std,
                                      log_sig_maj, log_sig_min, PA)
                                     offsets from cutout centre in pixels

Scene construction
------------------
A "centred source" of controlled morphology is placed at a random position
with sufficient margin for the cutout.  Distractor sources fill the rest of
the field.  The dirty image is ``PSF * full_sky + noise`` — no subtraction
of other sources — so the residual cutout contains realistic PSF contributions
from distractors as they would appear during an actual minor cycle.

The centred source morphology is sampled from ``morphology_balance``
(default 25 % each: point, blob, shell, filament).  The target always
refers to this centred source, regardless of what else falls in the cutout.

Coordinate convention
---------------------
Array indexing is (row, col).  Target uses (x=col-offset, y=row-offset)
from the cutout centre in pixels.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from mad_clean.data.extended_sky import (
    BEAM_SIGMA_PX,
    Target6D,
    assemble_mixed_field,
    render_filament,
    render_gaussian_blob,
    render_shell,
)
from mad_clean.data.psf_bank import PSFBank

__all__ = [
    "CutoutDataset",
    "LOG_FLUX_OFFSET",
    "LOG_FLUX_SCALE",
    "standardise_log_flux",
    "unstandardise_log_flux",
]

# D-config index per make_cond / mdn_asp.py (A=0, B=1, C=2, D=3)
_D_CONFIG_IDX = 3

LOG_FLUX_OFFSET = -5.76
LOG_FLUX_SCALE  = 3.45


def standardise_log_flux(log_flux: float) -> float:
    return (log_flux - LOG_FLUX_OFFSET) / LOG_FLUX_SCALE


def unstandardise_log_flux(z: float) -> float:
    return z * LOG_FLUX_SCALE + LOG_FLUX_OFFSET


class CutoutDataset(Dataset):
    """Dataset of (residual, psf, cond, target) quadruples for MDN-Asp training.

    Parameters
    ----------
    psf_bank:
        A :class:`PSFBank` instance with target_size == cutout_size.
    field_size:
        Side length of the full synthetic sky field (pixels).
    cutout_size:
        Side length of the square cutout (pixels). 128 per plan.
    sigma_noise:
        Gaussian noise standard deviation (Jy/beam).
    n_sources_per_field:
        Number of distractor sources per scene (int or (lo, hi) inclusive).
    extended_fraction:
        Per-distractor probability of being extended. Default 0.05.
    rng_seed:
        Base integer seed. Sample idx uses seed rng_seed + idx.
    length:
        Number of samples per epoch (__len__).
    config_idx:
        VLA configuration index (0=A, 1=B, 2=C, 3=D). Default 3.
    morphology_balance:
        Dict mapping morphology name to relative probability for the centred
        source. Keys must be a subset of {point, blob, shell, filament}.
        Default: equal weight across all four.
    snr_min:
        Minimum SNR for the centred source's PSF-convolved peak relative to
        sigma_noise. Sources below this are rescaled up. Default 5.0.
    return_sky:
        If True, append the true-sky cutout (same crop window as the
        residual, Jy/pixel, distractors included) as a fifth tensor.
        Used by the wavelet-NPE head, whose target is the sky image
        itself rather than the 6D parameter vector. Default False.
    compact_subtracted:
        Hybrid minor-cycle contract: assume a delta-function step has
        already (perfectly) subtracted all point sources before the
        learned solver sees the residual.  Point distractors are removed
        from the sky entirely (residual and target are extended-only),
        and the centred morphology must not be "point".  The wavelet
        codec cannot localise sub-beam structure (dropped w_1 plane), so
        training it on points would bake in systematic position errors
        that drift in the CLEAN loop. Default False.
    """

    _ALL_MORPHS = ("point", "blob", "shell", "filament")

    def __init__(
        self,
        psf_bank: PSFBank,
        field_size: int = 512,
        cutout_size: int = 128,
        sigma_noise: float = 1e-4,
        n_sources_per_field: int | tuple[int, int] = (5, 30),
        extended_fraction: float = 0.05,
        rng_seed: int = 42,
        length: int = 10_000,
        config_idx: int = _D_CONFIG_IDX,
        morphology_balance: dict[str, float] | None = None,
        snr_min: float = 5.0,
        return_sky: bool = False,
        compact_subtracted: bool = False,
    ) -> None:
        if len(psf_bank) == 0:
            raise ValueError("psf_bank is empty")
        if cutout_size < 1 or cutout_size > field_size:
            raise ValueError(
                f"cutout_size={cutout_size} must be in [1, field_size={field_size}]"
            )
        if morphology_balance is None:
            morphology_balance = {m: 0.25 for m in self._ALL_MORPHS}
        miss = set(morphology_balance) - set(self._ALL_MORPHS)
        if miss:
            raise ValueError(f"morphology_balance unknown keys: {sorted(miss)}")
        s = sum(morphology_balance.get(k, 0.0) for k in self._ALL_MORPHS)
        if s <= 0:
            raise ValueError("morphology_balance weights sum to <= 0")
        if compact_subtracted and morphology_balance.get("point", 0.0) > 0:
            raise ValueError(
                "compact_subtracted=True excludes point sources; remove "
                "'point' from morphology_balance"
            )
        self._morph_keys  = list(self._ALL_MORPHS)
        self._morph_probs = np.array(
            [morphology_balance.get(k, 0.0) / s for k in self._morph_keys],
            dtype=np.float64,
        )

        self._psf_bank    = psf_bank
        self._field_size  = int(field_size)
        self._cutout_size = int(cutout_size)
        self._sigma_noise = float(sigma_noise)
        self._n_sources   = n_sources_per_field
        self._ext_rate    = float(extended_fraction)
        self._rng_seed    = int(rng_seed)
        self._length      = int(length)
        self._config_idx  = int(config_idx)
        self._snr_min     = float(snr_min)
        self._return_sky  = bool(return_sky)
        self._compact_subtracted = bool(compact_subtracted)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return (residual_cutout, psf_cutout, cond, target).

        residual_cutout : float32 (H, W)
        psf_cutout      : float32 (H, W), peak = 1
        cond            : float32 (5,)
        target          : float32 (6,) = (x_off, y_off, log_flux_std,
                                          log_sig_maj, log_sig_min, PA)
        """
        rng = np.random.default_rng(self._rng_seed + idx)

        # 1. Pick centred-source morphology
        chosen_kind = self._morph_keys[
            int(rng.choice(len(self._morph_keys), p=self._morph_probs))
        ]

        # 2. Generate distractor scene
        if self._compact_subtracted:
            # Hybrid contract: point distractors removed entirely (the
            # delta-function step subtracted them before this solver runs).
            # Per-source rendering is requested so the extended-only sky
            # can be rebuilt; rng consumption matches the plain call, so
            # scenes stay deterministic per idx either way.
            _sky_all, tgts, per_source = assemble_mixed_field(
                size=self._field_size,
                n_sources=self._n_sources,
                extended_rate=self._ext_rate,
                rng=rng,
                return_per_source=True,
            )
            sky = np.zeros_like(_sky_all)
            for tgt, img in zip(tgts, per_source):
                if tgt.kind != "point":
                    sky = sky + img
        else:
            sky, _ = assemble_mixed_field(
                size=self._field_size,
                n_sources=self._n_sources,
                extended_rate=self._ext_rate,
                rng=rng,
            )

        # 3. Place centred source at a position with margin for the cutout
        margin = self._cutout_size // 2 + 4
        cx = float(rng.uniform(margin, self._field_size - margin))
        cy = float(rng.uniform(margin, self._field_size - margin))
        flux = float(np.exp(rng.uniform(math.log(1e-4), math.log(1e-1))))
        centred_img, centred_target = _render_kind(
            chosen_kind,
            size=self._field_size,
            cx=cx, cy=cy, flux_jy=flux, rng=rng,
        )

        # 4. PSF
        psf, _ = self._psf_bank.sample(rng)

        # Enforce minimum SNR: PSF-convolve the centred source alone and check
        # its peak against sigma_noise. If below snr_min, rescale flux so the
        # convolved peak just meets the threshold. This removes undetectable
        # training examples and matches the operational range of the minor cycle
        # (which never tries to fit sources below its stopping threshold).
        dirty_centred = fftconvolve(centred_img, psf, mode="same")
        convolved_peak = float(np.abs(dirty_centred).max())
        snr_min_threshold = self._snr_min * self._sigma_noise
        if convolved_peak < snr_min_threshold:
            scale = snr_min_threshold / max(convolved_peak, 1e-30)
            flux = flux * scale
            centred_img = centred_img * np.float32(scale)
            centred_target = centred_target._replace(
                log_flux=float(math.log(flux))
            )

        sky = sky + centred_img

        # 5. Dirty image: PSF * full_sky + noise (no subtraction of other sources)
        dirty_full = fftconvolve(sky, psf, mode="same").astype(np.float32)
        noise = rng.normal(0.0, self._sigma_noise, sky.shape).astype(np.float32)
        residual_full = dirty_full + noise

        # 6. Crop 128×128 around the centred source
        cr = int(round(centred_target.y))
        cc = int(round(centred_target.x))
        half = self._cutout_size // 2
        r0, r1 = cr - half, cr - half + self._cutout_size
        c0, c1 = cc - half, cc - half + self._cutout_size
        residual_cutout = _safe_crop(residual_full, r0, r1, c0, c1)

        psf_cutout = _crop_psf_centred(psf, self._cutout_size)

        # 7. Target in cutout-centred coordinates
        x_off = float(centred_target.x) - float(cc)
        y_off = float(centred_target.y) - float(cr)
        target_np = np.array([
            x_off,
            y_off,
            standardise_log_flux(centred_target.log_flux),
            centred_target.log_sig_maj,
            centred_target.log_sig_min,
            centred_target.pa,
        ], dtype=np.float32)

        # 8. Conditioning
        from mad_clean.models.mdn_asp import make_cond  # noqa: PLC0415
        sigma_local = _mad_sigma(residual_cutout)
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = max(self._sigma_noise, 1e-12)
        sigma_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t   = torch.tensor([self._config_idx], dtype=torch.long)
        cond    = make_cond(sigma_t, cfg_t).squeeze(0)  # (5,)

        residual_t = torch.from_numpy(residual_cutout)
        psf_t      = torch.from_numpy(psf_cutout)
        target_t   = torch.from_numpy(target_np)

        if self._return_sky:
            sky_cutout = _safe_crop(sky.astype(np.float32), r0, r1, c0, c1)
            return residual_t, psf_t, cond, target_t, torch.from_numpy(sky_cutout)

        return residual_t, psf_t, cond, target_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mad_sigma(arr: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(arr)))


def _render_kind(
    kind: str,
    *,
    size: int,
    cx: float,
    cy: float,
    flux_jy: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Target6D]:
    if kind == "point":
        img = np.zeros((size, size), dtype=np.float32)
        r = int(round(cy))
        c = int(round(cx))
        img[r, c] = np.float32(flux_jy)
        tgt = Target6D(
            x=float(c), y=float(r),
            log_flux=float(math.log(flux_jy)),
            log_sig_maj=float(math.log(BEAM_SIGMA_PX)),
            log_sig_min=float(math.log(BEAM_SIGMA_PX)),
            pa=0.0,
            kind="point",
        )
        return img, tgt
    if kind == "blob":
        return render_gaussian_blob(size=size, cx=cx, cy=cy, flux_jy=flux_jy, rng=rng)
    if kind == "shell":
        return render_shell(size=size, cx=cx, cy=cy, flux_jy=flux_jy, rng=rng)
    if kind == "filament":
        return render_filament(size=size, cx=cx, cy=cy, flux_jy=flux_jy, rng=rng)
    raise ValueError(f"Unknown morphology kind: {kind!r}")


def _safe_crop(
    arr: np.ndarray,
    r0: int, r1: int,
    c0: int, c1: int,
) -> np.ndarray:
    H, W = arr.shape
    out_h = r1 - r0
    out_w = c1 - c0
    out = np.zeros((out_h, out_w), dtype=arr.dtype)
    sr0 = max(0, r0); sr1 = min(H, r1)
    sc0 = max(0, c0); sc1 = min(W, c1)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    dr0 = sr0 - r0; dr1 = dr0 + (sr1 - sr0)
    dc0 = sc0 - c0; dc1 = dc0 + (sc1 - sc0)
    out[dr0:dr1, dc0:dc1] = arr[sr0:sr1, sc0:sc1]
    return out


def _crop_psf_centred(psf: np.ndarray, cutout_size: int) -> np.ndarray:
    if psf.shape == (cutout_size, cutout_size):
        return psf.copy()
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = cutout_size // 2
    return _safe_crop(psf, py - half, py - half + cutout_size,
                           px - half, px - half + cutout_size)
