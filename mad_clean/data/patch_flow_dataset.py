"""Dataset for patch-level conditional flow matching.

Each sample is (dirty_cutout, psf_cutout, sigma_local, clean_sky_cutout).
The clean_sky_cutout is the target for CFM training -- the deconvolved sky
patch for the centred source only (distractors excluded from the target).

Reuses CutoutDataset scene construction; replaces the 6D Gaussian target
with a pixel-level clean image patch.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from mad_clean.data.cutout_dataset import (
    _crop_psf_centred,
    _mad_sigma,
    _render_kind,
    _safe_crop,
)
from mad_clean.data.extended_sky import assemble_mixed_field
from mad_clean.data.psf_bank import PSFBank

__all__ = ["PatchFlowDataset"]


class PatchFlowDataset(Dataset):
    """Dataset of (dirty, psf, sigma, clean) tuples for CFM training.

    Parameters
    ----------
    psf_bank : PSFBank
    field_size : int
    cutout_size : int
        Must match the PatchFlow model input size (128).
    sigma_noise : float
        Gaussian noise std (Jy/beam).
    n_sources_per_field : int | tuple[int, int]
        Number of distractor sources per scene.
    morphology_balance : dict[str, float] | None
        Relative weights for centred-source morphology.
        Default: 50% point, ~17% each blob/shell/filament.
    snr_min : float
        Minimum SNR for the centred source. Default 5.0.
    rng_seed : int
    length : int
    """

    _ALL_MORPHS = ("point", "blob", "shell", "filament")
    _DEFAULT_BALANCE = {"point": 0.25, "blob": 0.25, "shell": 0.25, "filament": 0.25}

    def __init__(
        self,
        psf_bank: PSFBank,
        field_size: int = 512,
        cutout_size: int = 128,
        sigma_noise: float = 1e-4,
        n_sources_per_field: int | tuple[int, int] = (5, 30),
        extended_fraction: float = 0.25,
        morphology_balance: dict[str, float] | None = None,
        snr_min: float = 5.0,
        rng_seed: int = 42,
        length: int = 10_000,
    ) -> None:
        if morphology_balance is None:
            morphology_balance = self._DEFAULT_BALANCE
        s = sum(morphology_balance.get(k, 0.0) for k in self._ALL_MORPHS)
        if s <= 0:
            raise ValueError("morphology_balance weights sum to <= 0")
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
        self._snr_min     = float(snr_min)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        """Return (dirty, psf, sigma, clean) all float32 tensors.

        dirty : (1, H, W)  dirty cutout (PSF * full sky + noise)
        psf   : (1, H, W)  PSF cutout, peak = 1
        sigma : ()         scalar sigma_local estimate from dirty cutout
        clean : (1, H, W)  clean sky patch (centred source only, Jy/pixel)
        """
        rng = np.random.default_rng(self._rng_seed + idx)

        # Centred source morphology
        chosen_kind = self._morph_keys[
            int(rng.choice(len(self._morph_keys), p=self._morph_probs))
        ]

        # Distractor scene
        sky, _ = assemble_mixed_field(
            size=self._field_size,
            n_sources=self._n_sources,
            extended_rate=self._ext_rate,
            rng=rng,
        )

        # Centred source
        margin = self._cutout_size // 2 + 4
        cx = float(rng.uniform(margin, self._field_size - margin))
        cy = float(rng.uniform(margin, self._field_size - margin))
        flux = float(np.exp(rng.uniform(math.log(1e-4), math.log(1e-1))))
        centred_img, centred_target = _render_kind(
            chosen_kind, size=self._field_size,
            cx=cx, cy=cy, flux_jy=flux, rng=rng,
        )

        psf, _ = self._psf_bank.sample(rng)

        # SNR floor
        dirty_centred  = fftconvolve(centred_img, psf, mode="same")
        convolved_peak = float(np.abs(dirty_centred).max())
        snr_threshold  = self._snr_min * self._sigma_noise
        if convolved_peak < snr_threshold:
            scale = snr_threshold / max(convolved_peak, 1e-30)
            flux         *= scale
            centred_img   = centred_img * np.float32(scale)

        sky = sky + centred_img

        # Dirty image: PSF * full_sky + noise (same as CutoutDataset)
        dirty_full = fftconvolve(sky, psf, mode="same").astype(np.float32)
        noise      = rng.normal(0.0, self._sigma_noise, sky.shape).astype(np.float32)
        dirty_full = dirty_full + noise

        # Crop around centred source
        cr   = int(round(centred_target.y))
        cc   = int(round(centred_target.x))
        half = self._cutout_size // 2
        r0, r1 = cr - half, cr - half + self._cutout_size
        c0, c1 = cc - half, cc - half + self._cutout_size

        dirty_cut = _safe_crop(dirty_full, r0, r1, c0, c1)       # (H, W)
        clean_cut = _safe_crop(centred_img, r0, r1, c0, c1)      # (H, W) centred source only
        psf_cut   = _crop_psf_centred(psf, self._cutout_size)     # (H, W)

        sigma_local = _mad_sigma(dirty_cut)
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = self._sigma_noise

        beam_area = float(psf_cut.sum())
        if beam_area > 0:
            dirty_cut = dirty_cut / beam_area

        dirty_t = torch.from_numpy(dirty_cut).unsqueeze(0)   # (1, H, W)
        psf_t   = torch.from_numpy(psf_cut).unsqueeze(0)     # (1, H, W)
        clean_t = torch.from_numpy(clean_cut).unsqueeze(0)   # (1, H, W)
        sigma_t = torch.tensor(sigma_local, dtype=torch.float32)

        return dirty_t, psf_t, sigma_t, clean_t
