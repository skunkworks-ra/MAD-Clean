"""
mad_clean.data.cutout_dataset
==============================
PyTorch Dataset that emits (residual_cutout, psf_cutout, conditioning, target)
quadruples for training the MDN-Asp minor-cycle network.

Coordinate convention
---------------------
- All 2D arrays are indexed as (row, col) with origin at the top-left pixel.
- The 6D target stores (x, y, ...) where x = col-offset and y = row-offset
  from the cutout centre in pixels.  A source perfectly centred in the cutout
  has target x=0, y=0; a source one pixel to the right (higher col index) has
  x=1, y=0.
- This matches the (x=col, y=row) convention in extended_sky.Target6D and
  point_sky catalogue tuples.

Residual cutout construction
------------------------------
The residual cutout simulates a late-minor-cycle residual where only one
target Aspen remains unsubtracted.  For a chosen target at position (r, c):

    residual = dirty_full + noise − psf_response_of_all_other_sources
             = psf ⊛ sky_full + noise − (psf ⊛ sky_without_chosen)
             = psf ⊛ sky_chosen + noise

where ⊛ denotes full-field convolution.  In practice, since convolution is
linear, we convolve only the chosen source's image and add noise, which is
numerically identical.

PSF convolution
---------------
scipy.signal.fftconvolve(..., mode='same') is used throughout.  The dirty
image of each source component is the convolution of its pixel map with the
PSF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from mad_clean.data.extended_sky import assemble_mixed_field, Target6D
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

# log_flux standardisation. Default flux range is (1e-4, 1e-1) Jy, so
# log_flux spans roughly [-9.21, -2.30]. We standardise to a near-unit-range
# variable so the MDN does not have to learn to bias its output mean by ~6
# nats. OFFSET is the midpoint, SCALE is the half-range.
LOG_FLUX_OFFSET = -5.76
LOG_FLUX_SCALE  = 3.45


def standardise_log_flux(log_flux: float) -> float:
    """Map raw log_flux (nats) to standardised target space."""
    return (log_flux - LOG_FLUX_OFFSET) / LOG_FLUX_SCALE


def unstandardise_log_flux(z: float) -> float:
    """Invert standardise_log_flux."""
    return z * LOG_FLUX_SCALE + LOG_FLUX_OFFSET


class CutoutDataset(Dataset):
    """Dataset of (residual_cutout, psf_cutout, conditioning, target) quadruples.

    Each sample is generated on-the-fly from a synthetic mixed-field sky scene.
    The RNG is seeded deterministically from ``rng_seed + idx`` so the dataset
    is reproducible and individual samples are independent.

    Parameters
    ----------
    psf_bank:
        A :class:`PSFBank` instance.  The bank should have been constructed
        with ``target_size=cutout_size`` so PSF arrays are already 128×128.
    field_size:
        Side length of the full synthetic sky field (pixels).  Default 512.
    cutout_size:
        Side length of the square residual / PSF cutout (pixels).  Default 128.
    sigma_noise:
        Standard deviation of Gaussian noise added to the dirty image (Jy/beam).
    n_sources_per_field:
        Number of sources in each scene.  Passed as ``n_sources`` to
        ``assemble_mixed_field``.  Can be an int or (min, max) tuple.
    extended_fraction:
        Per-source probability of being extended.  Default 0.05.
    rng_seed:
        Base integer seed.  Sample ``idx`` uses seed ``rng_seed + idx``.
    length:
        Number of samples per epoch (``__len__``).  ``__getitem__`` accepts
        any non-negative idx; it does not wrap around internally — the caller
        is responsible for wrapping if ``idx >= length`` would be hit.
    config_idx:
        VLA configuration index (0=A, 1=B, 2=C, 3=D).  Default 3 (D-config).
    """

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
    ) -> None:
        if len(psf_bank) == 0:
            raise ValueError("psf_bank is empty")
        if cutout_size < 1 or cutout_size > field_size:
            raise ValueError(
                f"cutout_size={cutout_size} must be in [1, field_size={field_size}]"
            )
        self._psf_bank = psf_bank
        self._field_size = int(field_size)
        self._cutout_size = int(cutout_size)
        self._sigma_noise = float(sigma_noise)
        self._n_sources = n_sources_per_field
        self._ext_rate = float(extended_fraction)
        self._rng_seed = int(rng_seed)
        self._length = int(length)
        self._config_idx = int(config_idx)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (residual_cutout, psf_cutout, conditioning, target).

        residual_cutout : float32 (cutout_size, cutout_size)
        psf_cutout      : float32 (cutout_size, cutout_size), peak = 1
        conditioning    : float32 (5,)
        target          : float32 (6,) = (x, y, log_flux, log_sig_maj, log_sig_min, PA)
                          where (x, y) are offsets from the cutout centre in pixels.
        """
        rng = np.random.default_rng(self._rng_seed + idx)

        # 1. Generate scene — request per-source rendered images so the
        #    "chosen source contribution" is the AS-RENDERED image (shell,
        #    filament, blob), NOT the Gaussian fit of it. The supervised
        #    signal must contain real morphology; the network learns to
        #    regress 6D Gaussian-fit parameters from real shapes.
        sky, targets, per_source = assemble_mixed_field(
            size=self._field_size,
            n_sources=self._n_sources,
            extended_rate=self._ext_rate,
            rng=rng,
            return_per_source=True,
        )

        # Retry until we have at least one source
        if len(targets) == 0:
            rng2 = np.random.default_rng(self._rng_seed + idx + 1_000_000)
            sky, targets, per_source = assemble_mixed_field(
                size=self._field_size,
                n_sources=(2, 10),
                extended_rate=self._ext_rate,
                rng=rng2,
                return_per_source=True,
            )

        # 2. Pick a random PSF from the bank
        psf, _ = self._psf_bank.sample(rng)
        # psf is already peak-normalised (PSFBank does that at construction)

        # 3. Choose one target Aspen
        t_idx = int(rng.integers(0, len(targets)))
        chosen: Target6D = targets[t_idx]
        sky_chosen = per_source[t_idx]  # AS-RENDERED contribution

        # 4. Build the residual:
        #    residual = dirty_full + noise − PSF * (all other sources)
        #             = PSF * sky_chosen + noise        (by linearity)
        dirty_chosen = fftconvolve(sky_chosen, psf, mode="same").astype(np.float32)
        noise = rng.normal(0.0, self._sigma_noise, sky.shape).astype(np.float32)
        residual_full = dirty_chosen + noise

        # 5. Crop 128×128 around the chosen target centre
        cr = int(round(chosen.y))   # row = y
        cc = int(round(chosen.x))   # col = x
        half = self._cutout_size // 2

        r0 = cr - half
        r1 = r0 + self._cutout_size
        c0 = cc - half
        c1 = c0 + self._cutout_size

        residual_cutout = _safe_crop(residual_full, r0, r1, c0, c1)

        # 6. Crop / centre PSF to cutout_size
        #    PSFBank already returns arrays of target_size; if target_size == cutout_size
        #    the PSF is already the right shape and peak is at half, half.
        #    If the bank was built with a larger target_size, we re-crop here.
        psf_cutout = _crop_psf_centred(psf, self._cutout_size)

        # 7. Compute target in cutout-centred coordinates
        #    The chosen target is at absolute (row=chosen.y, col=chosen.x).
        #    Cutout centre is at absolute (cr, cc) = (round(chosen.y), round(chosen.x)).
        #    Offset in pixels: x_off = chosen.x - cc, y_off = chosen.y - cr.
        x_off = float(chosen.x) - float(cc)
        y_off = float(chosen.y) - float(cr)

        # log_flux is standardised; unstandardise downstream when reporting
        # human-readable errors. The MDN trains and predicts in this space.
        target_np = np.array([
            x_off,
            y_off,
            standardise_log_flux(chosen.log_flux),
            chosen.log_sig_maj,
            chosen.log_sig_min,
            chosen.pa,
        ], dtype=np.float32)

        # 8. Build conditioning via make_cond helper (deferred import to avoid
        #    circular init ordering in conftest when data sub-package loads before models)
        from mad_clean.models.mdn_asp import make_cond  # noqa: PLC0415
        sigma_t = torch.tensor([self._sigma_noise], dtype=torch.float32)
        cfg_t   = torch.tensor([self._config_idx], dtype=torch.long)
        cond    = make_cond(sigma_t, cfg_t).squeeze(0)  # (5,)

        # 9. Assemble tensors
        residual_t = torch.from_numpy(residual_cutout)
        psf_t      = torch.from_numpy(psf_cutout)
        target_t   = torch.from_numpy(target_np)

        return residual_t, psf_t, cond, target_t


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_crop(
    arr: np.ndarray,
    r0: int, r1: int,
    c0: int, c1: int,
) -> np.ndarray:
    """Crop arr[r0:r1, c0:c1], zero-padding outside array bounds."""
    H, W = arr.shape
    out_h = r1 - r0
    out_w = c1 - c0
    out = np.zeros((out_h, out_w), dtype=arr.dtype)

    sr0 = max(0, r0)
    sr1 = min(H, r1)
    sc0 = max(0, c0)
    sc1 = min(W, c1)

    if sr1 <= sr0 or sc1 <= sc0:
        return out  # fully outside

    dr0 = sr0 - r0
    dr1 = dr0 + (sr1 - sr0)
    dc0 = sc0 - c0
    dc1 = dc0 + (sc1 - sc0)

    out[dr0:dr1, dc0:dc1] = arr[sr0:sr1, sc0:sc1]
    return out


def _crop_psf_centred(psf: np.ndarray, cutout_size: int) -> np.ndarray:
    """Return a (cutout_size, cutout_size) view of psf centred on its peak.

    If psf is already (cutout_size, cutout_size), return it unchanged.
    """
    if psf.shape == (cutout_size, cutout_size):
        return psf.copy()

    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = cutout_size // 2
    return _safe_crop(psf, py - half, py - half + cutout_size,
                           px - half, px - half + cutout_size)
