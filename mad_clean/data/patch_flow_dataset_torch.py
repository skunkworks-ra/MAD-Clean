"""GPU-side dataset for PatchFlow CFM training.

Identical contract to PatchFlowDataset but runs entirely on a specified
torch device. Scene generation, PSF convolution, and normalisation all
happen on GPU when device='cuda'. Use with num_workers=0.

Each sample is (dirty, psf, sigma, clean) where dirty and clean are
peak-normalised by the dirty peak so that x1 has amplitude ~1 for CFM.
The peak value is NOT returned -- the training script does not need it.
Inference rescaling is handled by the minor cycle caller.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from mad_clean.data.extended_sky_torch import (
    BEAM_SIGMA_PX,
    SIGMA_MAX_PX,
    assemble_mixed_field_t,
    fft_convolve_t,
    render_gaussian_blob_t,
    render_shell_t,
    render_filament_t,
)
from mad_clean.data.psf_bank import PSFBank

__all__ = ["PatchFlowDatasetTorch"]


class PatchFlowDatasetTorch(Dataset):
    """GPU dataset of (dirty, psf, sigma, clean) tuples for CFM training.

    Parameters
    ----------
    psf_bank : PSFBank
    device : torch.device | str
    field_size : int
    cutout_size : int
    sigma_noise : float
    n_sources_per_field : int | tuple[int, int]
    extended_fraction : float
    snr_min : float
    rng_seed : int
    length : int
    """

    _MORPHS = ("point", "blob", "shell", "filament")

    def __init__(
        self,
        psf_bank: PSFBank,
        device: torch.device | str = "cuda",
        field_size: int = 512,
        cutout_size: int = 128,
        sigma_noise: float = 1e-4,
        n_sources_per_field: int | tuple[int, int] = (5, 30),
        extended_fraction: float = 0.25,
        snr_min: float = 5.0,
        rng_seed: int = 42,
        length: int = 10_000,
    ) -> None:
        self._psf_bank    = psf_bank
        self._device      = torch.device(device)
        self._field_size  = int(field_size)
        self._cutout_size = int(cutout_size)
        self._sigma_noise = float(sigma_noise)
        self._n_sources   = n_sources_per_field
        self._ext_rate    = float(extended_fraction)
        self._snr_min     = float(snr_min)
        self._rng_seed    = int(rng_seed)
        self._length      = int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        """Return (dirty, psf, sigma, clean) float32 tensors on self._device.

        dirty : (1, H, W)  peak-normalised dirty cutout
        psf   : (1, H, W)  PSF cutout, peak = 1
        sigma : ()         normalised noise estimate (sigma / dirty_peak)
        clean : (1, H, W)  peak-normalised clean sky patch
        """
        device = self._device
        rng_np = np.random.default_rng(self._rng_seed + idx)
        rng_t  = torch.Generator(device=device)
        rng_t.manual_seed(int(self._rng_seed + idx))

        H = W = self._field_size
        S = self._cutout_size
        half = S // 2

        # --- PSF (load from bank, move to device) ---
        psf_np, _ = self._psf_bank.sample(rng_np)
        psf_full  = torch.from_numpy(psf_np).float().to(device)   # (H_psf, W_psf)

        # Crop PSF to cutout size, centred on peak
        py, px = (psf_full == psf_full.max()).nonzero(as_tuple=False)[0]
        py, px = int(py), int(px)
        pr0, pr1 = py - half, py - half + S
        pc0, pc1 = px - half, px - half + S
        psf_cut = _safe_crop_t(psf_full, pr0, pr1, pc0, pc1, device)  # (S, S)

        # --- Distractor scene ---
        n_src = int(rng_np.integers(*self._n_sources) if isinstance(self._n_sources, tuple)
                    else self._n_sources)
        sky = assemble_mixed_field_t(
            size=H,
            n_sources=n_src,
            flux_range_jy=(1e-4, 1e-1),
            extended_rate=self._ext_rate,
            edge_margin=16,
            rng=rng_t,
            device=device,
        )

        # --- Centred source ---
        margin = half + 4
        cx = float(rng_np.uniform(margin, W - margin))
        cy = float(rng_np.uniform(margin, H - margin))
        flux = float(np.exp(rng_np.uniform(math.log(1e-4), math.log(1e-1))))

        morph = self._MORPHS[int(rng_np.integers(0, len(self._MORPHS)))]
        centred_img = _render_kind_t(morph, H, cx, cy, flux, rng_np, device)

        # SNR floor
        dirty_centred  = fft_convolve_t(centred_img, psf_full)
        convolved_peak = float(dirty_centred.abs().max())
        snr_threshold  = self._snr_min * self._sigma_noise
        if convolved_peak < snr_threshold:
            scale = snr_threshold / max(convolved_peak, 1e-30)
            flux          *= scale
            centred_img    = centred_img * scale

        sky = sky + centred_img

        # --- Dirty image ---
        dirty_full = fft_convolve_t(sky, psf_full)
        noise = torch.randn(H, W, device=device) * self._sigma_noise
        dirty_full = dirty_full + noise

        # --- Crop cutout around centred source centre ---
        cr = int(round(cy))
        cc = int(round(cx))
        r0, r1 = cr - half, cr - half + S
        c0, c1 = cc - half, cc - half + S

        dirty_cut = _safe_crop_t(dirty_full, r0, r1, c0, c1, device)   # (S, S)
        clean_cut = _safe_crop_t(centred_img, r0, r1, c0, c1, device)  # (S, S)

        # --- Beam area normalisation (dirty Jy/beam -> Jy/pixel scale) ---
        beam_area = float(psf_cut.sum())
        if beam_area > 0:
            dirty_cut = dirty_cut / beam_area

        # --- Peak normalisation (both dirty and clean scaled by dirty peak) ---
        dirty_peak = float(dirty_cut.abs().max())
        if dirty_peak > 0:
            dirty_cut = dirty_cut / dirty_peak
            clean_cut = clean_cut / dirty_peak

        # --- Noise estimate in normalised units ---
        sigma_local = float(1.4826 * dirty_cut.flatten().median().abs())
        if not math.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = self._sigma_noise / max(dirty_peak, 1e-30)

        return (
            dirty_cut.unsqueeze(0),                                    # (1, S, S)
            psf_cut.unsqueeze(0),                                      # (1, S, S)
            torch.tensor(sigma_local, dtype=torch.float32, device=device),
            clean_cut.unsqueeze(0),                                    # (1, S, S)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_crop_t(
    arr: torch.Tensor,
    r0: int, r1: int,
    c0: int, c1: int,
    device: torch.device,
) -> torch.Tensor:
    H, W = arr.shape
    out = torch.zeros(r1 - r0, c1 - c0, dtype=torch.float32, device=device)
    sr0 = max(0, r0); sr1 = min(H, r1)
    sc0 = max(0, c0); sc1 = min(W, c1)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
    return out


def _render_kind_t(
    kind: str,
    size: int,
    cx: float, cy: float,
    flux_jy: float,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    if kind == "point":
        img = torch.zeros(size, size, dtype=torch.float32, device=device)
        r = max(0, min(size - 1, int(round(cy))))
        c = max(0, min(size - 1, int(round(cx))))
        img[r, c] = flux_jy
        return img
    elif kind == "blob":
        sig_maj = float(rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX))
        sig_min = float(rng.uniform(BEAM_SIGMA_PX, sig_maj))
        pa      = float(rng.uniform(0.0, math.pi))
        return render_gaussian_blob_t(size, cx, cy, flux_jy, sig_maj, sig_min, pa, device)
    elif kind == "shell":
        radius    = float(rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX))
        thickness = float(rng.uniform(BEAM_SIGMA_PX * 0.5, max(BEAM_SIGMA_PX, radius * 0.5)))
        return render_shell_t(size, cx, cy, flux_jy, radius, thickness, device)
    else:  # filament
        length = float(rng.uniform(2 * BEAM_SIGMA_PX, 2 * SIGMA_MAX_PX))
        width  = float(rng.uniform(BEAM_SIGMA_PX, SIGMA_MAX_PX))
        pa     = float(rng.uniform(0.0, math.pi))
        return render_filament_t(size, cx, cy, flux_jy, length, width, pa, device)
