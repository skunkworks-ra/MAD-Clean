"""FITSCropDataset — random 128×128 crops from 768×768 corpus FITS fields.

Loads all dirty + model + psf FITS from a directory at init, then returns
random crops at __getitem__.  Optionally caches everything on GPU so
DataLoader workers are not needed (set num_workers=0 when gpu_cache=True).

Return contract matches PatchCorpusDataset: (dirty_crop, psf_crop, cond, sky_crop)
  dirty_crop : float32 (H, W)  signed dirty patch
  psf_crop   : float32 (H, W)  PSF center-cropped to patch size, peak=1
  cond       : float32 (5,)    (log10_sigma_local, config_one_hot[4])
  sky_crop   : float32 (H, W)  true sky patch, non-negative
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["FITSCropDataset"]

# CDELT arcsec → VLA config index (A=0 B=1 C=2 D=3)
_CONFIG_BREAKS = [0.5, 1.5, 5.0]   # arcsec


def _cdelt_to_config(cdelt_deg: float) -> int:
    cell_arcsec = abs(cdelt_deg) * 3600.0
    for i, thresh in enumerate(_CONFIG_BREAKS):
        if cell_arcsec < thresh:
            return i
    return 3


def _mad_sigma(arr: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(arr)))


def _load_field(dirty_path: Path, patch_size: int):
    """Load one field.  Returns (dirty, sky, psf_crop, config_idx) as float32 arrays.

    dirty  : (H, W) float32  — center-cropped to match model spatial size
    sky    : (H, W) float32  — Stokes I averaged over freq, non-negative
    psf_crop: (patch_size, patch_size) float32  — peak-normalised
    """
    from astropy.io import fits as afits

    stem = dirty_path.name.replace("_dirty.fits", "")
    field_dir = dirty_path.parent

    dirty_hdu = afits.open(dirty_path)[0]
    hdr       = dirty_hdu.header
    dirty_full = np.array(dirty_hdu.data, dtype=np.float32)
    if dirty_full.ndim > 2:
        dirty_full = dirty_full.squeeze()
    # dirty_full: (H_d, W_d)

    # Model FITS: (n_freq, n_stokes, H_m, W_m) in FITS convention.
    # Take Stokes I (axis -3, index 0) averaged over frequency.
    model_data = afits.open(field_dir / f"{stem}_model.fits")[0].data.astype(np.float32)
    if model_data.ndim == 4:
        sky_arr = model_data[:, 0, :, :].mean(axis=0)   # (H_m, W_m)
    elif model_data.ndim == 3:
        sky_arr = model_data[0]                          # (H_m, W_m)
    else:
        sky_arr = model_data
    sky_arr = np.clip(sky_arr, 0.0, None)
    H_m, W_m = sky_arr.shape

    # Center-crop dirty to match model spatial size
    H_d, W_d = dirty_full.shape
    r0 = (H_d - H_m) // 2
    c0 = (W_d - W_m) // 2
    dirty_arr = dirty_full[r0: r0 + H_m, c0: c0 + W_m]

    # PSF: center-crop to patch_size using CRPIX (1-indexed → 0-indexed)
    psf_full = afits.open(field_dir / f"{stem}_psf.fits")[0].data.astype(np.float32)
    if psf_full.ndim > 2:
        psf_full = psf_full.squeeze()
    cy = int(round(float(hdr.get("CRPIX2", psf_full.shape[0] / 2 + 0.5)))) - 1
    cx = int(round(float(hdr.get("CRPIX1", psf_full.shape[1] / 2 + 0.5)))) - 1
    half = patch_size // 2
    psf_crop = psf_full[cy - half: cy + half, cx - half: cx + half]
    peak = float(psf_crop.max())
    if peak > 0:
        psf_crop = psf_crop / peak

    config_idx = _cdelt_to_config(float(hdr.get("CDELT1", -2.4e-4)))

    return dirty_arr, sky_arr, psf_crop, config_idx


class FITSCropDataset(Dataset):
    """Random-crop dataset over real FITS corpus fields.

    Parameters
    ----------
    fits_dir : directory containing corpus_field_NNNN_dirty.fits etc.
    patch_size : spatial size of returned patches (default 128).
    length : virtual epoch length (default 100_000).
    gpu_cache : if True, move all tensors to `device` at init.
                Use num_workers=0 with DataLoader in this mode.
    device : torch device for gpu_cache (default cuda if available).
    """

    def __init__(
        self,
        fits_dir: str | Path,
        patch_size: int = 128,
        length: int = 100_000,
        gpu_cache: bool = False,
        device: torch.device | None = None,
    ) -> None:
        fits_dir = Path(fits_dir)
        dirty_paths = sorted(fits_dir.glob("*_dirty.fits"))
        if not dirty_paths:
            raise FileNotFoundError(f"No *_dirty.fits found in {fits_dir}")

        print(f"[FITSCropDataset] Loading {len(dirty_paths)} fields from {fits_dir} ...")
        dirties, skies, psfs, configs = [], [], [], []
        for p in dirty_paths:
            d, s, psf, cfg = _load_field(p, patch_size)
            dirties.append(torch.from_numpy(d))
            skies.append(torch.from_numpy(s))
            psfs.append(torch.from_numpy(psf))
            configs.append(cfg)

        self._dirties = dirties    # list of (H, W) tensors; H,W may vary per field
        self._skies   = skies
        self._psfs    = torch.stack(psfs)       # (F, patch_size, patch_size)
        self._configs = configs
        self._patch   = patch_size
        self._length  = length
        self._n       = len(dirty_paths)

        if gpu_cache:
            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._dirties = [t.to(dev) for t in self._dirties]
            self._skies   = [t.to(dev) for t in self._skies]
            self._psfs    = self._psfs.to(dev)
            print(f"[FITSCropDataset] Cached on {dev}.")

        print(f"[FITSCropDataset] Ready.  {self._n} fields, virtual length {length}.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        from mad_clean.models.mdn_asp import make_cond

        fid = idx % self._n
        dirty = self._dirties[fid]
        sky   = self._skies[fid]
        psf   = self._psfs[fid]
        cfg   = self._configs[fid]

        H, W = dirty.shape
        p = self._patch
        r0 = int(torch.randint(0, H - p + 1, (1,)).item())
        c0 = int(torch.randint(0, W - p + 1, (1,)).item())

        dirty_crop = dirty[r0: r0 + p, c0: c0 + p]
        sky_crop   = sky[r0: r0 + p, c0: c0 + p]

        sigma_local = _mad_sigma(dirty_crop.cpu().numpy())
        if not math.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = 1e-12

        sigma_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t   = torch.tensor([cfg], dtype=torch.long)
        cond    = make_cond(sigma_t, cfg_t).squeeze(0)

        return dirty_crop, psf, cond, sky_crop
