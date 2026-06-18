"""PatchCorpusDataset — thin torch Dataset over casa_sim memmap stacks.

The stacks are written by casa_sim's patchify stage (M4).  Each stack
directory contains::

    dirty.npy       (N, 128, 128) float32  — SIGNED, sidelobes present
    sky.npy         (N, 128, 128) float32  — non-negative true sky
    psf.npy         (F, 128, 128) float32  — one PSF per field, peak=1
    field_id.npy    (N,)          int32    — maps patch i → field index
    config_idx.npy  (F,)          int32    — maps field f → VLA config (0=A…3=D)
    manifest.json   (metadata)

Per-item return contract
------------------------
    (residual, psf, cond, sky)

    residual : float32 (128, 128)  dirty patch, SIGNED, unclipped
    psf      : float32 (128, 128)  PSF for this patch's field, peak=1
    cond     : float32 (5,)        (log10_sigma_local, config_one_hot[4])
    sky      : float32 (128, 128)  true sky patch, non-negative

The Dataset is torch-native so DataLoader num_workers / pin_memory /
.to(device) work out of the box.  No CASA dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["PatchCorpusDataset"]


def _mad_sigma(arr: np.ndarray) -> float:
    """Robust noise estimate: 1.4826 * MAD (matches CutoutDataset)."""
    return 1.4826 * float(np.median(np.abs(arr)))


class PatchCorpusDataset(Dataset):
    """Torch Dataset over casa_sim memmap stacks.

    Parameters
    ----------
    stacks_dir:
        Directory produced by casa_sim patchify (contains dirty.npy etc.).
    """

    def __init__(self, stacks_dir: str | Path) -> None:
        stacks_dir = Path(stacks_dir)
        if not stacks_dir.is_dir():
            raise FileNotFoundError(f"stacks_dir not found: {stacks_dir}")

        self._dirty      = np.load(stacks_dir / "dirty.npy",      mmap_mode="r")
        self._sky        = np.load(stacks_dir / "sky.npy",        mmap_mode="r")
        self._psf        = np.load(stacks_dir / "psf.npy",        mmap_mode="r")
        self._field_id   = np.load(stacks_dir / "field_id.npy",   mmap_mode="r")
        self._config_idx = np.load(stacks_dir / "config_idx.npy", mmap_mode="r")

        manifest_path = stacks_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as fh:
                self._manifest = json.load(fh)
        else:
            self._manifest = {}

        N = self._dirty.shape[0]
        if self._sky.shape[0] != N or self._field_id.shape[0] != N:
            raise ValueError(
                f"Stack size mismatch: dirty={N}, "
                f"sky={self._sky.shape[0]}, field_id={self._field_id.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self._dirty.shape[0])

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return (residual, psf, cond, sky).

        residual : float32 (H, W)  signed dirty patch
        psf      : float32 (H, W)  PSF for this field, peak=1
        cond     : float32 (5,)    conditioning vector
        sky      : float32 (H, W)  true-sky patch, non-negative
        """
        residual_np = np.array(self._dirty[idx], dtype=np.float32)
        sky_np      = np.array(self._sky[idx],   dtype=np.float32)

        fid = int(self._field_id[idx])
        psf_np = np.array(self._psf[fid], dtype=np.float32)

        cfg = int(self._config_idx[fid])

        sigma_local = _mad_sigma(residual_np)
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = 1e-12

        from mad_clean.models.mdn_asp import make_cond  # noqa: PLC0415
        sigma_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t   = torch.tensor([cfg],         dtype=torch.long)
        cond    = make_cond(sigma_t, cfg_t).squeeze(0)  # (5,)

        return (
            torch.from_numpy(residual_np),
            torch.from_numpy(psf_np),
            cond,
            torch.from_numpy(sky_np),
        )
