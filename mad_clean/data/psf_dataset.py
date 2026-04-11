"""
mad_clean.data.psf_dataset
===========================
PyTorch Dataset for PSF-conditioned deconvolution training.

Reads from the directory produced by scripts/generate_psf_data.py:
    crumb_data/psf_pairs/
        index.npz          — split metadata
        psf_0000.npz       — all images for PSF 0
        psf_0001.npz       — all images for PSF 1
        ...

Each sample is a (dirty, psf, clean) triple as float32 tensors.

Usage
-----
    from mad_clean.data.psf_dataset import PSFPairsDataset
    from torch.utils.data import DataLoader

    train_ds = PSFPairsDataset("crumb_data/psf_pairs", train=True)
    loader   = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

    for dirty, psf, clean in loader:
        # dirty: (B, H, W)  psf: (B, H, W)  clean: (B, H, W)
        ...
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["PSFPairsDataset"]


class PSFPairsDataset(Dataset):
    """
    Parameters
    ----------
    root    : path to psf_pairs/ directory
    train   : if True, use training PSFs; if False, use held-out test PSFs
    """

    def __init__(self, root: str | Path, train: bool = True):
        self.root = Path(root)

        index      = np.load(self.root / "index.npz")
        train_mask = index["train_mask"]           # (N_psf,) bool
        n_img      = int(index["n_img"])

        # PSF indices for this split
        self._psf_indices = np.where(train_mask if train else ~train_mask)[0]
        self._n_img       = n_img
        self._len         = len(self._psf_indices) * n_img

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psf_slot  = idx // self._n_img
        img_slot  = idx  % self._n_img
        psf_idx   = int(self._psf_indices[psf_slot])

        data  = self._load_psf_file(psf_idx)
        dirty = torch.from_numpy(data["dirty"][img_slot])   # (H, W)
        clean = torch.from_numpy(data["clean"][img_slot])   # (H, W)
        psf   = torch.from_numpy(data["psf"])               # (H, W)

        return dirty, psf, clean

    @lru_cache(maxsize=8)
    def _load_psf_file(self, psf_idx: int) -> dict:
        """Load and cache a PSF npz file. LRU cache keeps 8 files open."""
        path = self.root / f"psf_{psf_idx:04d}.npz"
        data = np.load(path)
        return {
            "dirty": data["dirty"],   # (N_img, H, W) float32
            "clean": data["clean"],   # (N_img, H, W) float32
            "psf"  : data["psf"],     # (H, W) float32
        }
