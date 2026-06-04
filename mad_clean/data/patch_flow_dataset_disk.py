"""Disk-based dataset for PatchFlow training.

Loads pre-generated .pt shards produced by generate_patch_flow_data.py.
Designed for fast training with multiple DataLoader workers.

Each shard contains:
    dirty : (N, 1, 128, 128) float32
    clean : (N, 1, 128, 128) float32
    psf   : (N, 1, 128, 128) float32
    sigma : (N,)             float32
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = ["PatchFlowDatasetDisk"]


class PatchFlowDatasetDisk(Dataset):
    """Dataset backed by pre-generated .pt shards.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing shard_XXXX.pt files.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._shards = sorted(Path(data_dir).glob("shard_*.pt"))
        if not self._shards:
            raise FileNotFoundError(f"No shard_*.pt files found in {data_dir}")

        # Build index: (shard_idx, sample_idx_within_shard)
        self._index: list[tuple[int, int]] = []
        self._cache: dict[int, dict] = {}
        for si, path in enumerate(self._shards):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            n = shard["dirty"].shape[0]
            self._index.extend((si, i) for i in range(n))
            self._cache[si] = shard

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si, li = self._index[idx]
        shard = self._cache[si]
        return (
            shard["dirty"][li],   # (1, H, W)
            shard["psf"][li],     # (1, H, W)
            shard["sigma"][li],   # ()
            shard["clean"][li],   # (1, H, W)
        )
