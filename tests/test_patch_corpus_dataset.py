"""tests/test_patch_corpus_dataset.py

Unit tests for PatchCorpusDataset.

All fixtures are synthetic (no CASA, no casa_sim dependency).  An optional
opportunistic check uses /tmp/m4_smoke_stacks if it is present.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_stacks_dir(
    tmp_path: Path,
    n_patches: int = 6,
    n_fields: int = 2,
    patch_size: int = 128,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write a minimal valid stacks directory to tmp_path."""
    if rng is None:
        rng = np.random.default_rng(0)

    # dirty: signed — include some negatives
    dirty = rng.standard_normal((n_patches, patch_size, patch_size)).astype(np.float32)
    # sky: non-negative
    sky = np.abs(rng.standard_normal((n_patches, patch_size, patch_size))).astype(np.float32)
    # psf: one per field, peak=1
    psf_raw = np.abs(rng.standard_normal((n_fields, patch_size, patch_size))).astype(np.float32)
    psf = psf_raw / psf_raw.max(axis=(1, 2), keepdims=True)
    # field_id: distribute patches across fields
    field_id = np.array([i % n_fields for i in range(n_patches)], dtype=np.int32)
    # config_idx: one per field, values in {0,1,2,3}
    config_idx = np.array([i % 4 for i in range(n_fields)], dtype=np.int32)

    np.save(tmp_path / "dirty.npy",      dirty)
    np.save(tmp_path / "sky.npy",        sky)
    np.save(tmp_path / "psf.npy",        psf)
    np.save(tmp_path / "field_id.npy",   field_id)
    np.save(tmp_path / "config_idx.npy", config_idx)

    manifest = {
        "n_fields": n_fields,
        "n_patches": n_patches,
        "patch_size": patch_size,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    return tmp_path


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestPatchCorpusDataset:

    def test_len(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=6)
        ds = PatchCorpusDataset(tmp_path)
        assert len(ds) == 6

    def test_item_is_4tuple(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=4, n_fields=2)
        ds = PatchCorpusDataset(tmp_path)
        item = ds[0]
        assert len(item) == 4, f"Expected 4-tuple, got {len(item)}-tuple"

    def test_shapes(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=4, n_fields=2, patch_size=128)
        ds = PatchCorpusDataset(tmp_path)
        residual, psf, cond, sky = ds[0]
        assert residual.shape == (128, 128), f"residual shape {residual.shape}"
        assert psf.shape      == (128, 128), f"psf shape {psf.shape}"
        assert cond.shape     == (5,),        f"cond shape {cond.shape}"
        assert sky.shape      == (128, 128),  f"sky shape {sky.shape}"

    def test_dtypes_float32(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=4, n_fields=2)
        ds = PatchCorpusDataset(tmp_path)
        residual, psf, cond, sky = ds[0]
        assert residual.dtype == torch.float32
        assert psf.dtype      == torch.float32
        assert cond.dtype     == torch.float32
        assert sky.dtype      == torch.float32

    def test_dirty_negatives_passthrough(self, tmp_path):
        """dirty contains negatives; they must not be clipped."""
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        rng = np.random.default_rng(42)
        _make_stacks_dir(tmp_path, n_patches=8, n_fields=2, rng=rng)
        ds = PatchCorpusDataset(tmp_path)
        # at least one item should have a negative pixel
        has_neg = any(ds[i][0].min().item() < 0 for i in range(len(ds)))
        assert has_neg, "All residuals are non-negative — negatives were clipped"

    def test_sky_nonnegative(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=6, n_fields=2)
        ds = PatchCorpusDataset(tmp_path)
        for i in range(len(ds)):
            _, _, _, sky = ds[i]
            assert sky.min().item() >= 0.0, f"sky[{i}] has negative values"

    def test_psf_indexed_by_field_id(self, tmp_path):
        """PSF returned for each patch must match psf[field_id[i]] exactly."""
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        rng = np.random.default_rng(7)
        n_patches, n_fields = 6, 2
        _make_stacks_dir(tmp_path, n_patches=n_patches, n_fields=n_fields,
                         patch_size=128, rng=rng)
        psf_arr   = np.load(tmp_path / "psf.npy")
        fid_arr   = np.load(tmp_path / "field_id.npy")

        ds = PatchCorpusDataset(tmp_path)
        for i in range(n_patches):
            _, psf_t, _, _ = ds[i]
            expected = torch.from_numpy(psf_arr[fid_arr[i]].astype(np.float32))
            assert torch.allclose(psf_t, expected), (
                f"PSF mismatch at patch {i}: field_id={fid_arr[i]}"
            )

    def test_cond_shape_5(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=4, n_fields=2)
        ds = PatchCorpusDataset(tmp_path)
        for i in range(len(ds)):
            _, _, cond, _ = ds[i]
            assert cond.shape == (5,), f"cond shape at {i}: {cond.shape}"

    def test_missing_dir_raises(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        with pytest.raises(FileNotFoundError):
            PatchCorpusDataset(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Collate / DataLoader tests
# ---------------------------------------------------------------------------

class TestCollate:
    """Verify that a DataLoader over PatchCorpusDataset produces the
    expected batched shapes matching the train_wavelet_npe contract:
        img  (B, 2, 128, 128)
        cond (B, 5)
        sky  (B, 128, 128)
    """

    def _collate(self, batch):
        res, psf, cond, sky = zip(*batch)
        img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
        return img, torch.stack(list(cond)), torch.stack(list(sky))

    def test_dataloader_batch_shapes(self, tmp_path):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        _make_stacks_dir(tmp_path, n_patches=8, n_fields=2, patch_size=128)
        ds = PatchCorpusDataset(tmp_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                            num_workers=0, collate_fn=self._collate)
        img, cond, sky = next(iter(loader))
        assert img.shape  == (4, 2, 128, 128), f"img shape: {img.shape}"
        assert cond.shape == (4, 5),            f"cond shape: {cond.shape}"
        assert sky.shape  == (4, 128, 128),     f"sky shape: {sky.shape}"
        assert img.dtype  == torch.float32
        assert cond.dtype == torch.float32
        assert sky.dtype  == torch.float32


# ---------------------------------------------------------------------------
# Opportunistic smoke-stack check
# ---------------------------------------------------------------------------

SMOKE_STACKS = Path("/tmp/m4_smoke_stacks")


@pytest.mark.skipif(not SMOKE_STACKS.exists(),
                    reason="/tmp/m4_smoke_stacks not present")
class TestSmokeStacks:

    def test_loads(self):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        ds = PatchCorpusDataset(SMOKE_STACKS)
        assert len(ds) > 0

    def test_item_shapes(self):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        ds = PatchCorpusDataset(SMOKE_STACKS)
        residual, psf, cond, sky = ds[0]
        assert residual.shape[-2:] == (128, 128)
        assert psf.shape[-2:]      == (128, 128)
        assert cond.shape          == (5,)
        assert sky.shape[-2:]      == (128, 128)

    def test_dtypes(self):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        ds = PatchCorpusDataset(SMOKE_STACKS)
        for t in ds[0]:
            assert t.dtype == torch.float32

    def test_psf_peak_one(self):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        ds = PatchCorpusDataset(SMOKE_STACKS)
        for i in range(min(len(ds), 4)):
            _, psf, _, _ = ds[i]
            assert abs(psf.max().item() - 1.0) < 1e-4, \
                f"PSF peak != 1 at patch {i}: {psf.max().item()}"

    def test_sky_nonnegative(self):
        from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
        ds = PatchCorpusDataset(SMOKE_STACKS)
        for i in range(len(ds)):
            _, _, _, sky = ds[i]
            assert sky.min().item() >= 0.0
