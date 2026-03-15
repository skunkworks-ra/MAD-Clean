"""
tests/test_filters.py — tests for mad_clean.filters.FilterBank
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mad_clean.filters import FilterBank


# ── helpers ───────────────────────────────────────────────────────────────────

def _random_atoms(k: int = 8, f: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((k, f, f)).astype(np.float32)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_filterbank_normalisation():
    """All atoms are normalised to unit L2 norm on construction."""
    atoms = _random_atoms(k=16, f=7)
    fb    = FilterBank(atoms)
    norms = fb.atoms.reshape(fb.K, -1).norm(dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(fb.K), atol=1e-5)


def test_filterbank_shape():
    """FilterBank exposes correct K, F and derived tensor shapes."""
    k, f  = 10, 9
    fb    = FilterBank(_random_atoms(k, f))
    assert fb.K == k
    assert fb.F == f
    assert fb.atoms.shape   == (k, f, f)
    assert fb.D.shape       == (f * f, k)
    assert fb.D_fft.shape   == (k, f, f // 2 + 1)
    assert fb.D_fft.is_complex()


def test_filterbank_bad_input():
    """ValueError raised when atoms array is not 3D."""
    with pytest.raises(ValueError):
        FilterBank(np.ones((5, 5), dtype=np.float32))


def test_filterbank_save_load_roundtrip():
    """Atoms saved with save() and reloaded with load() match within float32 tolerance."""
    atoms = _random_atoms(k=4, f=5)
    fb    = FilterBank(atoms)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "atoms.npy"
        fb.save(path)
        fb2  = FilterBank.load(path)
    np.testing.assert_allclose(
        fb.atoms.numpy(), fb2.atoms.numpy(), atol=1e-6
    )


def test_filterbank_to_device_cpu():
    """to('cpu') returns a new FilterBank on CPU with identical atoms."""
    fb  = FilterBank(_random_atoms())
    fb2 = fb.to("cpu")
    assert fb2.device == torch.device("cpu")
    np.testing.assert_allclose(fb.atoms.numpy(), fb2.atoms.numpy(), atol=1e-6)


def test_dead_atom_report_structure():
    """dead_atom_report returns a dict with the expected keys and sensible values."""
    fb     = FilterBank(_random_atoms(k=8, f=5))
    report = fb.dead_atom_report()
    for key in ("n_dead", "n_active", "norm_min", "norm_mean", "norm_max"):
        assert key in report
    assert report["n_dead"] + report["n_active"] == fb.K
    assert 0.0 < report["norm_min"] <= report["norm_mean"] <= report["norm_max"]


def test_dead_atom_report_counts_dead():
    """dead_atom_report correctly identifies atoms below the threshold."""
    atoms       = _random_atoms(k=4, f=5)
    atoms[0]   *= 0.0   # kill first atom
    fb          = FilterBank(atoms)
    # FilterBank normalises — a zero atom will remain zero (0/max(0, 1e-8) = 0)
    report = fb.dead_atom_report(threshold=0.5)
    assert report["n_dead"] >= 1
