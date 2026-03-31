"""
tests/test_psf_utils.py — tests for mad_clean.psf_utils.compute_psf_patch
"""

import torch
import pytest

from mad_clean.psf_utils import compute_psf_patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _delta_psf(H: int = 64) -> torch.Tensor:
    psf = torch.zeros(H, H)
    psf[H // 2, H // 2] = 1.0
    return psf


def _gaussian_psf(H: int = 64, sigma: float = 5.0) -> torch.Tensor:
    cy, cx = H // 2, H // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(H).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDeltaPSF:
    def test_patch_size_is_minimal(self):
        psf = _delta_psf(64)
        patch, (hh, hw) = compute_psf_patch(psf, energy_frac=0.99)
        assert patch.shape[0] == 2 * hh + 1
        assert patch.shape[1] == 2 * hw + 1

    def test_patch_contains_peak(self):
        psf = _delta_psf(64)
        patch, (hh, hw) = compute_psf_patch(psf, energy_frac=0.99)
        ph, pw = patch.shape
        assert float(patch[ph // 2, pw // 2]) == pytest.approx(1.0)


class TestGaussianPSF:
    def test_energy_fraction_captured(self):
        psf   = _gaussian_psf(64, sigma=5.0)
        frac  = 0.85
        patch, _ = compute_psf_patch(psf, energy_frac=frac)
        captured = float((patch ** 2).sum()) / float((psf ** 2).sum())
        assert captured >= frac - 1e-4

    def test_patch_shape_is_odd(self):
        """Patch should always be odd-sized (centred on peak)."""
        psf   = _gaussian_psf(64, sigma=5.0)
        patch, (hh, hw) = compute_psf_patch(psf)
        assert patch.shape[0] % 2 == 1
        assert patch.shape[1] % 2 == 1

    def test_patch_centred_on_peak(self):
        psf   = _gaussian_psf(64, sigma=5.0)
        patch, (hh, hw) = compute_psf_patch(psf)
        ph, pw = patch.shape
        cy_p, cx_p = ph // 2, pw // 2
        assert float(patch[cy_p, cx_p]) == pytest.approx(float(patch.max()), rel=1e-4)


class TestEdgeCases:
    def test_zero_psf_returns_patch(self):
        psf = torch.zeros(64, 64)
        patch, (hh, hw) = compute_psf_patch(psf)
        assert patch.shape == (2 * hh + 1, 2 * hw + 1)

    def test_high_energy_frac_gives_large_patch(self):
        psf = _gaussian_psf(64, sigma=10.0)
        _, (hh_lo, _) = compute_psf_patch(psf, energy_frac=0.50)
        _, (hh_hi, _) = compute_psf_patch(psf, energy_frac=0.99)
        assert hh_hi >= hh_lo
