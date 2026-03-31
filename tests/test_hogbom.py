"""
tests/test_hogbom.py — tests for mad_clean.hogbom.hogbom_clean
"""

import torch
import pytest

from mad_clean.hogbom import hogbom_clean


# ── helpers ───────────────────────────────────────────────────────────────────

def _delta_psf(H: int = 64) -> torch.Tensor:
    psf = torch.zeros(H, H)
    psf[H // 2, H // 2] = 1.0
    return psf


def _gaussian_psf(H: int = 64, sigma: float = 4.0) -> torch.Tensor:
    cy, cx = H // 2, H // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(H).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))


# ── delta PSF: residual must reach ~0 in one step at gain=1 ──────────────────

class TestDeltaPSF:
    @pytest.mark.parametrize("use_psf_patch", [False, True])
    def test_single_iteration_clears_residual(self, use_psf_patch):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 1.0
        psf   = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=10,
                         use_psf_patch=use_psf_patch)
        assert r["residual"].abs().max().item() < 1e-5

    @pytest.mark.parametrize("use_psf_patch", [False, True])
    def test_model_recovers_flux(self, use_psf_patch):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 2.5
        psf   = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=10,
                         use_psf_patch=use_psf_patch)
        assert abs(float(r["model"][32, 32]) - 2.5) < 1e-4


# ── flux conservation ─────────────────────────────────────────────────────────

class TestFluxConservation:
    def test_model_total_flux_converges(self):
        """After many iterations with delta PSF, model total flux ≈ dirty total flux."""
        H = 64
        dirty = torch.zeros(H, H)
        dirty[20, 20] = 1.0
        dirty[44, 44] = 0.6
        psf = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=200,
                         use_psf_patch=False)
        assert abs(float(r["model"].sum()) - float(dirty.sum())) < 1e-3


# ── convergence flag ─────────────────────────────────────────────────────────

class TestConvergence:
    def test_converged_true_when_threshold_met(self):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 0.01
        psf   = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=0.1, threshold=0.5, n_iter=100,
                         use_psf_patch=False)
        assert r["converged"] is True

    def test_converged_false_when_niter_reached(self):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 1.0
        psf   = _delta_psf(H)
        # threshold=0.0 disables early stopping; delta PSF has no sidelobes
        r = hogbom_clean(dirty, psf, gain=0.1, threshold=0.0, n_iter=5,
                         use_psf_patch=False)
        assert r["converged"] is False
        assert r["n_iter"] == 5

    def test_peak_flux_nonnegative(self):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 1.0
        psf   = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=0.5, threshold=1e-6, n_iter=50,
                         use_psf_patch=False)
        assert r["peak_flux"] >= 0.0

    def test_auto_threshold_delta_psf_cleans_fully(self):
        """threshold=None → 0.1 × dirty_peak = 0.1; delta PSF zeroes residual in 1 step."""
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 1.0
        psf   = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=None, n_iter=10,
                         use_psf_patch=False)
        assert r["residual"].abs().max().item() < 1e-5


# ── clean box ────────────────────────────────────────────────────────────────

class TestCleanBox:
    def test_source_outside_box_untouched(self):
        """Source outside clean_box should remain in residual, not cleaned."""
        H = 64
        dirty = torch.zeros(H, H)
        dirty[10, 10] = 1.0   # inside box
        dirty[50, 50] = 1.0   # outside box
        psf = _delta_psf(H)
        box = (0, 30, 0, 30)   # top-left quadrant only
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=50,
                         clean_box=box, use_psf_patch=False)
        # Source at (50, 50) should still be large in residual
        assert float(r["residual"][50, 50]) > 0.5


# ── multi-dimensional input ───────────────────────────────────────────────────

class TestMultiDim:
    def test_spectral_cube_no_error(self):
        """hogbom_clean should accept (nchan, H, W) without error."""
        nchan, H = 3, 32
        dirty = torch.zeros(nchan, H, H)
        for c in range(nchan):
            dirty[c, 16, 16] = float(c + 1)
        psf = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-5, n_iter=50,
                         use_psf_patch=False)
        assert r["model"].shape == (nchan, H, H)
        assert r["residual"].shape == (nchan, H, H)

    def test_spectral_residual_small_after_clean(self):
        """All channel peaks should be reduced after sufficient iterations."""
        nchan, H = 2, 32
        dirty = torch.zeros(nchan, H, H)
        dirty[0, 16, 16] = 1.0
        dirty[1, 16, 16] = 0.5
        psf = _delta_psf(H)
        r = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-5, n_iter=100,
                         use_psf_patch=False)
        assert r["residual"].abs().max().item() < 1e-4


# ── classic vs patch: same result on delta PSF ───────────────────────────────

class TestClassicVsPatch:
    def test_both_modes_agree_on_delta_psf(self):
        H = 64
        dirty = torch.zeros(H, H); dirty[32, 32] = 1.0
        psf   = _delta_psf(H)
        rc = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=10,
                          use_psf_patch=False)
        rp = hogbom_clean(dirty, psf, gain=1.0, threshold=1e-6, n_iter=10,
                          use_psf_patch=True)
        assert rc["residual"].abs().max().item() < 1e-5
        assert rp["residual"].abs().max().item() < 1e-5
