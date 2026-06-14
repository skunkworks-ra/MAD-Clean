"""
tests/test_forward.py
=====================
Unit tests for mad_clean.imaging.forward — the image-domain forward operator
for the field-posterior loop.  CPU, seconds, no GPU.

Load-bearing properties:
  * A and A^T are a true adjoint pair (assumption A0 in the design),
  * A is linear,
  * a zero-DC dirty beam turns positive sky into a field with a negative bowl
    (the instrumental negatives of §4.3),
  * the likelihood score vanishes at the truth (noiseless).
"""

from __future__ import annotations

import numpy as np
import torch

from mad_clean.imaging.forward import ImageDomainForward

SIZE = 64


def _gaussian_psf(size: int, fwhm: float = 4.0) -> torch.Tensor:
    """Peak-normalised, peak-centred Gaussian (a positive, sum>0 beam)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    c = size // 2
    ax = torch.arange(size, dtype=torch.float64) - c
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.max()


def _zero_dc_psf(size: int) -> torch.Tensor:
    """A crude dirty beam: positive core minus a broad pedestal so it sums to
    ≈0 (missing DC), peak-normalised."""
    psf = _gaussian_psf(size, fwhm=3.0)
    broad = _gaussian_psf(size, fwhm=20.0)
    psf = psf - broad * (psf.sum() / broad.sum())  # subtract to zero the sum
    return psf / psf.max()


def test_adjoint_pair():
    """⟨A x, y⟩ == ⟨x, A^T y⟩ to machine precision."""
    torch.manual_seed(0)
    op = ImageDomainForward(_gaussian_psf(SIZE))
    x = torch.randn(SIZE, SIZE, dtype=torch.float64)
    y = torch.randn(SIZE, SIZE, dtype=torch.float64)
    lhs = float((op.forward(x) * y).sum())
    rhs = float((x * op.adjoint(y)).sum())
    assert abs(lhs - rhs) < 1e-9, f"adjoint mismatch: {lhs} vs {rhs}"


def test_linearity():
    op = ImageDomainForward(_gaussian_psf(SIZE))
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64)
    lhs = op.forward(a + b)
    rhs = op.forward(a) + op.forward(b)
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_zero_dc_beam_makes_negative_bowl():
    """A positive sky convolved with a zero-DC beam acquires negatives."""
    psf = _zero_dc_psf(SIZE)
    assert abs(float(psf.sum())) < 1e-3 * float(psf.abs().sum()), "psf not zero-DC"
    op = ImageDomainForward(psf)
    sky = torch.zeros(SIZE, SIZE, dtype=torch.float64)
    sky[SIZE // 2, SIZE // 2] = 1.0  # a positive point source
    sky += 0.05  # plus a positive floor
    dirty = op.forward(sky)
    assert float(dirty.min()) < 0.0, "no negative bowl from zero-DC beam"


def test_batched_forward():
    op = ImageDomainForward(_gaussian_psf(SIZE))
    batch = torch.randn(3, SIZE, SIZE, dtype=torch.float64)
    out = op.forward(batch)
    assert out.shape == (3, SIZE, SIZE)
    # Each slice matches the unbatched call.
    torch.testing.assert_close(out[1], op.forward(batch[1]), rtol=1e-10, atol=1e-10)


def test_likelihood_score_zero_at_truth():
    """With d = A s_true and no noise, A^T N^{-1} r = 0 at s = s_true."""
    op = ImageDomainForward(_gaussian_psf(SIZE))
    s_true = torch.rand(SIZE, SIZE, dtype=torch.float64)
    d = op.make_dirty(s_true, noise_std=0.0)
    score = op.likelihood_score(s_true, d, noise_std=0.1)
    assert float(score.abs().max()) < 1e-9


def test_make_dirty_reproducible():
    op = ImageDomainForward(_gaussian_psf(SIZE))
    s = torch.rand(SIZE, SIZE, dtype=torch.float64)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    d1 = op.make_dirty(s, noise_std=0.1, generator=g1)
    d2 = op.make_dirty(s, noise_std=0.1, generator=g2)
    torch.testing.assert_close(d1, d2)
