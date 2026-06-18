"""Tests for the starlet transform and the decimated theta codec."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mad_clean.data.extended_sky import (
    render_filament,
    render_gaussian_blob,
    render_shell,
)
from mad_clean.wavelet.starlet import (
    StarletCodec,
    starlet_reconstruct,
    starlet_transform,
)


def _batch_of_skies(n: int = 16, size: int = 128, seed: int = 0) -> torch.Tensor:
    """Representative extended-source sky cutouts for calibration/round-trip."""
    rng = np.random.default_rng(seed)
    gens = [render_gaussian_blob, render_shell, render_filament]
    out = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        gen = gens[i % 3]
        img, _ = gen(
            size=size,
            cx=float(rng.uniform(40, size - 40)),
            cy=float(rng.uniform(40, size - 40)),
            flux_jy=float(np.exp(rng.uniform(np.log(1e-3), np.log(1e-1)))),
            rng=rng,
        )
        out[i] = img
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_starlet_exact_reconstruction():
    rng = np.random.default_rng(1)
    img = torch.from_numpy(rng.normal(size=(2, 128, 128)).astype(np.float32))
    planes = starlet_transform(img, n_scales=6)
    assert len(planes) == 7
    rec = starlet_reconstruct(planes)
    assert torch.allclose(rec, img, atol=1e-5)


def test_starlet_2d_input():
    img = torch.randn(64, 64)
    planes = starlet_transform(img, n_scales=4)
    assert planes[0].shape == (64, 64)
    rec = starlet_reconstruct(planes)
    assert torch.allclose(rec, img, atol=1e-5)


def test_starlet_scale_separation():
    """A broad Gaussian's power should sit in coarse planes, not fine ones."""
    size = 128
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    sig = 10.0
    img = torch.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * sig ** 2))
    planes = starlet_transform(img, n_scales=6)
    powers = torch.tensor([(p ** 2).sum() for p in planes])
    # Fine planes (w_1, w_2) carry a negligible share of the power.
    assert (powers[0] + powers[1]) / powers.sum() < 0.01


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

def test_codec_theta_dim_layout():
    codec = StarletCodec(image_size=128, n_scales=6, drop_scales=(1,))
    dims = codec.plane_dims()
    # Kept: w_2..w_6 + smooth, all at full resolution (no decimation)
    sides = [s for s, _ in dims]
    assert sides == [128, 128, 128, 128, 128, 128]
    assert codec.theta_dim == 6 * 128 ** 2


def test_codec_requires_calibration():
    codec = StarletCodec()
    with pytest.raises(RuntimeError):
        codec.encode(torch.zeros(1, 128, 128))


def test_codec_round_trip_extended_sources():
    """Encode/decode must preserve structure.

    No decimation; only the sub-beam plane is dropped (drop_scales=(1,) default).
    Round-trip error should be very small for extended sources.
    """
    skies = _batch_of_skies(n=12)
    codec = StarletCodec()
    codec.calibrate(skies)
    theta = codec.encode(skies)
    assert theta.shape == (12, codec.theta_dim)
    assert torch.isfinite(theta).all()
    rec = codec.decode(theta)
    # Relative L2 error per image
    err = (rec - skies).flatten(1).norm(dim=1) / skies.flatten(1).norm(dim=1)
    assert float(err.median()) < 0.35, f"median relative error {err.median():.3f}"


def test_codec_state_dict_round_trip():
    skies = _batch_of_skies(n=4)
    codec = StarletCodec()
    codec.calibrate(skies)
    codec2 = StarletCodec.from_state_dict(codec.state_dict())
    t1 = codec.encode(skies)
    t2 = codec2.encode(skies)
    assert torch.allclose(t1, t2)


def test_codec_decode_flux_conservation():
    """Total flux of the decoded image stays close to the original for
    extended sources (avg-pool decimation is mean-preserving per plane)."""
    skies = _batch_of_skies(n=12)
    codec = StarletCodec()
    codec.calibrate(skies)
    rec = codec.decode(codec.encode(skies))
    f_in  = skies.flatten(1).sum(dim=1)
    f_out = rec.flatten(1).sum(dim=1)
    rel = ((f_out - f_in).abs() / f_in.abs()).median()
    assert float(rel) < 0.15, f"median flux error {rel:.3f}"


# ---------------------------------------------------------------------------
# Support weights (training-loss shaping)
# ---------------------------------------------------------------------------

def test_support_weights_shape_and_range():
    skies = _batch_of_skies(n=4)
    codec = StarletCodec()
    codec.calibrate(skies)
    w = codec.support_weights(skies, outside_weight=0.05)
    assert w.shape == (4, codec.theta_dim)
    assert torch.isclose(w.min(), torch.tensor(0.05))
    assert torch.isclose(w.max(), torch.tensor(1.0))


def test_support_weights_localise_to_source():
    """A compact source in one corner: fine-scale weights must be 1 near
    it and outside_weight far away."""
    sky = torch.zeros(1, 128, 128)
    sky[0, 30:34, 30:34] = 1e-3
    codec = StarletCodec(drop_scales=())
    codec.calibrate(_batch_of_skies(n=4))
    w = codec.support_weights(sky, outside_weight=0.05)
    # First kept plane is w_1, undecimated 128x128.
    w1 = w[0, : 128 * 128].reshape(128, 128)
    assert float(w1[32, 32]) == 1.0
    assert abs(float(w1[100, 100]) - 0.05) < 1e-6


def test_codec_point_source_round_trip():
    """Points must survive the codec with w_1 kept (Option A).  Pilot v2
    (2026-06-11) lost ~99% of point flux to full-plane MAD calibration +
    decode z-clamp; calibration now uses active coefficients only."""
    skies = _batch_of_skies(n=8)
    points = torch.zeros(4, 128, 128)
    for i, (y, x, f) in enumerate(
            [(64, 64, 1e-3), (40, 80, 1e-4), (90, 33, 3e-3), (65, 63, 5e-4)]):
        points[i, y, x] = f
    codec = StarletCodec(drop_scales=())
    codec.calibrate(torch.cat([skies, points]))
    rec = codec.decode(codec.encode(points))
    f_in = points.flatten(1).sum(dim=1)
    f_out = rec.flatten(1).sum(dim=1)
    rel_flux = ((f_out - f_in).abs() / f_in).max()
    assert float(rel_flux) < 0.2, f"point flux error {rel_flux:.3f}"
    # Peak must stay on the source pixel (no shift)
    for i in range(4):
        idx = rec[i].argmax()
        iy, ix = int(idx // 128), int(idx % 128)
        ty, tx = int(points[i].argmax() // 128), int(points[i].argmax() % 128)
        assert abs(iy - ty) <= 1 and abs(ix - tx) <= 1
