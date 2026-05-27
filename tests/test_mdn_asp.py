"""tests/test_mdn_asp.py — unit tests for mad_clean.models.mdn_asp.

All tests run on CPU in seconds.  Use a tiny model (base_channels=8,
hidden=64) to keep wall time short.
"""

import math

import pytest
import torch

from mad_clean.models.mdn_asp import (
    MDNAsp,
    MixParams,
    decode_pa,
    encode_pa,
    make_cond,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    """Small MDNAsp for fast CPU tests."""
    return MDNAsp(base_channels=8, hidden=64, n_components=5)


@pytest.fixture(scope="module")
def batch():
    """Small batch of random inputs."""
    B = 4
    image = torch.randn(B, 2, 128, 128)
    sigma = torch.rand(B) * 0.1 + 0.01
    config = torch.randint(0, 4, (B,))
    cond = make_cond(sigma, config)
    return image, cond


# ---------------------------------------------------------------------------
# PA round-trip
# ---------------------------------------------------------------------------

def test_pa_roundtrip_known():
    """Encode a known PA and recover it modulo pi."""
    for pa_deg in [0.0, 30.0, -45.0, 89.0, -90.0]:
        pa = torch.tensor([math.radians(pa_deg)])
        enc = encode_pa(pa)
        pa_rec = decode_pa(enc[..., 0], enc[..., 1])
        # PA is defined mod pi; atan2(sin2t, cos2t)/2 is in (-pi/2, pi/2]
        # Check round-trip within that range
        diff = (pa_rec - pa).abs()
        close = diff < 1e-5
        wrap = (diff - math.pi).abs() < 1e-5
        assert (close | wrap).all(), (
            f"PA round-trip failed for {pa_deg} deg: got {pa_rec.item():.6f} rad"
        )


def test_pa_encode_shape():
    pa = torch.randn(3, 5)
    enc = encode_pa(pa)
    assert enc.shape == (3, 5, 2)


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_shape(tiny_model, batch):
    image, cond = batch
    params = tiny_model(image, cond)
    B, K = image.shape[0], tiny_model.K
    assert params.logits.shape == (B, K)
    assert params.mu.shape == (B, K, 7)
    assert params.log_std.shape == (B, K, 7)


def test_log_std_clamped(tiny_model, batch):
    """log_std must be within the declared bounds."""
    from mad_clean.models.mdn_asp import LOG_STD_MAX
    image, cond = batch
    params = tiny_model(image, cond)
    assert (params.log_std <= LOG_STD_MAX).all()
    # Each dim should be >= its floor (broadcasted check via min)
    assert params.log_std.min() >= -5.0 - 1e-6


# ---------------------------------------------------------------------------
# NLL is finite on random inputs
# ---------------------------------------------------------------------------

def test_nll_finite(tiny_model, batch):
    image, cond = batch
    params = tiny_model(image, cond)
    B = image.shape[0]
    targets = torch.randn(B, 6)
    loss = tiny_model.nll_loss(params, targets)
    assert torch.isfinite(loss), f"NLL not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Sampling shape
# ---------------------------------------------------------------------------

def test_sample_shape(tiny_model, batch):
    image, cond = batch
    params = tiny_model(image, cond)
    B = image.shape[0]
    n = 10
    samples = tiny_model.sample(params, n=n)
    assert samples.shape == (B, n, 6), f"Expected ({B}, {n}, 6), got {samples.shape}"


def test_sample_pa_in_range(tiny_model, batch):
    """Decoded PA should be in (-pi/2, pi/2]."""
    image, cond = batch
    params = tiny_model(image, cond)
    samples = tiny_model.sample(params, n=20)
    pa = samples[..., 5]
    assert (pa > -math.pi / 2 - 1e-4).all()
    assert (pa <= math.pi / 2 + 1e-4).all()


# ---------------------------------------------------------------------------
# Mode shape
# ---------------------------------------------------------------------------

def test_mode_shape(tiny_model, batch):
    image, cond = batch
    params = tiny_model(image, cond)
    mode = tiny_model.mode(params)
    B = image.shape[0]
    assert mode.shape == (B, 6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows():
    """loss.backward() produces non-zero grads on all parameters."""
    model = MDNAsp(base_channels=8, hidden=64, n_components=5)
    model.train()
    B = 2
    image = torch.randn(B, 2, 128, 128)
    sigma = torch.rand(B) * 0.05 + 0.01
    config = torch.zeros(B, dtype=torch.long)
    cond = make_cond(sigma, config)
    targets = torch.randn(B, 6)

    params = model(image, cond)
    loss = model.nll_loss(params, targets)
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
        elif p.grad.abs().max() == 0:
            no_grad.append(name)
    assert not no_grad, f"Zero/missing grads on: {no_grad}"


# ---------------------------------------------------------------------------
# make_cond shape
# ---------------------------------------------------------------------------

def test_make_cond_shape():
    B = 7
    sigma = torch.rand(B)
    config = torch.randint(0, 4, (B,))
    cond = make_cond(sigma, config)
    assert cond.shape == (B, 5)
    # One-hot part should sum to 1 per row
    assert (cond[:, 1:].sum(dim=-1) == 1).all()


# ---------------------------------------------------------------------------
# PA degenerate-case wrap (theta ↔ theta + pi identified under (sin 2θ, cos 2θ))
# ---------------------------------------------------------------------------

def test_pa_period_is_pi_under_encoding():
    """encode_pa(θ) == encode_pa(θ + π) for all θ. The encoding has period π."""
    thetas = torch.tensor([0.0, 0.3, 1.4, -1.1, math.pi / 3])
    a = encode_pa(thetas)
    b = encode_pa(thetas + math.pi)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_pa_decode_into_canonical_half_open_range():
    """decode_pa output sits in (-pi/2, pi/2]."""
    # Sample many encodings of arbitrary theta; decode should land in canonical range
    rng = torch.Generator().manual_seed(0)
    thetas = torch.empty(200).uniform_(-3.0, 3.0, generator=rng)
    enc = encode_pa(thetas)
    decoded = decode_pa(enc[..., 0], enc[..., 1])
    assert (decoded > -math.pi / 2 - 1e-6).all()
    assert (decoded <= math.pi / 2 + 1e-6).all()


def test_pa_degenerate_wrap_zero_under_period_pi_metric():
    """θ = +π/2 and θ = −π/2 are the same point in the encoded space.

    Decoded scalars may differ by ~π — atan2 of (sin π, cos π) ≈ (±ε, −1)
    returns ±π depending on the sign of float-ε, mapping to ±π/2 after the
    half. The correct distance is the period-π wrap.
    """
    a = encode_pa(torch.tensor(math.pi / 2))
    b = encode_pa(torch.tensor(-math.pi / 2))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    decoded_a = float(decode_pa(a[..., 0], a[..., 1]))
    decoded_b = float(decode_pa(b[..., 0], b[..., 1]))
    d = abs(decoded_a - decoded_b)
    wrap = min(d, math.pi - d)
    assert wrap < 1e-5, (
        f"period-pi wrap distance not zero: a={decoded_a:.6f}, "
        f"b={decoded_b:.6f}, wrap={wrap:.6e}"
    )
