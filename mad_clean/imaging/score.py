"""
mad_clean.imaging.score
=======================
Prior score model for the field-posterior loop (Fork A, design step 2).

A denoiser ``D_θ(x, σ)`` trained on the log-sky field ``f = log(s)`` by
denoising score matching, with EDM preconditioning (Karras et al. 2022).  The
prior score the Langevin sampler consumes is Tweedie's identity:

    s_θ(x, σ) = ∇_x log p_σ(x) = (D_θ(x, σ) - x) / σ²        (design §1.5).

Design notes
------------
- **EDM preconditioning adds no parameters.**  It is a σ-dependent rescaling of
  the network's input/output plus a loss weight; the capacity lives entirely in
  the conv U-Net backbone (:class:`UNet`).  It only makes the score well-behaved
  across noise levels, which matters for annealed Langevin mixing.
- **Conv-only, no attention.**  At full-field 512² attention is where memory and
  parameters explode; a plain conv U-Net keeps the budget deliberate.
- **Zero-initialised output conv.**  At initialisation ``F_θ ≡ 0`` so
  ``D_θ = c_skip · x`` — the exact Bayes denoiser for a unit-variance Gaussian
  prior.  This gives a closed-form sanity check (see tests) and a stable start.
- Training operates in a standardised log-sky space (zero mean, unit variance,
  so ``σ_data = 1``); the corpus mean/std are stored in the checkpoint and undone
  before the likelihood (which lives in linear sky space ``s = exp(f)``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["UNet", "EDMDenoiser", "edm_loss"]


# ---------------------------------------------------------------------------
# Noise-level embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a 1D tensor ``values`` (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = values[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # pad odd dim
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------------------------------------------------------------------------
# Conv U-Net backbone
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(min(32, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Compact conv U-Net, single-channel in/out, σ-conditioned.

    Parameters
    ----------
    base    : base channel count.
    mults   : channel multiplier per resolution level (length = depth).
    emb_dim : noise-embedding width.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        mults: tuple[int, ...] = (1, 2, 2, 4),
        emb_dim: int = 128,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        chans = [base * m for m in mults]

        self.in_conv = nn.Conv2d(in_ch, chans[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.down_blocks.append(_ResBlock(chans[i], chans[i], emb_dim))
            self.downsample.append(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1))

        # Bottleneck
        self.mid = _ResBlock(chans[-1], chans[-1], emb_dim)

        # Decoder (mirror); skip connections concatenated.
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(chans) - 1, 0, -1):
            self.upsample.append(
                nn.ConvTranspose2d(chans[i], chans[i - 1], 4, stride=2, padding=1)
            )
            self.up_blocks.append(_ResBlock(2 * chans[i - 1], chans[i - 1], emb_dim))

        self.out_norm = nn.GroupNorm(min(32, chans[0]), chans[0])
        self.out_conv = nn.Conv2d(chans[0], in_ch, 3, padding=1)
        # Zero-init so F_θ ≡ 0 at start → D = c_skip·x (Gaussian Bayes denoiser).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        emb = self.emb_mlp(_sinusoidal_embedding(c_noise, self.emb_dim))
        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsample):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, emb)
        for up, block in zip(self.upsample, self.up_blocks):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, emb)
        h = self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
        return h


# ---------------------------------------------------------------------------
# EDM preconditioning wrapper
# ---------------------------------------------------------------------------

class EDMDenoiser(nn.Module):
    """Wrap a backbone with EDM preconditioning and expose the Tweedie score.

    ``forward(x, sigma)`` returns the denoiser estimate ``D_θ(x, σ)``;
    ``score(x, sigma)`` returns ``(D_θ - x)/σ²`` = the prior score.
    """

    def __init__(self, backbone: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)

    def _coeffs(self, sigma: torch.Tensor):
        sd2 = self.sigma_data**2
        s2 = sigma**2
        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * self.sigma_data / torch.sqrt(s2 + sd2)
        c_in = 1.0 / torch.sqrt(s2 + sd2)
        c_noise = torch.log(sigma) / 4.0
        return c_skip, c_out, c_in, c_noise

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        s = sigma.view(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self._coeffs(s)
        F = self.backbone(c_in * x, c_noise.flatten())
        return c_skip * x + c_out * F

    def score(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(x.shape[0])
        s = sigma.view(-1, 1, 1, 1)
        return (self.forward(x, sigma) - x) / (s**2)


# ---------------------------------------------------------------------------
# Denoising score-matching loss (EDM weighting)
# ---------------------------------------------------------------------------

def edm_loss(
    model: EDMDenoiser,
    f0: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """EDM denoising score-matching loss on a clean log-sky batch ``f0`` (B,1,H,W).

    Samples ``ln σ ~ N(p_mean, p_std)`` per example, noises ``f0``, and returns
    the σ-weighted denoiser MSE ``λ(σ)‖D_θ(f0+n, σ) − f0‖²`` with
    ``λ(σ) = (σ²+σ_data²)/(σ·σ_data)²``."""
    b = f0.shape[0]
    ln_sigma = p_mean + p_std * torch.randn(b, generator=generator, device=f0.device)
    sigma = ln_sigma.exp()
    s = sigma.view(-1, 1, 1, 1)
    noise = torch.randn(f0.shape, generator=generator, device=f0.device, dtype=f0.dtype)
    x = f0 + s * noise
    d = model(x, sigma)
    sd = model.sigma_data
    weight = (s**2 + sd**2) / (s * sd) ** 2
    return (weight * (d - f0) ** 2).mean()
