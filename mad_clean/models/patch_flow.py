"""Patch-level conditional flow matching model for minor-cycle deconvolution.

Architecture
------------
U-Net velocity field estimator. Input: 3 channels (x_t, dirty, psf) at
128x128. Output: 1-channel velocity field at 128x128.

Time t and sigma_local are injected at every residual block via FiLM
(scale + shift).

Training (CFM, Lipman et al. 2022)
------------------------------------
x_0 ~ N(0, I),  x_1 = clean sky patch
x_t = (1 - t) * x_0 + t * x_1
v   = x_1 - x_0   (target velocity)
loss = ||u_theta(x_t, t, dirty, psf, sigma) - v||^2

Inference
---------
Start from x_0 ~ N(0, I), integrate velocity field with n_steps Euler steps
conditioned on (dirty, psf, sigma). Output is a clean sky patch (Jy/pixel).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["PatchFlow", "cfm_loss"]


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Affine conditioning: scale + shift from a conditioning vector."""
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W),  cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)   # (B, C) each
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film  = FiLM(cond_dim, channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res  = ResBlock(out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = F.silu(self.conv(x))
        x = self.res(x, cond)
        return self.down(x), x   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.res  = ResBlock(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = F.silu(self.conv(x))
        return self.res(x, cond)


# ---------------------------------------------------------------------------
# PatchFlow U-Net
# ---------------------------------------------------------------------------

class PatchFlow(nn.Module):
    """CFM velocity-field U-Net for 128x128 patches.

    Parameters
    ----------
    base_channels : int
        Channel count at the first encoder level. Doubles each level.
    depth : int
        Number of down/up-sampling stages (default 4 → 128→64→32→16→8).
    time_embed_dim : int
        Sinusoidal time embedding dimension.
    sigma_dim : int
        Dimension of sigma_local embedding (simple linear projection).
    """

    def __init__(
        self,
        base_channels: int = 32,
        depth: int = 4,
        time_embed_dim: int = 128,
        sigma_dim: int = 32,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Time and sigma conditioning → single cond vector
        cond_dim = time_embed_dim + sigma_dim
        self.time_embed = SinusoidalEmbed(time_embed_dim)
        self.sigma_proj = nn.Sequential(
            nn.Linear(1, sigma_dim),
            nn.SiLU(),
            nn.Linear(sigma_dim, sigma_dim),
        )

        # Input projection: 3 channels (x_t, dirty, psf) → base_channels
        self.input_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        # Encoder
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch  = channels[i]
            out_ch = channels[i + 1] if i + 1 < depth else channels[i]
            self.down_blocks.append(DownBlock(in_ch, out_ch, cond_dim))

        # Bottleneck
        bot_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bot_ch, cond_dim),
            ResBlock(bot_ch, cond_dim),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch   = channels[i + 1] if i + 1 < depth else channels[i]
            skip_ch = channels[i + 1] if i + 1 < depth else channels[i]
            out_ch  = channels[i]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, cond_dim))

        # Output projection → 1-channel velocity field
        self.output_conv = nn.Conv2d(base_channels, 1, 1)

    def _cond(self, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t_emb     = self.time_embed(t)                        # (B, time_embed_dim)
        sigma_emb = self.sigma_proj(sigma.unsqueeze(-1))      # (B, sigma_dim)
        return torch.cat([t_emb, sigma_emb], dim=-1)          # (B, cond_dim)

    def forward(
        self,
        x_t:    torch.Tensor,   # (B, 1, H, W) noisy/interpolated clean patch
        t:      torch.Tensor,   # (B,) in [0, 1]
        dirty:  torch.Tensor,   # (B, 1, H, W)
        psf:    torch.Tensor,   # (B, 1, H, W)
        sigma:  torch.Tensor,   # (B,) sigma_local
    ) -> torch.Tensor:
        cond = self._cond(t, sigma)                           # (B, cond_dim)

        x = torch.cat([x_t, dirty, psf], dim=1)              # (B, 3, H, W)
        x = F.silu(self.input_conv(x))

        skips = []
        for block in self.down_blocks:
            x, skip = block(x, cond)
            skips.append(skip)

        for res in self.bottleneck:
            x = res(x, cond)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip, cond)

        return self.output_conv(x)                            # (B, 1, H, W)

    @torch.no_grad()
    def sample(
        self,
        dirty:  torch.Tensor,   # (B, 1, H, W)
        psf:    torch.Tensor,   # (B, 1, H, W)
        sigma:  torch.Tensor,   # (B,)
        n_steps: int = 50,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Euler integration from noise to clean sky patch."""
        if device is None:
            device = dirty.device
        B, _, H, W = dirty.shape
        x = torch.randn(B, 1, H, W, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self(x, t, dirty, psf, sigma)
            x = x + v * dt
        return x


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cfm_loss(
    model:  PatchFlow,
    dirty:  torch.Tensor,   # (B, 1, H, W)
    psf:    torch.Tensor,   # (B, 1, H, W)
    clean:  torch.Tensor,   # (B, 1, H, W)  target clean sky patch
    sigma:  torch.Tensor,   # (B,)
) -> torch.Tensor:
    """CFM MSE loss.  Samples t and x_0, computes ||u_theta - v||^2."""
    B = dirty.shape[0]
    device = dirty.device

    t   = torch.rand(B, device=device)
    x_0 = torch.randn_like(clean)
    x_1 = clean
    x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1
    v   = x_1 - x_0   # target velocity

    v_pred = model(x_t, t, dirty, psf, sigma)
    return F.mse_loss(v_pred, v)
