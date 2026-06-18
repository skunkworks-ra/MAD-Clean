"""PSFCondFlow: conditional flow matching for single-source island deconvolution.

Maps a 2-channel (dirty, PSF) island cutout to a clean source pixel image via
conditional flow matching (CFM).  The encoder is the proven MDN-Asp 2-channel
CNN; the flow backbone is the score.py UNet rewired to fuse CFM time t with
the encoder context vector.

Design (psf_condflow_design.md)
-------------------------------
- ContextEncoder: 2-ch (residual, PSF) CNN + FiLM → ctx ∈ R^256
  Identical to CoeffFlow's encoder: per-sample residual normalization, log10
  of that scale appended to cond so absolute flux is recoverable from ctx.
- VelocityUNet: score.py UNet(base=32) with embedding recomputed from
  cat(sinusoidal(t), ctx) instead of sinusoidal(log_sigma).
- Training: CFM linear interpolant x_t = t*s + (1-t)*ε, MSE target u = s - ε.
  Pixel-weighted loss (flux-proportional) so sparse source pixels are not
  swamped by the zero-sky background in the MSE.
- Inference: Euler ODE from N(0,I), relu for non-negativity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mad_clean.imaging.score import UNet, _sinusoidal_embedding
from mad_clean.models.mdn_asp import COND_DIM, FiLMBlock

__all__ = ["PSFCondFlow", "cfm_loss"]


# ---------------------------------------------------------------------------
# Context encoder (mirrors coeff_flow._ContextEncoder exactly)
# ---------------------------------------------------------------------------

class _ContextEncoder(nn.Module):
    """2-channel (residual, PSF) CNN encoder with per-sample normalization.

    Per-sample residual normalization is essential: physical residuals are
    ~1e-4 Jy while the PSF channel peaks at 1, so without it GroupNorm
    statistics are PSF-dominated and all contexts collapse (same failure as
    2026-06-11 conditioning failure in CoeffFlow).  The normalization scale
    is appended to cond as log10 so the flow can recover absolute flux.
    """

    def __init__(self, base_channels: int, context_dim: int, cond_dim: int):
        super().__init__()
        c = base_channels

        def block(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
                nn.Conv2d(oc, oc, 3, stride=1, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
            )

        # 128 → 64 → 32 → 16 → 8 → 4
        self.enc = nn.Sequential(
            block(2, c),
            block(c, 2 * c),
            block(2 * c, 4 * c),
            block(4 * c, 4 * c),
            block(4 * c, 4 * c),
        )
        feat_dim = 4 * c * 4 * 4
        self.proj = nn.Linear(feat_dim, context_dim)
        self.film1 = FiLMBlock(context_dim, cond_dim + 1)
        self.film2 = FiLMBlock(context_dim, cond_dim + 1)

    def forward(self, image: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res, psf = image[:, 0:1], image[:, 1:2]
        scale = res.flatten(1).std(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        image = torch.cat([res / scale, psf], dim=1)
        cond = torch.cat([cond, torch.log10(scale.flatten(1))], dim=-1)
        h = self.enc(image).flatten(1)
        h = F.gelu(self.proj(h))
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        return h


# ---------------------------------------------------------------------------
# Velocity U-Net (score.py UNet with t+ctx embedding)
# ---------------------------------------------------------------------------

class _VelocityUNet(UNet):
    """score.py UNet with the embedding recomputed from cat(sin_emb(t), ctx).

    The only change from the parent: emb_mlp now takes
    (emb_dim + ctx_dim) → emb_dim instead of emb_dim → emb_dim, and
    forward(x, t, ctx) replaces forward(x, c_noise).  All ResBlocks,
    skip connections, and up/downsamplers are reused unchanged.
    """

    def __init__(
        self,
        base: int = 32,
        mults: tuple[int, ...] = (1, 2, 2, 4),
        emb_dim: int = 128,
        ctx_dim: int = 256,
    ):
        super().__init__(in_ch=1, base=base, mults=mults, emb_dim=emb_dim)
        # Replace emb_mlp: input is now (sin_emb || ctx)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim + ctx_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        """x (B,1,H,W), t (B,) ∈ [0,1], ctx (B, ctx_dim) → velocity (B,1,H,W)."""
        t_emb = _sinusoidal_embedding(t.float(), self.emb_dim)     # (B, emb_dim)
        combined = torch.cat([t_emb, ctx], dim=-1)                  # (B, emb_dim+ctx_dim)
        emb = self.emb_mlp(combined)                                # (B, emb_dim)
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
        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PSFCondFlow(nn.Module):
    """Conditional flow matching model for single-source island deconvolution.

    Parameters
    ----------
    base      : encoder + velocity net base channel count.
    mults     : channel multipliers per resolution level.
    emb_dim   : sinusoidal time embedding width.
    ctx_dim   : context vector dimension.
    cond_dim  : FiLM conditioning dim (5: sigma_local + config one-hot).
    """

    def __init__(
        self,
        base: int = 32,
        mults: tuple[int, ...] = (1, 2, 2, 4),
        emb_dim: int = 128,
        ctx_dim: int = 256,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.encoder = _ContextEncoder(base, ctx_dim, cond_dim)
        self.velocity = _VelocityUNet(base, mults, emb_dim, ctx_dim)

    def forward(
        self,
        x_t: torch.Tensor,   # (B, 1, H, W)  noisy interpolant
        t: torch.Tensor,     # (B,)           flow time in [0, 1]
        image: torch.Tensor, # (B, 2, H, W)  (dirty, PSF)
        cond: torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:
        """Predict velocity field (B, 1, H, W)."""
        ctx = self.encoder(image, cond)
        return self.velocity(x_t, t, ctx)

    @staticmethod
    def _flux_scale(image: torch.Tensor) -> torch.Tensor:
        """Peak of the dirty channel (ch 0) as the per-sample flux scale.

        For a single point source the dirty peak ≈ source flux (PSF is
        peak-normalised), so dividing s0 by this puts the target in [0,~1].
        For extended sources it is a reasonable order-of-magnitude scale.
        Returned shape: (B, 1, 1, 1) for broadcasting.
        """
        return image[:, 0].amax(dim=(-2, -1)).abs().clamp_min(1e-10).view(-1, 1, 1, 1)

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,    # (B, 2, H, W) or (2, H, W) single obs
        cond: torch.Tensor,     # (B, cond_dim) or (cond_dim,)
        n_samples: int = 8,
        n_steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw posterior samples. Returns (B, n_samples, H, W) non-negative.

        The ODE runs in flux-normalised space (s / dirty_peak) so the signal
        is O(1) and matches the N(0,1) initial noise.  Output is rescaled to
        physical Jy/pixel before returning.

        For a single observation (B=1 or unbatched), the leading dim is
        squeezed: returns (n_samples, H, W).
        """
        squeeze = image.ndim == 3
        if squeeze:
            image = image.unsqueeze(0)
            cond = cond.unsqueeze(0)
        B, _, H, W = image.shape

        # Replicate obs for the sample batch.
        img_rep = image.repeat_interleave(n_samples, dim=0)    # (B*n, 2, H, W)
        cond_rep = cond.repeat_interleave(n_samples, dim=0)    # (B*n, cond_dim)
        ctx = self.encoder(img_rep, cond_rep)                   # (B*n, ctx_dim)

        # Flux scale: (B*n, 1, 1, 1) for rescaling output.
        fscale = self._flux_scale(img_rep)                      # (B*n, 1, 1, 1)

        x = torch.randn(
            B * n_samples, 1, H, W,
            device=image.device, dtype=image.dtype, generator=generator,
        )
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full(
                (B * n_samples,), i * dt,
                device=image.device, dtype=image.dtype,
            )
            x = x + dt * self.velocity(x, t_val, ctx)

        s = torch.relu(x) * fscale                             # rescale to Jy
        return s.view(B, n_samples, H, W).squeeze(0) if squeeze else s.view(B, n_samples, H, W)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def cfm_loss(
    model: PSFCondFlow,
    s0: torch.Tensor,         # (B, 1, H, W) clean source image
    image: torch.Tensor,      # (B, 2, H, W) (dirty, PSF)
    cond: torch.Tensor,       # (B, cond_dim)
    pixel_weight_lambda: float = 10.0,
    weight_clip: float = 1e3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pixel-weighted CFM MSE loss.

    Linear interpolant: x_t = t*s0 + (1-t)*eps, target velocity: u = s0 - eps.

    pixel_weight_lambda: the source-pixel boost factor.  Source pixels carry
    ~0.1% of the image area for a point source, so without weighting the loss
    is dominated by learning to "remove noise from the empty background."  With
    lambda=10, source pixels get up to (1 + lambda*s_max/s_mean) x weight.
    Clip prevents a single hot pixel from dominating.
    """
    # Normalise s0 to O(1) amplitudes using the dirty image peak so that the
    # flow noise eps~N(0,1) and the target s0_norm are on the same scale.
    # Without this, s0 ~ 1e-2 Jy vs eps ~ O(1) creates a 100x amplitude
    # mismatch: the model must collapse near-unit noise to tiny residuals,
    # which the MSE loss cannot guide at all.
    fscale = PSFCondFlow._flux_scale(image)     # (B, 1, 1, 1)
    s0_norm = s0 / fscale                       # peak ~ 1 for a point source

    B = s0.shape[0]
    t = torch.rand(B, device=s0.device, generator=generator)
    eps = torch.randn(s0.shape, device=s0.device, dtype=s0.dtype, generator=generator)
    t_ = t.view(-1, 1, 1, 1)
    x_t = t_ * s0_norm + (1.0 - t_) * eps
    u_target = s0_norm - eps

    pred = model(x_t, t, image, cond)

    # Flux-proportional pixel weight (same lever as edm_loss pixel_weight).
    s_mean = s0_norm.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    w = (1.0 + pixel_weight_lambda * s0_norm / s_mean).clamp(max=weight_clip)

    return (w * (pred - u_target) ** 2).mean()
