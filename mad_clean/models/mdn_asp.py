"""Mixture Density Network for 6D Aspen posteriors.

Input:   2-channel 128x128 image (residual cutout, PSF cutout).
Conditioning: (sigma_local, config_one_hot[4]) — length 5.
Output:  K=5 diagonal-Gaussian mixture over 6D target:
           (x, y, log_flux, log_sigma_maj, log_sigma_minor, PA)

PA is emitted internally as (sin 2*theta, cos 2*theta) — 7 scalars per
component — then decoded back to a scalar PA at sampling/mode.  This
collapses the PA <-> PA + pi ambiguity.

Architecture is a direct port of radiosharp/radiosharp/models/mdn.py with:
  - Output dimension raised from 4D to 6D (plus PA encoding -> 7 emitted).
  - Conditioning vector swapped: (sigma_local, config_one_hot) length 5,
    NOT the radiosharp (HA, length, dec) vector.
  - LOG_STD_MIN_PER_DIM extended to 7 dims; PA dims use the full -5 floor
    (PA uncertainty is inherently wide for symmetric sources).
  - Everything else (CNN encoder, FiLM blocks, logsumexp NLL) is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_COMPONENTS = 5

# 7 emitted scalars per component:
#   x, y, log_flux, log_sigma_maj, log_sigma_minor, sin2pa, cos2pa
N_EMITTED = 7

# 6D target space (PA decoded to scalar at output time)
N_DIM = 6

LOG_STD_MAX = 3.0

# Per-dim floors for the 7 emitted dims.
# log_sigma_maj and log_sigma_minor get the tighter -1.5 floor used in
# radiosharp for the scale dim (same reasoning: stop shattering into
# near-delta peaks).  PA encoding dims (sin2pa, cos2pa) use -5 — their
# natural range is [-1, 1] so the network has to be able to go narrow there.
LOG_STD_MIN_PER_EMITTED = (-5.0, -5.0, -5.0, -1.5, -1.5, -5.0, -5.0)

# Conditioning vector dimension: (sigma_local [1], config_one_hot [4])
COND_DIM = 5

# Number of VLA configurations encoded in the one-hot part
N_CONFIGS = 4  # A, B, C, D


# ---------------------------------------------------------------------------
# PA encoding / decoding
# ---------------------------------------------------------------------------

def encode_pa(pa_rad: torch.Tensor) -> torch.Tensor:
    """Encode scalar PA (radians) to (sin 2*theta, cos 2*theta).

    Input shape:  (...,)
    Output shape: (..., 2)
    """
    two_pa = 2.0 * pa_rad
    return torch.stack([torch.sin(two_pa), torch.cos(two_pa)], dim=-1)


def decode_pa(sin2pa: torch.Tensor, cos2pa: torch.Tensor) -> torch.Tensor:
    """Decode (sin 2*theta, cos 2*theta) back to scalar PA in (-pi/2, pi/2].

    Uses atan2 then halves the result.  Output is in (-pi/2, pi/2].

    Input shapes: (...,) each.
    Output shape: (...,)
    """
    return torch.atan2(sin2pa, cos2pa) * 0.5


# ---------------------------------------------------------------------------
# FiLM block (identical to radiosharp)
# ---------------------------------------------------------------------------

class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc(h)
        gb = self.film(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        return F.gelu(h)


# ---------------------------------------------------------------------------
# Mixture parameters container
# ---------------------------------------------------------------------------

class MixParams(NamedTuple):
    """Raw MDN outputs before PA decoding.

    logits:  (B, K)         — unnormalised mixture log-weights
    mu:      (B, K, 7)      — component means over 7 emitted dims
    log_std: (B, K, 7)      — component log-stds over 7 emitted dims
    """
    logits:  torch.Tensor
    mu:      torch.Tensor
    log_std: torch.Tensor


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MDNAsp(nn.Module):
    """MDN for 6D Aspen posteriors conditioned on (sigma_local, config_one_hot).

    Parameters
    ----------
    base_channels:
        Width multiplier for the CNN encoder.  Default 32 matches radiosharp.
        Use a smaller value (e.g. 8) for unit tests on CPU.
    hidden:
        MLP hidden width.  Default 256.  Use 64 for tests.
    n_components:
        Number of mixture components K.  Default 5 per plan.
    cond_dim:
        Conditioning vector length.  Default 5 (1 + 4).
    """

    def __init__(
        self,
        base_channels: int = 32,
        hidden: int = 256,
        n_components: int = K_COMPONENTS,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.K = n_components
        self.cond_dim = cond_dim

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

        # 2-channel input: (residual, PSF); 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc = nn.Sequential(
            block(2, c),          # 64
            block(c, 2 * c),      # 32
            block(2 * c, 4 * c),  # 16
            block(4 * c, 4 * c),  # 8
            block(4 * c, 4 * c),  # 4
        )

        # Flatten (not global avg pool) — preserves sub-pixel localisation
        # information (see radiosharp rationale).
        feat_dim = 4 * c * 4 * 4

        self.proj = nn.Linear(feat_dim, hidden)
        self.film1 = FiLMBlock(hidden, cond_dim)
        self.film2 = FiLMBlock(hidden, cond_dim)

        # K logits + K * N_EMITTED means + K * N_EMITTED log_stds
        self.head = nn.Linear(hidden, self.K * (1 + 2 * N_EMITTED))

        # Per-dim log_std floor tensor — registered as buffer so it moves
        # with .to(device) automatically.
        self.register_buffer(
            "_log_std_min",
            torch.tensor(LOG_STD_MIN_PER_EMITTED, dtype=torch.float32),
        )

    def forward(
        self,
        image: torch.Tensor,   # (B, 2, 128, 128)
        cond:  torch.Tensor,   # (B, cond_dim)
    ) -> MixParams:
        """Return mixture parameters over the 7-emitted-dim space."""
        B = image.shape[0]
        h = self.enc(image)
        h = h.flatten(1)
        h = F.gelu(self.proj(h))
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        out = self.head(h)

        K, E = self.K, N_EMITTED
        logits  = out[:, :K]
        mu      = out[:, K : K + K * E].view(B, K, E)
        log_std = out[:, K + K * E :].view(B, K, E)

        log_std = torch.maximum(log_std, self._log_std_min)
        log_std = log_std.clamp_max(LOG_STD_MAX)

        return MixParams(logits=logits, mu=mu, log_std=log_std)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nll_loss(
        self,
        params:  MixParams,
        targets: torch.Tensor,  # (B, 6): x, y, log_flux, log_smaj, log_smin, PA
    ) -> torch.Tensor:
        """NLL of targets under the predicted mixture.

        PA (dim 5 of targets) is encoded to (sin2pa, cos2pa) before the NLL
        is computed, so the loss operates in the 7D emitted space.
        Returns a scalar (mean over batch).
        """
        B = targets.shape[0]
        pa = targets[:, 5]                          # (B,)
        pa_enc = encode_pa(pa)                      # (B, 2)
        # Rebuild target in 7D emitted space
        y7 = torch.cat([targets[:, :5], pa_enc], dim=-1)  # (B, 7)

        logits, mu, log_std = params

        y7_exp = y7.unsqueeze(1)                    # (B, 1, 7)
        var = torch.exp(2.0 * log_std)              # (B, K, 7)
        log_comp = -0.5 * (
            ((y7_exp - mu) ** 2) / var
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_comp = log_comp.sum(dim=-1)             # (B, K)
        log_w = F.log_softmax(logits, dim=-1)       # (B, K)
        log_p = torch.logsumexp(log_w + log_comp, dim=-1)  # (B,)
        return -log_p.mean()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mode(self, params: MixParams) -> torch.Tensor:
        """Argmax-component mean decoded to 6D.  Returns (B, 6)."""
        logits, mu, _ = params
        k = logits.argmax(dim=-1)                       # (B,)
        idx = k.view(-1, 1, 1).expand(-1, 1, N_EMITTED)
        mu_mode = mu.gather(1, idx).squeeze(1)          # (B, 7)
        return self._decode_7d_to_6d(mu_mode)

    @torch.no_grad()
    def sample(self, params: MixParams, n: int = 1) -> torch.Tensor:
        """Draw samples from the mixture.  Returns (B, n, 6)."""
        logits, mu, log_std = params
        B, K, E = mu.shape
        log_w = F.log_softmax(logits, dim=-1)
        probs = log_w.exp()
        comp = torch.multinomial(probs, n, replacement=True)  # (B, n)
        idx  = comp.unsqueeze(-1).expand(-1, -1, E)           # (B, n, E)
        mu_sel  = mu.gather(1, idx)                           # (B, n, E)
        std_sel = log_std.exp().gather(1, idx)
        eps = torch.randn_like(mu_sel)
        raw = mu_sel + std_sel * eps                          # (B, n, 7)
        # Decode each sample
        out = torch.stack(
            [self._decode_7d_to_6d(raw[:, i, :]) for i in range(n)],
            dim=1,
        )
        return out  # (B, n, 6)

    # ------------------------------------------------------------------
    # Set-prediction loss (Hungarian-matched NLL)
    # ------------------------------------------------------------------

    def set_nll_loss(
        self,
        params:      MixParams,
        targets:     torch.Tensor,  # (B, K_max, 6)
        target_mask: torch.Tensor,  # (B, K_max) bool
    ) -> torch.Tensor:
        """Set-prediction NLL: Hungarian-match K components to N≤K_max true
        sources per cutout; sum NLL over matched pairs; mean over batch
        elements that have at least one true source.

        Cutouts with zero true sources contribute nothing — the network is
        free to put weight anywhere when there's nothing to predict.
        """
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        B, K_max, _ = targets.shape
        logits, mu, log_std = params
        K = mu.shape[1]
        if K_max > K:
            raise ValueError(
                f"targets pad K_max={K_max} > model K={K}; cannot match"
            )

        # Encode PA into 7-emitted-dim space, broadcast targets and components
        pa = targets[:, :, 5]                          # (B, K_max)
        pa_enc = encode_pa(pa)                         # (B, K_max, 2)
        y7 = torch.cat([targets[:, :, :5], pa_enc], dim=-1)  # (B, K_max, 7)

        # log_w: log mixture weights (B, K). Not used in matching cost (we
        # match purely on per-component NLL of the target); used in the
        # weighted form? Keep matching cost = per-component Gaussian NLL.
        # (B, K, 1, 7) - (B, 1, K_max, 7)
        var = torch.exp(2.0 * log_std)                 # (B, K, 7)
        diff = mu.unsqueeze(2) - y7.unsqueeze(1)       # (B, K, K_max, 7)
        log_comp = -0.5 * (
            (diff ** 2) / var.unsqueeze(2)
            + 2.0 * log_std.unsqueeze(2)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)                                  # (B, K, K_max)

        # cost = -log_comp[k, j], lower = better fit
        cost = -log_comp

        losses = []
        cost_np = cost.detach().cpu().numpy()
        for b in range(B):
            mask_b = target_mask[b]
            n_b = int(mask_b.sum().item())
            if n_b == 0:
                continue
            # Submatrix: K rows, n_b columns. Hungarian picks n_b matched pairs.
            sub = cost_np[b, :, :n_b]                  # (K, n_b)
            row_ind, col_ind = linear_sum_assignment(sub)
            # Sum NLL over matched pairs (gradient flows through cost[...] which
            # is differentiable; the index arrays are constants).
            row_ind_t = torch.as_tensor(row_ind, dtype=torch.long, device=cost.device)
            col_ind_t = torch.as_tensor(col_ind, dtype=torch.long, device=cost.device)
            losses.append(cost[b, row_ind_t, col_ind_t].sum() / n_b)

        if not losses:
            # Whole batch had no true sources — return a zero with grad path
            return mu.sum() * 0.0

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Inference helper for set prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def all_modes(self, params: MixParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-component means decoded to 6D, plus mixture weights.

        Returns
        -------
        modes : (B, K, 6) — each component's mean in 6D.
        weights : (B, K)  — softmax of logits.
        """
        logits, mu, _ = params
        B, K, E = mu.shape
        flat = mu.reshape(B * K, E)
        out = self._decode_7d_to_6d(flat).reshape(B, K, 6)
        weights = F.softmax(logits, dim=-1)
        return out, weights

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_7d_to_6d(v: torch.Tensor) -> torch.Tensor:
        """Convert (B, 7) emitted to (B, 6) with PA decoded to scalar."""
        sin2pa = v[:, 5]
        cos2pa = v[:, 6]
        pa = decode_pa(sin2pa, cos2pa)
        return torch.cat([v[:, :5], pa.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# Convenience: build conditioning vector from raw inputs
# ---------------------------------------------------------------------------

def make_cond(
    sigma_local: torch.Tensor,   # (B,)
    config_idx:  torch.Tensor,   # (B,) int in {0, 1, 2, 3}
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the (B, 5) conditioning vector.

    sigma_local is not normalised here — caller is responsible.
    config_idx 0=A, 1=B, 2=C, 3=D.
    """
    B = sigma_local.shape[0]
    one_hot = F.one_hot(config_idx.long(), num_classes=N_CONFIGS).float()
    cond = torch.cat([sigma_local.unsqueeze(-1), one_hot], dim=-1)
    if device is not None:
        cond = cond.to(device)
    return cond
