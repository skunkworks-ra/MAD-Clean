"""
mad_clean.flow_dict
===================
Conditional Flow Matching for radio-interferometric island deconvolution (Variant C).

Trains a conditional flow matching (CFM) model that maps dirty→clean directly.
x_0 = dirty island, x_1 = clean island.  The velocity field v_θ learns the
straight-line path from dirty to clean.  At inference, the Euler ODE starts
from the dirty island and integrates to t=1 to produce the clean sky estimate.
No PSF is required at inference — the PSF is implicit in the training data.

Uncertainty (Component B — heteroscedastic head)
------------------------------------------------
UNetVelocityField outputs both a velocity field and a log-variance field from
two parallel 1×1 conv heads sharing all encoder/decoder features.  Training uses
the heteroscedastic NLL loss (Kendall & Gal 2017) so the model jointly learns
the mean direction and its confidence.  At inference, FlowModel.uncertainty_map()
evaluates at t=0 to return (μ_clean, σ_clean) per pixel in a single forward pass.

Classes
-------
UNetVelocityField
    Lightweight U-Net (~2.5M params).  Input: image + time embedding.
    Output: (velocity, log_var) — both (B, 1, H, W).

FlowModel
    Thin wrapper around UNetVelocityField.
    save(path)                       → writes state dict to .pt file
    load(path, device)               → class method, returns FlowModel
    velocity(x, t) → Tensor         → mean velocity v_θ(x, t)
    uncertainty_map(dirty) → (μ, σ) → per-pixel clean estimate + std, single pass

FlowTrainer
    fit(dirty, clean, device) → FlowModel
    Trains dirty→clean CFM with heteroscedastic NLL loss and 8-fold augmentation.
    Expects PSF-area-normalised inputs from SimulateObservations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_


__all__ = ["UNetVelocityField", "FlowModel", "FlowTrainer", "PriorTrainer"]


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Map scalar time values t ∈ [0,1] (shape B,) to sinusoidal embeddings (B, dim).

    Uses the same log-spaced frequency schedule as the original DDPM paper.
    Half the channels encode sin, the other half cos, so the embedding is
    injective and smooth across time.
    """
    assert dim % 2 == 0, "embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )                                          # (half,)
    angles = t[:, None].float() * freqs[None, :]  # (B, half)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Two 3×3 conv layers + GroupNorm + SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)  temb: (B, time_dim)
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(temb)[:, :, None, None]  # broadcast over spatial
        h = self.act(self.norm2(self.conv2(h)))
        return h


class _Downsample(nn.Module):
    """2× spatial downsample via strided conv (avoids aliasing vs max-pool)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    """2× learned upsample via transposed conv (avoids bilinear blur artefacts)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# C-1: UNetVelocityField
# ---------------------------------------------------------------------------

class UNetVelocityField(nn.Module):
    """
    Lightweight U-Net velocity field v_θ(x_t, t) for flow matching.

    Architecture
    ------------
    Input  : (B, 1, 150, 150) — noisy image at time t
    Output : (B, 1, 150, 150) — predicted velocity field

    Encoder : 150→75→37 (3 resolution levels, channels 1→32→64→128)
    Bottleneck : 37×37, 256 channels
    Decoder : 37→75→150 with skip connections from encoder (channels 128→64→32)
    Head    : 1×1 conv, 32→1

    Time conditioning: sinusoidal embedding dim=128, injected into every
    _ConvBlock via additive feature modulation.

    Parameter count: ~2.5M (within the 2–4M target).
    """

    TIME_DIM = 128

    def __init__(self):
        super().__init__()

        td = self.TIME_DIM

        # time MLP: sinusoidal → linear → SiLU → linear
        self.time_mlp = nn.Sequential(
            nn.Linear(td, td * 4),
            nn.SiLU(),
            nn.Linear(td * 4, td),
        )

        # Encoder
        self.enc1 = _ConvBlock(1,   32,  td)   # (B, 32,  150, 150)
        self.down1 = _Downsample(32)             # (B, 32,   75,  75)
        self.enc2 = _ConvBlock(32,  64,  td)   # (B, 64,   75,  75)
        self.down2 = _Downsample(64)             # (B, 64,   37,  37) [floor]

        # Bottleneck
        self.bot1  = _ConvBlock(64,  128, td)  # (B, 128,  37,  37)
        self.bot2  = _ConvBlock(128, 128, td)  # (B, 128,  37,  37)

        # Decoder — skip channels are concatenated before the conv block
        self.up2   = _Upsample(128)              # (B, 128,  74,  74) → pad to 75
        self.dec2  = _ConvBlock(128 + 64, 64, td)   # (B, 64,   75,  75)
        self.up1   = _Upsample(64)               # (B, 64,  150, 150)
        self.dec1  = _ConvBlock(64 + 32, 32, td)    # (B, 32,  150, 150)

        # Output projection — two parallel heads share all decoder features
        self.head     = nn.Conv2d(32, 1, 1)   # mean velocity
        self.var_head = nn.Conv2d(32, 1, 1)   # log-variance of velocity

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  — noisy image, H=W=150
        t : (B,)           — time values in [0, 1]

        Returns
        -------
        v       : (B, 1, H, W)  — mean velocity field
        log_var : (B, 1, H, W)  — log-variance of velocity (unclamped)
        """
        temb = _sinusoidal_embedding(t, self.TIME_DIM)
        temb = self.time_mlp(temb)  # (B, TIME_DIM)

        # Encoder
        e1   = self.enc1(x,           temb)   # (B, 32,  H,    W)
        e2   = self.enc2(self.down1(e1), temb) # (B, 64,  H//2, W//2)
        b    = self.bot1(self.down2(e2), temb) # (B, 128, H//4, W//4)  [floored]
        b    = self.bot2(b,           temb)

        # Decoder — bilinear upsample may differ from encoder size by 1 pixel;
        # interpolate skip to match before cat.
        d2   = self.up2(b)                                         # (B, 128, ?, ?)
        d2   = F_.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2   = self.dec2(torch.cat([d2, e2], dim=1), temb)        # (B, 64, H//2, W//2)

        d1   = self.up1(d2)                                        # (B, 64, ?, ?)
        d1   = F_.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1   = self.dec1(torch.cat([d1, e1], dim=1), temb)        # (B, 32, H, W)

        return self.head(d1), self.var_head(d1)                    # both (B, 1, H, W)


# ---------------------------------------------------------------------------
# C-2: FlowModel — thin wrapper
# ---------------------------------------------------------------------------

class FlowModel:
    """
    Wrapper around UNetVelocityField providing save/load/velocity interface.

    Usage
    -----
        fm = FlowModel(device="cuda")
        # ... train via FlowTrainer.fit() which calls fm._net ...
        fm.save("models/flow_model.pt")

        fm2 = FlowModel.load("models/flow_model.pt", device="cpu")
        v   = fm2.velocity(x_t, t)   # (B, 1, 150, 150)
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._net   = UNetVelocityField().to(self.device)

    # ------------------------------------------------------------------
    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate mean velocity v_θ(x, t).

        Parameters
        ----------
        x : (B, 1, H, W) float32 Tensor on any device
        t : (B,) float32 Tensor  — time values in [0, 1]

        Returns
        -------
        v : (B, 1, H, W) float32 Tensor on self.device
        """
        self._net.eval()
        x = x.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            v, _ = self._net(x, t)
            return v

    # ------------------------------------------------------------------
    def uncertainty_map(
        self, dirty: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-pixel (mean_clean, sigma_clean) in a single forward pass at t=0.

        Evaluates the model at t=0 where x_t = x_0 (the dirty image).
        The initial velocity uncertainty is a direct proxy for clean image
        uncertainty: the model's confidence about the direction dirty→clean
        at the starting point maps onto uncertainty in the final clean estimate.

        A single Euler step (μ = dirty + v) gives the mean clean estimate.
        σ = exp(0.5 * clamp(log_var, -10, 10)) gives the per-pixel std.

        Parameters
        ----------
        dirty : (B, 1, H, W) float32 Tensor

        Returns
        -------
        mu    : (B, 1, H, W) — mean clean estimate
        sigma : (B, 1, H, W) — per-pixel std of clean estimate
        """
        self._net.eval()
        dirty = dirty.to(self.device)
        t = torch.zeros(dirty.shape[0], device=self.device)
        with torch.no_grad():
            v, log_var = self._net(dirty, t)
            mu    = dirty + v
            sigma = (0.5 * log_var.clamp(-10, 10)).exp()
        return mu, sigma

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save network state dict and device string to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "device": str(self.device)},
            path,
        )
        print(f"FlowModel saved → {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "FlowModel":
        """
        Load a FlowModel from a .pt file.

        Parameters
        ----------
        path   : path to the .pt file written by FlowModel.save()
        device : override device (defaults to device stored in the file)
        """
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dev  = device if device is not None else ckpt.get("device", "cpu")
        fm   = cls(device=dev)
        missing, unexpected = fm._net.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        if missing:
            # Old deterministic checkpoint: var_head missing — random init.
            # Warm-start: velocity head loaded, var_head trains from scratch.
            print(f"FlowModel loaded ← {path}  (device={dev})  "
                  f"[warm-start: {len(missing)} keys missing — var_head randomly initialised]")
        else:
            print(f"FlowModel loaded ← {path}  (device={dev})")
        fm._net.eval()
        return fm

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self._net.parameters())
        return f"FlowModel(device={self.device}, params={n_params:,})"


# ---------------------------------------------------------------------------
# CR-2: FlowTrainer — conditional dirty→clean CFM
# ---------------------------------------------------------------------------

def _augment_pair(
    dirty: np.ndarray,
    clean: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same random D4 transform (4 rotations × optional flip) to a
    dirty/clean pair so spatial correspondence is preserved.
    """
    k    = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    dirty = np.rot90(dirty, k=k)
    clean = np.rot90(clean, k=k)
    if flip:
        dirty = np.fliplr(dirty)
        clean = np.fliplr(clean)
    return np.ascontiguousarray(dirty), np.ascontiguousarray(clean)


def _augment_single(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random D4 transform (4 rotations × optional flip) to a single image."""
    k    = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    img  = np.rot90(img, k=k)
    if flip:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


class FlowTrainer:
    """
    Train a FlowModel using Conditional Flow Matching (CFM), dirty→clean.

    Algorithm (per minibatch)
    -------------------------
    1. x0 = dirty image,  x1 = clean image  (naturally paired — no OT needed)
    2. Draw t ~ U[0,1]  (B,)
    3. Interpolate: x_t = (1 − t) · x0 + t · x1
    4. Target velocity: u = x1 − x0  (straight path from dirty to clean)
    5. CFM loss: ‖v_θ(x_t, t) − u‖²
    6. Adam step

    Normalisation: expects pre-normalised inputs (PSF-area normalisation applied
    by SimulateObservations). No additional normalisation is applied here.

    Parameters
    ----------
    n_epochs   : int    (default 100)
    batch_size : int    (default 8)
    lr         : float  Adam learning rate (default 1e-4)
    random_seed: int    (default 42)
    """

    def __init__(
        self,
        n_epochs   : int   = 100,
        batch_size : int   = 8,
        lr         : float = 1e-4,
        random_seed: int   = 42,
    ):
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_seed = random_seed

    def fit(
        self,
        dirty       : np.ndarray,
        clean       : np.ndarray,
        device      : str = "cpu",
        resume_from : Optional[str | Path] = None,
    ) -> FlowModel:
        """
        Train on paired (N, H, W) dirty and clean float32 images.

        Parameters
        ----------
        dirty       : np.ndarray (N, H, W) float32 — PSF-convolved + noise
        clean       : np.ndarray (N, H, W) float32 — ground truth sky
        device      : torch device string
        resume_from : path to an existing .pt checkpoint to resume from.
                      Loads weights into the model before training begins.
                      n_epochs additional epochs are run on top of the checkpoint.
        """
        assert dirty.shape == clean.shape, (
            f"dirty {dirty.shape} and clean {clean.shape} must match"
        )
        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        N, H, W = clean.shape

        if resume_from is not None:
            fm = FlowModel.load(resume_from, device=device)
            print(f"Resuming from {resume_from} — running {self.n_epochs} additional epochs")
        else:
            fm = FlowModel(device=device)
        optimizer = torch.optim.Adam(fm._net.parameters(), lr=self.lr)

        print(f"FlowTrainer (dirty→clean): N={N}  {H}×{W}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")

        for epoch in range(self.n_epochs):
            idx        = rng.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            fm._net.train()
            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]

                # 8-fold augmentation applied identically to dirty/clean pair
                pairs    = [_augment_pair(dirty[i], clean[i], rng) for i in batch_idx]
                dirty_np = np.stack([p[0] for p in pairs])   # (B, H, W)
                clean_np = np.stack([p[1] for p in pairs])   # (B, H, W)

                x0 = torch.from_numpy(dirty_np[:, None]).float().to(dev)  # (B,1,H,W)
                x1 = torch.from_numpy(clean_np[:, None]).float().to(dev)  # (B,1,H,W)
                B  = x0.shape[0]

                t  = torch.rand(B, device=dev)
                t4 = t[:, None, None, None]

                x_t      = (1.0 - t4) * x0 + t4 * x1   # dirty → clean interpolant
                u_target = x1 - x0                       # velocity: dirty → clean

                optimizer.zero_grad()
                v_pred, log_var = fm._net(x_t, t)
                log_var = log_var.clamp(-10, 10)
                # Heteroscedastic NLL (Kendall & Gal 2017):
                #   L = 0.5 * [ (v - u)² / exp(log_var) + log_var ]
                # exp(log_var) penalises overconfidence; log_var penalises underconfidence.
                # Optimal: log_var ≈ log((v_pred - u_target)²).
                loss = 0.5 * (
                    (v_pred - u_target) ** 2 / log_var.exp() + log_var
                ).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                n_batches  += 1

            mean_sigma = log_var.detach().mul(0.5).exp().mean().item()
            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}  "
                  f"mean_sigma={mean_sigma:.3f}", flush=True)

        fm._net.eval()
        return fm

    def __repr__(self) -> str:
        return (f"FlowTrainer(n_epochs={self.n_epochs}, "
                f"batch_size={self.batch_size}, lr={self.lr})")


# ---------------------------------------------------------------------------
# PR-1: PriorTrainer — unconditional CFM on clean images only
# ---------------------------------------------------------------------------

class PriorTrainer:
    """
    Train a FlowModel as an unconditional prior p(clean) using CFM.

    Unlike FlowTrainer (which maps dirty→clean), PriorTrainer maps noise→clean.
    The resulting model encodes source morphology without any PSF structure —
    it is fully instrument-agnostic and can be coupled to any PSF at inference
    via the likelihood gradient (see DPSSolver in solvers.py).

    Algorithm (per minibatch)
    -------------------------
    1. x0 = N(0, 1)  — pure Gaussian noise (no dirty image)
       x1 = clean image
    2. Draw t ~ U[0, 1]  (B,)
    3. Interpolate: x_t = (1 − t) · x0 + t · x1
    4. Target velocity: u = x1 − x0  (straight path from noise to clean)
    5. Heteroscedastic NLL loss (Kendall & Gal 2017):
           L = 0.5 * [ (v − u)² / exp(log_var) + log_var ]
    6. Adam step

    Parameters
    ----------
    n_epochs   : int    (default 500)
    batch_size : int    (default 16)
    lr         : float  Adam learning rate (default 1e-4)
    random_seed: int    (default 42)
    """

    def __init__(
        self,
        n_epochs   : int   = 500,
        batch_size : int   = 16,
        lr         : float = 1e-4,
        random_seed: int   = 42,
    ):
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_seed = random_seed

    def fit(
        self,
        clean       : np.ndarray,
        device      : str = "cpu",
        resume_from : Optional[str | Path] = None,
    ) -> FlowModel:
        """
        Train on (N, H, W) clean float32 images — no dirty images needed.

        Parameters
        ----------
        clean       : np.ndarray (N, H, W) float32 — ground truth sky images
        device      : torch device string
        resume_from : path to an existing .pt checkpoint to resume from.
                      Loads weights before training; n_epochs additional epochs run.
        """
        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        N, H, W = clean.shape

        if resume_from is not None:
            fm = FlowModel.load(resume_from, device=device)
            print(f"Resuming from {resume_from} — running {self.n_epochs} additional epochs")
        else:
            fm = FlowModel(device=device)
        optimizer = torch.optim.Adam(fm._net.parameters(), lr=self.lr)

        print(f"PriorTrainer (noise→clean): N={N}  {H}×{W}  "
              f"batch={self.batch_size}  epochs={self.n_epochs}  device={dev}")

        for epoch in range(self.n_epochs):
            idx        = rng.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            fm._net.train()
            for b_start in range(0, N, self.batch_size):
                batch_idx = idx[b_start : b_start + self.batch_size]

                clean_aug = [_augment_single(clean[i], rng) for i in batch_idx]
                clean_np  = np.stack(clean_aug)   # (B, H, W)

                x1 = torch.from_numpy(clean_np[:, None]).float().to(dev)  # (B,1,H,W)
                x0 = torch.randn_like(x1)                                   # pure noise
                B  = x1.shape[0]

                t  = torch.rand(B, device=dev)
                t4 = t[:, None, None, None]

                x_t      = (1.0 - t4) * x0 + t4 * x1   # noise → clean interpolant
                u_target = x1 - x0                       # velocity: noise → clean

                optimizer.zero_grad()
                v_pred, log_var = fm._net(x_t, t)
                log_var = log_var.clamp(-10, 10)
                loss = 0.5 * (
                    (v_pred - u_target) ** 2 / log_var.exp() + log_var
                ).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                n_batches  += 1

            mean_sigma = log_var.detach().mul(0.5).exp().mean().item()
            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}  "
                  f"mean_sigma={mean_sigma:.3f}", flush=True)

        fm._net.eval()
        return fm

    def __repr__(self) -> str:
        return (f"PriorTrainer(n_epochs={self.n_epochs}, "
                f"batch_size={self.batch_size}, lr={self.lr})")
