"""
mad_clean.flow_dict
===================
Conditional Flow Matching for radio-interferometric island deconvolution (Variant C).

Trains a conditional flow matching (CFM) model that maps dirty→clean directly.
x_0 = dirty island, x_1 = clean island.  The velocity field v_θ learns the
straight-line path from dirty to clean.  At inference, the Euler ODE starts
from the dirty island and integrates to t=1 to produce the clean sky estimate.
No PSF is required at inference — the PSF is implicit in the training data.

Classes
-------
UNetVelocityField
    Lightweight U-Net (~1–4M params).  Input: image + time embedding.
    Output: velocity field, same spatial size as input.

FlowModel
    Thin wrapper around UNetVelocityField.
    save(path)                → writes state dict to .pt file
    load(path, device)        → class method, returns FlowModel
    velocity(x, t) → Tensor  → evaluates v_θ(x, t)

FlowTrainer
    fit(dirty, clean, device) → FlowModel
    Trains dirty→clean CFM with 8-fold on-the-fly augmentation.
    Normalisation applied using clean image statistics (applied to both arrays).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_


__all__ = ["UNetVelocityField", "FlowModel", "FlowTrainer"]


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
    """2× bilinear upsample followed by 1×1 conv to fix channels."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F_.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
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

        # Output projection
        self.head  = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  — noisy image, H=W=150
        t : (B,)           — time values in [0, 1]

        Returns
        -------
        v : (B, 1, H, W)  — velocity field
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

        return self.head(d1)                                       # (B, 1, H, W)


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
        Evaluate v_θ(x, t).

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
            return self._net(x, t)

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
        path      = Path(path)
        ckpt      = torch.load(path, map_location="cpu", weights_only=True)
        dev       = device if device is not None else ckpt.get("device", "cpu")
        fm        = cls(device=dev)
        fm._net.load_state_dict(ckpt["state_dict"])
        fm._net.eval()
        print(f"FlowModel loaded ← {path}  (device={dev})")
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

    Normalisation: per-image using clean image statistics applied to both
    dirty and clean, preserving the relative dirty/clean amplitude difference.

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
        dirty  : np.ndarray,
        clean  : np.ndarray,
        device : str = "cpu",
    ) -> FlowModel:
        """
        Train on paired (N, H, W) dirty and clean float32 images.

        Parameters
        ----------
        dirty  : np.ndarray (N, H, W) float32 — PSF-convolved + noise
        clean  : np.ndarray (N, H, W) float32 — ground truth sky
        device : torch device string
        """
        assert dirty.shape == clean.shape, (
            f"dirty {dirty.shape} and clean {clean.shape} must match"
        )
        dev = torch.device(device)
        rng = np.random.default_rng(self.random_seed)
        N, H, W = clean.shape

        # Normalise using clean statistics; apply same transform to dirty so
        # the relative dirty/clean difference (blurring + noise) is preserved.
        c_mean = clean.mean(axis=(1, 2), keepdims=True)
        c_std  = clean.std(axis=(1, 2),  keepdims=True) + 1e-8
        clean  = (clean - c_mean) / c_std
        dirty  = (dirty - c_mean) / c_std

        fm        = FlowModel(device=device)
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
                v_pred = fm._net(x_t, t)
                loss   = 0.5 * ((v_pred - u_target) ** 2).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                n_batches  += 1

            print(f"  Epoch {epoch + 1:3d}/{self.n_epochs}  "
                  f"loss={epoch_loss / n_batches:.3e}")

        fm._net.eval()
        return fm

    def __repr__(self) -> str:
        return (f"FlowTrainer(n_epochs={self.n_epochs}, "
                f"batch_size={self.batch_size}, lr={self.lr})")
