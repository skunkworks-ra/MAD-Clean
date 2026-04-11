"""
mad_clean.data.simulator
========================
GPUSimulator — on-the-fly (dirty, clean) pair generation entirely on GPU.

End goal: Simulation-Based Inference (SBI).
The simulator reproduces the physical forward model:

    dirty = PSF ⊛ clean + noise,   noise ~ N(0, noise_std²)

where dirty is in Jy/beam and clean is in Jy/pixel (raw physical units, no
normalisation).  Posterior p(clean | dirty) is only meaningful in physical
units — any per-image rescaling breaks SBI calibration.

Design
------
- All 181 MB of clean images are loaded to GPU VRAM once at init.
  No DataLoader, no CPU↔GPU transfers during training.
- D4 augmentation applied per-sample (8 independent draws per batch):
  torch.rot90 + torch.flip — exact, zero interpolation error.
- PSF augmented per-batch: continuous rotation θ ~ U[0°, 360°) + optional
  flip via torchvision.transforms.v2.functional.rotate.
- Convolution: padded FFT (pad to next power-of-2 ≥ 2×max(H,W)).
  ifftshift moves PSF peak to (0,0) → linear (not circular) convolution.
- Batch size derived from vram_budget_gb so the user controls memory use
  without hand-tuning.

Classes
-------
GPUSimulator
    __init__(data_path, psf, noise_std, vram_budget_gb, device)
    forward(clean_batch) → (dirty, clean_aug)   both (B, 1, H, W)
    generate_epoch(shuffle)  → Iterator[Tensor (B, H, W)]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import rotate as tv_rotate

__all__ = ["GPUSimulator"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    """Smallest power of 2 that is ≥ n."""
    return 1 << (n - 1).bit_length()


def _d4_group() -> list[tuple[int, bool]]:
    """All 8 elements of the D4 symmetry group as (k_rot90, do_flip) pairs."""
    return [(k, flip) for k in range(4) for flip in (False, True)]


# ---------------------------------------------------------------------------
# GPUSimulator
# ---------------------------------------------------------------------------

class GPUSimulator:
    """
    On-the-fly (dirty, clean) pair generator running entirely on GPU.

    Parameters
    ----------
    data_path     : path to crumb_preprocessed.npz with 'images' key (N, H, W)
    psf           : (H, W) float32 array or Tensor — peak=1 PSF
    noise_std     : Gaussian noise std in Jy/beam (same units as dirty)
    vram_budget_gb: GPU VRAM budget in GB for the resident dataset + batch workspace.
                    Batch size is derived automatically.
    device        : torch device string (default 'cuda')
    seed          : RNG seed for reproducibility (default 42)
    """

    def __init__(
        self,
        data_path     : str | Path,
        psf           : np.ndarray | torch.Tensor,
        noise_std     : float,
        vram_budget_gb: float = 8.0,
        device        : str   = "cuda",
        seed          : int   = 42,
    ):
        self.device    = torch.device(device)
        self.noise_std = noise_std
        self._rng      = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

        # --- load clean images to GPU ---
        # Accepts npz files with either 'images' key (crumb_preprocessed.npz)
        # or 'clean' key (flow_pairs_vla.npz).
        data_path = Path(data_path)
        raw = np.load(data_path)
        if "images" in raw:
            images = raw["images"].astype(np.float32)
        elif "clean" in raw:
            images = raw["clean"].astype(np.float32)
        else:
            raise KeyError(f"{data_path} must contain an 'images' or 'clean' key")
        self.N, self.H, self.W = images.shape
        self._clean = torch.from_numpy(images).to(self.device)  # (N, H, W)

        # --- PSF ---
        if isinstance(psf, np.ndarray):
            psf = torch.from_numpy(psf.astype(np.float32))
        self._psf = psf.to(self.device)              # (H, W)
        assert self._psf.shape == (self.H, self.W), (
            f"PSF shape {self._psf.shape} must match image shape ({self.H}, {self.W})"
        )

        # --- padding size for linear FFT convolution ---
        self._pad = _next_pow2(2 * max(self.H, self.W))

        # --- derive batch size from VRAM budget ---
        # Dataset footprint (bytes already on GPU):
        dataset_bytes = self._clean.element_size() * self._clean.numel()
        # Per-sample workspace during forward (complex FFT intermediate dominates):
        #   padded real: pad² × 4
        #   padded complex rfft2: pad × (pad//2+1) × 8
        #   dirty output: H × W × 4
        bytes_per_sample = (
            self._pad ** 2 * 4          # padded clean
            + self._pad * (self._pad // 2 + 1) * 8   # complex FFT
            + self.H * self.W * 4       # dirty output
        )
        budget_bytes     = int(vram_budget_gb * 1024 ** 3)
        available        = budget_bytes - dataset_bytes
        self.batch_size  = max(1, available // bytes_per_sample)
        print(
            f"GPUSimulator: N={self.N}  {self.H}×{self.W}  "
            f"pad={self._pad}  noise_std={noise_std}  device={self.device}\n"
            f"  dataset={dataset_bytes/1024**2:.1f} MB  "
            f"batch_size={self.batch_size} "
            f"(from {vram_budget_gb} GB budget)"
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        clean_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a (dirty, clean_aug) pair batch entirely on GPU.

        Parameters
        ----------
        clean_batch : (B, H, W) float32 Tensor on self.device — raw Jy/pixel

        Returns
        -------
        dirty     : (B, 1, H, W) float32 — Jy/beam, PSF-convolved + noise
        clean_aug : (B, 1, H, W) float32 — Jy/pixel, D4-augmented
        """
        B = clean_batch.shape[0]
        x = clean_batch.clone()   # (B, H, W) — do not modify the resident dataset

        # --- Step 1: D4 augmentation per sample ---
        # Draw k ~ {0,1,2,3} and flip ~ {0,1} independently for each sample.
        k_vals   = torch.randint(0, 4, (B,), generator=self._rng, device=self.device)
        fl_vals  = torch.randint(0, 2, (B,), generator=self._rng, device=self.device).bool()

        out = torch.empty_like(x)
        for k in range(4):
            for flip in (False, True):
                mask = (k_vals == k) & (fl_vals == flip)
                if not mask.any():
                    continue
                sub = x[mask]                          # (n_sub, H, W)
                sub = torch.rot90(sub, k=k, dims=[1, 2])
                if flip:
                    sub = torch.flip(sub, dims=[2])
                out[mask] = sub
        clean_aug = out[:, None, :, :]                 # (B, 1, H, W)

        # --- Step 2: PSF rotation + optional flip (one draw per batch) ---
        theta_deg = torch.rand(1, generator=self._rng, device=self.device).item() * 360.0
        psf_flip  = torch.randint(0, 2, (1,), generator=self._rng, device=self.device).bool().item()

        # torchvision rotate expects (C, H, W) or (B, C, H, W), float or uint8
        psf_4d = self._psf[None, None, :, :]           # (1, 1, H, W)
        psf_rot = tv_rotate(psf_4d, angle=theta_deg,
                            interpolation=2,            # BILINEAR
                            expand=False)               # (1, 1, H, W)
        if psf_flip:
            psf_rot = torch.flip(psf_rot, dims=[-1])
        psf_2d = psf_rot[0, 0]                         # (H, W)

        # --- Step 3: Padded FFT convolution (batched) ---
        p = self._pad

        # Pad clean_aug to (p, p)
        pad_h = p - self.H
        pad_w = p - self.W
        # F.pad order: (left, right, top, bottom)
        clean_pad = F.pad(clean_aug, (0, pad_w, 0, pad_h))   # (B, 1, p, p)

        # Pad and ifftshift PSF — move peak from centre to (0,0)
        psf_pad = F.pad(psf_2d[None, None], (0, pad_w, 0, pad_h))  # (1, 1, p, p)
        psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))

        # FFT multiply
        clean_fft = torch.fft.rfft2(clean_pad)    # (B, 1, p, p//2+1) complex
        psf_fft   = torch.fft.rfft2(psf_pad)      # (1, 1, p, p//2+1) complex
        dirty_fft = clean_fft * psf_fft            # broadcast over batch

        dirty_pad = torch.fft.irfft2(dirty_fft, s=(p, p))   # (B, 1, p, p)

        # Crop back to (H, W)
        dirty = dirty_pad[:, :, :self.H, :self.W]            # (B, 1, H, W)

        # --- Step 4: Add Gaussian noise ---
        dirty = dirty + torch.randn(
            B, 1, self.H, self.W,
            device=self.device,
            generator=self._rng,
        ) * self.noise_std

        return dirty, clean_aug

    # ------------------------------------------------------------------
    def generate_epoch(
        self,
        shuffle: bool = True,
    ) -> Iterator[torch.Tensor]:
        """
        Yield batches of clean images from the GPU-resident dataset.

        Parameters
        ----------
        shuffle : if True, shuffle order each epoch (default True)

        Yields
        ------
        Tensor (B, H, W) — raw Jy/pixel clean images on self.device
        """
        if shuffle:
            idx = torch.randperm(self.N, device=self.device, generator=self._rng)
        else:
            idx = torch.arange(self.N, device=self.device)

        for start in range(0, self.N, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self._clean[batch_idx]
