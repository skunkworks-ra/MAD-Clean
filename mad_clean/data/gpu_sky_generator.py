"""On-GPU synthetic sky + dirty image generator.

Generates entire batches of (img, cond, sky) directly on the GPU.
No DataLoader, no workers, no CPU-GPU transfer bottleneck.

Morphologies (vectorised across the batch):
  point    -- delta function at a random pixel
  blob     -- 2-D Gaussian
  shell    -- thin ring
  filament -- rotated elongated Gaussian

Multiple sources per sample are accumulated before convolution.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from mad_clean.models.mdn_asp import COND_DIM


def _fft_convolve(sky: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Zero-padded FFT convolution. Both (B, H, W). Returns (B, H, W)."""
    B, H, W = sky.shape
    fH, fW = H + H - 1, W + W - 1
    sky_f = torch.fft.rfft2(sky, s=(fH, fW))
    psf_f = torch.fft.rfft2(psf, s=(fH, fW))
    out = torch.fft.irfft2(sky_f * psf_f, s=(fH, fW))
    y0, x0 = (fH - H) // 2, (fW - W) // 2
    return out[:, y0:y0 + H, x0:x0 + W]


class GPUSkyGenerator:
    """Generates synthetic (dirty, psf, cond, sky) batches on a CUDA device.

    Parameters
    ----------
    psf_npy : path to the (N_fields, H, W) PSF array.  Must already be
              peak-normalised and cropped to image_size.
    device   : torch device.
    image_size : spatial size (default 128).
    sigma_noise : RMS noise added to the dirty image.
    n_sources   : (min, max) sources per field sample.
    extended_fraction : fraction of sources that are non-point.
    """

    def __init__(
        self,
        psf_npy: str,
        device: torch.device,
        image_size: int = 128,
        sigma_noise: float = 1e-4,
        n_sources: tuple[int, int] = (1, 8),
        extended_fraction: float = 0.5,
    ):
        import numpy as np
        raw = np.load(psf_npy, mmap_mode="r")            # (N, H, W)
        self.psf_bank = torch.from_numpy(
            np.array(raw, dtype=np.float32)).to(device)   # (N, H, W)
        self.device        = device
        self.H = self.W    = image_size
        self.sigma_noise   = sigma_noise
        self.n_src_min, self.n_src_max = n_sources
        self.extended_frac = extended_fraction
        self.cond_dim      = COND_DIM

        # Coordinate grids, shared across calls
        ys = torch.arange(image_size, device=device).float()
        xs = torch.arange(image_size, device=device).float()
        # (1, 1, H, W) for broadcasting over (B, N_src, H, W)
        self._yg = ys.view(1, 1, image_size, 1).expand(1, 1, image_size, image_size)
        self._xg = xs.view(1, 1, 1, image_size).expand(1, 1, image_size, image_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (img, cond, sky) all on self.device.

        img  : (B, 2, H, W)  channels = [dirty, psf]
        cond : (B, COND_DIM)
        sky  : (B, H, W)
        """
        B, H, W, dev = batch_size, self.H, self.W, self.device

        # Random PSF per sample
        idx  = torch.randint(0, len(self.psf_bank), (B,), device=dev)
        psfs = self.psf_bank[idx]                         # (B, H, W)

        sky  = self._make_sky(B)                          # (B, H, W)
        dirty = _fft_convolve(sky, psfs)
        dirty = dirty + self.sigma_noise * torch.randn_like(dirty)

        img  = torch.stack([dirty, psfs], dim=1)          # (B, 2, H, W)
        cond = self._make_cond(dirty, B)                  # (B, COND_DIM)

        return img, cond, sky

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_cond(self, dirty: torch.Tensor, B: int) -> torch.Tensor:
        """sigma_local in slot 0; rest zeros (config one-hot not needed here)."""
        cond = torch.zeros(B, self.cond_dim, device=self.device)
        cond[:, 0] = dirty.flatten(1).std(dim=1)
        return cond

    def _make_sky(self, B: int) -> torch.Tensor:
        H, W, dev = self.H, self.W, self.device
        n_src = torch.randint(
            self.n_src_min, self.n_src_max + 1, (B,), device=dev)   # (B,)
        N = int(n_src.max().item())                                   # max sources

        # Random source centres (B, N)
        margin = 12
        cx = torch.randint(margin, W - margin, (B, N), device=dev).float()
        cy = torch.randint(margin, H - margin, (B, N), device=dev).float()

        # Random flux 1–30 mJy (normalised; sky_scale handles absolute level)
        flux = torch.rand(B, N, device=dev) * 0.029 + 0.001          # (B, N)

        # Morphology mask: 0=point, 1=blob, 2=shell, 3=filament
        is_extended = torch.rand(B, N, device=dev) < self.extended_frac
        morph = torch.where(
            is_extended,
            torch.randint(1, 4, (B, N), device=dev),
            torch.zeros(B, N, dtype=torch.long, device=dev),
        )                                                             # (B, N)

        # Source-valid mask: zero out slots beyond n_src[b]
        slot_idx = torch.arange(N, device=dev).view(1, N)            # (1, N)
        valid    = slot_idx < n_src.view(B, 1)                       # (B, N)

        # Coordinate grids: (1, 1, H, W) for broadcasting to (B, N, H, W)
        yg = self._yg[:, :, :H, :W]
        xg = self._xg[:, :, :H, :W]

        sky = torch.zeros(B, H, W, device=dev)

        # --- Point sources (morph == 0) ----------------------------------
        pm = (morph == 0) & valid                                     # (B, N)
        if pm.any():
            sky = self._add_points(sky, cx, cy, flux, pm)

        # --- Blobs (morph == 1) ------------------------------------------
        bm = (morph == 1) & valid
        if bm.any():
            sigma = torch.rand(B, N, device=dev) * 6 + 1.5           # 1.5–7.5 px
            sky   = self._add_gaussians(sky, cx, cy, flux, sigma, bm, xg, yg)

        # --- Shells (morph == 2) -----------------------------------------
        sm = (morph == 2) & valid
        if sm.any():
            r = torch.rand(B, N, device=dev) * 18 + 6                # 6–24 px
            w = torch.rand(B, N, device=dev) * 2 + 1                 # 1–3 px
            sky = self._add_shells(sky, cx, cy, flux, r, w, sm, xg, yg)

        # --- Filaments (morph == 3) --------------------------------------
        fm = (morph == 3) & valid
        if fm.any():
            length = torch.rand(B, N, device=dev) * 35 + 15          # 15–50 px
            width  = torch.rand(B, N, device=dev) * 2 + 1            # 1–3 px
            angle  = torch.rand(B, N, device=dev) * math.pi
            sky    = self._add_filaments(
                sky, cx, cy, flux, length, width, angle, fm, xg, yg)

        return sky.clamp(min=0)

    @staticmethod
    def _add_points(sky, cx, cy, flux, mask):
        B = sky.shape[0]
        for b in range(B):
            for n in range(cx.shape[1]):
                if mask[b, n]:
                    xi = int(cx[b, n].item())
                    yi = int(cy[b, n].item())
                    sky[b, yi, xi] = sky[b, yi, xi] + flux[b, n]
        return sky

    @staticmethod
    def _add_gaussians(sky, cx, cy, flux, sigma, mask, xg, yg):
        # Vectorised: (B, N, H, W) intermediate
        cx4  = cx.view(*cx.shape, 1, 1)
        cy4  = cy.view(*cy.shape, 1, 1)
        s4   = sigma.view(*sigma.shape, 1, 1)
        f4   = flux.view(*flux.shape, 1, 1)
        m4   = mask.view(*mask.shape, 1, 1).float()
        g    = torch.exp(-((xg - cx4)**2 + (yg - cy4)**2) / (2 * s4**2))
        peak = g.flatten(2).max(dim=2).values.clamp_min(1e-12).view(*g.shape[:2], 1, 1)
        sky  = sky + (f4 * g / peak * m4).sum(dim=1)
        return sky

    @staticmethod
    def _add_shells(sky, cx, cy, flux, r, w, mask, xg, yg):
        cx4 = cx.view(*cx.shape, 1, 1)
        cy4 = cy.view(*cy.shape, 1, 1)
        r4  = r.view(*r.shape, 1, 1)
        w4  = w.view(*w.shape, 1, 1)
        f4  = flux.view(*flux.shape, 1, 1)
        m4  = mask.view(*mask.shape, 1, 1).float()
        dist = ((xg - cx4)**2 + (yg - cy4)**2).sqrt()
        ring = torch.exp(-((dist - r4)**2) / (2 * w4**2))
        peak = ring.flatten(2).max(dim=2).values.clamp_min(1e-12).view(*ring.shape[:2], 1, 1)
        sky  = sky + (f4 * ring / peak * m4).sum(dim=1)
        return sky

    @staticmethod
    def _add_filaments(sky, cx, cy, flux, length, width, angle, mask, xg, yg):
        cx4 = cx.view(*cx.shape, 1, 1)
        cy4 = cy.view(*cy.shape, 1, 1)
        l4  = length.view(*length.shape, 1, 1)
        w4  = width.view(*width.shape, 1, 1)
        a4  = angle.view(*angle.shape, 1, 1)
        f4  = flux.view(*flux.shape, 1, 1)
        m4  = mask.view(*mask.shape, 1, 1).float()
        dx  = xg - cx4
        dy  = yg - cy4
        xr  =  dx * torch.cos(a4) + dy * torch.sin(a4)
        yr  = -dx * torch.sin(a4) + dy * torch.cos(a4)
        fil = torch.exp(-xr**2 / (2 * (l4 / 2)**2) - yr**2 / (2 * w4**2))
        peak = fil.flatten(2).max(dim=2).values.clamp_min(1e-12).view(*fil.shape[:2], 1, 1)
        sky  = sky + (f4 * fil / peak * m4).sum(dim=1)
        return sky
