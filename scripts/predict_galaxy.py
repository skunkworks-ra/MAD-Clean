#!/usr/bin/env python3
"""
Generate dirty_galaxy.npz for source 252, then run every solver and
plot the predicted clean model with individual colorbars.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ── 1. Load source 252 and VLA PSF ───────────────────────────────────────────

crumb = np.load("crumb_data/crumb_preprocessed.npz")
clean = crumb["images"][252].astype(np.float32)          # (150,150) Jy/pixel  [0,1]

psf = np.load("models/psf.npz")["psf"].astype(np.float32)  # peak=1, sum≈448

H, W = clean.shape
print(f"Source 252:  clean range=[{clean.min():.3f}, {clean.max():.3f}]")
print(f"PSF:         peak={psf.max():.4f}  sum={psf.sum():.2f}")

# ── 2. Simulate dirty (Jy/beam) ───────────────────────────────────────────────
#
#   dirty = psf(peak=1) ⊛ clean  +  noise
#
#   A point source at 1 Jy/pixel → dirty peak = 1 Jy/beam  (psf peak=1).
#   Extended emission is smoothed: flux spreads, peak drops.

psf_fft = np.fft.rfft2(np.fft.ifftshift(psf))
dirty   = np.fft.irfft2(np.fft.rfft2(clean) * psf_fft, s=(H, W)).astype(np.float32)
noise_std = 0.05
dirty  += np.random.default_rng(42).standard_normal(dirty.shape).astype(np.float32) * noise_std

print(f"Dirty:       range=[{dirty.min():.3f}, {dirty.max():.3f}]  noise_std={noise_std}")

torch.save({
    "clean"        : torch.from_numpy(clean),
    "dirty"        : torch.from_numpy(dirty),
    "psf"          : torch.from_numpy(psf),
    "psf_beam_area": torch.tensor(psf.sum()),
    "noise_std"    : torch.tensor(noise_std),
    "source_idx"   : torch.tensor(252),
}, "crumb_data/dirty_galaxy.pt")
print("Saved → crumb_data/dirty_galaxy.npz\n")

# ── 3. Run every solver ────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

from mad_clean import FilterBank, PatchSolver, ConvSolver, FlowSolver, DPSSolver
from mad_clean.training.flow import FlowModel

dirty_t = torch.from_numpy(dirty).to(device)
results = {}

print("── Variant A ──")
fb_a = FilterBank.load("models/cdl_filters_patch.npz", device=device)
results["A"] = PatchSolver(fb_a, n_nonzero=5, stride=8).decode_island(dirty_t).cpu().numpy()

print("── Variant B ──")
fb_b = FilterBank.load("models/cdl_filters_conv.npz", device=device)
results["B"] = ConvSolver(fb_b, lmbda=0.01, n_iter=100, psf=psf).decode_island(dirty_t).cpu().numpy()

print("── Variant C ──")
fm_c = FlowModel.load("models/flow_model.pt", device=device)
results["C"] = FlowSolver(fm_c, device=device, n_samples=1, n_steps=16).decode_island(dirty_t).cpu().numpy()

print("── DPS ──")
fm_p = FlowModel.load("models/prior_model.pt", device=device)
results["dps"] = DPSSolver(fm_p, psf=psf, noise_std=0.05, n_steps=50, n_samples=1, dps_weight=1.0, device=device).decode_island(dirty_t).cpu().numpy()

# ── 4. Plot ────────────────────────────────────────────────────────────────────

panels = [
    ("dirty (Jy/beam)",  dirty,          "Jy/beam"),
    ("truth (Jy/pixel)", clean,          "Jy/pixel"),
    ("A — patch",        results["A"],   "Jy/pixel"),
    ("B — conv",         results["B"],   "Jy/pixel"),
    ("C — flow",         results["C"],   "Jy/pixel"),
    ("DPS",              results["dps"], "Jy/pixel"),
]

fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 3.0, 3.8))

for ax, (name, img, unit) in zip(axes, panels):
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))
    im = ax.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(f"{name}\n[{vmin:.3f}, {vmax:.3f}]", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

fig.suptitle(f"Source 252 — model predictions  (noise_std={noise_std}, psf_beam_area={psf.sum():.0f})",
             fontsize=9, y=1.01)
plt.tight_layout()

Path("logs").mkdir(exist_ok=True)
out = "logs/source_252_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
