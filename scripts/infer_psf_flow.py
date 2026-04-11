#!/usr/bin/env python3
"""
Run PSFFlowModel inference on source 252 and plot the result.

Usage
-----
    pixi run -e gpu python scripts/infer_psf_flow.py
    pixi run -e gpu python scripts/infer_psf_flow.py --model models/psf_flow_model.best.pt
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from mad_clean.training.psf_flow import PSFFlowModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="models/psf_flow_model.pt")
    p.add_argument("--galaxy",   default="crumb_data/dirty_galaxy.npz")
    p.add_argument("--n_steps",  type=int, default=32)
    p.add_argument("--device",   default="cuda")
    p.add_argument("--out",      default="logs/infer_psf_flow.png")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = PSFFlowModel.load(args.model, device=device)

    # Load galaxy
    data  = np.load(args.galaxy)
    dirty = torch.from_numpy(data["dirty"]).float()
    clean = torch.from_numpy(data["clean"]).float()
    psf   = torch.from_numpy(data["psf"]).float()

    print(f"dirty: [{dirty.min():.3f}, {dirty.max():.3f}]")
    print(f"clean: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"psf:   peak={psf.max():.4f}  sum={psf.sum():.2f}")

    # Inference
    pred = model.decode_island(dirty, psf, n_steps=args.n_steps).cpu().numpy()
    dirty_np = dirty.numpy()
    clean_np = clean.numpy()
    psf_np   = psf.numpy()

    print(f"pred:  [{pred.min():.3f}, {pred.max():.3f}]")

    # Plot
    panels = [
        ("dirty (Jy/beam)", dirty_np, "Jy/beam"),
        ("PSF",             psf_np,   ""),
        ("truth (Jy/pixel)",clean_np, "Jy/pixel"),
        ("prediction",      pred,     "Jy/pixel"),
        ("residual",        pred - clean_np, "Jy/pixel"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 3.2, 3.8))

    for ax, (name, img, unit) in zip(axes, panels):
        vmin = float(np.nanmin(img))
        vmax = float(np.nanmax(img))
        cmap = "RdBu_r" if "residual" in name else "inferno"
        if "residual" in name:
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\n[{vmin:.3f}, {vmax:.3f}]", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    flux_truth = float(np.clip(clean_np, 0, None).sum())
    flux_pred  = float(np.clip(pred,     0, None).sum())
    flux_rec   = flux_pred / (flux_truth + 1e-12)

    fig.suptitle(
        f"PSFFlowModel — source 252  |  flux_rec={flux_rec:.3f}  |  n_steps={args.n_steps}",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()

    Path(args.out).parent.mkdir(exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    print(f"flux_rec = {flux_rec:.3f}  (target ≈ 1.0)")


if __name__ == "__main__":
    main()
