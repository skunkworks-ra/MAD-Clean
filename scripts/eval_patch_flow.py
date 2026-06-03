"""Eval diagnostic for a trained PatchFlow checkpoint.

Loads a saved checkpoint, samples a small fixed batch, and writes:
  - triptychs.png  (dirty | clean target | flow output, one row per sample)
  - metrics.json   (peak_flux_err, mae per sample)

Usage:
    pixi run -e gpu python scripts/eval_patch_flow.py \
        --checkpoint results/patch_flow_v1/best.pt \
        [--n_samples 8] [--n_steps 50] [--out_dir results/eval_patch_flow]
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mad_clean.data.patch_flow_dataset import PatchFlowDataset
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.patch_flow import PatchFlow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, default="results/patch_flow_v1/best.pt")
    p.add_argument("--n_samples",    type=int, default=8)
    p.add_argument("--n_steps",      type=int, default=50)
    p.add_argument("--base_channels",type=int, default=64)
    p.add_argument("--depth",        type=int, default=4)
    p.add_argument("--out_dir",      type=str, default="results/eval_patch_flow")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PatchFlow(base_channels=args.base_channels, depth=args.depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt['epoch']}  best_val={ckpt.get('best_val', 'n/a')})")

    # Fixed held-out samples (seed chosen to differ from train and val seeds)
    psf_bank = load_g55_psf_bank(_REPO_ROOT)
    ds = PatchFlowDataset(psf_bank=psf_bank, length=args.n_samples, rng_seed=99999)
    samples = [ds[i] for i in range(args.n_samples)]
    dirty = torch.stack([s[0] for s in samples]).to(device)
    psf   = torch.stack([s[1] for s in samples]).to(device)
    sigma = torch.stack([s[2] for s in samples]).to(device)
    clean = torch.stack([s[3] for s in samples]).to(device)

    # Sample
    with torch.no_grad():
        pred = model.sample(dirty, psf, sigma, n_steps=args.n_steps, device=device)

    # Metrics
    metrics = []
    for i in range(args.n_samples):
        c = clean[i, 0].cpu().numpy()
        p = pred[i, 0].cpu().numpy()
        peak_err = float(abs(p.max() - c.max()))
        mae      = float(np.abs(p - c).mean())
        metrics.append({"sample": i, "peak_flux_err": peak_err, "mae": mae})
        print(f"  sample {i}:  peak_flux_err={peak_err:.4f}  mae={mae:.6f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Triptychs
    fig, axes = plt.subplots(args.n_samples, 3, figsize=(9, 3 * args.n_samples))
    if args.n_samples == 1:
        axes = axes[None]
    for i in range(args.n_samples):
        d = dirty[i, 0].cpu().numpy()
        c = clean[i, 0].cpu().numpy()
        p = pred[i, 0].cpu().numpy()
        vmax = max(float(np.abs(c).max()), 1e-8)
        for ax, img, title in zip(axes[i], [d, c, p],
                                  ["dirty", "clean target", "flow output"]):
            ax.imshow(img, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.set_title(f"{title} (sample {i})", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "triptychs.png", dpi=120)
    plt.close()
    print(f"\nResults: {out_dir}/triptychs.png  {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
