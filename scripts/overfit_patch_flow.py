"""Overfit diagnostic for PatchFlow (gate 1).

Freezes N samples, trains PatchFlow to MSE floor, plots dirty/clean/output
triptychs. Gradient flow is real if loss descends and output patches match
the clean target visually.

Usage:
    pixi run -e gpu python scripts/overfit_patch_flow.py [--n_samples 8] \
        [--steps 2000] [--out_dir results/overfit_patch_flow]
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
import torch.optim as optim

from mad_clean.data.patch_flow_dataset import PatchFlowDataset
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.patch_flow import PatchFlow, cfm_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--steps",     type=int, default=2000)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--out_dir",   type=str, default="results/overfit_patch_flow")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_steps_sample", type=int, default=50,
                   help="Euler steps for sampling at eval time.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    psf_bank = load_g55_psf_bank(_REPO_ROOT)
    ds = PatchFlowDataset(psf_bank=psf_bank, length=args.n_samples, rng_seed=0)

    # Freeze N samples
    samples = [ds[i] for i in range(args.n_samples)]
    dirty  = torch.stack([s[0] for s in samples]).to(device)
    psf    = torch.stack([s[1] for s in samples]).to(device)
    sigma  = torch.stack([s[2] for s in samples]).to(device)
    clean  = torch.stack([s[3] for s in samples]).to(device)

    model = PatchFlow(base_channels=args.base_channels).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for step in range(args.steps):
        model.train()
        opt.zero_grad()
        loss = cfm_loss(model, dirty, psf, clean, sigma)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 200 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}")

    # Save loss curve
    plt.figure(figsize=(6, 3))
    plt.plot(losses)
    plt.xlabel("step"); plt.ylabel("CFM MSE loss")
    plt.title("Overfit patch flow -- loss curve")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close()

    # Sample and plot triptychs
    model.eval()
    with torch.no_grad():
        pred = model.sample(dirty, psf, sigma, n_steps=args.n_steps_sample, device=device)

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

    summary = {
        "final_loss": losses[-1],
        "min_loss":   min(losses),
        "n_samples":  args.n_samples,
        "steps":      args.steps,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfinal_loss={losses[-1]:.4f}  min_loss={min(losses):.4f}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
