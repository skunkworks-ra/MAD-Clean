"""Diagnostic 1 (flow_plan.md §Sanity checks): Overfit one frozen batch.

Freeze --n_samples cutouts; train MDNAsp on them until NLL floor.
This is a sanity check, NOT a pass/fail gate.  Overfit converging
tells us the loss and architecture are wired correctly; it does NOT
predict full-dataset performance (see lesson 5 in flow_plan.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for remote / SSH runs
import matplotlib.pyplot as plt  # noqa: E402

# Ensure repo root is on sys.path when script is invoked directly
# (pytest adds rootdir automatically; direct python invocation does not).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from mad_clean.data.cutout_dataset import (
    CutoutDataset,
    LOG_FLUX_SCALE,
    unstandardise_log_flux,
)
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.mdn_asp import MDNAsp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit one frozen batch — MDN-Asp sanity diagnostic 1.",
    )
    p.add_argument("--n_samples",  type=int,   default=8,
                   help="Number of frozen examples (default: 8).")
    p.add_argument("--steps",      type=int,   default=2000,
                   help="Training iterations (default: 2000).")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3).")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device (default: cuda if available, else cpu).")
    p.add_argument("--seed",       type=int,   default=0,
                   help="RNG seed (default: 0).")
    p.add_argument("--repo_root",  type=str,   default=".",
                   help="Repo root containing data/g55/chunk_*/psf.fits (default: .).")
    p.add_argument("--log_every",  type=int,   default=50,
                   help="Print loss every N steps (default: 50).")
    p.add_argument("--out_dir",    type=str,   default="results/overfit_one_batch",
                   help="Output directory for loss_curve.npy and summary.json.")
    # Small network knobs exposed for testing; sensible production defaults baked in.
    p.add_argument("--base_channels", type=int, default=32,
                   help="CNN base channel width (default: 32).")
    p.add_argument("--hidden",    type=int,   default=256,
                   help="MLP hidden width (default: 256).")
    p.add_argument("--extended_fraction", type=float, default=0.05,
                   help="Per-distractor probability of being extended (default 0.05).")
    p.add_argument("--morphologies", type=str, default="point,blob",
                   help="Comma-separated centred-source morphologies to sample from "
                        "(default: 'point,blob'). Choices: point, blob, shell, filament.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build PSF bank and dataset; pull frozen batch
    # ------------------------------------------------------------------
    print(f"[overfit] Loading PSF bank from {args.repo_root!r}/data/g55 ...")
    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root,
        target_size=128,
        rotation_augment=True,
    )
    print(f"[overfit] PSF bank size: {len(psf_bank)}")

    morph_keys = [m.strip() for m in args.morphologies.split(",")]
    morphology_balance = {m: 1.0 for m in morph_keys}
    print(f"[overfit] Morphology balance: {morphology_balance}")

    dataset = CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=128,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=args.seed,
        length=args.n_samples,
        morphology_balance=morphology_balance,
    )

    print(f"[overfit] Drawing {args.n_samples} frozen examples ...")
    residuals, psfs, conds, targets = [], [], [], []
    for i in range(args.n_samples):
        r, p, c, t = dataset[i]
        residuals.append(r)
        psfs.append(p)
        conds.append(c)
        targets.append(t)

    # Stack into (B, 1, H, W) then concat to (B, 2, H, W) — residual + PSF
    res_t  = torch.stack(residuals)              # (B, H, W)
    psf_t  = torch.stack(psfs)                   # (B, H, W)
    image  = torch.stack([res_t, psf_t], dim=1)  # (B, 2, H, W)
    cond   = torch.stack(conds)                  # (B, 5)
    target = torch.stack(targets)                # (B, 6)

    image  = image.to(device)
    cond   = cond.to(device)
    target = target.to(device)

    print(f"[overfit] Batch shapes: image={tuple(image.shape)}, "
          f"cond={tuple(cond.shape)}, target={tuple(target.shape)}")

    # ------------------------------------------------------------------
    # 2. Build MDNAsp — use args.base_channels / args.hidden for flexibility
    # ------------------------------------------------------------------
    model = MDNAsp(
        base_channels=args.base_channels,
        hidden=args.hidden,
        n_components=5,
        cond_dim=5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[overfit] Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 3. Train on the frozen batch
    # ------------------------------------------------------------------
    loss_curve = []
    best_loss = float("inf")
    best_state = None
    print(f"[overfit] Training for {args.steps} steps on {args.n_samples} frozen examples ...")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        params = model(image, cond)
        loss = model.nll_loss(params, target)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_curve.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  loss={loss_val:.4f}")

    # Restore best model state for evaluation
    model.load_state_dict(best_state)
    print(f"[overfit] Evaluating at best checkpoint (loss={best_loss:.4f})")

    # ------------------------------------------------------------------
    # 4. Per-example errors at mode
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        params = model(image, cond)
        mode   = model.mode(params)          # (B, 6)

    target_cpu = target.cpu().numpy()
    mode_cpu   = mode.cpu().numpy()

    # Dim names for the 6D target
    dim_names = ["x", "y", "log_flux", "log_sig_maj", "log_sig_min", "PA"]

    per_sample_errors = []
    print("\n[overfit] Per-example absolute errors at mode:")
    print(f"  {'idx':>3}  {'x':>7}  {'y':>7}  {'log_f':>7}  "
          f"{'log_smaj':>8}  {'log_smin':>8}  {'PA':>7}")

    for i in range(args.n_samples):
        t6 = target_cpu[i]
        m6 = mode_cpu[i]
        err = np.abs(m6 - t6)
        # log_flux error is in STANDARDISED units in the training space; convert
        # back to nats for human-readable reporting (multiply by SCALE — offset
        # cancels in a difference).
        err[2] = err[2] * LOG_FLUX_SCALE
        # PA period is pi under the (sin 2theta, cos 2theta) encoding —
        # theta and theta+pi map to the same point. decode_pa returns
        # (-pi/2, pi/2], so |delta| in [0, pi]; the wrap is min(|d|, pi-|d|).
        pa_raw = abs(float(m6[5]) - float(t6[5]))
        err[5] = min(pa_raw, math.pi - pa_raw)

        errd = {dim_names[d]: float(err[d]) for d in range(6)}
        per_sample_errors.append(errd)
        print(f"  {i:3d}  {err[0]:7.4f}  {err[1]:7.4f}  {err[2]:7.4f}  "
              f"{err[3]:8.4f}  {err[4]:8.4f}  {err[5]:7.4f}")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    loss_path = out_dir / "loss_curve.npy"
    np.save(loss_path, np.array(loss_curve, dtype=np.float32))
    print(f"\n[overfit] Loss curve saved to {loss_path}")

    summary = {
        "final_loss": loss_curve[-1],
        "min_loss":   min(loss_curve),
        "per_sample_errors": per_sample_errors,
        "hyperparams": {
            "n_samples":     args.n_samples,
            "steps":         args.steps,
            "lr":            args.lr,
            "seed":          args.seed,
            "base_channels": args.base_channels,
            "hidden":        args.hidden,
            "device":        args.device,
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[overfit] Summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # 6. Diagnostic plots
    # ------------------------------------------------------------------
    _save_plots(
        out_dir=out_dir,
        loss_curve=np.asarray(loss_curve, dtype=np.float32),
        image=image.cpu().numpy(),         # (B, 2, H, W)
        target=target_cpu,                  # (B, 6) standardised log_flux
        mode=mode_cpu,                      # (B, 6) standardised log_flux
        cutout_size=image.shape[-1],
    )
    print(f"[overfit] Diagnostic plots saved to {out_dir}/*.png")

    return summary


def _save_plots(
    out_dir: Path,
    loss_curve: np.ndarray,
    image: np.ndarray,
    target: np.ndarray,
    mode: np.ndarray,
    cutout_size: int,
) -> None:
    """Save three PNGs: loss curve, target-vs-mode scatter, residual cutouts grid."""

    # --- 1. Loss curve (linear + signed-log inset for spike visibility) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_curve, lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("NLL loss")
    ax.set_title(f"Overfit loss (final={loss_curve[-1]:.3f}, min={loss_curve.min():.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=110)
    plt.close(fig)

    # --- 2. Target vs predicted mode, per dim ---
    dim_labels = [
        ("x (px)",          target[:, 0],                          mode[:, 0]),
        ("y (px)",          target[:, 1],                          mode[:, 1]),
        ("log_flux (std)",  target[:, 2],                          mode[:, 2]),
        ("log_sig_maj",     target[:, 3],                          mode[:, 3]),
        ("log_sig_min",     target[:, 4],                          mode[:, 4]),
        ("PA (rad)",        target[:, 5],                          mode[:, 5]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (label, t_vals, m_vals) in zip(axes.flat, dim_labels):
        lo = float(min(t_vals.min(), m_vals.min()))
        hi = float(max(t_vals.max(), m_vals.max()))
        pad = max(1e-3, 0.05 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
        ax.scatter(t_vals, m_vals, s=40, alpha=0.8)
        ax.set_xlabel(f"target {label}")
        ax.set_ylabel(f"predicted {label}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Target vs predicted mode (y=x is perfect recovery)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_target_vs_mode.png", dpi=110)
    plt.close(fig)

    # --- 3. Residual cutouts grid with target + predicted markers ---
    B = image.shape[0]
    cols = min(4, B)
    rows = int(np.ceil(B / cols))
    centre = cutout_size // 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for i in range(B):
        ax = axes[i // cols, i % cols]
        res = image[i, 0]  # channel 0 = residual cutout
        vmax = float(np.abs(res).max())
        ax.imshow(res, cmap="seismic", vmin=-vmax, vmax=vmax, origin="upper")
        # Target marker: (x, y) are pixel offsets from cutout centre
        tx = centre + float(target[i, 0])
        ty = centre + float(target[i, 1])
        mx = centre + float(mode[i, 0])
        my = centre + float(mode[i, 1])
        ax.plot(tx, ty, "+", color="lime",   ms=12, mew=2, label="target")
        ax.plot(mx, my, "x", color="yellow", ms=10, mew=2, label="mode")
        ax.set_title(
            f"idx {i}: log_flux_z t={target[i,2]:+.2f} m={mode[i,2]:+.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide unused panels
    for j in range(B, rows * cols):
        axes[j // cols, j % cols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Residual cutouts (+ target, × predicted mode)")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_cutouts.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    run(args)
