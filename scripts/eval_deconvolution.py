#!/usr/bin/env python3
"""
eval_deconvolution.py — Deconvolution quality test for all three MAD-CLEAN variants.

Takes held-out dirty images from flow_pairs_vla.npz (PSF-convolved radio galaxies),
runs each trained solver, and compares the recovered image to the known clean ground truth.

Layout: one row per test image  ×  5 columns:
    Clean GT | Dirty (input) | A: Patch OMP | B: Conv+PSF | C: Flow ODE

Usage:
    pixi run -e gpu python scripts/eval_deconvolution.py \
        --data    crumb_data/flow_pairs_vla.npz \
        --atoms_a models/cdl_filters_patch.npz \
        --atoms_b models/cdl_filters_conv.npz \
        --flow    models/flow_model.pt \
        --n       5 \
        --device  cuda \
        --out     models/eval_deconvolution.png
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── mad_clean import shim ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "mad_clean" not in sys.modules:
    pkg = types.ModuleType("mad_clean")
    pkg.__path__    = [str(ROOT)]
    pkg.__package__ = "mad_clean"
    sys.modules["mad_clean"] = pkg

    def _load(key: str, filename: str):
        full_key = f"mad_clean.{key}"
        spec = importlib.util.spec_from_file_location(full_key, ROOT / filename)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[full_key] = mod
        spec.loader.exec_module(mod)
        return mod

    for _name, _file in {
        "filters"    : "filters.py",
        "detection"  : "detection.py",
        "io"         : "io.py",
        "patch_dict" : "patch_dict.py",
        "conv_dict"  : "conv_dict.py",
        "flow_dict"  : "flow_dict.py",
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }.items():
        setattr(pkg, _name, _load(_name, _file))
# ─────────────────────────────────────────────────────────────────────────────

import torch
from filters  import FilterBank
from solvers  import PatchSolver, ConvSolver, FlowSolver
from flow_dict import FlowModel


# ── helpers ───────────────────────────────────────────────────────────────────

def _held_out_indices(n_total: int, seed: int = 42) -> np.ndarray:
    """Return indices of the held-out 20% test split — mirrors the training split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return idx[int(0.8 * n_total):]


def _normalise_pairs(
    clean: np.ndarray,
    dirty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-image normalisation using clean image statistics — matches FlowTrainer.fit().
    Both clean and dirty are shifted/scaled by the same (mean, std) so the
    relative dirty/clean amplitude difference is preserved.
    """
    c_mean = clean.mean(axis=(1, 2), keepdims=True)
    c_std  = clean.std(axis=(1, 2),  keepdims=True) + 1e-8
    return (clean - c_mean) / c_std, (dirty - c_mean) / c_std


def _central_indices(clean: np.ndarray, n: int, max_dist_px: float = 5.0) -> np.ndarray:
    """Return indices of images whose brightest pixel is within max_dist_px of centre."""
    cy, cx = clean.shape[1] // 2, clean.shape[2] // 2
    indices = []
    for i, img in enumerate(clean):
        py, px = np.unravel_index(np.argmax(img), img.shape)
        if np.sqrt((py - cy)**2 + (px - cx)**2) <= max_dist_px:
            indices.append(i)
    return np.array(indices[:n])


def _run_a(dirty: np.ndarray, fb: FilterBank, device: str) -> np.ndarray:
    """Variant A: PatchSolver (OMP). Normalisation is handled per-patch internally."""
    solver = PatchSolver(fb)
    results = []
    for img in dirty:
        t = torch.from_numpy(img).float().to(device)
        r = solver.decode_island(t)
        results.append(r.cpu().numpy())
    return np.stack(results)


def _run_b(
    dirty: np.ndarray,
    fb   : FilterBank,
    psf  : np.ndarray,
    device: str,
    lmbda : float = 0.01,
    n_iter: int   = 100,
) -> np.ndarray:
    """
    Variant B: ConvSolver with PSF-residual FISTA.
    dirty images come in already per-image normalised (from flow_pairs_vla.npz).
    lmbda=0.01 matches training (train_all.sh).
    """
    solver = ConvSolver(fb, lmbda=lmbda, n_iter=n_iter, psf=psf)
    results = []
    for img in dirty:
        t = torch.from_numpy(img).float().to(device)
        r = solver.decode_island(t)
        results.append(r.cpu().numpy())
    return np.stack(results)


def _run_c(
    dirty  : np.ndarray,
    fm     : FlowModel,
    device : str,
    n_steps: int = 32,
) -> np.ndarray:
    """
    Variant C: FlowSolver Euler ODE (dirty → clean).

    Normalisation note: flow_pairs_vla.npz data is already per-image normalised
    by simulate_observations.py. FlowTrainer.fit() applies a second normalisation
    whose effect is approximately identity (data is already ~zero-mean unit-std).
    We pass dirty directly — this matches the effective training distribution.

    n_steps=32 (doubled from default 16) for better ODE accuracy in evaluation.
    """
    solver = FlowSolver(fm, device=device, n_samples=1, n_steps=n_steps,
                        perturb_std=0.0)   # no perturbation — deterministic mean
    results = []
    for img in dirty:
        t = torch.from_numpy(img).float().to(device)
        r = solver.decode_island(t)
        results.append(r.cpu().numpy())
    return np.stack(results)


# ── plotting ─────────────────────────────────────────────────────────────────

def _mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def _plot(
    clean : np.ndarray,   # (N, H, W)
    dirty : np.ndarray,
    recon_a: np.ndarray,
    recon_b: np.ndarray,
    recon_c: np.ndarray,
    out_path: Path | None,
) -> None:
    N      = len(clean)
    cols   = ["Clean (GT)", "Dirty (input)", "A: Patch OMP", "B: Conv+PSF", "C: Flow ODE"]
    images = [clean, dirty, recon_a, recon_b, recon_c]

    fig, axes = plt.subplots(N, 5, figsize=(15, N * 3.2))
    if N == 1:
        axes = axes[np.newaxis, :]

    for col_j, title in enumerate(cols):
        axes[0, col_j].set_title(title, fontsize=10, fontweight="bold")

    for row_i in range(N):
        gt   = clean[row_i]
        # Use same linear colour scale across all panels in a row
        vmin = gt.min()
        vmax = np.percentile(gt, 99.5)

        for col_j, img_stack in enumerate(images):
            img = img_stack[row_i]
            ax  = axes[row_i, col_j]

            if col_j == 0:
                # Ground truth — reference scale
                label = f"GT  peak={gt.max():.2f}"
            elif col_j == 1:
                m = _mse(img, gt)
                label = f"MSE={m:.3e}"
            else:
                m = _mse(img, gt)
                label = f"MSE={m:.3e}"

            ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            ax.set_xlabel(label, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    # Print summary MSE table
    header = f"\n{'Variant':<20}  {'Mean MSE':>12}  {'Min MSE':>12}  {'Max MSE':>12}"
    print(header)
    print("-" * len(header))
    for label, stack in [("Dirty (baseline)", dirty),
                          ("A: Patch OMP",     recon_a),
                          ("B: Conv+PSF",       recon_b),
                          ("C: Flow ODE",       recon_c)]:
        mses = np.array([_mse(stack[i], clean[i]) for i in range(N)])
        print(f"  {label:<18}  {mses.mean():>12.4e}  {mses.min():>12.4e}  {mses.max():>12.4e}")

    plt.suptitle(
        f"MAD-CLEAN deconvolution evaluation  ({N} held-out images)\n"
        "All images shown on the same linear scale per row",
        fontsize=11,
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"\nSaved → {out_path}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Deconvolution quality test — all three MAD-CLEAN variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",    default="crumb_data/flow_pairs_vla.npz",
                   help="flow_pairs .npz with clean, dirty, psf keys")
    p.add_argument("--atoms_a", default="models/cdl_filters_patch.npz",
                   help="Variant A FilterBank (.npz)")
    p.add_argument("--atoms_b", default="models/cdl_filters_conv.npz",
                   help="Variant B FilterBank (.npz)")
    p.add_argument("--flow",    default="models/flow_model.pt",
                   help="Variant C FlowModel (.pt)")
    p.add_argument("--n",       type=int, default=5,
                   help="Number of held-out test images")
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out",     default="models/eval_deconvolution.png",
                   help="Output plot path (or 'none' to display interactively)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Train/test split seed — must match training")
    p.add_argument("--central", action="store_true",
                   help="Pick images whose brightest source is near the image centre")
    p.add_argument("--steps_c", type=int, default=32,
                   help="Euler ODE steps for Variant C")
    p.add_argument("--lmbda_b", type=float, default=0.01,
                   help="FISTA L1 penalty for Variant B — must match training")
    args = p.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    d     = np.load(args.data)
    clean = d["clean"].astype(np.float32)   # (N, H, W) already normalised
    dirty = d["dirty"].astype(np.float32)
    psf   = d["psf"].astype(np.float32)     # (H, W)
    N_total, H, W = clean.shape
    print(f"  {N_total} images  {H}×{W}  PSF shape {psf.shape}")

    # Normalise pairs using clean statistics — matches FlowTrainer.fit()
    clean, dirty = _normalise_pairs(clean, dirty)
    print(f"  After normalisation: clean std={clean.std():.3f}  dirty std={dirty.std():.3f}")

    # Select test images
    if args.central:
        chosen = _central_indices(clean, n=args.n)
        print(f"  Using {len(chosen)} central-source images (indices {chosen.tolist()})")
    else:
        test_idx = _held_out_indices(N_total, seed=args.seed)
        chosen   = test_idx[:args.n]
        print(f"  Using {len(chosen)} held-out test images (indices {chosen.tolist()})")

    clean_t  = clean[chosen]
    dirty_t  = dirty[chosen]

    # ── load models ───────────────────────────────────────────────────────────
    print(f"\nLoading Variant A atoms from {args.atoms_a} ...")
    fb_a = FilterBank.load(args.atoms_a, device=args.device)
    print(f"  {fb_a}")

    print(f"Loading Variant B atoms from {args.atoms_b} ...")
    fb_b = FilterBank.load(args.atoms_b, device=args.device)
    print(f"  {fb_b}")

    print(f"Loading Variant C flow model from {args.flow} ...")
    fm   = FlowModel.load(args.flow, device=args.device)
    print(f"  {fm}")

    # ── reconstruct ───────────────────────────────────────────────────────────
    print(f"\nRunning Variant A (PatchSolver) on {len(clean_t)} images ...")
    recon_a = _run_a(dirty_t, fb_a, args.device)

    print(f"Running Variant B (ConvSolver + PSF, lmbda={args.lmbda_b}) ...")
    recon_b = _run_b(dirty_t, fb_b, psf, args.device, lmbda=args.lmbda_b)

    print(f"Running Variant C (FlowSolver, {args.steps_c} ODE steps) ...")
    recon_c = _run_c(dirty_t, fm, args.device, n_steps=args.steps_c)

    # ── plot and report ───────────────────────────────────────────────────────
    out_path = None if args.out.lower() == "none" else Path(args.out)
    _plot(clean_t, dirty_t, recon_a, recon_b, recon_c, out_path)


if __name__ == "__main__":
    main()
