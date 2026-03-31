#!/usr/bin/env python3
"""
reconstruct_crumb.py — Reconstruction test for trained FilterBank atoms.

Tests how well trained atoms can reconstruct held-out CRUMB images.
Uses the same 80/20 train/test split as training (seed=42).

Variant A: PatchSolver (OMP, handles per-patch normalisation internally)
Variant B: ConvSolver (FISTA, per-image normalisation applied here to match training)

Usage:
    python scripts/reconstruct_crumb.py --variant A --atoms models/cdl_filters_patch.npy \
        --data crumb_data/crumb_preprocessed.npz --out models/recon_A.png

    python scripts/reconstruct_crumb.py --variant B --atoms models/cdl_filters_conv.npy \
        --data crumb_data/crumb_preprocessed.npz --out models/recon_B.png
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
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }.items():
        setattr(pkg, _name, _load(_name, _file))
# ─────────────────────────────────────────────────────────────────────────────

import torch
from filters import FilterBank    # noqa: E402
from solvers  import PatchSolver, ConvSolver  # noqa: E402


def held_out_images(
    images : np.ndarray,
    n      : int,
    seed   : int = 42,
) -> np.ndarray:
    """Return n images from the held-out 20% test split (mirrors training split)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    test_idx = idx[int(0.8 * len(images)):]
    chosen   = rng.choice(test_idx, size=min(n, len(test_idx)), replace=False)
    return images[chosen]


def reconstruct_variant_a(
    images : np.ndarray,
    fb     : FilterBank,
    device : str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct images via PatchSolver (OMP). Returns (originals, reconstructions)."""
    solver = PatchSolver(fb)
    recons = []
    for img in images:
        t = torch.from_numpy(img).float().to(device)
        r = solver.decode_island(t)
        recons.append(r.cpu().numpy())
    return images, np.stack(recons)


def reconstruct_variant_b(
    images : np.ndarray,
    fb     : FilterBank,
    device : str,
    lmbda  : float,
    n_iter : int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct images via ConvSolver (FISTA).
    Applies per-image normalisation to match training, then maps back.
    """
    solver = ConvSolver(fb, lmbda=lmbda, n_iter=n_iter)
    recons = []
    for img in images:
        img_mean = img.mean()
        img_std  = img.std() + 1e-8
        img_n    = (img - img_mean) / img_std

        t     = torch.from_numpy(img_n).float().to(device)
        r_n   = solver.decode_island(t).cpu().numpy()

        # map reconstruction back to original image scale
        r = r_n * img_std + img_mean
        recons.append(r)
    return images, np.stack(recons)


def plot_reconstruction(
    originals : np.ndarray,
    recons    : np.ndarray,
    variant   : str,
    out_path  : Path | None,
) -> None:
    """Plot original | reconstruction | residual for each image, with MSE."""
    n    = len(originals)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Reconstruction", "Residual"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for i, (orig, recon) in enumerate(zip(originals, recons)):
        residual = orig - recon
        mse      = float(np.mean((orig - recon) ** 2))
        rel_err  = float(np.sqrt(mse) / (orig.std() + 1e-8))

        vmax_orig = np.percentile(np.sqrt(np.clip(orig, 0, None)), 99.5)

        # original — sqrt stretch
        axes[i, 0].imshow(np.sqrt(np.clip(orig,  0, None)),
                          cmap="hot", vmin=0, vmax=vmax_orig, origin="lower")
        axes[i, 0].set_ylabel(f"MSE={mse:.2e}\nrel={rel_err:.3f}", fontsize=8)

        # reconstruction — sqrt stretch, same scale
        axes[i, 1].imshow(np.sqrt(np.clip(recon, 0, None)),
                          cmap="hot", vmin=0, vmax=vmax_orig, origin="lower")

        # residual — symmetric, diverging
        rmax = np.abs(residual).max()
        axes[i, 2].imshow(residual, cmap="RdBu_r",
                          vmin=-rmax, vmax=rmax, origin="lower")

        for ax in axes[i]:
            ax.axis("off")

    # summary stats
    mse_all     = np.mean((originals - recons) ** 2, axis=(1, 2))
    rel_all     = np.sqrt(mse_all) / (originals.std(axis=(1, 2)) + 1e-8)
    fig.suptitle(
        f"Variant {variant} reconstruction  |  "
        f"mean MSE={mse_all.mean():.2e}  mean rel err={rel_all.mean():.3f}",
        fontsize=11,
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()

    print(f"\nReconstruction summary ({len(originals)} images):")
    print(f"  Mean MSE:     {mse_all.mean():.3e}  (±{mse_all.std():.3e})")
    print(f"  Mean rel err: {rel_all.mean():.3f}  (±{rel_all.std():.3f})")
    print(f"  Best  MSE:    {mse_all.min():.3e}")
    print(f"  Worst MSE:    {mse_all.max():.3e}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconstruction test for trained FilterBank atoms on held-out CRUMB images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",   required=True, choices=["A", "B"])
    p.add_argument("--atoms",     required=True, help="Path to .npy FilterBank")
    p.add_argument("--data",      required=True, help="Path to CRUMB .npz")
    p.add_argument("--n",         type=int, default=8,  help="Number of test images")
    p.add_argument("--device",    default="cpu",        help="cpu or cuda")
    p.add_argument("--out",       default=None,         help="Save plot to file")
    p.add_argument("--lmbda",     type=float, default=0.01,
                   help="[B] FISTA L1 penalty")
    p.add_argument("--n_iter",    type=int,   default=100,
                   help="[B] FISTA iterations at inference")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed — must match training split seed")
    args = p.parse_args()

    # ── load data and atoms ───────────────────────────────────────────────────
    data   = np.load(args.data)
    images = data["images"].astype(np.float32)
    print(f"Loaded {len(images)} images  shape={images.shape[1]}×{images.shape[2]}")

    test_images = held_out_images(images, n=args.n, seed=args.seed)
    print(f"Using {len(test_images)} held-out test images")

    fb = FilterBank.load(args.atoms, device=args.device)
    print(f"Loaded FilterBank  K={fb.K}  F={fb.F}  device={fb.device}")

    # ── reconstruct ───────────────────────────────────────────────────────────
    if args.variant == "A":
        originals, recons = reconstruct_variant_a(test_images, fb, args.device)
    else:
        originals, recons = reconstruct_variant_b(
            test_images, fb, args.device, args.lmbda, args.n_iter
        )

    # ── plot and report ───────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else None
    plot_reconstruction(originals, recons, args.variant, out_path)


if __name__ == "__main__":
    main()
