#!/usr/bin/env python3
"""
visualize_atoms.py — Plot learned dictionary atoms or CRUMB sample images.

Usage:
    # Visualise FilterBank atoms
    python scripts/visualize_atoms.py --atoms models/cdl_filters_patch.npy
    python scripts/visualize_atoms.py --atoms models/cdl_filters_conv.npy --out atoms.png

    # Visualise CRUMB sample images (sqrt stretch, one row per class)
    python scripts/visualize_atoms.py --data crumb_data/crumb_preprocessed.npz
    python scripts/visualize_atoms.py --data crumb_data/crumb_preprocessed.npz --out crumb_sample.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_atoms(atoms: np.ndarray, title: str, out_path: Path | None) -> None:
    """
    Plot (K, F, F) atoms as a grid. Each atom is independently normalised
    to [0, 1] so structure is visible regardless of scale.
    """
    K, F, _ = atoms.shape
    ncols = 16
    nrows = (K + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 1.2, nrows * 1.2))
    fig.suptitle(title, fontsize=11, y=1.01)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.05, wspace=0.05)

    for i, atom in enumerate(atoms):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        # Symmetric normalisation around zero — atoms are signed
        vmax = np.abs(atom).max() + 1e-8
        ax.imshow(atom, cmap="RdBu_r", interpolation="nearest",
                  vmin=-vmax, vmax=vmax)
        ax.axis("off")

    # Hide any unused cells
    for i in range(K, nrows * ncols):
        fig.add_subplot(gs[i // ncols, i % ncols]).axis("off")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def plot_crumb_sample(
    data_path : Path,
    n_per_class : int = 4,
    seed : int = 7,
    out_path : Path | None = None,
) -> None:
    """Plot sample images from CRUMB, one row per class, sqrt-stretched."""
    data   = np.load(data_path)
    images = data["images"]
    labels = data["labels"]

    class_names = {0: "FR-I", 1: "FR-II", 2: "Hybrid"}
    rng = np.random.default_rng(seed)
    rows = sorted(class_names.keys())
    idx  = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=n_per_class, replace=False)
        for c in rows
    ])

    fig, axes = plt.subplots(len(rows), n_per_class,
                             figsize=(n_per_class * 3, len(rows) * 3))
    for ax, i in zip(axes.flat, idx):
        img       = images[i]
        stretched = np.sqrt(np.clip(img, 0, None))
        vmax      = np.percentile(stretched, 99.5)
        ax.imshow(stretched, cmap="hot", vmin=0, vmax=vmax, origin="lower")
        ax.set_title(f"{class_names[labels[i]]}  #{i}", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"{data_path.name} — sqrt stretch  ({len(images)} images)", fontsize=10)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualise FilterBank atoms or CRUMB sample images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--atoms", help="Path to .npy FilterBank file")
    group.add_argument("--data",  help="Path to CRUMB .npz to show sample images")
    p.add_argument("--out",          default=None, help="Save to file instead of displaying")
    p.add_argument("--n_per_class",  type=int, default=4,
                   help="[--data] Images per class in sample grid")
    p.add_argument("--seed",         type=int, default=7,
                   help="[--data] Random seed for sample selection")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else None

    if args.atoms:
        path = Path(args.atoms)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        atoms = np.load(path)
        if atoms.ndim == 2:
            K = atoms.shape[0]
            F = int(atoms.shape[1] ** 0.5)
            atoms = atoms.reshape(K, F, F)
        print(f"Loaded {atoms.shape[0]} atoms  size={atoms.shape[1]}×{atoms.shape[2]}")
        plot_atoms(atoms,
                   title=f"{path.name}  ({atoms.shape[0]} atoms, {atoms.shape[1]}px)",
                   out_path=out_path)

    else:
        path = Path(args.data)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        plot_crumb_sample(path, n_per_class=args.n_per_class,
                          seed=args.seed, out_path=out_path)


if __name__ == "__main__":
    main()
