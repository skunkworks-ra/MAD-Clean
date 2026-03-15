#!/usr/bin/env python3
"""
run_train.py — CLI wrapper for MAD-CLEAN dictionary training.

Variant A  (patch OMP):
    python scripts/run_train.py --variant A \\
        --data crumb_data/crumb_preprocessed.npz \\
        --out models/cdl_filters_patch.npy \\
        [--k 32] [--atom_size 15] [--alpha 0.1] [--n_iter 1000] [--device cpu]

Variant B  (convolutional CDL):
    python scripts/run_train.py --variant B \\
        --data crumb_data/crumb_preprocessed.npz \\
        --out models/cdl_filters_conv.npy \\
        --k 32 --atom_size 15 \\
        [--batch_size 8] [--n_epochs 20] [--lr_d 1e-3] [--lmbda 0.1] [--device cuda]

The --data file must be a .npz with an 'images' key containing (N, H, W) float32 arrays.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch_dict import PatchDictTrainer  # noqa: E402
from conv_dict import ConvDictTrainer    # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a MAD-CLEAN filter bank (Variant A or B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",    required=True, choices=["A", "B"],
                   help="A = patch OMP (PatchDictTrainer), B = CDL (ConvDictTrainer)")
    p.add_argument("--data",       required=True,
                   help="Path to .npz containing 'images' key (N, H, W) float32")
    p.add_argument("--out",        required=True,
                   help="Output path for FilterBank atoms (.npy)")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])

    # Shared
    p.add_argument("--k",          type=int,   default=32,   help="Number of atoms")
    p.add_argument("--atom_size",  type=int,   default=15,   help="Atom size in pixels")

    # Variant A specific
    p.add_argument("--alpha",          type=float, default=0.1,
                   help="[A] sklearn sparsity regularisation")
    p.add_argument("--n_iter",         type=int,   default=1000,
                   help="[A] MiniBatchDictionaryLearning iterations")
    p.add_argument("--patches_per_img",type=int,   default=20,
                   help="[A] Random patches extracted per training image")

    # Variant B specific
    p.add_argument("--batch_size",       type=int,   default=8,
                   help="[B] Images per CDL minibatch")
    p.add_argument("--n_epochs",         type=int,   default=20,
                   help="[B] CDL training epochs")
    p.add_argument("--lr_d",             type=float, default=1e-3,
                   help="[B] Adam learning rate for D")
    p.add_argument("--lmbda",            type=float, default=0.1,
                   help="[B] FISTA L1 penalty (matches ConvSolver default)")
    p.add_argument("--fista_iter_train", type=int,   default=50,
                   help="[B] FISTA iterations per Z-step (fewer than inference)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    data   = np.load(data_path)
    images = data["images"].astype(np.float32)
    print(f"Loaded {len(images)} images  shape={images.shape[1]}×{images.shape[2]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── build trainer and train ───────────────────────────────────────────────
    if args.variant == "A":
        trainer = PatchDictTrainer(
            k               = args.k,
            atom_size       = args.atom_size,
            alpha           = args.alpha,
            n_iter          = args.n_iter,
            patches_per_img = args.patches_per_img,
        )
    else:
        trainer = ConvDictTrainer(
            k                = args.k,
            atom_size        = args.atom_size,
            batch_size       = args.batch_size,
            n_epochs         = args.n_epochs,
            lr_d             = args.lr_d,
            lmbda            = args.lmbda,
            fista_iter_train = args.fista_iter_train,
        )

    fb = trainer.fit(images, device=args.device)
    trainer.save(out_path, fb)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
