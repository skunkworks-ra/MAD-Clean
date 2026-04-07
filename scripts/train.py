#!/usr/bin/env python3
"""
scripts/train.py — CLI for MAD-CLEAN dictionary / flow model training.

Variant A  (patch OMP):
    mad-train --variant A --data crumb_data/flow_pairs.npz \\
        --out models/cdl_filters_patch [--k 32] [--atom_size 15] [--device cpu]

Variant B  (convolutional CDL, PSF-residual):
    mad-train --variant B --data crumb_data/flow_pairs.npz \\
        --out models/cdl_filters_conv --k 32 --atom_size 15 [--device cuda]

Variant C  (conditional flow matching):
    mad-train --variant C --data crumb_data/flow_pairs.npz \\
        --out models/flow_model.pt [--resume models/flow_model.pt] [--device cuda]

Variant P  (unconditional prior — clean images only):
    mad-train --variant P --data crumb_data/flow_pairs_vla.npz \\
        --out models/prior_model.pt [--n_epochs 500] [--batch_size 16] [--device cuda]

The --data file must be a .npz produced by scripts/simulate.py, with keys:
    clean     (N, H, W) float32
    dirty     (N, H, W) float32
    psf       (H, W)    float32  [required for Variant B]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from mad_clean.training import PatchDictTrainer, ConvDictTrainer, FlowTrainer, PriorTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a MAD-CLEAN filter bank or flow model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",    required=True, choices=["A", "B", "C", "P"],
                   help="A=patch OMP, B=CDL PSF-residual, C=flow matching, P=unconditional prior")
    p.add_argument("--data",       required=True,
                   help="Path to .npz with clean/dirty/psf keys")
    p.add_argument("--out",        required=True,
                   help="Output path for FilterBank (.npz) or FlowModel (.pt)")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--k",          type=int,   default=32,  help="Number of atoms")
    p.add_argument("--atom_size",  type=int,   default=15,  help="Atom size in pixels")

    p.add_argument("--lmbda",           type=float, default=0.1,
                   help="[A/B] FISTA L1 sparsity penalty")
    p.add_argument("--n_epochs",        type=int,   default=50,
                   help="[A/B/C] Training epochs")
    p.add_argument("--fista_iter",      type=int,   default=100,
                   help="[A] FISTA iterations per Z-step")
    p.add_argument("--lr_d",            type=float, default=1e-3,
                   help="[A/B] Adam learning rate for D")
    p.add_argument("--patches_per_img", type=int,   default=50,
                   help="[A] Random patches extracted per training image")

    p.add_argument("--batch_size",       type=int,   default=8,
                   help="[B/C] Images per minibatch")
    p.add_argument("--fista_iter_train", type=int,   default=50,
                   help="[B] FISTA iterations per Z-step during training")

    p.add_argument("--lr",     type=float, default=1e-4,
                   help="[C] Adam learning rate for FlowTrainer")
    p.add_argument("--resume", default=None,
                   help="[C] Path to existing .pt checkpoint to resume from")

    return p


def main() -> None:
    args = build_parser().parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(data_path)

    if "clean" in data:
        clean = data["clean"].astype(np.float32)
        dirty = data["dirty"].astype(np.float32) if "dirty" in data else None
    elif "images" in data:
        clean = data["images"].astype(np.float32)
        dirty = None
    else:
        print("ERROR: npz must contain 'clean' or 'images' key", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(clean)} images  shape={clean.shape[1]}×{clean.shape[2]}")
    if dirty is not None:
        print(f"  dirty images available  shape={dirty.shape[1]}×{dirty.shape[2]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "A":
        trainer = PatchDictTrainer(
            k               = args.k,
            atom_size       = args.atom_size,
            lmbda           = args.lmbda,
            n_epochs        = args.n_epochs,
            fista_iter      = args.fista_iter,
            lr_d            = args.lr_d,
            patches_per_img = args.patches_per_img,
        )
        fb = trainer.fit(clean, device=args.device)
        trainer.save(out_path, fb)

    elif args.variant == "B":
        if dirty is None:
            print("ERROR: --variant B requires dirty images. Run scripts/simulate.py first.",
                  file=sys.stderr)
            sys.exit(1)
        if "psf_norm" not in data:
            print("ERROR: --variant B requires a 'psf_norm' key in the .npz data file.",
                  file=sys.stderr)
            print("       Re-run scripts/simulate.py to regenerate the data.", file=sys.stderr)
            sys.exit(1)
        psf = data["psf_norm"].astype(np.float32)
        trainer = ConvDictTrainer(
            k                = args.k,
            atom_size        = args.atom_size,
            batch_size       = args.batch_size,
            n_epochs         = args.n_epochs,
            lr_d             = args.lr_d,
            lmbda            = args.lmbda,
            fista_iter_train = args.fista_iter_train,
        )
        fb = trainer.fit(dirty, psf, device=args.device)
        trainer.save(out_path, fb)

    elif args.variant == "C":
        if dirty is None:
            print("ERROR: --variant C requires dirty images. Run scripts/simulate.py first.",
                  file=sys.stderr)
            sys.exit(1)
        trainer = FlowTrainer(
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
        )
        fm = trainer.fit(dirty, clean, device=args.device, resume_from=args.resume)
        fm.save(out_path)

    else:  # P
        trainer = PriorTrainer(
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
        )
        fm = trainer.fit(clean, device=args.device, resume_from=args.resume)
        fm.save(out_path)

    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
