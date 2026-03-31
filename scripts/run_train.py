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
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# ── mad_clean import shim ─────────────────────────────────────────────────────
# Source files live flat at the project root and use `from mad_clean.xxx import`
# internally. This shim builds the mad_clean pseudo-package in sys.modules so
# imports resolve correctly without an editable install.

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "mad_clean" not in sys.modules:
    pkg = types.ModuleType("mad_clean")
    pkg.__path__    = [str(ROOT)]
    pkg.__package__ = "mad_clean"
    sys.modules["mad_clean"] = pkg

    def _load_local(key: str, filename: str):
        full_key = f"mad_clean.{key}"
        spec = importlib.util.spec_from_file_location(full_key, ROOT / filename)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[full_key] = mod
        spec.loader.exec_module(mod)
        return mod

    _mods = {
        "filters"    : "filters.py",
        "detection"  : "detection.py",
        "io"         : "io.py",
        "patch_dict" : "patch_dict.py",
        "conv_dict"  : "conv_dict.py",
        "flow_dict"  : "flow_dict.py",
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }
    for _name, _file in _mods.items():
        setattr(pkg, _name, _load_local(_name, _file))

    _train = types.ModuleType("mad_clean.train")
    _train.__package__ = "mad_clean.train"
    sys.modules["mad_clean.train"] = _train
    setattr(pkg, "train", _train)
    for _name in ("patch_dict", "conv_dict"):
        _mod = sys.modules[f"mad_clean.{_name}"]
        setattr(_train, _name, _mod)
        sys.modules[f"mad_clean.train.{_name}"] = _mod
# ─────────────────────────────────────────────────────────────────────────────

from patch_dict import PatchDictTrainer  # noqa: E402
from conv_dict import ConvDictTrainer    # noqa: E402
from flow_dict import FlowTrainer        # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a MAD-CLEAN filter bank (Variant A or B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant",    required=True, choices=["A", "B", "C"],
                   help="A = patch OMP (PatchDictTrainer), "
                        "B = CDL (ConvDictTrainer), "
                        "C = flow matching (FlowTrainer)")
    p.add_argument("--data",       required=True,
                   help="Path to .npz containing 'images' key (N, H, W) float32")
    p.add_argument("--out",        required=True,
                   help="Output path for FilterBank atoms (.npy)")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])

    # Shared
    p.add_argument("--k",          type=int,   default=32,   help="Number of atoms")
    p.add_argument("--atom_size",  type=int,   default=15,   help="Atom size in pixels")

    # Variant A specific
    p.add_argument("--lmbda",          type=float, default=0.1,
                   help="[A/B] FISTA L1 sparsity penalty")
    p.add_argument("--n_epochs",       type=int,   default=50,
                   help="[A/B] Alternating minimisation epochs")
    p.add_argument("--fista_iter",     type=int,   default=100,
                   help="[A] FISTA iterations per Z-step")
    p.add_argument("--lr_d",           type=float, default=1e-3,
                   help="[A/B] Adam learning rate for D")
    p.add_argument("--patches_per_img",type=int,   default=50,
                   help="[A] Random patches extracted per training image")

    # Variant B specific
    p.add_argument("--batch_size",       type=int,   default=8,
                   help="[B/C] Images per minibatch")
    p.add_argument("--fista_iter_train", type=int,   default=50,
                   help="[B] FISTA iterations per Z-step (fewer than inference)")

    # Variant C specific
    p.add_argument("--lr",               type=float, default=1e-4,
                   help="[C] Adam learning rate for FlowTrainer")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(data_path)

    # Unified npz format: may have 'images' (legacy clean-only) or 'clean'/'dirty' keys
    if "clean" in data:
        clean  = data["clean"].astype(np.float32)
        dirty  = data["dirty"].astype(np.float32) if "dirty" in data else None
        images = clean   # A/B use clean
    elif "images" in data:
        clean  = data["images"].astype(np.float32)
        dirty  = None
        images = clean
    else:
        print(f"ERROR: npz must contain 'clean' or 'images' key", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(clean)} images  shape={clean.shape[1]}×{clean.shape[2]}")
    if dirty is not None:
        print(f"  dirty images available  shape={dirty.shape[1]}×{dirty.shape[2]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── build trainer and train ───────────────────────────────────────────────
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
    elif args.variant == "B":
        trainer = ConvDictTrainer(
            k                = args.k,
            atom_size        = args.atom_size,
            batch_size       = args.batch_size,
            n_epochs         = args.n_epochs,
            lr_d             = args.lr_d,
            lmbda            = args.lmbda,
            fista_iter_train = args.fista_iter_train,
        )
    else:  # C
        trainer = FlowTrainer(
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
        )

    if args.variant == "C":
        if dirty is None:
            print("ERROR: --variant C requires dirty images. "
                  "Run simulate_observations.py first.", file=sys.stderr)
            sys.exit(1)
        fm = trainer.fit(dirty, clean, device=args.device)
        fm.save(out_path)
    else:
        fb = trainer.fit(images, device=args.device)
        trainer.save(out_path, fb)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
