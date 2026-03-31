#!/usr/bin/env python3
"""
sweep_patch.py — Parameter sweep for PatchDictTrainer (PyTorch, GPU).

Trains a grid of (k, lmbda, patches_per_img) combinations and evaluates
each on held-out CRUMB images using PatchSolver reconstruction quality
(relative error). Reports ranked results so you can pick parameters
before committing to a full training run.

Usage:
    pixi run --environment gpu python scripts/sweep_patch.py --device cuda
    pixi run python scripts/sweep_patch.py --device cpu
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

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
        "solvers"    : "solvers.py",
        "deconvolver": "deconvolver.py",
    }.items():
        setattr(pkg, _name, _load(_name, _file))
# ─────────────────────────────────────────────────────────────────────────────

from patch_dict import PatchDictTrainer   # noqa: E402
from solvers    import PatchSolver        # noqa: E402


def held_out_images(images: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return the 20% held-out split — mirrors PatchDictTrainer's split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    return images[idx[int(0.8 * len(images)):]]


def reconstruction_rel_error(
    images : np.ndarray,
    fb,
    device : str,
) -> float:
    """
    Mean relative reconstruction error over images using PatchSolver.
    rel_err = RMS(original - reconstruction) / std(original)
    """
    solver = PatchSolver(fb)
    rel_errors = []
    for img in images:
        t     = torch.from_numpy(img).float().to(device)
        recon = solver.decode_island(t).cpu().numpy()
        mse   = float(np.mean((img - recon) ** 2))
        rel   = float(np.sqrt(mse) / (img.std() + 1e-8))
        rel_errors.append(rel)
    return float(np.mean(rel_errors))


def run_sweep(images: np.ndarray, args: argparse.Namespace) -> None:
    test_images = held_out_images(images, seed=42)
    # Cap eval set for speed
    rng     = np.random.default_rng(0)
    eval_idx = rng.choice(len(test_images),
                          size=min(args.n_eval, len(test_images)), replace=False)
    eval_images = test_images[eval_idx]
    print(f"Evaluation set: {len(eval_images)} held-out images\n")

    ks              = args.ks
    lambdas         = args.lambdas
    patches_per_img = args.patches_per_img

    total = len(ks) * len(lambdas) * len(patches_per_img)
    results = []
    run = 0

    for ppi in patches_per_img:
        for k in ks:
            for lmbda in lambdas:
                run += 1
                print(f"[{run}/{total}]  k={k}  lmbda={lmbda}  "
                      f"patches/img={ppi}  epochs={args.n_epochs} …")

                trainer = PatchDictTrainer(
                    k               = k,
                    atom_size       = args.atom_size,
                    lmbda           = lmbda,
                    n_epochs        = args.n_epochs,
                    fista_iter      = args.fista_iter,
                    lr_d            = args.lr_d,
                    patches_per_img = ppi,
                    random_seed     = 42,
                )
                fb = trainer.fit(images, device=args.device)

                rel_err = reconstruction_rel_error(eval_images, fb, args.device)
                results.append((k, lmbda, ppi, rel_err))
                print(f"  → rel_err={rel_err:.4f}\n")

    print("=" * 60)
    print("RANKED BY RECONSTRUCTION RELATIVE ERROR (lower is better):")
    print(f"{'rank':>4}  {'k':>5}  {'lmbda':>7}  {'ppi':>5}  {'rel_err':>9}")
    print("-" * 60)
    results.sort(key=lambda x: x[3])
    for rank, (k, lmbda, ppi, rel_err) in enumerate(results, 1):
        print(f"{rank:>4}  {k:>5}  {lmbda:>7.3f}  {ppi:>5}  {rel_err:>9.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep PatchDictTrainer hyperparameters, rank by held-out reconstruction error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      default="crumb_data/crumb_preprocessed.npz")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--atom_size", type=int,   default=15)
    p.add_argument("--n_epochs",  type=int,   default=100,
                   help="Epochs per sweep point — short enough to be fast, "
                        "long enough to show convergence direction")
    p.add_argument("--fista_iter",type=int,   default=100)
    p.add_argument("--lr_d",      type=float, default=1e-3)
    p.add_argument("--n_eval",    type=int,   default=50,
                   help="Number of held-out images to evaluate reconstruction on")
    p.add_argument("--ks",           type=int,   nargs="+", default=[64, 128, 256])
    p.add_argument("--lambdas",      type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    p.add_argument("--patches_per_img", type=int, nargs="+", default=[20, 50, 100])
    args = p.parse_args()

    data   = np.load(args.data)
    images = data["images"].astype(np.float32)
    print(f"Loaded {len(images)} images  shape={images.shape[1]}×{images.shape[2]}")
    print(f"Sweep: k={args.ks}  lmbda={args.lambdas}  ppi={args.patches_per_img}")
    print(f"       {len(args.ks) * len(args.lambdas) * len(args.patches_per_img)} "
          f"combinations × {args.n_epochs} epochs each\n")

    run_sweep(images, args)


if __name__ == "__main__":
    main()
