#!/usr/bin/env python3
"""
scripts/deconvolve.py — MAD-CLEAN deconvolution CLI.

Variants
--------
hogbom   Classical Hogbom CLEAN
A        Patch dictionary (PatchSolver)
B        Convolutional dictionary (ConvSolver)
C        Conditional flow matching (FlowSolver)
dps      DPS prior + explicit likelihood (DPSSolver)
latent   Latent VAE DPS (LatentDPSSolver)

Inputs are numpy arrays (.npy) or FITS files. PSF must have sum=1 (use
models/psf_norm from the simulate step, or pass --normalise_psf to auto-divide).

Usage
-----
mad-deconvolve --variant hogbom \\
    --dirty dirty.npy --psf models/psf_norm.npy --out_dir results/

mad-deconvolve --variant C \\
    --dirty dirty.npy --psf models/psf_norm.npy \\
    --model models/flow_model.pt --out_dir results/

mad-deconvolve --variant dps \\
    --dirty dirty.npy --psf models/psf_norm.npy \\
    --prior models/prior_model.pt --out_dir results/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from mad_clean.io import load_image_data, save_fits
from mad_clean import (
    HogbomDeconvolver, MADCleanDeconvolver,
    FilterBank, PatchSolver, ConvSolver,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MAD-CLEAN deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", required=True,
                   choices=["hogbom", "A", "B", "C", "dps", "latent"],
                   help="Deconvolver variant")
    p.add_argument("--dirty",   required=True, help="Dirty image (.npy or .fits)")
    p.add_argument("--psf",     required=True, help="PSF with sum=1 (.npy or .fits)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--device",  default=None,
                   help="cuda or cpu (default: auto-detect)")

    p.add_argument("--normalise_psf", action="store_true",
                   help="Auto-divide PSF by its sum before use")

    # Model paths
    p.add_argument("--atoms",        default=None, help="[A/B] FilterBank .npz")
    p.add_argument("--model",        default="models/flow_model.pt",
                   help="[C] FlowModel .pt")
    p.add_argument("--prior",        default="models/prior_model.pt",
                   help="[dps] PriorTrainer FlowModel .pt")
    p.add_argument("--vae",          default="models/vae_d128.pt",
                   help="[latent] VAEModel .pt")
    p.add_argument("--latent_prior", default="models/latent_prior_d128.pt",
                   help="[latent] LatentFlowModel .pt")

    # Loop parameters
    p.add_argument("--gain",            type=float, default=0.1)
    p.add_argument("--noise_threshold", type=float, default=None,
                   help="Stop when residual peak < this. Default: auto (3σ)")
    p.add_argument("--sidelobe_level",  type=float, default=None,
                   help="Minor→major cycle trigger. Default: computed from PSF")
    p.add_argument("--n_major_cycles",  type=int,   default=10)

    # Solver-specific
    p.add_argument("--n_nonzero",   type=int,   default=5,    help="[A] OMP sparsity")
    p.add_argument("--stride",      type=int,   default=8,    help="[A] patch stride")
    p.add_argument("--lmbda",       type=float, default=0.1,  help="[B] FISTA L1")
    p.add_argument("--fista_iter",  type=int,   default=100,  help="[B] FISTA iterations")
    p.add_argument("--n_steps",     type=int,   default=50,   help="[C/dps/latent] ODE steps")
    p.add_argument("--n_samples",   type=int,   default=8,    help="[dps/latent] posterior draws")
    p.add_argument("--dps_weight",  type=float, default=1.0,  help="[dps/latent] likelihood scale")
    p.add_argument("--noise_std",   type=float, default=0.05, help="[dps/latent] noise std")
    p.add_argument("--latent_dim",  type=int,   default=128,  help="[latent] VAE latent dim")

    return p


def main() -> None:
    args    = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    dirty   = load_image_data(args.dirty).astype(np.float32)
    psf_raw = load_image_data(args.psf).astype(np.float32)

    if args.normalise_psf:
        psf_raw = psf_raw / (psf_raw.sum() + 1e-12)
        print(f"PSF normalised  sum={psf_raw.sum():.4f}")
    else:
        s = psf_raw.sum()
        if abs(s - 1.0) > 0.01:
            print(f"WARNING: PSF sum={s:.4f} (expected 1.0). "
                  f"Pass --normalise_psf to auto-divide.", file=sys.stderr)

    print(f"Dirty  shape={dirty.shape}  "
          f"range=[{dirty.min():.4e}, {dirty.max():.4e}]")

    # ── Build solver ─────────────────────────────────────────────────────────
    solver = None

    if args.variant in ("A", "B"):
        if args.atoms is None:
            print("ERROR: --atoms required for variants A/B", file=sys.stderr)
            sys.exit(1)
        fb = FilterBank.load(args.atoms, device=args.device or "cpu")
        if args.variant == "A":
            solver = PatchSolver(fb, n_nonzero=args.n_nonzero, stride=args.stride)
        else:
            solver = ConvSolver(fb, lmbda=args.lmbda, n_iter=args.fista_iter,
                                psf=psf_raw)

    elif args.variant == "C":
        from mad_clean.training.flow import FlowModel
        from mad_clean.solvers import FlowSolver
        fm     = FlowModel.load(args.model, device=args.device or "cpu")
        solver = FlowSolver(fm, device=args.device or "cpu",
                            n_steps=args.n_steps, n_samples=1)

    elif args.variant == "dps":
        from mad_clean.training.flow import FlowModel
        from mad_clean.solvers import DPSSolver
        prior_fm = FlowModel.load(args.prior, device=args.device or "cpu")
        solver   = DPSSolver(
            prior_fm, psf_norm=psf_raw,
            noise_std=args.noise_std, n_steps=args.n_steps,
            n_samples=args.n_samples, dps_weight=args.dps_weight,
            device=args.device or "cpu",
        )

    elif args.variant == "latent":
        from mad_clean.training.vae import VAEModel
        from mad_clean.training.latent_flow import LatentFlowModel
        from mad_clean.solvers import LatentDPSSolver
        vae_model   = VAEModel.load(args.vae, device=args.device or "cpu")
        latent_flow = LatentFlowModel.load(args.latent_prior, device=args.device or "cpu")
        solver      = LatentDPSSolver(
            vae_model=vae_model, latent_flow=latent_flow,
            psf_norm=psf_raw, noise_std=args.noise_std,
            n_steps=args.n_steps, n_samples=args.n_samples,
            dps_weight=args.dps_weight, device=args.device or "cpu",
        )

    # ── Build deconvolver ────────────────────────────────────────────────────
    common = dict(
        psf_norm        = psf_raw,
        gain            = args.gain,
        noise_threshold = args.noise_threshold,
        sidelobe_level  = args.sidelobe_level,
        n_major_cycles  = args.n_major_cycles,
        device          = args.device,
    )

    if args.variant == "hogbom":
        deconvolver = HogbomDeconvolver(**common)
    else:
        deconvolver = MADCleanDeconvolver(solver=solver, **common)

    # ── Run ──────────────────────────────────────────────────────────────────
    result = deconvolver.deconvolve(dirty)

    print(f"\nDone  variant={args.variant}  "
          f"major_cycles={result['n_major']}  "
          f"converged={result['converged']}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    lbl = args.variant
    save_fits(result["model"],    out_dir / f"{lbl}_model.fits")
    save_fits(result["residual"], out_dir / f"{lbl}_residual.fits")
    if result["uncertainty"] is not None:
        save_fits(result["uncertainty"], out_dir / f"{lbl}_uncertainty.fits")
    print(f"Outputs → {out_dir}/")


if __name__ == "__main__":
    main()
