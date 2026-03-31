#!/usr/bin/env python3
"""
reconstruct_ood.py — Out-of-distribution reconstruction test.

Three modes:

  1. Full run — preprocess raw PNG, reconstruct, plot:
        python scripts/reconstruct_ood.py \\
            --image SNR_G55_10s.briggs.image.png \\
            --flux  models/SNR_G55.npy \\
            --atoms_a models/cdl_filters_patch.npy \\
            --atoms_b models/cdl_filters_conv.npy \\
            --out models/recon_ood_G55.png --device cuda

  2. Re-reconstruct (skip preprocessing, flux already cached):
        python scripts/reconstruct_ood.py \\
            --flux models/SNR_G55.npy \\
            --atoms_a models/cdl_filters_patch.npy \\
            --atoms_b models/cdl_filters_conv.npy \\
            --out models/recon_ood_G55.png --device cuda

  3. Re-plot only (skip preprocessing and reconstruction, all cached):
        python scripts/reconstruct_ood.py \\
            --flux models/SNR_G55.npy \\
            --out models/recon_ood_G55.png \\
            --vmin 0.01 --vmax 0.3

Caches:
    --flux     : (H, W) float32 .npy — preprocessed flux image
    --recon_a  : (H, W) float32 .npy — Variant A reconstruction  [auto-derived]
    --recon_b  : (H, W) float32 .npy — Variant B reconstruction  [auto-derived]

If cache files exist and the relevant --atoms are not provided, cached results
are loaded directly and reconstruction is skipped.
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _setup_mad_clean() -> None:
    """Load mad_clean modules into sys.modules — only called when torch is needed."""
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



# ── preprocessing ─────────────────────────────────────────────────────────────

def load_casa_image(image_path: Path) -> np.ndarray:
    """
    Read a CASA image table using casatools.
    Squeezes degenerate Stokes/Frequency axes → (H, W) float32.
    Requires casatools — available in the data-analyst pixi environment.
    """
    try:
        import casatools
    except ImportError:
        raise RuntimeError(
            "casatools not found. Run with the data-analyst pixi environment:\n"
            "  cd /home/pjaganna/Software/data-analyst && "
            "pixi run python /home/pjaganna/Software/MAD-clean/scripts/reconstruct_ood.py ..."
        )
    ia = casatools.image()
    ia.open(str(image_path))
    data = ia.getchunk()          # (Ra, Dec, Stokes, Freq) or similar
    ia.close()
    arr  = np.squeeze(data).astype(np.float32)   # → (H, W)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got shape {arr.shape}")
    return arr


def load_png_image(image_path: Path) -> np.ndarray:
    """
    Load a false-colour CASA PNG, crop axes border, convert to flux proxy.
    R - B captures blue=low, red=high colour scheme; normalised to [0, 1].
    """
    arr = np.array(Image.open(image_path).convert("RGB"))
    # crop white border
    white = (arr.sum(axis=2) >= 750)
    rows  = np.where(~white.all(axis=1))[0]
    cols  = np.where(~white.all(axis=0))[0]
    arr   = arr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    R     = arr[:, :, 0].astype(np.float32)
    B     = arr[:, :, 2].astype(np.float32)
    flux  = np.clip(R - B, 0, None)
    flux /= (flux.max() + 1e-8)
    return flux


def preprocess(image_path: Path, flux_path: Path) -> np.ndarray:
    """Load image (CASA table or PNG), cache flux as .npy."""
    suffix = image_path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        flux = load_png_image(image_path)
    else:
        # Assume CASA image table (no extension or .image)
        flux = load_casa_image(image_path)

    print(f"  Shape: {flux.shape[1]}×{flux.shape[0]} px")
    print(f"  Flux range: min={flux.min():.4e}  max={flux.max():.4e}  "
          f"mean={flux.mean():.4e}  std={flux.std():.4e}")
    np.save(flux_path, flux)
    print(f"  Cached flux → {flux_path}")
    return flux


# ── reconstruction ─────────────────────────────────────────────────────────────

def reconstruct_a(flux: np.ndarray, fb, device: str) -> np.ndarray:
    import torch
    from filters import FilterBank
    from solvers  import PatchSolver
    solver = PatchSolver(fb)
    t      = torch.from_numpy(flux).float().to(device)
    return solver.decode_island(t).cpu().numpy()


def reconstruct_b(flux: np.ndarray, fb, device: str, lmbda: float, n_iter: int) -> np.ndarray:
    import torch
    from solvers import ConvSolver
    solver   = ConvSolver(fb, lmbda=lmbda, n_iter=n_iter)
    img_mean = flux.mean()
    img_std  = flux.std() + 1e-8
    t        = torch.from_numpy((flux - img_mean) / img_std).float().to(device)
    r_n      = solver.decode_island(t).cpu().numpy()
    return r_n * img_std + img_mean


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(
    flux    : np.ndarray,
    recon_a : np.ndarray,
    recon_b : np.ndarray,
    title   : str,
    out_path: Path | None,
    vmin    : float | None,
    vmax    : float | None,
) -> None:
    print(f"\nImage stats (linear flux):")
    print(f"  min={flux.min():.4e}  max={flux.max():.4e}  "
          f"mean={flux.mean():.4e}  std={flux.std():.4e}")
    print(f"  p0.1={np.percentile(flux,0.1):.4e}  p1={np.percentile(flux,1):.4e}  "
          f"p99={np.percentile(flux,99):.4e}  p99.9={np.percentile(flux,99.9):.4e}")

    if vmin is None:
        vmin = float(np.percentile(flux, 0.5))
    if vmax is None:
        vmax = float(np.percentile(flux, 99.5))
    print(f"  Using vmin={vmin:.4e}  vmax={vmax:.4e}")

    def mse_rel(orig, recon):
        mse = float(np.mean((orig - recon) ** 2))
        return mse, float(np.sqrt(mse) / (orig.std() + 1e-8))

    mse_a, rel_a = mse_rel(flux, recon_a)
    mse_b, rel_b = mse_rel(flux, recon_b)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    rows = [
        ("Variant A  (Patch OMP)", recon_a, mse_a, rel_a),
        ("Variant B  (CDL FISTA)", recon_b, mse_b, rel_b),
    ]
    axes[0, 0].set_title("Original",       fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Reconstruction", fontsize=10, fontweight="bold")
    axes[0, 2].set_title("Residual",       fontsize=10, fontweight="bold")

    for i, (label, recon, mse, rel) in enumerate(rows):
        residual = flux - recon
        rmax     = float(np.abs(residual).max())
        axes[i, 0].imshow(flux,  cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[i, 0].set_ylabel(f"{label}\nMSE={mse:.2e}  rel={rel:.3f}", fontsize=9)
        axes[i, 1].imshow(recon, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[i, 2].imshow(residual, cmap="RdBu_r", vmin=-rmax, vmax=rmax, origin="lower")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"\nSaved → {out_path}")
    else:
        plt.show()

    print(f"\nVariant A  MSE={mse_a:.3e}  rel_err={rel_a:.3f}")
    print(f"Variant B  MSE={mse_b:.3e}  rel_err={rel_b:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="OOD reconstruction test. Caches flux and reconstructions so "
                    "re-plotting is instant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--flux",     required=True,
                   help="Path to cached flux .npy (created on first run)")
    p.add_argument("--image",    default=None,
                   help="Raw false-colour PNG — only needed to (re)build --flux cache")
    p.add_argument("--atoms_a",  default=None,
                   help="Variant A FilterBank .npy — only needed to (re)run reconstruction")
    p.add_argument("--atoms_b",  default=None,
                   help="Variant B FilterBank .npy — only needed to (re)run reconstruction")
    p.add_argument("--out",      default=None,  help="Output figure path")
    p.add_argument("--title",    default="Out-of-distribution reconstruction test\n"
                                         "(atoms trained on CRUMB FR-I/FR-II/Hybrid only)")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--lmbda",    type=float, default=0.01,  help="[B] FISTA L1 penalty")
    p.add_argument("--n_iter",   type=int,   default=100,   help="[B] FISTA iterations")
    p.add_argument("--vmin",            type=float, default=None,
                   help="Plot vmin linear scale (stats printed on every run)")
    p.add_argument("--vmax",            type=float, default=None,
                   help="Plot vmax linear scale (stats printed on every run)")
    p.add_argument("--preprocess_only", action="store_true",
                   help="Only preprocess --image → --flux cache, then exit. "
                        "Run this step with the data-analyst pixi env (has casatools).")
    args = p.parse_args()

    flux_path = Path(args.flux)

    # ── flux cache ─────────────────────────────────────────────────────────────
    if not flux_path.exists():
        if args.image is None:
            print(f"ERROR: {flux_path} not found and --image not provided.", file=sys.stderr)
            sys.exit(1)
        print("Preprocessing …")
        flux = preprocess(Path(args.image), flux_path)
    else:
        flux = np.load(flux_path)
        print(f"Loaded flux cache  {flux.shape[1]}×{flux.shape[0]} px  → {flux_path}")

    if args.preprocess_only:
        print("Preprocessing done. Re-run without --preprocess_only to reconstruct and plot.")
        sys.exit(0)

    # ── reconstruction caches ──────────────────────────────────────────────────
    cache_a = flux_path.with_name(flux_path.stem + "_recon_a.npy")
    cache_b = flux_path.with_name(flux_path.stem + "_recon_b.npy")

    if args.atoms_a or not cache_a.exists():
        if args.atoms_a is None:
            print(f"ERROR: {cache_a} not found and --atoms_a not provided.", file=sys.stderr)
            sys.exit(1)
        print("Reconstructing Variant A …")
        _setup_mad_clean()
        from filters import FilterBank
        fb_a    = FilterBank.load(args.atoms_a, device=args.device)
        recon_a = reconstruct_a(flux, fb_a, args.device)
        np.save(cache_a, recon_a)
        print(f"  Cached → {cache_a}")
    else:
        recon_a = np.load(cache_a)
        print(f"Loaded Variant A cache → {cache_a}")

    if args.atoms_b or not cache_b.exists():
        if args.atoms_b is None:
            print(f"ERROR: {cache_b} not found and --atoms_b not provided.", file=sys.stderr)
            sys.exit(1)
        print("Reconstructing Variant B …")
        _setup_mad_clean()
        from filters import FilterBank
        fb_b    = FilterBank.load(args.atoms_b, device=args.device)
        recon_b = reconstruct_b(flux, fb_b, args.device, args.lmbda, args.n_iter)
        np.save(cache_b, recon_b)
        print(f"  Cached → {cache_b}")
    else:
        recon_b = np.load(cache_b)
        print(f"Loaded Variant B cache → {cache_b}")

    # ── plot ───────────────────────────────────────────────────────────────────
    plot_comparison(flux, recon_a, recon_b, args.title,
                    Path(args.out) if args.out else None,
                    vmin=args.vmin, vmax=args.vmax)


if __name__ == "__main__":
    main()
