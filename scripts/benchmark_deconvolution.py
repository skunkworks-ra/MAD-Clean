#!/usr/bin/env python3
"""
scripts/benchmark_deconvolution.py — End-to-end deconvolution benchmark.

Extracts 6 stratified test sources from flow_pairs_vla.npz (2 compact, 2 mid,
2 extended), runs all MAD-CLEAN variants through the full deconvolver loop, and
produces per-source comparison figures + a summary metrics table.

Methods benchmarked
-------------------
  hogbom   Classical Hogbom CLEAN (baseline)
  A        Patch dictionary (PatchSolver)
  B        Convolutional dictionary + PSF (ConvSolver)
  C        Conditional flow matching (FlowSolver) — with uncertainty
  dps      DPS prior + explicit likelihood (DPSSolver) — with uncertainty

Figures (per source, saved to work area)
-----------------------------------------
  source_N_comparison.png
    Row 0  Reconstruction   : Dirty | Hogbom | A | B | C | DPS | Truth
    Row 1  Residual vs truth: ───── | Hogbom | A | B | C | DPS | ─────
    Row 2  Fractional error : ───── | Hogbom | A | B | C | DPS | ─────
    Row 3  Uncertainty std  : ───── | ─────  | ─ | ─ | C | DPS | ─────

  summary_metrics.csv + printed table

Flux uncertainty
----------------
  FlowSolver  : per-pixel std accumulated by deconvolver across minor cycles.
  DPSSolver   : per-pixel std from a separate posterior sample() on the dirty image.
  Flux std    : sqrt(sum(std_map²)) — upper bound assuming independent pixels.

Usage
-----
    pixi run -e gpu python scripts/benchmark_deconvolution.py \\
        --data  crumb_data/flow_pairs_vla.npz \\
        --atoms_a models/cdl_filters_patch.npz \\
        --atoms_b models/cdl_filters_conv.npz \\
        --model   models/flow_model.pt \\
        --prior   models/prior_model.pt \\
        --device  cuda
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch


# ---------------------------------------------------------------------------
# Work area
# ---------------------------------------------------------------------------

def _make_work_area(base: str = "logs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work  = Path(base) / f"deconv_benchmark_{stamp}"
    work.mkdir(parents=True)
    print(f"Work area: {work}/")
    return work


# ---------------------------------------------------------------------------
# Stratified source selection
# ---------------------------------------------------------------------------

def select_sources(
    clean       : np.ndarray,
    n_per_tier  : int = 2,
    seed        : int = 42,
) -> list[dict]:
    """
    Return 2*n_per_tier source dicts, stratified by compactness.

    Compactness = peak / total_positive_flux.  High = point-like, low = extended.
    Tiers: top third (compact), middle third, bottom third (extended).
    Within each tier, pick n_per_tier at evenly-spaced quantiles for
    reproducibility and good coverage.
    """
    total_flux  = np.sum(np.clip(clean, 0, None), axis=(1, 2))
    peak_flux   = clean.max(axis=(1, 2))
    compactness = peak_flux / (total_flux + 1e-8)

    order = np.argsort(compactness)   # ascending: most extended first
    N     = len(order)
    tier_size = N // 3

    tiers = {
        "extended" : order[:tier_size],
        "mid"      : order[tier_size : 2 * tier_size],
        "compact"  : order[2 * tier_size :],
    }

    rng     = np.random.default_rng(seed)
    sources = []
    for tier_name, tier_idx in tiers.items():
        # Pick n_per_tier evenly-spaced positions within the sorted tier
        positions = np.linspace(0, len(tier_idx) - 1, n_per_tier, dtype=int)
        for pos in positions:
            idx = int(tier_idx[pos])
            sources.append({
                "idx"        : idx,
                "tier"       : tier_name,
                "compactness": float(compactness[idx]),
            })

    return sources


# ---------------------------------------------------------------------------
# Solver construction
# ---------------------------------------------------------------------------

def build_solvers(args, psf: np.ndarray, device: str) -> dict:
    """Construct all available solvers. Missing checkpoints are skipped."""
    solvers = {}

    # Variant A
    if Path(args.atoms_a).exists():
        try:
            from mad_clean.filters import FilterBank
            from mad_clean.solvers import PatchSolver
            fb = FilterBank.load(args.atoms_a, device=device)
            solvers["A"] = PatchSolver(fb, n_nonzero=args.n_nonzero,
                                       stride=args.stride)
            print(f"  Variant A: {solvers['A']}")
        except Exception as e:
            print(f"  WARNING: Variant A failed — {e}")

    # Variant B
    if Path(args.atoms_b).exists():
        try:
            from mad_clean.filters import FilterBank
            from mad_clean.solvers import ConvSolver
            fb = FilterBank.load(args.atoms_b, device=device)
            solvers["B"] = ConvSolver(fb, lmbda=args.lmbda,
                                      n_iter=args.fista_iter, psf=psf)
            print(f"  Variant B: {solvers['B']}")
        except Exception as e:
            print(f"  WARNING: Variant B failed — {e}")

    # Variant C (FlowSolver — has decode_island_with_uncertainty)
    if Path(args.model).exists():
        try:
            from mad_clean.training.flow import FlowModel
            from mad_clean.solvers import FlowSolver
            fm = FlowModel.load(args.model, device=device)
            solvers["C"] = FlowSolver(
                fm, device=device,
                n_samples   = args.n_samples,
                n_steps     = args.flow_steps,
                perturb_std = args.perturb_std,
            )
            print(f"  Variant C: {solvers['C']}")
        except Exception as e:
            print(f"  WARNING: Variant C failed — {e}")

    # DPS
    if Path(args.prior).exists():
        try:
            from mad_clean.training.flow import FlowModel
            from mad_clean.solvers import DPSSolver
            prior_fm = FlowModel.load(args.prior, device=device)
            solvers["dps"] = DPSSolver(
                prior_fm, psf        = psf,
                noise_std  = args.noise_std,
                n_steps    = args.dps_steps,
                n_samples  = args.n_samples,
                dps_weight = args.dps_weight,
                device     = device,
            )
            print(f"  DPS: {solvers['dps']}")
        except Exception as e:
            print(f"  WARNING: DPS failed — {e}")

    return solvers


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

_DECONV_COMMON: dict = {}   # populated in main() from args


def run_deconvolution(
    dirty    : np.ndarray,
    psf      : np.ndarray,
    solvers  : dict,
    device   : str,
) -> dict[str, dict]:
    """
    Run all variants on a single dirty image.
    Returns {method_name: deconvolver result dict}.
    """
    from mad_clean import HogbomDeconvolver, MADCleanDeconvolver

    results = {}

    # Hogbom baseline
    hog = HogbomDeconvolver(psf=psf, device=device, **_DECONV_COMMON)
    results["hogbom"] = hog.deconvolve(dirty)

    # Learned variants
    for name, solver in solvers.items():
        dec = MADCleanDeconvolver(
            solver   = solver,
            psf      = psf,
            device   = device,
            **_DECONV_COMMON,
        )
        results[name] = dec.deconvolve(dirty)

    return results


def add_dps_uncertainty(
    results  : dict[str, dict],
    dirty    : np.ndarray,
    solvers  : dict,
    device   : str,
) -> None:
    """
    DPSSolver has no decode_island_with_uncertainty, so the deconvolver loop
    can't accumulate a std map.  Run a single posterior sample on the full dirty
    image as a post-hoc uncertainty estimate and store it in results["dps"].
    """
    if "dps" not in solvers or "dps" not in results:
        return
    if results["dps"]["uncertainty"] is not None:
        return   # already populated

    solver = solvers["dps"]
    island = torch.from_numpy(dirty).float()
    try:
        _, std = solver.sample(island, n_samples=solver.n_samples)
        results["dps"]["uncertainty"] = std.cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"  WARNING: DPS uncertainty estimation failed — {e}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: np.ndarray, truth: np.ndarray,
                    uncertainty: Optional[np.ndarray]) -> dict:
    model_pos = np.clip(model, 0, None)
    truth_pos = np.clip(truth, 0, None)

    # Pearson correlation
    a, b  = model.ravel() - model.mean(), truth.ravel() - truth.mean()
    corr  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # RMS residual
    rms   = float(np.sqrt(np.mean((model - truth) ** 2)))

    # Flux recovery
    flux_truth = float(truth_pos.sum()) + 1e-12
    flux_rec   = float(model_pos.sum()) / flux_truth

    # Flux uncertainty (upper bound: sqrt(sum(std²)) / flux_truth)
    flux_std_frac = None
    if uncertainty is not None:
        flux_std_frac = float(np.sqrt((uncertainty ** 2).sum())) / flux_truth

    return {
        "corr"         : corr,
        "rms"          : rms,
        "flux_rec"     : flux_rec,
        "flux_std_frac": flux_std_frac,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_METHODS_ORDERED = ["hogbom", "A", "B", "C", "dps"]
_METHOD_LABEL    = {
    "hogbom": "Hogbom", "A": "Variant A",
    "B": "Variant B",   "C": "Variant C",
    "dps": "DPS",
}
_COLS_ORDERED = ["dirty"] + _METHODS_ORDERED + ["truth"]
_NCOLS        = len(_COLS_ORDERED)   # 7
_NROWS        = 4


def _blank(ax: plt.Axes, text: str = "") -> None:
    ax.set_facecolor("#efefef")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if text:
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")


def _imshow_with_cbar(
    fig: plt.Figure,
    ax: plt.Axes,
    img: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    unit_label: str = "",
    ticks: list | None = None,
) -> None:
    im   = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    if unit_label:
        cbar.set_label(unit_label, fontsize=6)
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def plot_source(
    source   : dict,
    dirty    : np.ndarray,
    truth    : np.ndarray,
    results  : dict[str, dict],
    metrics  : dict[str, dict],
    out_path : Path,
) -> None:
    """
    4 × 7 comparison figure for one source.

    Row 0  Reconstruction
    Row 1  Residual vs truth  (RdBu_r, symmetric)
    Row 2  Fractional error   (RdBu_r, [-1, 1])
    Row 3  Uncertainty std    (viridis, only C and DPS)
    """
    fig, axes = plt.subplots(
        _NROWS, _NCOLS,
        figsize=(_NCOLS * 2.8, _NROWS * 2.8),
        squeeze=False,
    )

    vmax_img  = float(truth.max())
    col_order = _COLS_ORDERED   # dirty, hogbom, A, B, C, dps, truth

    # ── Row 0: Reconstructions ────────────────────────────────────────────────
    for col, name in enumerate(col_order):
        ax  = axes[0, col]
        img = dirty if name == "dirty" else (
              truth if name == "truth" else
              results[name]["model"] if name in results else None)

        if img is None:
            _blank(ax, "missing")
            if col == 0: ax.set_title("Dirty",  fontsize=9, fontweight="bold")
            continue

        _imshow_with_cbar(fig, ax, img, "inferno", 0, vmax_img, "Jy/px")
        if name in ("dirty", "truth"):
            title = "Dirty" if name == "dirty" else "Truth"
        else:
            m = metrics.get(name, {})
            title = (f"{_METHOD_LABEL[name]}\n"
                     f"corr={m.get('corr', 0):.3f}  fr={m.get('flux_rec', 0):.2f}")
        ax.set_title(title, fontsize=8, fontweight="bold")

    # ── Row 1: Residual (recon − truth) ────────────────────────────────────────
    for col, name in enumerate(col_order):
        ax = axes[1, col]
        if name in ("dirty", "truth") or name not in results:
            _blank(ax)
            continue
        resid    = results[name]["model"] - truth
        res_abs  = float(np.abs(resid).max()) + 1e-8
        _imshow_with_cbar(fig, ax, resid, "RdBu_r", -res_abs, res_abs, "Jy/px")
        if col == 0:
            ax.set_ylabel("Residual", fontsize=8, rotation=90, labelpad=4)

    axes[1, 0].set_ylabel("Residual", fontsize=8)

    # ── Row 2: Fractional error ────────────────────────────────────────────────
    for col, name in enumerate(col_order):
        ax = axes[2, col]
        if name in ("dirty", "truth") or name not in results:
            _blank(ax)
            continue
        frac = np.clip(
            (results[name]["model"] - truth) / (vmax_img + 1e-8),
            -1.0, 1.0,
        )
        _imshow_with_cbar(fig, ax, frac, "RdBu_r", -1.0, 1.0, "frac.",
                          ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])

    axes[2, 0].set_ylabel("Frac. error", fontsize=8)

    # ── Row 3: Uncertainty std ─────────────────────────────────────────────────
    for col, name in enumerate(col_order):
        ax = axes[3, col]
        unc = results.get(name, {}).get("uncertainty") if name not in ("dirty", "truth") else None

        if unc is None:
            label = "N/A" if name not in ("dirty", "truth") else ""
            _blank(ax, label)
            continue

        flux_std_frac = metrics.get(name, {}).get("flux_std_frac")
        title_unc = (f"flux_std={flux_std_frac:.3f}" if flux_std_frac is not None else "")
        _imshow_with_cbar(fig, ax, unc, "viridis", 0, float(unc.max()) + 1e-8, "Jy/px")
        ax.set_title(title_unc, fontsize=7)

    axes[3, 0].set_ylabel("Uncertainty σ", fontsize=8)

    # ── Row labels ────────────────────────────────────────────────────────────
    row_labels = ["Reconstruction", "Residual", "Frac. Error", "Uncertainty σ"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, labelpad=4)

    fig.suptitle(
        f"Source idx={source['idx']}  tier={source['tier']}  "
        f"compactness={source['compactness']:.4f}",
        fontsize=10, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Diagnostic plots: dirty / model / residual per method, own colourscale
# ---------------------------------------------------------------------------

def plot_diagnostics(
    source   : dict,
    dirty    : np.ndarray,
    truth    : np.ndarray,
    results  : dict[str, dict],
    out_path : Path,
) -> None:
    """
    Per-method diagnostic: dirty, model, residual — each on its own colourscale.
    One row per method, three columns.
    """
    methods = [m for m in _METHODS_ORDERED if m in results]
    n_methods = len(methods)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(
        n_methods, 3,
        figsize=(3 * 3.2, n_methods * 3.0),
        squeeze=False,
    )

    col_titles = ["Dirty", "Model", "Residual"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, method in enumerate(methods):
        res   = results[method]
        model = res["model"]
        resid = res["residual"]
        label = _METHOD_LABEL.get(method, method)

        panels = [dirty, model, resid]
        cmaps  = ["inferno", "inferno", "RdBu_r"]

        for col, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            if cmap == "RdBu_r":
                vlim = float(np.abs(img).max()) + 1e-12
                vmin, vmax = -vlim, vlim
            else:
                vmin, vmax = float(img.min()), float(img.max()) + 1e-12
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold",
                                rotation=90, labelpad=8)

    fig.suptitle(
        f"Diagnostics: source_{source['idx']}_{source['tier']}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary_table(all_metrics: list[dict]) -> None:
    header = f"{'source':<22} {'method':<10} {'corr':>6} {'rms':>8} {'flux_rec':>9} {'flux_std':>9}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for row in all_metrics:
        fs = f"{row['flux_std_frac']:.3f}" if row["flux_std_frac"] is not None else "   N/A"
        print(
            f"{row['source_label']:<22} {row['method']:<10} "
            f"{row['corr']:>6.3f} {row['rms']:>8.4f} "
            f"{row['flux_rec']:>9.3f} {fs:>9}"
        )
    print("─" * len(header))


def save_summary_csv(all_metrics: list[dict], out_path: Path) -> None:
    keys = ["source_label", "tier", "compactness", "method",
            "corr", "rms", "flux_rec", "flux_std_frac"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_metrics)
    print(f"Summary CSV → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Stratified deconvolution benchmark across all MAD-CLEAN variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",        default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--atoms_a",     default="models/cdl_filters_patch.npz")
    p.add_argument("--atoms_b",     default="models/cdl_filters_conv.npz")
    p.add_argument("--model",       default="models/flow_model.pt",
                   help="Variant C FlowModel checkpoint.")
    p.add_argument("--prior",       default="models/prior_model.pt",
                   help="DPS prior FlowModel checkpoint.")
    p.add_argument("--skip",        nargs="*", default=[],
                   help="Solver keys to skip (e.g. --skip A C).")
    p.add_argument("--n_per_tier",  type=int,   default=2,
                   help="Sources per compactness tier (total = 3 × n_per_tier).")
    p.add_argument("--n_samples",   type=int,   default=8,
                   help="Posterior draws for FlowSolver and DPS uncertainty.")
    p.add_argument("--perturb_std", type=float, default=0.05,
                   help="Starting perturbation std for FlowSolver ensemble.")
    p.add_argument("--flow_steps",  type=int,   default=16,
                   help="Euler steps for FlowSolver.")
    p.add_argument("--dps_steps",   type=int,   default=50,
                   help="Euler steps for DPS.")
    p.add_argument("--dps_weight",  type=float, default=0.05,
                   help="DPS likelihood gradient scale ζ.")
    p.add_argument("--noise_std",   type=float, default=0.05,
                   help="Estimated dirty image noise std for DPS likelihood.")
    p.add_argument("--n_nonzero",   type=int,   default=5,   help="[A] OMP sparsity")
    p.add_argument("--stride",      type=int,   default=8,   help="[A] patch stride")
    p.add_argument("--lmbda",       type=float, default=0.01, help="[B] FISTA L1")
    p.add_argument("--fista_iter",  type=int,   default=100, help="[B] FISTA iterations")
    p.add_argument("--n_major",      type=int,   default=10,  help="Max major CLEAN cycles")
    p.add_argument("--max_minor",    type=int,   default=100, help="Hard cap on minor iter per major cycle")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable — falling back to cpu.")
        device = "cpu"

    # ── Shared deconvolver settings ───────────────────────────────────────────
    _DECONV_COMMON.update(
        gain           = 0.1,
        n_major_cycles = args.n_major,
        max_minor_iter = args.max_minor,
        verbose        = True,
    )

    # ── Work area ─────────────────────────────────────────────────────────────
    work = _make_work_area()

    # ── Data ──────────────────────────────────────────────────────────────────
    data     = np.load(args.data)
    clean    = data["clean"].astype(np.float32)
    dirty    = data["dirty"].astype(np.float32)
    psf = None
    for key in ("psf", "psf_norm"):
        if key in data:
            psf = data[key].astype(np.float32)
            break
    if psf is None:
        raise RuntimeError("No PSF key found in data file.")

    # Ensure peak=1 (CASA convention)
    psf_peak = psf.max()
    if psf_peak > 0:
        psf = psf / psf_peak

    N = len(clean)
    print(f"Loaded {N} image pairs  shape={clean.shape[1]}×{clean.shape[2]}")
    print(f"PSF peak={psf.max():.4f}  sum={psf.sum():.1f}")

    # Save shared PSF for reference
    np.save(work / "psf.npy", psf)

    # ── Stratified source selection ───────────────────────────────────────────
    sources = select_sources(clean, n_per_tier=args.n_per_tier, seed=args.seed)
    print(f"\nSelected {len(sources)} sources:")
    for s in sources:
        print(f"  idx={s['idx']:4d}  tier={s['tier']:8s}  "
              f"compactness={s['compactness']:.4f}")

    # Save source arrays
    src_dir = work / "sources"
    src_dir.mkdir()
    for s in sources:
        label = f"source_{s['idx']}_{s['tier']}"
        np.save(src_dir / f"{label}_dirty.npy",  dirty[s["idx"]])
        np.save(src_dir / f"{label}_clean.npy",  clean[s["idx"]])

    # ── Build solvers (once, reused across all sources) ───────────────────────
    print("\nBuilding solvers …")
    solvers = build_solvers(args, psf, device)
    for key in args.skip:
        if key in solvers:
            del solvers[key]
            print(f"  Skipped solver {key}")
    if not solvers:
        print("WARNING: no learned solvers loaded — only Hogbom will run.")

    # ── Deconvolve + evaluate ─────────────────────────────────────────────────
    all_metrics = []

    for s in sources:
        idx   = s["idx"]
        label = f"source_{idx}_{s['tier']}"
        print(f"\n{'─'*60}")
        print(f"Source {label}  compactness={s['compactness']:.4f}")

        d_img = dirty[idx]
        c_img = clean[idx]

        # Run all deconvolvers
        results = run_deconvolution(d_img, psf, solvers, device)

        # Post-hoc DPS uncertainty (run sample() on full dirty image)
        add_dps_uncertainty(results, d_img, solvers, device)

        # Save model outputs
        res_dir = work / "results" / label
        res_dir.mkdir(parents=True)
        np.save(res_dir / "dirty.npy",  d_img)
        np.save(res_dir / "truth.npy",  c_img)
        for method, res in results.items():
            np.save(res_dir / f"{method}_model.npy",    res["model"])
            np.save(res_dir / f"{method}_residual.npy", res["residual"])
            if res["uncertainty"] is not None:
                np.save(res_dir / f"{method}_uncertainty.npy", res["uncertainty"])

        # Metrics
        source_metrics = {}
        for method, res in results.items():
            m = compute_metrics(res["model"], c_img, res["uncertainty"])
            source_metrics[method] = m
            fs = f"±{m['flux_std_frac']:.3f}" if m["flux_std_frac"] is not None else ""
            print(f"  {_METHOD_LABEL.get(method, method):12s}  "
                  f"corr={m['corr']:.3f}  rms={m['rms']:.4f}  "
                  f"flux_rec={m['flux_rec']:.3f}{fs}  "
                  f"cycles={results[method]['n_major']}")
            all_metrics.append({
                "source_label" : label,
                "tier"         : s["tier"],
                "compactness"  : round(s["compactness"], 4),
                "method"       : method,
                **m,
            })

        # Plot
        plot_source(
            source   = s,
            dirty    = d_img,
            truth    = c_img,
            results  = results,
            metrics  = source_metrics,
            out_path = work / f"{label}_comparison.png",
        )
        plot_diagnostics(
            source   = s,
            dirty    = d_img,
            truth    = c_img,
            results  = results,
            out_path = work / f"{label}_diagnostics.png",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(all_metrics)
    save_summary_csv(all_metrics, work / "summary_metrics.csv")
    print(f"\nAll outputs in {work}/")


if __name__ == "__main__":
    main()
