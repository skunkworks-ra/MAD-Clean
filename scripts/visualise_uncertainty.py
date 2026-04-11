#!/usr/bin/env python3
"""
scripts/visualise_uncertainty.py — Per-galaxy uncertainty visualisation.

For Variant C (FlowSolver, stochastic ensemble) and DPS (prior+likelihood),
runs posterior sampling on a random selection of galaxies and produces a
diagnostic figure per method showing:

    Dirty | Reconstruction | Truth | Residual | Fractional Error | Std

Colormaps
---------
  Dirty / Recon / Truth  : inferno   (sequential, physical units)
  Residual               : RdBu_r    (diverging, symmetric ±vmax)
  Fractional error       : RdBu_r    (diverging, fixed [-1, 1])
  Std                    : viridis   (sequential)

Fractional error is defined as (recon − truth) / truth.max(), giving a
unitless quantity on [−1, 1] for a perfect reconstruction.

Usage
-----
    pixi run -e gpu python scripts/visualise_uncertainty.py \\
        --data  crumb_data/flow_pairs_vla.npz \\
        --model models/flow_model.pt \\
        --prior models/prior_model.pt \\
        --out   logs/uncertainty \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Panel layout constants
# ---------------------------------------------------------------------------

_COLS   = ["Dirty", "Reconstruction", "Truth", "Residual", "Frac. Error", "Std"]
_NCOLS  = len(_COLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_galaxies(n_total: int, n_pick: int, seed: int) -> np.ndarray:
    """Return n_pick randomly chosen indices from [0, n_total)."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n_pick, n_total), replace=False)


def _run_flow(dirty: np.ndarray, solver) -> tuple[np.ndarray, np.ndarray]:
    """FlowSolver → (mean, std) via decode_island_with_uncertainty."""
    import torch
    island = torch.from_numpy(dirty).float().to(solver.device)
    mean, std = solver.decode_island_with_uncertainty(island)
    return mean.cpu().numpy(), std.cpu().numpy()


def _run_dps(dirty: np.ndarray, solver) -> tuple[np.ndarray, np.ndarray]:
    """DPSSolver → (mean, std) via sample."""
    import torch
    island = torch.from_numpy(dirty).float()
    mean, std = solver.sample(island)
    return mean.cpu().numpy(), std.cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_method(
    method_label : str,
    galaxy_idx   : np.ndarray,
    dirty        : np.ndarray,
    clean        : np.ndarray,
    flux_ratios  : np.ndarray,
    results      : list[dict],   # list of {mean, std} per galaxy
    out_dir      : Path,
    fname        : str,
) -> None:
    """
    Produce a (n_galaxies × 6) figure for one method.

    results[i] keys: "mean" (H,W), "std" (H,W)
    """
    n = len(galaxy_idx)
    fig, axes = plt.subplots(
        n, _NCOLS,
        figsize=(_NCOLS * 2.6, n * 2.6),
        squeeze=False,
    )
    fig.suptitle(method_label, fontsize=12, fontweight="bold", y=1.01)

    for row, (idx, res) in enumerate(zip(galaxy_idx, results)):
        d_img   = dirty[idx]
        c_img   = clean[idx]
        mean    = res["mean"]
        std     = res["std"]

        residual  = mean - c_img
        frac_err  = residual / (c_img.max() + 1e-8)
        frac_err  = np.clip(frac_err, -1.0, 1.0)

        # Shared intensity scale for dirty / recon / truth
        vmax_img = float(c_img.max())

        # Symmetric colorbar limit for residual
        res_abs = float(np.abs(residual).max())

        panels = [
            dict(img=d_img,   cmap="inferno",  vmin=0,        vmax=vmax_img, label="Jy/px"),
            dict(img=mean,    cmap="inferno",  vmin=0,        vmax=vmax_img, label="Jy/px"),
            dict(img=c_img,   cmap="inferno",  vmin=0,        vmax=vmax_img, label="Jy/px"),
            dict(img=residual, cmap="RdBu_r",  vmin=-res_abs, vmax=res_abs,  label="Jy/px"),
            dict(img=frac_err, cmap="RdBu_r",  vmin=-1.0,     vmax=1.0,      label="frac."),
            dict(img=std,     cmap="viridis",  vmin=0,        vmax=std.max() + 1e-8, label="Jy/px"),
        ]

        for col, (ax, panel) in enumerate(zip(axes[row], panels)):
            im = ax.imshow(
                panel["img"], origin="lower",
                cmap=panel["cmap"],
                vmin=panel["vmin"], vmax=panel["vmax"],
            )
            ax.set_xticks([]); ax.set_yticks([])

            # Column titles on first row only
            if row == 0:
                ax.set_title(_COLS[col], fontsize=9, fontweight="bold")

            # Row label: compactness on first column
            if col == 0:
                ax.set_ylabel(
                    f"idx={idx}\ncompact={flux_ratios[idx]:.3f}",
                    fontsize=8,
                )

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label(panel["label"], fontsize=6)

            # Fractional error: annotate ±1 ticks explicitly
            if col == 4:
                cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                cbar.ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter("%.1f")
                )

    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-galaxy uncertainty visualisation for Variant C and DPS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",        default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--model",       default="models/flow_model.pt",
                   help="Variant C FlowModel checkpoint.")
    p.add_argument("--prior",       default="models/prior_model.pt",
                   help="DPS prior FlowModel checkpoint.")
    p.add_argument("--n_galaxies",  type=int,   default=4,
                   help="Number of random galaxies to visualise.")
    p.add_argument("--n_samples",   type=int,   default=16,
                   help="Posterior draws per galaxy (higher → smoother std map).")
    p.add_argument("--perturb_std", type=float, default=0.05,
                   help="Starting perturbation std for FlowSolver ensemble.")
    p.add_argument("--dps_weight",  type=float, default=0.05,
                   help="DPS likelihood gradient scale ζ.")
    p.add_argument("--noise_std",   type=float, default=0.05,
                   help="Estimated dirty image noise std for DPS likelihood.")
    p.add_argument("--dps_steps",   type=int,   default=50,
                   help="Euler steps for DPS integration.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out",         default="logs/uncertainty")
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA unavailable; falling back to cpu.")
        device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    data  = np.load(args.data)
    clean = data["clean"].astype(np.float32)
    dirty = data["dirty"].astype(np.float32)
    N     = len(clean)
    print(f"Loaded {N} image pairs  shape={clean.shape[1]}×{clean.shape[2]}")

    psf_norm = None
    for key in ("psf_norm", "psf"):
        if key in data:
            psf_norm = data[key].astype(np.float32)
            break

    total_flux  = np.sum(np.clip(clean, 0, None), axis=(1, 2))
    peak_flux   = clean.max(axis=(1, 2))
    flux_ratios = peak_flux / (total_flux + 1e-8)

    galaxy_idx = _pick_galaxies(N, args.n_galaxies, args.seed)
    print(f"Selected galaxies: {galaxy_idx.tolist()}")
    print(f"Compactness: {flux_ratios[galaxy_idx].round(3).tolist()}")

    # ── Variant C — FlowSolver ───────────────────────────────────────────────
    flow_results = None
    model_path   = Path(args.model)
    if model_path.exists():
        try:
            from mad_clean.training.flow import FlowModel
            from mad_clean.solvers import FlowSolver
            fm = FlowModel.load(str(model_path), device=device)
            solver_c = FlowSolver(
                fm,
                device      = device,
                n_samples   = args.n_samples,
                n_steps     = 16,
                perturb_std = args.perturb_std,
            )
            print(f"Variant C loaded: {solver_c}")

            flow_results = []
            for i, idx in enumerate(galaxy_idx):
                mean, std = _run_flow(dirty[idx], solver_c)
                flow_results.append({"mean": mean, "std": std})
                print(f"  Flow galaxy {i + 1}/{args.n_galaxies} done", flush=True)

            _plot_method(
                method_label = f"Variant C — FlowSolver  "
                               f"(n_samples={args.n_samples}, "
                               f"perturb_std={args.perturb_std})",
                galaxy_idx   = galaxy_idx,
                dirty        = dirty,
                clean        = clean,
                flux_ratios  = flux_ratios,
                results      = flow_results,
                out_dir      = out_dir,
                fname        = "uncertainty_flow.png",
            )
        except Exception as e:
            print(f"WARNING: Variant C failed — {e}")
    else:
        print(f"Variant C checkpoint not found at {model_path} — skipping.")

    # ── DPS — prior + explicit likelihood ────────────────────────────────────
    dps_results = None
    prior_path  = Path(args.prior)
    if prior_path.exists():
        if psf_norm is None:
            print("WARNING: DPS requires psf_norm in the data file — skipping.")
        else:
            try:
                from mad_clean.training.flow import FlowModel
                from mad_clean.solvers import DPSSolver
                prior_fm   = FlowModel.load(str(prior_path), device=device)
                solver_dps = DPSSolver(
                    prior_fm,
                    psf_norm   = psf_norm,
                    noise_std  = args.noise_std,
                    n_steps    = args.dps_steps,
                    n_samples  = args.n_samples,
                    dps_weight = args.dps_weight,
                    device     = device,
                )
                print(f"DPS loaded: {solver_dps}")

                dps_results = []
                for i, idx in enumerate(galaxy_idx):
                    mean, std = _run_dps(dirty[idx], solver_dps)
                    dps_results.append({"mean": mean, "std": std})
                    print(f"  DPS galaxy {i + 1}/{args.n_galaxies} done", flush=True)

                _plot_method(
                    method_label = f"DPS — prior + likelihood  "
                                   f"(n_samples={args.n_samples}, "
                                   f"ζ={args.dps_weight})",
                    galaxy_idx   = galaxy_idx,
                    dirty        = dirty,
                    clean        = clean,
                    flux_ratios  = flux_ratios,
                    results      = dps_results,
                    out_dir      = out_dir,
                    fname        = "uncertainty_dps.png",
                )
            except Exception as e:
                print(f"WARNING: DPS failed — {e}")
    else:
        print(f"Prior checkpoint not found at {prior_path} — skipping.")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
