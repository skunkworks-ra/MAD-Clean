#!/usr/bin/env python3
"""
scripts/evaluate_morphology.py — Morphology reconstruction experiment.

Compares Gaussian fitting (ASP CLEAN single-component equivalent) against
Variant A (PatchSolver), Variant B (ConvSolver+PSF), and Variant C (FlowModel)
across the full range of source complexity.

The key question: for extended sources (high flux_ratio), does the learned
representation capture morphology that a Gaussian basis cannot?

Flux ratio = sum(positive flux) / peak flux, computed on clean ground truth.
Compact sources: low flux_ratio (point-like, well-described by a Gaussian).
Extended sources: high flux_ratio (jets, lobes, diffuse — Gaussian fails here).

Usage:
    pixi run -e gpu python scripts/evaluate_morphology.py \\
        --data    crumb_data/flow_pairs_vla.npz \\
        --model_a models/cdl_filters_patch.npz \\
        --model_b models/cdl_filters_conv.npz \\
        --model   models/flow_model.pt \\
        --prior   models/prior_model.pt \\
        --out     logs/morphology_experiment \\
        --n_samples 200 \\
        --dps_samples 8 \\
        [--device cuda]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Gaussian fitting baseline
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return (bg + amp * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))).ravel()


def fit_gaussian(image: np.ndarray) -> np.ndarray | None:
    """
    Fit a single 2D Gaussian to image. Returns the fitted model image,
    or None if the optimiser fails.
    """
    H, W = image.shape
    y_g, x_g = np.mgrid[0:H, 0:W].astype(np.float32)

    peak_yx = np.unravel_index(image.argmax(), image.shape)
    p0 = [float(image.max()), float(peak_yx[1]), float(peak_yx[0]),
          H / 8.0, H / 8.0, 0.0, float(np.percentile(image, 10))]
    lo = [0.0,   0.0, 0.0, 0.3, 0.3, -np.pi / 2, -np.inf]
    hi = [np.inf, W,   H,   W,   H,   np.pi / 2,  np.inf]

    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            (x_g.ravel(), y_g.ravel()),
            image.ravel(),
            p0=p0, bounds=(lo, hi),
            maxfev=10_000,
        )
        return _gaussian_2d((x_g, y_g), *popt).reshape(H, W)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation — proxy for structural similarity."""
    a, b = a.ravel(), b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def residual_rms(recon: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((recon - truth) ** 2)))


def flux_recovery(recon: np.ndarray, truth: np.ndarray) -> float:
    denom = np.sum(np.clip(truth, 0, None)) + 1e-12
    return float(np.sum(np.clip(recon, 0, None)) / denom)


# ---------------------------------------------------------------------------
# FlowModel inference (deterministic Euler, 20 steps)
# ---------------------------------------------------------------------------

def flowmodel_reconstruct(
    dirty: np.ndarray,
    fm,
    device: str,
    n_steps: int = 20,
    n_samples: int = 1,
    perturb_std: float = 0.0,
) -> np.ndarray:
    """
    ODE integration: dirty → clean estimate.

    n_samples=1, perturb_std=0.0 : deterministic Euler (mean prediction, blurrier)
    n_samples>1, perturb_std>0   : stochastic ensemble mean (sharper, more variance)
    """
    import torch
    from mad_clean.solvers import FlowSolver
    solver = FlowSolver(
        fm, device=device,
        n_samples=n_samples,
        n_steps=n_steps,
        perturb_std=perturb_std,
    )
    island = torch.from_numpy(dirty).float().to(device)
    return solver.decode_island(island).cpu().numpy()


def dps_reconstruct(
    dirty: np.ndarray,
    dps_solver,
) -> np.ndarray:
    """
    DPS posterior mean: noise → clean guided by explicit likelihood gradient.
    Returns (H, W) float32 posterior mean.
    """
    import torch
    island = torch.from_numpy(dirty).float()
    mean, _ = dps_solver.sample(island)
    return mean.cpu().numpy()


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(flux_ratios: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    """
    Sample n_total indices stratified across the flux_ratio range.
    Ensures representation from compact, intermediate, and extended sources.
    """
    rng = np.random.default_rng(seed)
    quartiles = np.quantile(flux_ratios, [0.25, 0.5, 0.75])
    bins = [
        np.where(flux_ratios <= quartiles[0])[0],
        np.where((flux_ratios > quartiles[0]) & (flux_ratios <= quartiles[1]))[0],
        np.where((flux_ratios > quartiles[1]) & (flux_ratios <= quartiles[2]))[0],
        np.where(flux_ratios > quartiles[2])[0],
    ]
    per_bin = n_total // 4
    idx = np.concatenate([
        rng.choice(b, size=min(per_bin, len(b)), replace=False)
        for b in bins
    ])
    return idx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Consistent colour/label map for all methods
_METHOD_STYLE = {
    "gauss":  {"color": "#e07b39", "label": "Gaussian fit"},
    "var_a":  {"color": "#2ca02c", "label": "Variant A (patch)"},
    "var_b":  {"color": "#9467bd", "label": "Variant B (conv+PSF)"},
    "flow":   {"color": "#3a86ff", "label": "Variant C (flow)"},
    "dps":    {"color": "#d62728", "label": "DPS (prior + likelihood)"},
}


def plot_scatter(flux_ratios, method_metrics: dict, out_dir: Path) -> None:
    """
    Scatter corr and rms vs flux_ratio for all available methods.

    method_metrics : {method_key: {"corr": [...], "rms": [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Reconstruction quality across source complexity",
        fontsize=11, fontweight="bold",
    )

    for ax, metric_key, ylabel in zip(
        axes,
        ["corr", "rms"],
        ["Pearson correlation (↑ better)", "Residual RMS (↓ better)"],
    ):
        for key, vals_dict in method_metrics.items():
            style = _METHOD_STYLE[key]
            vals = vals_dict[metric_key]
            ax.scatter(
                flux_ratios, vals,
                alpha=0.4, s=12,
                color=style["color"], label=style["label"], zorder=2,
            )
            # Running-median trend line
            order = np.argsort(flux_ratios)
            window = max(10, len(flux_ratios) // 10)
            sorted_vals = np.array(vals)[order]
            smooth = np.convolve(sorted_vals, np.ones(window) / window, mode="valid")
            x_smooth = np.sort(flux_ratios)[window // 2: window // 2 + len(smooth)]
            ax.plot(x_smooth, smooth, color=style["color"], linewidth=2, zorder=4)

        ax.set_xlabel("Flux ratio  (sum positive / peak)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axvline(50, color="gray", linestyle="--", alpha=0.4,
                   label="compact/extended boundary")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "scatter_reconstruction_quality.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_examples(
    indices,
    clean, dirty,
    method_recons: dict,   # {method_key: [recon, ...]} in top_extended order
    flux_ratios,
    out_dir: Path,
) -> None:
    """Multi-panel comparison for the most extended examples."""
    n = min(4, len(indices))
    # Column order: dirty, then all methods in a fixed order, then truth
    method_keys = list(method_recons.keys())
    col_titles = (
        ["Dirty"]
        + [_METHOD_STYLE[k]["label"] for k in method_keys]
        + ["Clean truth"]
    )
    cols = len(col_titles)

    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.8, n * 2.8))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices[:n]):
        panels = (
            [dirty[idx]]
            + [method_recons[k][row] for k in method_keys]
            + [clean[idx]]
        )
        # Use a shared scale across all panels so CDL/flow output isn't
        # black against the clean image range. 99th percentile of the clean
        # truth is a stable anchor without being dominated by outlier peaks.
        vmin = 0.0
        vmax = float(np.percentile(clean[idx], 99))

        for col, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            ax.imshow(panel, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"flux_ratio={flux_ratios[idx]:.0f}", fontsize=9)

    plt.suptitle("Extended source reconstruction — most complex examples", fontsize=11)
    plt.tight_layout()
    path = out_dir / "example_extended_sources.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare all MAD-CLEAN variants across source complexity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--model_a",    default="models/cdl_filters_patch.npz",
                   help="Variant A FilterBank .npz. Skip if not found.")
    p.add_argument("--model_b",    default="models/cdl_filters_conv.npz",
                   help="Variant B FilterBank .npz. Skip if not found.")
    p.add_argument("--model",      default="models/flow_model.pt",
                   help="Variant C FlowModel .pt. Skip if not found.")
    p.add_argument("--prior",      default="models/prior_model.pt",
                   help="DPS prior FlowModel .pt (PriorTrainer output). Skip if not found.")
    p.add_argument("--dps_samples", type=int,   default=8,
                   help="Posterior draws per island for DPS.")
    p.add_argument("--dps_weight",  type=float, default=1.0,
                   help="DPS likelihood gradient scale ζ. Tune so flux_rec ≈ 1.0.")
    p.add_argument("--noise_std",   type=float, default=0.05,
                   help="Estimated dirty image noise std for DPS likelihood.")
    p.add_argument("--lmbda_b",    type=float, default=0.01,
                   help="L1 penalty for Variant B FISTA inference.")
    p.add_argument("--fista_iter", type=int,   default=100,
                   help="FISTA iterations for Variant B inference.")
    p.add_argument("--out",           default="logs/morphology_experiment")
    p.add_argument("--n_samples",     type=int,   default=200)
    p.add_argument("--flow_samples",  type=int,   default=1,
                   help="Stochastic ODE trajectories for Variant C (1=deterministic).")
    p.add_argument("--flow_perturb",  type=float, default=0.0,
                   help="Noise std for stochastic FlowSolver trajectories.")
    p.add_argument("--device",        default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    # ── Device check ──────────────────────────────────────────────────────────
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; falling back to cpu.")
        device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data  = np.load(args.data)
    clean = data["clean"].astype(np.float32)
    dirty = data["dirty"].astype(np.float32)
    N     = len(clean)
    print(f"Loaded {N} image pairs  shape={clean.shape[1]}×{clean.shape[2]}")

    # PSF for Variant B — must match the normalisation used at training time.
    # train.py feeds data["psf_norm"] to ConvDictTrainer, so we use the same key.
    psf_norm = data["psf_norm"].astype(np.float32) if "psf_norm" in data else None
    if psf_norm is None:
        print("WARNING: 'psf_norm' key not found in data file — Variant B will run without PSF.")

    # ── Flux ratio stratification ──────────────────────────────────────────────
    flux_ratios = (
        np.sum(np.clip(clean, 0, None), axis=(1, 2))
        / (clean.max(axis=(1, 2)) + 1e-8)
    )
    print(f"Flux ratio  min={flux_ratios.min():.1f}  "
          f"median={np.median(flux_ratios):.1f}  max={flux_ratios.max():.1f}")

    sample_idx = stratified_sample(flux_ratios, args.n_samples)
    print(f"Sampled {len(sample_idx)} sources (stratified across flux_ratio range)")

    # ── Load Variant A (PatchSolver) ──────────────────────────────────────────
    solver_a = None
    if Path(args.model_a).exists():
        try:
            from mad_clean.filters import FilterBank
            from mad_clean.solvers import PatchSolver
            fb_a     = FilterBank.load(args.model_a, device=device)
            solver_a = PatchSolver(fb_a, n_nonzero=5, stride=8)
            print(f"Variant A loaded: {solver_a}")
        except Exception as e:
            print(f"WARNING: could not load Variant A — {e}")
    else:
        print(f"Variant A checkpoint not found at {args.model_a} — skipping.")

    # ── Load Variant B (ConvSolver + PSF) ─────────────────────────────────────
    solver_b = None
    if Path(args.model_b).exists():
        try:
            from mad_clean.filters import FilterBank
            from mad_clean.solvers import ConvSolver
            fb_b     = FilterBank.load(args.model_b, device=device)
            solver_b = ConvSolver(
                fb_b,
                lmbda=args.lmbda_b,
                n_iter=args.fista_iter,
                psf=psf_norm,   # psf_norm from data file — matches training
            )
            print(f"Variant B loaded: {solver_b}")
        except Exception as e:
            print(f"WARNING: could not load Variant B — {e}")
    else:
        print(f"Variant B checkpoint not found at {args.model_b} — skipping.")

    # ── Load Variant C (FlowModel) ────────────────────────────────────────────
    fm = None
    model_path = Path(args.model)
    if model_path.exists():
        try:
            from mad_clean.training.flow import FlowModel
            fm = FlowModel.load(str(model_path), device=device)
            print(f"Variant C loaded: {fm}")
        except Exception as e:
            print(f"WARNING: could not load FlowModel — {e}")
            print("         Running without Variant C.")
    else:
        print(f"FlowModel checkpoint not found at {model_path} — skipping.")

    # ── Load DPS solver (prior + explicit likelihood) ─────────────────────────
    dps_solver = None
    prior_path = Path(args.prior)
    if prior_path.exists():
        try:
            from mad_clean.training.flow import FlowModel
            from mad_clean.solvers import DPSSolver
            prior_fm = FlowModel.load(str(prior_path), device=device)
            if psf_norm is None:
                print("WARNING: DPS requires psf_norm — skipping DPS (no psf_norm in data).")
            else:
                if device == "cpu":
                    print("WARNING: DPS on CPU is very slow (50 steps × n_samples). "
                          "Consider --device cuda.")
                dps_solver = DPSSolver(
                    prior_fm,
                    psf_norm      = psf_norm,
                    noise_std     = args.noise_std,
                    n_steps       = 50,
                    n_samples     = args.dps_samples,
                    dps_weight    = args.dps_weight,
                    device        = device,
                )
                print(f"DPS solver loaded: {dps_solver}")
        except Exception as e:
            print(f"WARNING: could not load DPS solver — {e}")
    else:
        print(f"Prior model not found at {prior_path} — skipping DPS.")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    # Per-method metric lists and reconstruction stores
    g_corr,  g_rms,  g_flux_rec  = [], [], []
    a_corr,  a_rms,  a_flux_rec  = [], [], []
    b_corr,  b_rms,  b_flux_rec  = [], [], []
    f_corr,  f_rms,  f_flux_rec  = [], [], []
    d_corr,  d_rms,  d_flux_rec  = [], [], []

    gauss_recons = []
    varA_recons  = []
    varB_recons  = []
    flow_recons  = []
    dps_recons   = []

    failed_gauss = 0

    for k, idx in enumerate(sample_idx):
        d_img = dirty[idx]
        c_img = clean[idx]

        # Gaussian fit
        gauss = fit_gaussian(d_img)
        if gauss is None:
            gauss = np.zeros_like(d_img)
            failed_gauss += 1
        gauss_recons.append(gauss)
        g_corr.append(pearson_corr(gauss, c_img))
        g_rms.append(residual_rms(gauss, c_img))
        g_flux_rec.append(flux_recovery(gauss, c_img))

        # Variant A
        if solver_a is not None:
            island = torch.from_numpy(d_img).float().to(solver_a.device)
            recon  = solver_a.decode_island(island).cpu().numpy()
            varA_recons.append(recon)
            a_corr.append(pearson_corr(recon, c_img))
            a_rms.append(residual_rms(recon, c_img))
            a_flux_rec.append(flux_recovery(recon, c_img))

        # Variant B
        if solver_b is not None:
            island = torch.from_numpy(d_img).float().to(solver_b.device)
            recon  = solver_b.decode_island(island).cpu().numpy()
            varB_recons.append(recon)
            b_corr.append(pearson_corr(recon, c_img))
            b_rms.append(residual_rms(recon, c_img))
            b_flux_rec.append(flux_recovery(recon, c_img))

        # Variant C (FlowModel)
        if fm is not None:
            recon = flowmodel_reconstruct(
                d_img, fm, device,
                n_samples=args.flow_samples,
                perturb_std=args.flow_perturb,
            )
            flow_recons.append(recon)
            f_corr.append(pearson_corr(recon, c_img))
            f_rms.append(residual_rms(recon, c_img))
            f_flux_rec.append(flux_recovery(recon, c_img))

        # DPS (prior + explicit likelihood)
        if dps_solver is not None:
            recon = dps_reconstruct(d_img, dps_solver)
            dps_recons.append(recon)
            d_corr.append(pearson_corr(recon, c_img))
            d_rms.append(residual_rms(recon, c_img))
            d_flux_rec.append(flux_recovery(recon, c_img))

        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(sample_idx)} evaluated")

    if failed_gauss:
        print(f"WARNING: Gaussian fit failed on {failed_gauss}/{len(sample_idx)} sources (set to zero)")

    # ── Summary statistics ─────────────────────────────────────────────────────
    sampled_ratios = flux_ratios[sample_idx]
    compact_mask   = sampled_ratios < 50
    extended_mask  = sampled_ratios > 100

    def _print_summary(label, corr, rms, flux_rec):
        print(f"\n── {label} {'─' * max(0, 44 - len(label))}")
        for name, mask in [("all", np.ones(len(sample_idx), bool)),
                           ("compact (ratio<50)", compact_mask),
                           ("extended (ratio>100)", extended_mask)]:
            if mask.sum() == 0:
                continue
            print(f"  {name:25s}  n={mask.sum():4d}  "
                  f"corr={np.mean(np.array(corr)[mask]):.3f}  "
                  f"rms={np.mean(np.array(rms)[mask]):.4f}  "
                  f"flux_rec={np.mean(np.array(flux_rec)[mask]):.3f}")

    _print_summary("Gaussian fit", g_corr, g_rms, g_flux_rec)
    if solver_a is not None:
        _print_summary("Variant A (patch)", a_corr, a_rms, a_flux_rec)
    if solver_b is not None:
        _print_summary("Variant B (conv+PSF)", b_corr, b_rms, b_flux_rec)
    if fm is not None:
        _print_summary("Variant C (flow)", f_corr, f_rms, f_flux_rec)
    if dps_solver is not None:
        _print_summary("DPS (prior+likelihood)", d_corr, d_rms, d_flux_rec)

    # ── Plots ──────────────────────────────────────────────────────────────────
    method_metrics = {"gauss": {"corr": g_corr, "rms": g_rms}}
    if solver_a is not None:
        method_metrics["var_a"] = {"corr": a_corr, "rms": a_rms}
    if solver_b is not None:
        method_metrics["var_b"] = {"corr": b_corr, "rms": b_rms}
    if fm is not None:
        method_metrics["flow"]  = {"corr": f_corr, "rms": f_rms}
    if dps_solver is not None:
        method_metrics["dps"]   = {"corr": d_corr, "rms": d_rms}

    plot_scatter(sampled_ratios, method_metrics, out_dir)

    # Most extended sources for the panel plot
    top_extended = np.argsort(sampled_ratios)[-4:][::-1]
    method_recons_panel = {}
    if varA_recons:
        method_recons_panel["var_a"] = [varA_recons[i] for i in top_extended]
    if varB_recons:
        method_recons_panel["var_b"] = [varB_recons[i] for i in top_extended]
    if flow_recons:
        method_recons_panel["flow"]  = [flow_recons[i] for i in top_extended]
    if dps_recons:
        method_recons_panel["dps"]   = [dps_recons[i]  for i in top_extended]

    plot_examples(
        sample_idx[top_extended],
        clean, dirty,
        # Always include gaussian in the panel
        {"gauss": [gauss_recons[i] for i in top_extended], **method_recons_panel},
        flux_ratios,
        out_dir,
    )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    header = ["idx", "flux_ratio",
              "gauss_corr", "gauss_rms", "gauss_flux_rec",
              "varA_corr",  "varA_rms",  "varA_flux_rec",
              "varB_corr",  "varB_rms",  "varB_flux_rec",
              "flow_corr",  "flow_rms",  "flow_flux_rec",
              "dps_corr",   "dps_rms",   "dps_flux_rec"]
    rows = [header]
    for k, idx in enumerate(sample_idx):
        def _fmt(lst, i):
            return f"{lst[i]:.4f}" if lst else "NA"
        rows.append([
            idx, f"{sampled_ratios[k]:.2f}",
            f"{g_corr[k]:.4f}", f"{g_rms[k]:.6f}", f"{g_flux_rec[k]:.4f}",
            _fmt(a_corr, k), _fmt(a_rms, k), _fmt(a_flux_rec, k),
            _fmt(b_corr, k), _fmt(b_rms, k), _fmt(b_flux_rec, k),
            _fmt(f_corr, k), _fmt(f_rms, k), _fmt(f_flux_rec, k),
            _fmt(d_corr, k), _fmt(d_rms, k), _fmt(d_flux_rec, k),
        ])
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("\n".join(",".join(str(v) for v in row) for row in rows))
    print(f"Saved → {csv_path}")
    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
