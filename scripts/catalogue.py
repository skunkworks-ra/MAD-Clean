#!/usr/bin/env python3
"""
scripts/catalogue.py — Per-source uncertainty catalogue from DPS posteriors.

For each source island detected in a dirty field, runs DPS posterior sampling
and extracts calibrated per-source statistics:

  - flux_mean / flux_std       integrated flux and its posterior std
  - peak_flux_mean / _std      peak pixel flux and posterior std
  - peak_row_mean / _std       peak row position (sub-pixel via mean)
  - peak_col_mean / _std       peak col position (sub-pixel via mean)
  - detection_prob             fraction of posterior samples with peak > threshold
  - morphology_conf            mean pairwise Pearson correlation between samples
                               (1.0 = all samples agree on morphology; near 0 = diffuse uncertainty)
  - bbox_r0/r1/c0/c1           bounding box of the detected island

Also saves:
  - logs/catalogue/mean_field.npy   — posterior mean image
  - logs/catalogue/std_field.npy    — posterior std image (uncertainty map)
  - logs/catalogue/sources.csv      — one row per source

Usage:
    pixi run -e gpu python scripts/catalogue.py \\
        --data        crumb_data/flow_pairs_vla.npz \\
        --prior       models/prior_model.pt \\
        --idx         0 \\
        --n_samples   20 \\
        --dps_weight  1.0 \\
        --device      cuda

    # Run on all examples and save CSV:
    pixi run -e gpu python scripts/catalogue.py \\
        --data        crumb_data/flow_pairs_vla.npz \\
        --prior       models/prior_model.pt \\
        --n_images    50 \\
        --n_samples   20 \\
        --device      cuda
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Per-source statistics from posterior samples
# ---------------------------------------------------------------------------

def morphology_confidence(samples: np.ndarray) -> float:
    """
    Mean pairwise Pearson correlation between posterior samples.

    1.0 → all samples agree on morphology (high confidence)
    ~0  → samples disagree (high morphological uncertainty)

    Parameters
    ----------
    samples : (S, H, W) — posterior draws for one island

    Returns
    -------
    conf : float in [-1, 1]
    """
    S = samples.shape[0]
    if S == 1:
        return 1.0
    flat    = samples.reshape(S, -1)          # (S, H*W)
    flat    = flat - flat.mean(axis=1, keepdims=True)
    norms   = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    normed  = flat / norms                    # (S, H*W)
    corr_mat = normed @ normed.T              # (S, S) — all pairwise correlations
    # Average upper triangle (excluding diagonal)
    n_pairs = S * (S - 1) / 2
    total   = (corr_mat.sum() - np.trace(corr_mat)) / 2.0
    return float(total / (n_pairs + 1e-12))


def source_stats(
    samples      : np.ndarray,  # (S, H_i, W_i)
    noise_thresh : float = 3.0, # detection: peak > thresh * noise_std
    noise_std    : float = 0.05,
) -> dict:
    """
    Compute per-source statistics from S posterior samples.

    Returns a dict with all catalogue columns for one source.
    """
    S = samples.shape[0]

    # Integrated flux (positive only, matching flux_recovery convention)
    flux_per_sample = np.clip(samples, 0, None).sum(axis=(1, 2))    # (S,)

    # Peak flux and position per sample
    flat_idx = samples.reshape(S, -1).argmax(axis=1)                 # (S,)
    H, W     = samples.shape[1], samples.shape[2]
    peak_rows = flat_idx // W
    peak_cols = flat_idx % W
    peak_flux = samples.reshape(S, -1)[np.arange(S), flat_idx]      # (S,)

    # Detection probability: fraction of samples where peak > threshold
    detected  = (peak_flux > noise_thresh * noise_std).astype(float)

    return {
        "flux_mean":      float(flux_per_sample.mean()),
        "flux_std":       float(flux_per_sample.std()),
        "peak_flux_mean": float(peak_flux.mean()),
        "peak_flux_std":  float(peak_flux.std()),
        "peak_row_mean":  float(peak_rows.mean()),
        "peak_row_std":   float(peak_rows.std()),
        "peak_col_mean":  float(peak_cols.mean()),
        "peak_col_std":   float(peak_cols.std()),
        "detection_prob": float(detected.mean()),
        "morphology_conf": morphology_confidence(samples),
    }


# ---------------------------------------------------------------------------
# Field-level processing
# ---------------------------------------------------------------------------

def process_field(
    dirty      : np.ndarray,   # (H, W)
    solver,                     # DPSSolver or AmortisedSolver
    n_samples  : int,
    noise_std  : float,
    sigma_thresh: float,
    device     : str,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Detect islands, run posterior sampling on each, return catalogue rows
    and the (mean_field, std_field) uncertainty maps.

    Returns
    -------
    rows       : list of dicts, one per detected source
    mean_field : (H, W) — posterior mean for the whole field
    std_field  : (H, W) — posterior std (uncertainty map)
    """
    from mad_clean.detection import IslandDetector

    H, W = dirty.shape
    detector = IslandDetector(
        sigma_thresh=sigma_thresh,
        min_island=9,
        atom_size=15,
        device=device,
    )

    dirty_t = torch.from_numpy(dirty).float().to(device)
    bboxes, rms = detector.detect(dirty_t)
    print(f"  IslandDetector: {len(bboxes)} islands  (rms={rms:.4f})")

    mean_field = np.zeros((H, W), dtype=np.float32)
    std_field  = np.zeros((H, W), dtype=np.float32)
    rows       = []

    for k, (r0, r1, c0, c1) in enumerate(bboxes):
        island  = dirty[r0:r1, c0:c1]
        island_t = torch.from_numpy(island).float()

        # Posterior samples for this island: (S, H_i, W_i)
        samples = solver.sample_all(island_t, n_samples=n_samples).cpu().numpy()

        # Per-source statistics
        stats = source_stats(samples, noise_std=noise_std)
        stats.update({
            "source_id": k,
            "bbox_r0": r0, "bbox_r1": r1,
            "bbox_c0": c0, "bbox_c1": c1,
            # Absolute peak position in the full field
            "peak_row_abs_mean": stats["peak_row_mean"] + r0,
            "peak_col_abs_mean": stats["peak_col_mean"] + c0,
        })
        rows.append(stats)

        # Paint into field uncertainty maps
        mean_field[r0:r1, c0:c1] = samples.mean(axis=0)
        std_field [r0:r1, c0:c1] = samples.std(axis=0)

        print(f"  Source {k:3d}: flux={stats['flux_mean']:.3f}±{stats['flux_std']:.3f}  "
              f"det_prob={stats['detection_prob']:.2f}  "
              f"morph_conf={stats['morphology_conf']:.3f}")

    return rows, mean_field, std_field


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_field(
    dirty      : np.ndarray,
    mean_field : np.ndarray,
    std_field  : np.ndarray,
    rows       : list[dict],
    out_path   : Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    vmax = float(np.percentile(dirty, 99.5))

    for ax, data, title in zip(
        axes,
        [dirty, mean_field, std_field],
        ["Dirty (input)", "Posterior mean", "Posterior std (uncertainty)"],
    ):
        ax.imshow(data, origin="lower", cmap="inferno",
                  vmin=0, vmax=vmax if title != "Posterior std (uncertainty)" else None)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Overlay source positions on mean field
    for row in rows:
        r, c = row["peak_row_abs_mean"], row["peak_col_abs_mean"]
        axes[1].plot(c, r, "w+", ms=8, mew=1.5)
        axes[1].text(c + 2, r + 2,
                     f"p={row['detection_prob']:.1f}", color="white", fontsize=6)

    plt.suptitle(f"{len(rows)} sources catalogued", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "source_id", "image_idx",
    "flux_mean", "flux_std",
    "peak_flux_mean", "peak_flux_std",
    "peak_row_abs_mean", "peak_col_abs_mean",
    "peak_row_std", "peak_col_std",
    "detection_prob", "morphology_conf",
    "bbox_r0", "bbox_r1", "bbox_c0", "bbox_c1",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-source uncertainty catalogue from DPS posteriors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="crumb_data/flow_pairs_vla.npz")
    p.add_argument("--prior",        default="models/prior_model.pt",
                   help="Prior FlowModel (PriorTrainer). Can also point to an "
                        "AmortisedPosteriorModel once Phase 2 is trained.")
    p.add_argument("--out",          default="logs/catalogue")
    p.add_argument("--idx",          type=int,   default=None,
                   help="Single image index to process. Overrides --n_images.")
    p.add_argument("--n_images",     type=int,   default=10,
                   help="Number of images to process (from start of data file).")
    p.add_argument("--n_samples",    type=int,   default=20,
                   help="Posterior draws per island.")
    p.add_argument("--dps_weight",   type=float, default=1.0)
    p.add_argument("--noise_std",    type=float, default=0.05)
    p.add_argument("--sigma_thresh", type=float, default=3.0,
                   help="IslandDetector detection threshold (σ units).")
    p.add_argument("--amortised",    action="store_true",
                   help="Use AmortisedSolver instead of DPSSolver (Phase 2).")
    p.add_argument("--amortised_model", default="models/amortised_posterior.pt",
                   help="Path to AmortisedPosteriorModel checkpoint.")
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        args.device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data     = np.load(args.data)
    dirty_all = data["dirty"].astype(np.float32)
    psf_norm  = data["psf_norm"].astype(np.float32)

    if args.idx is not None:
        indices = [args.idx]
    else:
        indices = list(range(min(args.n_images, len(dirty_all))))
    print(f"Processing {len(indices)} image(s)")

    # ── Build solver ───────────────────────────────────────────────────────────
    if args.amortised:
        from mad_clean.training.posterior import ConditionalFlowModel
        from mad_clean.solvers import AmortisedSolver
        cond_model = ConditionalFlowModel.load(args.amortised_model, device=args.device)
        solver = AmortisedSolver(
            cond_model,
            device    = args.device,
            n_samples = args.n_samples,
            n_steps   = 20,
        )
        print(f"Amortised solver: {solver}")
    else:
        from mad_clean.training.flow import FlowModel
        from mad_clean.solvers import DPSSolver
        prior  = FlowModel.load(args.prior, device=args.device)
        solver = DPSSolver(
            prior,
            psf_norm   = psf_norm,
            noise_std  = args.noise_std,
            n_steps    = 50,
            n_samples  = args.n_samples,
            dps_weight = args.dps_weight,
            device     = args.device,
        )
        print(f"DPS solver: {solver}")

    # ── Process images ─────────────────────────────────────────────────────────
    all_rows = []
    for img_i in indices:
        dirty = dirty_all[img_i]
        print(f"\nImage {img_i}:")

        rows, mean_field, std_field = process_field(
            dirty        = dirty,
            solver       = solver,
            n_samples    = args.n_samples,
            noise_std    = args.noise_std,
            sigma_thresh = args.sigma_thresh,
            device       = args.device,
        )

        for row in rows:
            row["image_idx"] = img_i
        all_rows.extend(rows)

        # Save per-image arrays and plot
        img_dir = out_dir / f"image_{img_i:04d}"
        img_dir.mkdir(exist_ok=True)
        np.save(img_dir / "mean_field.npy", mean_field)
        np.save(img_dir / "std_field.npy",  std_field)
        plot_field(dirty, mean_field, std_field, rows, img_dir / "field.png")

    # ── Write catalogue CSV ────────────────────────────────────────────────────
    csv_path = out_dir / "sources.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCatalogue → {csv_path}  ({len(all_rows)} sources)")
    print(f"Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
