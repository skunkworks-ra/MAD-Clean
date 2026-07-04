"""Stage-0 census of the realistic (T-RECS) corpus: how many distinct true
sources sit in each patch, and how many *single-source* islands the corpus can
yield for PSFCondFlow training.

This is a measurement, not an extraction — it commits no GPU time and writes no
cutouts.  It answers the one open risk before realistic-source training: the
T-RECS density tuning was deferred, so the corpus may be too sparse to yield
enough clean single-source islands (see project_psf_condflow_validated).

Sources are counted on the clean `sky` stack (non-negative truth), thresholded
per-patch at a fraction of that patch's peak and connected-component labelled.
This is the truth census; the deployment detector (IslandDetector on the dirty)
is a separate, later concern.

    pixi run -e gpu python scripts/corpus_island_census.py \
        --stacks /mnt/Data/Data/corpus_stacks/train
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Census true sources per corpus patch.")
    p.add_argument("--stacks", type=str, required=True,
                   help="corpus_stacks dir (contains sky.npy, field_id.npy).")
    p.add_argument("--peak_frac", type=float, default=0.05,
                   help="Per-patch detection threshold as a fraction of the "
                        "patch peak. Sources below this of the brightest source "
                        "in the same patch are not counted as separate islands.")
    p.add_argument("--min_pix", type=int, default=2,
                   help="Minimum connected area (px) to count as a source.")
    p.add_argument("--extent", action="store_true",
                   help="Measure the central source's spatial extent per patch "
                        "and rank the most extended (fluffy) fields, with a figure.")
    p.add_argument("--top", type=int, default=9,
                   help="Number of most-extended patches to show in --extent mode.")
    p.add_argument("--fig", type=str, default="results/corpus_extent_top.png")
    return p.parse_args(argv)


def extent_census(args):
    """Rank patches by the central source's extent (the fluffy CFD sources).

    For each patch: take the connected component (above peak_frac*peak) that
    contains the brightest pixel, and report its area (px), effective radius
    sqrt(area/pi), and concentration peak/sum (low = diffuse/fluffy).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stacks = Path(args.stacks)
    sky = np.load(stacks / "sky.npy", mmap_mode="r")
    field_id = np.asarray(np.load(stacks / "field_id.npy", mmap_mode="r"))
    n = sky.shape[0]
    structure = np.ones((3, 3), dtype=bool)

    area = np.zeros(n); conc = np.ones(n)
    for i in range(n):
        patch = np.asarray(sky[i], dtype=np.float32)
        peak = float(patch.max())
        if peak <= 0:
            continue
        labels, nlab = ndimage.label(patch > args.peak_frac * peak, structure=structure)
        if nlab == 0:
            continue
        pk = np.unravel_index(int(patch.argmax()), patch.shape)
        lab = labels[pk]
        comp = labels == lab
        a = int(comp.sum())
        area[i] = a
        conc[i] = peak / float(patch[comp].sum())

    order = np.argsort(-area)
    print(f"[extent] stacks={stacks}  patches={n}  peak_frac={args.peak_frac}")
    print(f"[extent] most extended (fluffy) central sources:")
    print(f"  {'rank':>4} {'field':>6} {'area_px':>8} {'eff_r_px':>9} {'peak/sum':>9}")
    for r, i in enumerate(order[:args.top]):
        print(f"  {r:>4} {int(field_id[i]):>6} {int(area[i]):>8} "
              f"{np.sqrt(area[i]/np.pi):>9.1f} {conc[i]:>9.4f}")

    k = min(args.top, n)
    cols = int(np.ceil(np.sqrt(k))); rows = int(np.ceil(k / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.6*cols, 2.6*rows))
    ax = np.atleast_1d(ax).ravel()
    for j in range(rows*cols):
        ax[j].axis("off")
    for j, i in enumerate(order[:k]):
        p = np.asarray(sky[i], dtype=np.float32)
        ax[j].imshow(p, origin="lower", cmap="inferno",
                     vmax=np.percentile(p[p > 0], 99.5) if (p > 0).any() else 1)
        ax[j].set_title(f"f{int(field_id[i])} a={int(area[i])}", fontsize=8)
    fig.suptitle("Most extended central sources (corpus truth)", fontsize=11)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120); plt.close(fig)
    print(f"[extent] figure → {args.fig}")


def main(argv=None):
    args = parse_args(argv)
    if args.extent:
        extent_census(args)
        return
    stacks = Path(args.stacks)
    sky = np.load(stacks / "sky.npy", mmap_mode="r")          # (N, H, W)
    field_id = np.load(stacks / "field_id.npy", mmap_mode="r")  # (N,)
    n = sky.shape[0]

    per_patch_counts = np.zeros(n, dtype=np.int32)
    structure = np.ones((3, 3), dtype=bool)   # 8-connectivity

    for i in range(n):
        patch = np.asarray(sky[i], dtype=np.float32)
        peak = float(patch.max())
        if peak <= 0.0:
            continue
        mask = patch > args.peak_frac * peak
        labels, nlab = ndimage.label(mask, structure=structure)
        if nlab == 0:
            continue
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, nlab + 1))
        per_patch_counts[i] = int((np.asarray(sizes) >= args.min_pix).sum())

    hist = Counter(per_patch_counts.tolist())
    empty = hist.get(0, 0)
    single = hist.get(1, 0)
    multi = n - empty - single
    total_sources = int(per_patch_counts.sum())
    n_fields = len(np.unique(np.asarray(field_id)))

    print(f"[census] stacks={stacks}")
    print(f"[census] patches={n}  fields={n_fields}  "
          f"peak_frac={args.peak_frac}  min_pix={args.min_pix}")
    print(f"[census] total true sources counted: {total_sources}  "
          f"(~{total_sources / max(n_fields, 1):.2f} per field)")
    print(f"[census] patches  empty={empty} ({100*empty/n:.1f}%)  "
          f"single={single} ({100*single/n:.1f}%)  "
          f"multi>=2={multi} ({100*multi/n:.1f}%)")
    print("[census] sources-per-patch histogram (count: n_patches):")
    for k in sorted(hist):
        print(f"           {k:>3d}: {hist[k]}")
    print(f"[census] => single-source islands directly available: {single}")
    print(f"[census] => (multi-source patches are splittable into more islands "
          f"by the deployment detector, not counted here)")


if __name__ == "__main__":
    main()
