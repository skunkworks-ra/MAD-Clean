"""Characterise the sparsity of the simulated sky.

The gradient-collapse argument rests on a factual premise: the clean sky a
prior must learn is a sparse, high-information signal sitting in a dense
background.  Before committing to any representation or loss change we measure
whether that premise actually holds for the sky we simulate, across three
regimes:

  1. ``points``        — pure point-source field (point_sky), the "few points,
                         noise, beam" sky.  This is the clean target in the
                         delta regime.
  2. ``corpus_default``— field_sky.assemble_corpus_field at its current
                         defaults (diffuse_flux_jy = 1.0): the diffuse-dominated
                         corpus the overnight prior was trained on.
  3. ``corpus_sparse`` — the same generator with the diffuse turned down, i.e.
                         a source-prominent sky.

For each regime we report, on the CLEAN sky (what the prior learns):

  - support fraction          : pixels above 1e-3 * peak / total pixels
  - top-0.1% flux fraction     : flux in the brightest 0.1% of pixels (the
                                "compactness" metric used in the field-posterior
                                eval; corpus ~0.20, diffuse prior generated ~0.07)
  - Gini coefficient           : 0 = uniform, 1 = all flux in one pixel
  - flux split                 : diffuse vs points vs ridges share of total flux

This is the §4.6 corpus-calibration diagnostic; rerun it whenever the corpus
statistics constants change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mad_clean.data.field_sky import assemble_corpus_field
from mad_clean.data.point_sky import generate_point_source_field


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative flux image (flattened)."""
    v = np.sort(x.ravel().astype(np.float64))
    n = v.size
    s = v.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * v)) / (n * s))


def _metrics(sky: np.ndarray) -> dict:
    flat = sky.ravel().astype(np.float64)
    n = flat.size
    total = flat.sum()
    peak = flat.max()
    support = float((flat > 1e-3 * peak).sum()) / n
    k = max(1, int(0.001 * n))  # top 0.1% of pixels
    top = np.partition(flat, n - k)[n - k:]
    top_frac = float(top.sum() / total) if total > 0 else 0.0
    return {
        "support_frac": support,
        "top0.1pct_flux_frac": top_frac,
        "gini": _gini(sky),
        "peak": float(peak),
        "total_flux": float(total),
        "dyn_range": float(peak / (total / n)) if total > 0 else 0.0,
    }


def _report(name: str, m: dict, extra: str = "") -> None:
    print(
        f"{name:16s}  support={m['support_frac']*100:7.3f}%  "
        f"top0.1%flux={m['top0.1pct_flux_frac']*100:6.2f}%  "
        f"gini={m['gini']:.3f}  "
        f"peak/mean={m['dyn_range']:8.1f}  {extra}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--n_fields", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sz = args.size
    print(f"# sky sparsity, size={sz}, n_fields={args.n_fields}\n")

    # --- 1. pure point field -------------------------------------------------
    ms = []
    for _ in range(args.n_fields):
        sky, cat = generate_point_source_field(
            size=sz, n_sources=(5, 30), rng=rng)
        ms.append(_metrics(sky))
    mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    _report("points", mean, "(clean delta sky, no diffuse)")

    # --- 2. corpus at current defaults (diffuse-dominated) -------------------
    ms, splits = [], []
    for _ in range(args.n_fields):
        sky, comp = assemble_corpus_field(
            size=sz, diffuse_flux_jy=1.0, return_components=True, rng=rng)
        ms.append(_metrics(sky))
        t = sky.sum()
        splits.append({k: float(comp[k].sum() / t) for k in comp})
    mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    sp = {k: float(np.mean([s[k] for s in splits])) for k in splits[0]}
    _report("corpus_default", mean,
            f"flux: diffuse={sp['diffuse']*100:.0f}% "
            f"points={sp['points']*100:.0f}% ridges={sp['ridges']*100:.0f}%")

    # --- 3. corpus, source-prominent (diffuse turned down) -------------------
    ms, splits = [], []
    for _ in range(args.n_fields):
        sky, comp = assemble_corpus_field(
            size=sz, diffuse_flux_jy=0.02, return_components=True, rng=rng)
        ms.append(_metrics(sky))
        t = sky.sum()
        splits.append({k: float(comp[k].sum() / t) for k in comp})
    mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    sp = {k: float(np.mean([s[k] for s in splits])) for k in splits[0]}
    _report("corpus_sparse", mean,
            f"flux: diffuse={sp['diffuse']*100:.0f}% "
            f"points={sp['points']*100:.0f}% ridges={sp['ridges']*100:.0f}%")


if __name__ == "__main__":
    main()
