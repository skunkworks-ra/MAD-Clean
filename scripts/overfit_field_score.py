"""Overfit gate: can the field-posterior score prior learn SOURCE structure?

The overnight full-train prior reproduced the diffuse floor but generated no
compact sources (top-0.1% flux fraction ~0.07 vs corpus ~0.20).  The open
question is whether that is a capacity wall (score model can't represent sharp
sources → the field-posterior program has no scope) or a loss-balance artifact
(sources are ~0.1% of pixels, so pixel-uniform DSM gives them ~0.1% of the
gradient and they are simply not learned).

This gate decides it.  On a small FIXED set of source-prominent fields it
overfits the same EDM denoiser under two losses:

  plain : pixel-uniform DSM (the overnight recipe)
  flux  : flux-weighted DSM (per-pixel weight ∝ clean flux, mean-1)

then GENERATES unconditional samples from each (annealed prior-only Langevin —
generation, not denoising, because a source pixel is ~+10σ in log-sky and is
trivially high-SNR per pixel, so denoising hides the failure; the prior score
away from data is where 'did it learn sources' actually lives) and measures
source reproduction by compactness.

Verdict
-------
  flux reproduces sources (compactness ≈ training), plain does not
      → capacity is fine; the overnight failure was the loss weighting → SCOPE.
  both fail to reproduce sources even when overfitting a handful of fields
      → representation wall → NO SCOPE (stop).

Example
-------
    pixi run -e gpu python scripts/overfit_field_score.py \\
        --size 128 --n_fields 8 --steps 4000 --device cuda \\
        --fig results/overfit_field_gate.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mad_clean.data.field_sky import assemble_corpus_field
from mad_clean.imaging.langevin import annealed_langevin, geometric_sigma_schedule
from mad_clean.imaging.score import EDMDenoiser, UNet, edm_loss

_S_FLOOR = 1e-8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Overfit gate: can the prior learn sources?")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--n_fields", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_clip", type=float, default=1e3,
                   help="Cap on the flux weight so a single source pixel can't dominate.")
    p.add_argument("--n_gen", type=int, default=6, help="Unconditional samples per model.")
    p.add_argument("--diffuse_flux", type=float, default=0.05,
                   help="Total diffuse flux (Jy); low → faint background, sources dominate.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig", type=str, default="results/overfit_field_gate.png")
    return p.parse_args(argv)


def compactness(s: np.ndarray) -> float:
    """Fraction of total flux in the brightest 0.1% of pixels (the source proxy)."""
    f = s.ravel()
    k = max(1, int(0.001 * f.size))
    return float(np.sort(f)[-k:].sum() / f.sum())


def make_fields(n, size, seed, diffuse_flux):
    """Fixed source-prominent corpus fields: bright points + ridges on a FAINT
    diffuse background, so the sources clearly dominate the flux (a meaningful
    'can it reproduce sources' target).  Points are single-pixel deltas (the
    sharpest possible feature); ridges are extended shapes."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        out[i] = assemble_corpus_field(
            size=size,
            diffuse_flux_jy=diffuse_flux,
            n_points=(8, 15),
            point_flux_range_jy=(1e-2, 1e-1),   # bright, clearly visible
            n_ridges=(1, 2),
            ridge_flux_range_jy=(1e-2, 1e-1),
            rng=rng)
    return out


def overfit(f0, pixel_weight, steps, base, lr, device, seed):
    """Overfit one EDM denoiser on the fixed batch; return the trained model."""
    torch.manual_seed(seed)
    model = EDMDenoiser(UNet(in_ch=1, base=base), sigma_data=1.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = edm_loss(model, f0, generator=gen, pixel_weight=pixel_weight)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"    step {step:>5d}  loss {float(loss):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def generate(model, n, size, mu, tau, device, seed):
    """Unconditional samples via annealed prior-only Langevin → linear sky s."""
    gen = torch.Generator(device=device).manual_seed(seed)
    sig = geometric_sigma_schedule(5.0, 0.01, 40, device=device)
    prior = lambda f, s: model.score(f.unsqueeze(1), s).squeeze(1)
    init = torch.randn(n, size, size, generator=gen, device=device)
    f = annealed_langevin(prior, init, sig, n_steps_per_level=40, step_size=2e-5,
                          generator=gen, project=lambda x: x.clamp(-20, 20))
    return torch.exp(mu + tau * f).cpu().numpy()


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    # ── fixed source-prominent training set ──────────────────────────────────
    s_fields = make_fields(args.n_fields, args.size, args.seed, args.diffuse_flux)
    train_compact = np.mean([compactness(s) for s in s_fields])
    print(f"Training set: {args.n_fields} fields @ {args.size}²  "
          f"compactness(top-0.1% flux)={train_compact:.3f}")
    if train_compact < 0.10:
        print(f"!! ABORT: training compactness {train_compact:.3f} < 0.10 — the "
              f"fields are not source-prominent, so 'can it reproduce sources' is "
              f"vacuous. Lower --diffuse_flux / brighten points before trusting a "
              f"verdict.")
        return

    f_log = np.log(np.maximum(s_fields, _S_FLOOR)).astype(np.float32)
    mu, tau = float(f_log.mean()), float(f_log.std() + 1e-8)
    f0 = ((torch.from_numpy(f_log) - mu) / tau)[:, None].to(device)  # (N,1,H,W)
    print(f"  mu={mu:.3f}  tau={tau:.3f}")

    # Flux weight ∝ clean s, mean-1 per field, capped (the rebalancing lever).
    s_t = torch.from_numpy(s_fields)[:, None].to(device)
    w = s_t / s_t.mean(dim=(2, 3), keepdim=True)
    w = w.clamp(max=args.weight_clip)

    results = {}
    for name, pw in [("plain", None), ("flux", w)]:
        print(f"\n[{name}] overfitting {args.steps} steps "
              f"({'pixel-uniform' if pw is None else 'flux-weighted'} DSM)…")
        model = overfit(f0, pw, args.steps, args.base_channels, args.lr,
                        device, args.seed + 1)
        gen = generate(model, args.n_gen, args.size, mu, tau, device, args.seed + 2)
        comp = np.mean([compactness(g) for g in gen])
        results[name] = (gen, comp)
        print(f"[{name}] generated compactness={comp:.3f}  "
              f"(training {train_compact:.3f})")

    # ── verdict ──────────────────────────────────────────────────────────────
    pc, fc = results["plain"][1], results["flux"][1]
    print("\n── overfit gate verdict ──")
    print(f"  training compactness   {train_compact:.3f}")
    print(f"  plain  generated       {pc:.3f}   ({pc/train_compact:.0%} of training)")
    print(f"  flux   generated       {fc:.3f}   ({fc/train_compact:.0%} of training)")
    if fc >= 0.6 * train_compact and fc > 1.5 * pc:
        print("  → SCOPE: flux-weighting recovers sources the plain loss drops; "
              "capacity is fine, the overnight failure was the weighting.")
    elif fc < 0.4 * train_compact and pc < 0.4 * train_compact:
        print("  → NO SCOPE: neither loss reproduces sources even on overfit; "
              "representation wall.")
    else:
        print("  → INCONCLUSIVE: partial source recovery; read the figure.")

    _save_fig(args.fig, s_fields, results, train_compact, pc, fc)


def _save_fig(path, s_fields, results, tc, pc, fc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(6, s_fields.shape[0], results["plain"][0].shape[0])
    rows = [("training", s_fields, tc),
            ("plain gen", results["plain"][0], pc),
            ("flux gen", results["flux"][0], fc)]
    lt = np.log10(np.maximum(s_fields, 1e-12))
    vmax = lt.max()
    vmin = vmax - 4.0
    fig, ax = plt.subplots(3, n, figsize=(2.2 * n, 7))
    for r, (label, arr, comp) in enumerate(rows):
        for c in range(n):
            a = ax[r, c]
            a.imshow(np.log10(np.maximum(arr[c], 1e-12)), origin="lower",
                     cmap="inferno", vmin=vmin, vmax=vmax)
            a.axis("off")
            if c == 0:
                a.set_ylabel(f"{label}\nc={comp:.2f}", rotation=0, labelpad=40,
                             va="center", fontsize=10)
    fig.suptitle("Overfit gate: log10 sky — training / plain-DSM gen / flux-DSM gen",
                 fontsize=12)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    print(f"\nfigure → {path}")


if __name__ == "__main__":
    main()
