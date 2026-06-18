"""Per-patch, in-loop overfit gate for the field-posterior prior.

The agreed goal (2026-06-14): make a *non-parametric* overfit gate pass.  Overfit
a small fixed set of source-prominent patches with the field-posterior score
prior, then show the sources SURVIVE when the prior is run inside the explicit-
PSF loop (prior score + data score), per patch.  Approximate is fine; the outer
major/minor residual loop absorbs the rest.

Why this gate and not ``overfit_field_score.py``
-------------------------------------------------
``overfit_field_score`` tested *unconditional generation* — can the prior dream
sources from pure noise.  That is strictly harder than what the loop needs and
it failed (plain-DSM generation came out empty).  The loop never asks the prior
to invent the source: ``reconstruct_point_source`` showed the likelihood alone
(Landweber) recovers an off-centre point exactly.  The loop's real ask is weaker
— given the data, the prior must regularise WITHOUT destroying the source.  This
gate tests exactly that, and it overfits the prior on the very patches that hold
the sources, so a failure here is the cleanest possible indictment of the
representation: the prior cannot hold a source it has memorised.

Levers (the "percentages are knobs" point)
-------------------------------------------
- ``--diffuse_flux``  : background brightness; low = source-prominent.
- ``--pixel_weight``  : flux-proportional DSM weight (the loss lever).
- ``--steps``         : overfit budget.
- ``--data_scale``    : data-term coupling (default = noise_std, as in
                        reconstruct_point_source).
This first version runs in LOG-sky (the current pipeline).  The representation
lever (linear-sky, no obligatory diffuse floor) is the next step, driven by what
this gate shows; see field_posterior_design.md.

Example
-------
    pixi run -e gpu python scripts/overfit_field_inloop.py \\
        --size 128 --n_patches 8 --steps 4000 --device cuda \\
        --fig results/overfit_field_inloop.png
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
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.imaging.langevin import (
    measure_data_scale,
    sample_field_posterior,
)
from mad_clean.imaging.score import EDMDenoiser, UNet, edm_loss

_S_FLOOR = 1e-8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-patch in-loop overfit gate.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--space", choices=["log", "linear"], default="log",
                   help="Prior/sampler representation. 'linear' keeps a sparse "
                        "sky sparse (no log floor); the representation lever.")
    p.add_argument("--morphology", choices=["points", "extended"],
                   default="points",
                   help="'points' = sharp delta sources; 'extended' = "
                        "ridge/arc-dominated patches (the Cyg-A-like test of "
                        "whether linear positivity rearranges flux over a "
                        "resolved region).")
    p.add_argument("--n_patches", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--diffuse_flux", type=float, default=0.05,
                   help="Total diffuse flux (Jy); low → sources dominate.")
    p.add_argument("--pixel_weight", action="store_true",
                   help="Use flux-proportional DSM weight (the loss lever).")
    p.add_argument("--weight_clip", type=float, default=1e3)
    p.add_argument("--peak_snr", type=float, default=100.0)
    p.add_argument("--n_landweber", type=int, default=2000)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--ckpt", type=str, default="",
                   help="If set, LOAD this trained prior and skip overfitting — "
                        "turns the gate into a held-out per-patch in-loop eval. "
                        "Space/mu/tau are read from the checkpoint.")
    p.add_argument("--data_scale", type=str, default="sigma_n",
                   help="Data-term coupling base: 'auto' (|prior|/|data|, the "
                        "principled scalar), 'sigma_n' (noise std), or a float.")
    p.add_argument("--data_scale_mult", type=float, default=1.0,
                   help="Multiplier on the data-term coupling. >1 lets "
                        "well-constrained data pull sampler-undershot spikes "
                        "back toward true flux (prior-vs-data balance knob).")
    p.add_argument("--patch_idx", type=int, default=0,
                   help="Which overfit patch to reconstruct in the loop.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig", type=str, default="results/overfit_field_inloop.png")
    return p.parse_args(argv)


# ── small reconstruction helpers (mirrors reconstruct_point_source) ───────────

def peak_rc(a: np.ndarray) -> tuple[int, int]:
    return tuple(int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))


def windowed_flux(a: np.ndarray, rc, half: int = 4) -> float:
    """Positive flux summed in a small window around (row, col)."""
    r0, c0 = rc
    H, W = a.shape
    r1, r2 = max(0, r0 - half), min(H, r0 + half + 1)
    c1, c2 = max(0, c0 - half), min(W, c0 + half + 1)
    return float(np.clip(a[r1:r2, c1:c2], 0, None).sum())


def landweber(fwd, d, n_iter):
    """Positive least-squares (likelihood-only) deconvolution, no prior."""
    L = float((fwd._H.abs() ** 2).max())
    eta = 1.0 / L
    s = torch.zeros(fwd.shape, device=d.device, dtype=d.dtype)
    for _ in range(n_iter):
        s = torch.relu(s + eta * fwd.adjoint(d - fwd.forward(s)))
    return s


# ── corpus + overfit ──────────────────────────────────────────────────────────

def make_patches(n, size, seed, diffuse_flux, morphology):
    """Fixed source-prominent patches.

    ``points``   : bright single-pixel deltas (the sharpest feature) plus an
                   occasional ridge — the point-survival test.
    ``extended`` : ridge/arc-dominated patches (limb-brightened shock fronts,
                   the G55 shell / Cyg-A-lobe-edge proxy) with at most a couple
                   of points — the resolved-flux-rearrangement test.

    Returns skies, per-patch point catalogs, and per-patch ridge layers (the
    extended support)."""
    rng = np.random.default_rng(seed)
    skies = np.empty((n, size, size), dtype=np.float32)
    catalogs, ridges = [], []
    if morphology == "extended":
        kw = dict(n_points=(0, 2), point_flux_range_jy=(1e-2, 5e-2),
                  n_ridges=(2, 3), ridge_flux_range_jy=(3e-2, 1e-1))
    else:
        kw = dict(n_points=(5, 10), point_flux_range_jy=(1e-2, 1e-1),
                  n_ridges=(0, 1), ridge_flux_range_jy=(1e-2, 1e-1))
    for i in range(n):
        sky, comp = assemble_corpus_field(
            size=size, diffuse_flux_jy=diffuse_flux, edge_margin=12,
            return_components=True, rng=rng, **kw)
        skies[i] = sky
        pts = comp["points"]
        rows, cols = np.nonzero(pts > 0)
        catalogs.append([(int(r), int(c), float(pts[r, c]))
                         for r, c in zip(rows, cols)])
        ridges.append(comp["ridges"])
    return skies, catalogs, ridges


def extended_fidelity(truth, recon, mask):
    """Flux ratio, rel-L2, and Pearson correlation on an extended support mask.

    flux_ratio : recovered positive flux / true flux inside the support — did
                 the right TOTAL flux land on the resolved structure.
    rel_l2     : ‖recon − truth‖ / ‖truth‖ on the support — shape fidelity; >1
                 means the reconstruction is further from the truth than zero is
                 (structure lost/fragmented).
    corr       : spatial pattern match on the support."""
    t = truth[mask]
    r = np.clip(recon, 0.0, None)[mask]
    tf = float(t.sum())
    flux_ratio = float(r.sum() / tf) if tf > 0 else float("nan")
    rel_l2 = float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))
    if t.std() > 0 and r.std() > 0:
        corr = float(np.corrcoef(t, r)[0, 1])
    else:
        corr = float("nan")
    return flux_ratio, rel_l2, corr


def overfit(f0, pixel_weight, steps, base, lr, device, seed):
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


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    # ── fixed source-prominent patches ───────────────────────────────────────
    skies, catalogs, ridges = make_patches(
        args.n_patches, args.size, args.seed, args.diffuse_flux, args.morphology)
    print(f"{args.n_patches} {args.morphology} patches @ {args.size}²  "
          f"diffuse_flux={args.diffuse_flux}  "
          f"points/patch={[len(c) for c in catalogs]}")

    if args.ckpt:
        # Held-out eval: load a trained prior, do not overfit. The patches above
        # are fresh draws of the (infinite) corpus, so they are held out from
        # training by construction.
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        space = ckpt.get("space", "log")
        mu, tau = float(ckpt["mu"]), float(ckpt["tau"])
        model = EDMDenoiser(UNet(in_ch=1, base=ckpt["config"]["base_channels"]),
                            sigma_data=ckpt["sigma_data"]).to(device)
        model.load_state_dict(ckpt.get("ema", ckpt["model"]))
        model.eval()
        print(f"\n[held-out eval] loaded prior {args.ckpt}  space={space}  "
              f"mu={mu:.4g} tau={tau:.4g}  step={ckpt.get('step', '?')} — no overfit")
    else:
        space = args.space
        if space == "log":
            field = np.log(np.maximum(skies, _S_FLOOR)).astype(np.float32)
        else:  # linear sky: sparse signal stays large-magnitude, no dense floor
            field = skies.astype(np.float32)
        mu, tau = float(field.mean()), float(field.std() + 1e-8)
        f0 = ((torch.from_numpy(field) - mu) / tau)[:, None].to(device)
        print(f"  {space}-sky standardisation: mu={mu:.4g}  tau={tau:.4g}")

        pw = None
        if args.pixel_weight:
            s_t = torch.from_numpy(skies)[:, None].to(device)
            pw = (s_t / s_t.mean(dim=(2, 3), keepdim=True)).clamp(max=args.weight_clip)

        print(f"\n[overfit] {args.steps} steps "
              f"({'flux-weighted' if pw is not None else 'pixel-uniform'} DSM)…")
        model = overfit(f0, pw, args.steps, args.base_channels, args.lr,
                        device, args.seed + 1)

    # ── reconstruct one patch inside the loop ────────────────────────────────
    i = args.patch_idx
    s_true = torch.from_numpy(skies[i]).to(device)
    cat = catalogs[i]

    bank = load_g55_psf_bank(repo_root=_REPO_ROOT, target_size=args.size)
    psf_np, meta = bank.sample(np.random.default_rng(args.seed))
    fwd = ImageDomainForward(torch.from_numpy(psf_np).to(device))
    print(f"\nPSF: {Path(meta['source_path']).parent.name} "
          f"rot={meta['rotation_deg']}°  reconstructing patch {i} "
          f"({len(cat)} points)")

    d_clean = fwd.forward(s_true)
    sigma_n = float(d_clean.abs().max()) / args.peak_snr
    gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    d = fwd.make_dirty(s_true, noise_std=sigma_n, generator=gen)
    if args.data_scale == "auto":
        base_scale = measure_data_scale(fwd, model, d, mu, tau, sigma_n, space=space)
        base_label = "auto |prior|/|data|"
    elif args.data_scale == "sigma_n":
        base_scale = sigma_n
        base_label = "sigma_n"
    else:
        base_scale = float(args.data_scale)
        base_label = "user"
    data_scale = base_scale * args.data_scale_mult
    print(f"  noise_std={sigma_n:.3e}  data_scale={data_scale:.3e} "
          f"(base {base_scale:.3e} [{base_label}] × mult {args.data_scale_mult})")

    with torch.no_grad():
        s_lik = landweber(fwd, d, args.n_landweber)
        samp = sample_field_posterior(
            fwd, model, d, mu, tau, sigma_n, n_samples=args.n_samples,
            data_scale=data_scale, space=space, generator=gen)
    s_post = samp.mean(0)

    t = s_true.cpu().numpy()
    lik = s_lik.cpu().numpy()
    post = s_post.cpu().numpy()

    # ── per-point survival (the sharp-source test) ───────────────────────────
    point_pass = None
    if cat:
        print("\n  per-source windowed flux (half=4 px):")
        print(f"  {'(row,col)':>12s} {'true':>9s} {'likeli':>9s} {'post':>9s} "
              f"{'post/true':>9s}")
        ratios = []
        for (r, c, f) in sorted(cat, key=lambda x: -x[2]):
            ft = windowed_flux(t, (r, c))
            fl = windowed_flux(lik, (r, c))
            fp = windowed_flux(post, (r, c))
            ratio = fp / ft if ft > 0 else float("nan")
            ratios.append(ratio)
            print(f"  {f'({r},{c})':>12s} {ft:9.3e} {fl:9.3e} {fp:9.3e} {ratio:9.2f}")
        pr, pc = peak_rc(post)
        near = min(((pr - r) ** 2 + (pc - c) ** 2) ** 0.5 for r, c, _ in cat)
        med_ratio = float(np.nanmedian(ratios))
        print(f"\n  posterior argmax ({pr},{pc}); nearest true source {near:.1f} px")
        print(f"  median post/true windowed-flux ratio: {med_ratio:.2f}")
        point_pass = 0.3 <= med_ratio <= 3.0 and near <= 3.0

    # ── extended-support fidelity (the resolved-flux-rearrangement test) ──────
    ext_pass = None
    ridge = ridges[i]
    if ridge.max() > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(ridge > 1e-3 * ridge.max(), iterations=3)
        print(f"\n  extended-support fidelity (ridge mask, {int(mask.sum())} px):")
        print(f"  {'':10s} {'flux_ratio':>10s} {'rel_l2':>8s} {'corr':>6s}")
        for name, rec in [("likelihood", lik), ("posterior", post)]:
            fr, rl, cr = extended_fidelity(t, rec, mask)
            print(f"  {name:10s} {fr:10.2f} {rl:8.2f} {cr:6.2f}")
        fr, rl, cr = extended_fidelity(t, post, mask)
        ext_pass = (0.3 <= fr <= 3.0) and (cr >= 0.5)

    # Verdict from the test matching the morphology (pass lines are knobs).
    verdict = ext_pass if args.morphology == "extended" else point_pass
    label = "extended structure" if args.morphology == "extended" else "sources"
    if verdict is None:
        print(f"\n  → no {label} to score")
    else:
        print(f"\n  → {label.upper()} "
              f"{'SURVIVE' if verdict else 'LOST'} (in-loop overfit gate)")

    _save_fig(args.fig, t, d.cpu().numpy(), lik, post, cat)


def _save_fig(path, s_true, d, s_lik, s_post, cat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def L(a):
        return np.log10(np.maximum(a, 1e-12))

    panels = [("truth  log10", L(s_true), "inferno"),
              ("dirty  (linear)", d, "RdBu_r"),
              ("likelihood  log10", L(s_lik), "inferno"),
              ("overfit posterior mean  log10", L(s_post), "inferno")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, (title, im, cmap) in zip(ax, panels):
        kw = dict(vmin=-np.abs(im).max(), vmax=np.abs(im).max()) if cmap == "RdBu_r" else {}
        m = a.imshow(im, origin="lower", cmap=cmap, **kw)
        a.set_title(title); a.axis("off")
        for r, c, _ in cat:
            a.plot(c, r, "c+", ms=8, mew=1.0)
        fig.colorbar(m, ax=a, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    print(f"\nfigure → {path}")


if __name__ == "__main__":
    main()
