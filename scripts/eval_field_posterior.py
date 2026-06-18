"""Drive the annealed-ULA field-posterior sampler on one simulated field
(Fork A, design step 3, field_posterior_design.md §2.3 / §4.5).

This is the first end-to-end run of ``sample_field_posterior`` against the
*trained* score prior: one true sky from the corpus generator, one real G55
dirty beam, an honest forward-modelled dirty image, then an ensemble of
annealed-Langevin sweeps.  It reports the diagnostics that decide whether the
sampler is behaving, before the full coverage / SBC test (§2.8):

  1. Posterior mean vs ground truth     — RMSE, peak-flux ratio, total-flux ratio.
  2. Per-pixel posterior std            — the uncertainty map.
  3. Posterior-predictive whiten check  — r = d - A·mean; std(r)/σ_n and reduced
     χ² should sit near 1.  A structured residual is the §4.5 OOD alarm: a prior
     that cannot represent the data shows up here, not as a confident wrong mean.

Honesty flags carried from the engine (langevin.py): the data score is the
clean-data gradient added at full strength at every annealing level (§2.3,
exact only in the slow-annealing limit), and the image-domain white-noise model
is the §2.7-A0 approximation.  This run is a smoke test, not a calibration claim.

Example
-------
    pixi run -e gpu python scripts/eval_field_posterior.py \\
        --ckpt models/field_score.pt --size 512 --n_samples 16 \\
        --device cuda --fig results/field_posterior_smoke.png
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
from mad_clean.imaging.langevin import measure_data_scale, sample_field_posterior
from mad_clean.imaging.score import EDMDenoiser, UNet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Drive the field-posterior ULA sampler.")
    p.add_argument("--ckpt", type=str, default="models/field_score.pt")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=4,
                   help="Samples per GPU sweep; ensemble drawn in chunks to fit memory.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Sample with EMA weights (default; warm at 100k steps).")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--seed", type=int, default=2026,
                   help="Held-out seed (sky, PSF draw, sampler noise).")
    p.add_argument("--peak_snr", type=float, default=50.0,
                   help="σ_n set so peak(A s*) / σ_n = peak_snr.")
    # Sampler knobs (defaults = langevin.py defaults; the GOTCHA band).
    p.add_argument("--step_size", type=float, default=1e-5)
    p.add_argument("--sigma_max", type=float, default=5.0)
    p.add_argument("--sigma_min", type=float, default=1e-2)
    p.add_argument("--n_levels", type=int, default=20)
    p.add_argument("--n_steps_per_level", type=int, default=20)
    p.add_argument("--data_scale", type=str, default="auto",
                   help="Scalar on the data term. 'auto' = |prior|/|data| at the top "
                        "σ level (puts the data term on the prior score's scale); or a float.")
    p.add_argument("--fig", type=str, default="results/field_posterior_smoke.png")
    return p.parse_args(argv)


def load_model(ckpt: dict, use_ema: bool, device) -> EDMDenoiser:
    base = ckpt["config"]["base_channels"]
    model = EDMDenoiser(UNet(in_ch=1, base=base), sigma_data=ckpt["sigma_data"]).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    return model


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mu, tau = float(ckpt["mu"]), float(ckpt["tau"])
    space = ckpt.get("space", "log")
    print(f"Checkpoint {args.ckpt}  step={ckpt.get('step', '?')}  space={space}  "
          f"mu={mu:.4g} tau={tau:.4g}  ema={args.use_ema}  device={device}")
    model = load_model(ckpt, args.use_ema, device)

    # ── ground-truth sky + real dirty beam ───────────────────────────────────
    s_true_np = assemble_corpus_field(size=args.size, rng=np_rng).astype(np.float32)
    psf_bank = load_g55_psf_bank(repo_root=_REPO_ROOT, target_size=args.size,
                                 rotation_augment=True)
    psf_np, psf_meta = psf_bank.sample(np_rng)
    print(f"PSF: {Path(psf_meta['source_path']).parent.name}  "
          f"rot={psf_meta['rotation_deg']}°   bank size={len(psf_bank)}")

    s_true = torch.from_numpy(s_true_np).to(device)
    psf = torch.from_numpy(psf_np).to(device)
    fwd = ImageDomainForward(psf)

    # ── honest forward-modelled dirty image ──────────────────────────────────
    d_clean = fwd.forward(s_true)
    sigma_n = float(d_clean.abs().max()) / args.peak_snr
    gen = torch.Generator(device=device).manual_seed(args.seed)
    d = fwd.make_dirty(s_true, noise_std=sigma_n, generator=gen)
    print(f"σ_n={sigma_n:.3e}  (peak_snr={args.peak_snr})   "
          f"s_true: peak={float(s_true.max()):.3e} total={float(s_true.sum()):.3e}")

    # ── data-term scale (the coupling fix) ───────────────────────────────────
    if args.data_scale == "auto":
        with torch.no_grad():
            data_scale = measure_data_scale(
                fwd, model, d, mu, tau, sigma_n, args.sigma_max, space)
        print(f"data_scale=auto → λ={data_scale:.3e}  "
              f"(|prior|/|data| at σ_max, cold init, space={space})")
    else:
        data_scale = float(args.data_scale)
        print(f"data_scale={data_scale:.3e}  (user)")

    # ── ensemble of posterior samples (chunked, no autograd graph) ───────────
    chunks = []
    drawn = 0
    with torch.no_grad():
        while drawn < args.n_samples:
            nb = min(args.batch, args.n_samples - drawn)
            chunks.append(sample_field_posterior(
                fwd, model, d, mu, tau, sigma_n,
                n_samples=nb,
                sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                n_levels=args.n_levels, n_steps_per_level=args.n_steps_per_level,
                step_size=args.step_size, data_scale=data_scale, space=space,
                generator=gen,
            ))
            drawn += nb
            print(f"  drawn {drawn}/{args.n_samples}", flush=True)
    samples = torch.cat(chunks, dim=0)
    n_nan = int(torch.isnan(samples).any(dim=(1, 2)).sum())
    if n_nan:
        print(f"!! {n_nan}/{args.n_samples} samples contain NaN "
              f"(log-space stiffness — drop step_size or raise peak_snr).")
    finite = samples[~torch.isnan(samples).any(dim=(1, 2))]
    if finite.numel() == 0:
        print("All samples NaN; aborting diagnostics.")
        return

    post_mean = finite.mean(dim=0)
    post_std = finite.std(dim=0)

    # ── diagnostics ──────────────────────────────────────────────────────────
    rmse = float((post_mean - s_true).pow(2).mean().sqrt())
    peak_ratio = float(post_mean.max() / s_true.max())
    flux_ratio = float(post_mean.sum() / s_true.sum())
    resid = d - fwd.forward(post_mean)
    whiten = float(resid.std() / sigma_n)
    chi2_red = float((resid.pow(2).mean()) / (sigma_n ** 2))

    print("\n── posterior diagnostics ──")
    print(f"  usable samples       {finite.shape[0]}/{args.n_samples}")
    print(f"  RMSE(mean, truth)    {rmse:.3e}")
    print(f"  peak flux ratio      {peak_ratio:.3f}   (1 = perfect)")
    print(f"  total flux ratio     {flux_ratio:.3f}   (1 = perfect)")
    print(f"  mean post. std       {float(post_std.mean()):.3e}")
    print(f"  whiten std(r)/σ_n    {whiten:.3f}   (≈1 = data fit)")
    print(f"  reduced χ²           {chi2_red:.3f}   (≈1 = data fit)")

    if args.fig:
        _save_fig(args.fig, s_true, d, post_mean, post_std)


def _save_fig(path, s_true, d, post_mean, post_std):
    """truth / mean / std in log10 (the source is a few bright pixels over a
    ~1e-6 floor — invisible on a linear stretch).  truth and mean SHARE the
    log colour scale so 'did the mean recover the source' is read directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = s_true.detach().cpu().numpy()
    m = post_mean.detach().cpu().numpy()
    sd = post_std.detach().cpu().numpy()
    dd = d.detach().cpu().numpy()
    lt = np.log10(np.maximum(t, 1e-12))
    lm = np.log10(np.maximum(m, 1e-12))
    vmax = lt.max()
    vmin = vmax - 4.0  # 4 decades of dynamic range below the true peak

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    p0 = ax[0].imshow(lt, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax[0].set_title("true sky  log10 s")
    p1 = ax[1].imshow(dd, origin="lower", cmap="RdBu_r",
                      vmin=-np.abs(dd).max(), vmax=np.abs(dd).max())
    ax[1].set_title("dirty d  (linear)")
    p2 = ax[2].imshow(lm, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax[2].set_title("posterior mean  log10 s  (shared scale)")
    p3 = ax[3].imshow(np.log10(np.maximum(sd, 1e-12)), origin="lower", cmap="viridis")
    ax[3].set_title("posterior std  log10")
    for a, p in zip(ax, (p0, p1, p2, p3)):
        a.axis("off")
        fig.colorbar(p, ax=a, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    print(f"\nfigure → {path}")


if __name__ == "__main__":
    main()
