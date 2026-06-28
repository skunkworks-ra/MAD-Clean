"""Overfit gate for PSFCondFlow.

Run BEFORE committing GPU time to full training.  Overfit a small fixed set
of single-source patches, then reconstruct one patch in the CFM loop.  If
sources do not survive here, the velocity net embedding or encoder wiring is
wrong; fix it before training.

Gate rule (psf_condflow_design.md §5):
  - Points:   median post/true windowed-flux >= 0.5, argmax within 3 px.
  - Extended: flux_ratio 0.5-2.0, corr >= 0.6 on the dilated ridge mask.

Example
-------
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --morphology points --n_patches 8 --steps 2000 --device cuda

Held-out eval (trained prior, no overfit):
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --ckpt models/psf_condflow.pt --morphology points --device cuda
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
import torch.nn.functional as F

from mad_clean.data.field_sky import assemble_corpus_field, render_ridge
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.models.mdn_asp import COND_DIM
from mad_clean.models.psf_condflow import PSFCondFlow, cfm_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PSFCondFlow overfit gate.")
    p.add_argument("--size",      type=int,   default=128)
    p.add_argument("--morphology", choices=["points", "extended", "gaussian"],
                   default="points")
    p.add_argument("--n_patches", type=int,   default=8)
    p.add_argument("--steps",     type=int,   default=2000)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--pw_lambda", type=float, default=10.0,
                   help="Pixel-weight lambda: source pixels get 1+lambda*s/s_mean weight.")
    p.add_argument("--peak_snr",  type=float, default=50.0)
    p.add_argument("--n_samples", type=int,   default=8)
    p.add_argument("--n_steps",   type=int,   default=50,
                   help="ODE integration steps at inference.")
    p.add_argument("--asinh_a",   type=float, default=0.0,
                   help="If >0, transport the flow in asinh-compressed space "
                        "y=asinh(s/a)/asinh(1/a) so brightness decades are "
                        "resolved. Set near the normalised noise level. 0=linear.")
    p.add_argument("--ckpt",      type=str,   default="",
                   help="Load a trained checkpoint instead of overfitting.")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--fig",       type=str,
                   default="results/overfit_psf_condflow.png")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Single-source corpus
# ---------------------------------------------------------------------------

def _single_point_patch(size, rng, flux_range=(1e-2, 1e0), edge_margin=16):
    sky, comp = assemble_corpus_field(
        size=size, include_diffuse=False,
        n_points=(1, 1), point_flux_range_jy=flux_range,
        n_ridges=(0, 0), edge_margin=edge_margin,
        return_components=True, rng=rng,
    )
    pts = comp["points"]
    rows, cols = np.nonzero(pts > 0)
    catalog = [(int(r), int(c), float(pts[r, c])) for r, c in zip(rows, cols)]
    return sky.astype(np.float32), catalog


def _single_ridge_patch(size, rng, flux_range=(5e-2, 5e-1), edge_margin=16):
    sky, comp = assemble_corpus_field(
        size=size, include_diffuse=False,
        n_points=(0, 0), n_ridges=(1, 1),
        ridge_flux_range_jy=flux_range, edge_margin=edge_margin,
        return_components=True, rng=rng,
    )
    ridge = comp["ridges"]
    return sky.astype(np.float32), ridge.astype(np.float32)


def _single_gaussian_patch(size, rng, peak_range=(1e-1, 5e-1),
                           sigma_range=(12.0, 20.0), edge_margin=28):
    """One large, smooth, peak-normalised 2D Gaussian blob.

    The cleanest extended source: single-peaked, smooth, big spatial extent,
    a continuous center-to-edge brightness ramp.  If the flow cannot reproduce
    this on overfit it cannot do extended emission at all.
    """
    cy = rng.integers(edge_margin, size - edge_margin)
    cx = rng.integers(edge_margin, size - edge_margin)
    sy = rng.uniform(*sigma_range)
    sx = rng.uniform(*sigma_range)
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2)
                 + ((xx - cx) ** 2) / (2 * sx ** 2)))
    g = (g * rng.uniform(*peak_range)).astype(np.float32)   # peak ∈ peak_range
    return g, g.copy()


def make_patches(n, size, seed, morphology):
    rng = np.random.default_rng(seed)
    skies, catalogs, ridges = [], [], []
    for _ in range(n):
        if morphology == "points":
            sky, cat = _single_point_patch(size, rng)
            skies.append(sky)
            catalogs.append(cat)
            ridges.append(np.zeros((size, size), dtype=np.float32))
        elif morphology == "gaussian":
            sky, blob = _single_gaussian_patch(size, rng)
            skies.append(sky)
            catalogs.append([])
            ridges.append(blob)
        else:
            sky, ridge = _single_ridge_patch(size, rng)
            skies.append(sky)
            catalogs.append([])
            ridges.append(ridge)
    return (np.stack(skies),
            catalogs,
            np.stack(ridges))


# ---------------------------------------------------------------------------
# Reconstruction helpers (mirrors overfit_field_inloop.py)
# ---------------------------------------------------------------------------

def windowed_flux(a, rc, half=4):
    r0, c0 = rc
    H, W = a.shape
    r1, r2 = max(0, r0 - half), min(H, r0 + half + 1)
    c1, c2 = max(0, c0 - half), min(W, c0 + half + 1)
    return float(np.clip(a[r1:r2, c1:c2], 0, None).sum())


def peak_rc(a):
    return tuple(int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))


def extended_fidelity(truth, recon, mask):
    t = truth[mask]
    r = np.clip(recon, 0.0, None)[mask]
    tf = float(t.sum())
    flux_ratio = float(r.sum() / tf) if tf > 0 else float("nan")
    rel_l2 = float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))
    corr = float(np.corrcoef(t, r)[0, 1]) if (t.std() > 0 and r.std() > 0) else float("nan")
    return flux_ratio, rel_l2, corr


def _zero_cond(B, device, dtype):
    return torch.zeros(B, COND_DIM, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Overfit
# ---------------------------------------------------------------------------

def overfit(s0_batch, d_batch, psf_batch, sigma_n, steps, base, lr, pw_lambda,
            device, seed, asinh_a=0.0):
    """Overfit PSFCondFlow on a small fixed batch of (s0, d, psf) pairs."""
    torch.manual_seed(seed)
    model = PSFCondFlow(base=base, asinh_a=asinh_a).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)

    # (B, 2, H, W) image stack: channel 0 = dirty, channel 1 = PSF
    image = torch.cat([d_batch[:, None], psf_batch[:, None]], dim=1)
    s0 = s0_batch[:, None]                              # (B, 1, H, W)
    cond = _zero_cond(s0.shape[0], device, s0.dtype)   # zeros: sigma_n passed via encoder scale

    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = cfm_loss(model, s0, image, cond,
                        pixel_weight_lambda=pw_lambda, generator=gen)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"    step {step:>5d}  loss {float(loss):.5f}", flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    skies, catalogs, ridges = make_patches(
        args.n_patches, args.size, args.seed, args.morphology)
    print(f"{args.n_patches} {args.morphology} patches @ {args.size}²")

    bank = load_g55_psf_bank(repo_root=_REPO_ROOT, target_size=args.size)
    rng_psf = np.random.default_rng(args.seed + 10)

    # Build dirty images for all patches using the same PSF (patch 0).
    psf_np, meta = bank.sample(rng_psf)
    print(f"PSF: {Path(meta['source_path']).parent.name}  rot={meta['rotation_deg']}°")
    fwd = ImageDomainForward(torch.from_numpy(psf_np).to(device))

    s_torch = torch.from_numpy(skies).to(device)        # (N, H, W)
    psf_torch = torch.from_numpy(psf_np).to(device)     # (H, W)
    psf_batch = psf_torch.unsqueeze(0).expand(args.n_patches, -1, -1)  # (N, H, W)

    gen_noise = torch.Generator(device=device).manual_seed(args.seed + 99)
    sigma_n = float(s_torch.amax(dim=(1, 2)).max()) / args.peak_snr
    d_batch = torch.stack([
        fwd.make_dirty(s_torch[i], noise_std=sigma_n, generator=gen_noise)
        for i in range(args.n_patches)
    ])

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = PSFCondFlow(
            base=cfg["base_channels"],
            asinh_a=cfg.get("asinh_a", 0.0),   # restore target space; old ckpts=linear
        ).to(device)
        model.load_state_dict(ckpt.get("ema", ckpt["model"]))
        model.eval()
        print(f"\n[held-out eval] {args.ckpt}  step={ckpt.get('step','?')}")
    else:
        print(f"\n[overfit] {args.steps} steps  pw_lambda={args.pw_lambda}  sigma_n={sigma_n:.3e}")
        model = overfit(
            s_torch, d_batch, psf_batch, sigma_n,
            args.steps, args.base_channels, args.lr, args.pw_lambda, device,
            args.seed + 1, asinh_a=args.asinh_a,
        )

    # ── reconstruct patch 0 ───────────────────────────────────────────────────
    i = 0
    s_true = s_torch[i]
    d_obs = d_batch[i]
    image_in = torch.stack([d_obs, psf_torch], dim=0).unsqueeze(0).to(device)  # (1,2,H,W)
    cond_in = _zero_cond(1, device, d_obs.dtype)

    gen_samp = torch.Generator(device=device).manual_seed(args.seed + 200)
    with torch.no_grad():
        samp = model.sample(image_in, cond_in, n_samples=args.n_samples,
                            n_steps=args.n_steps, generator=gen_samp)   # (1,n,H,W)
    s_post = samp[0].mean(0).cpu().numpy()   # (H, W) posterior mean
    t_np = s_true.cpu().numpy()
    d_np = d_obs.cpu().numpy()

    cat = catalogs[i]
    ridge = ridges[i]

    point_pass = None
    if cat:
        print(f"\n  per-source windowed flux (half=4 px), patch {i}:")
        print(f"  {'(row,col)':>12s} {'true':>9s} {'post':>9s} {'post/true':>9s}")
        ratios = []
        for (r, c, f) in sorted(cat, key=lambda x: -x[2]):
            ft = windowed_flux(t_np, (r, c))
            fp = windowed_flux(s_post, (r, c))
            ratio = fp / ft if ft > 0 else float("nan")
            ratios.append(ratio)
            print(f"  {f'({r},{c})':>12s} {ft:9.3e} {fp:9.3e} {ratio:9.2f}")
        pr, pc = peak_rc(s_post)
        near = min(((pr - r) ** 2 + (pc - c) ** 2) ** 0.5 for r, c, _ in cat)
        med = float(np.nanmedian(ratios))
        print(f"\n  posterior argmax ({pr},{pc}); nearest true source {near:.1f} px")
        print(f"  median post/true: {med:.2f}")
        point_pass = 0.3 <= med <= 3.0 and near <= 3.0

    ext_pass = None
    if ridge.max() > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(ridge > 1e-3 * ridge.max(), iterations=3)
        print(f"\n  extended fidelity ({int(mask.sum())} px support):")
        print(f"  {'':10s} {'flux_ratio':>10s} {'rel_l2':>8s} {'corr':>6s}")
        fr, rl, cr = extended_fidelity(t_np, s_post, mask)
        print(f"  {'posterior':10s} {fr:10.2f} {rl:8.2f} {cr:6.2f}")
        ext_pass = (0.3 <= fr <= 3.0) and (cr >= 0.6)

    verdict = ext_pass if args.morphology in ("extended", "gaussian") else point_pass
    label = "extended" if args.morphology in ("extended", "gaussian") else "sources"
    if verdict is None:
        print(f"\n  → no {label} to score")
    elif verdict:
        print(f"\n  → {label.upper()} SURVIVE (PSFCondFlow overfit gate PASS)")
    else:
        print(f"\n  → {label.upper()} LOST (PSFCondFlow overfit gate FAIL)")

    _save_fig(args.fig, t_np, d_np, s_post, cat, samp[0].cpu().numpy())
    return verdict


def _save_fig(path, s_true, d, s_post, cat, samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(4, samples.shape[0])
    ncols = 3 + n_show
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    def L(a):
        return np.log10(np.maximum(a, 1e-12))

    panels = [
        ("truth  log10", L(s_true), "inferno"),
        ("dirty  linear", d, "RdBu_r"),
        ("post mean  log10", L(s_post), "inferno"),
    ]
    for k in range(n_show):
        panels.append((f"sample {k}  log10", L(samples[k]), "inferno"))

    for ax, (title, im, cmap) in zip(axes, panels):
        kw = dict(vmin=-np.abs(im).max(), vmax=np.abs(im).max()) if cmap == "RdBu_r" else {}
        m = ax.imshow(im, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        for r, c, _ in cat:
            ax.plot(c, r, "c+", ms=8, mew=1.0)
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    print(f"\nfigure → {path}")


if __name__ == "__main__":
    main()
