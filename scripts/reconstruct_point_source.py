"""Minimal honest test: forward-model a single OFF-CENTRE point source through a
real beam, then reconstruct it.  A centred point aligns with the PSF origin and
hides position/ifftshift errors, so the source is placed off-centre and the
recovered peak location is reported against the truth.

Two reconstructions:
  likelihood  : positive least-squares (Landweber) on ‖d − A s‖² — the explicit-
                PSF loop's data-consistency, NO prior.  Tests whether the forward
                model + adjoint recover a point at the right place and flux.
  posterior   : the trained-prior field-posterior ULA (mean), for contrast — what
                the diffuse prior does to a clean point.

Example
-------
    pixi run -e gpu python scripts/reconstruct_point_source.py \\
        --ckpt models/field_score.pt --size 256 --row 70 --col 180 \\
        --device cuda --fig results/point_reconstruct.png
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

from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.imaging.langevin import sample_field_posterior
from mad_clean.imaging.score import EDMDenoiser, UNet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reconstruct one off-centre point source.")
    p.add_argument("--ckpt", type=str, default="models/field_score.pt")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--source", choices=["point", "gaussian"], default="point")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="2D Gaussian sigma (px) when --source gaussian.")
    p.add_argument("--row", type=int, default=70, help="Source row (off-centre).")
    p.add_argument("--col", type=int, default=180, help="Source col (off-centre).")
    p.add_argument("--flux", type=float, default=1.0)
    p.add_argument("--peak_snr", type=float, default=100.0)
    p.add_argument("--n_landweber", type=int, default=2000)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig", type=str, default="results/point_reconstruct.png")
    return p.parse_args(argv)


def peak_rc(a: np.ndarray) -> tuple[int, int]:
    return tuple(int(x) for x in np.unravel_index(int(np.argmax(a)), a.shape))


def render_gaussian(size, row, col, sigma, flux, device):
    """Off-centre 2D Gaussian, pixel sum = flux."""
    rr = torch.arange(size, device=device).view(-1, 1).float()
    cc = torch.arange(size, device=device).view(1, -1).float()
    g = torch.exp(-((rr - row) ** 2 + (cc - col) ** 2) / (2 * sigma ** 2))
    return g * (flux / g.sum())


def windowed_moments(a: np.ndarray, true_rc, half=20):
    """Flux-weighted centroid, flux, and rms width in a window around true_rc.

    Windowed so PSF sidelobes / prior texture far from the source don't bias the
    centroid; returns None if the window holds no positive flux."""
    r0, c0 = true_rc
    H, W = a.shape
    r1, r2 = max(0, r0 - half), min(H, r0 + half + 1)
    c1, c2 = max(0, c0 - half), min(W, c0 + half + 1)
    win = np.clip(a[r1:r2, c1:c2], 0, None)
    tot = win.sum()
    if tot <= 0:
        return None
    rr, cc = np.mgrid[r1:r2, c1:c2]
    cr = float((rr * win).sum() / tot)
    cc_ = float((cc * win).sum() / tot)
    var = float((((rr - cr) ** 2 + (cc - cc_) ** 2) * win).sum() / tot)
    return cr, cc_, float(tot), float(np.sqrt(var / 2.0))


def report(name, recon, true_rc, true_flux):
    a = recon.detach().cpu().numpy()
    pr, pc = peak_rc(a)
    m = windowed_moments(a, true_rc)
    print(f"[{name:10s}] argmax_peak=({pr},{pc}) Δ=({pr-true_rc[0]:+d},{pc-true_rc[1]:+d})  "
          f"peak_flux={a.max():.3e}")
    if m is None:
        print(f"             windowed: NO positive flux near the true source.")
    else:
        cr, cc, tot, sig = m
        print(f"             centroid=({cr:.1f},{cc:.1f}) "
              f"Δ=({cr-true_rc[0]:+.1f},{cc-true_rc[1]:+.1f})  "
              f"win_flux={tot:.3f} (true {true_flux})  rms_width={sig:.2f}px")


def landweber(fwd, d, n_iter):
    """Positive least-squares deconvolution (likelihood-only, no prior).

    s ← relu(s + η Aᵀ(d − A s)),  η = 1/max|H(k)|²  (guaranteed-stable step)."""
    L = float((fwd._H.abs() ** 2).max())
    eta = 1.0 / L
    s = torch.zeros(fwd.shape, device=d.device, dtype=d.dtype)
    for _ in range(n_iter):
        s = torch.relu(s + eta * fwd.adjoint(d - fwd.forward(s)))
    return s


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    # ── off-centre source ────────────────────────────────────────────────────
    if args.source == "gaussian":
        s_true = render_gaussian(args.size, args.row, args.col, args.sigma,
                                 args.flux, device)
        print(f"true gaussian: centre=({args.row},{args.col}) sigma={args.sigma}px "
              f"flux={args.flux} size={args.size} (centre would be {args.size//2})")
    else:
        s_true = torch.zeros(args.size, args.size, device=device)
        s_true[args.row, args.col] = args.flux
        print(f"true point: ({args.row},{args.col}) flux={args.flux} size={args.size} "
              f"(centre would be {args.size//2})")
    true_rc = (args.row, args.col)

    # ── real beam, honest dirty image ────────────────────────────────────────
    bank = load_g55_psf_bank(repo_root=_REPO_ROOT, target_size=args.size)
    psf_np, meta = bank.sample(np.random.default_rng(args.seed))
    fwd = ImageDomainForward(torch.from_numpy(psf_np).to(device))
    print(f"PSF: {Path(meta['source_path']).parent.name} rot={meta['rotation_deg']}°")

    d_clean = fwd.forward(s_true)
    sigma_n = float(d_clean.abs().max()) / args.peak_snr
    gen = torch.Generator(device=device).manual_seed(args.seed)
    d = fwd.make_dirty(s_true, noise_std=sigma_n, generator=gen)

    # ── likelihood-only reconstruction ───────────────────────────────────────
    with torch.no_grad():
        s_lik = landweber(fwd, d, args.n_landweber)
    print()
    report("likelihood", s_lik, true_rc, args.flux)

    # ── trained-prior posterior (mean), for contrast ─────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mu, tau = float(ckpt["mu"]), float(ckpt["tau"])
    space = ckpt.get("space", "log")
    model = EDMDenoiser(UNet(in_ch=1, base=ckpt["config"]["base_channels"]),
                        sigma_data=ckpt["sigma_data"]).to(device)
    model.load_state_dict(ckpt["ema"]); model.eval()
    print(f"prior space={space}  mu={mu:.4g} tau={tau:.4g}")
    with torch.no_grad():
        samp = sample_field_posterior(
            fwd, model, d, mu, tau, sigma_n, n_samples=args.n_samples,
            data_scale="auto", space=space, generator=gen)
    s_post = samp.mean(0)
    report("posterior", s_post, true_rc, args.flux)

    _save_fig(args.fig, s_true, d, s_lik, s_post, true_rc)


def _save_fig(path, s_true, d, s_lik, s_post, true_rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def L(a):
        return np.log10(np.maximum(a.detach().cpu().numpy(), 1e-12))

    panels = [("true point  log10", L(s_true), "inferno"),
              ("dirty d  (linear)", d.detach().cpu().numpy(), "RdBu_r"),
              ("likelihood recon  log10", L(s_lik), "inferno"),
              ("posterior mean  log10", L(s_post), "inferno")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, (title, im, cmap) in zip(ax, panels):
        kw = dict(vmin=-np.abs(im).max(), vmax=np.abs(im).max()) if cmap == "RdBu_r" else {}
        m = a.imshow(im, origin="lower", cmap=cmap, **kw)
        a.set_title(title); a.axis("off")
        a.plot(true_rc[1], true_rc[0], "c+", ms=12, mew=1.5)  # true location marker
        fig.colorbar(m, ax=a, fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=90)
    print(f"\nfigure → {path}")


if __name__ == "__main__":
    main()
