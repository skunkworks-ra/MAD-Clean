"""Evaluation suite for wavelet-NPE checkpoints (CoeffFlow + StarletCodec).

Run against any checkpoint produced by train_wavelet_npe.py:

    pixi run -e gpu python scripts/eval_wavelet_npe.py \
        --checkpoint results/wavelet_npe_train/best.pt

Tests, in order of importance:

1. cross-assignment — THE conditioning gate.  L[b, c] = log q(theta_c |
   image_b) over N held-out scenes.  A conditional model scores its own
   theta best (diagonal rank 1); an unconditional model is uniform random.
   This caught the 2026-06-11 failure where concat-conditioning was
   ignored entirely (diag nll == offdiag nll to 4 decimals).
2. codec round-trip — truth -> theta -> image with no flow involved;
   isolates representation loss from inference loss.
3. posterior grid — residual | truth | posterior median | posterior std
   on held-out scenes, with flux discrimination per scene.

Outputs to --out_dir: cross_assignment.json, codec_roundtrip.png,
posterior_grid.png, eval_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from mad_clean.data.cutout_dataset import CutoutDataset  # noqa: E402
from mad_clean.data.psf_bank import load_g55_psf_bank  # noqa: E402
from mad_clean.models.coeff_flow import CoeffFlow  # noqa: E402
from mad_clean.wavelet.starlet import StarletCodec  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a wavelet-NPE checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir",    type=str, default=None,
                   help="Default: <checkpoint dir>/eval_<step>/")
    p.add_argument("--repo_root",  type=str, default=".")
    p.add_argument("--n_scenes",   type=int, default=16,
                   help="Held-out scenes for all tests.")
    p.add_argument("--n_posterior", type=int, default=32)
    p.add_argument("--seed_offset", type=int, default=10_000_000,
                   help="Held-out scene seed offset (must differ from training).")
    p.add_argument("--morphologies", type=str, default="blob,shell,filament")
    p.add_argument("--extended_fraction", type=float, default=0.05)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # Model architecture (must match the checkpoint)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=512)
    p.add_argument("--n_layers",      type=int, default=8)
    return p.parse_args(argv)


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    codec = StarletCodec.from_state_dict(ckpt["codec"])
    flow = CoeffFlow(
        theta_dim=codec.theta_dim,
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    flow.load_state_dict(ckpt["model"])
    flow.eval()
    return flow, codec, ckpt.get("step", -1)


def make_scenes(args, psf_bank):
    morph = {m.strip(): 1.0 for m in args.morphologies.split(",")}
    ds = CutoutDataset(
        psf_bank=psf_bank, field_size=512, cutout_size=128,
        sigma_noise=1e-4, n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=args.seed_offset, length=args.n_scenes,
        morphology_balance=morph, return_sky=True, compact_subtracted=True,
    )
    res, psf, cond, sky = [], [], [], []
    for i in range(args.n_scenes):
        r, p, c, _, s = ds[i]
        res.append(r); psf.append(p); cond.append(c); sky.append(s)
    img = torch.stack([torch.stack(res), torch.stack(psf)], dim=1)
    return img, torch.stack(cond), torch.stack(sky)


def cross_assignment(flow, theta, img, cond, device):
    """Rank of each scene's own theta among all scenes' thetas."""
    N = theta.shape[0]
    L = torch.zeros(N, N)
    with torch.no_grad():
        for b in range(N):
            ib = img[b:b + 1].expand(N, -1, -1, -1).to(device)
            cb = cond[b:b + 1].expand(N, -1).to(device)
            L[b] = flow.log_prob(theta.to(device), ib, cb).cpu()
    ranks = [int((L[b] > L[b, b]).sum()) + 1 for b in range(N)]
    D = theta.shape[1]
    off = (L.sum() - L.diag().sum()) / (N * N - N)
    return {
        "ranks": ranks,
        "diagonal_wins": int(sum(r == 1 for r in ranks)),
        "n_scenes": N,
        "diag_nll_per_dim": float(-L.diag().mean() / D),
        "offdiag_nll_per_dim": float(-off / D),
    }


def codec_roundtrip_figure(codec, sky, path):
    rec = codec.decode(codec.encode(sky))
    N = sky.shape[0]
    rel = [float((rec[b] - sky[b]).norm() / max(sky[b].norm(), 1e-12))
           for b in range(N)]
    n_show = min(N, 8)
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    axes = np.atleast_2d(axes)
    for b in range(n_show):
        for k, (panel, title) in enumerate([
            (sky[b], "truth"), (rec[b], "codec round-trip"),
            (rec[b] - sky[b], "error"),
        ]):
            im = axes[b, k].imshow(panel.numpy(), origin="lower")
            axes[b, k].set_title(title if b == 0 else "")
            axes[b, k].axis("off")
            fig.colorbar(im, ax=axes[b, k], fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return rel


def posterior_grid(flow, codec, img, cond, sky, n_post, device, path):
    N = sky.shape[0]
    with torch.no_grad():
        samples = flow.sample(img.to(device), cond.to(device), n=n_post)
    B, n, D = samples.shape
    dec = codec.decode(samples.reshape(B * n, D).cpu()).reshape(B, n, 128, 128)
    med = dec.median(dim=1).values
    std = dec.std(dim=1)

    per_scene = []
    for b in range(N):
        flux_samples = dec[b].sum(dim=(1, 2))
        per_scene.append({
            "true_flux": float(sky[b].sum()),
            "post_median_flux": float(med[b].sum()),
            "post_flux_iqr": float(
                flux_samples.quantile(0.75) - flux_samples.quantile(0.25)),
            "rel_l2_post_median": float(
                (med[b] - sky[b]).norm() / max(sky[b].norm(), 1e-12)),
        })

    n_show = min(N, 8)
    fig, axes = plt.subplots(n_show, 4, figsize=(13, 3.2 * n_show))
    axes = np.atleast_2d(axes)
    titles = ["residual", "true sky", "posterior median", "posterior std"]
    for b in range(n_show):
        panels = [img[b, 0], sky[b], med[b], std[b]]
        for k, (panel, title) in enumerate(zip(panels, titles)):
            im = axes[b, k].imshow(panel.numpy(), origin="lower")
            axes[b, k].set_title(title if b == 0 else "")
            axes[b, k].axis("off")
            fig.colorbar(im, ax=axes[b, k], fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return per_scene


def run(args):
    device = torch.device(args.device)
    flow, codec, step = load_model(args, device)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.checkpoint).parent / f"eval_{step:07d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] checkpoint step {step}; outputs in {out_dir}/")

    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=128, rotation_augment=True)
    img, cond, sky = make_scenes(args, psf_bank)
    theta = codec.encode(sky)

    xa = cross_assignment(flow, theta, img, cond, device)
    print(f"[eval] cross-assignment: diagonal wins {xa['diagonal_wins']}"
          f"/{xa['n_scenes']}; diag nll/dim {xa['diag_nll_per_dim']:.4f}, "
          f"offdiag {xa['offdiag_nll_per_dim']:.4f}")
    with open(out_dir / "cross_assignment.json", "w") as fh:
        json.dump(xa, fh, indent=2)

    rel = codec_roundtrip_figure(codec, sky, out_dir / "codec_roundtrip.png")
    print(f"[eval] codec round-trip rel L2: median "
          f"{float(np.median(rel)):.3f}, max {max(rel):.3f}")

    per_scene = posterior_grid(flow, codec, img, cond, sky,
                               args.n_posterior, device,
                               out_dir / "posterior_grid.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "cross_assignment": xa,
        "codec_roundtrip_rel_l2": rel,
        "per_scene": per_scene,
    }
    with open(out_dir / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[eval] done.")
    return summary


if __name__ == "__main__":
    run(parse_args())
