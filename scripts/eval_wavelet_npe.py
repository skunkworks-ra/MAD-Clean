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
4. decisive samples — for held-out extended scenes (shell, filament), the
   codec round-trip (ceiling) | posterior median | individual posterior
   draws, all on ONE shared colour scale.  A blurry median is ambiguous:
   if each individual draw is sharp but displaced between draws, the
   posterior is correctly broad and the median is just the wrong summary;
   if each draw is itself mush, that is a genuine generalisation failure.
   This figure separates those two cases by eye.

Outputs to --out_dir: cross_assignment.json, codec_roundtrip.png,
posterior_grid.png, decisive_samples.png, eval_summary.json.
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
    p.add_argument("--morphologies", type=str,
                   default="point,blob,shell,filament")
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
        morphology_balance=morph, return_sky=True,
        compact_subtracted=("point" not in morph),
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
        # True sky and posterior median share the truth's colour scale, so
        # the median is judged at the amplitude it must reproduce — not
        # self-scaled, which hides flat/structureless fields.
        smax = float(sky[b].abs().max().clamp_min(1e-12))
        scales = [None, (-smax, smax), (-smax, smax), None]
        panels = [img[b, 0], sky[b], med[b], std[b]]
        for k, (panel, title, sc) in enumerate(zip(panels, titles, scales)):
            kw = {} if sc is None else {"vmin": sc[0], "vmax": sc[1]}
            im = axes[b, k].imshow(panel.numpy(), origin="lower", **kw)
            axes[b, k].set_title(title if b == 0 else "")
            axes[b, k].axis("off")
            fig.colorbar(im, ax=axes[b, k], fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return per_scene


def make_morph_scenes(args, psf_bank, morph_name, n, seed_offset):
    """Held-out scenes whose CENTRED source is forced to one morphology.

    CutoutDataset always places the centred source's morphology from
    ``morphology_balance``, so balance={morph: 1.0} guarantees the scene
    is centred on a shell / filament — the extended cases the decisive
    figure needs.  compact_subtracted matches the rest of the eval (tied
    to whether 'point' is among the requested morphologies, i.e. the
    model's training domain), so the conditioning image is the same kind
    of residual the flow was trained on."""
    compact = "point" not in {m.strip() for m in args.morphologies.split(",")}
    ds = CutoutDataset(
        psf_bank=psf_bank, field_size=512, cutout_size=128,
        sigma_noise=1e-4, n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=seed_offset, length=n,
        morphology_balance={morph_name: 1.0}, return_sky=True,
        compact_subtracted=compact,
    )
    res, psf, cond, sky = [], [], [], []
    for i in range(n):
        r, p, c, _, s = ds[i]
        res.append(r); psf.append(p); cond.append(c); sky.append(s)
    img = torch.stack([torch.stack(res), torch.stack(psf)], dim=1)
    return img, torch.stack(cond), torch.stack(sky)


def _decisive_panels(rows, roundtrip, median, dec, n_draws, path, crop=None):
    """Render the decisive grid: truth | posterior median | draw 1, one
    shared symmetric colour scale per scene (anchored to the truth).  With
    ``crop`` (half-width in px) every panel is centred-cropped so the
    centred source's structure is resolvable rather than swamped by the
    background-decode speckle that fills the frame periphery."""
    R = len(rows)
    titles = ["truth", "posterior median", "posterior draw"]

    def view(t):
        if crop is None:
            return t
        c = t.shape[-1] // 2
        return t[..., c - crop:c + crop, c - crop:c + crop]

    fig, axes = plt.subplots(R, 3, figsize=(8.4, 2.8 * R))
    axes = np.atleast_2d(axes)
    for b in range(R):
        label, truth_b = rows[b][0], rows[b][1]
        # Anchor on the truth's amplitude, never self-scaled: a flat field
        # must look flat, not be stretched into apparent structure.
        m = float(truth_b.abs().max().clamp_min(1e-12))
        panels = [truth_b, median[b], dec[b, 0]]
        for k, (panel, title) in enumerate(zip(panels, titles)):
            im = axes[b, k].imshow(view(panel).numpy(), origin="lower",
                                   vmin=-m, vmax=m)
            if b == 0:
                axes[b, k].set_title(title)
            axes[b, k].set_xticks([]); axes[b, k].set_yticks([])
            fig.colorbar(im, ax=axes[b, k], fraction=0.046)
        axes[b, 0].set_ylabel(label, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def decisive_samples_figure(flow, codec, psf_bank, args, device, path,
                            n_per_morph=2, n_sample_cols=3, crop=28):
    """The fork-deciding figure.  Per held-out extended scene, on ONE
    shared symmetric colour scale (anchored to the truth's amplitude):
    truth | codec round-trip (ceiling) | posterior median | individual
    posterior draws.

    Reading it:
      - draws sharp but DISPLACED between columns  => posterior correctly
        broad; the median blurs because it averages displaced structure.
        Report samples, not the median; v4 is fine.
      - each draw itself MUSH                       => genuine
        generalisation failure; that is the bug to fix.

    Writes two files: the full 128x128 frame (shows the background-decode
    speckle) and a ``_zoom`` centred crop (resolves the source structure).
    """
    morphs = [("shell", args.seed_offset + 1_000),
              ("filament", args.seed_offset + 2_000)]

    rows = []  # each: (label, truth(128,128), img(1,2,128,128), cond(1,C))
    for name, seed in morphs:
        img, cond, sky = make_morph_scenes(args, psf_bank, name,
                                           n_per_morph, seed)
        for i in range(n_per_morph):
            rows.append((f"{name} #{i}", sky[i], img[i:i + 1], cond[i:i + 1]))

    truth = torch.stack([r[1] for r in rows])                       # (R,128,128)
    img_all = torch.cat([r[2] for r in rows], dim=0)                # (R,2,128,128)
    cond_all = torch.cat([r[3] for r in rows], dim=0)               # (R,C)

    # Ceiling: codec round-trip of the truth (no flow involved).
    roundtrip = codec.decode(codec.encode(truth))                  # (R,128,128)

    # Posterior draws; median over all, plus the first few individual draws.
    with torch.no_grad():
        samples = flow.sample(img_all.to(device), cond_all.to(device),
                              n=args.n_posterior)                   # (R,n,D)
    R_, n, D = samples.shape
    dec = codec.decode(samples.reshape(R_ * n, D).cpu()).reshape(R_, n, 128, 128)
    median = dec.median(dim=1).values                              # (R,128,128)
    n_draws = min(n_sample_cols, n)

    _decisive_panels(rows, roundtrip, median, dec, n_draws, path, crop=None)
    zoom_path = path.with_name(path.stem + "_zoom" + path.suffix)
    _decisive_panels(rows, roundtrip, median, dec, n_draws, zoom_path, crop=crop)


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

    decisive_samples_figure(flow, codec, psf_bank, args, device,
                            out_dir / "decisive_samples.png")
    print(f"[eval] decisive samples figure -> "
          f"{out_dir / 'decisive_samples.png'}")

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
