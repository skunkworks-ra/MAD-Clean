"""Probe a wavelet-NPE checkpoint on a single corpus field.

Two modes:
  --patch_mode   dirty | true sky | N posterior draws per patch, brightest first.
  --field_mode   (default) run all patches, recompose full 512x512 field,
                 show dirty | true sky | posterior median | posterior std.

Usage:
    pixi run -e gpu python scripts/probe_corpus_field.py \
        --checkpoint results/wavelet_npe_train_corpus/best.pt \
        --stacks_dir /path/to/corpus_stacks/val \
        --field_idx 2 \
        --n_draws 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
from mad_clean.models.coeff_flow import CoeffFlow
from mad_clean.wavelet.starlet import StarletCodec


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--stacks_dir",    required=True)
    p.add_argument("--field_idx",     type=int, default=0)
    p.add_argument("--n_draws",       type=int, default=16,
                   help="Posterior draws per patch (field mode uses median over these).")
    p.add_argument("--patch_mode",    action="store_true",
                   help="Per-patch figures (brightest first) instead of full-field.")
    p.add_argument("--n_patches",     type=int, default=4,
                   help="Number of patches to show in --patch_mode.")
    p.add_argument("--out_dir",       type=str, default=None)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=128)
    p.add_argument("--n_layers",      type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"[probe] loaded checkpoint step {ckpt.get('step', -1)}")
    return flow, codec


def sample_patch(flow, codec, dirty, psf, cond, n_draws, device):
    img = torch.stack([dirty, psf], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        samples = flow.sample(img, cond.unsqueeze(0).to(device), n=n_draws)
    return codec.decode(samples[0].cpu())  # (n_draws, 128, 128)


def _asinh_stretch(arr, a):
    return np.arcsinh(arr / a) / np.arcsinh(1.0 / a)


def _show(ax, data, title, cmap, vmin, vmax, a=None):
    if a is not None:
        data = _asinh_stretch(np.clip(data, 0, None), a)
        vmin, vmax = 0, 1
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


# ── field-level reconstruction ────────────────────────────────────────────────

def run_field(flow, codec, ds, field_info, args, out_dir, device):
    start, end = field_info["patch_start"], field_info["patch_end"]
    config = field_info["vla_config"]
    cell   = field_info.get("cell_arcsec", "?")
    n_side = int(np.sqrt(end - start))   # 4 for 16 patches
    patch  = 128
    field_px = n_side * patch            # 512

    dirty_field  = np.zeros((field_px, field_px), dtype=np.float32)
    sky_field    = np.zeros((field_px, field_px), dtype=np.float32)
    median_field = np.zeros((field_px, field_px), dtype=np.float32)
    std_field    = np.zeros((field_px, field_px), dtype=np.float32)

    print(f"[probe] field {args.field_idx}: VLA-{config}, cell={cell}\", "
          f"{end-start} patches -> {field_px}x{field_px}")

    for local_idx, global_idx in enumerate(range(start, end)):
        row = local_idx // n_side
        col = local_idx % n_side
        r0, c0 = row * patch, col * patch

        dirty, psf, cond, sky = ds[global_idx]
        draws = sample_patch(flow, codec, dirty, psf, cond, args.n_draws, device)
        # draws: (n_draws, 128, 128)
        med = draws.median(dim=0).values.numpy()
        std = draws.std(dim=0).numpy()

        dirty_field [r0:r0+patch, c0:c0+patch] = dirty.numpy()
        sky_field   [r0:r0+patch, c0:c0+patch] = sky.numpy()
        median_field[r0:r0+patch, c0:c0+patch] = med
        std_field   [r0:r0+patch, c0:c0+patch] = std

        sky_pk = float(sky.abs().max())
        med_pk = float(np.abs(med).max())
        print(f"  patch {local_idx:2d} (global {global_idx}): "
              f"dirty_peak={float(dirty.abs().max()):.4f}  "
              f"sky_peak={sky_pk:.4f}  med_peak={med_pk:.4f}")

    residual_field = dirty_field - median_field
    # Each panel scaled to its own data range.
    def vrange(data, sym=False):
        hi = float(np.percentile(np.abs(data), 99.9)) or 1e-12
        return (-hi, hi) if sym else (0, hi)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        ("dirty",                    dirty_field,   "RdBu_r", True),
        ("true sky",                 sky_field,     "inferno", False),
        ("posterior median",         median_field,  "inferno", False),
        ("residual (dirty - median)",residual_field,"RdBu_r", True),
    ]
    for ax, (title, data, cmap, sym) in zip(axes, panels):
        lo, hi = vrange(data, sym)
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"field {args.field_idx} | VLA-{config} | cell={cell}\" | "
        f"{args.n_draws} draws/patch | step {ckpt_step}",
        fontsize=10)
    fig.tight_layout()
    out = out_dir / f"field{args.field_idx}_reconstruction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[probe] {out}")


# ── per-patch figures ─────────────────────────────────────────────────────────

def run_patches(flow, codec, ds, field_info, args, out_dir, device):
    start, end = field_info["patch_start"], field_info["patch_end"]
    config = field_info["vla_config"]

    peaks = [(float(ds[i][0].abs().max()), i) for i in range(start, end)]
    peaks.sort(reverse=True)

    for rank, (peak, idx) in enumerate(peaks[:args.n_patches]):
        dirty, psf, cond, sky = ds[idx]
        draws = sample_patch(flow, codec, dirty, psf, cond, args.n_draws, device)

        all_pos = np.concatenate([sky.numpy().ravel(), draws.numpy().ravel()])
        smax = float(all_pos.max()) or 1e-12
        a    = max(smax * 0.01, 1e-9)
        dvmax = float(dirty.abs().max()) or 1e-12

        n_cols = 2 + args.n_draws
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.8))
        _show(axes[0], dirty.numpy(), "dirty", "RdBu_r", -dvmax, dvmax)
        fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.02)
        _show(axes[1], sky.numpy(), f"true sky\npeak={float(sky.max()):.4f}",
              "inferno", None, None, a=a)
        fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.02)
        for i in range(args.n_draws):
            _show(axes[2+i], draws[i].numpy(),
                  f"draw {i}\npeak={float(draws[i].max()):.4f}",
                  "inferno", None, None, a=a)
            fig.colorbar(axes[2+i].images[0], ax=axes[2+i], fraction=0.046, pad=0.02)

        label = (f"field {args.field_idx} | patch {idx} (rank {rank+1}) | "
                 f"VLA-{config} | dirty peak={peak:.4f}")
        fig.suptitle(label, fontsize=9)
        fig.tight_layout()
        out = out_dir / f"patch_rank{rank+1:02d}_idx{idx:03d}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[probe] {out}")


# ── entry point ───────────────────────────────────────────────────────────────

ckpt_step = -1


def run(args):
    global ckpt_step
    device = torch.device(args.device)
    flow, codec = load_model(args, device)
    ckpt_step = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False).get("step", -1)

    ds = PatchCorpusDataset(args.stacks_dir)
    with open(Path(args.stacks_dir) / "manifest.json") as fh:
        manifest = json.load(fh)
    field_info = manifest["fields"][args.field_idx]

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(args.checkpoint).parent / f"probe_field{args.field_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_mode:
        run_patches(flow, codec, ds, field_info, args, out_dir, device)
    else:
        run_field(flow, codec, ds, field_info, args, out_dir, device)

    print("[probe] done.")


if __name__ == "__main__":
    run(parse_args())
