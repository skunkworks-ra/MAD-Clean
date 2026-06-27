"""Full-scale training for the pixel-space flow deconvolution model.

Mirrors train_wavelet_npe.py but operates directly in pixel space:
  theta = sky / sky_scale  (flattened 128*128 = 16384 dims)

Loss: sky-value-weighted NLL + L1 sparsity on posterior samples.
  - Sky weighting: floor + sky_value, normalised to mean=1 per sample.
    Upweights source pixels without zeroing background gradient.
  - Sparsity: L1 on a single posterior sample per step, pushes
    background pixels toward zero.

Proven working on 8-sample overfit (overfit_pixel_sparse settings).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mad_clean.data.cutout_dataset import CutoutDataset
from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset
from mad_clean.data.psf_bank import load_g55_psf_bank, load_corpus_psf_bank
from mad_clean.models.coeff_flow import CoeffFlow


THETA_DIM  = 128 * 128   # pixel-space target dimension
IMAGE_SIZE = 128


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train pixel-space flow deconvolution.")
    p.add_argument("--out_dir",      type=str, default="results/pixel_flow_train")
    p.add_argument("--repo_root",    type=str, default=".")
    p.add_argument("--steps",        type=int, default=50_000)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--dataset_size", type=int, default=200_000)
    p.add_argument("--val_size",     type=int, default=1_000)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--morphologies", type=str,
                   default="point,blob,shell,filament")
    p.add_argument("--extended_fraction", type=float, default=0.5)
    p.add_argument("--corpus_psf_dir", type=str, default=None)
    p.add_argument("--stacks_dir",     type=str, default=None,
                   help="Path to PatchCorpusDataset stacks directory (train). "
                        "When set, uses real corpus patches instead of synthetic.")
    p.add_argument("--val_stacks_dir", type=str, default=None,
                   help="Stacks dir for validation. Falls back to --stacks_dir.")
    # Loss
    p.add_argument("--sky_weight_floor", type=float, default=0.1,
                   help="Additive floor on per-pixel NLL weights. floor + sky_value, "
                        "normalised to mean=1. Keeps background in loss while "
                        "upweighting source pixels.")
    p.add_argument("--sparsity_weight",  type=float, default=1.0,
                   help="Weight on L1 sparsity prior on posterior samples. "
                        "Drives background pixels toward zero.")
    p.add_argument("--theta_jitter",     type=float, default=0.0,
                   help="Gaussian noise std added to normalised theta during "
                        "training. 0 disables.")
    # Model
    p.add_argument("--base_channels",   type=int, default=32)
    p.add_argument("--context_dim",     type=int, default=256)
    p.add_argument("--hidden",          type=int, default=128)
    p.add_argument("--n_layers",        type=int, default=8)
    p.add_argument("--grad_clip",       type=float, default=10.0)
    # Logging / checkpoints
    p.add_argument("--log_every",        type=int, default=100)
    p.add_argument("--val_every",        type=int, default=1_000)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device",  type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--resume",  type=str, default=None)
    return p.parse_args(argv)


def make_dataset(psf_bank, args, size, seed_offset, stacks_dir=None):
    if stacks_dir is not None:
        print(f"[train] Using PatchCorpusDataset from {stacks_dir!r}")
        return PatchCorpusDataset(stacks_dir)
    morph = {m.strip(): 1.0 for m in args.morphologies.split(",")}
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=IMAGE_SIZE,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=seed_offset,
        length=size,
        morphology_balance=morph,
        return_sky=True,
    )


def collate(batch):
    # PatchCorpusDataset: (res, psf, cond, sky) — 4-tuple
    # CutoutDataset:      (res, psf, cond, _, sky) — 5-tuple
    if len(batch[0]) == 4:
        res, psf, cond, sky = zip(*batch)
    else:
        res, psf, cond, _, sky = zip(*batch)
    img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
    return img, torch.stack(list(cond)), torch.stack(list(sky))


def build_val_cache(val_ds, device, num_workers):
    print(f"[train] Building val cache on {device} ...")
    loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"),
                        collate_fn=collate)
    imgs, conds, skies = [], [], []
    for img, cond, sky in loader:
        imgs.append(img); conds.append(cond); skies.append(sky)
    imgs  = torch.cat(imgs).to(device)
    conds = torch.cat(conds).to(device)
    skies = torch.cat(skies).to(device)
    # Normalise val skies with training sky_scale (set after first batch;
    # for val we compute per-batch scale consistently).
    print("[train] Val cache ready.")
    return imgs, conds, skies


def eval_val(flow, val_cache, device, sky_weight_floor):
    imgs, conds, skies = val_cache
    flow.eval()
    nlls = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            img  = imgs[i:i+64]
            cond = conds[i:i+64]
            sky  = skies[i:i+64]
            sky_scale = sky.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
            theta = (sky / sky_scale.view(-1, 1, 1)).flatten(1)
            sky_w = sky_weight_floor + theta.clamp(min=0)
            sky_w = sky_w / sky_w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
            nll = flow.nll_loss(theta, img, cond, dim_weights=sky_w)
            nlls.append(float(nll.item()))
    flow.train()
    return float(np.mean(nlls)) / THETA_DIM


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.stacks_dir is not None:
        psf_bank = None
        print("[train] Corpus stacks mode: PSF bank not loaded.")
    elif args.corpus_psf_dir is not None:
        psf_bank = load_corpus_psf_bank(
            corpus_fits_dir=args.corpus_psf_dir, target_size=IMAGE_SIZE,
            rotation_augment=True)
        print(f"[train] PSF bank size: {len(psf_bank)}")
    else:
        psf_bank = load_g55_psf_bank(
            repo_root=args.repo_root, target_size=IMAGE_SIZE,
            rotation_augment=True)
        print(f"[train] PSF bank size: {len(psf_bank)}")

    val_stacks = args.val_stacks_dir or args.stacks_dir
    train_ds = make_dataset(psf_bank, args, args.dataset_size,
                            seed_offset=0,          stacks_dir=args.stacks_dir)
    val_ds   = make_dataset(psf_bank, args, args.val_size,
                            seed_offset=10_000_000, stacks_dir=val_stacks)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate, persistent_workers=(args.num_workers > 0),
    )

    flow = CoeffFlow(
        theta_dim=THETA_DIM,
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[train] Model parameters: {n_params:,}")
    print(f"[train] theta_dim = {THETA_DIM}  (pixel space, no codec)")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        flow.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed at step {start_step}")

    val_cache = build_val_cache(val_ds, device, args.num_workers)
    optimizer = optim.Adam(flow.parameters(), lr=args.lr, foreach=False)

    def save(path, step, extra=None):
        d = {"step": step, "model": flow.state_dict(),
             "theta_dim": THETA_DIM, "image_size": IMAGE_SIZE}
        if extra:
            d.update(extra)
        torch.save(d, path)

    log = []
    best_val_nll = float("inf")
    step = start_step
    t0 = time.time()

    print(f"[train] Training for {args.steps} steps, batch={args.batch_size}, "
          f"device={args.device}")
    flow.train()

    while step < args.steps:
        for img, cond, sky in train_loader:
            if step >= args.steps:
                break
            img, cond, sky = img.to(device), cond.to(device), sky.to(device)

            # Per-sample normalisation: flow sees O(1) targets.
            sky_scale = sky.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
            theta = (sky / sky_scale.view(-1, 1, 1)).flatten(1)

            if args.theta_jitter > 0:
                theta = theta + args.theta_jitter * torch.randn_like(theta)

            # Sky-value weighting: upweight source pixels, keep background in loss.
            sky_w = args.sky_weight_floor + theta.clamp(min=0)
            sky_w = sky_w / sky_w.mean(dim=-1, keepdim=True).clamp_min(1e-12)

            optimizer.zero_grad()
            nll = -flow.log_prob(theta, img, cond, dim_weights=sky_w).mean()
            loss = nll / THETA_DIM

            sparsity_val = 0.0
            if args.sparsity_weight > 0:
                theta_s = flow.sample_with_grad(img, cond, n=1).squeeze(1)
                sky_s   = theta_s * sky_scale.view(-1, 1)
                sparsity = sky_s.abs().mean()
                loss = loss + args.sparsity_weight * sparsity
                sparsity_val = float(sparsity.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"  step {step:6d}/{args.steps}  "
                      f"loss={float(loss.item()):8.4f}  "
                      f"nll/dim={float(nll.item()) / THETA_DIM:8.4f}  "
                      f"sparse={sparsity_val:.4e}  "
                      f"elapsed={elapsed / 60:.1f}m")
                log.append({"step": step,
                            "train_loss": float(loss.item()),
                            "train_nll_per_dim": float(nll.item()) / THETA_DIM,
                            "train_sparsity": sparsity_val})

            if step % args.val_every == 0:
                val_nll = eval_val(flow, val_cache, device, args.sky_weight_floor)
                print(f"  step {step:6d}  val_nll/dim={val_nll:.4f}")
                if log:
                    log[-1]["val_nll_per_dim"] = val_nll
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    save(out_dir / "best.pt", step, {"val_nll_per_dim": val_nll})

            if step % args.checkpoint_every == 0:
                save(out_dir / f"ckpt_{step:07d}.pt", step)
                with open(out_dir / "log.json", "w") as fh:
                    json.dump(log, fh, indent=2)

    save(out_dir / "final.pt", step)
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[train] Done. Best val_nll/dim={best_val_nll:.4f}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
