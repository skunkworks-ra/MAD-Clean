"""Full-scale training script for MDNAsp (flow_plan.md §Training).

Trains on on-the-fly generated CutoutDataset scenes with the G55 D-config
L-band PSF bank. Checkpoints saved every --checkpoint_every steps.
Validation NLL logged on a fixed held-out set.
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
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.mdn_asp import MDNAsp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MDNAsp on synthetic cutouts.")
    p.add_argument("--out_dir",    type=str, default="results/mdn_asp_train",
                   help="Output directory for checkpoints and logs.")
    p.add_argument("--repo_root",  type=str, default=".",
                   help="Repo root containing data/g55/chunk_*/psf.fits.")
    p.add_argument("--steps",      type=int, default=50_000,
                   help="Total gradient steps (default: 50000).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dataset_size", type=int, default=200_000,
                   help="Number of unique training scenes (default: 200000).")
    p.add_argument("--val_size",   type=int, default=1_000,
                   help="Fixed held-out validation scenes (default: 1000).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4).")
    p.add_argument("--morphologies", type=str, default="point,blob",
                   help="Centred-source morphologies (default: 'point,blob').")
    p.add_argument("--extended_fraction", type=float, default=0.05,
                   help="Per-distractor extended probability (default: 0.05).")
    p.add_argument("--snr_min",    type=float, default=5.0,
                   help="Minimum convolved SNR for centred source (default: 5.0).")
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--val_every",  type=int, default=1_000)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--hidden",        type=int, default=256)
    p.add_argument("--n_components",  type=int, default=5)
    p.add_argument("--resume",        type=str, default=None,
                   help="Path to checkpoint .pt to resume from.")
    return p.parse_args(argv)


def make_dataset(psf_bank, morphologies, extended_fraction, snr_min,
                 size, seed_offset):
    morph_keys = [m.strip() for m in morphologies.split(",")]
    morphology_balance = {m: 1.0 for m in morph_keys}
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=128,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=extended_fraction,
        rng_seed=seed_offset,
        length=size,
        morphology_balance=morphology_balance,
        snr_min=snr_min,
    )


def collate(batch):
    res, psf, cond, tgt = zip(*batch)
    img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
    return img, torch.stack(list(cond)), torch.stack(list(tgt))


def eval_val(model, val_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for img, cond, tgt in val_loader:
            img, cond, tgt = img.to(device), cond.to(device), tgt.to(device)
            params = model(img, cond)
            losses.append(model.nll_loss(params, tgt).item())
    model.train()
    return float(np.mean(losses))


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[train] Loading PSF bank from {args.repo_root!r}/data/g55 ...")
    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=128, rotation_augment=True,
    )
    print(f"[train] PSF bank size: {len(psf_bank)}")

    train_ds = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.dataset_size, seed_offset=0)
    val_ds   = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.val_size,   seed_offset=10_000_000)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=collate, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=(args.device == "cuda"),
        collate_fn=collate, persistent_workers=(args.num_workers > 0),
    )

    model = MDNAsp(
        base_channels=args.base_channels,
        hidden=args.hidden,
        n_components=args.n_components,
        cond_dim=5,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {n_params:,}")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed from {args.resume} at step {start_step}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log = []
    best_val_loss = float("inf")
    step = start_step
    t0 = time.time()

    print(f"[train] Training for {args.steps} steps, batch={args.batch_size}, "
          f"device={args.device}, morphologies={args.morphologies}")
    model.train()

    while step < args.steps:
        for img, cond, tgt in train_loader:
            if step >= args.steps:
                break

            img, cond, tgt = img.to(device), cond.to(device), tgt.to(device)
            optimizer.zero_grad()
            params = model(img, cond)
            loss = model.nll_loss(params, tgt)
            loss.backward()
            optimizer.step()

            step += 1
            loss_val = float(loss.item())

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"  step {step:6d}/{args.steps}  "
                      f"train_loss={loss_val:8.3f}  "
                      f"elapsed={elapsed/60:.1f}m")
                log.append({"step": step, "train_loss": loss_val})

            if step % args.val_every == 0:
                val_loss = eval_val(model, val_loader, device)
                print(f"  step {step:6d}  val_loss={val_loss:.3f}")
                if log:
                    log[-1]["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {"step": step, "model": model.state_dict(),
                         "val_loss": val_loss},
                        out_dir / "best.pt",
                    )

            if step % args.checkpoint_every == 0:
                torch.save(
                    {"step": step, "model": model.state_dict()},
                    out_dir / f"ckpt_{step:07d}.pt",
                )
                with open(out_dir / "log.json", "w") as fh:
                    json.dump(log, fh, indent=2)

    # Final save
    torch.save({"step": step, "model": model.state_dict()},
               out_dir / "final.pt")
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[train] Done. Best val_loss={best_val_loss:.3f}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
