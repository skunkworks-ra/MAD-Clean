"""Full training script for PatchFlow.

Usage:
    pixi run -e gpu python scripts/train_patch_flow.py \
        --out_dir results/patch_flow_v1 \
        --epochs 200 \
        --batch_size 64 \
        --base_channels 64
"""
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mad_clean.data.patch_flow_dataset import PatchFlowDataset
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.models.patch_flow import PatchFlow, cfm_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",       type=str,   default="results/patch_flow_v1")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--base_channels", type=int,   default=64)
    p.add_argument("--depth",         type=int,   default=4)
    p.add_argument("--samples_per_epoch", type=int, default=10_000)
    p.add_argument("--val_samples",   type=int,   default=1_000)
    p.add_argument("--n_workers",     type=int,   default=4)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint to resume from.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)

    psf_bank = load_g55_psf_bank(_REPO_ROOT)

    train_ds = PatchFlowDataset(
        psf_bank=psf_bank,
        length=args.samples_per_epoch,
        rng_seed=args.seed,
    )
    val_ds = PatchFlowDataset(
        psf_bank=psf_bank,
        length=args.val_samples,
        rng_seed=args.seed + 10_000_000,  # held-out split
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True,
    )

    model = PatchFlow(base_channels=args.base_channels, depth=args.depth).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best_val    = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_epoch}  best_val={best_val:.4f}")

    log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for dirty, psf, sigma, clean in train_loader:
            dirty, psf, sigma, clean = (
                dirty.to(device), psf.to(device),
                sigma.to(device), clean.to(device),
            )
            opt.zero_grad()
            loss = cfm_loss(model, dirty, psf, clean, sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss) * dirty.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dirty, psf, sigma, clean in val_loader:
                dirty, psf, sigma, clean = (
                    dirty.to(device), psf.to(device),
                    sigma.to(device), clean.to(device),
                )
                val_loss += float(cfm_loss(model, dirty, psf, clean, sigma)) * dirty.shape[0]
        val_loss /= len(val_ds)

        sched.step()
        elapsed = time.time() - t0

        print(f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  t={elapsed:.1f}s")

        entry = {"epoch": epoch, "train": train_loss, "val": val_loss}
        log.append(entry)
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        # Checkpoint every epoch
        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "best_val": best_val,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> best checkpoint  val={best_val:.4f}")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {out_dir}/best.pt  {out_dir}/last.pt")


if __name__ == "__main__":
    main()
