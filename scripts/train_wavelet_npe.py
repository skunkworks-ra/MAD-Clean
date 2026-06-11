"""Full-scale training for the wavelet-NPE head (CoeffFlow + StarletCodec).

Mirrors train_mdn_asp.py: on-the-fly CutoutDataset scenes with the G55
D-config PSF bank, fixed held-out validation NLL, periodic checkpoints.
The codec calibration constants are stored in every checkpoint — they are
part of the model contract.
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

import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mad_clean.data.cutout_dataset import CutoutDataset  # noqa: E402
from mad_clean.data.psf_bank import load_g55_psf_bank  # noqa: E402
from mad_clean.models.coeff_flow import CoeffFlow  # noqa: E402
from mad_clean.wavelet.starlet import StarletCodec  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train wavelet NPE on synthetic cutouts.")
    p.add_argument("--out_dir",    type=str, default="results/wavelet_npe_train")
    p.add_argument("--repo_root",  type=str, default=".")
    p.add_argument("--steps",      type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dataset_size", type=int, default=200_000)
    p.add_argument("--val_size",   type=int, default=1_000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--morphologies", type=str,
                   default="point,blob,shell,filament")
    p.add_argument("--extended_fraction", type=float, default=0.05)
    p.add_argument("--snr_min",    type=float, default=5.0)
    p.add_argument("--compact_subtracted", action="store_true", default=True,
                   help="Hybrid contract: point sources removed from sky "
                        "and morphologies (a delta-function step handles "
                        "them in the loop). Default on for wavelet NPE.")
    p.add_argument("--no_compact_subtracted", dest="compact_subtracted",
                   action="store_false")
    p.add_argument("--calib_samples", type=int, default=1024,
                   help="Sky cutouts for codec calibration.")
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--val_every",  type=int, default=1_000)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=512)
    p.add_argument("--n_layers",      type=int, default=8)
    p.add_argument("--grad_clip",     type=float, default=10.0)
    p.add_argument("--resume",        type=str, default=None)
    return p.parse_args(argv)


def make_dataset(psf_bank, morphologies, extended_fraction, snr_min,
                 size, seed_offset, compact_subtracted=False):
    morph = {m.strip(): 1.0 for m in morphologies.split(",")}
    if compact_subtracted and "point" in morph:
        del morph["point"]
        print("[train] compact_subtracted: dropped 'point' from centred "
              "morphologies")
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=128,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=extended_fraction,
        rng_seed=seed_offset,
        length=size,
        morphology_balance=morph,
        snr_min=snr_min,
        return_sky=True,
        compact_subtracted=compact_subtracted,
    )


def collate(batch):
    res, psf, cond, _tgt, sky = zip(*batch)
    img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
    return img, torch.stack(list(cond)), torch.stack(list(sky))


def eval_val(flow, codec, val_loader, device):
    flow.eval()
    losses = []
    with torch.no_grad():
        for img, cond, sky in val_loader:
            theta = codec.encode(sky).to(device)
            img, cond = img.to(device), cond.to(device)
            losses.append(flow.nll_loss(theta, img, cond).item())
    flow.train()
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

    # --- Codec calibration (resume restores it from the checkpoint) -------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        codec = StarletCodec.from_state_dict(ckpt["codec"])
        print(f"[train] Codec restored from {args.resume}")
    else:
        calib_ds = make_dataset(psf_bank, args.morphologies,
                                args.extended_fraction, args.snr_min,
                                args.calib_samples, seed_offset=20_000_000,
                                compact_subtracted=args.compact_subtracted)
        skies = torch.stack([calib_ds[i][4] for i in range(args.calib_samples)])
        codec = StarletCodec(image_size=128)
        codec.calibrate(skies)
    print(f"[train] theta_dim = {codec.theta_dim}")

    train_ds = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.dataset_size, seed_offset=0,
                            compact_subtracted=args.compact_subtracted)
    val_ds   = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.val_size,   seed_offset=10_000_000,
                            compact_subtracted=args.compact_subtracted)

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

    flow = CoeffFlow(
        theta_dim=codec.theta_dim,
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[train] Model parameters: {n_params:,}")

    start_step = 0
    if args.resume:
        flow.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed at step {start_step}")

    optimizer = optim.Adam(flow.parameters(), lr=args.lr)

    def save(path, step, extra=None):
        d = {"step": step, "model": flow.state_dict(),
             "codec": codec.state_dict()}
        if extra:
            d.update(extra)
        torch.save(d, path)

    log = []
    best_val_loss = float("inf")
    step = start_step
    t0 = time.time()
    D = codec.theta_dim

    print(f"[train] Training for {args.steps} steps, batch={args.batch_size}, "
          f"device={args.device}, morphologies={args.morphologies}")
    flow.train()

    while step < args.steps:
        for img, cond, sky in train_loader:
            if step >= args.steps:
                break
            theta = codec.encode(sky).to(device)
            img, cond = img.to(device), cond.to(device)
            optimizer.zero_grad()
            loss = flow.nll_loss(theta, img, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            optimizer.step()

            step += 1
            loss_val = float(loss.item())

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"  step {step:6d}/{args.steps}  "
                      f"nll/dim={loss_val / D:8.4f}  "
                      f"elapsed={elapsed / 60:.1f}m")
                log.append({"step": step, "train_nll_per_dim": loss_val / D})

            if step % args.val_every == 0:
                val_loss = eval_val(flow, codec, val_loader, device)
                print(f"  step {step:6d}  val_nll/dim={val_loss / D:.4f}")
                if log:
                    log[-1]["val_nll_per_dim"] = val_loss / D
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save(out_dir / "best.pt", step, {"val_loss": val_loss})

            if step % args.checkpoint_every == 0:
                save(out_dir / f"ckpt_{step:07d}.pt", step)
                with open(out_dir / "log.json", "w") as fh:
                    json.dump(log, fh, indent=2)

    save(out_dir / "final.pt", step)
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[train] Done. Best val_nll/dim="
          f"{best_val_loss / D if best_val_loss < float('inf') else float('nan'):.4f}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
