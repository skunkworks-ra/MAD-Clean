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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mad_clean.data.cutout_dataset import CutoutDataset  # noqa: E402
from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset  # noqa: E402
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
    p.add_argument("--compact_subtracted", action="store_true", default=False,
                   help="Hybrid contract: point sources removed from sky "
                        "and morphologies. Default OFF — with w_1 restored "
                        "(--drop_scales '') points are representable and "
                        "belong in training.")
    p.add_argument("--no_compact_subtracted", dest="compact_subtracted",
                   action="store_false")
    p.add_argument("--drop_scales", type=str, default="",
                   help="Comma-separated 1-based detail planes to drop from "
                        "theta. Default none: w_1 restored (Option A). The "
                        "old sub-beam drop was '1'.")
    p.add_argument("--outside_weight", type=float, default=0.05,
                   help="Loss weight on coefficients outside the true-sky "
                        "support (1.0 inside). 1.0 disables weighting. "
                        "Validation NLL is always unweighted.")
    p.add_argument("--theta_jitter", type=float, default=0.05,
                   help="Std of Gaussian dequantisation noise added to "
                        "theta (standardised units). True skies are "
                        "noiseless, so empty-sky coefficients are exactly "
                        "zero and the exact NLL is unbounded below; jitter "
                        "puts a floor on it. Applied in train AND val so "
                        "the val NLL stays comparable. 0 disables.")
    p.add_argument("--calib_samples", type=int, default=1024,
                   help="Sky cutouts for codec calibration.")
    p.add_argument("--infonce_weight", type=float, default=0.0,
                   help="Weight lambda on the in-batch InfoNCE term. 0 => "
                        "pure NLL (prior behaviour). Both terms are in "
                        "per-dim units, so lambda is a direct balance. "
                        "Proven on the overfit harness at lambda=5.")
    p.add_argument("--infonce_temp", type=float, default=0.0,
                   help="Softmax temperature tau for the L[i,j] logits. "
                        "0 => divide by theta_dim (per-dim logits, O(1)).")
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--val_every",  type=int, default=1_000)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=128)
    p.add_argument("--n_layers",      type=int, default=8)
    p.add_argument("--grad_clip",     type=float, default=10.0)
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--stacks_dir",    type=str, default=None,
                   help="Path to a casa_sim memmap stacks directory.  When "
                        "set, PatchCorpusDataset is used for training (and "
                        "validation if --val_stacks_dir is also set) instead "
                        "of the synthetic CutoutDataset.  PSF bank is not loaded.")
    p.add_argument("--val_stacks_dir", type=str, default=None,
                   help="Stacks dir for validation set.  Falls back to "
                        "--stacks_dir if omitted (uses same stacks).")
    return p.parse_args(argv)


def make_dataset(psf_bank, morphologies, extended_fraction, snr_min,
                 size, seed_offset, compact_subtracted=False,
                 stacks_dir=None):
    """Build a training/validation dataset.

    When *stacks_dir* is given, returns a PatchCorpusDataset (casa_sim
    memmap stacks); the psf_bank / morphology / snr_min arguments are
    ignored.  Otherwise falls back to the synthetic CutoutDataset path.
    """
    if stacks_dir is not None:
        print(f"[train] Using PatchCorpusDataset from {stacks_dir!r}")
        return PatchCorpusDataset(stacks_dir)

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
    res, psf, cond, sky = zip(*batch)
    img = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
    return img, torch.stack(list(cond)), torch.stack(list(sky))


def make_collate(codec, outside_weight, theta_jitter):
    """Collate that encodes sky→theta and computes dim_weights on CPU workers."""
    def _collate(batch):
        res, psf, cond, sky = zip(*batch)
        img  = torch.stack([torch.stack(list(res)), torch.stack(list(psf))], dim=1)
        cond = torch.stack(list(cond))
        sky  = torch.stack(list(sky))
        theta = codec.encode(sky)
        if theta_jitter > 0:
            theta = theta + theta_jitter * torch.randn_like(theta)
        dim_w = None
        if outside_weight < 1.0:
            dim_w = codec.support_weights(sky, outside_weight=outside_weight)
        return img, cond, theta, dim_w
    return _collate


def build_val_cache(val_ds, codec, device, val_size, theta_jitter,
                    num_workers=4, seed=0):
    """One DataLoader pass → everything encoded and cached on GPU."""
    print(f"[train] Building val cache ({val_size} scenes) on {device} ...")
    loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate)
    gen = torch.Generator(device=device).manual_seed(seed)
    imgs, conds, thetas, masks = [], [], [], []
    with torch.no_grad():
        for img, cond, sky in loader:
            img, cond = img.to(device), cond.to(device)
            sky_g = sky.to(device)
            theta = codec.encode(sky_g)
            if theta_jitter > 0:
                theta = theta + theta_jitter * torch.randn(theta.shape, generator=gen, device=device, dtype=theta.dtype)
            mask = codec.support_weights(sky_g, outside_weight=0.0) > 0.5
            imgs.append(img); conds.append(cond)
            thetas.append(theta); masks.append(mask)
    print(f"[train] Val cache ready.")
    return (torch.cat(imgs), torch.cat(conds),
            torch.cat(thetas), torch.cat(masks))


def eval_val(flow, val_cache, device, batch_size=64):
    """Val NLL from pre-cached GPU tensors -- no data generation overhead."""
    imgs, conds, thetas, masks = val_cache
    flow.eval()
    all_pd, in_pd, out_pd = [], [], []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            img  = imgs[i:i+batch_size]
            cond = conds[i:i+batch_size]
            theta = thetas[i:i+batch_size]
            mask  = masks[i:i+batch_size]
            nll_pd = -flow.log_prob_per_dim(theta, img, cond)  # (B,D)
            all_pd.append(nll_pd.mean().item())
            in_pd.append(nll_pd[mask].mean().item())
            out_pd.append(nll_pd[~mask].mean().item())
    flow.train()
    return {
        "all": float(np.mean(all_pd)),
        "in_support": float(np.mean(in_pd)),
        "out_support": float(np.mean(out_pd)),
    }


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.stacks_dir is None:
        print(f"[train] Loading PSF bank from {args.repo_root!r}/data/g55 ...")
        psf_bank = load_g55_psf_bank(
            repo_root=args.repo_root, target_size=128, rotation_augment=True,
        )
        print(f"[train] PSF bank size: {len(psf_bank)}")
    else:
        psf_bank = None

    # --- Codec calibration (resume restores it from the checkpoint) -------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        codec = StarletCodec.from_state_dict(ckpt["codec"])
        print(f"[train] Codec restored from {args.resume}")
    else:
        # sky is always the last element of the tuple regardless of dataset type
        # (index 3 for 4-tuple PatchCorpusDataset; index 4 for 5-tuple CutoutDataset)
        calib_ds = make_dataset(psf_bank, args.morphologies,
                                args.extended_fraction, args.snr_min,
                                args.calib_samples, seed_offset=20_000_000,
                                compact_subtracted=args.compact_subtracted,
                                stacks_dir=args.stacks_dir)
        n_calib = min(args.calib_samples, len(calib_ds))
        skies = torch.stack([calib_ds[i][-1] for i in range(n_calib)])
        drops = tuple(
            int(x) for x in args.drop_scales.split(",") if x.strip())
        codec = StarletCodec(image_size=128, drop_scales=drops)
        print(f"[train] Calibrating codec on {device} ...")
        codec.calibrate(skies.to(device))
    print(f"[train] theta_dim = {codec.theta_dim}")

    val_stacks = args.val_stacks_dir or args.stacks_dir
    train_ds = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.dataset_size, seed_offset=0,
                            compact_subtracted=args.compact_subtracted,
                            stacks_dir=args.stacks_dir)
    val_ds   = make_dataset(psf_bank, args.morphologies, args.extended_fraction,
                            args.snr_min, args.val_size,   seed_offset=10_000_000,
                            compact_subtracted=args.compact_subtracted,
                            stacks_dir=val_stacks)

    train_collate = make_collate(codec, args.outside_weight, args.theta_jitter)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        collate_fn=train_collate, persistent_workers=(args.num_workers > 0),
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

    val_cache = build_val_cache(
        val_ds, codec, device, args.val_size, args.theta_jitter,
        num_workers=args.num_workers)

    optimizer = optim.Adam(flow.parameters(), lr=args.lr, foreach=False)

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
    tau = args.infonce_temp if args.infonce_temp > 0 else float(D)

    print(f"[train] Training for {args.steps} steps, batch={args.batch_size}, "
          f"device={args.device}, morphologies={args.morphologies}")
    flow.train()

    while step < args.steps:
        for img, cond, theta, dim_w in train_loader:
            if step >= args.steps:
                break
            img, cond, theta = img.to(device), cond.to(device), theta.to(device)
            if dim_w is not None:
                dim_w = dim_w.to(device)
            optimizer.zero_grad()
            # NLL in per-dim units so the InfoNCE term (also per-dim scale)
            # is an interpretable balance; the summed 21k-dim NLL otherwise
            # dwarfs it.  Proven on the overfit harness (lambda=5).
            nll = flow.nll_loss(theta, img, cond, dim_weights=dim_w) / D
            loss = nll
            infonce_val = 0.0
            if args.infonce_weight > 0:
                L = flow.log_prob_matrix(theta, img, cond)  # (B, B)
                tgt = torch.arange(L.shape[0], device=device)
                infonce = F.cross_entropy(L / tau, tgt)
                loss = nll + args.infonce_weight * infonce
                infonce_val = float(infonce.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            optimizer.step()

            step += 1
            nll_val = float(nll.item())

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                total_val = nll_val + args.infonce_weight * infonce_val
                print(f"  step {step:6d}/{args.steps}  "
                      f"loss={total_val:8.4f}  nll/dim={nll_val:8.4f}  "
                      f"infonce={infonce_val:7.4f}  elapsed={elapsed / 60:.1f}m")
                log.append({"step": step, "train_loss": total_val,
                            "train_nll_per_dim": nll_val,
                            "train_infonce": infonce_val})

            if step % args.val_every == 0:
                val = eval_val(flow, val_cache, device)
                print(f"  step {step:6d}  val_nll/dim={val['all']:.4f}  "
                      f"in_support={val['in_support']:.4f}  "
                      f"out={val['out_support']:.4f}")
                if log:
                    log[-1]["val_nll_per_dim"] = val["all"]
                    log[-1]["val_nll_in_support"] = val["in_support"]
                    log[-1]["val_nll_out_support"] = val["out_support"]
                # Gate on in-support NLL: the global mean hides source misses.
                if val["in_support"] < best_val_loss:
                    best_val_loss = val["in_support"]
                    save(out_dir / "best.pt", step,
                         {"val_in_support": val["in_support"],
                          "val_nll_per_dim": val["all"]})

            if step % args.checkpoint_every == 0:
                save(out_dir / f"ckpt_{step:07d}.pt", step)
                with open(out_dir / "log.json", "w") as fh:
                    json.dump(log, fh, indent=2)

    save(out_dir / "final.pt", step)
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[train] Done. Best val_nll/dim (in_support)="
          f"{best_val_loss if best_val_loss < float('inf') else float('nan'):.4f}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
