"""Train PSFCondFlow: conditional flow matching for single-source islands.

Infinite on-the-fly corpus: single source per patch (50% point, 30% compact
Gaussian, 20% arc), no diffuse background, simulated dirty image via the G55
PSF bank.  Loss is pixel-weighted CFM MSE (source pixels up-weighted by
lambda=10 to avoid floor domination from the zero-sky background).

GPU required.

Example
-------
    pixi run -e gpu python scripts/train_psf_condflow.py \\
        --out models/psf_condflow.pt --steps 100000 --batch_size 16

Held-out eval after training:
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --ckpt models/psf_condflow.pt --morphology points --device cuda
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --ckpt models/psf_condflow.pt --morphology extended --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from scipy.ndimage import gaussian_filter

from mad_clean.data.field_sky import assemble_corpus_field
from mad_clean.data.psf_bank import load_g55_psf_bank
from mad_clean.imaging.forward import ImageDomainForward
from mad_clean.models.mdn_asp import COND_DIM
from mad_clean.models.psf_condflow import PSFCondFlow, cfm_loss


# ---------------------------------------------------------------------------
# Single-source corpus sample
# ---------------------------------------------------------------------------

def _sample_source(size, rng, morph_probs=(0.5, 0.3, 0.2)):
    """Draw one (sky, morph_tag) pair from the single-source corpus.

    morph_probs = (point, compact_gaussian, arc).
    No diffuse background: sources dominate, loss is not floor-dominated.
    """
    morph = rng.choice(["point", "compact", "arc"], p=morph_probs)

    if morph == "point":
        sky, _ = assemble_corpus_field(
            size=size, include_diffuse=False,
            n_points=(1, 1), point_flux_range_jy=(1e-2, 1e0),
            n_ridges=(0, 0), rng=rng,
        )

    elif morph == "compact":
        sky, _ = assemble_corpus_field(
            size=size, include_diffuse=False,
            n_points=(1, 1), point_flux_range_jy=(1e-2, 1e0),
            n_ridges=(0, 0), rng=rng,
        )
        total = float(sky.sum())
        sigma = float(rng.uniform(0.5, 2.5))
        sky = gaussian_filter(sky.astype(np.float64), sigma=sigma).astype(np.float32)
        if sky.sum() > 1e-12:
            sky = sky * (total / sky.sum())

    else:  # arc
        sky, _ = assemble_corpus_field(
            size=size, include_diffuse=False,
            n_points=(0, 0), n_ridges=(1, 1),
            ridge_flux_range_jy=(5e-2, 5e-1), rng=rng,
        )

    return sky.astype(np.float32)


class _SingleSourceStream(IterableDataset):
    """Infinite stream of (dirty, PSF, clean_source) triples, generated on the fly.

    Each DataLoader worker seeds its own RNG so workers produce non-overlapping
    streams.  The PSF bank is loaded once per worker.
    """

    def __init__(
        self,
        size: int,
        base_seed: int,
        repo_root: Path,
        snr_range: tuple[float, float],
        morph_probs: tuple[float, float, float],
    ):
        self.size = size
        self.base_seed = base_seed
        self.repo_root = repo_root
        self.snr_range = snr_range
        self.morph_probs = morph_probs

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.base_seed + wid)
        bank = load_g55_psf_bank(repo_root=self.repo_root, target_size=self.size)

        while True:
            sky = _sample_source(self.size, rng, self.morph_probs)
            psf_np, _ = bank.sample(rng)
            fwd = ImageDomainForward(torch.from_numpy(psf_np))

            sky_t = torch.from_numpy(sky)
            peak = float(sky_t.max())
            if peak < 1e-10:
                continue
            snr = float(rng.uniform(*self.snr_range))
            noise_std = peak / snr
            d = fwd.make_dirty(sky_t, noise_std=noise_std)

            # Stack dirty+PSF into 2-channel image
            image = torch.stack([d, torch.from_numpy(psf_np)], dim=0)  # (2, H, W)
            yield image, sky_t  # dirty+psf, clean source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train PSFCondFlow.")
    p.add_argument("--out",         type=str,   default="models/psf_condflow.pt")
    p.add_argument("--size",        type=int,   default=128)
    p.add_argument("--steps",       type=int,   default=100_000)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--pw_lambda",   type=float, default=10.0,
                   help="Pixel-weight lambda for source pixels in the CFM loss.")
    p.add_argument("--snr_min",     type=float, default=5.0)
    p.add_argument("--snr_max",     type=float, default=100.0)
    p.add_argument("--ema_decay",   type=float, default=0.9999)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = PSFCondFlow(base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PSFCondFlow  base={args.base_channels}  params={n_params:,}")
    print(f"size={args.size}  batch={args.batch_size}  steps={args.steps}  "
          f"pw_lambda={args.pw_lambda}  device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def save(step):
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema,
                "config": {
                    "base_channels": args.base_channels,
                    "size": args.size,
                    "pw_lambda": args.pw_lambda,
                },
            },
            out_path,
        )

    ds = _SingleSourceStream(
        size=args.size,
        base_seed=args.seed + 1,
        repo_root=_REPO_ROOT,
        snr_range=(args.snr_min, args.snr_max),
        morph_probs=(0.5, 0.3, 0.2),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    # Zero conditioning: sigma_local and config_one_hot are zero; the encoder
    # recovers absolute flux via log10(residual_scale) appended internally.
    zero_cond = torch.zeros(args.batch_size, COND_DIM, device=device)

    model.train()
    t0 = time.time()
    running = 0.0
    step = 0
    gen = torch.Generator(device=device).manual_seed(args.seed + 99)

    for image, s0 in loader:
        step += 1
        if step > args.steps:
            break

        image = image.to(device, non_blocking=True)    # (B, 2, H, W)
        s0 = s0.to(device, non_blocking=True)[:, None] # (B, 1, H, W)
        cond = zero_cond[:image.shape[0]]              # handle last batch

        opt.zero_grad()
        loss = cfm_loss(model, s0, image, cond,
                        pixel_weight_lambda=args.pw_lambda, generator=gen)
        loss.backward()
        opt.step()

        with torch.no_grad():
            d = args.ema_decay
            for k, v in model.state_dict().items():
                ema[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

        running += float(loss)
        if step % args.log_every == 0:
            rate = step / (time.time() - t0)
            print(f"step {step:>7d}  loss {running/args.log_every:.5f}  "
                  f"{rate:.1f} steps/s")
            running = 0.0
        if step % args.checkpoint_every == 0:
            save(step)
            print(f"  checkpoint → {out_path}  (step {step})")

    save(args.steps)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
