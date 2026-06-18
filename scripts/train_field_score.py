"""Train the field-posterior prior score model (Fork A, design step 2).

An EDM-preconditioned conv U-Net denoiser (mad_clean.imaging.score) trained by
denoising score matching on the §4 statistics corpus (mad_clean.data.field_sky),
generated on the fly at full field size — infinite data, no stored npz.

Training is in standardised log-sky space: f = log(s), then (f - mu)/tau so the
field has zero mean / unit variance (σ_data = 1).  mu, tau are measured once from
a warmup of corpus fields and stored in the checkpoint; the Langevin loop undoes
them before the likelihood (which lives in linear sky space s = exp(f)).

GPU env required (memory: pixi run -e gpu); CPU silently falls back and is slow.

Example
-------
    pixi run -e gpu train-field-score-gpu
    # or explicitly:
    pixi run -e gpu python scripts/train_field_score.py \\
        --out models/field_score.pt --size 512 --base_channels 32 \\
        --steps 100000 --batch_size 8 --lr 1e-4
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mad_clean.data.field_sky import assemble_corpus_field
from mad_clean.imaging.score import EDMDenoiser, UNet, edm_loss

# Floor on s before log, so log(s) is finite even where the field is ~0.
_S_FLOOR = 1e-8


def _to_field(s: np.ndarray, space: str) -> np.ndarray:
    """Map a linear sky ``s`` to the training space.

    'log'    : f = log(max(s, floor)) — positivity by exp, but densifies a
               sparse sky into a full-support floor (the source-destroying
               representation isolated by the overfit gate).
    'linear' : the sky itself — empty sky is genuinely zero, sparse stays
               sparse; positivity is enforced in the sampler by clamp."""
    if space == "log":
        return np.log(np.maximum(s, _S_FLOOR)).astype(np.float32)
    return s.astype(np.float32)


class CorpusStream(IterableDataset):
    """Infinite stream of clean fields in the chosen space, generated on the fly.

    Each DataLoader worker seeds its own RNG (base_seed + worker_id) so workers
    produce distinct, non-overlapping field streams; with ``num_workers > 0`` the
    corpus is generated in parallel and prefetched while the GPU trains."""

    def __init__(self, size: int, base_seed: int, space: str, diffuse_flux: float):
        self.size = size
        self.base_seed = base_seed
        self.space = space
        self.diffuse_flux = diffuse_flux

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.base_seed + wid)
        while True:
            s = assemble_corpus_field(
                size=self.size, diffuse_flux_jy=self.diffuse_flux, rng=rng)
            yield _to_field(s, self.space)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the field prior score model.")
    p.add_argument("--out", type=str, default="models/field_score.pt")
    p.add_argument("--space", choices=["log", "linear"], default="log",
                   help="Training representation. 'linear' keeps a sparse sky "
                        "sparse (the gate-validated fix); 'log' is the original.")
    p.add_argument("--size", type=int, default=512, help="Full field side (px).")
    p.add_argument("--diffuse_flux", type=float, default=1.0,
                   help="Total diffuse flux (Jy) of the corpus. Low (e.g. 0.05) "
                        "= source-prominent sky (the gate-validated regime); 1.0 "
                        "= diffuse-dominated (the original, which collapses).")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA of weights for sampling (standard diffusion practice).")
    p.add_argument("--warmup_fields", type=int, default=256,
                   help="Corpus fields used to measure log-sky mean/std once.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers generating the corpus in parallel.")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def make_field_batch(
    batch_size: int, size: int, rng: np.random.Generator, space: str,
    diffuse_flux: float,
) -> np.ndarray:
    """A batch of clean fields in ``space``, shape (B, H, W) float32."""
    out = np.empty((batch_size, size, size), dtype=np.float32)
    for i in range(batch_size):
        s = assemble_corpus_field(size=size, diffuse_flux_jy=diffuse_flux, rng=rng)
        out[i] = _to_field(s, space)
    return out


def measure_standardisation(
    n_fields: int, size: int, rng: np.random.Generator, space: str,
    diffuse_flux: float,
) -> tuple[float, float]:
    """Measure (mu, tau) = (mean, std) of the clean field over ``n_fields``."""
    vals = make_field_batch(n_fields, size, rng, space, diffuse_flux)
    return float(vals.mean()), float(vals.std() + 1e-8)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── standardisation (measured once) ──────────────────────────────────────
    print(f"Measuring {args.space}-sky standardisation over {args.warmup_fields} fields…")
    mu, tau = measure_standardisation(
        args.warmup_fields, args.size, rng, args.space, args.diffuse_flux)
    print(f"  space={args.space}  diffuse_flux={args.diffuse_flux}  "
          f"mu={mu:.4g}  tau={tau:.4g}  (σ_data set to 1.0 in standardised space)")

    # ── model ────────────────────────────────────────────────────────────────
    net = UNet(in_ch=1, base=args.base_channels)
    model = EDMDenoiser(net, sigma_data=1.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: EDM denoiser, base={args.base_channels}, params={n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def save(step: int) -> None:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema,
                "sigma_data": 1.0,
                "mu": mu,
                "tau": tau,
                "space": args.space,
                "config": {"size": args.size, "base_channels": args.base_channels,
                           "space": args.space, "diffuse_flux": args.diffuse_flux},
            },
            out_path,
        )

    # ── corpus stream (parallel generation, prefetched) ──────────────────────
    loader = DataLoader(
        CorpusStream(args.size, base_seed=args.seed + 1, space=args.space,
                     diffuse_flux=args.diffuse_flux),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    running = 0.0
    step = 0
    for batch in loader:
        step += 1
        if step > args.steps:
            break
        f0 = batch.to(device, non_blocking=True)[:, None]  # (B,1,H,W)
        f0 = (f0 - mu) / tau

        opt.zero_grad()
        loss = edm_loss(model, f0)
        loss.backward()
        opt.step()

        with torch.no_grad():
            decay = args.ema_decay
            for k, v in model.state_dict().items():
                ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)

        running += float(loss)
        if step % args.log_every == 0:
            rate = step / (time.time() - t0)
            print(f"step {step:>7d}  loss {running/args.log_every:.4f}  "
                  f"{rate:.1f} steps/s")
            running = 0.0
        if step % args.checkpoint_every == 0:
            save(step)
            print(f"  checkpoint → {out_path}  (step {step})")

    save(args.steps)
    print(f"Done. Final checkpoint → {out_path}")
    print(json.dumps({"steps": args.steps, "params": n_params,
                      "mu": mu, "tau": tau}))


if __name__ == "__main__":
    main()
