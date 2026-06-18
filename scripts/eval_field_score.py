"""Evaluate a trained field-posterior score model (Fork A, design step 2).

Diagnostics that actually track learning (unlike the noisy DSM training loss):

  1. Per-σ denoising error of D_θ vs the Gaussian-denoiser baseline (c_skip·x,
     i.e. what the zero-init net outputs).  If the model beats the baseline,
     especially at low/mid σ, it has learned the non-Gaussian structure.
  2. A clean / noisy / denoised figure at one σ for eyeballing.

Reads a checkpoint written by train_field_score.py.  Runs on CPU by default so
it does not contend with a running GPU training job.  Use raw weights early in
training (EMA needs ~1/(1-ema_decay) steps to warm up).

Example
-------
    pixi run python scripts/eval_field_score.py --ckpt models/field_score.pt \\
        --size 256 --n_fields 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mad_clean.data.field_sky import assemble_corpus_field
from mad_clean.imaging.score import EDMDenoiser, UNet

_S_FLOOR = 1e-8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the field score model.")
    p.add_argument("--ckpt", type=str, default="models/field_score.pt")
    p.add_argument("--size", type=int, default=256,
                   help="Field size for the held-out eval set (≤ training size is fine).")
    p.add_argument("--n_fields", type=int, default=8)
    p.add_argument("--use_ema", action="store_true",
                   help="Use EMA weights (only meaningful once EMA has warmed up).")
    p.add_argument("--sigmas", type=str, default="0.1,0.3,0.5,1.0,2.0")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=12345,
                   help="Held-out seed, distinct from the training stream.")
    p.add_argument("--fig", type=str, default="",
                   help="Optional path to save a clean/noisy/denoised figure.")
    return p.parse_args(argv)


def load_model(ckpt: dict, use_ema: bool, device) -> EDMDenoiser:
    base = ckpt["config"]["base_channels"]
    model = EDMDenoiser(UNet(in_ch=1, base=base), sigma_data=ckpt["sigma_data"]).to(device)
    state = ckpt["ema"] if use_ema else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def held_out_batch(n_fields: int, size: int, mu: float, tau: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    out = np.empty((n_fields, size, size), dtype=np.float32)
    for i in range(n_fields):
        s = assemble_corpus_field(size=size, rng=rng)
        out[i] = np.log(np.maximum(s, _S_FLOOR))
    f = torch.from_numpy(out)[:, None]  # (N,1,H,W)
    return (f - mu) / tau


@torch.no_grad()
def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    step = ckpt.get("step", "?")
    mu, tau = ckpt["mu"], ckpt["tau"]
    print(f"Checkpoint {args.ckpt}  step={step}  mu={mu:.3f} tau={tau:.3f}  "
          f"ema={args.use_ema}")

    model = load_model(ckpt, args.use_ema, device)
    f0 = held_out_batch(args.n_fields, args.size, mu, tau, args.seed).to(device)

    sigmas = [float(x) for x in args.sigmas.split(",")]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"\n{'sigma':>6} {'model_relerr':>13} {'gauss_relerr':>13} {'improve%':>9}")
    rows = []
    for sg in sigmas:
        noise = torch.randn(f0.shape, generator=gen, device=device)
        x = f0 + sg * noise
        sigma = torch.full((f0.shape[0],), sg, device=device)
        d_model = model(x, sigma)
        c_skip = model.sigma_data**2 / (sg**2 + model.sigma_data**2)
        d_gauss = c_skip * x  # zero-init / Gaussian-prior denoiser
        denom = f0.norm()
        em = float((d_model - f0).norm() / denom)
        eg = float((d_gauss - f0).norm() / denom)
        improve = 100.0 * (eg - em) / eg
        rows.append((sg, em, eg, improve))
        print(f"{sg:>6.2f} {em:>13.4f} {eg:>13.4f} {improve:>9.1f}")

    mean_improve = float(np.mean([r[3] for r in rows]))
    print(f"\nMean improvement over Gaussian baseline: {mean_improve:.1f}%")
    if mean_improve > 2.0:
        print("→ Model beats the Gaussian denoiser: it is learning structure, "
              "not stuck.")
    else:
        print("→ Model ~= Gaussian denoiser: little non-Gaussian structure "
              "learned yet (or eval σ range off).")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sg = sigmas[len(sigmas) // 2]
        noise = torch.randn(f0.shape, generator=torch.Generator(device=device).manual_seed(0),
                            device=device)
        x = f0 + sg * noise
        d = model(x, torch.full((f0.shape[0],), sg, device=device))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, (t, im) in zip(ax, [("clean f'", f0[0, 0]),
                                   (f"noisy σ={sg}", x[0, 0]),
                                   ("denoised", d[0, 0])]):
            a.imshow(im.cpu().numpy(), origin="lower", cmap="inferno")
            a.set_title(t)
            a.axis("off")
        plt.tight_layout()
        plt.savefig(args.fig, dpi=90)
        print(f"figure → {args.fig}")


if __name__ == "__main__":
    main()
