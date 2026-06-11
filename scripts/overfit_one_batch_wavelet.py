"""Diagnostic 1 for the wavelet-NPE head: overfit one frozen batch.

Freeze --n_samples cutouts; train CoeffFlow on them until the NLL floors.
Sanity check only — overfit converging tells us loss/architecture/codec
are wired correctly, not how the full model will generalise.

Outputs in --out_dir:
- loss_curve.png      : training NLL vs step
- recon_grid.png      : per-sample (residual | true sky | posterior mean |
                        posterior std) panels from 32 posterior samples
- summary.json        : final NLL, per-sample decode metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

from mad_clean.data.cutout_dataset import CutoutDataset  # noqa: E402
from mad_clean.data.psf_bank import load_g55_psf_bank  # noqa: E402
from mad_clean.models.coeff_flow import CoeffFlow  # noqa: E402
from mad_clean.wavelet.starlet import StarletCodec  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit one frozen batch — wavelet-NPE diagnostic 1.",
    )
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--steps",     type=int, default=2000)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--repo_root", type=str, default=".",
                   help="Repo root containing data/g55/chunk_*/psf.fits.")
    p.add_argument("--out_dir",   type=str, default="results/overfit_wavelet")
    p.add_argument("--extended_fraction", type=float, default=0.05)
    p.add_argument("--morphologies", type=str, default="point,blob,shell,filament")
    p.add_argument("--compact_subtracted", action="store_true", default=True,
                   help="Hybrid contract: point sources removed (delta step "
                        "handles them in the loop). Default on.")
    p.add_argument("--no_compact_subtracted", dest="compact_subtracted",
                   action="store_false")
    p.add_argument("--calib_samples", type=int, default=256,
                   help="Sky cutouts used to calibrate the codec.")
    p.add_argument("--n_posterior", type=int, default=32)
    # Model size (defaults match the planned full run)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--context_dim",   type=int, default=256)
    p.add_argument("--hidden",        type=int, default=512)
    p.add_argument("--n_layers",      type=int, default=8)
    return p.parse_args(argv)


def make_dataset(args, psf_bank, length, seed_offset=0):
    morph = {m.strip(): 1.0 for m in args.morphologies.split(",")}
    if args.compact_subtracted and "point" in morph:
        del morph["point"]
    return CutoutDataset(
        psf_bank=psf_bank,
        field_size=512,
        cutout_size=128,
        sigma_noise=1e-4,
        n_sources_per_field=(5, 20),
        extended_fraction=args.extended_fraction,
        rng_seed=args.seed + seed_offset,
        length=length,
        morphology_balance=morph,
        return_sky=True,
        compact_subtracted=args.compact_subtracted,
    )


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=128, rotation_augment=True,
    )

    # --- Codec calibration on a separate sample stream -------------------
    calib_ds = make_dataset(args, psf_bank, args.calib_samples,
                            seed_offset=1_000_000)
    calib_skies = torch.stack(
        [calib_ds[i][4] for i in range(args.calib_samples)]
    )
    codec = StarletCodec(image_size=128)
    codec.calibrate(calib_skies)
    print(f"[overfit] codec theta_dim = {codec.theta_dim}")

    # --- Frozen batch -----------------------------------------------------
    ds = make_dataset(args, psf_bank, args.n_samples)
    res, psf, cond, sky = [], [], [], []
    for i in range(args.n_samples):
        r, p, c, _, s = ds[i]
        res.append(r); psf.append(p); cond.append(c); sky.append(s)
    image = torch.stack([torch.stack(res), torch.stack(psf)], dim=1).to(device)
    cond  = torch.stack(cond).to(device)
    sky   = torch.stack(sky)
    theta = codec.encode(sky).to(device)

    # --- Model and training ------------------------------------------------
    flow = CoeffFlow(
        theta_dim=codec.theta_dim,
        base_channels=args.base_channels,
        context_dim=args.context_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[overfit] model parameters: {n_params:,}")

    opt = optim.Adam(flow.parameters(), lr=args.lr)
    losses = []
    t0 = time.time()
    flow.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss = flow.nll_loss(theta, image, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % 200 == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  nll/dim="
                  f"{losses[-1] / codec.theta_dim:8.4f}  "
                  f"elapsed={(time.time() - t0) / 60:.1f}m")

    # --- Cross-assignment gate ---------------------------------------------
    # A flow that memorised 8 image->theta mappings must score each scene's
    # own theta best (rank 1).  An unconditional flow is uniform random.
    # This is the PASS/FAIL gate before any full training run (added after
    # the 2026-06-11 concat-conditioning failure).
    flow.eval()
    B = args.n_samples
    L = torch.zeros(B, B)
    with torch.no_grad():
        for b in range(B):
            ib = image[b:b + 1].expand(B, -1, -1, -1)
            cb = cond[b:b + 1].expand(B, -1)
            L[b] = flow.log_prob(theta, ib, cb).cpu()
    ranks = [int((L[b] > L[b, b]).sum()) + 1 for b in range(B)]
    diagonal_wins = int(sum(r == 1 for r in ranks))
    off = (L.sum() - L.diag().sum()) / (B * B - B)
    cross = {
        "ranks": ranks,
        "diagonal_wins": diagonal_wins,
        "n_samples": B,
        "diag_nll_per_dim": float(-L.diag().mean() / codec.theta_dim),
        "offdiag_nll_per_dim": float(-off / codec.theta_dim),
        "gate_passed": diagonal_wins == B,
    }
    print(f"[overfit] cross-assignment gate: {diagonal_wins}/{B} diagonal "
          f"wins — {'PASS' if cross['gate_passed'] else 'FAIL'} "
          f"(diag nll/dim {cross['diag_nll_per_dim']:.4f}, "
          f"offdiag {cross['offdiag_nll_per_dim']:.4f})")

    # --- Posterior reconstruction diagnostics ------------------------------
    samples = flow.sample(image, cond, n=args.n_posterior)  # (B, n, D)
    B, n, D = samples.shape
    dec = codec.decode(samples.reshape(B * n, D).cpu()).reshape(B, n, 128, 128)
    # Median, not mean: the sinh decode amplifies posterior tail samples
    # exponentially, so the pixelwise mean is dominated by outliers.
    post_med = dec.median(dim=1).values
    post_std = dec.std(dim=1)

    per_sample = []
    for b in range(B):
        truth = sky[b]
        err = float((post_med[b] - truth).norm() / max(truth.norm(), 1e-12))
        flux_samples = dec[b].sum(dim=(1, 2))
        per_sample.append({
            "rel_l2_post_median": err,
            "true_flux": float(truth.sum()),
            "post_median_flux": float(post_med[b].sum()),
            "post_flux_iqr": float(
                flux_samples.quantile(0.75) - flux_samples.quantile(0.25)
            ),
        })

    # --- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(losses) / codec.theta_dim)
    ax.set_xlabel("step"); ax.set_ylabel("NLL / dim"); ax.set_yscale("symlog")
    fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(B, 4, figsize=(13, 3.2 * B))
    axes = np.atleast_2d(axes)
    titles = ["residual", "true sky", "posterior median", "posterior std"]
    for b in range(B):
        panels = [image[b, 0].cpu(), sky[b], post_med[b], post_std[b]]
        for k, (panel, title) in enumerate(zip(panels, titles)):
            im = axes[b, k].imshow(panel.numpy(), origin="lower")
            axes[b, k].set_title(title if b == 0 else "")
            axes[b, k].axis("off")
            fig.colorbar(im, ax=axes[b, k], fraction=0.046)
    fig.tight_layout(); fig.savefig(out_dir / "recon_grid.png", dpi=120)
    plt.close(fig)

    summary = {
        "cross_assignment": cross,
        "final_nll_per_dim": losses[-1] / codec.theta_dim,
        "theta_dim": codec.theta_dim,
        "n_params": n_params,
        "steps": args.steps,
        "per_sample": per_sample,
        "codec": codec.state_dict(),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[overfit] done. final nll/dim = {summary['final_nll_per_dim']:.4f}; "
          f"outputs in {out_dir}/")
    return summary


if __name__ == "__main__":
    run(parse_args())
