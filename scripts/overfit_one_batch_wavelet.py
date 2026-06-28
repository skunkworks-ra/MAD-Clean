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
from mad_clean.data.patch_corpus_dataset import PatchCorpusDataset  # noqa: E402
from mad_clean.data.psf_bank import load_g55_psf_bank, load_corpus_psf_bank  # noqa: E402
from mad_clean.data.gpu_sky_generator import GPUSkyGenerator  # noqa: E402
from mad_clean.models.coeff_flow import CoeffFlow  # noqa: E402
from mad_clean.models.conv_pixel_flow import ConvPixelFlow  # noqa: E402


def _fft_convolve(sky: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Zero-padded FFT convolution, both (B, H, W). Returns (B, H, W)."""
    B, H, W = sky.shape
    fH, fW = H + H - 1, W + W - 1
    sky_f = torch.fft.rfft2(sky, s=(fH, fW))
    psf_f = torch.fft.rfft2(psf, s=(fH, fW))
    out = torch.fft.irfft2(sky_f * psf_f, s=(fH, fW))
    y0, x0 = (fH - H) // 2, (fW - W) // 2
    return out[:, y0:y0 + H, x0:x0 + W]


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
    p.add_argument("--corpus_psf_dir", type=str, default=None,
                   help="Directory of corpus_field_XXXX_psf.fits. "
                        "When set, uses corpus PSF bank instead of G55.")
    p.add_argument("--psf_npy", type=str, default=None,
                   help="Path to (N_fields, H, W) psf.npy. When set, uses "
                        "GPUSkyGenerator to draw the frozen batch.")
    p.add_argument("--residual_weight", type=float, default=0.0,
                   help="Weight on RESOLVE-style residual loss. 0 => pure NLL.")
    p.add_argument("--l1_outside_weight", type=float, default=0.0,
                   help="Weight on L1 penalty for predicted sky outside the "
                        "truth footprint mask. Computed from sky_truth > 1%% "
                        "of sky_truth.max() per sample.")
    p.add_argument("--stacks_dir", type=str, default=None,
                   help="Path to PatchCorpusDataset stacks directory. When set, "
                        "loads the N brightest patches by sky flux instead of "
                        "generating synthetic cutouts.")
    p.add_argument("--out_dir",   type=str, default="results/overfit_wavelet")
    p.add_argument("--extended_fraction", type=float, default=0.05)
    p.add_argument("--morphologies", type=str, default="point,blob,shell,filament")
    p.add_argument("--compact_subtracted", action="store_true", default=False,
                   help="Hybrid contract: point sources removed (delta step "
                        "handles them in the loop). Default off.")
    p.add_argument("--no_compact_subtracted", dest="compact_subtracted",
                   action="store_false")
    p.add_argument("--drop_scales", type=str, default="",
                   help="Comma-separated 1-based detail planes to drop. "
                        "Default none (w_1 kept), matching train_wavelet_npe.")
    p.add_argument("--theta_jitter", type=float, default=0.05,
                   help="Dequantisation noise std on theta during training "
                        "(matches train_wavelet_npe). 0 disables.")
    p.add_argument("--calib_samples", type=int, default=256,
                   help="Sky cutouts used to calibrate the codec.")
    p.add_argument("--n_posterior", type=int, default=32)
    # Contrastive (InfoNCE) auxiliary.  MLE alone does not penalise a model
    # that puts the same (marginal-mean) mass on every scene; the softmax
    # over candidate thetas does, by normalising across alternatives.  This
    # is the cross-assignment gate turned into a training signal.
    p.add_argument("--sky_weight", action="store_true", default=True,
                   help="Weight NLL per pixel by true sky value (normalised per "
                        "sample). Suppresses background-pixel dominance.")
    p.add_argument("--no_sky_weight", dest="sky_weight", action="store_false")
    p.add_argument("--sky_weight_floor", type=float, default=0.1,
                   help="Additive floor on sky weights before normalisation. "
                        "Keeps background pixels in the loss (prevents "
                        "conditioning collapse) while still upweighting sources.")
    p.add_argument("--sky_sigma_cut", type=float, default=0.0,
                   help="If > 0, hard-cut NLL weighting: only pixels whose true "
                        "sky exceeds (cut * sigma_noise) are scored; sub-threshold "
                        "pixels get zero weight. Bypasses --sky_weight_floor. "
                        "Replaces the sparsity penalty as the source-selection "
                        "mechanism. 0 => floor-based soft weighting (default).")
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight on L1 sparsity prior applied to posterior samples. "
                        "Pushes background pixels toward zero; sky-weighted NLL "
                        "opposes it at source pixels. Use sample_with_grad.")
    p.add_argument("--infonce_weight", type=float, default=0.0,
                   help="Weight lambda on the InfoNCE term. 0 => pure NLL "
                        "(reproduces prior behaviour).")
    p.add_argument("--infonce_temp", type=float, default=0.0,
                   help="Softmax temperature tau for the L[b,c] logits. "
                        "0 => divide by theta_dim (per-dim logits, O(1) "
                        "scale); otherwise logits are divided by tau.")
    # Model size (defaults match the planned full run)
    p.add_argument("--base_channels",     type=int, default=32)
    p.add_argument("--context_dim",       type=int, default=256)
    p.add_argument("--hidden",            type=int, default=128)
    p.add_argument("--n_layers",          type=int, default=8)
    p.add_argument("--conv_flow",         action="store_true", default=False,
                   help="Use ConvPixelFlow (CNN coupling + checkerboard masks) "
                        "instead of CoeffFlow (MLP coupling + random masks).")
    p.add_argument("--coupling_channels", type=int, default=32,
                   help="Channels inside each ConvPixelFlow coupling CNN.")
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

    if args.psf_npy is not None:
        print(f"[overfit] GPUSkyGenerator from {args.psf_npy!r} ...")
        torch.manual_seed(args.seed)
        morph_list = [m.strip() for m in args.morphologies.split(",")]
        gen = GPUSkyGenerator(
            psf_npy=args.psf_npy, device=device,
            image_size=128, sigma_noise=1e-4,
            n_sources=(1, 8), extended_fraction=0.5,
            morphologies=morph_list,
        )
        img_t, cond_t, sky_t = gen.sample(args.n_samples)
        res_list  = list(img_t[:, 0].cpu())
        psf_list  = list(img_t[:, 1].cpu())
        cond_list = list(cond_t.cpu())
        sky_list  = list(sky_t.cpu())
    elif args.stacks_dir is not None:
        # Load the N brightest patches from a real corpus stacks directory.
        print(f"[overfit] Loading real patches from {args.stacks_dir!r} ...")
        corpus_ds = PatchCorpusDataset(args.stacks_dir)
        sky_flux = np.array([
            corpus_ds[i][3].sum().item() for i in range(len(corpus_ds))
        ])
        top_idx = np.argsort(sky_flux)[::-1][:args.n_samples]
        print(f"[overfit] Selected patch indices: {top_idx.tolist()}")
        print(f"[overfit] Sky fluxes: {sky_flux[top_idx].tolist()}")
        res_list, psf_list, cond_list, sky_list = [], [], [], []
        for i in top_idx:
            r, p, c, s = corpus_ds[i]
            res_list.append(r); psf_list.append(p)
            cond_list.append(c); sky_list.append(s)
    else:
        if args.corpus_psf_dir is not None:
            psf_bank = load_corpus_psf_bank(
                corpus_fits_dir=args.corpus_psf_dir, target_size=128,
                rotation_augment=True,
            )
        else:
            psf_bank = load_g55_psf_bank(
                repo_root=args.repo_root, target_size=128, rotation_augment=True,
            )
        print(f"[overfit] PSF bank size: {len(psf_bank)}")
        ds = make_dataset(args, psf_bank, args.n_samples)
        res_list, psf_list, cond_list, sky_list = [], [], [], []
        for i in range(args.n_samples):
            r, p, c, _, s = ds[i]
            res_list.append(r); psf_list.append(p)
            cond_list.append(c); sky_list.append(s)

    # --- Pixel-space target -----------------------------------------------
    # No codec. Flow operates directly on flattened sky pixels.
    # Normalise by per-batch sky peak so the flow sees O(1) targets
    # regardless of absolute flux level. Scale is stored for decode.
    image = torch.stack([torch.stack(res_list), torch.stack(psf_list)], dim=1).to(device)
    cond  = torch.stack(cond_list).to(device)
    sky   = torch.stack(sky_list)           # (B, 128, 128)
    sky_scale = sky.abs().max().clamp_min(1e-12)
    theta = (sky / sky_scale).flatten(1).to(device)   # (B, 16384)
    THETA_DIM = 128 * 128
    print(f"[overfit] pixel-space theta_dim = {THETA_DIM}  sky_scale = {sky_scale:.4e}")

    # --- Model and training ------------------------------------------------
    if args.conv_flow:
        flow = ConvPixelFlow(
            image_size=128,
            base_channels=args.base_channels,
            context_dim=args.context_dim,
            coupling_channels=args.coupling_channels,
            n_layers=args.n_layers,
        ).to(device)
    else:
        flow = CoeffFlow(
            theta_dim=THETA_DIM,
            base_channels=args.base_channels,
            context_dim=args.context_dim,
            hidden=args.hidden,
            n_layers=args.n_layers,
        ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"[overfit] model parameters: {n_params:,}")

    # Precompute truth footprint mask per sample: sky > 1% of sky.max().
    # Used for the L1 outside-footprint penalty during training.
    # Shape (B, H, W), on device, computed from truth sky not the prediction.
    sky_dev = sky.to(device)
    # Noise-aware footprint: sky > 3 * MAD(sky), not a fraction of peak.
    # Fraction-of-peak collapses to noise for faint extended sources.
    sky_mad = 1.4826 * sky_dev.flatten(1).median(dim=1).values.abs().clamp_min(1e-12)
    footprint = sky_dev > 3.0 * sky_mad.view(-1, 1, 1)

    opt = optim.Adam(flow.parameters(), lr=args.lr, foreach=False)
    losses = []
    nlls = []
    infonces = []
    residuals = []
    l1_outsides = []
    B = args.n_samples
    tau = args.infonce_temp if args.infonce_temp > 0 else float(THETA_DIM)
    targets = torch.arange(B, device=device)
    t0 = time.time()
    flow.train()
    import torch.nn.functional as F  # noqa: E402
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        theta_step = theta
        if args.theta_jitter > 0:
            theta_step = theta + args.theta_jitter * torch.randn_like(theta)
        if args.sky_weight:
            # Sky-value weighting: bright source pixels dominate the NLL;
            # near-zero background pixels get near-zero weight so the flow
            # cannot satisfy the loss by outputting zeros everywhere.
            # Weights are normalised per sample so the total loss magnitude
            # is unchanged (mean weight = 1).
            if args.sky_sigma_cut > 0:
                # Hard sigma-cut: score only pixels above k*sigma_noise.
                # threshold in theta-space = k*sigma_noise/sky_scale (sky_scale
                # is the batch peak used to normalise theta). Mean-normalised so
                # the surviving pixels carry mass THETA_DIM (scale-stable vs /D).
                threshold = args.sky_sigma_cut * 1e-4 / sky_scale.to(device)
                sky_w = theta.clamp(min=0)
                sky_w = sky_w * (sky_w > threshold)
                sky_w = sky_w / sky_w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                sky_w = args.sky_weight_floor + theta.clamp(min=0)
                sky_w = sky_w / sky_w.mean(dim=-1, keepdim=True).clamp_min(1e-12)
            nll = -flow.log_prob(theta_step, image, cond, dim_weights=sky_w).mean()
        else:
            nll = flow.nll_loss(theta_step, image, cond)
        # Per-dim NLL so it is O(1), comparable to InfoNCE; lambda is then
        # an interpretable balance, not fighting the 21k-dim summed scale.
        loss = nll / THETA_DIM
        infonce = torch.zeros((), device=device)
        residual_loss = torch.zeros((), device=device)
        if args.infonce_weight > 0:
            rows = []
            for b in range(B):
                ib = image[b:b + 1].expand(B, -1, -1, -1)
                cb = cond[b:b + 1].expand(B, -1)
                rows.append(flow.log_prob(theta_step, ib, cb))
            L = torch.stack(rows, dim=0)  # (B, B), requires grad
            infonce = F.cross_entropy(L / tau, targets)
            loss = nll / THETA_DIM + args.infonce_weight * infonce
        sparsity_loss = torch.zeros((), device=device)
        l1_outside = torch.zeros((), device=device)
        if args.sparsity_weight > 0 or args.residual_weight > 0 or args.l1_outside_weight > 0:
            theta_s = flow.sample_with_grad(image, cond, n=1).squeeze(1)
            sky_s = theta_s.reshape(-1, 128, 128) * sky_scale.to(device)
            if args.residual_weight > 0:
                psf_ch = image[:, 1]
                dirty_pred = _fft_convolve(sky_s, psf_ch)
                dirty_obs  = image[:, 0]
                sigma = 1e-4
                residual_loss = ((dirty_obs - dirty_pred) ** 2).mean() / (sigma ** 2)
                loss = loss + args.residual_weight * residual_loss
            if args.sparsity_weight > 0:
                # L1 on all predicted pixels: sparse prior on the sky.
                # Sky-weighted NLL opposes this at source pixels (large sky_w
                # gradient); at background pixels sky_w ~ floor so L1 wins
                # and drives them toward zero.
                sparsity_loss = sky_s.abs().mean()
                loss = loss + args.sparsity_weight * sparsity_loss
            if args.l1_outside_weight > 0:
                # Penalise any predicted flux outside the truth source footprint.
                # sky_s.clamp(min=0) because the flow can predict negative values
                # which are unphysical; we only want to penalise positive outside flux.
                outside_flux = sky_s.clamp(min=0)[~footprint]
                l1_outside = outside_flux.mean()
                loss = loss + args.l1_outside_weight * l1_outside
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        losses.append(float(loss.item()))
        nlls.append(float(nll.item()))
        infonces.append(float(infonce.item()))
        residuals.append(float(residual_loss.item()))
        l1_outsides.append(float(l1_outside.item()))
        if step % 200 == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  nll/dim="
                  f"{nlls[-1] / THETA_DIM:8.4f}  "
                  f"infonce={infonces[-1]:7.4f}  "
                  f"resid={residuals[-1]:10.2f}  "
                  f"sparse={float(sparsity_loss.item()):.4e}  "
                  f"l1_out={l1_outsides[-1]:.4e}  "
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
        "diag_nll_per_dim": float(-L.diag().mean() / THETA_DIM),
        "offdiag_nll_per_dim": float(-off / THETA_DIM),
        "gate_passed": diagonal_wins == B,
        # Per-sample diagnostics: distinguishes "this scene's own theta is
        # poorly fit" (context collision) from decode-side artefacts.
        "per_sample_diag_nll_per_dim":
            (-L.diag() / THETA_DIM).tolist(),
        "log_prob_matrix_per_dim": (L / THETA_DIM).tolist(),
    }
    print(f"[overfit] cross-assignment gate: {diagonal_wins}/{B} diagonal "
          f"wins — {'PASS' if cross['gate_passed'] else 'FAIL'} "
          f"(diag nll/dim {cross['diag_nll_per_dim']:.4f}, "
          f"offdiag {cross['offdiag_nll_per_dim']:.4f})")

    # --- Posterior reconstruction diagnostics ------------------------------
    samples = flow.sample(image, cond, n=args.n_posterior)  # (B, n, D)
    B, n, D = samples.shape
    dec = (samples.reshape(B * n, D).cpu() * sky_scale).reshape(B, n, 128, 128)
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
    ax.plot(np.array(losses) / THETA_DIM)
    ax.set_xlabel("step"); ax.set_ylabel("NLL / dim"); ax.set_yscale("symlog")
    fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # Codec round-trip: truth → theta → decode. Isolates whether failure is
    # in the codec (can't represent) or the flow (can't predict theta).
    roundtrip = sky  # pixel basis: encode/decode is identity
    fig_rt, axes_rt = plt.subplots(B, 3, figsize=(10, 3.2 * B))
    axes_rt = np.atleast_2d(axes_rt)
    for b in range(B):
        for k, (panel, title) in enumerate(zip(
            [sky[b], roundtrip[b], sky[b] - roundtrip[b]],
            ["true sky", "codec round-trip", "residual (truth - rt)"],
        )):
            im = axes_rt[b, k].imshow(panel.numpy(), origin="lower")
            axes_rt[b, k].set_title(title if b == 0 else "")
            axes_rt[b, k].axis("off")
            fig_rt.colorbar(im, ax=axes_rt[b, k], fraction=0.046)
    fig_rt.tight_layout()
    fig_rt.savefig(out_dir / "codec_roundtrip.png", dpi=120)
    plt.close(fig_rt)

    from matplotlib.colors import TwoSlopeNorm  # noqa: E402
    import matplotlib.ticker as ticker  # noqa: E402

    def _sym_norm(arr):
        v = float(np.abs(arr).max()) or 1.0
        return TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    def _pos_norm(arr):
        v = float(np.abs(arr).max()) or 1.0
        return plt.Normalize(vmin=0, vmax=v)

    fig, axes = plt.subplots(B, 4, figsize=(13, 3.2 * B))
    axes = np.atleast_2d(axes)
    col_titles = ["dirty (input)", "true sky", "posterior median", "posterior std"]
    for b in range(B):
        dirty_np = image[b, 0].cpu().numpy()
        sky_np   = sky[b].numpy()
        med_np   = post_med[b].numpy()
        std_np   = post_std[b].numpy()
        panels = [
            (dirty_np, "RdBu_r",  _sym_norm(dirty_np)),
            (sky_np,   "inferno", _pos_norm(sky_np)),
            (med_np,   "inferno", _pos_norm(sky_np)),   # same scale as truth
            (std_np,   "inferno", _pos_norm(std_np)),
        ]
        for k, (panel, cmap, norm) in enumerate(panels):
            im = axes[b, k].imshow(panel, origin="lower", cmap=cmap, norm=norm,
                                   interpolation="nearest")
            if b == 0:
                axes[b, k].set_title(col_titles[k], fontsize=8)
            axes[b, k].axis("off")
            cb = fig.colorbar(im, ax=axes[b, k], fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    fig.tight_layout()
    fig.savefig(out_dir / "recon_grid.png", dpi=130)
    plt.close(fig)

    summary = {
        "cross_assignment": cross,
        "final_nll_per_dim": nlls[-1] / THETA_DIM,
        "final_infonce": infonces[-1],
        "infonce_weight": args.infonce_weight,
        "infonce_temp": tau,
        "theta_dim": THETA_DIM,
        "sky_scale": float(sky_scale),
        "n_params": n_params,
        "steps": args.steps,
        "per_sample": per_sample,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[overfit] done. final nll/dim = {summary['final_nll_per_dim']:.4f}; "
          f"outputs in {out_dir}/")
    return summary


if __name__ == "__main__":
    run(parse_args())
