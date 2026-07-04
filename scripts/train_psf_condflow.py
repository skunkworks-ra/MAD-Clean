"""Train PSFCondFlow: conditional flow matching for single-source islands.

Infinite on-the-fly corpus: single source per patch (50% point, 30% Gaussian
spanning compact-to-extended, 20% arc), NO rings, no diffuse background.
Carries both 2026-06-28 fixes: spatial conditioning (dirty+PSF concatenated to
x_t, built into PSFCondFlow) and asinh target space (--asinh_a, validated to
recover extended dynamic range).  Loss is pixel-weighted CFM MSE.

GPU required.

Example (general deconvolution over a corpus PSF bank, on-GPU generation)
------------------------------------------------------------------------
    pixi run -e gpu python scripts/train_psf_condflow.py \\
        --out models/psf_condflow.pt --steps 100000 --batch_size 128 \\
        --asinh_a 1e-2 --psf_npy /path/to/corpus_stacks/train/psf.npy --gpu_gen

--gpu_gen synthesises batches entirely on the GPU (no DataLoader/CPU workers),
removing the data bottleneck that starves fast GPUs.  Without it, an on-the-fly
CPU DataLoader is used (omit --psf_npy to fall back to the in-repo G55 bank).

Held-out eval after training (restores asinh_a from the checkpoint):
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --ckpt models/psf_condflow.pt --morphology points --device cuda
    pixi run -e gpu python scripts/overfit_psf_condflow.py \\
        --ckpt models/psf_condflow.pt --morphology gaussian --device cuda
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
from mad_clean.data.gpu_sky_generator import GPUSkyGenerator
from mad_clean.data.psf_bank import load_g55_psf_bank, load_psf_bank_from_npy
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
        sky = assemble_corpus_field(
            size=size, include_diffuse=False,
            n_points=(1, 1), point_flux_range_jy=(1e-2, 1e0),
            n_ridges=(0, 0), rng=rng,
        )

    elif morph == "compact":
        sky = assemble_corpus_field(
            size=size, include_diffuse=False,
            n_points=(1, 1), point_flux_range_jy=(1e-2, 1e0),
            n_ridges=(0, 0), rng=rng,
        )
        total = float(sky.sum())
        # Span compact-to-extended Gaussians (sub-beam to large blob) so the
        # model sees the extended regime validated in the asinh overfit test.
        sigma = float(rng.uniform(1.0, 16.0))
        sky = gaussian_filter(sky.astype(np.float64), sigma=sigma).astype(np.float32)
        if sky.sum() > 1e-12:
            sky = sky * (total / sky.sum())

    else:  # arc
        sky = assemble_corpus_field(
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
        psf_npy: str | None = None,
    ):
        self.size = size
        self.base_seed = base_seed
        self.repo_root = repo_root
        self.snr_range = snr_range
        self.morph_probs = morph_probs
        self.psf_npy = psf_npy

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.base_seed + wid)
        # Corpus PSF bank (--psf_npy) for general deconvolution across PSFs;
        # falls back to the in-repo G55 bank when not supplied.
        if self.psf_npy is not None:
            bank = load_psf_bank_from_npy(self.psf_npy, target_size=self.size)
        else:
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
    p.add_argument("--asinh_a",     type=float, default=1e-2,
                   help="asinh target-space softening (validated extended fix). "
                        "Set near the normalised noise level. 0 => linear space.")
    p.add_argument("--psf_npy",     type=str,   default=None,
                   help="Corpus PSF stack (N_fields,H,W).npy for general "
                        "deconvolution. Omit to use the in-repo G55 bank.")
    p.add_argument("--gpu_gen",     action="store_true",
                   help="Generate batches entirely on the GPU (GPUSkyGenerator): "
                        "no DataLoader/CPU workers. Removes the data bottleneck "
                        "on fast GPUs. Requires --psf_npy. Morphologies: "
                        "point/blob/filament (NO rings); blob sigma 1.5-16 px.")
    p.add_argument("--gpu_noise",   type=float, default=1e-4,
                   help="Dirty-image RMS noise in --gpu_gen mode.")
    p.add_argument("--extended_fraction", type=float, default=0.5,
                   help="Fraction of non-point sources in --gpu_gen mode.")
    p.add_argument("--snr_min",     type=float, default=5.0)
    p.add_argument("--snr_max",     type=float, default=100.0)
    p.add_argument("--ema_decay",   type=float, default=0.9999)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--val_every",   type=int,   default=5_000,
                   help="Held-out PIXEL validation every N steps (EMA weights): "
                        "rel-L2(posterior median, truth) on a fixed held-out set. "
                        "Saves <out>.best.pt by rel-L2 (NOT by loss) + a val PNG. "
                        "0 disables. Requires --psf_npy.")
    p.add_argument("--val_size",    type=int,   default=64)
    p.add_argument("--val_draws",   type=int,   default=8,
                   help="Posterior draws per held-out scene for the median.")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--islands",     type=str,   default=None,
                   help="Train on realistic (T-RECS) island stacks at this dir "
                        "(dirty.npy/sky.npy/psf.npy/field_id.npy) instead of the "
                        "synthetic generator. Held-out validation is BY FIELD.")
    p.add_argument("--init_ckpt",   type=str,   default=None,
                   help="Warm-start model+EMA from this checkpoint (e.g. the "
                        "synthetic best.pt). Fine-tune: keeps the localization "
                        "that transfers; builds morphology+photometry that don't.")
    p.add_argument("--val_select",  type=str,   default="field",
                   choices=["field", "extended"],
                   help="'field': held-out fields (honest). 'extended': rank by "
                        "central extendedness for visualizing resolved sources "
                        "(NOT held out — illustrative only).")
    p.add_argument("--val_fields",  type=int,   default=10,
                   help="Number of whole fields held out for validation in "
                        "--islands mode (tests generalization to unseen fields).")
    p.add_argument("--eval_ckpt",   type=str,   default=None,
                   help="Held-out eval only: load this checkpoint's weights, run "
                        "the trusted pixel recon on the fixed held-out set, report "
                        "rel-L2/flux plus the background speckle floor vs PSF "
                        "sidelobe, save the val PNG, and exit. Requires --psf_npy.")
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

    model = PSFCondFlow(base=args.base_channels, asinh_a=args.asinh_a).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PSFCondFlow  base={args.base_channels}  params={n_params:,}")
    print(f"size={args.size}  batch={args.batch_size}  steps={args.steps}  "
          f"pw_lambda={args.pw_lambda}  asinh_a={args.asinh_a}  "
          f"psf={'corpus:' + args.psf_npy if args.psf_npy else 'g55'}  device={device}")

    if args.init_ckpt is not None:
        ick = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ick["model"])
        print(f"Warm-started model from {args.init_ckpt} (step {ick.get('step')})")

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
                    "asinh_a": args.asinh_a,
                },
            },
            out_path,
        )

    # Zero conditioning: sigma_local and config_one_hot are zero; the encoder
    # recovers absolute flux via log10(residual_scale) appended internally.
    # Kept consistent with the held-out eval (overfit_psf_condflow uses zeros).
    zero_cond = torch.zeros(args.batch_size, COND_DIM, device=device)

    # ── Held-out pixel validation set (fixed, matched distribution) ──────────
    # Drawn once with a dedicated seed so the metric is comparable over time
    # and never overlaps the training stream.  Validation = reconstruction in
    # PIXELS (rel-L2 of the posterior median vs truth), NOT held-out loss:
    # the loss floors at the irreducible posterior variance and is not a
    # quality signal (the very thing that misled us — judge pixels).
    val_gen = None
    val_img = val_sky = None
    ema_model = None
    best_val = float("inf")
    if args.val_every > 0 and args.psf_npy is not None:
        val_gen = GPUSkyGenerator(
            psf_npy=args.psf_npy, device=device, image_size=args.size,
            sigma_noise=args.gpu_noise, n_sources=(1, 1),
            extended_fraction=args.extended_fraction,
            morphologies=["point", "blob", "filament"], blob_sigma=(1.5, 16.0),
        )
        torch.manual_seed(args.seed + 7777)
        val_img, _, val_sky = val_gen.sample(args.val_size)
        val_img, val_sky = val_img.detach(), val_sky.detach()
        torch.manual_seed(args.seed)                       # restore train stream
        ema_model = PSFCondFlow(base=args.base_channels,
                                asinh_a=args.asinh_a).to(device)
        val_cond = torch.zeros(args.val_size, COND_DIM, device=device)
        print(f"Validation: {args.val_size} held-out scenes, "
              f"{args.val_draws} draws, every {args.val_every} steps (EMA).")
    elif args.val_every > 0 and args.islands is None:
        print("Validation disabled: --psf_npy not set (needs a PSF source).")

    # ── Realistic (T-RECS) island data + held-out-by-FIELD validation ────────
    isl_imgs = isl_skies = None
    if args.islands is not None:
        stacks = Path(args.islands)
        dirty_np = np.load(stacks / "dirty.npy")             # (N, H, W)
        sky_np   = np.load(stacks / "sky.npy")               # (N, H, W)
        psf_np   = np.load(stacks / "psf.npy")               # (F, H, W)
        fid_np   = np.load(stacks / "field_id.npy")          # (N,)
        N = dirty_np.shape[0]
        # 2-channel images = [dirty, field PSF], built once on device.
        dirty_t = torch.from_numpy(dirty_np).float().to(device)
        sky_t   = torch.from_numpy(sky_np).float().to(device)
        psf_t   = torch.from_numpy(psf_np).float().to(device)
        fid_t   = torch.from_numpy(fid_np).long().to(device)
        # PSF layout: either one-per-field (indexed by field_id) or already
        # one-per-window (aligned to dirty).  Detect by matching the first dim.
        psf_img = psf_t if psf_t.shape[0] == N else psf_t[fid_t]
        imgs    = torch.stack([dirty_t, psf_img], dim=1)       # (N, 2, H, W)

        # Hold out whole fields (the last val_fields field ids) for validation.
        uniq = np.unique(fid_np)
        rng_v = np.random.default_rng(args.seed)
        val_field_ids = set(rng_v.choice(uniq, size=min(args.val_fields, len(uniq)),
                                         replace=False).tolist())
        is_val = np.array([f in val_field_ids for f in fid_np])
        tr_idx = torch.from_numpy(np.where(~is_val)[0]).to(device)
        va_idx = torch.from_numpy(np.where(is_val)[0]).to(device)

        # Optional: rank the val set by central extendedness (sum/peak over the
        # central 48px crop; ~1 for a point, >>1 for resolved structure).  Used
        # only to VISUALIZE extended reconstruction — note these windows are NOT
        # held out from training (illustrative of capability, not generalization).
        if args.val_select == "extended":
            H0, W0 = sky_np.shape[-2:]
            h = 24
            crop = sky_np[:, H0 // 2 - h:H0 // 2 + h, W0 // 2 - h:W0 // 2 + h]
            csum = crop.reshape(N, -1).sum(1)
            cpk = crop.reshape(N, -1).max(1).clip(1e-12)
            ext = csum / cpk
            va_idx = torch.from_numpy(np.argsort(-ext)).to(device)
            print(f"Islands: val_select=extended — top central sum/peak = "
                  f"{ext[np.argsort(-ext)[:3]].round(1).tolist()} (leakage; viz only).")

        isl_imgs  = imgs[tr_idx]
        isl_skies = sky_t[tr_idx]
        n_va = min(args.val_size, va_idx.numel())
        va_sel = va_idx[:n_va]
        val_img  = imgs[va_sel].detach()
        val_sky  = sky_t[va_sel].detach()
        val_cond = torch.zeros(n_va, COND_DIM, device=device)
        ema_model = PSFCondFlow(base=args.base_channels,
                                asinh_a=args.asinh_a).to(device)
        print(f"Islands: {isl_imgs.shape[0]} train / {n_va} val "
              f"({len(val_field_ids)} held-out fields), "
              f"{int(fid_t.max())+1} PSFs.")

    def validate(step):
        nonlocal best_val
        ema_model.load_state_dict(ema)
        ema_model.eval()
        with torch.no_grad():
            draws = ema_model.sample(val_img, val_cond,
                                     n_samples=args.val_draws, n_steps=50)
            med = draws.median(dim=1).values                # (B, H, W)
            num = (med - val_sky).flatten(1).norm(dim=1)
            den = val_sky.flatten(1).norm(dim=1).clamp_min(1e-12)
            rel_l2 = float((num / den).median())
            flux = float((med.flatten(1).sum(1)
                          / val_sky.flatten(1).sum(1).clamp_min(1e-12)).median())
        improved = rel_l2 < best_val
        if improved:
            best_val = rel_l2
            torch.save(
                {"step": step, "model": ema, "ema": ema,
                 "config": {"base_channels": args.base_channels,
                            "size": args.size, "pw_lambda": args.pw_lambda,
                            "asinh_a": args.asinh_a},
                 "val_rel_l2": rel_l2},
                out_path.with_suffix(".best.pt"))
        print(f"  [val] step {step}: rel_l2={rel_l2:.3f}  flux_ratio={flux:.2f}"
              f"{'  *best → ' + str(out_path.with_suffix('.best.pt')) if improved else ''}")
        _save_val_fig(step, med)

    def _save_val_fig(step, med):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        n = min(4, args.val_size)
        fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
        ax = np.atleast_2d(ax)
        for i in range(n):
            for k, (img, title) in enumerate([
                (val_sky[i].cpu().numpy(), "truth"),
                (val_img[i, 0].cpu().numpy(), "dirty"),
                (med[i].cpu().numpy(), "post median"),
            ]):
                a = ax[i, k]
                d = np.log10(np.clip(img, 1e-12, None)) if k != 1 else img
                a.imshow(d, origin="lower", cmap="inferno" if k != 1 else "RdBu_r")
                if i == 0:
                    a.set_title(title, fontsize=9)
                a.axis("off")
        fig.tight_layout()
        fig.savefig(out_path.parent / f"{out_path.stem}_val.png", dpi=110)
        # Also keep a step-tagged copy so the progression is preserved.
        prog = out_path.parent / f"{out_path.stem}_val_progression"
        prog.mkdir(exist_ok=True)
        fig.savefig(prog / f"step{step:06d}.png", dpi=110)
        plt.close(fig)

    # ── Held-out eval only ──────────────────────────────────────────────────
    # Load a checkpoint, run the trusted pixel recon on the fixed held-out set,
    # and additionally quantify the background speckle floor against the PSF
    # sidelobe level: this is the number that decides whether a simple threshold
    # cut (below the first sidelobe) cleanly removes the salt-and-pepper floor
    # without eating faint sources.
    if args.eval_ckpt is not None:
        if val_gen is None and isl_imgs is None:
            raise SystemExit("--eval_ckpt requires --psf_npy (synthetic) or "
                             "--islands (realistic held-out fields).")
        ck = torch.load(args.eval_ckpt, map_location=device, weights_only=True)
        ema_model.load_state_dict(ck["model"])
        ema_model.eval()
        with torch.no_grad():
            draws = ema_model.sample(val_img, val_cond,
                                     n_samples=args.val_draws, n_steps=50)
            med = draws.median(dim=1).values                  # (B, H, W)
            num = (med - val_sky).flatten(1).norm(dim=1)
            den = val_sky.flatten(1).norm(dim=1).clamp_min(1e-12)
            rel = (num / den)
            true_flux = val_sky.flatten(1).sum(1).clamp_min(1e-12)
            flux = (med.flatten(1).sum(1) / true_flux)
            # Operational threshold: clear pixels below t * predicted peak (the
            # cleaned image's own peak, sidelobe-unaware), at 1% — far under the
            # 27% first sidelobe.  Recompute flux to test the speckle removal.
            pred_peak = med.flatten(1).max(1).values.clamp_min(1e-12)
            med_thr = torch.where(med >= 0.01 * pred_peak.view(-1, 1, 1),
                                  med, torch.zeros_like(med))
            flux_thr = (med_thr.flatten(1).sum(1) / true_flux)

            # Per-scene true peak and background speckle floor.  Background =
            # pixels where truth < 1% of that scene's peak.  Floor reported as a
            # fraction of true peak (99.9th-pct of |median| in the background).
            true_peak = val_sky.flatten(1).max(1).values.clamp_min(1e-12)   # (B,)
            bg_mask = val_sky < (0.01 * true_peak.view(-1, 1, 1))
            big = torch.where(bg_mask, med.abs(), torch.zeros_like(med))
            bg_floor = torch.quantile(big.flatten(1), 0.999, dim=1)         # (B,)
            floor_frac = (bg_floor / true_peak)                             # (B,)

        # PSF first-sidelobe level: max of |psf|/peak outside the central beam.
        # Synthetic uses the single generator PSF; islands use per-field PSFs, so
        # report the median first-sidelobe across the field PSF bank.
        if val_gen is not None:
            psf_bank = val_gen.psf_bank
            psf_stack = (psf_bank if psf_bank.ndim == 3
                         else psf_bank.unsqueeze(0)).to(device).float()
        else:
            psf_stack = psf_t.to(device).float()              # (F, H, W)
        H, W = psf_stack.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing="ij")
        r = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2).sqrt()
        outer = r > 5
        psf_n = psf_stack / psf_stack.abs().amax(dim=(-2, -1),
                                                 keepdim=True).clamp_min(1e-12)
        sidelobe = float(psf_n.abs()[:, outer].max(dim=1).values.median())

        def _q(x):
            return (float(torch.quantile(x, 0.25)), float(x.median()),
                    float(torch.quantile(x, 0.75)))
        rl, fl, ft, ff = _q(rel), _q(flux), _q(flux_thr), _q(floor_frac)
        print(f"[eval] ckpt={args.eval_ckpt}  step={ck.get('step', -1)}  "
              f"n={args.val_size} held-out scenes, {args.val_draws} draws")
        print(f"[eval] rel_L2(median,truth):  median={rl[1]:.3f}  "
              f"IQR=[{rl[0]:.3f}, {rl[2]:.3f}]   (lower better)")
        print(f"[eval] total flux ratio:      median={fl[1]:.3f}  "
              f"IQR=[{fl[0]:.3f}, {fl[2]:.3f}]  (1.0 = exact)")
        print(f"[eval] flux ratio @1% thresh: median={ft[1]:.3f}  "
              f"IQR=[{ft[0]:.3f}, {ft[2]:.3f}]  (speckle removed)")
        print(f"[eval] speckle floor / peak:  median={ff[1]:.4f}  "
              f"IQR=[{ff[0]:.4f}, {ff[2]:.4f}]  (background 99.9th pct)")
        print(f"[eval] PSF first sidelobe:    {sidelobe:.4f} of peak  "
              f"(threshold below this clears speckle if floor < sidelobe)")
        verdict = ("CLEAN: speckle floor below sidelobe → a single threshold "
                   "removes it" if ff[1] < sidelobe else
                   "OVERLAP: speckle floor >= sidelobe → thresholding risks "
                   "eating faint real flux")
        print(f"[eval] verdict: {verdict}")

        # ── Eval figures: linear stretch pinned to truth's peak ─────────────
        # (log10 flatters the speckle floor; linear shows true recovery.)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sky_np  = val_sky.cpu().numpy()
        med_np  = med.cpu().numpy()
        dirty_np_ = val_img[:, 0].cpu().numpy()
        B = sky_np.shape[0]
        figdir = out_path.parent
        stem = out_path.stem

        def _panel(a, img, vmin, vmax, cmap):
            a.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            a.axis("off")

        # Full-field pages, 4 scenes each: truth | dirty | post median | residual
        cols = ["truth", "dirty", "post median", "residual (rec−truth)"]
        per_page = 4
        n_pages = min(4, (B + per_page - 1) // per_page)
        for pg in range(n_pages):
            idx = list(range(pg * per_page, min((pg + 1) * per_page, B)))
            fig, ax = plt.subplots(len(idx), 4, figsize=(10, 2.5 * len(idx)))
            ax = np.atleast_2d(ax)
            for r, i in enumerate(idx):
                vmax = float(np.clip(sky_np[i].max(), 1e-12, None))
                resid = med_np[i] - sky_np[i]
                rlim = vmax
                _panel(ax[r, 0], sky_np[i], 0.0, vmax, "inferno")
                _panel(ax[r, 1], dirty_np_[i], None, None, "RdBu_r")
                _panel(ax[r, 2], med_np[i], 0.0, vmax, "inferno")
                _panel(ax[r, 3], resid, -rlim, rlim, "RdBu_r")
                if r == 0:
                    for k, t in enumerate(cols):
                        ax[r, k].set_title(t, fontsize=9)
            fig.tight_layout()
            p = figdir / f"{stem}_eval_page{pg}.png"
            fig.savefig(p, dpi=120); plt.close(fig)
            print(f"[eval] full-field page → {p}")

        # Center zoom-ins on the most complex central sources.  Window is
        # source-centred; rank scenes by total truth flux inside the central
        # crop (a proxy for extended/complex central structure), take top 4.
        H, W = sky_np.shape[-2:]
        half = 24                                    # 48px central crop
        cy, cx = H // 2, W // 2
        csl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        central_flux = sky_np[:, csl[0], csl[1]].reshape(B, -1).sum(1)
        order = np.argsort(-central_flux)[:4]
        fig, ax = plt.subplots(len(order), 3, figsize=(7.5, 2.5 * len(order)))
        ax = np.atleast_2d(ax)
        zcols = ["truth (center)", "post median (center)", "residual"]
        for r, i in enumerate(order):
            t = sky_np[i][csl]; m = med_np[i][csl]
            vmax = float(np.clip(t.max(), 1e-12, None))
            _panel(ax[r, 0], t, 0.0, vmax, "inferno")
            _panel(ax[r, 1], m, 0.0, vmax, "inferno")
            _panel(ax[r, 2], m - t, -vmax, vmax, "RdBu_r")
            if r == 0:
                for k, tt in enumerate(zcols):
                    ax[r, k].set_title(tt, fontsize=9)
        fig.tight_layout()
        pz = figdir / f"{stem}_eval_zoom.png"
        fig.savefig(pz, dpi=130); plt.close(fig)
        print(f"[eval] center zoom → {pz}")
        return

    model.train()
    t0 = time.time()
    running = 0.0
    step = 0
    gen = torch.Generator(device=device).manual_seed(args.seed + 99)

    def train_step(image, s0):
        """One optimisation step on a device-resident (image, s0) batch."""
        nonlocal step, running
        cond = zero_cond[:image.shape[0]]
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
        if val_img is not None and step % args.val_every == 0:
            validate(step)

    if args.islands is not None:
        # Sample random minibatches from the device-resident real islands.
        n_tr = isl_imgs.shape[0]
        ib_gen = torch.Generator(device=device).manual_seed(args.seed + 7)
        while step < args.steps:
            sel = torch.randint(0, n_tr, (args.batch_size,),
                                device=device, generator=ib_gen)
            step += 1
            train_step(isl_imgs[sel], isl_skies[sel][:, None])
    elif args.gpu_gen:
        # On-GPU generation: no DataLoader, no CPU workers.
        if args.psf_npy is None:
            raise SystemExit("--gpu_gen requires --psf_npy (corpus PSF stack).")
        sky_gen = GPUSkyGenerator(
            psf_npy=args.psf_npy, device=device, image_size=args.size,
            sigma_noise=args.gpu_noise, n_sources=(1, 1),
            extended_fraction=args.extended_fraction,
            morphologies=["point", "blob", "filament"],   # NO rings
            blob_sigma=(1.5, 16.0),                         # compact-to-extended
        )
        print(f"GPU generator: {len(sky_gen.psf_bank)} PSFs, single-source, "
              f"point/blob/filament")
        while step < args.steps:
            img, _, sky = sky_gen.sample(args.batch_size)   # img=[dirty,psf]
            step += 1
            train_step(img, sky[:, None])
    else:
        ds = _SingleSourceStream(
            size=args.size,
            base_seed=args.seed + 1,
            repo_root=_REPO_ROOT,
            snr_range=(args.snr_min, args.snr_max),
            morph_probs=(0.5, 0.3, 0.2),
            psf_npy=args.psf_npy,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=(device.type == "cuda"),
        )
        for image, s0 in loader:
            if step >= args.steps:
                break
            step += 1
            train_step(image.to(device, non_blocking=True),
                       s0.to(device, non_blocking=True)[:, None])

    save(args.steps)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
