"""One-window flow diagnostic: does the per-window flow output carry a
square pedestal (window-boundary artefact) attached to the source?

This is the cheap test that settles whether the box seams seen in the
accumulated model can be removed by a connectivity/support mask (pedestal
DISCONNECTED from source -> mask kills it) or are baked into the flow's
per-window output (pedestal ATTACHED to source -> no loop surgery helps,
needs a retrain).

Procedure (no training, one inference call):
  1. tclean(niter=0) on 3C391 -> fresh dirty (.residual) + .psf
  2. pick the global peak inside the mask
  3. extract the 128 window centred on the peak, run flow.sample
  4. save an asinh-stretched panel of the raw median output + the window
     boundary so a low-level pedestal is visible.

Run:
  PYTHONUNBUFFERED=1 pixi run -e gpu python scripts/diag_one_window.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from casatasks import tclean
from run_imaging_3c391 import (
    read_casa_image, make_mask, _measure_noise_rms, crop_psf_centred,
)
from mad_clean.models.psf_condflow import PSFCondFlow
from mad_clean.models.mdn_asp import COND_DIM


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--vis", default="/home/pjaganna/Data/imaging/3c391_ctm_mosaic_spw0.ms")
    p.add_argument("--flow_ckpt", default="models/psf_condflow_real.best.pt")
    p.add_argument("--out_dir", default="results/diag_one_window")
    p.add_argument("--imsize", type=int, default=512)
    p.add_argument("--cell", default="2.5arcsec")
    p.add_argument("--gridder", default="mosaic")
    p.add_argument("--mask_radius_px", type=int, default=200)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--floor_frac", type=float, default=0.0,
                   help="Pre-rescale relative amplitude cut in sample().")
    p.add_argument("--flow_base_channels", type=int, default=32)
    p.add_argument("--flow_asinh_a", type=float, default=1e-2)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def asinh_norm(a, scale=None):
    scale = scale if scale is not None else (np.abs(a).max() or 1.0)
    return np.arcsinh(a / (0.02 * scale))


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = args.imsize
    imgname = str(out_dir / "diag")
    casa_mask = f"circle[[{S//2}pix,{S//2}pix],{args.mask_radius_px}pix]"

    print(f"[diag] tclean niter=0 -> dirty + psf ...")
    tclean(vis=args.vis, imagename=imgname, field='', spw='', specmode='mfs',
           gridder=args.gridder, imsize=[S, S], cell=[args.cell, args.cell],
           stokes='I', weighting='briggs', robust=0.5, niter=0, mask=casa_mask,
           calcpsf=True, calcres=True, pbcor=False)

    psf = np.nan_to_num(read_casa_image(imgname + ".psf"))
    dirty = np.nan_to_num(read_casa_image(imgname + ".residual"))
    mask = make_mask(S, args.mask_radius_px)

    pr, pc = np.unravel_index(int(np.where(mask, dirty, -np.inf).argmax()), dirty.shape)
    peak = float(dirty[pr, pc])
    print(f"[diag] global peak {peak:.4e} Jy at (row={pr}, col={pc})")

    half = args.tile // 2
    r0, c0 = pr - half, pc - half
    res_cut = dirty[r0:r0 + args.tile, c0:c0 + args.tile].astype(np.float32)
    psf_cut = crop_psf_centred(psf, args.tile).astype(np.float32)

    ck = torch.load(args.flow_ckpt, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    flow = PSFCondFlow(base=cfg.get("base_channels", args.flow_base_channels),
                       asinh_a=cfg.get("asinh_a", args.flow_asinh_a))
    flow.load_state_dict(ck["model"]); flow.to(device).eval()
    print(f"[diag] loaded {args.flow_ckpt} (step {ck.get('step','?')})")

    img_t = torch.stack([torch.from_numpy(res_cut.copy()).to(device),
                         torch.from_numpy(psf_cut).to(device)])[None]
    zero_cond = torch.zeros(1, COND_DIM, device=device)
    with torch.no_grad():
        draws = flow.sample(img_t, zero_cond, n_samples=args.n_samples,
                            n_steps=args.n_steps,
                            floor_frac=args.floor_frac).squeeze(0)
    med = draws.median(dim=0).values.cpu().numpy().astype(np.float32)
    mad = (draws - draws.median(dim=0, keepdim=True).values).abs().median(dim=0).values.cpu().numpy()
    wpk = float(med.max())

    # row-cut through the peak and a column-cut, to expose a flat box pedestal
    pk = np.unravel_index(int(med.argmax()), med.shape)
    print(f"[diag] window peak {wpk:.4e}  median(min)={med.min():.3e}  "
          f"edge_mean={med[[0,-1],:].mean():.3e},{med[:,[0,-1]].mean():.3e}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    im0 = ax[0,0].imshow(asinh_norm(res_cut), origin="lower", cmap="magma")
    ax[0,0].set_title("dirty window (asinh)"); plt.colorbar(im0, ax=ax[0,0])
    im1 = ax[0,1].imshow(asinh_norm(med, wpk), origin="lower", cmap="magma")
    ax[0,1].set_title("flow median (asinh) — look for box pedestal"); plt.colorbar(im1, ax=ax[0,1])
    # hard low-level stretch: anything 0.1%..3% of peak (the pedestal band)
    im2 = ax[0,2].imshow(np.clip(med, 0, 0.03*wpk), origin="lower", cmap="magma")
    ax[0,2].set_title("flow median clipped 0..3% peak (pedestal band)"); plt.colorbar(im2, ax=ax[0,2])
    im3 = ax[1,0].imshow(med > 0.01*wpk, origin="lower", cmap="gray")
    ax[1,0].set_title("support @1% peak (is it square?)"); plt.colorbar(im3, ax=ax[1,0])
    ax[1,1].plot(med[pk[0], :]); ax[1,1].axhline(0.01*wpk, color="r", ls=":")
    ax[1,1].set_title(f"row cut through peak (row {pk[0]})"); ax[1,1].set_yscale("symlog", linthresh=1e-5)
    ax[1,2].plot(med[:, pk[1]]); ax[1,2].axhline(0.01*wpk, color="r", ls=":")
    ax[1,2].set_title(f"col cut through peak (col {pk[1]})"); ax[1,2].set_yscale("symlog", linthresh=1e-5)
    fig.suptitle(f"3C391 one-window flow diagnostic — peak {peak:.3e} Jy, window peak {wpk:.3e}")
    fig.tight_layout()
    tag = f"_floor{args.floor_frac:g}" if args.floor_frac > 0 else ""
    outpng = out_dir / f"one_window_pedestal{tag}.png"
    fig.savefig(outpng, dpi=110); plt.close(fig)
    print(f"[diag] wrote {outpng}")


if __name__ == "__main__":
    main()
