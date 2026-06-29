"""Greedy flow-CLEAN on 3C391 — ONE major cycle, with snapshots.

Standalone (does not touch run_imaging_3c391.py). Runs a single major cycle so
you can watch the model build up component-by-component, à-la-CLEAN:

  1. tclean(niter=0)               -> dirty (.residual) + .psf
  2. greedy minor cycle:           peak -> 128 window -> flow -> connected
                                   component about the centre -> commit gain*comp
                                   -> subtract gain*PSF(conv)comp -> next peak
     (writes .model_snapNNNN every --snapshot_every commits for CARTA scrubbing)
  3. write the accumulated model into .model
  4. tclean(niter=0, calcres)      -> the residual after this one major cycle
  5. STOP

Run:
  PYTHONUNBUFFERED=1 pixi run -e gpu python scripts/run_greedy_flow_3c391.py \
      --device cuda --global_threshold 8e-3 --gain 0.6 --snapshot_every 20
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_REPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from casatasks import tclean

# reuse the CASA helpers from the existing script (import, do not duplicate)
from run_imaging_3c391 import (
    read_casa_image, write_casa_image, make_mask,
    _measure_noise_rms, crop_psf_centred,
)
from casatools import image as iatool

from mad_clean.minor_cycle import minor_cycle_flow_greedy
from mad_clean.models.psf_condflow import PSFCondFlow


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--vis", default="/home/pjaganna/Data/imaging/3c391_ctm_mosaic_spw0.ms")
    p.add_argument("--flow_ckpt", default="models/psf_condflow_real.best.pt")
    p.add_argument("--out_dir", default="results/3c391_greedy_flow")
    p.add_argument("--imsize", type=int, default=512)
    p.add_argument("--cell", default="2.5arcsec")
    p.add_argument("--gridder", default="mosaic")
    p.add_argument("--mask_radius_px", type=int, default=200)
    p.add_argument("--global_threshold", type=float, default=8e-3,
                   help="Peak stop (Jy/beam). Set from the value the data reaches.")
    p.add_argument("--gain", type=float, default=0.6,
                   help="Fraction of the connected component committed per visit.")
    p.add_argument("--eps_frac", type=float, default=0.01,
                   help="Binarise the flow window at this fraction of its peak for "
                        "connected-component despeckling.")
    p.add_argument("--max_major", type=int, default=20,
                   help="Max major cycles (run to convergence).")
    p.add_argument("--max_components", type=int, default=2000)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--snapshot_every", type=int, default=20,
                   help="Write .model_snapNNNN every N commits (0=off).")
    p.add_argument("--flow_base_channels", type=int, default=32)
    p.add_argument("--flow_asinh_a", type=float, default=1e-2)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = args.imsize
    imgname = str(out_dir / "3c391_greedy")
    casa_mask = f"circle[[{S//2}pix,{S//2}pix],{args.mask_radius_px}pix]"

    print(f"[greedy] Initial tclean (niter=0, calcpsf) ... mask={casa_mask}")
    tclean(vis=args.vis, imagename=imgname, field='', spw='', specmode='mfs',
           gridder=args.gridder, imsize=[S, S], cell=[args.cell, args.cell],
           stokes='I', weighting='briggs', robust=0.5, niter=0, mask=casa_mask,
           calcpsf=True, calcres=True, pbcor=False)

    psf      = np.nan_to_num(read_casa_image(imgname + ".psf"))
    residual = np.nan_to_num(read_casa_image(imgname + ".residual"))
    mask = make_mask(S, args.mask_radius_px)

    # --- load flow ---
    ck = torch.load(args.flow_ckpt, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    flow = PSFCondFlow(base=cfg.get("base_channels", args.flow_base_channels),
                       asinh_a=cfg.get("asinh_a", args.flow_asinh_a))
    flow.load_state_dict(ck["model"]); flow.to(device).eval()
    print(f"[greedy] Loaded {args.flow_ckpt} (step {ck.get('step','?')})")

    sigma_rms = _measure_noise_rms(residual, mask)
    threshold = args.global_threshold
    res_peak = float(np.where(mask, residual, 0.0).max())
    print(f"[greedy] dirty_peak={res_peak:.4e}  off-source 3-sigma={3*sigma_rms:.4e}  "
          f"stop_threshold={threshold:.4e} (override)  gain={args.gain}")

    # snapshot writer: copy the (zeroed) .model once as a template, then putchunk
    snap_template = imgname + ".model"
    def snapshot_cb(model_update, n):
        snap = f"{imgname}.model_snap{n:04d}"
        if Path(snap).exists():
            shutil.rmtree(snap)
        shutil.copytree(snap_template, snap)
        write_casa_image(snap, model_update)
        print(f"  [snap] {n} components -> {snap}")

    # --- major cycle loop to convergence ---
    for major in range(args.max_major):
        residual = np.nan_to_num(read_casa_image(imgname + ".residual"))
        res_peak = float(np.where(mask, residual, 0.0).max())
        print(f"\n[greedy] Major cycle {major}  residual_peak={res_peak:.4e}  "
              f"stop_threshold={threshold:.4e}")
        if res_peak < threshold:
            print("[greedy] Residual peak below threshold — converged.")
            break

        masked_residual = np.where(mask, residual, 0.0).astype(np.float32)
        model_update, commits, stop_reason = minor_cycle_flow_greedy(
            residual=masked_residual, psf=psf, sigma=sigma_rms, model=flow,
            threshold=threshold, gain=args.gain, eps_frac=args.eps_frac,
            n_samples=args.n_samples, n_steps=args.n_steps,
            max_components=args.max_components, image_mask=mask,
            snapshot_every=args.snapshot_every, snapshot_cb=snapshot_cb,
            device=args.device,
        )
        if not commits:
            print(f"[greedy] No components committed (stop={stop_reason}) — stopping.")
            break

        # accumulate into the persistent CASA model
        model_img = read_casa_image(imgname + ".model")
        model_img += model_update
        write_casa_image(imgname + ".model", model_img)
        print(f"[greedy]   committed {len(commits)} components (stop={stop_reason})")
        print(f"[greedy]   total model flux cleaned this major cycle = {model_update.sum():.4f} Jy  "
              f"(cumulative model = {model_img.sum():.4f} Jy)")

        # per-major-cycle model snapshot for CARTA scrubbing
        snap = f"{imgname}.model_maj{major:02d}"
        if Path(snap).exists():
            shutil.rmtree(snap)
        shutil.copytree(imgname + ".model", snap)

        # exact re-image
        tclean(vis=args.vis, imagename=imgname, field='', spw='', specmode='mfs',
               gridder=args.gridder, imsize=[S, S], cell=[args.cell, args.cell],
               stokes='I', weighting='briggs', robust=0.5, niter=0, mask=casa_mask,
               calcpsf=False, calcres=True, restart=True, pbcor=False)

        new_res = np.nan_to_num(read_casa_image(imgname + ".residual"))
        new_peak = float(np.where(mask, new_res, 0.0).max())
        print(f"[greedy]   residual peak: {res_peak:.4e} -> {new_peak:.4e}")
        if new_peak > res_peak * 1.2:
            print(f"[greedy] Residual peak rose >20% — stopping (divergence).")
            break

    print(f"\n[greedy] Done. model={imgname}.model  residual={imgname}.residual")
    print(f"[greedy] Per-major snapshots: {imgname}.model_majNN  (open in CARTA to scrub)")


if __name__ == "__main__":
    main()
