#!/usr/bin/env bash
# retrain_phase2.sh — Retrain Phase 2 pipeline: VAE(β=0) → z codes → latent flow
#
# Usage:
#   bash retrain_phase2.sh [cuda|cpu]
#
# This retrains ONLY the Phase 2 components (VAE, z codes, latent flow prior).
# Variants A, B, C, P are not touched.

set -euo pipefail

DEVICE="${1:-cuda}"
DATA="crumb_data/flow_pairs_vla.npz"
MODELS="models"
LOGS="logs"

PIXI_ENV="gpu"
[[ "$DEVICE" == "cpu" ]] && PIXI_ENV="default"

mkdir -p "$MODELS" "$LOGS"

echo "============================================================"
echo " Phase 2 retraining:  VAE(β=0) → z codes → latent flow"
echo " Device : $DEVICE"
echo " Data   : $DATA"
echo " Started: $(date)"
echo "============================================================"

# ── Step 1: Retrain VAE with β=0 (deterministic AE) ─────────────────────────
echo ""
echo ">>> STEP 1/3  VAE (β=0, deterministic AE)"
echo "    Output : $MODELS/vae_d128.pt"

pixi run -e "$PIXI_ENV" python -u scripts/train.py \
    --variant    V \
    --data       "$DATA" \
    --out        "$MODELS/vae_d128.pt" \
    --device     "$DEVICE" \
    --n_epochs   200 \
    --batch_size 16 \
    --lr         1e-3 \
    --latent_dim 128 \
    --beta       0.0 \
    2>&1 | tee "$LOGS/retrain_vae.log"

echo ">>> VAE complete"

# ── Step 2: Collect z codes ──────────────────────────────────────────────────
echo ""
echo ">>> STEP 2/3  Collect z codes"
echo "    Output : crumb_data/z_codes_d128.npz"

pixi run -e "$PIXI_ENV" python -u scripts/collect_z_codes.py \
    --data   "$DATA" \
    --vae    "$MODELS/vae_d128.pt" \
    --out    crumb_data/z_codes_d128.npz \
    --device "$DEVICE" \
    2>&1 | tee "$LOGS/retrain_collect_z.log"

echo ">>> Z codes collected"

# ── Step 3: Retrain latent flow prior ────────────────────────────────────────
echo ""
echo ">>> STEP 3/3  Latent flow prior"
echo "    Output : $MODELS/latent_prior_d128.pt"

pixi run -e "$PIXI_ENV" python -u scripts/train.py \
    --variant    Q \
    --data       crumb_data/z_codes_d128.npz \
    --out        "$MODELS/latent_prior_d128.pt" \
    --device     "$DEVICE" \
    --n_epochs   300 \
    --batch_size 64 \
    --lr         1e-3 \
    2>&1 | tee "$LOGS/retrain_latent_flow.log"

echo ">>> Latent flow prior complete"

# ── Quick sanity check ───────────────────────────────────────────────────────
echo ""
echo ">>> Sanity check: VAE encode→decode + latent flow sample→decode"

pixi run -e "$PIXI_ENV" python -u -c "
import numpy as np, torch
from mad_clean.training.vae import VAEModel
from mad_clean.training.latent_flow import LatentFlowModel

vae = VAEModel.load('$MODELS/vae_d128.pt', device='$DEVICE')
lf  = LatentFlowModel.load('$MODELS/latent_prior_d128.pt', device='$DEVICE')

# Encode/decode check
clean = np.load('$DATA')['clean'][:10]
x = torch.from_numpy(clean[:, None]).float().to('$DEVICE')
peak = x.amax(dim=(2,3), keepdim=True).clamp(min=1e-8)
x = x / peak
with torch.no_grad():
    mu, _ = vae._net.encode(x)
    recon = vae._net.decode(mu)
print(f'Encoder mu std: {mu.std():.4f}  (want >> 0.01)')
print(f'Recon range: [{recon.min():.4f}, {recon.max():.4f}]  (want close to [0, 1])')
rms = ((recon[:,0].cpu().numpy() - x.cpu().numpy()[:,0])**2).mean()**0.5
print(f'Recon RMS error: {rms:.4f}')

# Flow prior check
z_codes = np.load('crumb_data/z_codes_d128.npz')['z_codes']
z_samples = lf.sample(100, device='$DEVICE')
with torch.no_grad():
    x_from_prior = vae._net.decode(z_samples.to('$DEVICE'))
print(f'Encoder z std:  {z_codes.std():.4f}')
print(f'Prior z std:    {z_samples.std():.4f}  (want similar to encoder)')
print(f'Decoded from prior: [{x_from_prior.min():.4f}, {x_from_prior.max():.4f}]')
"

echo ""
echo "============================================================"
echo " Phase 2 retraining complete: $(date)"
echo " Models:"
ls -lh "$MODELS/vae_d128.pt" "$MODELS/latent_prior_d128.pt" crumb_data/z_codes_d128.npz
echo ""
echo " Next: run evaluate_morphology.py and tarp_test.py with"
echo "   --data crumb_data/flow_pairs_vla.npz  (not flow_pairs.npz!)"
echo "============================================================"
