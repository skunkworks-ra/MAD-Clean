#!/usr/bin/env bash
# train_all.sh — Full VLA PSF training pipeline: extract → simulate → A → B → C
#
# Usage:
#   bash train_all.sh [cuda|cpu] [/path/to/casa.psf]
#
# Resume Variant C only (continues from existing flow_model.pt):
#   bash train_all.sh cuda resume_c [extra_epochs]
#   e.g.  bash train_all.sh cuda resume_c 200
#
# Arguments:
#   $1  device        cuda (default) or cpu
#   $2  PSF_SOURCE    CASA image table path for the VLA PSF, OR "resume_c"
#                     (default: /home/pjaganna/Data/taylor_home/three_points/dirty_spw4.psf)
#   $3  extra_epochs  [resume_c only] additional epochs to train (default 200)
#
# Outputs (all .npz except the PyTorch flow model):
#   models/psf.npz                  extracted VLA PSF
#   crumb_data/flow_pairs_vla.npz   simulated dirty/clean pairs
#   models/cdl_filters_patch.npz    Variant A atoms
#   models/cdl_filters_conv.npz     Variant B atoms  (PSF-residual training)
#   models/flow_model.pt            Variant C flow model
#   logs/                           per-step logs

set -euo pipefail

DEVICE="${1:-cuda}"
PSF_SOURCE="${2:-/home/pjaganna/Data/taylor_home/three_points/dirty_spw4.psf}"

PSF_NPZ="models/psf.npz"
DATA="crumb_data/flow_pairs_vla.npz"
MODELS="models"
LOGS="logs"

PIXI_ENV="gpu"
[[ "$DEVICE" == "cpu" ]] && PIXI_ENV="default"

mkdir -p "$MODELS" "$LOGS"

# ── Resume Variant C only ─────────────────────────────────────────────────────
if [[ "$PSF_SOURCE" == "resume_c" ]]; then
    EXTRA_EPOCHS="${3:-200}"
    CHECKPOINT="$MODELS/flow_model.pt"

    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: no checkpoint found at $CHECKPOINT — run full training first." >&2
        exit 1
    fi
    if [[ ! -f "$DATA" ]]; then
        echo "ERROR: training data not found at $DATA — run full training first." >&2
        exit 1
    fi

    echo "============================================================"
    echo " Resuming Variant C — $EXTRA_EPOCHS additional epochs"
    echo " Checkpoint : $CHECKPOINT"
    echo " Data       : $DATA"
    echo " Device     : $DEVICE"
    echo " Started    : $(date)"
    echo "============================================================"

    pixi run -e "$PIXI_ENV" python scripts/run_train.py \
        --variant    C \
        --data       "$DATA" \
        --out        "$CHECKPOINT" \
        --device     "$DEVICE" \
        --n_epochs   "$EXTRA_EPOCHS" \
        --batch_size 8 \
        --lr         1e-4 \
        --resume     "$CHECKPOINT" \
        2>&1 | tee "$LOGS/resume_C.log"

    echo ""
    echo "============================================================"
    echo " Resume complete: $(date)"
    echo " Model: $(ls -lh $CHECKPOINT)"
    echo "============================================================"
    exit 0
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " MAD-CLEAN full training pipeline"
echo " Device    : $DEVICE  (pixi env: $PIXI_ENV)"
echo " PSF source: $PSF_SOURCE"
echo " Data      : $DATA"
echo " Started   : $(date)"
echo "============================================================"

# ── guard: clean images must exist ──────────────────────────────────────────
if [[ ! -f "crumb_data/crumb_preprocessed.npz" ]]; then
    echo "ERROR: crumb_data/crumb_preprocessed.npz not found. Run 'pixi run fetch' first." >&2
    exit 1
fi

# ── Step 0a: extract VLA PSF (skipped if already exists) ─────────────────────
echo ""
if [[ -f "$PSF_NPZ" ]]; then
    echo ">>> STEP 0a  PSF already extracted — $PSF_NPZ  (skipping)"
else
    echo ">>> STEP 0a  extract VLA PSF from CASA image"
    echo "    Source : $PSF_SOURCE"
    echo "    Output : $PSF_NPZ"
    echo "    Log    : $LOGS/extract_psf.log"

    python /home/pjaganna/Software/MAD-clean/scripts/extract_psf.py \
        --psf  "$PSF_SOURCE" \
        --out  "$PSF_NPZ" \
        --size 150 \
        2>&1 | tee "$LOGS/extract_psf.log"

    echo ">>> PSF extracted — $PSF_NPZ"
fi

# ── Step 0b: simulate dirty/clean pairs ──────────────────────────────────────
echo ""
echo ">>> STEP 0b  simulate dirty/clean pairs with VLA PSF"
echo "    Output : $DATA"
echo "    Log    : $LOGS/simulate.log"

pixi run -e "$PIXI_ENV" python scripts/simulate_observations.py \
    --data      crumb_data/crumb_preprocessed.npz \
    --psf       "$PSF_NPZ" \
    --noise_std 0.05 \
    --out       "$DATA" \
    2>&1 | tee "$LOGS/simulate.log"

echo ">>> Simulation complete — $DATA"

# ── Variant A — Patch dictionary (clean-image representation) ─────────────────
echo ""
echo ">>> VARIANT A  (patch dictionary, OMP — trains on clean images)"
echo "    Output : $MODELS/cdl_filters_patch.npz"
echo "    Log    : $LOGS/train_A.log"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant        A \
    --data           "$DATA" \
    --out            "$MODELS/cdl_filters_patch.npz" \
    --device         "$DEVICE" \
    --k              128 \
    --atom_size      15 \
    --lmbda          0.5 \
    --n_epochs       500 \
    --fista_iter     100 \
    --lr_d           1e-3 \
    --patches_per_img 50 \
    2>&1 | tee "$LOGS/train_A.log"

echo ">>> Variant A complete — $MODELS/cdl_filters_patch.npz"

# ── Variant B — CDL with PSF-residual loss ────────────────────────────────────
echo ""
echo ">>> VARIANT B  (CDL FISTA — PSF-residual loss, trains on dirty images)"
echo "    Output : $MODELS/cdl_filters_conv.npz"
echo "    Log    : $LOGS/train_B.log"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant          B \
    --data             "$DATA" \
    --out              "$MODELS/cdl_filters_conv.npz" \
    --device           "$DEVICE" \
    --k                128 \
    --atom_size        15 \
    --lmbda            0.01 \
    --n_epochs         50 \
    --lr_d             1e-4 \
    --batch_size       64 \
    --fista_iter_train 100 \
    2>&1 | tee "$LOGS/train_B.log"

echo ">>> Variant B complete — $MODELS/cdl_filters_conv.npz"

# ── Variant C — Conditional flow matching ────────────────────────────────────
echo ""
echo ">>> VARIANT C  (conditional flow matching, dirty→clean)"
echo "    Output : $MODELS/flow_model.pt"
echo "    Log    : $LOGS/train_C.log"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant    C \
    --data       "$DATA" \
    --out        "$MODELS/flow_model.pt" \
    --device     "$DEVICE" \
    --n_epochs   100 \
    --batch_size 8 \
    --lr         1e-4 \
    2>&1 | tee "$LOGS/train_C.log"

echo ">>> Variant C complete — $MODELS/flow_model.pt"

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All training complete: $(date)"
echo " Models:"
ls -lh "$MODELS/"*.npz "$MODELS/"*.pt 2>/dev/null || echo "  (no model files found — check logs)"
echo "============================================================"
