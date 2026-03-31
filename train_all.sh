#!/usr/bin/env bash
# train_all.sh — Run full training pipeline: simulate → Variant A → B → C
#
# Usage:
#   bash train_all.sh [cuda|cpu]
#
# Defaults to cuda. Pass 'cpu' for testing only (very slow for B and C).
#
# Output:
#   crumb_data/flow_pairs.npz       simulated dirty/clean pairs
#   models/cdl_filters_patch.npy    Variant A atoms
#   models/cdl_filters_conv.npy     Variant B atoms
#   models/flow_model.pt            Variant C flow model
#   logs/simulate.log
#   logs/train_A.log
#   logs/train_B.log
#   logs/train_C.log

set -euo pipefail

DEVICE="${1:-cuda}"
PIXI_ENV="gpu"
if [[ "$DEVICE" == "cpu" ]]; then
    PIXI_ENV="default"
fi

DATA="crumb_data/flow_pairs.npz"
MODELS="models"
LOGS="logs"

mkdir -p "$MODELS" "$LOGS"

echo "============================================================"
echo " MAD-CLEAN full training pipeline"
echo " Device  : $DEVICE  (pixi env: $PIXI_ENV)"
echo " Data    : $DATA"
echo " Started : $(date)"
echo "============================================================"

# ── guard: clean images must exist ──────────────────────────────────────────
if [[ ! -f "crumb_data/crumb_preprocessed.npz" ]]; then
    echo "ERROR: crumb_data/crumb_preprocessed.npz not found. Run 'pixi run fetch' first." >&2
    exit 1
fi

# ── Step 0: simulate dirty/clean pairs ──────────────────────────────────────
echo ""
echo ">>> STEP 0  simulate dirty/clean training pairs"
echo "    Output : $DATA"
echo "    Log    : $LOGS/simulate.log"
echo "    Started: $(date)"

pixi run simulate 2>&1 | tee "$LOGS/simulate.log"

echo "    Finished: $(date)"
echo ">>> Simulation complete — $DATA"

# ── Variant A — Patch dictionary ─────────────────────────────────────────────
echo ""
echo ">>> VARIANT A  (patch dictionary, OMP)"
echo "    Output : $MODELS/cdl_filters_patch.npy"
echo "    Log    : $LOGS/train_A.log"
echo "    Started: $(date)"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant A \
    --data    "$DATA" \
    --out     "$MODELS/cdl_filters_patch.npy" \
    --device  "$DEVICE" \
    --k 128 \
    --atom_size 15 \
    --lmbda 0.5 \
    --n_epochs 500 \
    --fista_iter 100 \
    --lr_d 1e-3 \
    --patches_per_img 50 \
    2>&1 | tee "$LOGS/train_A.log"

echo "    Finished: $(date)"
echo ">>> Variant A complete — $MODELS/cdl_filters_patch.npy"

# ── Variant B — Convolutional dictionary ─────────────────────────────────────
echo ""
echo ">>> VARIANT B  (convolutional CDL, PyTorch)"
echo "    Output : $MODELS/cdl_filters_conv.npy"
echo "    Log    : $LOGS/train_B.log"
echo "    Started: $(date)"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant B \
    --data    "$DATA" \
    --out     "$MODELS/cdl_filters_conv.npy" \
    --device  "$DEVICE" \
    --k 128 \
    --atom_size 15 \
    --lmbda 0.01 \
    --n_epochs 50 \
    --lr_d 1e-4 \
    --batch_size 64 \
    --fista_iter_train 100 \
    2>&1 | tee "$LOGS/train_B.log"

echo "    Finished: $(date)"
echo ">>> Variant B complete — $MODELS/cdl_filters_conv.npy"

# ── Variant C — Conditional flow matching ────────────────────────────────────
echo ""
echo ">>> VARIANT C  (conditional flow matching, dirty→clean)"
echo "    Output : $MODELS/flow_model.pt"
echo "    Log    : $LOGS/train_C.log"
echo "    Started: $(date)"

pixi run -e "$PIXI_ENV" python scripts/run_train.py \
    --variant C \
    --data    "$DATA" \
    --out     "$MODELS/flow_model.pt" \
    --device  "$DEVICE" \
    --n_epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    2>&1 | tee "$LOGS/train_C.log"

echo "    Finished: $(date)"
echo ">>> Variant C complete — $MODELS/flow_model.pt"

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All training complete: $(date)"
echo " Models:"
ls -lh "$MODELS/"*.npy "$MODELS/"*.pt 2>/dev/null || echo "  (no model files found — check logs)"
echo "============================================================"
