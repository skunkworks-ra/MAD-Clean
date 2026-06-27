#!/usr/bin/env bash
# Two-phase pixel-space flow training.
#
# Phase 1: Synthetic morphologies (CutoutDataset + G55 PSFs).
#           Learns the general deconvolution mapping across all morphology types.
# Phase 2: Real sky fine-tune (PatchCorpusDataset).
#           Adapts to real data distributions at lower LR.
#           Skipped if CORPUS_STACKS_DIR is not set.
#
# Stdout is shown live and also tee'd to a log file in OUT_DIR.
#
# Usage (set paths before running):
#   CORPUS_STACKS_DIR=/path/to/corpus_stacks/train \
#   VAL_STACKS_DIR=/path/to/corpus_stacks/val \
#   bash scripts/run_pixel_flow_train.sh

set -euo pipefail

# --- Paths (override via environment) ---
OUT_DIR="${OUT_DIR:-results/pixel_flow_train}"
REPO_ROOT="${REPO_ROOT:-.}"
CORPUS_STACKS_DIR="${CORPUS_STACKS_DIR:-}"
VAL_STACKS_DIR="${VAL_STACKS_DIR:-}"

# --- Phase 1 hyperparameters ---
P1_STEPS="${P1_STEPS:-50000}"
P1_BATCH="${P1_BATCH:-32}"
P1_LR="${P1_LR:-3e-4}"
P1_WORKERS="${P1_WORKERS:-4}"

# --- Phase 2 hyperparameters ---
P2_STEPS="${P2_STEPS:-20000}"
P2_BATCH="${P2_BATCH:-32}"
P2_LR="${P2_LR:-1e-4}"
P2_WORKERS="${P2_WORKERS:-4}"

# --- Shared loss settings (proven on overfit) ---
SKY_WEIGHT_FLOOR="${SKY_WEIGHT_FLOOR:-0.1}"
SPARSITY_WEIGHT="${SPARSITY_WEIGHT:-1.0}"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Phase 1: Synthetic
# ---------------------------------------------------------------------------
echo "================================================================"
echo " Phase 1: Synthetic training  (${P1_STEPS} steps)"
echo " Output: ${OUT_DIR}/phase1/"
echo "================================================================"

mkdir -p "${OUT_DIR}/phase1"

pixi run -e gpu python scripts/train_pixel_flow.py \
    --out_dir        "${OUT_DIR}/phase1" \
    --repo_root      "${REPO_ROOT}" \
    --steps          "${P1_STEPS}" \
    --batch_size     "${P1_BATCH}" \
    --lr             "${P1_LR}" \
    --num_workers    "${P1_WORKERS}" \
    --morphologies   point,blob,shell,filament \
    --extended_fraction 0.5 \
    --sky_weight_floor  "${SKY_WEIGHT_FLOOR}" \
    --sparsity_weight   "${SPARSITY_WEIGHT}" \
    --log_every      100 \
    --val_every      1000 \
    --checkpoint_every 5000 \
  2>&1 | tee "${OUT_DIR}/phase1/train.log"

echo ""
echo "Phase 1 complete. Best checkpoint: ${OUT_DIR}/phase1/best.pt"

# ---------------------------------------------------------------------------
# Phase 2: Real sky fine-tune (optional)
# ---------------------------------------------------------------------------
if [ -z "${CORPUS_STACKS_DIR}" ]; then
    echo ""
    echo "CORPUS_STACKS_DIR not set — skipping phase 2 fine-tune."
    echo "To run phase 2 later:"
    echo "  CORPUS_STACKS_DIR=/path/to/stacks/train \\"
    echo "  VAL_STACKS_DIR=/path/to/stacks/val \\"
    echo "  bash scripts/run_pixel_flow_train.sh"
    exit 0
fi

echo ""
echo "================================================================"
echo " Phase 2: Real sky fine-tune  (${P2_STEPS} steps)"
echo " Corpus:  ${CORPUS_STACKS_DIR}"
echo " Output:  ${OUT_DIR}/phase2/"
echo "================================================================"

mkdir -p "${OUT_DIR}/phase2"

pixi run -e gpu python scripts/train_pixel_flow.py \
    --out_dir        "${OUT_DIR}/phase2" \
    --repo_root      "${REPO_ROOT}" \
    --steps          "${P2_STEPS}" \
    --batch_size     "${P2_BATCH}" \
    --lr             "${P2_LR}" \
    --num_workers    "${P2_WORKERS}" \
    --stacks_dir     "${CORPUS_STACKS_DIR}" \
    --val_stacks_dir "${VAL_STACKS_DIR}" \
    --sky_weight_floor  "${SKY_WEIGHT_FLOOR}" \
    --sparsity_weight   "${SPARSITY_WEIGHT}" \
    --resume         "${OUT_DIR}/phase1/best.pt" \
    --log_every      100 \
    --val_every      500 \
    --checkpoint_every 2000 \
  2>&1 | tee "${OUT_DIR}/phase2/train.log"

echo ""
echo "Phase 2 complete. Best checkpoint: ${OUT_DIR}/phase2/best.pt"
