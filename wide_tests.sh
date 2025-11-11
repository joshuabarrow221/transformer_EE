#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config — edit these as needed
# -----------------------------

# Sweep values
NHEADS=(16 32)
DROPOUTS=(0.1 0.15 0.2 0.3)

# Shared training hyperparameters
DMODEL=256
NUM_LAYERS=6
OPTIMIZER="adamw"
LR="3e-4"
WEIGHT_DECAY="0.01"
EPOCHS=50

# Paths (base pieces that don't change across runs)
BASE_SAVE_PREFIX="/exp/dune/data/users/cborden/save_genie-dmodel_TEST/model"
BASE_NAME="GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MSE_E_Px_Py_Pz_EID_adamw_dropouts"
WANDB_DIR="/exp/dune/data/users/cborden/save_genie-dmodel_TEST"

# Python entrypoint
PYTHON_BIN="python3"
ENTRYPOINT="train_wide.py"

# Optional: extra args
EXTRA_ARGS=""

# -----------------------------
# Sweep
# -----------------------------

echo "[INFO] Starting sweep over nhead x dropout (fixed d_model=${DMODEL})"

for h in "${NHEADS[@]}"; do

  # Validate divisibility: d_model must be divisible by nhead
  if (( DMODEL % h != 0 )); then
    echo "[WARN] Skipping nhead=${h} for d_model=${DMODEL} (not divisible)"
    continue
  fi

  for dr in "${DROPOUTS[@]}"; do

    # Safe string for dropout in names (e.g., 0.1 -> 0p1)
    DR_STR="${dr/./p}"

    # Compose identifiers/paths
    SUFFIX="dmodel${DMODEL}_nh${h}_do${DR_STR}"
    WANDB_ID="${BASE_NAME}_${SUFFIX}"
    SAVE_PATH="${BASE_SAVE_PREFIX}/${WANDB_ID}"

    # Ensure save directory exists
    mkdir -p "${SAVE_PATH}"

    # Log what we're about to run
    echo "------------------------------------------------------------"
    echo "[RUN ] d_model=${DMODEL}, nhead=${h}, dropout=${dr}"
    echo "[ID  ] ${WANDB_ID}"
    echo "[SAVE] ${SAVE_PATH}"
    echo "------------------------------------------------------------"

    # Launch training
    ${PYTHON_BIN} "${ENTRYPOINT}" \
      --d-model "${DMODEL}" \
      --nhead "${h}" \
      --num-layers "${NUM_LAYERS}" \
      --optimizer "${OPTIMIZER}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --dropout "${dr}" \
      --epochs "${EPOCHS}" \
      --wandb-id "${WANDB_ID}" \
      --save-path "${SAVE_PATH}" \
      --wandb-dir "${WANDB_DIR}" \
      ${EXTRA_ARGS}

    echo "[DONE] ${WANDB_ID}"
    echo
  done
done

echo "[ALL DONE] Sweep complete."