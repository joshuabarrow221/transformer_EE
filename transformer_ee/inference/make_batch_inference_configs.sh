#!/usr/bin/env bash
set -euo pipefail

# This wrapper assumes the 6 input text files are in the current directory.
# Edit the arrays below if your filenames/locations differ.

ATM_FILES=(
  "Train_Atmospheric_Flat_Models_rrichi.txt"
  "Train_Atmospheric_Flat_Models_jbarrow.txt"
  "Train_Atmospheric_Flat_Models_cborden.txt"
)

BEAM_FILES=(
  "Train_DUNEBeam_Flat_Models_rrich.txt"
  "Train_DUNEBeam_Flat_Models_jbarrow.txt"
  "Train_DUNEBeam_Flat_Models_cborden.txt"
)

OUTDIR="."

python3 generate_batch_inference_configs.py \
  --outdir "${OUTDIR}" \
  --atm-files "${ATM_FILES[@]}" \
  --beam-files "${BEAM_FILES[@]}"
