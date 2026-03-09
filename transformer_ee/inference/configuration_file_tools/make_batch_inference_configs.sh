#!/usr/bin/env bash
set -euo pipefail

# This wrapper assumes the expected input text files are in the current directory.
# Edit the arrays below if your filenames/locations differ.

ATM_FILES=(
  "Train_Atmospheric_Flat_Models_rrichi.txt"
  "Train_Atmospheric_Flat_Models_jbarrow.txt"
  "Train_Atmospheric_Flat_Models_cborden.txt"
)

DUNE_BEAM_FLAT_FILES=(
  "Train_DUNEBeam_Flat_Models_rrichi.txt"
  "Train_DUNEBeam_Flat_Models_jbarrow.txt"
  "Train_DUNEBeam_Flat_Models_cborden.txt"
)

DUNE_BEAM_NAT_FILES=(
  "Train_DUNEBeam_Nat_Models_rrichi.txt"
  "Train_DUNEBeam_Nat_Models_jbarrow.txt"
  "Train_DUNEBeam_Nat_Models_cborden.txt"
)

NOVA_BEAM_NAT_FILES=(
  "Train_NOvABeam_Nat_Models_rrichi.txt"
  "Train_NOvABeam_Nat_Models_jbarrow.txt"
  "Train_NOvABeam_Nat_Models_cborden.txt"
)

BEAM_FILES=(
  "${DUNE_BEAM_FLAT_FILES[@]}"
  "${DUNE_BEAM_NAT_FILES[@]}"
  "${NOVA_BEAM_NAT_FILES[@]}"
)

OUTDIR="."

# Override model search roots with either:
#  1) MODEL_SEARCH_ROOTS as a bash array in this script, or
#  2) MODEL_SEARCH_ROOTS env var as a colon-separated list.
#
# Example:
#   MODEL_SEARCH_ROOTS="/path/to/a:/path/to/b" ./make_batch_inference_configs.sh
MODEL_SEARCH_ROOTS=(
  "/exp/dune/data/users/cborden/MLProject/Training_Samples"
  "/exp/dune/data/users/rrichi/MLProject/Training_Samples"
  "/exp/dune/data/users/jbarrow/MLProject/Training_Samples"
)

if [[ -n "${MODEL_SEARCH_ROOTS:-}" && "${MODEL_SEARCH_ROOTS}" == *":"* ]]; then
  IFS=':' read -r -a MODEL_SEARCH_ROOTS <<< "${MODEL_SEARCH_ROOTS}"
fi

python3 generate_batch_inference_configs.py \
  --outdir "${OUTDIR}" \
  --atm-files "${ATM_FILES[@]}" \
  --beam-files "${BEAM_FILES[@]}" \
  --model-search-roots "${MODEL_SEARCH_ROOTS[@]}"
