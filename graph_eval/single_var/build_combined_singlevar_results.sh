#!/usr/bin/env bash
set -euo pipefail

# build_combined_singlevar_results.sh
#
# Bash driver that:
#  - navigates type dirs under single_var_models/
#  - finds newest model_*/result.csv for each (var,loss)
#  - loops over all 32 desired combinations
#  - calls ROOT helper macro merge_result_csv_two.C to iteratively build combined CSVs
#
# Usage:
#   ./build_combined_singlevar_results.sh SINGLE_VAR_BASE OUT_ROOT MACRO_DIR
#
# Example:
#   ./build_combined_singlevar_results.sh \
#     /exp/dune/data/users/cborden/eval_model/single_var_models \
#     /exp/dune/data/users/cborden/eval_model/single_var_models_combined \
#     /exp/dune/data/users/cborden/eval_model

SINGLE_VAR_BASE="${1:-}"
OUT_ROOT="${2:-}"
MACRO_DIR="${3:-}"

if [[ -z "$SINGLE_VAR_BASE" || -z "$OUT_ROOT" || -z "$MACRO_DIR" ]]; then
  echo "Usage: $0 SINGLE_VAR_BASE OUT_ROOT MACRO_DIR"
  exit 1
fi

if [[ ! -d "$SINGLE_VAR_BASE" ]]; then
  echo "ERROR: SINGLE_VAR_BASE not a directory: $SINGLE_VAR_BASE" >&2
  exit 1
fi

MACRO_PATH="${MACRO_DIR%/}/merge_result_csv_two.C"
if [[ ! -f "$MACRO_PATH" ]]; then
  echo "ERROR: macro not found: $MACRO_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

TYPES=(
  "DUNEAtmoNat"
  "DUNEAtmoFlat"
  "DUNEOnAxisNDBeamNat"
  "DUNEBeamFlat"
  "NOvABeamND"
)

LOSSES=(MAE MSE MAPE MACE)

ts(){ date +"%Y-%m-%d %H:%M:%S"; }

# Find newest model_*/result.csv under the training directory (model_*/result.csv).
newest_result_csv_in_training_dir() {
  local train_dir="$1"
  # shellcheck disable=SC2012
  local newest
  newest="$(ls -1dt "$train_dir"/model_*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "$newest" ]]; then
    return 1
  fi
  if [[ ! -f "$newest/result.csv" ]]; then
    return 1
  fi
  echo "$newest/result.csv"
}

# Find training directory name matching "NpNpi_<VAR>_<LOSS>_Topology"
find_training_dir_for_var_loss() {
  local base_dir="$1"
  local var="$2"
  local loss="$3"
  local needle="NpNpi_${var}_${loss}_Topology"

  # Look only at immediate subdirectories of base_dir
  local d
  for d in "$base_dir"/*/; do
    [[ -d "$d" ]] || continue
    local bn
    bn="$(basename "${d%/}")"
    if [[ "$bn" == *"$needle"* ]]; then
      echo "${d%/}"
      return 0
    fi
  done
  return 1
}

# Resolve the result.csv for (var, loss) under base_dir
get_result_csv_for_var_loss() {
  local base_dir="$1"
  local var="$2"
  local loss="$3"

  local train_dir
  if ! train_dir="$(find_training_dir_for_var_loss "$base_dir" "$var" "$loss")"; then
    echo "ERROR: could not find training dir in $base_dir for ${var}_${loss}" >&2
    return 2
  fi

  local csv
  if ! csv="$(newest_result_csv_in_training_dir "$train_dir")"; then
    echo "ERROR: could not find newest model_*/result.csv under $train_dir" >&2
    return 3
  fi

  echo "$csv"
}

# Call ROOT macro helper
root_merge_two() {
  local in_combined="$1"   # "" for bootstrap
  local in_new="$2"
  local out_combined="$3"
  local var_token="$4"
  local tag="$5"
  local warn_file="$6"

  root -l -b -q "${MACRO_PATH}(\"${in_combined}\",\"${in_new}\",\"${out_combined}\",\"${var_token}\",\"${tag}\",\"${warn_file}\")"
}

for type in "${TYPES[@]}"; do
  base_dir="${SINGLE_VAR_BASE%/}/${type}"
  if [[ ! -d "$base_dir" ]]; then
    echo "[$(ts)] [SKIP] Missing type dir: $base_dir"
    continue
  fi

  out_dir="${OUT_ROOT%/}/${type}"
  mkdir -p "$out_dir"
  log="${out_dir}/build_combined.log"

  warn_file="${out_dir}/WARNINGS.txt"
  : > "$warn_file"   # truncate

  {
    echo "============================================================"
    echo "[$(ts)] TYPE     = $type"
    echo "[$(ts)] BASE_DIR = $base_dir"
    echo "[$(ts)] OUT_DIR  = $out_dir"
    echo "============================================================"
  } | tee -a "$log"

  # Pre-resolve all needed CSVs into associative arrays for speed / determinism
  declare -A CSV_E CSV_MX CSV_MY CSV_MZ CSV_TH

  for loss in "${LOSSES[@]}"; do
    CSV_E["$loss"]="$(get_result_csv_for_var_loss "$base_dir" "Energy" "$loss")"
    CSV_TH["$loss"]="$(get_result_csv_for_var_loss "$base_dir" "Theta"  "$loss")"
    CSV_MX["$loss"]="$(get_result_csv_for_var_loss "$base_dir" "Mom_X"  "$loss")"
    CSV_MY["$loss"]="$(get_result_csv_for_var_loss "$base_dir" "Mom_Y"  "$loss")"
    CSV_MZ["$loss"]="$(get_result_csv_for_var_loss "$base_dir" "Mom_Z"  "$loss")"
  done

  # ---- 16 Energy + Momentum combos ----
  for eLoss in "${LOSSES[@]}"; do
    for pLoss in "${LOSSES[@]}"; do
      out="${out_dir}/combined_result__E-${eLoss}__P-${pLoss}.csv"

      echo "[$(ts)] [BUILD] $(basename "$out")" | tee -a "$log"

      # Bootstrap with Energy
      root_merge_two ""   "${CSV_E[$eLoss]}"  "$out" "Energy" "Energy_${eLoss}" "$warn_file" >>"$log" 2>&1

      # Append Mom_X / Mom_Y / Mom_Z (same pLoss)
      root_merge_two "$out" "${CSV_MX[$pLoss]}" "$out" "Mom_X" "Mom_X_${pLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "${CSV_MY[$pLoss]}" "$out" "Mom_Y" "Mom_Y_${pLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "${CSV_MZ[$pLoss]}" "$out" "Mom_Z" "Mom_Z_${pLoss}" "$warn_file" >>"$log" 2>&1
    done
  done

  # ---- 16 Energy + Theta combos ----
  for eLoss in "${LOSSES[@]}"; do
    for tLoss in "${LOSSES[@]}"; do
      out="${out_dir}/combined_result__E-${eLoss}__Th-${tLoss}.csv"

      echo "[$(ts)] [BUILD] $(basename "$out")" | tee -a "$log"

      # Bootstrap with Energy
      root_merge_two ""   "${CSV_E[$eLoss]}"  "$out" "Energy" "Energy_${eLoss}" "$warn_file" >>"$log" 2>&1

      # Append Theta
      root_merge_two "$out" "${CSV_TH[$tLoss]}" "$out" "Theta"  "Theta_${tLoss}" "$warn_file" >>"$log" 2>&1
    done
  done

  echo "[$(ts)] Done type=$type" | tee -a "$log"
  echo | tee -a "$log"

  if [[ -s "$warn_file" ]]; then
    echo "[$(ts)] WARNING SUMMARY for type=$type (see $warn_file):" | tee -a "$log"
    tail -n 20 "$warn_file" | sed 's/^/  /' | tee -a "$log"
  else
    echo "[$(ts)] No warnings for type=$type" | tee -a "$log"
  fi

  # cleanup assoc arrays (bash requires unset to avoid cross-type contamination)
  unset CSV_E CSV_MX CSV_MY CSV_MZ CSV_TH
done

echo "[$(ts)] All done. Outputs under: $OUT_ROOT/<TYPE>/combined_result__*.csv"