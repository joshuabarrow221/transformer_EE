#!/usr/bin/env bash
set -euo pipefail

# build_combined_singlevar_inference_results.sh
#
# Build pseudo-multivariate combined CSVs from single-variable inference outputs.
#
# It scans inference result directories named like:
#   ..._NpNpi_<VAR>_<LOSS>_Topology_<SUFFIX>/
# and locates the single CSV file in each directory, then builds the same
# combinations used by the training-time single_var combiner:
#   - 16 x (Energy loss, Momentum loss)
#   - 16 x (Energy loss, Angular loss)
#
# Optional: run graph_eval/run_eval_all.sh on each output group directory so
# combined_output.root and ellipse_fraction.csv are updated/appended in a common
# eval output directory.

INFER_BASE="${1:-}"
OUT_ROOT="${2:-}"
MACRO_DIR="${3:-}"
EVAL_DIR="${4:-}"

if [[ -z "$INFER_BASE" || -z "$OUT_ROOT" || -z "$MACRO_DIR" ]]; then
  echo "Usage: $0 INFER_BASE OUT_ROOT MACRO_DIR [COMMON_EVAL_OUTPUT_DIR]"
  exit 1
fi

if [[ ! -d "$INFER_BASE" ]]; then
  echo "ERROR: INFER_BASE not a directory: $INFER_BASE" >&2
  exit 1
fi

MACRO_PATH="${MACRO_DIR%/}/merge_result_csv_two.C"
if [[ ! -f "$MACRO_PATH" ]]; then
  echo "ERROR: macro not found: $MACRO_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$(cd -- "${SCRIPT_DIR}/.." && pwd)/run_eval_all.sh"

LOSSES=(MAE MSE MAPE MACE)

ts(){ date +"%Y-%m-%d %H:%M:%S"; }

canon_var() {
  local raw="$1"
  case "$raw" in
    Energy|Nu_Energy) echo "Energy" ;;
    Mom_X|Nu_Mom_X|Nu_MomX|MomX) echo "Mom_X" ;;
    Mom_Y|Nu_Mom_Y|Nu_MomY|MomY) echo "Mom_Y" ;;
    Mom_Z|Nu_Mom_Z|Nu_MomZ|MomZ) echo "Mom_Z" ;;
    Theta|Nu_Theta) echo "Theta" ;;
    Nu_CosTheta|CosTheta|Cos_Theta) echo "CosTheta" ;;
    Phi|Nu_Phi) echo "Phi" ;;
    *) echo "" ;;
  esac
}

root_merge_two() {
  local in_combined="$1"
  local in_new="$2"
  local out_combined="$3"
  local var_token="$4"
  local tag="$5"
  local warn_file="$6"

  root -l -b -q "${MACRO_PATH}(\"${in_combined}\",\"${in_new}\",\"${out_combined}\",\"${var_token}\",\"${tag}\",\"${warn_file}\")"
}

sanitize_name() {
  local x="$1"
  x="${x//\//__}"
  x="${x// /_}"
  x="${x//:/_}"
  x="${x//[^A-Za-z0-9_.-]/_}"
  echo "$x"
}

declare -A CSV_BY_KEY MTIME_BY_KEY

echo "[$(ts)] Scanning inference directories under: $INFER_BASE"

while IFS= read -r d; do
  bn="$(basename "$d")"

  if [[ ! "$bn" =~ ^(.*)_NpNpi_(.*)_(MAE|MSE|MAPE|MACE)_Topology(_.*)?$ ]]; then
    continue
  fi

  group_prefix="${BASH_REMATCH[1]}_NpNpi"
  raw_var="${BASH_REMATCH[2]}"
  loss="${BASH_REMATCH[3]}"

  var="$(canon_var "$raw_var")"
  if [[ -z "$var" ]]; then
    continue
  fi

  csv="$(find "$d" -maxdepth 1 -type f -name "*.csv" -print | sort | tail -n 1)"
  if [[ -z "$csv" || ! -f "$csv" ]]; then
    continue
  fi

  mt="$(stat -c %Y "$csv" 2>/dev/null || echo 0)"
  key="${group_prefix}|${var}|${loss}"
  old_mt="${MTIME_BY_KEY[$key]:-0}"

  if (( mt >= old_mt )); then
    MTIME_BY_KEY[$key]="$mt"
    CSV_BY_KEY[$key]="$csv"
  fi
done < <(find "$INFER_BASE" -type d -print)

if (( ${#CSV_BY_KEY[@]} == 0 )); then
  echo "[$(ts)] ERROR: No matching single-variable inference directories found."
  exit 2
fi

mkdir -p "$OUT_ROOT"

# unique group list
mapfile -t GROUPS < <(
  for k in "${!CSV_BY_KEY[@]}"; do
    echo "${k%%|*}"
  done | sort -u
)

echo "[$(ts)] Found ${#GROUPS[@]} inference group(s)."

for group in "${GROUPS[@]}"; do
  out_dir="${OUT_ROOT%/}/$(sanitize_name "$group")"
  mkdir -p "$out_dir"

  log="${out_dir}/build_combined_inference.log"
  warn_file="${out_dir}/WARNINGS.txt"
  : > "$warn_file"

  {
    echo "============================================================"
    echo "[$(ts)] GROUP    = $group"
    echo "[$(ts)] OUT_DIR  = $out_dir"
    echo "============================================================"
  } | tee -a "$log"

  declare -A CSV_E CSV_MX CSV_MY CSV_MZ CSV_ANG ANG_KIND

  for loss in "${LOSSES[@]}"; do
    CSV_E["$loss"]="${CSV_BY_KEY[${group}|Energy|${loss}]:-}"
    CSV_MX["$loss"]="${CSV_BY_KEY[${group}|Mom_X|${loss}]:-}"
    CSV_MY["$loss"]="${CSV_BY_KEY[${group}|Mom_Y|${loss}]:-}"
    CSV_MZ["$loss"]="${CSV_BY_KEY[${group}|Mom_Z|${loss}]:-}"

    if [[ -n "${CSV_BY_KEY[${group}|Theta|${loss}]:-}" ]]; then
      CSV_ANG["$loss"]="${CSV_BY_KEY[${group}|Theta|${loss}]}"
      ANG_KIND["$loss"]="Theta"
    elif [[ -n "${CSV_BY_KEY[${group}|CosTheta|${loss}]:-}" ]]; then
      CSV_ANG["$loss"]="${CSV_BY_KEY[${group}|CosTheta|${loss}]}"
      ANG_KIND["$loss"]="CosTheta"
    elif [[ -n "${CSV_BY_KEY[${group}|Phi|${loss}]:-}" ]]; then
      CSV_ANG["$loss"]="${CSV_BY_KEY[${group}|Phi|${loss}]}"
      ANG_KIND["$loss"]="Phi"
    else
      CSV_ANG["$loss"]=""
      ANG_KIND["$loss"]=""
    fi
  done

  # Energy + Momentum combos
  for eLoss in "${LOSSES[@]}"; do
    for pLoss in "${LOSSES[@]}"; do
      if [[ -z "${CSV_E[$eLoss]}" || -z "${CSV_MX[$pLoss]}" || -z "${CSV_MY[$pLoss]}" || -z "${CSV_MZ[$pLoss]}" ]]; then
        echo "[$(ts)] [SKIP] E-${eLoss} + P-${pLoss}: missing one or more required CSVs" | tee -a "$log"
        continue
      fi

      out="${out_dir}/combined_result__E-${eLoss}__P-${pLoss}.csv"
      echo "[$(ts)] [BUILD] $(basename "$out")" | tee -a "$log"

      root_merge_two ""   "${CSV_E[$eLoss]}"  "$out" "Energy" "Energy_${eLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "${CSV_MX[$pLoss]}" "$out" "Mom_X" "Mom_X_${pLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "${CSV_MY[$pLoss]}" "$out" "Mom_Y" "Mom_Y_${pLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "${CSV_MZ[$pLoss]}" "$out" "Mom_Z" "Mom_Z_${pLoss}" "$warn_file" >>"$log" 2>&1
    done
  done

  # Energy + angular combos (Theta, CosTheta, or Phi)
  for eLoss in "${LOSSES[@]}"; do
    for aLoss in "${LOSSES[@]}"; do
      ang_csv="${CSV_ANG[$aLoss]}"
      ang_kind="${ANG_KIND[$aLoss]}"

      if [[ -z "${CSV_E[$eLoss]}" || -z "$ang_csv" || -z "$ang_kind" ]]; then
        echo "[$(ts)] [SKIP] E-${eLoss} + Th-${aLoss}: missing Energy/angular CSV" | tee -a "$log"
        continue
      fi

      out="${out_dir}/combined_result__E-${eLoss}__Th-${aLoss}.csv"
      echo "[$(ts)] [BUILD] $(basename "$out") [angular=${ang_kind}]" | tee -a "$log"

      root_merge_two ""   "${CSV_E[$eLoss]}" "$out" "Energy" "Energy_${eLoss}" "$warn_file" >>"$log" 2>&1
      root_merge_two "$out" "$ang_csv" "$out" "$ang_kind" "${ang_kind}_${aLoss}" "$warn_file" >>"$log" 2>&1
    done
  done

  if [[ -n "$EVAL_DIR" ]]; then
    mkdir -p "$EVAL_DIR"
    if [[ ! -f "$EVAL_SCRIPT" ]]; then
      echo "[$(ts)] [WARN] Eval requested but run_eval_all.sh not found at $EVAL_SCRIPT" | tee -a "$log"
    else
      echo "[$(ts)] [EVAL] Running eval over $out_dir (cwd=$EVAL_DIR)" | tee -a "$log"
      (
        cd "$EVAL_DIR"
        bash "$EVAL_SCRIPT" "$out_dir"
      ) >>"$log" 2>&1
    fi
  fi

  if [[ -s "$warn_file" ]]; then
    echo "[$(ts)] [WARN] Warnings were emitted; see $warn_file" | tee -a "$log"
  fi

  unset CSV_E CSV_MX CSV_MY CSV_MZ CSV_ANG ANG_KIND

done

echo "[$(ts)] Done. Combined inference outputs under: $OUT_ROOT"
if [[ -n "$EVAL_DIR" ]]; then
  echo "[$(ts)] Eval outputs appended in: $EVAL_DIR (ellipse_fraction.csv, combined_output.root)"
fi
