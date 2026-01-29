#!/usr/bin/env bash
set -euo pipefail

BEAM_ONLY=false

usage() {
  echo "Usage: $0 [--beam-only] [OUT_BASE_DIR]"
  echo "  --beam-only   Run only beam training types (ND beam / NOvA beam / etc.)"
}

# Parse optional flags
while (( $# > 0 )); do
  case "$1" in
    --beam-only)
      BEAM_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# Driver script:
# - Runs ./run_eval_all.sh over multiple users + multiple training types
# - Keeps per-type aggregated outputs in dedicated directories:
#   DUNEAtmoNat_FullEval, DUNEAtmoFlat_FullEval, DUNEOnAxisNDBeamNat_FullEval,
#   DUNEBeamFlat_FullEval, NOvABeamND_FullEval

USERS=(cborden jbarrow rrichi)

# Base "users" directory
BASE_USERS_DIR="/exp/dune/data/users"

# Where to store aggregated outputs (directories named *_FullEval).
# Default: current working directory. You can override by passing an argument.
OUT_BASE_DIR="${1:-$PWD}"

# Path to the evaluator script (must be runnable)
# Resolve directory that this driver script lives in (robust even if called from elsewhere)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use absolute path so it still works after pushd into output dirs
EVAL_SCRIPT="${SCRIPT_DIR}/run_eval_all.sh"

if [[ ! -x "$EVAL_SCRIPT" ]]; then
  echo "ERROR: run_eval_all.sh not found or not executable at: $EVAL_SCRIPT" >&2
  echo "Fix: put run_eval_all.sh next to run_full_evals_all_users.sh, or edit EVAL_SCRIPT." >&2
  exit 1
fi

# Training types:
# Each entry is: OUTDIR_NAME|RELATIVE_PATH_UNDER_USER_DIR
# (RELATIVE path starts at: /exp/dune/data/users/<USER>/ )
# --------------------------
# 1) ORIGINAL training dirs (per-user; rel_path is RELATIVE under /exp/dune/data/users/<USER>/)
# --------------------------
TRAINING_TYPES_ORIG=(
  "DUNEAtmoNat_FullEval|MLProject/Training_Samples/Atmospherics_DUNE_Like/Natural_Spectra"
  "DUNEAtmoFlat_FullEval|MLProject/Training_Samples/Atmospherics_DUNE_Like/Flat_Spectra"
  "DUNEOnAxisNDBeamNat_FullEval|MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisND"
  "DUNEBeamFlat_FullEval|MLProject/Training_Samples/Beam_Like/Flat_Spectra/Ar40"
  "NOvABeamND_FullEval|MLProject/Training_Samples/Beam_Like/Natural_Spectra/NOvAND"
)

# --------------------------
# 2) SINGLE-VAR combined CSV dirs (absolute paths; run once, no users)
# --------------------------
COMBINED_BASE="/exp/dune/data/users/cborden/eval_model/single_var_models_combined"

TRAINING_TYPES_SINGLEVAR=(
  "DUNEAtmoNat_FullEval|${COMBINED_BASE}/DUNEAtmoNat"
  "DUNEAtmoFlat_FullEval|${COMBINED_BASE}/DUNEAtmoFlat"
  "DUNEOnAxisNDBeamNat_FullEval|${COMBINED_BASE}/DUNEOnAxisNDBeamNat"
  "DUNEBeamFlat_FullEval|${COMBINED_BASE}/DUNEBeamFlat"
  "NOvABeamND_FullEval|${COMBINED_BASE}/NOvABeamND"
)

filter_beam_only() {
  local -n in_arr=$1
  local -a out=()
  for entry in "${in_arr[@]}"; do
    outdir_name="${entry%%|*}"
    case "$outdir_name" in
      *Beam*|*NOvABeam*|*OnAxisNDBeam*)
        out+=( "$entry" )
        ;;
    esac
  done
  in_arr=( "${out[@]}" )
}

if [[ "$BEAM_ONLY" == true ]]; then
  filter_beam_only TRAINING_TYPES_ORIG
  filter_beam_only TRAINING_TYPES_SINGLEVAR
fi

is_beam_type() {
  local outdir_name="$1"
  case "$outdir_name" in
    *Beam*|*NOvABeam*|*OnAxisNDBeam*)
      return 0 ;;  # yes, beam
    *)
      return 1 ;;  # not beam
  esac
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] OUT_BASE_DIR = $OUT_BASE_DIR"
echo "[$(timestamp)] USERS        = ${USERS[*]}"
echo "[$(timestamp)] EVAL_SCRIPT   = $EVAL_SCRIPT"
echo

mkdir -p "$OUT_BASE_DIR"

# Run one training type across all users, aggregating into a single output directory.
run_type_across_users() {
  local outdir_name="$1"
  local rel_path="$2"

  local beam_arg=()
  if is_beam_type "$outdir_name"; then
    beam_arg=(--beam)
  fi


  local outdir="${OUT_BASE_DIR%/}/$outdir_name"
  mkdir -p "$outdir"

  local logfile="${outdir}/full_eval.log"
  {
    echo "============================================================"
    echo "[$(timestamp)] Starting type: $outdir_name"
    echo "[$(timestamp)] Relative path: $rel_path"
    echo "[$(timestamp)] Output dir:    $outdir"
    echo "============================================================"
  } | tee -a "$logfile"

  # IMPORTANT:
  # run_eval_all.sh updates combined_output.root and ellipse_fraction.csv
  # when run from the same directory. So we run it *inside* $outdir.
  pushd "$outdir" >/dev/null
  echo "[$(timestamp)] [INFO] CWD for aggregation: $(pwd)" | tee -a "$logfile"
  echo "[$(timestamp)] [INFO] Using EVAL_SCRIPT: $EVAL_SCRIPT" | tee -a "$logfile"

  # For combined CSV directories, rel_path is an absolute path. Run once, no users.
  if [[ "$rel_path" = /* ]]; then
    echo "[$(timestamp)] [RUN ] $EVAL_SCRIPT ${beam_arg[*]} \"$rel_path\"" | tee -a "$logfile"

    { time "$EVAL_SCRIPT" "${beam_arg[@]}" "$rel_path"; } >>"$logfile" 2>&1 || {
      echo "[$(timestamp)] [ERROR] run_eval_all.sh failed for $rel_path (see $logfile)" | tee -a "$logfile"
    }

    popd >/dev/null
    echo "[$(timestamp)] Finished type: $outdir_name" | tee -a "$logfile"
    echo | tee -a "$logfile"
    return 0
  fi

  for user in "${USERS[@]}"; do
    local basepath="${BASE_USERS_DIR%/}/${user}/${rel_path}"

    if [[ ! -d "$basepath" ]]; then
      echo "[$(timestamp)] [SKIP] Missing: $basepath" | tee -a "$logfile"
      continue
    fi

    echo "[$(timestamp)] [RUN ] User=$user Path=$basepath" | tee -a "$logfile"

    # Run eval; append stdout/stderr to the per-type log
    # Use `time` to see roughly how long each user takes.
    echo "[$(timestamp)] [RUN ] $EVAL_SCRIPT ${beam_arg[*]} \"$basepath\"" | tee -a "$logfile"

    set +e
    { time "$EVAL_SCRIPT" "${beam_arg[@]}" "$basepath"; } >>"$logfile" 2>&1
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
    echo "[$(timestamp)] [ERROR] run_eval_all.sh failed for user=$user (rc=$rc). See: $logfile" | tee -a "$logfile"
    # Choose one behavior:
    # 1) continue to next user:
    continue
    # 2) OR hard fail instead (comment out continue and uncomment next line):
    # exit $rc
    fi


    # Optional: stamp outputs to help debugging / provenance
    if [[ -f combined_output.root ]]; then
      echo "[$(timestamp)] [INFO] combined_output.root updated (user=$user)" | tee -a "$logfile"
    fi
    if [[ -f ellipse_fraction.csv ]]; then
      echo "[$(timestamp)] [INFO] ellipse_fraction.csv updated (user=$user)" | tee -a "$logfile"
    fi
  done

  popd >/dev/null

  echo "[$(timestamp)] Finished type: $outdir_name" | tee -a "$logfile"
  echo | tee -a "$logfile"
}

# --------------------------
# Run ORIGINAL training dirs first (per-user)
# --------------------------
for entry in "${TRAINING_TYPES_ORIG[@]}"; do
  IFS="|" read -r outdir_name rel_path <<<"$entry"
  run_type_across_users "$outdir_name" "$rel_path"
done

# --------------------------
# Run SINGLE-VAR combined dirs second (absolute paths)
# --------------------------
for entry in "${TRAINING_TYPES_SINGLEVAR[@]}"; do
  IFS="|" read -r outdir_name rel_path <<<"$entry"
  run_type_across_users "$outdir_name" "$rel_path"
done

echo "[$(timestamp)] All full evals complete."
echo "Outputs created/updated under: $OUT_BASE_DIR"
