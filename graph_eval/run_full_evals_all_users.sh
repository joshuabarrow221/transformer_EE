#!/usr/bin/env bash
set -euo pipefail

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

COMBINED_BASE="/exp/dune/data/users/cborden/eval_model/single_var_models_combined"

# Training types:
# Each entry is: OUTDIR_NAME|RELATIVE_PATH_UNDER_USER_DIR
# (RELATIVE path starts at: /exp/dune/data/users/<USER>/ )
TRAINING_TYPES=(
  "DUNEAtmoNat_FullEval|${COMBINED_BASE}/DUNEAtmoNat"
  "DUNEAtmoFlat_FullEval|${COMBINED_BASE}/DUNEAtmoFlat"
  "DUNEOnAxisNDBeamNat_FullEval|${COMBINED_BASE}/DUNEOnAxisNDBeamNat"
  "DUNEBeamFlat_FullEval|${COMBINED_BASE}/DUNEBeamFlat"
  "NOvABeamND_FullEval|${COMBINED_BASE}/NOvABeamND"
)


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
    echo "[$(timestamp)] [RUN ] Combined CSV dir: $rel_path" | tee -a "$logfile"

    { time "$EVAL_SCRIPT" "$rel_path"; } >>"$logfile" 2>&1 || {
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
    echo "[$(timestamp)] [RUN ] $EVAL_SCRIPT \"$basepath\"" | tee -a "$logfile"

    set +e
    { time "$EVAL_SCRIPT" "$basepath"; } >>"$logfile" 2>&1
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

# Main loop over training types
for entry in "${TRAINING_TYPES[@]}"; do
  IFS="|" read -r outdir_name rel_path <<<"$entry"
  run_type_across_users "$outdir_name" "$rel_path"
done

echo "[$(timestamp)] All full evals complete."
echo "Outputs created/updated under: $OUT_BASE_DIR"
