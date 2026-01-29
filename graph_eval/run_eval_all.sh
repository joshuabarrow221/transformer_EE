#!/usr/bin/env bash
set -euo pipefail

# Directory where this script lives (so we can find eval_model.C reliably)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACRO="${SCRIPT_DIR}/eval_model.C"

if [[ ! -f "$MACRO" ]]; then
    echo "Error: eval_model.C not found at: $MACRO" >&2
    exit 1
fi

# Usage: ./run_eval_all.sh /path/to/base_dir
BEAM_MODE=false
BASE_DIR=""

usage() {
    echo "Usage: $0 [--beam|-b] BASE_DIR"
}

# Parse optional flag(s)
while (( $# > 0 )); do
    case "$1" in
        --beam|-b)
            BEAM_MODE=true
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
            echo "Error: unknown option '$1'" >&2
            usage
            exit 1
            ;;
        *)
            BASE_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$BASE_DIR" ]]; then
    usage
    exit 1
fi


if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: BASE_DIR '$BASE_DIR' is not a directory."
    exit 1
fi

# ROOT macro call (second argument is beam_mode bool)
ROOT_CALL_BOOL=$([[ "$BEAM_MODE" == true ]] && echo "true" || echo "false")

# Where the individual single-variable training dirs live (used to sum runtimes for combined_result__*.csv)
SINGLE_VAR_MODELS_BASE="/exp/dune/data/users/cborden/eval_model/single_var_models"

# Extract W&B runtime (hours) for a given training directory.
# Expects W&B runs under:
#   <train_dir>/wandb/wandb/run-*/files/wandb-summary.json
# Prints a numeric value (hours) or -1 if not found.
get_wandb_runtime_hours() {
    local train_dir="$1"
    local wandb_root="${train_dir%/}/wandb/wandb"

    # Default: unknown
    local out="-1"

    if [[ ! -d "$wandb_root" ]]; then
        echo "$out"
        return 0
    fi

    out=$(python3 - "$wandb_root" <<'PY'
import glob, json, math, os, sys

wandb_root = sys.argv[1]
runs = sorted(glob.glob(os.path.join(wandb_root, 'run-*')), key=os.path.getmtime)
if not runs:
    print(-1)
    sys.exit(0)

run_dir = runs[-1]
summ = os.path.join(run_dir, 'files', 'wandb-summary.json')

try:
    with open(summ, 'r') as f:
        d = json.load(f)
    rt = d.get('_runtime', None)
    if rt is None:
        rt = (d.get('_wandb', {}) or {}).get('runtime', None)
    if isinstance(rt, (int, float)) and math.isfinite(rt) and rt > 0:
        print(float(rt) / 3600.0)
    else:
        print(-1)
except Exception:
    print(-1)
PY
)

    echo "$out"
}

# Parse tokens from combined CSV name:
# combined_result__E-MAE__P-MAE.csv -> tokens: E-MAE  P-MAE
parse_combined_tokens() {
    local csv="$1"
    local b
    b="$(basename "$csv")"
    b="${b#combined_result__}"
    b="${b%.csv}"

    # Replace "__" with spaces, then print one token per line
    b="${b//__/ }"
    for t in $b; do
        echo "$t"
    done
}

# Find the best matching single-var training directory for a given token.
# We pick the newest matching directory if multiple match.
find_singlevar_training_dir_for_token() {
    local type="$1"   # e.g. DUNEAtmoNat
    local token="$2"  # e.g. E-MAE
    local root="${SINGLE_VAR_MODELS_BASE%/}/${type}"

    [[ -d "$root" ]] || { echo ""; return 0; }

    local var="${token%%-*}"         # E, P, CosTheta, Px, ...
    local metric="${token#*-}"       # MAE, MSE, MACE, ...

    # Some tokens may not contain "-" (be defensive)
    if [[ "$var" == "$token" ]]; then
        metric=""
    fi

    # Candidate patterns (ordered from most specific to least)
    local pats=()

    case "$var" in
        E)
            # Prefer scalar-style naming, then vector-style
            pats+=("*Nu_Energy_${metric}*")
            pats+=("*Energy_${metric}*")
            pats+=("*Nu_Energy*")
            pats+=("*Energy*")
            ;;

        Th|Theta)
            pats+=("*Nu_Theta_${metric}*")
            pats+=("*Theta_${metric}*")
            pats+=("*Nu_Theta*")
            pats+=("*Theta*")
            ;;

        MomX|Nu_MomX)
            pats+=("*Nu_MomX_${metric}*")
            pats+=("*Mom_X_${metric}*")
            pats+=("*Nu_MomX*")
            pats+=("*Mom_X*")
            ;;

        MomY|Nu_MomY)
            pats+=("*Nu_MomY_${metric}*")
            pats+=("*Mom_Y_${metric}*")
            pats+=("*Nu_MomY*")
            pats+=("*Mom_Y*")
            ;;

        MomZ|Nu_MomZ)
            pats+=("*Nu_MomZ_${metric}*")
            pats+=("*Mom_Z_${metric}*")
            pats+=("*Nu_MomZ*")
            pats+=("*Mom_Z*")
            ;;

        CosTheta|Cos_Theta|cosTheta)
            pats+=("*Nu_CosTheta_${metric}*")
            pats+=("*CosTheta_${metric}*")
            pats+=("*Nu_CosTheta*")
            pats+=("*CosTheta*")
            ;;

        *)
            local t1="${token//-/_}"
            pats+=("*${token}*")
            pats+=("*${t1}*")
            [[ -n "$metric" ]] && pats+=("*${var}*${metric}*")
            ;;
    esac

    local best=""
    local best_mtime=0

    shopt -s nullglob
    for pat in "${pats[@]}"; do
        for d in "$root"/$pat/; do
            [[ -d "$d" ]] || continue
            [[ "$(basename "$d")" == "wandb" ]] && continue
            local mt
            mt=$(stat -c %Y "$d" 2>/dev/null || echo 0)
            if (( mt > best_mtime )); then
                best_mtime=$mt
                best="$d"
            fi
        done
        [[ -n "$best" ]] && break
    done
    shopt -u nullglob

    echo "$best"
}

# For a token like:
#   E-MSE  -> one training dir (Nu_Energy_MSE)
#   Th-MAE -> one training dir (Nu_Theta_MAE)
#   P-MSE  -> three training dirs (Nu_MomX_MSE, Nu_MomY_MSE, Nu_MomZ_MSE)
# Prints one directory per line (may print nothing if not found).
token_to_training_dirs() {
    local type="$1"    # e.g. DUNEAtmoNat
    local token="$2"   # e.g. P-MSE

    local var="${token%%-*}"
    local metric="${token#*-}"
    if [[ "$var" == "$token" ]]; then
        metric=""
    fi

    if [[ "$var" == "P" ]]; then
        # Expand P into MomX/MomY/MomZ with same metric
        local d
        d="$(find_singlevar_training_dir_for_token "$type" "MomX-${metric}")"
        [[ -n "$d" ]] && echo "$d"
        d="$(find_singlevar_training_dir_for_token "$type" "MomY-${metric}")"
        [[ -n "$d" ]] && echo "$d"
        d="$(find_singlevar_training_dir_for_token "$type" "MomZ-${metric}")"
        [[ -n "$d" ]] && echo "$d"
    else
        local d
        d="$(find_singlevar_training_dir_for_token "$type" "$token")"
        [[ -n "$d" ]] && echo "$d"
    fi
}

echo "Scanning base directory: $BASE_DIR"
echo

# Make globs that don't match expand to nothing instead of themselves
shopt -s nullglob

# if this looks like a "combined CSV output" directory, evaluate those CSVs directly ---
combined_csvs=( "$BASE_DIR"/combined_result__*.csv )
if (( ${#combined_csvs[@]} > 0 )); then
    echo "Detected combined CSV directory (found ${#combined_csvs[@]} combined_result__*.csv files)."

    # Infer training type from the combined directory name, e.g.
    # .../single_var_models_combined/DUNEAtmoNat -> DUNEAtmoNat
    TYPE_NAME="$(basename "${BASE_DIR%/}")"
    echo "Inferred training type for runtime summing: $TYPE_NAME"
    echo

    for csv in "${combined_csvs[@]}"; do
        echo "---------------------------------------------"
        echo "Processing combined CSV: $csv"

        total_runtime_hours="0.0"
        any_found=false

        while IFS= read -r tok; do
            [[ -n "$tok" ]] || continue

            # Expand token into 1+ component training dirs (P -> MomX/MomY/MomZ)
            mapfile -t dirs < <(token_to_training_dirs "$TYPE_NAME" "$tok")

            if (( ${#dirs[@]} == 0 )); then
                echo "  WARNING: could not resolve token '$tok' to any training dirs under ${SINGLE_VAR_MODELS_BASE}/${TYPE_NAME}"
                continue
            fi

            for train_dir in "${dirs[@]}"; do
                rh="$(get_wandb_runtime_hours "$train_dir")"
                if [[ "$rh" == "-1" ]]; then
                    echo "  WARNING: runtime not found for '$tok' component (dir=$train_dir)"
                    continue
                fi

                any_found=true
                echo "  Component '$tok' -> $(basename "$train_dir") : runtime_hours=$rh"

                # floating-point sum
                total_runtime_hours="$(python3 - <<PY
import sys
print(float(sys.argv[1]) + float(sys.argv[2]))
PY
"$total_runtime_hours" "$rh")"
            done

        done < <(parse_combined_tokens "$csv")

        if [[ "$any_found" == true ]]; then
            runtime_hours="$total_runtime_hours"
            echo "  Total combined runtime_hours = $runtime_hours"
        else
            runtime_hours="-1"
            echo "  Total combined runtime_hours = not found (leaving blank)"
        fi

        if ! root -l -b -q "${MACRO}(\"$csv\", ${ROOT_CALL_BOOL}, ${runtime_hours})"; then
            echo "  ERROR: ROOT failed for $csv (continuing)"
            continue
        fi
    done

    echo
    echo "Done."
    exit 0
fi

# Loop over all immediate subdirectories (training runs)
for train_dir in "$BASE_DIR"/*/; do
    [[ -d "$train_dir" ]] || continue

    echo "---------------------------------------------"
    echo "Checking training directory: $train_dir"

    # All model_* subdirectories in this training directory
    model_dirs=( "$train_dir"model_*/ )

    if (( ${#model_dirs[@]} == 0 )); then
        echo "  No model_* directories found here."
        continue
    fi

    # If more than one model_* dir, print them all
    if (( ${#model_dirs[@]} > 1 )); then
        echo "  Found ${#model_dirs[@]} model_* directories:"
        for md in "${model_dirs[@]}"; do
            echo "    - $md"
        done
    fi

    # Track the most recent model_* that actually has a result.csv
    latest_dir=""
    latest_time=0

    for md in "${model_dirs[@]}"; do
        result_file="${md}result.csv"

        if [[ -f "$result_file" ]]; then
            # Use directory mtime as proxy for "created most recently"
            mtime=$(stat -c %Y "$md")
            if (( mtime > latest_time )); then
                latest_time=$mtime
                latest_dir="$md"
            fi
        else
            echo "  WARNING: $md does not contain result.csv"
        fi
    done

    if [[ -z "$latest_dir" ]]; then
        echo "  No result.csv found in any model_* directory under $train_dir"
        continue
    fi

    if (( ${#model_dirs[@]} > 1 )); then
        echo "  Using most recent model_* with result.csv: $latest_dir"
    fi

    result_file="${latest_dir}result.csv"
    echo "  Processing result.csv: $result_file"

    # Extract W&B runtime (hours) from the training directory that contains model_*/ and wandb/
    runtime_hours=$(get_wandb_runtime_hours "$train_dir")
    if [[ "$runtime_hours" != "-1" ]]; then
        echo "  W&B runtime (hours): $runtime_hours"
    else
        echo "  W&B runtime (hours): not found"
    fi

    # Run ROOT in batch mode, calling eval_model.C("full/path/to/result.csv")
    if ! root -l -b -q "${MACRO}(\"$result_file\", ${ROOT_CALL_BOOL}, ${runtime_hours})"; then
        echo "  ERROR: ROOT failed for $result_file (continuing to next training directory)"
        continue
    fi
done

echo
echo "Done."