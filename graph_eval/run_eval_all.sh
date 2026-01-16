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
BASE_DIR="${1:-}"

if [[ -z "$BASE_DIR" ]]; then
    echo "Usage: $0 BASE_DIR"
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: BASE_DIR '$BASE_DIR' is not a directory."
    exit 1
fi

echo "Scanning base directory: $BASE_DIR"
echo

# Make globs that don't match expand to nothing instead of themselves
shopt -s nullglob

# if this looks like a "combined CSV output" directory, evaluate those CSVs directly ---
combined_csvs=( "$BASE_DIR"/combined_result__*.csv )
if (( ${#combined_csvs[@]} > 0 )); then
    echo "Detected combined CSV directory (found ${#combined_csvs[@]} combined_result__*.csv files)."
    for csv in "${combined_csvs[@]}"; do
        echo "---------------------------------------------"
        echo "Processing combined CSV: $csv"
        if ! root -l -b -q "${MACRO}(\"$csv\")"; then
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

    # Run ROOT in batch mode, calling eval_model.C("full/path/to/result.csv")
    if ! root -l -b -q "${MACRO}(\"$result_file\")"; then
        echo "  ERROR: ROOT failed for $result_file (continuing to next training directory)"
        continue
    fi
done

echo
echo "Done."