# Single-Variable Result CSV Merger

This directory contains two utilities used to **combine multiple single-variable transformer training outputs (`result.csv`) into a single, merged `result.csv`** that can be evaluated with the same tools used for multivariate trainings (e.g. `eval_model.C`).

---

## Files

### `build_combined_singlevar_results.sh`

A **bash driver script** that:
- Searches a base directory containing *single-variable* training outputs
- Finds the **most recent `model_*/result.csv`** for each variable and loss type
- Automatically builds many **combined CSV files** by repeatedly calling the ROOT macro below

---

### `merge_result_csv_two.C`

A helper **ROOT macro** that:
- Merges `true_*` and `pred_*` columns for **one variable at a time**
- Either:
  - **Bootstraps** a new combined CSV, or
  - **Appends** new variable columns to an existing combined CSV
- Enforces row-by-row alignment using `true_Topology`
- Guarantees that the **last column header encodes the combined “model name”**, which downstream evaluation macros rely on

This macro is normally invoked by the bash script, but can also be run manually.

---

## Expected directory layout

The bash script assumes a structure like:
(SINGLE_VAR_BASE)/(TYPE)/NpNpi_Energy_MAE_Topology/model_000/result.csv


Key assumptions:
- Each training directory contains one or more `model_*/` subdirectories
- The **newest** `model_*/result.csv` is the one you want
- Training directory names encode both the **variable** and **loss**
- All single-variable CSVs were produced from the *same underlying event set*

---

## How to run (recommended workflow)

### 1. Make the bash script executable
```bash
chmod +x build_combined_singlevar_results.sh
```
### 2. Run the script
```bash
./build_combined_singlevar_results.sh \
  (SINGLE_VAR_BASE) \
  (OUTPUT_BASE) \
  (MACRO_DIR)
  ```
Explicit example:
```bash
./build_combined_singlevar_results.sh \
  /exp/dune/data/users/USERNAME/single_var_models \
  /exp/dune/data/users/USERNAME/single_var_models_combined \
  /exp/dune/data/users/USERNAME/eval_macros
  ```
### 3. Check outputs
The script should create directories like
(OUTPUT_BASE)/(TYPE)/
  combined_result__E-MAE__P-MSE.csv
  combined_result__E-MAPE__Th-MAE.csv
  ...
  build_combined.log
  WARNINGS.txt

Each combined_result__*.csv is ready to be passed directly to evaluation macros.

---

## Inference-mode combiner

For inference outputs produced by `transformer_ee/inference/batch_inference.py`, use:

- `build_combined_singlevar_inference_results.sh`

This script scans recursively for directories named like:

- `..._NpNpi_<VAR>_<LOSS>_Topology_<SUFFIX>/`

and for each such directory picks the newest CSV by modification time (e.g. when a directory contains multiple batch-inference samples). It then builds pseudo-multivariate combined CSVs using the same
combination grid as the training combiner:

- `combined_result__E-<LOSS>__P-<LOSS>.csv` (Energy + Mom_X + Mom_Y + Mom_Z)
- `combined_result__E-<LOSS>__Th-<LOSS>.csv` (Energy + angular variable)
- `combined_result__E-<LOSS>__Th-<LOSS>__Phi-<LOSS>.csv` (Energy + Theta + Phi)
- `combined_result__E-<LOSS>__CosTheta-<LOSS>__Phi-<LOSS>.csv` (Energy + CosTheta + Phi)

Angular variables are resolved in priority order per loss for 2-variable Energy+angular outputs (the filename token becomes `Th`, `CosTheta`, or `Phi` to match what was selected):

1. `Theta`
2. `Nu_CosTheta`
3. `Phi`

For 3-variable outputs, `Phi` is required and the script builds whichever of `Theta` and/or `CosTheta` is available for each loss pairing.

If a momentum component is missing for a loss (for example `Mom_X` missing for `MACE`), the script logs an explicit `[ERROR]` line and skips only combinations that require that missing momentum loss. Other valid momentum combinations continue to build.

Optional 4th argument: a common eval output directory. If provided, the script runs
`graph_eval/run_eval_all.sh` on each output group directory while `cd`'d into that
common directory so existing `ellipse_fraction.csv` is **appended/expanded** rather
than replaced.

### Usage

```bash
chmod +x build_combined_singlevar_inference_results.sh
./build_combined_singlevar_inference_results.sh \
  /exp/dune/data/users/USERNAME/MLProject/Inference_Samples \
  /exp/dune/data/users/USERNAME/MLProject/Inference_Samples/Combined_SingleVar \
  /path/to/transformer_EE/graph_eval/single_var \
  /exp/dune/data/users/USERNAME/MLProject/Inference_Samples/CommonEval
```

If the 4th argument is omitted, the script only builds the combined CSVs.


### Troubleshooting scanner appears to stop after "Scanning inference directories..."

If your filesystem has unreadable subdirectories (common on shared storage), older versions of the script could exit early during recursive `find`. The scanner now ignores unreadable branches and prints a summary like:

- `candidate_dirs=<N>, matched_dirs_with_csv=<M>, unique_keys=<K>`

This helps distinguish between:
- no matching directory names,
- matching names but no CSV files,
- or successful discovery with deduplication to unique `(group,var,loss)` keys.


For 2-variable Energy+angular outputs, filename tokens now match the selected variable: `Th`, `CosTheta`, or `Phi` (instead of always using `Th`).

For directory names ending with run labels like `_J1`, `_R1`, `_C1`: these suffixes are treated as **immaterial** for matching. The scanner keys each candidate by `(group, variable, loss)` and then keeps whichever CSV has the newest modification time for that key.

`group` (shown as `GROUP_KEY` in logs) is simply the shared prefix before `_NpNpi_<VAR>_<LOSS>_Topology...` and is used to ensure only compatible single-variable outputs are merged together.

Output subdirectories are now named with a human-readable tag (shown as `GROUP_TAG`), for example:

- `DUNEAtmoFlat_Infer_Vector_NpNpi_SV`

If multiple distinct groups map to the same tag, the script appends `__dupN` to keep output paths unique.

