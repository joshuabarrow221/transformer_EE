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
