# Evaluation scripts

This repository includes two convenience scripts for evaluating many trainings by repeatedly running `eval_model.C` over relevant result.csv files. `eval_model.C` is a ROOT macro and the core script for evalutating models. NOTE: both bash scripts rely on the ROOT macro `eval_model.C` and thus require ROOT to be setup before they can be run.

## run_eval_all.sh

`run_eval_all.sh` scans a base directory and runs `eval_model.C` in ROOT batch mode for each `result.csv` it finds.

### What it expects

You call it with a single argument: the base directory to scan.

It supports **two input layouts**:

1. **Training directories:** `BASE_DIR` contains multiple training run subdirectories, each with one or more `model_*/` folders that contain `result.csv`.
   - If multiple `model_*/` exist, the script evaluates the **most recently modified** one that contains `result.csv`.

2. **Combined-CSV directories:** `BASE_DIR` directly contains one or more files named:
  - `combined_result__*.csv`
   In this case, the script evaluates each combined CSV directly.

### How to run run_eval_all.sh
```bash
chmod +x run_eval_all.sh
./run_eval_all.sh /path/to/base_dir
```
For example:
```bash
./run_eval_all.sh /exp/dune/data/users/cborden/MLProject/Training_Samples/Beam_Like/Natural_Spectra/NOvAND/
```

### Outputs for run_eval_all.sh
All outputs are written to the current working directory by default and consist of a .root file and a ellipse_fraction.csv file including evaluation tools for all of the trainings found in the given base directory.

## run_full_evals_all_users.sh

`run_full_evals_all_users.sh` is a driver that runs `run_eval_all.sh` repeatedly across a predefined set of training types and users, and aggregates outputs into per-type directories.

### What it does

For each configured training type, it:

1. Creates an output directory named like `*_FullEval/`
2. `cd`s into that directory
3. Runs `run_eval_all.sh <BASE_DIR_FOR_THIS_TYPE>`
4. Appends stdout/stderr to a log file:
   - `*_FullEval/full_eval.log`

Because it runs evaluations *inside* each `*_FullEval/` directory, each training type gets its own:
- `combined_output.root`
- `ellipse_fraction.csv`
- `full_eval.log`

### How to run run_full_evals_all_users.sh
The set of USERS, TRAINING_TYPES, and corresponding base directories must be adjusted to match the file path of the trainings that are intended to be run over.

After adjusting those configuration to match your directory structure, simply run
```bash
chmod +x run_eval_all.sh run_full_evals_all_users.sh
./run_full_evals_all_users.sh /path/where/you/want/outputs
```

## eval_model.C

`eval_model.C` is a ROOT macro for evaluating neutrino energy and momentum reconstruction performance using a CSV file containing **true** and **predicted** kinematic variables. The script automatically generates a large collection of 1D/2D histograms, resolution plots, contour plots, and summary metrics used for model evaluation. All outputs are appended to a central ROOT file and a CSV summary file.

## Features

* Automatically parses any `true_*` / `pred_*` variable pairs from the input CSV.
* Produces 1D histograms for every CSV column plus derived physics quantities:
  - Neutrino θ and cos θ (true and predicted)
  - Baseline estimates (true and predicted)
  - Mass-squared (true and predicted)
* Generates percent-resolution histograms for every matched variable pair.
* Creates 2D “resolution vs truth” graphs using mean ± RMS or mean ± std (with resolution clamped to ±200%).
* Creates 2D truth-vs-reco histograms for every variable pair.
* Builds a special 2D histogram of **energy resolution (%) vs Δθ**, including:
  - 95%, 90%, and 68% highest-density contours
  - A fixed ellipse (±10% × ±30°)
  - Computation of the fraction of events inside the ellipse
* Stores the ellipse fractions in a cumulative `ellipse_fraction.csv` file (auto-created and auto-expanded).
* Appends or updates a directory inside `combined_output.root` (or a custom ROOT output name) named after the model (taken from the final CSV column header).
* Runs in ROOT **batch mode** so plots are written to file without opening GUI windows.

## Requirements

* ROOT 6.x with C++17 support (standard on most Linux HEP systems)
* A CSV file with:
  - Any number of scalar columns
  - Columns beginning with `true_` and `pred_` for matching variable pairs
  - Neutrino momentum components if angular/ baseline/ mass-squared quantities are desired:
    * `true_Nu_Mom_X`, `true_Nu_Mom_Y`, `true_Nu_Mom_Z`
    * `pred_Nu_Mom_X`, `pred_Nu_Mom_Y`, `pred_Nu_Mom_Z`
  - The **last column header must be the model name**, which determines the output directory inside the ROOT file.

## Output Files

### combined_output.root
A structured ROOT file containing:

* All 1D histograms
* All 2D histograms
* All TGraphErrors resolution plots
* Contour plots and energy-vs-angle canvases
* All outputs organized under a directory named after the model

If the ROOT file already exists and is non-empty, new outputs are appended.

### ellipse_fraction.csv
A cumulative CSV summary containing:

* A `model_name` column
* Columns of the form:  
  `Fraction inside ellipse (center 0 0; a=10; b=30)`
* A new row is added each time the macro runs.

## Usage

Run the macro from a shell:

```bash
root -l 'eval_model.C("result.csv")'
```

Optional arguments can set the output directory, PNG export path/size, and the ROOT
output filename:

```bash
root -l 'eval_model.C("result.csv", "./Results", "energy_theta.png", 3000, 2000, "combined_output.root")'
```
or
```bash
root -l 'neweval_model.C("result.csv", "false", "-1.0",./Results", "energy_theta.png", 3000, 2000, "combined_output.root")'
```
