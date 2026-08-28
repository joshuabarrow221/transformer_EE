# Evaluation scripts

This repository includes two convenience scripts for evaluating many trainings by repeatedly running `eval_model.C` over relevant result.csv files. `eval_model.C` is a ROOT macro and the core script for evalutating models. NOTE: both bash scripts rely on the ROOT macro `eval_model.C` and thus require ROOT to be setup before they can be run.

## run_eval_all.sh

`run_eval_all.sh` scans a base directory and runs `eval_model.C` in ROOT batch mode for each `result.csv` it finds.

### What it expects

You call it with a base directory to scan.
It also supports:

- `--beam` / `-b` to enable beam-mode behavior inside `eval_model.C`
- `--output-dir DIR` to choose where `ellipse_fraction.csv` and the ROOT file are written
- `--root-output-name NAME` to choose the ROOT filename, for example `combined_inference_output.root`

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
Beam-mode example:
```bash
./run_eval_all.sh --beam /exp/dune/data/users/cborden/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisND/
```

Evaluate a folder of combined inference CSVs and write outputs into a dedicated directory:
```bash
./run_eval_all.sh \
  --output-dir /path/to/eval_outputs \
  --root-output-name combined_inference_output.root \
  /path/to/folder/containing/combined_result__*.csv
```

### Outputs for run_eval_all.sh
All outputs are written to the current working directory by default, unless `--output-dir` is provided. Outputs consist of a ROOT file and an `ellipse_fraction.csv` file including evaluation tools for all of the trainings found in the given base directory.

## run_full_evals_all_users.sh

`run_full_evals_all_users.sh` is a driver that runs `run_eval_all.sh` repeatedly across a predefined set of training types and users, and aggregates outputs into per-type directories. It also evaluates combined single-variable CSV directories (named `combined_result__*.csv`) when configured.

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
* Generates 1D resolution histograms for every matched variable pair.
* Resolution histograms for `cos(theta)` variables use absolute residuals with fixed axes `(-1, 1)` and 1000 bins.
* Resolution histograms for `Nu_Theta` use absolute residuals with fixed axes `(-180, 180)` and 1000 bins (works even if theta is derived from momentum).
* Resolution histograms for all other variables use percent resolution with fixed axes `(-200, 200)` and 1000 bins.
* Creates 2D “resolution vs truth” graphs using mean ± RMS or mean ± std (with resolution clamped to ±200%).
* Creates 2D truth-vs-reco histograms for every variable pair.
* Builds a special 2D histogram of **energy resolution (%) vs Δθ**, including:
  - 95%, 90%, and 68% highest-density contours
  - A fixed ellipse (±10% × ±30°)
  - Computation of the fraction of events inside the ellipse
* Stores the ellipse fractions in a cumulative `ellipse_fraction.csv` file (auto-created and auto-expanded), including a `wandb_runtime_hours` column when provided.
* Appends or updates a directory inside `combined_output.root` named after the model (taken from the final CSV column header).
* Runs in ROOT **batch mode** so plots are written to file without opening GUI windows.
* Optional selection controls for robustness studies: filter by `true_Topology` codes (comma-separated list), required proton/pion counts, true-energy range, and true-theta range.
* Per-topology overlay outputs are **not** generated. When topology composition cuts are used, all plots are produced for the filtered subset using the provided `name_prefix`.

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

If the ROOT file already exists, new outputs are appended.

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

Parameter order for `eval_model.C(...)` (full list, in order):
```text
filename,
beam_mode,
wandb_runtime_hours,
output_dir,
png_path,
png_width,
png_height,
root_output_name,
name_prefix,
topology_codes_csv,
required_n_proton,
required_total_pions,
true_energy_min,
true_energy_max,
true_theta_min_deg,
true_theta_max_deg
```

Optional selection controls (examples):
```bash
root -l 'eval_model.C("result.csv", false, -1.0, ".", "", 0, 0, "combined_output.root", "", "301000000000000,300000100000000", 1, 1, 1.0, 5.0, 80.0, 100.0)'
```

## plot_csv_overlays.C (new generalized CSV overlay tool)

`plot_csv_overlays.C` is a generic ROOT macro to rapidly compare up to **8** distributions from arbitrary CSV files, with on-the-fly controls tuned for energy/angle/mass-resolution style plots.

### Supported capabilities

1. Normalize each histogram independently (`normalize=true/false`)
2. Overlay up to 8 curves at once
3. Per-curve color, line style, and line width
4. User-defined binning (`nbins`)
5. User-defined x/y display ranges (`xmin/xmax`, optional `ymin/ymax`)
6. Vertical guides at `x=0` and optional symmetric guides at `±value`
7. Output to a single ROOT file with `TDirectory` placement

### Plot-spec format

The first argument is a semicolon-separated list, where each item is:

```text
file.csv|column_name|legend label|color|line_style|line_width
```

- `color`: ROOT token (`kRed`, `kBlue`, etc.) or integer color code
- You may omit trailing style fields; defaults are used.

### Direct ROOT CLI example

```bash
root -l -b -q 'plot_csv_overlays.C(
"Noised_FlatvsNat_MVSV_CMCs/E-MAPE__P-MAE/DBNat_on_DBNat_Vector_SV_E_MAPE_P_MAE_AR23.csv|E_MAPE|SV AR23 w/Noise|kRed|1|2;
Noised_FlatvsNat_MVSV_CMCs/E-MAPE__P-MAE/DBNat_on_DBNat_Vector_SV_E_MAPE_P_MAE_G2111a.csv|E_MAPE|SV G2111a w/Noise|kBlue|1|2",
220,-4,4,true,-1,-1,true,0.5,
"Energy Resolution (%)","Normalized Events",
"DUNE ND-like beam: SV comparison","",
"combined_output.root","csv_overlay/SV","sv_energy_overlay","sv_energy_overlay.png")'
```

### Bash wrapper

For faster repeated use, run:

```bash
./run_csv_overlay.sh \
  --plots "file1.csv|E_MAPE|SV AR23 Natural|kRed|1|2;file2.csv|E_MAPE|SV G2111a Natural|kBlue|1|2" \
  --bins 220 --xmin -4 --xmax 4 --normalize true \
  --draw-v0 true --sym-vline 0.5 \
  --xtitle "Energy Resolution (%)" --ytitle "Normalized Events" \
  --title "DUNE Beam Flat vs Natural" \
  --output-root combined_output.root --tdir csv_overlay/SV \
  --canvas-name sv_comp --png sv_comp.png
```

The wrapper forwards all options to `plot_csv_overlays.C` in ROOT batch mode.
