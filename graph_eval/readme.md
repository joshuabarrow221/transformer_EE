# eval_model.C

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
* Appends or updates a directory inside `combined_output.root` named after the model (taken from the final CSV column header).
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
