# Correlation plot macro (`eval_model_correlations.C`)

This macro is designed for outputs from:

- `transformer_ee/inference/correlation_inference.py`

and creates scatter-style correlation plots, optionally color-coded by a third variable.

## File

- `graph_eval/eval_model_correlations.C`

## Usage

### Basic (no color variable)

```bash
root -l -b -q 'graph_eval/eval_model_correlations.C(
  "results/sample_with_corr.csv",
  "true_NuMomMag",
  "pred_NuMomMag",
  "",
  "results/plots",
  "numomag_corr",
  -1,
  true
)'
```

### Colored by a third column

```bash
root -l -b -q 'graph_eval/eval_model_correlations.C(
  "results/sample_with_corr.csv",
  "true_LeptKE",
  "pred_LeptKE",
  "VisibleEnergyFraction",
  "results/plots",
  "leptke_vs_true_colored",
  200000,
  true
)'
```

## Arguments

1. `csv_path`
2. `x_column`
3. `y_column`
4. `color_column` (empty string disables color-coding)
5. `output_dir`
6. `output_stem`
7. `max_points` (`-1` means all rows)
8. `draw_unity_line`

## Outputs

In `output_dir`:

- `<output_stem>.png` (main scatter plot)
- `<output_stem>.root` (canvas saved into ROOT file)
- `<output_stem>_colorbar.png` (when `color_column` is provided)

## Notes

- Rows with missing/non-numeric values in required columns are skipped.
- For large files, use `max_points` to keep rendering responsive.
- The unity line (`y=x`) is useful for quick regression-quality checks.
