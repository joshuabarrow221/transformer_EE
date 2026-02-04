# Batch inference tooling

This directory includes a configurable batch inference runner (`batch_inference.py`) that can pair multiple trained model exports with multiple inference CSVs. The typical workflow is:

1. Create a JSON config describing which model trainings you want to run (by training name or by explicit path), and which CSV inference samples to use.
2. Run `batch_inference.py --pair-config <config.json>`.
3. Review outputs under the `--output-dir` (default: `InferenceTests/Results`).

## Quick start

Create a config from the example:

```bash
cp transformer_ee/inference/batch_inference_config.example.json ./batch_inference_config.json
```

Edit the `model_search_roots`, `models`, `samples`, and `pairs` sections to match your environment. Then run:

```bash
python transformer_ee/inference/batch_inference.py \
  --pair-config ./batch_inference_config.json \
  --output-dir ./InferenceTests/Results \
  --output-format both \
  --device cuda:0
```

## How model discovery works

There are two ways to describe a model:

- **By training name**: supply `training_name` in the `models` list. The script searches each `model_search_roots` entry for a directory containing both `input.json` and `best_model.zip` with a path segment matching the training name. If multiple matches are found (e.g., the same training was run by multiple users), **all matches are used** and each match becomes a separate inference task.
- **By explicit path**: supply `path` (or `dir`) in the `models` list pointing directly at a model export directory (the one that contains `best_model.zip` and `input.json`).

### Labeling and uniqueness

The `name` field in the `models` list is the label used in output files. It **can be shorter and more human-readable** than the training name. When a training name resolves to multiple model exports, the script appends a suffix derived from the search-root-relative path so each model gets a unique label like:

```
MyShortLabel::Beam_Like/Flat_Spectra/Ar40/<training_name>/model_<hash>
```

This keeps outputs distinct across users and duplicate trainings.

## How pair configs are resolved

The config JSON has four sections:

- `model_search_roots`: list of directories to search for model exports when using `training_name`.
- `models`: list of model definitions (each with `name` and either `training_name` or `path`).
- `samples`: list of CSV samples (each with `name` and `path`).
- `pairs`: list of pairings, where `model` and `sample` refer to the **names** defined above.

A minimal example that uses training-name lookup:

```json
{
  "model_search_roots": [
    "/exp/dune/data/users/cborden/MLProject/Training_Samples",
    "/exp/dune/data/users/rrichi/MLProject/Training_Samples",
    "/exp/dune/data/users/jbarrow/MLProject/Training_Samples"
  ],
  "models": [
    {
      "name": "TopModel",
      "training_name": "Numu_CC_Train_DUNEBeam_Flat_Ar40_p1to10_VectorLeptwNC_eventnum_All_NpNpi_Energy_MAPE_Mom_X_MSE_Mom_Y_MSE_Mom_Z_MSE_Topology_MAE"
    }
  ],
  "samples": [
    {
      "name": "sample_placeholder",
      "path": "/path/to/inference/sample.csv"
    }
  ],
  "pairs": [
    {
      "model": "TopModel",
      "sample": "sample_placeholder"
    }
  ]
}
```

### Defaults and matching rules

- In `models`, `name` is **optional**. If omitted, the label defaults to the resolved path relative to `model_search_roots` (or the file name if no root applies).
- In `samples`, `name` is **optional**. If omitted, the label defaults to the file name (or relative to `--sample-root` if used).
- In `pairs`, `model` and `sample` should reference the **labels** (names) defined in `models` and `samples`. The `model` and `sample` fields do **not** automatically default to `training_name` or `path`—they are labels, and should match whatever label you want to pair.

## Testing & validation

The script supports light-weight smoke tests:

```bash
python transformer_ee/inference/batch_inference.py \
  --pair-config ./batch_inference_config.json \
  --max-samples 100 \
  --device cpu
```

- `--max-samples` limits the number of rows per CSV.
- `--device` can be `cpu`, `cuda`, or `cuda:<index>`.

Note: Some versions of `pred_wBatch.Predictor` do not accept `device` or `num_workers`
arguments or return batch timing metadata. The batch inference runner detects this and
falls back to the supported signature automatically, so these flags may be ignored in
older predictor implementations.

Outputs include:

- CSV/NPZ outputs per model + sample pair in `--output-dir`. Use `--output-format csv`
  to write only CSV, or `--output-format npz` for legacy NPZ-only output.
- `timings.csv` for per-batch timing.
- `summary.json` with metadata for each task.

## Optional ROOT-based evaluation

You can optionally run the ROOT macro `graph_eval/eval_model.C` for each generated
CSV to produce `combined_inference_output.root` and `ellipse_fraction.csv`, and to
save the energy/theta 2D canvas as a PNG alongside each per-model output CSV.

Example:

```bash
python transformer_ee/inference/batch_inference.py \
  --pair-config ./batch_inference_config.json \
  --output-dir ./InferenceTests/Results \
  --eval-macro-path ./graph_eval/eval_model.C \
  --eval-output-dir ./InferenceTests/Results \
  --beam-mode true \
  --eval-save-png \
  --eval-png-width 3000 \
  --eval-png-height 2000
```

When enabled, the `summary.json` entries include an `eval_model` section with the
macro path, output directory, optional PNG path, `stdout`/`stderr` from the macro,
and any matching row from `ellipse_fraction.csv` (matched against the per-task
model label). If ROOT fails to execute the macro, `eval_model.error` will explain
why.

Use `--beam-mode true` for beam-neutrino plots with narrowed ranges, or
`--beam-mode false` (the default) for atmospheric-style ranges.
