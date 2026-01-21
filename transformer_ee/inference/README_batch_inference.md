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

Outputs include:

- CSV/NPZ outputs per model + sample pair in `--output-dir`.
- `timings.csv` for per-batch timing.
- `summary.json` with metadata for each task.
