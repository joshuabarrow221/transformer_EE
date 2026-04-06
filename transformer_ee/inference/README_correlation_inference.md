# Correlation-focused inference workflow

This workflow is intended for **validation plots that correlate model outputs with additional truth-level quantities** carried in the same inference CSV.

It adds a dedicated script:

- `transformer_ee/inference/correlation_inference.py`

which writes a single output CSV containing:

1. passthrough columns from the source inference file,
2. `true_<target>` columns (if available),
3. `pred_<target>` columns,
4. `resid_<target> = pred - true` columns.

This keeps everything needed for later studies in one table.

---

## Why this is useful

For plots like:

- `predicted KE` vs `true KE`,
- marker color from another truth variable (`visible fraction`, topology, etc.),

you need those additional truth columns to survive inference. Rather than rebuilding joins later, this script writes a **correlation-ready table** directly.

---

## Command-line usage

```bash
python transformer_ee/inference/correlation_inference.py \
  --model-dir /path/to/model_export \
  --input-csv /path/to/sample.csv \
  --output-csv /path/to/results/sample_with_corr.csv \
  --copy-targets \
  --copy-columns EventID,NuMomX,NuMomY,NuMomZ \
  --copy-regex "^(Vis|True|Topology|Nu)" \
  --write-manifest
```

### Important flags

- `--copy-columns`
  - Explicit comma-separated list of columns to copy unchanged.
- `--copy-regex`
  - Regex to copy any matching source columns.
- `--copy-scalars`
  - Copy scalar inputs used by the model (`input.json -> scalar`).
- `--copy-targets`
  - Copy target columns if present (truth validation mode).
- `--write-manifest`
  - Writes `<output>.manifest.json` with selected column metadata.
- `--max-rows`
  - Useful for smoke tests.

---

## Recommended pattern for truth-correlation studies

1. Include event identifiers (`event`, `EventID`, etc.) in `--copy-columns`.
2. Include key truth controls via `--copy-columns` and/or `--copy-regex`.
3. Include targets via `--copy-targets` so each row has `true_` and `pred_` pairs.
4. Keep the manifest (`--write-manifest`) for reproducibility.

---

## Output schema

Each row corresponds to one input row.

- `event_index`
- passthrough columns (selected by CLI)
- `true_<target_i>` (NaN if target absent in input CSV)
- `pred_<target_i>`
- `resid_<target_i>`

---

## Notes

- This script is intentionally separate from `batch_inference.py` to keep standard production inference outputs unchanged.
- If you need this for many model/sample pairs, you can call this script in shell loops or adapt `batch_inference.py` later.
