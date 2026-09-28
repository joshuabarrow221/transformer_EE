# Where Polars and GPUs help

Polars is useful for particle-vector string parsing and columnar normalization;
GPU execution supports the frozen network's forward passes and the optional cuML
t-SNE backend. The default openTSNE backend remains CPU-based. Neither backend
changes the latent dimensionality. [GPU.md](GPU.md) describes the A100 workflow
and coordinate-only redraws; GPU timings remain to be measured on that host.

## Measured preparation comparison

A local benchmark read the first 199,990 raw AR23 beam rows and normalized the
configured vector/scalar fields using the saved flat/no-noise MV model statistics.
Three separate processes per engine were run in alternating order on Linux/ARM,
with four Polars threads. The input was on the mounted data drive. Medians were:

| Preparation measurement | pandas 2.3.3 | Polars 1.44.2 |
|---|---:|---:|
| CSV reading | 1.328 s | 0.458 s |
| Vector/scalar normalization | 2.419 s | 0.063 s |
| Combined timed phases | 3.747 s | 0.521 s |
| Peak process RSS | 476.7 MiB | 289.7 MiB |

This was approximately **7.2 times faster** and **39% lower peak process memory**
for the measured preparation representation. All event IDs, topology strings,
vector lengths, and normalized values at network float32 input precision matched
exactly by hashes. Float64 intermediate expression results were not bit-identical,
which is why equivalence is explicitly stated at the network's input precision.

This is an **unbatched preparation microbenchmark**, not an end-to-end timing of
the packaged runner. It retains normalized variable-length columns, excludes
complete truth/topology auditing, padding, model execution, t-SNE, and rendering,
and does not flush the OS filesystem cache. Hashing is outside the timed phases
but is included in the peak RSS. The production runner already uses bounded
chunks and disk-backed output arrays; the table is not a claim of a 39% memory
reduction for that runner or a 7.2-times faster total analysis.

Reproduce on your own host using explicit locations:

```bash
python -m pip install -r plotting/tSNE/requirements-polars.txt
python plotting/tSNE/benchmark_io.py \
  --csv "$TSNE_SAMPLE" --model-directory "$TSNE_MODEL" \
  --rows 199990 --threads 4 --repeats 3 --output "$TSNE_BENCHMARK"
```

The command runs independent workers and refuses to report equivalent preparation
if prepared float32 values or identifiers differ. Output includes all runs,
versions, timings, and comparison hashes. It does not modify the source files.

## Optional production backend

`run.py --prepare-engine polars` replaces Python loops for normalized batch-array
preparation with Polars list expressions. It includes truncation and normalized
zero padding. Saved statistics are used, calculations stay float64 until the
final float32 cast, and the same original Dataset reference check runs afterward.
The default remains `--prepare-engine pandas` for the established reference path.
Changing backend requires a fresh output directory and is recorded in metadata.

CSV event selection and detailed truth/topology checks currently remain shared
pandas code. Thus the production switch does not implement the entire benchmark's
Polars CSV-loading path. A subsequent optimization could keep audited sources
as typed list columns in Parquet and use lazy scans with projection/filter
pushdown, avoiding repeated CSV parsing. Polars documents those optimizations
[here](https://docs.pola.rs/user-guide/lazy/optimizations/). It would require the
same exact row-selection, topology, normalization, and prediction comparisons;
the old Polars Dataset should not be substituted blindly because it casts vector
values to float32 before normalization and applies its own null-row filtering.

## Remaining dominant resources

The largest current row contains 999,950 events. At 1,088 float32 SV features per
event, its dense feature matrix alone needs 4,351,782,400 bytes, about 4.05 GiB.
That matrix has the same size whether the CSV was read by pandas or Polars.
PCA, approximate-neighbor structures, optimizer working arrays, and rendering
require additional host memory. Polars cannot make openTSNE GPU-based.

GPU inference, optional Polars batch preparation, reusable event cohorts, and
separate CPU plotting are complementary optimizations. Measure the full run on
the destination machine before deciding which stage deserves further work.

## Measured coordinate-only redraw

On the local Linux/ARM CPU host, `render.py` redrew the completed beam MV
E_MAPE/P_MAE grid: four training conditions by five generators, 199,990 events
per panel, **3,999,800 points total**. It wrote both PNG and rasterized PDF at
180 DPI in **33.80 seconds wall time**, with **1,857,048 KiB peak RSS (1.77 GiB)**,
as measured by `/usr/bin/time -v`. No inference, PCA or t-SNE ran. Existing
historical coordinate CSVs were read from the mounted drive; OS caches were not
flushed. This is one measured redraw, not a repeated benchmark or GPU timing.
All panels retained their original coordinates and counts; the layout was
visually inspected. New exports also have coordinate checksum verification.
