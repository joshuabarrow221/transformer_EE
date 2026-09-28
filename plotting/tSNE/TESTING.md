# Release validation

The portable package was checked on Linux/aarch64 with Python 3.12.3 and CPU
PyTorch 2.14.0 execution (the installed CUDA-capable build exposed no CUDA device).
Other installed versions were NumPy 2.4.0, pandas 2.3.3, SciPy 1.16.3,
Matplotlib 3.10.8, scikit-learn 1.9.0, openTSNE 1.0.4, Annoy 1.17.3,
threadpoolctl 3.6.0, and Pillow 12.1.0. These are the observed test environment,
not a requirement to obtain the same development/PyTorch build on another host.

## Checks

- Unit/integration tests cover exact topology decoding, pion-charge summation,
  invalid-row selection, legacy SV prediction alignment, real original/optimized
  forward equivalence, a full small FFT run, safe resume, relocation of feature
  outputs, checkpoint mutation rejection, duplicate target rejection, verified
  asset copying, destination protection, and explicit CUDA-unavailable failure.
- The optional Polars backend is tested against the reference forward path,
  including normalized zero padding and over-width event truncation. The
  preparation microbenchmark and its scope are in [PERFORMANCE.md](PERFORMANCE.md).
  On the two real 128-event MV/SV smoke exports, Polars and pandas produced
  bit-identical saved features, penultimate activations, predictions, and IDs.
- Preflight checked all 37 selected real checkpoint bundles and nine sources:
  83 source/model-bundle exports in the full manifest.
- A real-checkpoint smoke run exported the first 128 usable AR23 beam events
  through the flat/no-noise MV E_MAPE/P_MAE model and all four flat/no-noise SV
  E_MAPE/P_MAE components. It produced both PNG/PDF grids and coordinate files.
- Those exports were compared with the corresponding prefixes of the earlier
  validated 199,990-event campaign. Source rows, event IDs, and topology were
  identical. All predictions, head-input features, and penultimate activations
  passed `rtol=2e-4, atol=1e-5`.

| Bundle | Maximum absolute prediction difference | Maximum head-input difference |
|---|---:|---:|
| MV E_MAPE/P_MAE | 9.54e-7 | 2.86e-6 |
| Composed SV E_MAPE/P_MAE | 7.16e-7 | 7.63e-6 |

Prediction differences use the stored units (GeV or GeV/c); activations have no
physical units. Small differences can arise from batch/padding arrangement and
floating-point parsing. The smoke plots used perplexity 10 and 300 iterations
to validate mechanics; they are not replacements for the full scientific fits.

The mapped scratch arrays are explicitly closed before cleanup, including on
the mounted NTFS workspace. Figure caches include renderer code, numerical
library versions, source provenance, and SV block definitions. The relocation
test verifies plotting after both original model and raw-input folders become
unavailable. Original study artifacts were not overwritten.

## Limits

CUDA execution and GPU speedup were not benchmarked on this host. At runtime,
every extraction chunk compares optimized inference to the original Dataset
path on the same selected device, including examples of each observed prong
count. CPU/GPU floating-point variation can affect t-SNE neighborhood choices;
coordinate identity across devices or dependency versions is not promised.

This release smoke test did not repeat the multi-hour full 199,990-event fitting
campaign. The full study's checkpoint selections and methodology were packaged,
and small real-model runs checked the portable implementation. An external GPU
run should begin with the documented small quick-start, then use a separate
directory for the full event count.
