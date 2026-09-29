# A100 execution and coordinate-only redraws

`--backend cuml` runs cuML FFT t-SNE on CUDA. It is optional: `fft` remains the
CPU openTSNE reference backend. `--device cuda:0` selects **inference** hardware;
`--tsne-device 0` independently selects the **t-SNE** GPU (indices are relative
to `CUDA_VISIBLE_DEVICES`). GPU requests fail explicitly; they never silently
fall back, reduce event counts, or install packages.

CPU PCA, SV block scaling and event selection remain shared with the reference
pipeline. cuML performs its own GPU neighbor search and FFT optimization. No
labels enter either fit. All 199,990 events per generator participate; five
beam generators therefore yield 999,950 jointly embedded events per model row.
A row's float32 PCA matrix at 50 dimensions is about 200 MB. This is **not** the
GPU memory requirement: neighbor graphs, FFT grids and other workspaces are
additional. The 1,088-dimensional SV matrix still occupies about 4.35 GB of host
RAM. An A100 with 80 GB is a suitable target for evaluation, but full-size peak
memory and wall time must be measured there. Run one fit process per GPU first.

## Persistent installation

Use a persistent environment, such as the README's virtual environment. Install
its base requirements and a PyTorch build appropriate to your machine. For cuML
and CuPy, use the [official RAPIDS installation selector](https://docs.rapids.ai/install/)
for your Linux distribution, Python, CUDA toolkit and NVIDIA driver. Choose cuML;
it supplies compatible GPU dependencies. Do not install an unrelated package
named `cuml` from an arbitrary index. CUDA wheel families must match the host.

The adapter targets the documented cuML 26.x API (`max_iter`, `method='fft'`,
`learning_rate_method='none'`, `output_type='numpy'`). Dependencies remain
optional, rather than forcing a particular CUDA wheel family onto CPU machines.
Record the environment after successful target-host tests:

```bash
python -m pip freeze > "$TSNE_RUN-environment.txt"
python -c 'from cupy.cuda import runtime; import cuml; print(cuml.__version__, runtime.getDeviceCount())'
```

The package also records cuML/CuPy versions, CUDA driver/runtime versions, GPU
name and memory, explicit optimizer parameters, timings and fit diagnostics.
See the [cuML API](https://docs.nvidia.com/cuml/latest/api/generated/cuml.manifold.TSNE/).

## Test before the full production run

From the repository root, run the ordinary suite and the explicitly enabled
CUDA test. Enabling it makes a missing/broken CUDA installation a failure, not a
skip. It fits a 1,500-event synthetic dataset and reports diagnostics; this is
an integration check, not validation of neutrino physics or million-event speed.

```bash
PYTHONPATH=plotting/tSNE python -m unittest discover -s plotting/tSNE/tests -v
TRANSFORMEREE_TEST_GPU=1 PYTHONPATH=plotting/tSNE \
  python -m unittest discover -s plotting/tSNE/tests -p test_gpu_render.py -v
```

Then try real AR23 data with the supplied quick-start configuration, in a fresh
external directory. Start small and retain the log:

```bash
python -u plotting/tSNE/run.py \
  --manifest plotting/tSNE/examples/quickstart.json \
  --models-root "$TSNE_ASSETS/models" --samples-root "$TSNE_ASSETS/samples" \
  --output "$TSNE_RUN-gpu-smoke" --events 1500 \
  --device cuda:0 --backend cuml --tsne-device 0 --threads 8
```

For the full study, use `examples/study.json`, `--events 199990`, and a fresh
output directory. All other options can remain the same. The default 256-event
inference minibatch is independent of t-SNE's joint sample size. With existing
extracted features and no previous plotting settings in that run directory:

```bash
python -u plotting/tSNE/run.py --stage plot --output "$TSNE_RUN" \
  --backend cuml --tsne-device 0 --threads 16
```

If that run already has CPU plot settings, keep them and write a separate GPU
comparison from its resolved activation manifest:

```bash
python -u plotting/tSNE/plot_tsne.py \
  --manifest "$TSNE_RUN/latent_manifest.json" --output "$TSNE_RUN/gpu-plots" \
  --representation latent --events 199990 --sampling first --strict \
  --lazy-latent-load --backend cuml --tsne-device 0
```

The resolved manifest contains local activation paths. After relocating a run,
`run.py --stage plot` rebuilds that manifest from its plan and new output root;
otherwise update the paths in the resolved manifest before using it directly.

## Scientific comparison and tuning

PCA initialization, Euclidean distance, perplexity 30, 90 neighbors, early
exaggeration 12 for 250 iterations and 1,000 total iterations are explicit.
cuML adaptive tuning is disabled so it cannot change the requested neighbor
count or exaggeration according to population size. `--gpu-learning-rate`
defaults to cuML's documented input rate of 200 and is configurable. This is a
baseline, **not a tuned setting for one million events**. cuML's learning-rate
schedule and neighbor implementation differ from openTSNE's automatic schedule;
matching numeric seeds/settings does not imply equivalent coordinates.

Before adopting GPU results, compare CPU/GPU fits on the same real event cohort,
check finite coordinates, topology counts and the recorded 1,000-event-subset
trustworthiness diagnostic, and inspect seed/perplexity/learning-rate sensitivity.
Use separate output directories for changed settings. The bounded diagnostic
is not the full population's trustworthiness; compare like sample sizes only.
Do not compare absolute island positions across fits or choose a fit solely
because it produces more visually separated topology colors. Optimization may
need a larger learning rate or more iterations at full scale. Report those
settings alongside the figures. Hardware/library changes can change results.

The original development host was CPU-only. Synthetic A100 MIG 40 GB execution
has now passed; see [host validation and setup](HOST_VALIDATION.md). Real-data
and full-size performance checks remain pending.

## Redraw without models, activations or CUDA

Once plots exist, only `plots/<group>/metadata.json` and its coordinate CSV.gz
files are needed. The separate render stage bypasses inference, PCA and t-SNE:

```bash
python plotting/tSNE/run.py --stage render --output "$TSNE_RUN" \
  --style plotting/tSNE/examples/render-style.json
```

Output goes to `TSNE_RUN/redraw/<group>/grid.png` and `.pdf`. `--render-output`
changes that destination; `--only GROUP_ID` selects grids. To redraw a copied
GPU comparison folder or a single completed grid directly:

```bash
python plotting/tSNE/render.py --source "$TSNE_RUN/gpu-plots" \
  --output "$TSNE_RUN/gpu-redraw" --style plotting/tSNE/examples/render-style.json
```

Colors, title, point size, opacity and DPI are presentation overrides only. All
points and coordinates are retained. Redraws do not change the scientific cache.
New exports have SHA-256 coordinate inventories; redraw verifies these plus row
identities, truth labels, finite coordinates and joint-fit counts. Historical
exports lacking hashes remain supported with a warning and structural checks.
`render.json` records the input hashes and presentation options. It contains no
new embedding. Keep this data outside Git.

This is faster regeneration of static figures, not a browser-based interactive
viewer. Decompressing CSVs and rendering millions of points still takes time;
these costs remain after eliminating inference and t-SNE.
