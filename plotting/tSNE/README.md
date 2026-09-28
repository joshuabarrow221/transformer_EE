# TransformerEE topology t-SNE

Portable event-level inference and latent visualization for the `wide_model_test`
branch and its GPU development branch. Model/sample pairings are configuration, not Python source edits. Use
existing AR23-trained models on GENIE or NuWro inputs; no retraining is needed.

Each point is one interaction, colored by its truth proton/pion topology. The
network's regression-head input is embedded; its inactive topology prediction
is never used. [METHODS.md](METHODS.md) explains interpretation and selection.

## What belongs in Git

Keep this small package, manifests, documentation, and tests in Git. Keep raw
CSVs, weights, activations, coordinates, logs, and generated plots in external
asset/run directories. The complete supplied study requires 37 checkpoints and
nine CSVs: 120 asset files totaling 5,299,005,374 bytes (about 5.3 GB). Those files
are **not included in this package**. Disk requirements for extracted features,
scratch arrays, and t-SNE results are additional; allow tens of GB for the full run.

No tool contains personal machine paths. The runner accepts `--repo`,
`--models-root`, `--samples-root`, and `--output`. Asset paths in manifests are
relative to those roots. The full study manifest records checkpoint IDs/hashes
and sample filenames as editable configuration, not executable path constants.
Private maps from old local folders to these assets belong outside Git.

## Install once on the execution machine

Use Linux or WSL, Python 3.10+, and a persistent virtual environment. From the
repository root, for example:

```bash
python3 -m venv "$HOME/.venvs/transformeree-tsne"
source "$HOME/.venvs/transformeree-tsne/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r plotting/tSNE/requirements.txt
```

Install PyTorch separately so its build matches your GPU/driver. Use the current
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Only `torch` is required here; torchvision/torchaudio are not needed. For a CPU
installation, the official CPU wheel index can be selected explicitly:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU extraction, install the CUDA build instead and verify:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

If Annoy needs compilation, the host needs a C++ compiler and Python development
headers. No package downloads or environment installation are performed by the
runner. Caches respect `XDG_CACHE_HOME`, `MPLCONFIGDIR`, and `NUMBA_CACHE_DIR`.
On an affected ARM virtual CPU only, `NUMBA_CPU_NAME=generic` can work around the
optional Numba startup issue encountered during the original study.

## Prepare external assets

Choose directories appropriate for your machine. These environment variables
are convenient examples, not required names or fixed locations:

```bash
export TSNE_ASSETS="$HOME/transformeree-assets"
export TSNE_RUN="$HOME/transformeree-runs/study"
```

The shipped configurations expect this organization below `TSNE_ASSETS`:

```text
models/
  model_<checkpoint-id>/
    best_model.zip
    input.json
    trainset_stat.json
samples/
  Beam_Like/Natural_Spectra/DUNEOnAxisND/<sample>.csv
  Atmospherics_DUNE_Like/Natural_Spectra/<sample>.csv
```

You can instead edit the manifest's relative paths to match your own layout.
Never substitute normalization computed from inference events: use the saved
training statistics. An SV bundle needs four compatible component networks.

To collect the existing study assets on the originating machine, use the
separately supplied **private** inventory. Set `TSNE_INVENTORY` to that JSON file
and `TSNE_SOURCE_ROOT` to the directory its `source` entries are relative to:

```bash
python plotting/tSNE/stage_assets.py \
  --manifest plotting/tSNE/examples/study.json \
  --inventory "$TSNE_INVENTORY" --source-root "$TSNE_SOURCE_ROOT" \
  --destination "$TSNE_ASSETS"
```

This is a dry run. Add `--copy` to copy the selected files and verify SHA-256
checksums. Existing different destination files are refused. The resulting
`TRANSFER.json` records every asset hash, including raw samples. Transfer the
asset directory using your usual storage/rsync/scp workflow, separately from Git.
The inventory schema is documented in [examples/README.md](examples/README.md).

## Quick start and full inference

Run these commands from the repository root. The quick-start manifest selects
one real beam model and AR23 sample; use a fresh output directory for a different
event count or checkpoint selection:

```bash
python plotting/tSNE/run.py \
  --manifest plotting/tSNE/examples/quickstart.json \
  --models-root "$TSNE_ASSETS/models" --samples-root "$TSNE_ASSETS/samples" \
  --output "$TSNE_RUN-smoke" --events 1500 --device cpu --threads 4
```

The default `--repo` is inferred from the package's position in the checkout.
Supply `--repo` explicitly if the package is elsewhere. A preflight checks all
required files and checkpoint hashes without producing features:

```bash
python plotting/tSNE/run.py --stage check \
  --manifest plotting/tSNE/examples/study.json \
  --models-root "$TSNE_ASSETS/models" --samples-root "$TSNE_ASSETS/samples" \
  --output "$TSNE_RUN" --device cuda:0
```

Extract all available study configurations on a GPU:

```bash
python -u plotting/tSNE/run.py --stage extract \
  --manifest plotting/tSNE/examples/study.json \
  --models-root "$TSNE_ASSETS/models" --samples-root "$TSNE_ASSETS/samples" \
  --output "$TSNE_RUN" --events 199990 --device cuda:0 \
  --batch-size 256 --chunk-size 8192 --threads 4
```

Use `--device cpu` for CPU execution. CUDA requests fail clearly if CUDA is
unavailable; they never silently fall back. Start with batch size 256 and measure
on the target GPU before increasing it. Lower it if device memory is exhausted.
`--chunk-size` bounds host preprocessing memory independently of GPU minibatches.
Model evaluation stays FP32 with TF32 disabled; there is no mixed-precision mode.

For optional Polars vector parsing, normalization, truncation, and padding:

```bash
python -m pip install -r plotting/tSNE/requirements-polars.txt
```

Then add `--prepare-engine polars` to extraction/check commands and use a new
output directory when changing engines. The default is `pandas`. Both paths are
checked against the original Dataset forward pass. CSV selection and the truth
audit remain shared; this option does not replace t-SNE's NumPy matrices.
See [PERFORMANCE.md](PERFORMANCE.md) for a measured preparation comparison and
its limits. `POLARS_MAX_THREADS`, if set, overrides the default thread setting.

The [full manifest](examples/study.json) contains six main/supplemental grids and
83 source/model-bundle exports: MV MAE (20), MV MAPE (15), two atmospheric grids
(4 each), and both completed SV loss families (20 each). This includes the later
corrected SV noise bundles, superseding their missing rows in the earlier study.
Atmospheric NuWro input and beam MV MAPE natural/no-noise weights remain absent.
The atmospheric natural/noise checkpoint is a separately labeled replica. The
flat/noise SV MAPE configuration explicitly selects the verified J1 Px model.

Use `--only GROUP_ID [GROUP_ID ...]` to choose whole grids. For other pairings,
edit `groups[].rows` in a manifest; each row declares its models and sources.
See [examples/README.md](examples/README.md). Changes to the selected models,
inputs, code, or event count require a new output directory, preventing accidental
reuse of incompatible activations. Repeating the same command resumes completed
exports after checking their hashes. An interrupted export is recomputed.

## Fit and render on CPU

```bash
python -u plotting/tSNE/run.py --stage plot \
  --output "$TSNE_RUN" --threads 16
```

This requires the completed `plan.json` and `features/` directory only. They can
be moved to another host; the old absolute paths recorded as provenance are not
required to exist there. `--stage plot` obtains the event count from `plan.json`.
`RESULTS.md` links each PNG/PDF. CSV.gz coordinates, counts, and fit metadata are
written under `plots/`. The default `--stage all` performs extraction and plotting
in sequence on one host.

`--device cuda` accelerates model forward passes. Optional `--backend cuml`
adds GPU neighbor search and FFT t-SNE; see [GPU.md](GPU.md) for A100 setup,
validation and examples. PCA, SV scaling and Matplotlib drawing remain on CPU.
The default `fft` backend continues to use CPU openTSNE. See openTSNE's
[parallelism documentation](https://opentsne.readthedocs.io/en/stable/api/index.html).
A full beam row jointly fits 999,950 events. Its SV feature matrix alone is about
4.35 GB in float32; PCA, neighbor structures, and plotting need additional RAM.
64 GB or more provides practical headroom for full rows, but is not a measured
minimum or a guarantee for every library version. GPU speedup has not been
benchmarked in this package's CPU-only test environment.

Multiple GPUs can run independent task shards. Launch one process per GPU with
the same manifest, output, event count, and extraction settings, and distinct
`--worker-index` values in `[0, --workers)`. For example, two extraction commands
use `--workers 2 --worker-index 0 --device cuda:0` and
`--workers 2 --worker-index 1 --device cuda:1`. Tasks are assigned deterministically;
locks protect shared selections and duplicate exports. Run plotting once after
**all** shards finish. There is no distributed training or collective operation.

For the original seed/perplexity sensitivity check, after a normal plotting run:

```bash
python plotting/tSNE/plot_tsne.py \
  --manifest "$TSNE_RUN/latent_manifest.json" --output "$TSNE_RUN/sensitivity" \
  --representation latent --events 199990 --sampling first --backend fft \
  --seed 7 --perplexity 50 --iterations 1000 --strict --lazy-latent-load \
  --only DB_MV_E_MAPE_P_MAE_latent_25epochs
```

## Redraw saved coordinates

```bash
python plotting/tSNE/run.py --stage render --output "$TSNE_RUN" \
  --style plotting/tSNE/examples/render-style.json
```

This uses only completed plot metadata and coordinate CSV.gz files, with no
models, activations, CUDA or refitting. Figures go to `redraw/`; original fits
stay intact. `--render-output` and `--only` control the destination and groups.
See [GPU.md](GPU.md) for transfer, integrity checks, and style options.

## Tests and provenance

```bash
PYTHONPATH=plotting/tSNE python -m unittest discover -s plotting/tSNE/tests -v
```

Tests use a tiny real TransformerEE network to check corrupt-input exclusion,
original-vs-optimized forward equivalence, complete extraction/FFT plotting,
resume, relocated plotting without source assets, and rejection of changed
checkpoints or duplicate target assignments. The CUDA-unavailable behavior is
tested on CPU hosts. [TESTING.md](TESTING.md) records the release checks and limits.

Per-export metadata preserves model configurations, training statistics, hashes,
source selection, device, batch/chunk sizes, and reference-forward checks.
All execution metadata remains in the external output directory. The package
uses the repository's model and Dataset implementations directly; it does not
vendor a competing network architecture.
