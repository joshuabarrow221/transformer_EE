# CONFIGS_README.md — Batch inference config generation for `transformer_EE`

This README explains how to **generate** and **use** `batch_inference_config.*.json` files for the `transformer_EE` repository (we typically are now using the branch `wide_model_test` which will eventually become `main`). Specifically, the batch inference runner located at:

- `transformer_ee/inference/batch_inference.py` (run inference for many model/sample pairs)

The goal of the helper scripts is to take our **model-name text lists** (across potentially multiple analyzers), **deduplicate** them, **separate Vector vs Scalar models**, and emit configuration JSON files that the batch inference tool can consume via `--pair-config`.

---

## Files this workflow currently uses

### 1) Input: Current model-name lists (text files)

We provide lists like:

- `Train_Atmospheric_Flat_Models_rrichi.txt`
- `Train_Atmospheric_Flat_Models_cborden.txt`
- `Train_Atmospheric_Flat_Models_jbarrow.txt`
- `Train_DUNEBeam_Flat_Models_rrich.txt`
- `Train_DUNEBeam_Flat_Models_cborden.txt`
- `Train_DUNEBeam_Flat_Models_jbarrow.txt`
- `Train_DUNEBeam_Nat_Models_rrichi.txt`
- `Train_DUNEBeam_Nat_Models_cborden.txt`
- `Train_DUNEBeam_Nat_Models_jbarrow.txt`
- `Train_NOvABeam_Nat_Models_rrichi.txt`
- `Train_NOvABeam_Nat_Models_cborden.txt`
- `Train_NOvABeam_Nat_Models_jbarrow.txt`

Each file contains many lines/model names returned from a simple `ls` command; the generator extracts the *trained model training names* (strings starting at the moment with `Numu_CC_Train_...`) and then filters/splits as needed:
- Atmospheric configs are filtered by optional substring lists (see `--atm-require-substrs`)
- Beam configs are separated into flat vs natural (and DUNE vs NOvA) based on training-name tokens

The tool then deduplicates across all provided lists.

### 2) Generator scripts

- `generate_batch_inference_configs.py` — core generator which reads lists, dedupes, builds `models`, `samples`, and `pairs`
- `make_batch_inference_configs.sh` — convenient bash shell wrapper which calls the Python generator

### 3) Output: the 4 configuration files

The generator currently writes out configuration files pertinent to flat atmospheric neutrino spectra - trained models to be used on separate natural atmospheric neutrino spectra inference samples:

1. `batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json`
2. `batch_inference_config.DUNEBeamFlat-to-DUNEFDBeamOsc.json`
3. `batch_inference_config.DUNEBeamFlat-to-DUNEND39mOffAxisBeamNat.json`
4. `batch_inference_config.DUNEBeamFlat-to-DUNENDBeamNat.json`
5. `batch_inference_config.DUNEBeamNat-to-DUNEFDBeamOsc.json`
6. `batch_inference_config.DUNEBeamNat-to-DUNEND39mOffAxisBeamNat.json`
7. `batch_inference_config.NOvABeamNat-to-NOvAFDBeamOsc.json`

---

## What a batch inference config contains (schema)

Each config JSON contains four top-level keys (including a map):

1) **`model_search_roots`**
- List of directories that hold trained model exports (the batch tool recursively searches these).

2) **`models`**
- Each entry is a dict containing at minimum:
  - `name` (a unique shorthand label used in `pairs`)
  - `training_name` (the long training name string to locate the model on disk)

3) **`samples`**
- Each entry is a dict containing:
  - `name` (unique shorthand label)
  - `path` (absolute path to a CSV sample)

4) **`pairs`**
- A list of `{ "model": <models[i].name>, "sample": <samples[j].name> }` entries
- The batch tool runs one inference job per pair

### Why `training_name` matters

When we provide a `training_name`, `batch_inference.py` searches the configured roots for trained model directories by scanning for `input.json` files and checking whether `training_name` appears as a path component of that model export directory tree.

**Devil’s advocate / failure mode:** if our exported model directory layout does *not* include the full training name in its path components, the resolver won’t end up finding it. In that case, we must either:
- switch to using explicit model directory `path`s in `models`, **or**
- adjust how exports are stored (so the training name appears in the directory path)

---

## Vector vs Scalar pairing logic

Our training names contain either `...Vector...` or `...Scalar...` in the name, implicating the different training/inference data format which either brings the charge current lepton into the full list (vector) of visible final state particles or instead lead the final state lepton in a single-valued column of data.

The generator currently applies a couple of strict rules which are pretty important:
- Vector models are paired **only** with Vector CSV samples
- Scalar models are paired **only** with Scalar CSV samples

If a model name contains neither substring, it is treated as “Unknown” and is not paired (it may still appear in `models`, depending on how we want to run--in principle this shouldn't happen given our current scope of data preprocessing, associated naming, and data sample organization).

**Devil’s advocate:** if a training name uses a different convention (e.g., `Vect`/`Scal` or missing those substrings), it will inevitably be miscategorized. We can fix this by editing the `_model_kind()` function in the generator.

---

## Sample CSVs used (and where to edit them)

The generator hardcodes the exact CSV sample paths in a `SAMPLES` dictionary near the top of `generate_batch_inference_configs.py`.

The current inference sample paths in that file correspond to:

1. DUNE-like Atmospheric Natural Spectra, already oscillated and assumed to only pertain to FD (p1to10 = 0.1 - 10 GeV of incoming neutrino energy):
- Vector: `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Atmospherics_DUNE_Like/Natural_Spectra/...Vector...csv`
- Scalar: `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Atmospherics_DUNE_Like/Natural_Spectra/...Scalar...csv`

2. DUNE-like Beam Natural Spectra, OnAxis FD Oscillated (p1to6 = 0.1 - 6 GeV of incoming neutrino energy):
- Vector + Scalar in `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisFDOsc/`

3. DUNE-like Beam Natural Spectra, but now 39m OffAxis ND and so unoscillated (p1to2 = 0.1 - 2 GeV of incoming neutrino enegy):
- Vector + Scalar in `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNE39mOffAxis/`

4. DUNE-like Beam Natural Spectra, OnAxis ND and so unoscillated (p1to6 = 0.1 - 6 GeV of incoming neutrino energy):
- Vector + Scalar in `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisND/`

5. NOvA-like Beam Natural Spectra, OnAxis FD Oscillated:
- Vector + Scalar in `/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/NOvAFDOsc/`

Note that some/all of these are under Richi's `Training_Samples` area -- while it is true they were used for training, because they are statistically independent than the flat spectra, we can use them for inference purposes without issue, while also guaranteeing that we have high statistics for inference-related performance plots.

If we move samples or want different ones, edit only the paths in this dictionary—no other logic changes needed.

---

## How to generate the configs

### Step -1 — make relevant text files from the analyzers' directories listing model names

```bash
ls /exp/dune/data/users/<user>/MLProject/Training_Samples/Atmospherics_DUNE_Like/Flat_Spectra/ > Train_Atmospheric_Flat_Models_<user>.txt
```

### Step 0 — clone the repo and check out the right branch

```bash
git clone https://github.com/joshuabarrow221/transformer_EE.git
cd transformer_EE
git checkout wide_model_test
```
Note that eventually this may be under `main`

### Step 1 — put the generator scripts somewhere convenient

For example, from the repo root:

```bash
cp /path/to/generate_batch_inference_configs.py .
cp /path/to/make_batch_inference_configs.sh .
chmod +x make_batch_inference_configs.sh
```

### Step 2 — put the text files next to the scripts (or update paths)

If they are not in the current working directory, either:
- move them there, or
- edit `ATM_FILES` / `BEAM_FILES` arrays in `make_batch_inference_configs.sh`, or
- call the python generator directly with `--atm-files` and `--beam-files`

### Step 3 — run the wrapper

```bash
./make_batch_inference_configs.sh
```

This writes the (currently seven) JSON files into the current directory (or the configured output dir).

### Re-running behavior (backups)

By default the generator creates timestamped backups if an output JSON already exists:

- `batch_inference_config....json.bakYYYYMMDD_HHMMSS`

To overwrite without backups:

```bash
python3 generate_batch_inference_configs.py --no-backup ...
```

---

## How to run batch inference using the generated configs

The batch inference runner supports several modes, but the recommended one here is **pair-config mode**:

```bash
python3 transformer_ee/inference/batch_inference.py   --pair-config batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json   --output-dir /exp/dune/data/users/$USER/MLProject/Inference_Samples/HondaDUNEOsc/Results   --device cuda:0
```

Repeat with the other configs as needed, e.g.

```bash
python3 transformer_ee/inference/batch_inference.py   --pair-config batch_inference_config.DUNEBeamFlat-to-DUNEFDBeamOsc.json   --output-dir /exp/dune/data/users/$USER/MLProject/Inference_Samples/DUNEBeam/Results   --device cuda:0
```

### Optional ROOT macro evaluation

If we want a ROOT-based plotting + ellipse (basic 2D performance metric) evaluation per output:

```bash
python3 transformer_ee/inference/batch_inference.py   --pair-config batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json   --output-dir /exp/dune/data/users/$USER/MLProject/Inference_Samples/HondaDUNEOsc/Results   --device cuda:0   --eval-macro-path ./graph_eval/eval_model.C   --eval-output-dir /exp/dune/data/users/$USER/MLProject/Inference_Samples/HondaDUNEOsc/Results   --beam-mode false   --eval-save-png   --eval-png-width 3000   --eval-png-height 2000
```

For beam-neutrino inference, pass `--beam-mode true` to apply the narrowed
plotting ranges in `eval_model.C`.

---

## How the generator ensures uniqueness of names

The batch inference tool requires that `models[].name` and `samples[].name` are unique within the file, because `pairs` references them by string.

To avoid collisions, the generator:
- creates stable short names for models that include a SHA1 hash fragment of the `training_name`
- uses fixed names for each sample entry (one for Vector, one for Scalar)

**Devil’s advocate:** if we want *human-readable* model shorthand names (instead of hashed), we could replace the `_make_model_name()` function with a readable template; just keep a collision-resistant suffix (hash, or a monotonically increasing ID).

---

## Troubleshooting checklist

### “My model isn’t found” / missing models reported

Likely causes:
- the model export directory does not contain the `training_name` as a directory component
- your `model_search_roots` do not include the actual export locations
- the model export is incomplete (missing `best_model.zip` or `input.json`)

Fixes:
- verify each `model_search_roots` path exists on the machine where you run inference
- locate the export directory and confirm it contains both `input.json` and `best_model.zip`
- if necessary, switch from `training_name` resolution to explicit `path` resolution in the config
- the `batch_inference.py` now supports output for a `models_not_found.txt` which can be saved to the area wherever we output the finished inference samples, allowing for us to go back and retrain if need be if we are somehow missing the `best_model.zip` files storing the model weights. Why this has happened isn't exactly clear, but it can and has happened sadly.
- if you pass `--missing-models-file` (to skip training names that were already reported missing), the run writes `models_missing_previously.txt` alongside the outputs so you still have a record of which training names were skipped. This keeps `models_not_found.txt` focused on *new* missing exports discovered in the current run.

### “We got multiple models for one training_name because we accidentally ran over each others training ranges...”

This is expected if multiple users have ended up with accidentally training the same model, ending up under different roots.

The batch tool will run inference for each discovered instance. If we want **only one**, either:
- we can narrow `model_search_roots` (easy), or
- we can adjust the labeling/selection logic in `batch_inference.py` down the line (more advanced)

---

## Minimal “smoke test” (fast sanity run)

Run only a few rows from the CSVs:

```bash
python3 transformer_ee/inference/batch_inference.py   --pair-config batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json   --output-dir ./InferenceSmokeTest   --device cpu   --max-samples 200
```

If that works, switch to GPU and full-stat inference.

---

## Notes

- This workflow currently assumes our “flat” training name strings are correct and correspond to actual exported model directories searchable under `model_search_roots`.
- Natural-spectrum inference samples can absolutely live under `Training_Samples/` -- see above explanation.
