"""Per-event feature importance for trained transformer_ee checkpoints.

Motivation
----------
``feature_diagnostics.py`` answers *global* questions: "averaged over the test
split, how much does the network rely on ``particle.calE``?".  This script
answers the *local* version: "for THIS event, which inputs actually moved the
prediction, by how many GeV, and in which direction?"  That distinction matters
because a feature can be irrelevant for 95% of events and decisive for the
5% that sit in some corner of phase space -- a global permutation score
averages that structure away, while the per-event table below preserves it.

Method
------
Two complementary sensitivity measures, both computed on the trained network
with NO retraining and NO gradients:

* ``occlusion`` (default, deterministic):  replace one feature with the
  training-set mean and record the signed change in the prediction.  Because
  the pipeline z-scores every input with train-set statistics
  (``pd_dataset.normalize``), "training mean" is exactly 0 in network space,
  so occlusion is literally zeroing one channel.  We report

      contrib(event, feature, target)
          = pred(all features) - pred(feature knocked out)

  i.e. how far the feature's actual value pushed the prediction, relative to
  an average-valued stand-in.  Positive = pushed the energy estimate up.
  Contributions are in physical target units (targets are never normalised in
  this pipeline, so predictions come out in e.g. GeV).

* ``permutation`` (stochastic): swap the feature's value with that of a
  random other event, ``--n-repeats`` times, and record the mean |change| in
  prediction.  Unsigned, but respects the feature's marginal distribution
  instead of collapsing it to the mean.  Use it as a cross-check when a
  feature's distribution is wildly non-Gaussian and "mean value" is a
  strange counterfactual (e.g. ``particle.is_shower``, which is binary).

Every event row also carries context columns: the truth target(s), the
baseline prediction(s), the absolute error, the prong multiplicity, and the
per-event z-scores of each input feature (how unusual each input was).  Sort
by ``abs_err__<target>`` and read across: you immediately see *which* inputs
were extreme and *which* the network actually listened to for its worst
events.

Checkpoint requirements
-----------------------
Point ``--model-dir`` at a trainer output directory.  Only two files are
needed: ``input.json`` (config) and ``best_model.zip`` (state_dict).
``trainset_stat.json`` is NOT required -- normalisation statistics are
recomputed from the raw CSV via the identical seeded pipeline, which is handy
for recovered checkpoints where that file was lost.  If ``input.json`` is the
missing file instead, supply a reconstructed config with ``--config`` plus the
weights with ``--checkpoint``.

Examples
--------
Occlusion on the first 20k test events, plus a breakdown plot of event 137::

    python per_event_importance.py --model-dir save/model/dune/NC/model_abc \
        --event 137 --output-dir per_event_out

Permutation-based cross-check, 8 repeats, whole test split::

    python per_event_importance.py --model-dir ... --mode permutation \
        --n-repeats 8 --max-events -1
"""
from __future__ import annotations


# ===========================================================================
# READING GUIDE
# ===========================================================================
#
# This file answers a LOCAL interpretability question:
#
#     "For one event, how much did each input feature move each model output?"
#
# Complete workflow:
#
#   1. Read the model configuration and checkpoint.
#   2. Rebuild the requested validation/test split.
#   3. Normalize input features using training-set statistics.
#   4. Run the untouched event through the network to get a baseline prediction.
#   5. Perturb one complete feature channel at a time.
#   6. Rerun inference and compare with the baseline prediction.
#   7. Save one row per event and produce aggregate/event-level plots.
#
# Two supported perturbation methods:
#
#   OCCLUSION
#       Replace a normalized feature channel with zero.
#
#       Because the inputs are z-scored, zero represents the training-set mean.
#       The signed contribution is:
#
#           baseline prediction - prediction with feature removed
#
#       A positive value means the real feature value pushed the prediction up.
#       A negative value means the real feature value pushed the prediction down.
#
#   PERMUTATION
#       Replace a feature with the value from another randomly selected event.
#       The reported quantity is the mean absolute prediction change over repeats.
#
# Important interpretation limits:
#
#   * These are model-sensitivity values, not causal physical effects.
#   * Feature effects are not additive because the neural network is nonlinear.
#   * Correlated features can substitute for one another.
#   * Topology is retained in the output table if the checkpoint produces it,
#     but it is excluded from plots because it is only a placeholder here.
#
# Tensor naming used throughout:
#
#   vector : particle/prong-level features, shape (events, prongs, features)
#   scalar : one scalar feature vector per event
#   mask   : True for padded/nonphysical prong slots
#   target : physical truth values
#   pred   : model outputs in physical target units
#
# ===========================================================================


import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless-safe: render straight to PNG, never a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_ee.dataloader.load import get_sample_indices
from transformer_ee.dataloader.pd_dataset import (
    Normalized_pandas_Dataset_with_cache,
)
from transformer_ee.model import create_model
from transformer_ee.utils import get_gpu
from transformer_ee.utils.weights import NullWeights


# ---------------------------------------------------------------------------
# Checkpoint / data loading
# ---------------------------------------------------------------------------
def load_config_and_model(args: argparse.Namespace) -> Tuple[Dict, torch.nn.Module]:
    """Resolve (config, model-with-weights) from the CLI arguments.

    Two entry paths:
    * ``--model-dir``: read ``input.json`` and ``best_model.zip`` from the
      trainer's output directory (mirrors inference/load_model_checkpoint,
      minus the trainset_stat.json dependency).
    * ``--config`` + ``--checkpoint``: explicit files, for checkpoints whose
      directory structure did not survive.
    """
    if args.model_dir:
        cfg_path = os.path.join(args.model_dir, "input.json")
        ckpt_path = os.path.join(args.model_dir, "best_model.zip")
    else:
        if not (args.config and args.checkpoint):
            raise SystemExit(
                "Provide either --model-dir, or both --config and --checkpoint."
            )
        cfg_path, ckpt_path = args.config, args.checkpoint

    with open(cfg_path, encoding="UTF-8") as fh:
        config = json.load(fh)

    # Analysis-only defaults; nothing on disk is modified.
    config.setdefault("dataframe_type", "pandas")
    config.setdefault("seed", 0)
    # Worker processes buy nothing for a single sequential pass and are a
    # reliable source of pain on Windows (spawn semantics); force 0 unless
    # the user asks otherwise.
    config["num_workers"] = args.num_workers

    if args.data_path:
        # Convenience override: configs often carry cluster paths; this lets
        # you point the same checkpoint at a local copy of the CSV.
        config["data_path"] = args.data_path

    # Analysis-only metadata. These keys are ignored by create_model.
    config["_analysis_stats_path"] = args.stats
    config["_analysis_config_path"] = cfg_path
    config["dataframe_type"] = "pandas"

    # Build architecture from config, then load weights.  map_location="cpu"
    # so CUDA-trained checkpoints load on CPU-only machines; the caller moves
    # the model to the compute device afterwards.
    model = create_model(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return config, model


def _finite_mean_std(values: np.ndarray, feature_name: str) -> List[float]:
    """Finite-only training mean/std for one input feature."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError(
            f"Feature {feature_name!r} has no finite values in the train split."
        )

    mean = float(np.mean(values))
    std = float(np.std(values))

    if not np.isfinite(std) or std < 1.0e-10:
        print(
            f"[warn] {feature_name}: invalid or tiny training std={std}; "
            "using std=1.0"
        )
        std = 1.0

    return [mean, std]


def _compute_finite_stats(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    vector_names: Sequence[str],
    scalar_names: Sequence[str],
) -> Dict[str, List[float]]:
    """Compute normalization statistics while ignoring NaN/Inf values."""
    stat: Dict[str, List[float]] = {}

    for name in vector_names:
        series = dataframe[name].iloc[train_indices]
        valid_text = series.map(
            lambda value: isinstance(value, str) and len(value.strip()) > 0
        )
        exploded = (
            series.where(valid_text, "")
            .str.split(",")
            .explode()
        )
        values = pd.to_numeric(exploded, errors="coerce").to_numpy(np.float64)
        stat[name] = _finite_mean_std(values, name)

    for name in scalar_names:
        values = pd.to_numeric(
            dataframe[name].iloc[train_indices], errors="coerce"
        ).to_numpy(np.float64)
        stat[name] = _finite_mean_std(values, name)

    return stat


def _load_or_compute_stats(
    config: Dict,
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
) -> Dict:
    """Use an existing finite stats file, otherwise recompute robustly."""
    feature_names = list(config["vector"]) + list(config["scalar"])
    stat_path = config.get("_analysis_stats_path")

    if stat_path and os.path.isfile(stat_path):
        with open(stat_path, encoding="UTF-8") as handle:
            stat = json.load(handle)
        source = stat_path
    else:
        print(
            "[warn] No usable trainset_stat.json was supplied; "
            "recomputing finite-only statistics from the training split."
        )
        stat = _compute_finite_stats(
            dataframe,
            train_indices,
            list(config["vector"]),
            list(config["scalar"]),
        )
        source = "finite-only recomputation"

    for name in feature_names:
        if name not in stat:
            raise RuntimeError(
                f"Normalization statistics are missing feature {name!r}."
            )

        pair = stat[name]
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            raise RuntimeError(f"Malformed statistics for {name!r}: {pair!r}")

        mean = float(pair[0])
        std = float(pair[1])

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise RuntimeError(
                f"Invalid normalization statistics for {name!r}: "
                f"mean={mean}, std={std}"
            )

    print(f"[per-event] normalization statistics source: {source}")
    return stat


def materialise_split(
    config: Dict, split: str, max_events: int | None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Materialize a deterministic pandas split with finite normalization."""

    if split == "train":
        raise SystemExit(
            "Per-event analysis is restricted to test or valid splits."
        )

    data_path = config.get("data_path")
    if not data_path or not os.path.isfile(data_path):
        raise RuntimeError(f"CSV does not exist: {data_path}")

    vector_names = list(config["vector"])
    scalar_names = list(config["scalar"])
    target_names = list(config["target"])

    required = vector_names + scalar_names + target_names

    print(f"[per-event] reading {data_path} ...")
    dataframe = pd.read_csv(
        data_path,
        dtype={name: str for name in vector_names},
    )

    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")

    train_idx, valid_idx, test_idx = get_sample_indices(len(dataframe), config)
    split_idx = {
        "valid": np.asarray(valid_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }[split]

    if max_events is not None:
        split_idx = split_idx[:max_events]

    stat = _load_or_compute_stats(config, dataframe, np.asarray(train_idx))

    selected = dataframe.iloc[split_idx].reset_index(drop=True).copy()
    dataset = Normalized_pandas_Dataset_with_cache(
        config,
        selected,
        weighter=NullWeights(),
    )
    dataset.normalize(stat)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
    )

    vecs: List[torch.Tensor] = []
    scas: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []

    for vector, scalar, mask, target, _weight in loader:
        vecs.append(vector.float())
        scas.append(scalar.float())
        masks.append(mask.bool())
        tgts.append(target.float())

    vector_t = torch.cat(vecs)
    scalar_t = torch.cat(scas)
    mask_t = torch.cat(masks)
    target_t = torch.cat(tgts)
    csv_rows = split_idx.copy()

    # Report and remove events containing invalid normalized model inputs.
    vector_bad = ~torch.isfinite(vector_t).reshape(vector_t.shape[0], -1).all(dim=1)
    scalar_bad = ~torch.isfinite(scalar_t).all(dim=1)
    bad_inputs = vector_bad | scalar_bad

    if bad_inputs.any():
        count = int(bad_inputs.sum().item())
        print(
            f"[clean] dropping {count}/{vector_t.shape[0]} events with "
            "NaN/Inf normalized inputs"
        )
        keep = ~bad_inputs
        vector_t = vector_t[keep]
        scalar_t = scalar_t[keep]
        mask_t = mask_t[keep]
        target_t = target_t[keep]
        csv_rows = csv_rows[keep.numpy()]

    if vector_t.shape[0] == 0:
        raise RuntimeError("No finite-input events remain after cleaning.")

    # Padded values are inert, but explicitly zero them for clean perturbations.
    vector_t = vector_t.masked_fill(mask_t.unsqueeze(-1), 0.0)

    # MultiheadAttention can generate NaN when every token in an event is masked.
    all_masked = mask_t.all(dim=1)
    if all_masked.any():
        count = int(all_masked.sum().item())
        print(
            f"[clean] {count} events had all prongs masked; "
            "inserting one zero-valued dummy prong"
        )
        mask_t[all_masked, 0] = False
        vector_t[all_masked, 0, :] = 0.0

    print(
        "[check] finite normalized tensors: "
        f"vector={bool(torch.isfinite(vector_t).all())}, "
        f"scalar={bool(torch.isfinite(scalar_t).all())}"
    )

    return vector_t, scalar_t, mask_t, target_t, csv_rows


@torch.no_grad()
def predict_in_chunks(
    model: torch.nn.Module,
    vector: torch.Tensor,
    scalar: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    chunk: int = 4096,
) -> torch.Tensor:
    """model(vector, scalar, mask) over the whole split, chunked for memory.

    Inputs/outputs live on CPU; only the active chunk visits the device.
    """
    outs: List[torch.Tensor] = []
    for lo in range(0, vector.shape[0], chunk):
        hi = lo + chunk
        outs.append(
            model(
                vector[lo:hi].to(device),
                scalar[lo:hi].to(device),
                mask[lo:hi].to(device),
            ).cpu()
        )
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Per-event sensitivity
# ---------------------------------------------------------------------------
def per_event_deltas(
    model: torch.nn.Module,
    vector: torch.Tensor,
    scalar: torch.Tensor,
    mask: torch.Tensor,
    baseline_pred: torch.Tensor,
    feature_names: Sequence[str],
    n_vector: int,
    mode: str,
    device: torch.device,
    n_repeats: int,
    seed: int,
    chunk: int,
) -> np.ndarray:
    """Core loop: perturb one feature at a time, diff against the baseline.

    Returns an array of shape (n_events, n_features, n_targets):

    * ``occlusion``   -> SIGNED  contrib = baseline - knocked_out.
      One extra forward pass per feature; fully deterministic.
    * ``permutation`` -> UNSIGNED mean |perturbed - baseline| over repeats.
      n_repeats forward passes per feature.

    Perturbation granularity is one whole channel per event: for a vector
    feature the event's entire prong sequence in that channel is zeroed /
    swapped (padding included -- padded positions never reach the attention
    thanks to src_key_padding_mask, so their content is inert); for a scalar
    it is the single value.
    """
    n_events = vector.shape[0]
    n_targets = baseline_pred.shape[1]
    deltas = np.zeros((n_events, len(feature_names), n_targets), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for f_idx, f_name in enumerate(feature_names):
        if mode == "occlusion":
            if f_idx < n_vector:
                v_mod = vector.clone()
                v_mod[:, :, f_idx] = 0.0  # 0 in z-space == training mean.
                pred = predict_in_chunks(model, v_mod, scalar, mask, device, chunk)
            else:
                s_mod = scalar.clone()
                s_mod[:, f_idx - n_vector] = 0.0
                pred = predict_in_chunks(model, vector, s_mod, mask, device, chunk)
            # Signed: "the feature's actual value moved the prediction by
            # this much, versus an average-valued stand-in".
            # This is the signed event-level occlusion contribution:
            #
            #   positive -> the real feature pushed the output upward
            #   negative -> the real feature pushed the output downward
            #
            # It is measured in the physical units of each target.
            deltas[:, f_idx, :] = (baseline_pred - pred).numpy()
        elif mode == "permutation":
            acc = np.zeros((n_events, n_targets), dtype=np.float64)
            for _rep in range(n_repeats):
                perm = torch.from_numpy(rng.permutation(n_events))
                if f_idx < n_vector:
                    v_mod = vector.clone()
                    v_mod[:, :, f_idx] = vector[perm, :, f_idx]
                    pred = predict_in_chunks(
                        model, v_mod, scalar, mask, device, chunk
                    )
                else:
                    s_mod = scalar.clone()
                    s_idx = f_idx - n_vector
                    s_mod[:, s_idx] = scalar[perm, s_idx]
                    pred = predict_in_chunks(
                        model, vector, s_mod, mask, device, chunk
                    )
                acc += (pred - baseline_pred).abs().numpy()
            deltas[:, f_idx, :] = acc / n_repeats
        else:
            raise ValueError(f"Unknown mode: {mode}")
        print(f"[per-event] {mode} done for {f_name}")

    return deltas


def build_event_table(
    config: Dict,
    vector: torch.Tensor,
    scalar: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    baseline_pred: torch.Tensor,
    deltas: np.ndarray,
    csv_rows: np.ndarray,
    split: str,
    mode: str,
) -> pd.DataFrame:
    """Assemble the one-row-per-event diagnostic table.

    Column groups, in order:
    * bookkeeping : ``event`` (ordinal within the split, matches --event),
                    ``csv_row`` (row in the source CSV; -1 if unknown),
                    ``split``, ``n_prongs``
    * truth/pred  : ``target__<t>``, ``pred__<t>``, ``abs_err__<t>``
    * z-context   : ``z__<feature>`` -- how unusual each input was for this
                    event, in train-set standard deviations.  For vector
                    features this is the mean over REAL prongs of the
                    z-scored channel (padding excluded via the mask); scalars
                    are already z-scores.
    * sensitivity : occlusion  -> ``contrib__<feature>__<t>``  (signed GeV)
                    permutation-> ``sens__<feature>__<t>``     (mean |dGeV|)

    Everything a "why is event X mispredicted?" investigation needs lives in
    one row: sort by ``abs_err``, then scan z (was some input extreme?) and
    contrib (did the network act on it?).
    """
    target_names = list(config["target"])
    feature_names = list(config["vector"]) + list(config["scalar"])
    n_vector = len(config["vector"])
    n_events = vector.shape[0]

    real = (~mask).to(torch.float64)                     # (N, L): 1 = real prong
    n_prongs = real.sum(dim=1)                           # (N,)
    safe_n = torch.clamp(n_prongs, min=1.0)              # avoid 0-division on
                                                         # empty (all-pad) events

    cols: Dict[str, np.ndarray] = {
        "event": np.arange(n_events, dtype=np.int64),
        "csv_row": csv_rows,
        "split": np.full(n_events, split),
        "n_prongs": n_prongs.numpy(),
    }
    for t_idx, t_name in enumerate(target_names):
        truth = target[:, t_idx].numpy().astype(np.float64)
        pred = baseline_pred[:, t_idx].numpy().astype(np.float64)

        truth_valid = np.isfinite(truth) & (np.abs(truth) <= 1.0e5)
        truth_clean = np.where(truth_valid, truth, np.nan)
        abs_error = np.where(truth_valid, np.abs(pred - truth), np.nan)

        cols[f"target__{t_name}"] = truth_clean
        cols[f"pred__{t_name}"] = pred
        cols[f"abs_err__{t_name}"] = abs_error

    for f_idx, f_name in enumerate(feature_names):
        if f_idx < n_vector:
            # Masked mean over real prongs of the already-z-scored channel.
            z = (vector[:, :, f_idx].to(torch.float64) * real).sum(dim=1) / safe_n
        else:
            z = scalar[:, f_idx - n_vector].to(torch.float64)
        cols[f"z__{f_name}"] = z.numpy()

    prefix = "contrib" if mode == "occlusion" else "sens"
    for f_idx, f_name in enumerate(feature_names):
        for t_idx, t_name in enumerate(target_names):
            cols[f"{prefix}__{f_name}__{t_name}"] = deltas[:, f_idx, t_idx]

    return pd.DataFrame(cols)


def _targets_for_plots(config: Dict) -> List[str]:
    """Return scientifically meaningful targets for visualization.

    The checkpoint still contains a Topology output neuron, so its prediction is
    preserved in per_event_importance.csv. However, Topology is only a placeholder
    in this atmospheric sample and does not enter the loss, so plotting its
    sensitivity would invite a misleading interpretation.
    """
    return [name for name in config["target"] if str(name).lower() != "topology"]



# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_summary(
    deltas: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    mode: str,
    out_dir: str,
) -> None:
    """Two split-level views of the per-event matrix.

    1. ``summary_mean_abs.png``: mean |delta| per feature, one bar group per
       target.  This is the bridge back to the global permutation-importance
       ranking -- the two orderings should broadly agree.
    2. ``per_event_spread.png``: log-scale box plot of |delta| across events
       (first target).  This is the plot the global number cannot give you:
       a long upper whisker on an otherwise small box is the signature of a
       feature that only matters for a subpopulation of events.
    """
    abs_d = np.abs(deltas)
    order = np.argsort(abs_d.mean(axis=(0, 2)))          # rank by overall mean

    # --- 1. mean |delta| bars ---------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8.0, max(4.0, 0.4 * len(feature_names)))
    )
    height = 0.8 / len(target_names)
    y = np.arange(len(feature_names), dtype=np.float64)
    for t_idx, t_name in enumerate(target_names):
        ax.barh(
            y + t_idx * height,
            abs_d[:, order, t_idx].mean(axis=0),
            height=height,
            label=t_name,
        )
    ax.set_yticks(y + 0.4 - height / 2)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel(
        "mean |prediction shift| when feature is "
        + ("occluded" if mode == "occlusion" else "permuted")
        + " (target units)"
    )
    ax.set_title(f"Per-event sensitivity, split average ({mode})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_mean_abs.png"), dpi=300)
    plt.close(fig)

    # --- 2. event-to-event spread (first target) --------------------------
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.5 * len(feature_names)), 5.0)
    )
    data = [abs_d[:, i, 0] for i in order[::-1]]         # biggest first
    ax.boxplot(data, showfliers=False)                   # fliers: millions of
    ax.set_yscale("log")                                 # dots, no information
    ax.set_xticklabels(
        [feature_names[i] for i in order[::-1]], rotation=90, fontsize=8
    )
    ax.set_ylabel(f"|prediction shift| on {target_names[0]} (log)")
    ax.set_title("Event-to-event spread of feature sensitivity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_event_spread.png"), dpi=300)
    plt.close(fig)


def plot_single_event(
    table: pd.DataFrame,
    event: int,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    mode: str,
    out_dir: str,
) -> None:
    """Bar-chart breakdown for one event: the "why THIS prediction" plot.

    One horizontal bar per feature and target.  In occlusion mode bars are
    signed: right = this input pushed the estimate up, left = down; the
    per-feature z-score is annotated so you can see at a glance whether a big
    push came from an unusual input or from an ordinary one the network is
    simply very sensitive to.
    """
    row = table.loc[table["event"] == event]
    if row.empty:
        raise SystemExit(
            f"--event {event} is outside the materialised range "
            f"(0..{table['event'].max()}); raise --max-events."
        )
    row = row.iloc[0]
    prefix = "contrib" if mode == "occlusion" else "sens"

    fig, axes = plt.subplots(
        1,
        len(target_names),
        figsize=(6.0 * len(target_names), max(4.0, 0.35 * len(feature_names))),
        squeeze=False,
    )
    for t_idx, t_name in enumerate(target_names):
        ax = axes[0][t_idx]
        vals = np.array(
            [row[f"{prefix}__{f}__{t_name}"] for f in feature_names]
        )
        order = np.argsort(np.abs(vals))                 # biggest on top
        ax.barh(
            np.arange(len(feature_names)),
            vals[order],
            color=["#b04a3b" if v < 0 else "#3b6aa0" for v in vals[order]],
        )
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(
            [
                # e.g. "particle.calE  (z=+2.31)"
                f"{feature_names[i]}  (z={row[f'z__{feature_names[i]}']:+.2f})"
                for i in order
            ],
            fontsize=8,
        )
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(
            ("signed contribution" if mode == "occlusion" else "mean |shift|")
            + " (target units)"
        )
        ax.set_title(
            f"{t_name}\ntruth={row[f'target__{t_name}']:.3f}  "
            f"pred={row[f'pred__{t_name}']:.3f}"
        )
    fig.suptitle(
        f"Event {event} (csv_row={int(row['csv_row'])}, "
        f"n_prongs={int(row['n_prongs'])}) -- {mode} breakdown"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"event_{event}_breakdown.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Trainer output dir containing input.json + best_model.zip.",
    )
    parser.add_argument("--config", default=None, help="Config JSON (alt path).")
    parser.add_argument(
        "--checkpoint", default=None, help="state_dict path (alt path)."
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Optional trainset_stat.json. If omitted, finite-only statistics "
        "are recomputed from the CSV train split.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override config['data_path'] (e.g. local copy of the CSV).",
    )
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument(
        "--mode",
        choices=("occlusion", "permutation"),
        default="occlusion",
        help="occlusion: signed, deterministic, 1 pass/feature. "
        "permutation: unsigned, distribution-respecting, n-repeats passes.",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5, help="Permutation mode only."
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20000,
        help="Events to analyse (leading block of the unshuffled split; "
        "deterministic). -1 = all events; watch RAM on huge splits.",
    )
    parser.add_argument(
        "--event",
        type=int,
        default=None,
        help="Also render a breakdown plot for this split-ordinal event.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4096, help="Forward batch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="per_event_out")
    args = parser.parse_args()

    max_events = None if args.max_events == -1 else args.max_events
    os.makedirs(args.output_dir, exist_ok=True)

    config, model = load_config_and_model(args)
    device = get_gpu()  # cuda > mps > cpu, same helper the trainer uses.
    model.to(device)

    print(f"[per-event] materialising '{args.split}' split ...")
    vector, scalar, mask, target, csv_rows = materialise_split(
        config, args.split, max_events
    )
    print(f"[per-event] {vector.shape[0]} events materialised.")

    baseline_pred = predict_in_chunks(
        model, vector, scalar, mask, device, args.chunk
    )

    if not torch.isfinite(baseline_pred).all():
        bad = int((~torch.isfinite(baseline_pred)).sum().item())
        raise RuntimeError(
            f"Baseline predictions contain {bad} NaN/Inf values. "
            "Refusing to generate misleading empty plots."
        )

    for target_index, target_name in enumerate(config["target"]):
        values = baseline_pred[:, target_index].numpy()
        print(
            f"[check] predicted {target_name}: "
            f"mean={np.mean(values):.6g}, "
            f"std={np.std(values):.6g}, "
            f"min={np.min(values):.6g}, "
            f"max={np.max(values):.6g}"
        )

    feature_names = list(config["vector"]) + list(config["scalar"])
    deltas = per_event_deltas(
        model=model,
        vector=vector,
        scalar=scalar,
        mask=mask,
        baseline_pred=baseline_pred,
        feature_names=feature_names,
        n_vector=len(config["vector"]),
        mode=args.mode,
        device=device,
        n_repeats=args.n_repeats,
        seed=args.seed,
        chunk=args.chunk,
    )

    if not np.isfinite(deltas).all():
        bad = int((~np.isfinite(deltas)).sum())
        raise RuntimeError(
            f"Occlusion/permutation output contains {bad} NaN/Inf values. "
            "No plots were written."
        )

    max_shift = float(np.max(np.abs(deltas)))
    mean_shift = float(np.mean(np.abs(deltas)))
    print(
        f"[check] sensitivity shifts: mean_abs={mean_shift:.6g}, "
        f"max_abs={max_shift:.6g}"
    )

    if max_shift == 0.0:
        raise RuntimeError(
            "Every sensitivity shift is exactly zero. "
            "The model/input mapping must be checked."
        )

    table = build_event_table(
        config,
        vector,
        scalar,
        mask,
        target,
        baseline_pred,
        deltas,
        csv_rows,
        args.split,
        args.mode,
    )
    csv_path = os.path.join(args.output_dir, "per_event_importance.csv")
    table.to_csv(csv_path, index=False)
    print(f"[per-event] wrote {csv_path}  ({len(table)} rows)")

    plot_targets = _targets_for_plots(config)

    if not plot_targets:
        raise RuntimeError("No non-topology targets remain to plot.")

    target_indices = [list(config["target"]).index(name) for name in plot_targets]
    deltas_for_plot = deltas[:, :, target_indices]

    plot_summary(
        deltas_for_plot, feature_names, plot_targets, args.mode, args.output_dir
    )
    if args.event is not None:
        plot_single_event(
            table,
            args.event,
            feature_names,
            plot_targets,
            args.mode,
            args.output_dir,
        )
    print(f"[per-event] plots written to {args.output_dir}")


if __name__ == "__main__":
    # Required on Windows whenever num_workers > 0 (spawned workers re-import
    # this file); harmless everywhere else.
    main()