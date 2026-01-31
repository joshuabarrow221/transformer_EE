"""Feature correlation and permutation importance diagnostics.

This module provides a light-weight command line entry point that mirrors the
configuration-driven training pipeline used elsewhere in the project.  The goal
is to let analysts reuse existing JSON config files (with their definitions of
vector, scalar and target columns) to quickly

* build flattened feature tables for classical statistics such as the Pearson
  correlation coefficient, and
* reuse a *trained* model checkpoint to estimate permutation feature importance
  scores without writing another training loop.

Because many of the functions are reused in notebooks, each helper tries to
return Pandas / NumPy objects whenever possible, while keeping the computation
device-agnostic for portability between CPU and GPU environments.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformer_ee.dataloader.load import (
    get_sample_indices,
    get_sample_sizes,
    get_train_valid_test_dataloader,
)
from transformer_ee.dataloader.pd_dataset import string_to_float_list
from transformer_ee.model import create_model


@dataclass
class AggregationSpec:
    """Describe how to summarise sequence (per-prong) features.

    A ``vector`` feature in the training CSV is stored as a JSON-like string
    containing one value per prong element.  In order to compare those per-prong
    sequences with scalar features we have to reduce them to a single statistic
    per event.  ``AggregationSpec`` stores a human-readable name together with
    the callable that performs that reduction (e.g. ``np.mean`` or ``np.std``).
    """

    name: str
    func: callable


DEFAULT_AGGREGATIONS: Tuple[AggregationSpec, ...] = (
    AggregationSpec("mean", np.mean),
    AggregationSpec("std", np.std),
)


def _normalise_config(config: Dict) -> Dict:
    """Return a config copy with defaults filled for downstream utilities.

    Some downstream helpers (like ``get_sample_indices``) expect optional fields
    such as ``dataframe_type`` and ``seed`` to be present in the config.  When
    the user supplies an analysis-only config we therefore inject conservative
    defaults to avoid ``KeyError`` surprises without mutating the original
    dictionary.
    """

    cfg = dict(config)
    cfg.setdefault("dataframe_type", "pandas")
    cfg.setdefault("seed", 0)
    return cfg


def _train_subset(config: Dict) -> pd.DataFrame:
    """Return the training split as a pandas ``DataFrame``.

    The correlation analysis is intentionally anchored to the same distribution
    of events used during training, so the split logic mirrors
    ``get_train_valid_test_dataloader`` exactly.  For pandas-backed training we
    reuse ``get_sample_indices``.  For polars-backed training we shuffle and
    slice using the same polars calls before converting the subset to pandas for
    downstream numeric analysis.
    """

    cfg = _normalise_config(config)
    path = cfg["data_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not locate dataset: {path}")

    dataframe_type = cfg.get("dataframe_type", "pandas")
    if dataframe_type == "pandas":
        df = pd.read_csv(path)
        train_idx, _, _ = get_sample_indices(len(df), cfg)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        return train_df
    if dataframe_type == "polars":
        from transformer_ee.dataloader.pl_dataset import (  # pylint: disable=C0415
            get_polars_df_from_file,
        )

        df = get_polars_df_from_file(path)
        randomdf = df.sample(fraction=1.0, seed=cfg["seed"], shuffle=True)
        sizes = get_sample_sizes(randomdf.height, cfg)
        train_df = randomdf.slice(offset=0, length=sizes[0]).to_pandas()
        return train_df.reset_index(drop=True)
    raise ValueError(
        f"Unknown dataframe_type: {dataframe_type}. Supported types: ['pandas', 'polars']"
    )


def _sequence_to_array(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert string-encoded sequences into lists of floats in-place."""
    for name in columns:
        df[name] = df[name].apply(string_to_float_list)
    return df


def summarise_features(
    df: pd.DataFrame,
    vector_cols: Sequence[str],
    scalar_cols: Sequence[str],
    aggregations: Sequence[AggregationSpec] | None = None,
) -> pd.DataFrame:
    """Create a flat feature table suitable for correlation analysis.

    Parameters
    ----------
    df:
        DataFrame containing at least the columns referenced by ``vector_cols``
        and ``scalar_cols``.
    vector_cols / scalar_cols:
        Column names (as they appear in the training config) identifying
        sequence and scalar inputs respectively.
    aggregations:
        Optional list of :class:`AggregationSpec` controlling which statistics
        to compute for each sequence.  Defaults to ``mean`` and ``std``.

    Returns
    -------
    pandas.DataFrame
        A new table where each column is a numeric feature ready for classical
        statistics.  Per-prong sequences are summarised according to the chosen
        aggregations, while scalar columns are simply converted to floating
        point values.  Columns that cannot be converted or that are constant are
        dropped automatically because they would otherwise break correlation
        computations.
    """

    aggregations = tuple(aggregations or DEFAULT_AGGREGATIONS)
    df = df.copy()
    _sequence_to_array(df, vector_cols)

    # summarise per sequence
    feature_dict: Dict[str, List[float]] = {}
    for vec_name in vector_cols:
        seq_values = df[vec_name]
        for agg in aggregations:
            feat_name = f"{vec_name}__{agg.name}"
            feature_dict[feat_name] = [float(agg.func(v) if len(v) else float("nan")) for v in seq_values]

    # Scalars are already one value per event
    for sca_name in scalar_cols:
        feature_dict[sca_name] = pd.to_numeric(df[sca_name], errors="coerce").astype(float).tolist()

    feature_frame = pd.DataFrame(feature_dict)
    # Drop columns that are entirely NaN (e.g. empty sequences for all events)
    feature_frame = feature_frame.dropna(axis=1, how="all")
    # Remaining NaNs are typically caused by per-event missing values.  Replace
    # them by the column-wise mean to avoid holes in the correlation matrix.
    feature_frame = feature_frame.fillna(feature_frame.mean())

    # Drop constant columns which break Pearson correlation
    nunique = feature_frame.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index
    feature_frame = feature_frame.drop(columns=constant_cols)

    return feature_frame


def compute_target_correlations(
    feature_frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Return the Pearson correlation between each feature and each target."""
    combined = feature_frame.join(targets.reset_index(drop=True))
    corr_matrix = combined.corr(method="pearson")
    # Slice: rows=features, columns=targets
    corr_df = corr_matrix.loc[feature_frame.columns, targets.columns]
    return corr_df


def plot_correlation_heatmap(corr_df: pd.DataFrame, out_path: str):
    """Render ``corr_df`` as a heatmap saved to ``out_path``."""
    fig, ax = plt.subplots(figsize=(0.4 * len(corr_df.columns) + 3, 0.4 * len(corr_df.index) + 3))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Pearson correlation: features vs targets")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation coefficient")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _stack_batches(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0) if tensors else torch.empty(0)


def _collect_dataset(loader: torch.utils.data.DataLoader, max_samples: int | None = None):
    """Materialise all tensors from ``loader`` into memory.

    ``permutation_feature_importance`` repeatedly shuffles features; keeping the
    entire split in memory avoids re-running the DataLoader (and, by extension,
    any expensive on-the-fly augmentations).  ``max_samples`` offers a way to
    cap the workload for quick exploratory runs.
    """
    vecs: List[torch.Tensor] = []
    scas: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    wgts: List[torch.Tensor] = []

    total = 0
    for batch in loader:
        vec, sca, mask, tgt, wgt = batch
        batch_size = vec.shape[0]
        if max_samples is not None and total >= max_samples:
            break
        if max_samples is not None and total + batch_size > max_samples:
            take = max_samples - total
            vec = vec[:take]
            sca = sca[:take]
            mask = mask[:take]
            tgt = tgt[:take]
            wgt = wgt[:take]
            batch_size = take
        vecs.append(vec)
        scas.append(sca)
        masks.append(mask)
        tgts.append(tgt)
        wgts.append(wgt)
        total += batch_size

    vectors = _stack_batches(vecs)
    scalars = _stack_batches(scas)
    masks = _stack_batches(masks)
    targets = _stack_batches(tgts)
    weights = _stack_batches(wgts)

    return vectors, scalars, masks, targets, weights


def _batched_metric(
    model: torch.nn.Module,
    vectors: torch.Tensor,
    scalars: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    # Evaluation is always performed with gradients disabled to avoid autograd
    # bookkeeping overhead during repeated forward passes.
    model.eval()
    n_samples = vectors.shape[0]
    losses: List[float] = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            vec = vectors[start:end].to(device)
            sca = scalars[start:end].to(device)
            mask = masks[start:end].to(device)
            tgt = targets[start:end].to(device)
            wgt = weights[start:end].to(device)

            pred = model(vec, sca, mask)
            diff = (pred - tgt) ** 2
            if wgt.numel() > 0:
                diff = diff * wgt.view(-1, 1)
            losses.append(diff.mean().item())
    return float(np.mean(losses)) if losses else float("nan")


def permutation_feature_importance(
    model: torch.nn.Module,
    vectors: torch.Tensor,
    scalars: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Compute permutation feature importance scores for the provided tensors.

    The implementation follows the classic algorithm introduced by Breiman: the
    baseline mean-squared error is measured once, then each feature is randomly
    permuted across events and the degradation in loss is recorded.  By
    permuting instead of zeroing-out a column we preserve the marginal feature
    distribution, which reduces spurious artefacts for non-zero-mean inputs.
    """

    base_loss = _batched_metric(
        model, vectors, scalars, masks, targets, weights, device, batch_size
    )

    rng = np.random.default_rng(seed=0)
    importances: Dict[str, float] = {}

    # Scalars
    for idx in range(scalars.shape[1]):
        losses = []
        for _ in range(n_repeats):
            perm = rng.permutation(scalars.shape[0])
            perturbed = scalars.clone()
            perturbed[:, idx] = scalars[perm, idx]
            loss = _batched_metric(
                model,
                vectors,
                perturbed,
                masks,
                targets,
                weights,
                device,
                batch_size,
            )
            losses.append(loss)
        importances[f"scalar::{idx}"] = float(np.mean(losses) - base_loss)

    # Vector features (per prong component)
    for idx in range(vectors.shape[2]):
        losses = []
        for _ in range(n_repeats):
            perm = rng.permutation(vectors.shape[0])
            perturbed = vectors.clone()
            perturbed[:, :, idx] = vectors[perm, :, idx]
            loss = _batched_metric(
                model,
                perturbed,
                scalars,
                masks,
                targets,
                weights,
                device,
                batch_size,
            )
            losses.append(loss)
        importances[f"vector::{idx}"] = float(np.mean(losses) - base_loss)

    return importances


def plot_feature_importance(importances: Dict[str, float], out_path: str, feature_labels: Sequence[str], vector_labels: Sequence[str]):
    """Bar-plot helper used for visualising permutation importances."""
    labels: List[str] = []
    values: List[float] = []

    for key in sorted(importances.keys()):
        kind, index_str = key.split("::")
        idx = int(index_str)
        if kind == "scalar":
            label = f"scalar: {feature_labels[idx]}"
        elif kind == "vector":
            label = f"vector: {vector_labels[idx]}"
        else:
            label = key
        labels.append(label)
        values.append(importances[key])

    fig, ax = plt.subplots(figsize=(0.5 * len(labels) + 3, 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ loss (MSE)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _feature_labels(config: Dict) -> Tuple[List[str], List[str]]:
    """Extract scalar and vector feature labels from ``config``."""
    scalars = list(config["scalar"])
    vectors = list(config["vector"])
    return scalars, vectors


def run_analysis(args):
    """Entry point shared by the CLI and potential notebook wrappers."""
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    vector_cols = config["vector"]
    scalar_cols = config["scalar"]
    target_cols = config["target"]

    # Correlations are computed on the training split to mirror the feature
    # distribution seen during optimization.  If you want to probe generalised
    # behaviour instead, consider re-running this analysis on a held-out split
    # by adapting this helper.
    train_df = _train_subset(config)
    feature_frame = summarise_features(train_df, vector_cols, scalar_cols)

    targets = train_df[target_cols].apply(pd.to_numeric, errors="coerce")
    corr_df = compute_target_correlations(feature_frame, targets)

    os.makedirs(args.output_dir, exist_ok=True)
    corr_path = os.path.join(args.output_dir, "pearson_feature_target_correlation.png")
    plot_correlation_heatmap(corr_df, corr_path)
    corr_df.to_csv(os.path.join(args.output_dir, "pearson_feature_target_correlation.csv"))

    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(config)

        # ``torch.load`` reads the weights stored in the checkpoint file and
        # returns the ``state_dict`` created by the trainer.  No optimisation or
        # additional training happens here: we simply restore the model to the
        # exact state at which the checkpoint was saved (typically the best
        # validation epoch or the final epoch depending on user choice).
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        trainloader, validloader, testloader, _ = get_train_valid_test_dataloader(config)
        loader = {"train": trainloader, "valid": validloader, "test": testloader}[args.split]
        vectors, scalars, masks, targets, weights = _collect_dataset(loader, max_samples=args.max_samples)
        if vectors.numel() == 0:
            raise RuntimeError("No samples collected for permutation importance analysis.")

        importances = permutation_feature_importance(
            model,
            vectors,
            scalars,
            masks,
            targets,
            weights,
            device,
            batch_size=args.batch_size,
            n_repeats=args.n_repeats,
        )
        scalar_labels, vector_labels = _feature_labels(config)
        plot_feature_importance(
            importances,
            os.path.join(args.output_dir, "permutation_feature_importance.png"),
            scalar_labels,
            vector_labels,
        )
        pd.Series(importances).to_csv(
            os.path.join(args.output_dir, "permutation_feature_importance.csv"),
            header=["delta_loss"],
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Feature correlation diagnostics for transformer_EE")
    parser.add_argument("--config", required=True, help="Path to a training config JSON file")
    parser.add_argument(
        "--checkpoint",
        help="Optional path to a trained model checkpoint (.zip) for permutation importance",
    )
    parser.add_argument("--output-dir", default="analysis_output", help="Directory to store plots and tables")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="valid",
        help="Dataset split to use when computing permutation importance",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation during importance calculation")
    parser.add_argument("--n-repeats", type=int, default=5, help="Number of shuffles per feature for permutation importance")
    parser.add_argument("--max-samples", type=int, help="Optional cap on the number of samples used for permutation importance")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
