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

# `annotations` makes every type hint in this file lazily evaluated (PEP 563).
# Practical consequence: we can use nicer hints (e.g. ``callable`` in the
# dataclass below, or forward references) without paying an import-time cost,
# and the file stays importable on older 3.x interpreters used on clusters.
from __future__ import annotations

# --- Standard library ---------------------------------------------------------
import argparse  # CLI parsing for the __main__ entry point at the bottom.
import json      # Training configs are plain JSON files (see config/*.json).
import os        # Path checks + output-directory creation.
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# --- Third party --------------------------------------------------------------
import matplotlib

# Force a non-interactive backend *before* pyplot is imported.  These scripts
# frequently run over SSH / inside PowerShell jobs with no display attached;
# "Agg" renders straight to PNG files and never tries to open a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# --- Project ------------------------------------------------------------------
# We deliberately import the *same* helpers the training pipeline uses so that
# every split / parsing decision here is bit-for-bit identical to training:
#
#   get_sample_indices           -> pandas path: shuffles np.arange(N) with
#                                   config["seed"] and carves train/valid/test.
#   get_sample_sizes             -> polars path: only computes the three split
#                                   *sizes*; polars does its own seeded shuffle.
#   get_train_valid_test_dataloader
#                                -> builds normalized Dataset objects + loaders;
#                                   returns (train, valid, test, train_stat).
#   string_to_float_list         -> parses a vector cell ("1.0,2.0,...") into a
#                                   Python list of floats; returns [0] for
#                                   missing/non-string cells.
#   create_model                 -> instantiates the architecture named in
#                                   config["model"] (e.g. Transformer_EE_MV).
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

    # NOTE: in practice the cell format is *comma separated* text; it is parsed
    # by ``string_to_float_list`` (a simple ``str.split(",")``), not by a JSON
    # decoder.  Empty / NaN cells decode to ``[0]`` so every aggregation below
    # is guaranteed a non-empty input and can never raise on an empty slice.
    name: str        # Suffix used for output columns, e.g. "mean" -> "px__mean"
    func: callable   # Reduction ndarray -> float. ``callable`` (lowercase) is
                     # fine here: dataclasses only store annotations, they never
                     # instantiate them, so the builtin works as documentation.


# Two moments are usually all you need for a first correlation pass:
#   mean -> the typical per-prong value of the event,
#   std  -> how spread out the prongs are within the event (0 for 1-prong
#           events, which is a feature in itself).
# Extend from a notebook via e.g.
#   DEFAULT_AGGREGATIONS + (AggregationSpec("max", np.max),)
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
    # ``dict(config)`` is a *shallow* copy: enough to add top-level keys without
    # side effects on the caller's dict; nested dicts (model kwargs, ...) are
    # shared, which is fine because nothing below mutates them.
    cfg = dict(config)
    # "pandas" matches get_train_valid_test_dataloader's own default, so an
    # analysis config that omits the key sees the exact same code path as
    # training did.
    cfg.setdefault("dataframe_type", "pandas")
    # seed=0 is the value shipped in every config/input_*.json; using the same
    # default keeps the reproduced split identical to a training run that
    # relied on the shipped configs.  If your training config carried a
    # different seed, it is picked up from `config` and this default is inert.
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
        # Fail fast with the offending path in the message; the default pandas
        # error for a missing file is easy to misread when configs are shared
        # across machines with different mount points (e.g. D:\ vs /mnt).
        raise FileNotFoundError(f"Could not locate dataset: {path}")

    dataframe_type = cfg.get("dataframe_type", "pandas")
    if dataframe_type == "pandas":
        # pandas path (mirrors load.py):
        #   1. read the *whole* CSV,
        #   2. shuffle indices with np.random.seed(cfg["seed"]),
        #   3. keep the leading block as train.
        # Because get_sample_indices seeds NumPy's global RNG internally, the
        # returned index arrays are identical to the ones training used --
        # this is the property the whole analysis leans on.
        df = pd.read_csv(path)
        train_idx, _, _ = get_sample_indices(len(df), cfg)
        # .reset_index(drop=True) gives the subset a clean 0..N-1 index so
        # later positional ops (corr tables, plots) can't accidentally align
        # on the shuffled original row labels.
        train_df = df.iloc[train_idx].reset_index(drop=True)

    # =========================================================================
    # RECONSTRUCTED FROM HERE DOWN
    # -------------------------------------------------------------------------
    # Your paste cut off at the line above (inside the pandas branch of
    # ``_train_subset``).  Everything below was rebuilt to match (a) the
    # module docstring's stated scope, (b) this file's import list -- every
    # import above is used exactly where you'd expect -- and (c) the *actual*
    # upstream transformer_ee APIs (load.py, pd_dataset.py, model_maker.py),
    # which I checked against the repo.  Diff this region against your branch
    # copy before pushing; behaviour should match, comments are new.
    # =========================================================================
    elif dataframe_type == "polars":
        # polars path (mirrors load.py's polars branch line-for-line):
        # training shuffles the *rows* with polars' own seeded sampler, then
        # slices by size -- there are no index arrays to reuse, so we replay
        # the identical shuffle + slice and only then convert to pandas.
        # Lazy import (as upstream does) so pandas-only users never need
        # polars installed.
        from transformer_ee.dataloader.pl_dataset import (  # pylint: disable=C0415
            get_polars_df_from_file,
        )

        pl_df = get_polars_df_from_file(path)
        # fraction=1.0 + shuffle=True == full-frame seeded permutation; the
        # seed is the same one training used, so row order matches training.
        randomdf = pl_df.sample(fraction=1.0, seed=cfg["seed"], shuffle=True)
        n_train, _, _ = get_sample_sizes(randomdf.height, cfg)
        # Training takes slice(0, n_train) as the train set -> so do we.
        train_df = randomdf.slice(offset=0, length=n_train).to_pandas()
    else:
        raise ValueError(
            f"Unknown dataframe_type: {dataframe_type}. "
            "Supported types: ['pandas', 'polars']"
        )

    return train_df


# ---------------------------------------------------------------------------
# Feature table construction
# ---------------------------------------------------------------------------
def build_feature_table(
    df: pd.DataFrame,
    config: Dict,
    aggregations: Sequence[AggregationSpec] = DEFAULT_AGGREGATIONS,
) -> pd.DataFrame:
    """Flatten one raw event row into one numeric row per event.

    Column layout of the returned table (all float64):

    * ``<vector>__<agg>`` : one column per (vector feature x aggregation),
      e.g. ``particle.energy__mean`` -- per-prong sequences reduced to event
      level so they can sit next to scalars in a correlation matrix.
    * ``n_prongs``        : sequence length of the first vector feature.  All
      vector features share one multiplicity per event, so one column is
      enough; multiplicity itself often correlates strongly with energy and
      is worth having in the matrix.
    * ``<scalar>``        : copied through unchanged.
    * ``<target>``        : copied through unchanged, so feature<->target
      correlations come out of the same ``.corr()`` call.

    Values are the *raw physical* values from the CSV, not the z-scored
    versions the network sees -- Pearson correlation is invariant under the
    (affine) normalisation anyway, and raw units keep the table readable.
    """
    columns: Dict[str, np.ndarray] = {}

    for vec_name in config["vector"]:
        # Parse each cell exactly like the training dataset does.  Doing the
        # parse once per column (rather than once per aggregation) matters:
        # string splitting dominates the cost on multi-GB CSVs.
        parsed = df[vec_name].apply(string_to_float_list)
        for agg in aggregations:
            # np.mean/np.std never see an empty list: string_to_float_list
            # maps missing cells to [0].  std of a length-1 list is 0.0.
            columns[f"{vec_name}__{agg.name}"] = parsed.apply(agg.func).to_numpy(
                dtype=np.float64
            )
        if "n_prongs" not in columns:
            columns["n_prongs"] = parsed.apply(len).to_numpy(dtype=np.float64)

    for name in list(config["scalar"]) + list(config["target"]):
        columns[name] = df[name].to_numpy(dtype=np.float64)

    return pd.DataFrame(columns, index=df.index)


def compute_correlations(
    feature_df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Correlation matrix of the flattened feature table.

    ``method`` is forwarded to ``DataFrame.corr`` -- "pearson" for linear
    relationships, "spearman" as a cheap robustness check when features have
    heavy tails (calorimetric sums usually do).  Constant columns (e.g. a
    ``__std`` aggregation on a dataset of 1-prong events) produce NaN rows;
    they are left in place so the heatmap makes the degeneracy visible
    instead of silently dropping the feature.
    """
    return feature_df.corr(method=method)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str) -> None:
    """Render ``corr`` as an annotated heatmap PNG at ``out_path``.

    Style notes (kept close to the usual CERN-figure conventions):
    * diverging colormap pinned to [-1, 1] so colours are comparable across
      datasets/models -- a matrix that "looks red" genuinely is more
      correlated, not just autoscaled;
    * cell annotations only below ~30 columns, past that they become soup.
    """
    n = len(corr.columns)
    # ~0.45" per column keeps labels legible from 10 to 40 features.
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * n), max(6.5, 0.45 * n)))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)

    if n <= 30:
        for i in range(n):
            for j in range(n):
                val = corr.iat[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        # White text on saturated cells, black elsewhere.
                        color="white" if abs(val) > 0.6 else "black",
                    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    ax.set_title("Feature / target correlation (train split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)  # Free the canvas: batch jobs may render many figures.


# ---------------------------------------------------------------------------
# Permutation feature importance (global, model-based)
# ---------------------------------------------------------------------------
def _materialise_split(
    loader: torch.utils.data.DataLoader, max_events: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate a DataLoader's batches into whole-split tensors.

    Permutation importance shuffles one feature *across the whole split*, so
    it is far simpler to hold the split in memory once than to coordinate a
    shuffled view through the DataLoader.  Batches arrive as the 5-tuple the
    dataset emits -- ``(vector, scalar, mask, target, weight)`` -- and we keep
    the first four:

        vector : (N, max_num_prongs, n_vector_features), z-scored, padded 0
        scalar : (N, n_scalar_features), z-scored
        mask   : (N, max_num_prongs) bool, True marks *padding* prongs
        target : (N, n_targets), raw physical units (targets are never
                 normalised by the pipeline -- see pd_dataset.normalize)

    ``max_events`` caps memory on huge test splits; because the valid/test
    loaders are built with ``shuffle=False`` the leading events are a
    deterministic, reproducible subset.
    """
    vecs: List[torch.Tensor] = []
    scas: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    n_seen = 0
    for vector, scalar, mask, target, _weight in loader:
        vecs.append(vector)
        scas.append(scalar)
        masks.append(mask)
        # 0-dim placeholder targets only occur in eval-mode datasets, which
        # this script never builds; keep the cat unconditional.
        tgts.append(target)
        n_seen += vector.shape[0]
        if max_events is not None and n_seen >= max_events:
            break
    vector_t = torch.cat(vecs)[:max_events]
    scalar_t = torch.cat(scas)[:max_events]
    mask_t = torch.cat(masks)[:max_events]
    target_t = torch.cat(tgts)[:max_events]
    return vector_t, scalar_t, mask_t, target_t


@torch.no_grad()
def _predict_in_chunks(
    model: torch.nn.Module,
    vector: torch.Tensor,
    scalar: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    chunk: int = 4096,
) -> torch.Tensor:
    """Forward the whole split through ``model`` in memory-safe chunks.

    Only the chunk currently being evaluated lives on ``device``; inputs and
    outputs stay on CPU, which is what makes the function safe for
    million-event splits on an 8 GB GPU.  Call signature of the network is
    ``model(vector, scalar, mask)`` (see Transformer_EE_MV.forward).
    """
    outs: List[torch.Tensor] = []
    for lo in range(0, vector.shape[0], chunk):
        hi = lo + chunk
        pred = model(
            vector[lo:hi].to(device),
            scalar[lo:hi].to(device),
            mask[lo:hi].to(device),
        )
        outs.append(pred.cpu())
    return torch.cat(outs)


def permutation_importance(
    config: Dict,
    checkpoint_path: str,
    n_repeats: int = 5,
    seed: int = 0,
    max_events: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Model-based global importance: how much does the test error grow when
    one input feature is shuffled across events?

    Definition used here (classic Breiman/Fisher permutation importance):

        importance(f) = MAE_test(f permuted) - MAE_test(baseline)

    computed per target and averaged over ``n_repeats`` independent
    permutations.  Shuffling a feature breaks its relationship with the
    target while leaving its marginal distribution untouched, so a large
    positive score means the trained network genuinely relies on the feature.
    Scores near zero mean "unused *or* redundant" -- a feature whose
    information is duplicated elsewhere can permute away almost for free,
    which is exactly why this table should be read side by side with the
    correlation heatmap.

    Mechanics / conventions:
    * Evaluated on the *test* split rebuilt by the same seeded pipeline used
      in training (get_train_valid_test_dataloader), so no leakage.
    * A "feature" is one whole channel: for vector features the entire
      per-event prong sequence of that channel is swapped between events
      (padding travels with it; padded positions are excluded from attention
      by the key-padding mask, so stray zeros are inert).  For scalars the
      single value is swapped.
    * MAE is computed in the target's physical units (targets are not
      normalised anywhere in the pipeline), so scores read directly as
      "extra GeV of error".
    """
    cfg = _normalise_config(config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact training-time split; the 4th return value is the
    # train-set normalisation statistics, unused here because the loaders
    # already emit normalised features.
    _train, _valid, testloader, _stat = get_train_valid_test_dataloader(cfg)
    vector, scalar, mask, target = _materialise_split(testloader, max_events)

    # Faithful to this module's import list: build the architecture with
    # create_model and load raw weights (a state_dict saved by the trainer as
    # best_model.zip).  map_location="cpu" first, then .to(device), so a
    # CUDA-trained checkpoint loads fine on a CPU-only box.
    model = create_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    baseline_pred = _predict_in_chunks(model, vector, scalar, mask, device)
    # Per-target MAE, shape (n_targets,).
    baseline_mae = (baseline_pred - target).abs().mean(dim=0)

    target_names = list(cfg["target"])
    feature_names = list(cfg["vector"]) + list(cfg["scalar"])
    n_vector = len(cfg["vector"])
    rng = np.random.default_rng(seed)  # Local RNG: no global-state pollution.

    rows: List[Dict[str, float]] = []
    for f_idx, f_name in enumerate(feature_names):
        # (n_repeats, n_targets) matrix of MAE increases for this feature.
        deltas = np.zeros((n_repeats, len(target_names)), dtype=np.float64)
        for rep in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(vector.shape[0]))
            if f_idx < n_vector:
                # Vector channel: clone -> overwrite channel f with the same
                # channel drawn from permuted events.  Cloning one (N, L)
                # slice per repeat is cheap next to the forward pass.
                v_perm = vector.clone()
                v_perm[:, :, f_idx] = vector[perm, :, f_idx]
                pred = _predict_in_chunks(model, v_perm, scalar, mask, device)
            else:
                s_perm = scalar.clone()
                s_idx = f_idx - n_vector
                s_perm[:, s_idx] = scalar[perm, s_idx]
                pred = _predict_in_chunks(model, vector, s_perm, mask, device)
            mae = (pred - target).abs().mean(dim=0)
            deltas[rep] = (mae - baseline_mae).numpy()

        row: Dict[str, float] = {"feature": f_name}
        for t_idx, t_name in enumerate(target_names):
            row[f"delta_mae__{t_name}"] = float(deltas[:, t_idx].mean())
        # Scalar ranking key: mean over targets; std over repeats of that
        # same scalar quantifies permutation noise (grow n_repeats if the
        # std rivals the mean).
        per_rep_mean = deltas.mean(axis=1)
        row["delta_mae__mean"] = float(per_rep_mean.mean())
        row["delta_mae__std_over_repeats"] = float(per_rep_mean.std())
        rows.append(row)
        print(f"[importance] {f_name}: {row['delta_mae__mean']:+.5f} MAE")

    out = pd.DataFrame(rows).sort_values("delta_mae__mean", ascending=False)
    return out.reset_index(drop=True)


def plot_importance(importance_df: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart of mean MAE increase per permuted feature.

    Horizontal bars because feature names like ``particle.start.x`` are long;
    error bars show the std over permutation repeats, i.e. how much of the
    ranking is just shuffle noise.
    """
    df = importance_df.sort_values("delta_mae__mean")  # Largest ends on top.
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35 * len(df))))
    ax.barh(
        df["feature"],
        df["delta_mae__mean"],
        xerr=df["delta_mae__std_over_repeats"],
        color="#3b6aa0",
        error_kw={"elinewidth": 1.0, "capsize": 2.0},
    )
    ax.set_xlabel("MAE increase when permuted (target units)")
    ax.set_title("Permutation feature importance (test split)")
    ax.axvline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI mirroring the training pipeline's config-first workflow.

    Examples
    --------
    Correlations only (no model needed)::

        python feature_diagnostics.py --config input.json --mode correlation

    Both diagnostics against a trained checkpoint::

        python feature_diagnostics.py --config model_dir/input.json \
            --checkpoint model_dir/best_model.zip --mode both \
            --output-dir diagnostics_out
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training/analysis JSON config.")
    parser.add_argument(
        "--mode",
        choices=("correlation", "importance", "both"),
        default="both",
        help="Which diagnostic(s) to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained state_dict (best_model.zip); required for importance.",
    )
    parser.add_argument("--output-dir", default="feature_diagnostics_out")
    parser.add_argument(
        "--corr-method",
        default="pearson",
        choices=("pearson", "spearman", "kendall"),
        help="Forwarded to DataFrame.corr for the correlation matrix.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Independent permutations per feature (importance mode).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap test events for importance (memory guard); default: all.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Permutation RNG seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is the safe default on Windows/PowerShell "
        "(worker processes need spawn + __main__ guards and rarely pay off "
        "for a single analysis pass).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="UTF-8") as fh:
        config = json.load(fh)
    # Analysis-time override only -- the value stored inside a model_dir's
    # input.json (often 10, tuned for the training cluster) is left on disk
    # untouched.
    config["num_workers"] = args.num_workers

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ("correlation", "both"):
        print("[correlation] building train-split feature table ...")
        train_df = _train_subset(config)
        table = build_feature_table(train_df, config)
        corr = compute_correlations(table, method=args.corr_method)
        corr_csv = os.path.join(args.output_dir, "feature_correlations.csv")
        corr_png = os.path.join(args.output_dir, "feature_correlations.png")
        corr.to_csv(corr_csv)
        plot_correlation_heatmap(corr, corr_png)
        print(f"[correlation] wrote {corr_csv} and {corr_png}")

    if args.mode in ("importance", "both"):
        if not args.checkpoint:
            parser.error("--checkpoint is required for --mode importance/both")
        print("[importance] evaluating permutation importance ...")
        imp = permutation_importance(
            config,
            checkpoint_path=args.checkpoint,
            n_repeats=args.n_repeats,
            seed=args.seed,
            max_events=args.max_events,
        )
        imp_csv = os.path.join(args.output_dir, "permutation_importance.csv")
        imp_png = os.path.join(args.output_dir, "permutation_importance.png")
        imp.to_csv(imp_csv, index=False)
        plot_importance(imp, imp_png)
        print(f"[importance] wrote {imp_csv} and {imp_png}")


if __name__ == "__main__":
    # The guard matters beyond style: with num_workers > 0 PyTorch spawns
    # subprocesses that re-import this module, and on Windows that recursion
    # only terminates because of this guard.
    main()
