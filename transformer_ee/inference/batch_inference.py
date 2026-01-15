#!/usr/bin/env python3
"""
Batch inference runner that pairs multiple trained models with multiple CSV samples.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

from transformer_ee.inference.pred import Predictor


@dataclass(frozen=True)
class InferenceTask:
    model_name: str
    model_dir: Path
    sample_name: str
    sample_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference over many model/sample combinations."
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="InferenceTests/Models",
        help="Directory that stores trained model exports (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-root",
        type=str,
        default="InferenceTests/InferenceSamples",
        help="Directory that stores CSV inference samples (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Explicit model directories. If omitted, discover all models under --model-root.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Explicit CSV sample paths. If omitted, discover all CSV files under --sample-root.",
    )
    parser.add_argument(
        "--pair-config",
        type=str,
        help="Optional JSON file that lists explicit model/sample pairs.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["cartesian", "zip"],
        default="cartesian",
        help="How to pair models and samples when --pair-config is not supplied.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="InferenceTests/Results",
        help="Where to store inference outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override such as 'cpu' or 'cuda:0'. Leave empty to auto-detect.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader worker count.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples processed per CSV (useful for smoke tests).",
    )
    return parser.parse_args()


def discover_model_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    candidates = {
        path.parent.resolve()
        for path in root.rglob("input.json")
        if (path.parent / "best_model.zip").exists()
    }
    return sorted(candidates, key=lambda p: str(p))


def discover_sample_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"), key=lambda p: str(p))


def resolve_model_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Model directory '{path}' does not exist.")
    if path.is_file():
        if path.name == "input.json":
            return path.parent
        raise ValueError(f"Model path '{path}' must be a directory.")
    if (path / "input.json").exists() and (path / "best_model.zip").exists():
        return path
    direct_candidates = [
        p.parent for p in path.glob("*/input.json") if (p.parent / "best_model.zip").exists()
    ]
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    raise ValueError(
        f"Could not resolve a unique model export under '{path}'. "
        "Please point to the directory that directly holds best_model.zip."
    )


def resolve_sample_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Sample CSV '{path}' does not exist.")
    if path.is_dir():
        raise ValueError(f"Sample path '{path}' must be a CSV file.")
    return path


def default_label(path: Path, root: Optional[Path]) -> str:
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def safe_name(label: str) -> str:
    # strip common file extensions and sanitize separators/spaces
    stripped = label
    if stripped.endswith(".csv"):
        stripped = stripped[: -len(".csv")]
    name = stripped.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
    return name.replace(" ", "_")


def resolve_relative(candidate: str, root: Optional[Path]) -> Path:
    path = Path(candidate)
    if path.exists():
        return path
    if root is not None:
        alt = (root / candidate).resolve()
        if alt.exists():
            return alt
    return path


def load_pairs_from_config(
    config_path: Path, model_root: Optional[Path], sample_root: Optional[Path]
) -> List[InferenceTask]:
    with open(config_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "pairs" not in payload:
        raise ValueError("Pair config must include a 'pairs' list.")

    model_lookup: Dict[str, Path] = {}
    sample_lookup: Dict[str, Path] = {}

    for entry in payload.get("models", []):
        label, resolved = parse_entity(entry, model_root, resolve_model_dir)
        model_lookup[label] = resolved

    for entry in payload.get("samples", []):
        label, resolved = parse_entity(entry, sample_root, resolve_sample_path)
        sample_lookup[label] = resolved

    tasks: List[InferenceTask] = []
    for pair in payload["pairs"]:
        model_label, model_path = parse_pair_entry(
            pair,
            key="model",
            lookup=model_lookup,
            root=model_root,
            resolver=resolve_model_dir,
        )
        sample_label, sample_path = parse_pair_entry(
            pair,
            key="sample",
            lookup=sample_lookup,
            root=sample_root,
            resolver=resolve_sample_path,
        )
        tasks.append(
            InferenceTask(
                model_name=model_label,
                model_dir=model_path,
                sample_name=sample_label,
                sample_path=sample_path,
            )
        )
    return tasks


def parse_entity(
    entry, root: Optional[Path], resolver
) -> Tuple[str, Path]:
    if isinstance(entry, dict):
        path_value = entry.get("path") or entry.get("dir") or entry.get("file")
        if path_value is None:
            raise ValueError("Each entity dict must include a 'path' field.")
        label = entry.get("name")
    else:
        path_value = entry
        label = None
    resolved_path = resolver(resolve_relative(path_value, root))
    if label is None:
        label = default_label(resolved_path, root)
    return label, resolved_path


def parse_pair_entry(
    pair_entry,
    key: str,
    lookup: Dict[str, Path],
    root: Optional[Path],
    resolver,
) -> Tuple[str, Path]:
    if isinstance(pair_entry, dict):
        if key in pair_entry:
            value = pair_entry[key]
        else:
            value = None
            for alt_key in (f"{key}_path", f"{key}_dir", f"{key}_file"):
                if alt_key in pair_entry:
                    value = pair_entry[alt_key]
                    break
            if value is None:
                raise ValueError(f"Pair entries must define '{key}'.")
    elif isinstance(pair_entry, (list, tuple)) and len(pair_entry) == 2:
        value = pair_entry[0 if key == "model" else 1]
    else:
        raise ValueError(f"Pair entries must provide '{key}'.")

    if isinstance(value, str) and value in lookup:
        return value, lookup[value]
    if isinstance(value, dict):
        label, resolved = parse_entity(value, root, resolver)
    else:
        resolved = resolver(resolve_relative(value, root))
        label = default_label(resolved, root)
    lookup[label] = resolved
    return label, resolved


def build_tasks_from_lists(
    model_paths: Sequence[Path],
    sample_paths: Sequence[Path],
    pair_mode: str,
    model_root: Optional[Path],
    sample_root: Optional[Path],
) -> List[InferenceTask]:
    model_entries = [
        (default_label(path, model_root), resolve_model_dir(path)) for path in model_paths
    ]
    sample_entries = [
        (default_label(path, sample_root), resolve_sample_path(path)) for path in sample_paths
    ]

    tasks: List[InferenceTask] = []
    if pair_mode == "cartesian":
        for model_label, model_dir in model_entries:
            for sample_label, sample_path in sample_entries:
                tasks.append(
                    InferenceTask(
                        model_name=model_label,
                        model_dir=model_dir,
                        sample_name=sample_label,
                        sample_path=sample_path,
                    )
                )
    else:  # zip
        for (model_label, model_dir), (sample_label, sample_path) in zip(
            model_entries, sample_entries
        ):
            tasks.append(
                InferenceTask(
                    model_name=model_label,
                    model_dir=model_dir,
                    sample_name=sample_label,
                    sample_path=sample_path,
                )
            )
    return tasks


def trim_dataframe(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.iloc[:max_rows].copy()


def record_batch_timings(
    output_dir: Path,
    task: InferenceTask,
    batch_times: List[float],
    batch_sizes: List[int],
    total_rows: int,
) -> float:
    """
    Append per-batch timings for a task to a CSV in the output directory.
    """

    timing_path = output_dir / "timings.csv"
    df = pd.DataFrame(
        {
            "model_name": task.model_name,
            "sample_name": task.sample_name,
            "batch_index": list(range(1, len(batch_times) + 1)),
            "batch_time_sec": batch_times,
            "batch_size": batch_sizes,
            "total_rows": total_rows,
        }
    )
    df.to_csv(timing_path, mode="a", header=not timing_path.exists(), index=False)
    return float(np.sum(batch_times))


def run_task(
    task: InferenceTask,
    output_dir: Path,
    device: Optional[str],
    num_workers: Optional[int],
    max_rows: Optional[int],
) -> Dict[str, str]:
    print(
        f"[INFO] Running model '{task.model_name}' on sample '{task.sample_name}' "
        f"({task.sample_path.name})"
    )
    dataframe = pd.read_csv(task.sample_path)
    dataframe = trim_dataframe(dataframe, max_rows)
    predictor = Predictor(
        str(task.model_dir),
        dataframe,
        device=device,
        num_workers=num_workers,
    )
    predictions, batch_times, batch_sizes = predictor.go(return_batch_times=True)
    target_names = predictor.train_config.get("target", [])
    if len(target_names) != predictions.shape[1]:
        target_names = [f"target_{idx}" for idx in range(predictions.shape[1])]
    raw_truth_df = predictor.original_df
    truth_available = all(name in raw_truth_df.columns for name in target_names)
    if truth_available:
        truth_vals = raw_truth_df[target_names].to_numpy()
    else:
        truth_vals = np.full_like(predictions, np.nan, dtype=float)

    result_df = pd.DataFrame()
    # add truth and prediction columns, aligned by target (legacy eval_model style)
    for idx, name in enumerate(target_names):
        result_df[f"true_{name}"] = truth_vals[:, idx]
    for idx, name in enumerate(target_names):
        result_df[f"pred_{name}"] = predictions[:, idx]
    # trailing model name column (value placeholder to mirror legacy exports)
    result_df[task.model_name] = -999999
    model_bucket = output_dir / safe_name(task.model_name)
    model_bucket.mkdir(parents=True, exist_ok=True)
    base_name = safe_name(task.sample_name)
    csv_path = model_bucket / f"{base_name}.csv"
    npz_path = model_bucket / f"{base_name}.npz"
    result_df.to_csv(csv_path, index=False)
    np.savez(
        npz_path,
        trueval=truth_vals,
        prediction=predictions,
        target_names=np.array(target_names),
        sample_row_index=raw_truth_df.index.to_numpy(),
    )
    print(f"[INFO] Saved predictions to {csv_path} and {npz_path}")
    total_time = record_batch_timings(
        output_dir=output_dir,
        task=task,
        batch_times=batch_times,
        batch_sizes=batch_sizes,
        total_rows=len(result_df),
    )
    return {
        "model_name": task.model_name,
        "model_dir": str(task.model_dir),
        "sample_name": task.sample_name,
        "sample_path": str(task.sample_path),
        "output_csv": str(csv_path),
        "output_npz": str(npz_path),
        "num_rows": len(result_df),
        "num_outputs": len(target_names),
        "truth_available": truth_available,
        "total_time_sec": total_time,
    }


def main():
    args = parse_args()
    model_root = Path(args.model_root).resolve() if args.model_root else None
    sample_root = Path(args.sample_root).resolve() if args.sample_root else None

    if args.pair_config:
        tasks = load_pairs_from_config(Path(args.pair_config), model_root, sample_root)
    else:
        if args.models:
            model_paths = [resolve_relative(p, model_root) for p in args.models]
        else:
            if not model_root:
                raise ValueError("Either --models or --model-root must be provided.")
            model_paths = discover_model_dirs(model_root)
        if args.samples:
            sample_paths = [resolve_relative(p, sample_root) for p in args.samples]
        else:
            if not sample_root:
                raise ValueError("Either --samples or --sample-root must be provided.")
            sample_paths = discover_sample_paths(sample_root)
        if not model_paths:
            raise ValueError("No trained models were found.")
        if not sample_paths:
            raise ValueError("No inference samples were found.")
        tasks = build_tasks_from_lists(
            model_paths,
            sample_paths,
            args.pair_mode,
            model_root,
            sample_root,
        )

    if not tasks:
        print("[WARN] No inference tasks to execute.")
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, str]] = []
    for task_idx, task in enumerate(tasks, start=1):
        print(f"[INFO] Task {task_idx} / {len(tasks)}")
        meta = run_task(
            task,
            output_dir=output_dir,
            device=args.device,
            num_workers=args.num_workers,
            max_rows=args.max_samples,
        )
        summary.append(meta)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
