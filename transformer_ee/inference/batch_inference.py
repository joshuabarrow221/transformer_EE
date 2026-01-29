#!/usr/bin/env python3
"""
Batch inference runner that pairs multiple trained models with multiple CSV samples.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import inspect
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np

from transformer_ee.inference.pred_wBatch import Predictor


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
    parser.add_argument(
        "--eval-macro-path",
        type=str,
        default=None,
        help="Optional ROOT macro (eval_model.C) to run on each output CSV.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help="Directory to store eval_model outputs (combined_output.root, ellipse_fraction.csv).",
    )
    parser.add_argument(
        "--eval-save-png",
        action="store_true",
        help="Save energy_theta_2d_canvas as a PNG per model/sample.",
    )
    parser.add_argument(
        "--eval-png-width",
        type=int,
        default=3000,
        help="PNG width for eval_model output (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-png-height",
        type=int,
        default=2000,
        help="PNG height for eval_model output (default: %(default)s).",
    )
    parser.add_argument(
        "--start-task",
        type=int,
        default=1,
        help="Start from this 1-based task index (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue after task failures and record errors in failed_tasks.jsonl.",
    )
    parser.add_argument(
        "--missing-models-file",
        type=str,
        default=None,
        help=(
            "Path to a models_not_found.txt file. If provided, skip training_name lookups "
            "that were already recorded as missing."
        ),
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


def shorten_component(name: str, max_length: int = 120) -> str:
    if len(name) <= max_length:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    keep = max_length - (len(digest) + 1)
    return f"{name[:keep]}_{digest}"


def ensure_bucket(output_dir: Path, model_name: str) -> Tuple[Path, str]:
    base = safe_name(model_name)
    candidate = output_dir / base
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate, base
    except OSError as exc:
        if exc.errno != errno.ENAMETOOLONG:
            raise
    shortened = shorten_component(base, max_length=120)
    candidate = output_dir / shortened
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate, shortened


def safe_output_base(label: str, max_length: int = 180) -> str:
    base = safe_name(label)
    return shorten_component(base, max_length=max_length)


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
    config_path: Path,
    model_root: Optional[Path],
    sample_root: Optional[Path],
    known_missing_models: Optional[Iterable[str]] = None,
) -> Tuple[List[InferenceTask], List[str]]:
    with open(config_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "pairs" not in payload:
        raise ValueError("Pair config must include a 'pairs' list.")

    training_search_roots = [
        Path(path).expanduser().resolve() for path in payload.get("model_search_roots", [])
    ]

    model_lookup: Dict[str, Union[Path, List[Path]]] = {}
    sample_lookup: Dict[str, Path] = {}
    missing_models: List[str] = []
    known_missing = set(known_missing_models or [])

    for entry in payload.get("models", []):
        label, resolved = parse_entity(
            entry,
            model_root,
            resolve_model_dir,
            training_search_roots=training_search_roots,
            missing_models=missing_models,
            known_missing=known_missing,
        )
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
            training_search_roots=training_search_roots,
            missing_models=missing_models,
            known_missing=known_missing,
        )
        sample_label, sample_path = parse_pair_entry(
            pair,
            key="sample",
            lookup=sample_lookup,
            root=sample_root,
            resolver=resolve_sample_path,
            training_search_roots=None,
        )
        for resolved_label, resolved_path in normalize_model_paths(
            model_label, model_path, training_search_roots
        ):
            tasks.append(
                InferenceTask(
                    model_name=resolved_label,
                    model_dir=resolved_path,
                    sample_name=sample_label,
                    sample_path=sample_path,
                )
            )
    return tasks, missing_models


def parse_entity(
    entry,
    root: Optional[Path],
    resolver,
    training_search_roots: Optional[Sequence[Path]] = None,
    missing_models: Optional[List[str]] = None,
    known_missing: Optional[Iterable[str]] = None,
) -> Tuple[str, Union[Path, List[Path]]]:
    if isinstance(entry, dict):
        if "training_name" in entry:
            training_name = entry["training_name"]
            label = entry.get("name", training_name)
            if not training_search_roots:
                raise ValueError(
                    "Model entries with 'training_name' require 'model_search_roots'."
                )
            resolved_path = resolve_model_from_training_name(
                training_name,
                training_search_roots,
                missing_models,
                known_missing=known_missing,
            )
            return label, resolved_path
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
    lookup: Dict[str, Union[Path, List[Path]]],
    root: Optional[Path],
    resolver,
    training_search_roots: Optional[Sequence[Path]] = None,
    missing_models: Optional[List[str]] = None,
    known_missing: Optional[Iterable[str]] = None,
) -> Tuple[str, Union[Path, List[Path]]]:
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
        label, resolved = parse_entity(
            value,
            root,
            resolver,
            training_search_roots=training_search_roots,
            missing_models=missing_models,
            known_missing=known_missing,
        )
    else:
        resolved = resolver(resolve_relative(value, root))
        label = default_label(resolved, root)
    lookup[label] = resolved
    return label, resolved


def resolve_model_from_training_name(
    training_name: str,
    search_roots: Sequence[Path],
    missing_models: Optional[List[str]] = None,
    known_missing: Optional[Iterable[str]] = None,
) -> List[Path]:
    if known_missing and training_name in set(known_missing):
        return []
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("input.json"):
            if training_name in path.parts and (path.parent / "best_model.zip").exists():
                candidates.append(path.parent.resolve())
    unique = sorted({path.resolve() for path in candidates}, key=lambda p: str(p))
    if not unique:
        if missing_models is not None and training_name not in missing_models:
            missing_models.append(training_name)
        return []
    return unique


def normalize_model_paths(
    label: str,
    model_path: Union[Path, List[Path]],
    search_roots: Optional[Sequence[Path]],
) -> List[Tuple[str, Path]]:
    if isinstance(model_path, list):
        if not model_path:
            return []
        if len(model_path) == 1:
            return [(label, model_path[0])]
        return [
            (label_with_origin(label, path, search_roots), path) for path in model_path
        ]
    return [(label, model_path)]


def label_with_origin(
    label: str, model_dir: Path, search_roots: Optional[Sequence[Path]]
) -> str:
    if search_roots:
        for root in search_roots:
            try:
                relative = model_dir.relative_to(root)
            except ValueError:
                continue
            return f"{label}::{relative}"
    return f"{label}::{model_dir.name}"


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


def run_eval_model(
    csv_path: Path,
    model_name: str,
    model_label: str,
    eval_macro_path: Path,
    eval_output_dir: Path,
    eval_save_png: bool,
    eval_png_size: Optional[Tuple[int, int]],
) -> Dict[str, object]:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    png_path = ""
    if eval_save_png:
        png_base = f"{safe_name(model_name)}__{safe_name(csv_path.stem)}__energy_theta_2d.png"
        png_path = str((eval_output_dir / png_base).resolve())
    width, height = eval_png_size or (0, 0)
    cmd = [
        "root",
        "-l",
        "-b",
        "-q",
        (
            f"{eval_macro_path}("
            f"\"{csv_path}\","
            f"\"{eval_output_dir}\","
            f"\"{png_path}\","
            f"{width},"
            f"{height}"
            f",\"combined_inference_output.root\""
            f")"
        ),
    ]
    result = {
        "macro_path": str(eval_macro_path),
        "output_dir": str(eval_output_dir),
        "png_path": png_path or None,
        "returncode": None,
        "ellipse_fraction": None,
        "model_label": model_label,
    }
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result["error"] = "root command not found on PATH"
        return result
    result["returncode"] = completed.returncode
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip() or completed.stdout.strip()
        return result

    ellipse_path = eval_output_dir / "ellipse_fraction.csv"
    if ellipse_path.exists():
        try:
            ellipse_df = pd.read_csv(ellipse_path)
            matched = ellipse_df[ellipse_df["model_name"] == model_label]
            if not matched.empty:
                result["ellipse_fraction"] = matched.tail(1).to_dict(orient="records")[0]
        except Exception as exc:
            result["error"] = f"Failed to parse ellipse_fraction.csv: {exc}"
    return result


def run_task(
    task: InferenceTask,
    output_dir: Path,
    device: Optional[str],
    num_workers: Optional[int],
    max_rows: Optional[int],
    eval_macro_path: Optional[Path] = None,
    eval_output_dir: Optional[Path] = None,
    eval_save_png: bool = False,
    eval_png_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    print(
        f"[INFO] Running model '{task.model_name}' on sample '{task.sample_name}' "
        f"({task.sample_path.name})"
    )
    dataframe = pd.read_csv(task.sample_path)
    dataframe = trim_dataframe(dataframe, max_rows)
    predictor_kwargs = {}
    predictor_params = inspect.signature(Predictor).parameters
    if "device" in predictor_params:
        predictor_kwargs["device"] = device
    if "num_workers" in predictor_params:
        predictor_kwargs["num_workers"] = num_workers
    predictor = Predictor(str(task.model_dir), dataframe, **predictor_kwargs)

    go_params = inspect.signature(predictor.go).parameters
    if "return_batch_times" in go_params:
        predictions, batch_times, batch_sizes = predictor.go(return_batch_times=True)
    else:
        predictions = predictor.go()
        batch_times = []
        batch_sizes = []
    target_names = predictor.train_config.get("target", [])
    if len(target_names) != predictions.shape[1]:
        target_names = [f"target_{idx}" for idx in range(predictions.shape[1])]
    raw_truth_df = getattr(predictor, "original_df", dataframe)
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
    eval_label = task.model_name
    if eval_macro_path is not None:
        eval_label = f"{safe_name(task.model_name)}__{safe_name(task.sample_name)}"
    result_df[eval_label] = -999999
    model_bucket, bucket_label = ensure_bucket(output_dir, task.model_name)
    base_name = safe_output_base(task.sample_name)
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
    eval_meta: Dict[str, object] = {}
    if eval_macro_path is not None:
        eval_meta = run_eval_model(
            csv_path=csv_path,
            model_name=task.model_name,
            model_label=eval_label,
            eval_macro_path=eval_macro_path,
            eval_output_dir=eval_output_dir or output_dir,
            eval_save_png=eval_save_png,
            eval_png_size=eval_png_size,
        )
    total_time = 0.0
    if batch_times:
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
        "output_bucket": bucket_label,
        "output_csv": str(csv_path),
        "output_npz": str(npz_path),
        "num_rows": len(result_df),
        "num_outputs": len(target_names),
        "truth_available": truth_available,
        "total_time_sec": total_time,
        "eval_model": eval_meta,
    }


def main():
    args = parse_args()
    model_root = Path(args.model_root).resolve() if args.model_root else None
    sample_root = Path(args.sample_root).resolve() if args.sample_root else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_models: List[str] = []

    known_missing = set()
    if args.missing_models_file:
        missing_path = Path(args.missing_models_file).expanduser().resolve()
        if missing_path.exists():
            with open(missing_path, encoding="utf-8") as handle:
                known_missing = {
                    line.strip()
                    for line in handle
                    if line.strip() and not line.startswith("#")
                }

    if args.pair_config:
        tasks, missing_models = load_pairs_from_config(
            Path(args.pair_config),
            model_root,
            sample_root,
            known_missing_models=known_missing,
        )
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
    if missing_models:
        missing_path = output_dir / "models_not_found.txt"
        with open(missing_path, "w", encoding="utf-8") as handle:
            for name in sorted(set(missing_models)):
                handle.write(f"{name}\n")
        print(
            "[WARN] Missing model exports for training names. "
            f"See {missing_path} for details."
        )

    if not tasks:
        print("[WARN] No inference tasks to execute.")
        return

    summary: List[Dict[str, str]] = []
    failed_tasks_path = output_dir / "failed_tasks.jsonl"
    start_idx = max(1, args.start_task)
    for task_idx, task in enumerate(tasks, start=1):
        if task_idx < start_idx:
            print(f"[INFO] Skipping task {task_idx} / {len(tasks)}")
            continue
        print(f"[INFO] Task {task_idx} / {len(tasks)}")
        try:
            meta = run_task(
                task,
                output_dir=output_dir,
                device=args.device,
                num_workers=args.num_workers,
                max_rows=args.max_samples,
                eval_macro_path=Path(args.eval_macro_path).expanduser().resolve()
                if args.eval_macro_path
                else None,
                eval_output_dir=Path(args.eval_output_dir).expanduser().resolve()
                if args.eval_output_dir
                else None,
                eval_save_png=args.eval_save_png,
                eval_png_size=(args.eval_png_width, args.eval_png_height)
                if args.eval_save_png
                else None,
            )
            summary.append(meta)
        except Exception as exc:
            error_meta = {
                "model_name": task.model_name,
                "model_dir": str(task.model_dir),
                "sample_name": task.sample_name,
                "sample_path": str(task.sample_path),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            with open(failed_tasks_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_meta) + "\n")
            print(f"[ERROR] Task {task_idx} failed: {exc}")
            if not args.skip_errors:
                raise

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
