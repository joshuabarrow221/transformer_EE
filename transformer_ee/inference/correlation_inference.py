#!/usr/bin/env python3
"""
Inference helper that keeps model predictions plus selected source columns
for downstream correlation studies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np
import pandas as pd

from transformer_ee.inference.pred_wBatch import Predictor


def _split_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference and write a correlation-ready output table containing "
            "predictions, truths, and selected passthrough columns from the input CSV."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Directory with input.json + best_model.zip")
    parser.add_argument("--input-csv", required=True, help="Inference sample CSV")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated input CSV columns to passthrough (e.g. EventID,NuMomX)",
    )
    parser.add_argument(
        "--copy-regex",
        default="",
        help="Regex applied to input CSV column names. Any match is copied.",
    )
    parser.add_argument(
        "--copy-scalars",
        action="store_true",
        help="Copy all scalar features listed in the model input.json.",
    )
    parser.add_argument(
        "--copy-targets",
        action="store_true",
        help="Copy target columns from input CSV if present (useful for truth validation samples).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a JSON manifest next to output CSV describing included columns.",
    )
    return parser.parse_args()


def resolve_passthrough_columns(
    dataframe: pd.DataFrame,
    explicit: Sequence[str],
    regex_pattern: str,
    scalar_names: Sequence[str],
    target_names: Sequence[str],
    copy_scalars: bool,
    copy_targets: bool,
) -> List[str]:
    selected: Set[str] = set()

    for column in explicit:
        if column in dataframe.columns:
            selected.add(column)
        else:
            print(f"[WARN] Requested passthrough column '{column}' was not found in input CSV.")

    if copy_scalars:
        for name in scalar_names:
            if name in dataframe.columns:
                selected.add(name)

    if copy_targets:
        for name in target_names:
            if name in dataframe.columns:
                selected.add(name)

    if regex_pattern:
        matcher = re.compile(regex_pattern)
        for name in dataframe.columns:
            if matcher.search(name):
                selected.add(name)

    return sorted(selected)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    df = pd.read_csv(input_csv)
    if args.max_rows is not None and args.max_rows >= 0:
        df = df.iloc[: args.max_rows].copy()

    predictor = Predictor(str(model_dir), df.copy())
    predictions = predictor.go()

    target_names = predictor.train_config.get("target", [])
    scalar_names = predictor.train_config.get("scalar", [])
    if len(target_names) != predictions.shape[1]:
        target_names = [f"target_{i}" for i in range(predictions.shape[1])]

    passthrough_columns = resolve_passthrough_columns(
        dataframe=df,
        explicit=_split_csv_list(args.copy_columns),
        regex_pattern=args.copy_regex,
        scalar_names=scalar_names,
        target_names=target_names,
        copy_scalars=args.copy_scalars,
        copy_targets=args.copy_targets,
    )

    result = pd.DataFrame()
    result["event_index"] = np.arange(len(df), dtype=int)

    for column in passthrough_columns:
        result[column] = df[column].values

    for idx, name in enumerate(target_names):
        if name in df.columns:
            result[f"true_{name}"] = df[name].to_numpy()
        else:
            result[f"true_{name}"] = np.full(len(df), np.nan)

    for idx, name in enumerate(target_names):
        pred_col = predictions[:, idx]
        result[f"pred_{name}"] = pred_col
        truth_col = result[f"true_{name}"]
        result[f"resid_{name}"] = pred_col - truth_col

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"[INFO] Wrote correlation-ready inference table: {output_csv}")

    if args.write_manifest:
        manifest = {
            "model_dir": str(model_dir),
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "rows": len(result),
            "targets": target_names,
            "passthrough_columns": passthrough_columns,
            "truth_available": [name in df.columns for name in target_names],
        }
        manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"[INFO] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
