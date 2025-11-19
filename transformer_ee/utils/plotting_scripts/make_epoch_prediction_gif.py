"""Create a GIF visualising prediction distributions across epochs."""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import numpy as np

from transformer_ee.utils import binstat

EPOCH_PATTERN = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)


def _expand_files(patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            files.extend(expanded)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return sorted(set(files))


def _epoch_from_path(path: str) -> int:
    match = EPOCH_PATTERN.search(Path(path).stem)
    if match:
        return int(match.group(1))
    return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-epoch prediction histograms and combine them into a GIF."
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        help="List of result files or glob patterns (e.g. 'result_epoch*.npz').",
    )
    parser.add_argument(
        "--variable-index",
        type=int,
        default=0,
        help="Index of the predicted variable to visualise.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to store intermediate plots and the final GIF.",
    )
    parser.add_argument(
        "--prefix",
        default="prediction",
        help="Prefix for the generated plot filenames.",
    )
    parser.add_argument(
        "--gif-name",
        default="training_evolution.gif",
        help="Filename for the generated GIF (within output-dir).",
    )
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=0.6,
        help="Duration (in seconds) of each frame in the GIF.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for the histogram.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI used when saving histogram images.",
    )
    parser.add_argument(
        "--value-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Optional range for the histogram (min max).",
    )
    parser.add_argument(
        "--xlabel",
        default="Predicted value",
        help="Label for the histogram x-axis.",
    )
    parser.add_argument(
        "--ylabel",
        default="Event count",
        help="Label for the histogram y-axis.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Epoch",
        help="Prefix for the histogram title (epoch number appended).",
    )
    parser.add_argument(
        "--file-extension",
        default="png",
        help="Image format used for intermediate plots (e.g. png, pdf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_files = _expand_files(args.result_files)
    if not result_files:
        raise FileNotFoundError("No result files matched the provided patterns.")

    result_files.sort(key=_epoch_from_path)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_paths: List[str] = []
    for file_path in result_files:
        epoch = _epoch_from_path(file_path)
        if epoch < 0:
            print(f"Warning: could not infer epoch number from '{file_path}'. Skipping.")
            continue

        with np.load(file_path) as data:
            if "prediction" not in data:
                print(f"Warning: 'prediction' not found in '{file_path}'. Skipping.")
                continue
            predictions = data["prediction"]

        if args.variable_index < 0 or args.variable_index >= predictions.shape[1]:
            raise IndexError(
                f"variable-index {args.variable_index} out of bounds for predictions with "
                f"shape {predictions.shape}."
            )

        values = predictions[:, args.variable_index]
        plot_base = os.path.join(args.output_dir, f"{args.prefix}_epoch{epoch:03d}")

        plot_kwargs = {
            "bins": args.bins,
            "dpi": args.dpi,
            "ext": args.file_extension,
            "xlabel": args.xlabel,
            "ylabel": args.ylabel,
            "title": f"{args.title_prefix} {epoch:03d}",
        }
        if args.value_range is not None:
            plot_kwargs["range"] = (args.value_range[0], args.value_range[1])

        binstat.plot_y_hist(values, name=plot_base, **plot_kwargs)
        plot_paths.append(f"{plot_base}.{args.file_extension}")

    if not plot_paths:
        raise RuntimeError("No plots were generated; check the result files provided.")

    gif_path = os.path.join(args.output_dir, args.gif_name)
    frames = [imageio.imread(path) for path in plot_paths]
    imageio.mimsave(gif_path, frames, duration=args.frame_duration)
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
