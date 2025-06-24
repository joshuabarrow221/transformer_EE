# ------------------------- loss_track.py ------------------------

import os
from typing import Dict, Sequence, List, Union
import numpy as np
import torch
import matplotlib.pyplot as plt

# Alias for type hints – list/tuple/ndarray/torch.Tensor of scalars
ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]


def _to_cpu_floats(seq: ArrayLike) -> List[float]:
    """Ensure *seq* is a list of Python floats on CPU (handles nested tensors)."""
    if torch.is_tensor(seq):  # a tensor of any shape
        seq = seq.detach().cpu().flatten().tolist()
    out: List[float] = []
    for v in seq:
        if torch.is_tensor(v):
            out.append(float(v.detach().cpu().item()))
        else:
            out.append(float(v))
    return out


def _clip_nonpositive(vals: List[float], *, eps: float) -> List[float]:
    """Replace non‑positive entries with a small positive *eps* for log scale."""
    return [v if v > 0 else eps for v in vals]


def plot_loss(metrics: Dict[str, ArrayLike], save_path: str, *, filename: str = "loss.png",
              logy: bool = True, ymin: float = 1e-4):
    """Plot multiple learning curves on a single canvas.

    Parameters
    ----------
    metrics : dict[str, ArrayLike]
        Keys become legend labels. All sequences must share length == n_epochs.
    save_path : str
        Directory where the figure will be written. Created if absent.
    filename : str, default "loss.png"
        Output file name inside *save_path*.
    logy : bool, default True
        If True, use a logarithmic y‑axis.
    ymin : float, default 1e‑3
        Minimum y‑axis value when *logy* is True. Also used to replace any
        non‑positive metric values so log(0) does not raise.
    """
    if not metrics:
        raise ValueError("'metrics' dictionary is empty – nothing to plot.")

    # Convert every series to CPU floats first
    metrics_cpu = {k: _to_cpu_floats(v) for k, v in metrics.items()}

    n_epochs = len(next(iter(metrics_cpu.values())))
    if any(len(v) != n_epochs for v in metrics_cpu.values()):
        raise ValueError("All metric sequences must have identical length.")

    # Clip / adjust values if log scale requested
    if logy:
        metrics_cpu = {k: _clip_nonpositive(v, eps=ymin * 0.1) for k, v in metrics_cpu.items()}

    epochs = range(1, n_epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, values) in enumerate(metrics_cpu.items()):
        ax.plot(epochs, values, label=label, linewidth=1.5,
                alpha=0.85, color=colours[idx % len(colours)])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training‑time Metrics")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(bottom=ymin)
    ax.legend(fontsize="small")
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=200)
    plt.close(fig)