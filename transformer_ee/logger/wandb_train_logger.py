"""
A logger module which logs the training information to wandb.
NOTE: wandb is not necessary for this project.
"""

import wandb
from .train_logger import BaseLogger


class WandBLogger(BaseLogger):
    """
    A class to log the training information to wandb.
    """

    def __init__(self, **kwargs):
        # Preserve existing behavior: pass kwargs straight through.
        # (Optionally: wandb.require("service") if you like, but not required.)
        self.run = wandb.init(**kwargs)

    # NEW: generic logger used by MVtrainer.train_LCL()
    def log(self, metrics: dict, step: int | None = None):
        # Backward compatible: if step is provided, include it and also
        # expose a "global_step" key for cleaner charts; otherwise behave as before.
        if step is not None:
            wandb.log({"global_step": int(step), **metrics}, step=int(step))
        else:
            wandb.log(metrics)

    # Keep existing API (used elsewhere); forward to the new log()
    def log_scalar(self, scalars: dict, step: int, epoch: int):
        # Preserve signature; epoch is unused here but kept for compatibility.
        self.log(scalars, step=step)

    def close(self):
        wandb.finish()
