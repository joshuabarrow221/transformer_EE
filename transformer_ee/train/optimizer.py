"""
A function to create an optimizer from a config.
"""

import torch.optim as optim
import torch.nn as nn


def create_optimizer(config: dict, model: nn.Module):
    """
    Create an optimizer from a config.
    Args:
        config:
        {
            "optimizer": {
                "name": <str>,        # e.g. "AdamW" or "adamw"
                "kwargs": <dict>      # args for the optimizer ctor
            },
            ...
        }
        model: an nn.Module to optimize

    Returns:
        an optimizer
    """
    # --- Backward-compatible reads ---
    opt_block = config.get("optimizer", {})
    name_raw = opt_block.get("name", "sgd")
    name = str(name_raw).lower()                 # accept "AdamW" and "adamw"
    kw = dict(opt_block.get("kwargs", {}) or {}) # safe copy; default to {}

    # (Optional but harmless) If some configs accidentally pass None values,
    # strip them so torch doesn't choke on kwarg=None.
    kw = {k: v for k, v in kw.items() if v is not None}

    # --- Mapping (unchanged behavior; just case-insensitive) ---
    if name == "adam":
        optimizer = optim.Adam(model.parameters(), **kw)
    elif name == "adamw":
        optimizer = optim.AdamW(model.parameters(), **kw)
    elif name == "adamax":
        optimizer = optim.Adamax(model.parameters(), **kw)
    elif name == "sgd":
        optimizer = optim.SGD(model.parameters(), **kw)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), **kw)
    else:
        # Avoid attribute access on dict in the error path
        raise ValueError("Unsupported optimizer: {}".format(name_raw))

    return optimizer
