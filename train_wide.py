#!/usr/bin/env python3
# Flexible launch script for Transformer_EE_MV with widened d_model & extras.

import json
import os
import argparse
from copy import deepcopy

from transformer_ee.train import MVtrainer
from transformer_ee.logger.wandb_train_logger import WandBLogger
import wandb


# ---------- Helpers ----------
def kset(d, dotted_key, value):
    """Set a nested config key using dotted path, creating parents as needed."""
    if value is None:
        return
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def ensure_model_kwargs(cfg):
    """Make sure cfg['model']['kwargs'] exists."""
    if "model" not in cfg:
        cfg["model"] = {}
    if "kwargs" not in cfg["model"]:
        cfg["model"]["kwargs"] = {}
    return cfg["model"]["kwargs"]


def fail_if(cond, msg):
    if cond:
        raise ValueError(msg)


def validate_and_autofill(cfg):
    """
    Validate divisibility constraints and fill sensible defaults
    (keeps legacy behavior if you don't opt in).
    """
    mkw = ensure_model_kwargs(cfg)

    nhead = int(mkw.get("nhead", 4))
    # Legacy behavior (no widening) if both d_model and auto_d_model are absent/False.
    explicit_d = mkw.get("d_model", None)
    auto_d = bool(mkw.get("auto_d_model", False))

    if explicit_d is not None:
        explicit_d = int(explicit_d)
        fail_if(explicit_d % nhead != 0,
                f"d_model={explicit_d} must be divisible by nhead={nhead}")
        # If user didn't set dim_feedforward, give a sane default tied to width
        if "dim_feedforward" not in mkw:
            mkw["dim_feedforward"] = max(4 * explicit_d, 128)

    # If auto_d_model is true, the class will compute d_model internally,
    # but we still warn if the user set incompatible min_head_dim & nhead.
    if auto_d:
        # Nothing to validate here beyond types; real selection happens in model.
        # Still: if user supplied d_model together with auto_d_model, prefer explicit.
        if explicit_d is not None:
            # Be explicit about precedence to avoid silent surprises.
            print("[WARN] Both d_model and auto_d_model were set; using explicit d_model.")

    # If neither explicit nor auto, legacy path: d_model == len(vector),
    # which *must* be divisible by nhead. We can’t check that here without opening the
    # base JSON config, but the model will raise with a clear error if it’s not.

    # Suggest a dim_feedforward if user widened but forgot to scale it.
    if explicit_d is not None and "dim_feedforward" in mkw:
        dff = int(mkw["dim_feedforward"])
        if dff < 2 * explicit_d:
            print(f"[WARN] dim_feedforward={dff} may be small for d_model={explicit_d}. "
                  f"Consider >= {4 * explicit_d}.")


def apply_profile(cfg, profile):
    """
    Fast presets so you can A/B ideas without typing a wall of flags.
    Profiles only *set/override* a few keys; everything else remains unchanged.
    """
    mkw = ensure_model_kwargs(cfg)

    if profile == "legacy":
        # exact legacy behavior; nothing to override
        pass

    elif profile == "wide":
        # widen with explicit d_model; keep legacy head and pooling
        mkw["d_model"] = max(128, mkw.get("nhead", 4) * 32)  # safeish starting point
        mkw["auto_d_model"] = False
        mkw["scalar_as_token"] = False
        mkw["use_new_head"] = False
        mkw.setdefault("dim_feedforward", max(4 * mkw["d_model"], 128))

    elif profile == "auto":
        # let the model compute d_model; keep legacy head
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = False
        mkw["use_new_head"] = False
        # dim_feedforward will be set in the model layer if not provided

    elif profile == "token":
        # scalar token + auto width; still use legacy head for apples-to-apples
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = True
        mkw["use_new_head"] = False

    elif profile == "newhead":
        # scalar token + auto width + modern head
        mkw["auto_d_model"] = True
        mkw["min_head_dim"] = 16
        mkw["base_multiple"] = 4
        mkw["scalar_as_token"] = True
        mkw["use_new_head"] = True

    else:
        fail_if(True, f"Unknown profile '{profile}'. Valid: legacy|wide|auto|token|newhead")


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Launch Transformer_EE_MV with flexible d_model / scalar token / head."
    )

    # Core I/O
    p.add_argument("--base-config",
                   default="/home/cborden/git/josh_transformer_EE/transformer_EE/transformer_ee/config/wEID/input_GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_E_Px_Py_Pz_EID.json")
    p.add_argument("--data-path",
                   default="/exp/dune/app/users/rrichi/FinalCodes/Numu_CC_Thresh_p1to1_VectorLeptWithoutNC_eventnum_All_NpNpi.csv")
    p.add_argument("--save-path",
                   default="/exp/dune/data/users/cborden/save_genie-dmodel_TEST/model/test/GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_E_Px_Py_Pz_EID")
    p.add_argument("--dataframe-type", default="polars")
    p.add_argument("--num-workers", type=int, default=10)

    # Training / Optim
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)  # only for SGD
    p.add_argument("--weight-decay", type=float, default=None)

    # Transformer knobs (override as needed)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--dim-ff", type=int, default=None)

    # Width control
    p.add_argument("--d-model", type=int, default=None,
                   help="Explicit transformer width (must be divisible by nhead).")
    p.add_argument("--auto-d-model", action="store_true",
                   help="If set, pick d_model automatically inside the model.")
    p.add_argument("--min-head-dim", type=int, default=None,
                   help="Used when --auto-d-model is set. Head dim >= this.")
    p.add_argument("--base-multiple", type=int, default=None,
                   help="Used when --auto-d-model is set. d_model >= base_multiple * nvec.")

    # Sequence & head style
    p.add_argument("--scalar-as-token", action="store_true",
                   help="Prepend a scalar-derived token to the sequence.")
    p.add_argument("--use-new-head", action="store_true",
                   help="LayerNorm+GELU MLP head on pooled token. If off, legacy head is used.")
    p.add_argument("--norm-first", action="store_true",
                   help="Use norm_first=True in TransformerEncoderLayer.")

    # fast presets
    p.add_argument("--profile", default=None,
                   help="Quick preset: legacy|wide|auto|token|newhead")

    # WandB
    p.add_argument("--wandb-project", default="GENIE_Atmo")
    p.add_argument("--wandb-entity", default="neutrinoenenergyestimators")
    p.add_argument("--wandb-dir", default="/exp/dune/data/users/cborden/save_genie-dmodel_TEST/wandb")
    p.add_argument("--wandb-id",
                   default="GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_E_Px_Py_Pz_EID")

    return p


def main():
    args = build_argparser().parse_args()

    # Load base config
    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # --- Apply common overrides (I/O, dataloader, training) ---
    kset(cfg, "data_path", args.data_path)
    kset(cfg, "save_path", args.save_path)
    kset(cfg, "dataframe_type", args.dataframe_type)
    kset(cfg, "num_workers", args.num_workers)

    kset(cfg, "model.epochs", args.epochs)

    # Optimizer
    kset(cfg, "optimizer.name", args.optimizer)
    kset(cfg, "optimizer.kwargs.lr", args.lr)
    if args.optimizer == "sgd":
        kset(cfg, "optimizer.kwargs.momentum", args.momentum)
    if args.weight_decay is not None:
        kset(cfg, "optimizer.kwargs.weight_decay", args.weight_decay)

    # Transformer basics
    mkw = ensure_model_kwargs(cfg)
    mkw["nhead"] = int(args.nhead) if args.nhead is not None else mkw.get("nhead", 4)
    mkw["num_layers"] = int(args.num_layers) if args.num_layers is not None else mkw.get("num_layers", 6)
    if args.dropout is not None:
        mkw["dropout"] = float(args.dropout)
    if args.dim_ff is not None:
        mkw["dim_feedforward"] = int(args.dim_ff)

    # Profiles (apply late, they override the above if needed)
    if args.profile:
        apply_profile(cfg, args.profile)

    # Width / style toggles (explicit CLI wins over profile)
    if args.d_model is not None:
        mkw["d_model"] = int(args.d_model)
        mkw["auto_d_model"] = False
    if args.auto_d_model:
        mkw["auto_d_model"] = True
    if args.min_head_dim is not None:
        mkw["min_head_dim"] = int(args.min_head_dim)
    if args.base_multiple is not None:
        mkw["base_multiple"] = int(args.base_multiple)
    if args.scalar_as_token:
        mkw["scalar_as_token"] = True
    if args.use_new_head:
        mkw["use_new_head"] = True
    if args.norm_first:
        mkw["norm_first"] = True

    # Validate & add implied defaults
    validate_and_autofill(cfg)

    # --- WandB logger ---
    my_logger = WandBLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=cfg,
        dir=args.wandb_dir,
        id=args.wandb_id,
    )

    my_trainer = MVtrainer(cfg, logger=my_logger)
    my_trainer.train_LCL()
    my_trainer.eval()


if __name__ == "__main__":
    main()
