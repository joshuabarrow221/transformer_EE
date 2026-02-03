#!/usr/bin/env python3
"""
Generate batch_inference_config.*.json files for transformer_ee batch inference.

Reads one or more "model list" text files (e.g. Train_Atmospheric_Flat_Models_*.txt)
and extracts unique trained-model training names, then pairs Vector models with
Vector samples and Scalar models with Scalar samples.

Only standard-library Python is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_MODEL_SEARCH_ROOTS = [
    "/exp/dune/data/users/cborden/MLProject/Training_Samples",
    "/exp/dune/data/users/rrichi/MLProject/Training_Samples",
    "/exp/dune/data/users/jbarrow/MLProject/Training_Samples",
]

# --- Sample paths from Josh's prompt (edit here if your paths change) ---
SAMPLES = {
    # 1) DUNEAtmFlat-to-DUNEAtmNat
    "batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json": [
        ("Vector", "DUNEAtmo_Nat_p1to10_NpNpi_Vector",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Atmospherics_DUNE_Like/Natural_Spectra/Numu_CC_Train_DUNEAtmo_Natural_p1to10_VectorLeptwNC_eventnum_All_NpNpi.csv"),
        ("Scalar", "DUNEAtmo_Nat_p1to10_NpNpi_Scalar",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Atmospherics_DUNE_Like/Natural_Spectra/Numu_CC_Train_DUNEAtmo_Natural_p1to10_ScalarLeptwNC_eventnum_All_NpNpi.csv"),
    ],

    # 2) DUNEBeamFlat-to-DUNEFDBeamOsc
    "batch_inference_config.DUNEBeamFlat-to-DUNEFDBeamOsc.json": [
        ("Vector", "DUNEBeam_Nat_OnAxisFD_p1to6_NpNpi_Vector",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisFDOsc/Numu_CC_Train_DUNEBeam_Natural_OnAxisFD_p1to6_VectorLeptwNC_eventnum_All_NpNpi.csv"),
        ("Scalar", "DUNEBeam_Nat_OnAxisFD_p1to6_NpNpi_Scalar",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisFDOsc/Numu_CC_Train_DUNEBeam_Natural_OnAxisFD_p1to6_ScalarLeptwNC_eventnum_All_NpNpi.csv"),
    ],

    # 3) DUNEBeamFlat-to-DUNEND39mOffAxisBeamNat
    "batch_inference_config.DUNEBeamFlat-to-DUNEND39mOffAxisBeamNat.json": [
        ("Vector", "DUNEBeam_Nat_OffAxisND39m_p1to2_NpNpi_Vector",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNE39mOffAxis/Numu_CC_Train_DUNEBeam_Natural_OffAxisND_p1to2_VectorLeptwNC_eventnum_All_NpNpi.csv"),
        ("Scalar", "DUNEBeam_Nat_OffAxisND39m_p1to2_NpNpi_Scalar",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNE39mOffAxis/Numu_CC_Train_DUNEBeam_Natural_OffAxisND_p1to2_ScalarLeptwNC_eventnum_All_NpNpi.csv"),
    ],

    # 4) DUNEBeamFlat-to-DUNENDBeamNat
    "batch_inference_config.DUNEBeamFlat-to-DUNENDBeamNat.json": [
        ("Vector", "DUNEBeam_Nat_OnAxisND_p1to6_NpNpi_Vector",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisND/Numu_CC_Train_DUNEBeam_Natural_OnAxisND_p1to6_VectorLeptwNC_eventnum_All_NpNpi.csv"),
        ("Scalar", "DUNEBeam_Nat_OnAxisND_p1to6_NpNpi_Scalar",
         "/exp/dune/data/users/rrichi/MLProject/Training_Samples/Beam_Like/Natural_Spectra/DUNEOnAxisND/Numu_CC_Train_DUNEBeam_Natural_OnAxisND_p1to6_ScalarLeptwNC_eventnum_All_NpNpi.csv"),
    ],
}

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def _model_kind(training_name: str) -> str:
    tn = training_name.lower()
    if "vector" in tn:
        return "Vector"
    if "scalar" in tn:
        return "Scalar"
    return "Unknown"

def _make_model_name(training_name: str) -> str:
    """Create a stable, unique, *short-ish* name for the config's 'models[].name'."""
    kind = _model_kind(training_name)
    tag = "DUNEAtmo" if "DUNEAtmo" in training_name else ("DUNEBeam" if "DUNEBeam" in training_name else "Model")
    h = hashlib.sha1(training_name.encode("utf-8")).hexdigest()[:10]

    toks = training_name.split("_")
    tail = "_".join(toks[-8:])  # keep the last few loss tokens + Topology_MAE, etc
    human = _slug(tail)[:50]
    name = f"{tag}_{kind}_{human}_{h}"
    return name[:160]

def _extract_training_names(paths: Iterable[Path], require_substr: str) -> List[str]:
    """
    Extract training names from the text documents.

    Heuristic: keep lines that
      - start with "Numu_CC_Train_"
      - contain require_substr (e.g. "DUNEAtmo_Flat" or "DUNEBeam_Flat")
      - contain "_Topology_" (to avoid grabbing raw datasets / csv/root filenames)
      - do NOT end in .csv/.root/.json/.txt
    """
    out = set()
    for p in paths:
        for raw in p.read_text(errors="replace").splitlines():
            s = raw.strip()
            if not s:
                continue
            if not s.startswith("Numu_CC_Train_"):
                continue
            if require_substr not in s:
                continue
            if s.endswith((".csv", ".root", ".json", ".txt")):
                continue
            if "_Topology_" not in s:
                continue
            out.add(s)
    return sorted(out)

def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak{ts}")
    path.rename(bak)

def _build_config(training_names: List[str], samples: List[Tuple[str, str, str]], model_search_roots: List[str]) -> Dict:
    model_objs = []
    pairs = []

    sample_objs = [{"name": name, "path": path} for (_kind, name, path) in samples]
    sample_by_kind = {kind: name for (kind, name, _path) in samples}

    used_names = set()
    for tn in training_names:
        mn = _make_model_name(tn)
        if mn in used_names:
            # should never happen, but keep it bulletproof
            mn = f"{mn}_{hashlib.sha1((tn+'x').encode('utf-8')).hexdigest()[:14]}"
        used_names.add(mn)
        model_objs.append({"name": mn, "training_name": tn})

        k = _model_kind(tn)
        if k not in sample_by_kind:
            continue
        pairs.append({"model": mn, "sample": sample_by_kind[k]})

    return {
        "model_search_roots": model_search_roots,
        "models": model_objs,
        "samples": sample_objs,
        "pairs": pairs,
    }

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="Where to write batch_inference_config.*.json")
    ap.add_argument("--atm-files", nargs="+", default=[
        "Train_Atmospheric_Flat_Models_rrichi.txt",
        "Train_Atmospheric_Flat_Models_jbarrow.txt",
        "Train_Atmospheric_Flat_Models_cborden.txt",
    ], help="Atmospheric flat model list text files")
    ap.add_argument("--beam-files", nargs="+", default=[
        "Train_DUNEBeam_Flat_Models_rrichi.txt",
        "Train_DUNEBeam_Flat_Models_jbarrow.txt",
        "Train_DUNEBeam_Flat_Models_cborden.txt",
    ], help="Beam flat model list text files")
    ap.add_argument("--model-search-roots", nargs="+", default=DEFAULT_MODEL_SEARCH_ROOTS)
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak* backups when output exists")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    atm_paths = [Path(p) for p in args.atm_files]
    beam_paths = [Path(p) for p in args.beam_files]

    missing = [str(p) for p in (atm_paths + beam_paths) if not p.exists()]
    if missing:
        print("ERROR: missing input files:", *missing, sep="\n  - ", file=sys.stderr)
        return 2

    atm_models = _extract_training_names(atm_paths, require_substr="DUNEAtmo_Flat")
    beam_models = _extract_training_names(beam_paths, require_substr="DUNEBeam_Flat")

    # Build + write all configs
    specs = {
        "batch_inference_config.DUNEAtmFlat-to-DUNEAtmNat.json": atm_models,
        "batch_inference_config.DUNEBeamFlat-to-DUNEFDBeamOsc.json": beam_models,
        "batch_inference_config.DUNEBeamFlat-to-DUNEND39mOffAxisBeamNat.json": beam_models,
        "batch_inference_config.DUNEBeamFlat-to-DUNENDBeamNat.json": beam_models,
    }

    for outname, models in specs.items():
        samples = SAMPLES[outname]
        cfg = _build_config(models, samples, args.model_search_roots)
        outpath = outdir / outname
        if outpath.exists() and not args.no_backup:
            _backup_if_exists(outpath)
        outpath.write_text(json.dumps(cfg, indent=2))
        print(f"Wrote {outpath}  (models={len(cfg['models'])}, samples={len(cfg['samples'])}, pairs={len(cfg['pairs'])})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
