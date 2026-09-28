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
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_MODEL_SEARCH_ROOTS = [
    "/exp/dune/data/users/cborden/MLProject/Training_Samples",
    "/exp/dune/data/users/rrichi/MLProject/Training_Samples",
    "/exp/dune/data/users/jbarrow/MLProject/Training_Samples",
]

# Flat single-variable models without training noise, evaluated on three GENIE tunes.
# Keep output names and sample definitions together so they cannot drift apart.
SAMPLES = {}
for tune, label in (("G18_10a_02_11a", "G1810a0211a"),
                    ("G18_10a_02_11b", "G1810a0211b"),
                    ("G21_11a", "G2111a")):
    outname = f"batch_inference_config.DUNEBeamFlat_SV_NoNoise-to-DUNENDBeamNatNDOnAxis_GENIE_{label}.json"
    SAMPLES[outname] = [
        (kind, f"DUNEBeam_Nat_OnAxisND_p1to5_NpNpi_{kind}_{label}",
         "/exp/dune/data/users/jbarrow/MLProject/Inference_Samples/DUNEOnAxisND/GENIE_CMCs/"
         f"{tune}_DUNENDNat_Numu_CC_Thresh_p1to5_eventnum_{kind}LeptwNC_eventnum_All_NpNpi.csv")
        for kind in ("Vector", "Scalar")
    ]


def _model_kind(training_name: str) -> str:
    tn = training_name.lower()
    if "vector" in tn:
        return "Vector"
    if "scalar" in tn:
        return "Scalar"
    return "Unknown"

def _user_code_from_path(path: Path) -> str:
    user = path.stem.split("_")[-1].lower()
    return {"jbarrow": "J", "cborden": "C", "rrichi": "R"}.get(user, "U")

def _user_from_code(user_code: str) -> str:
    return {"J": "jbarrow", "C": "cborden", "R": "rrichi"}.get(user_code, "")

def _extract_training_names(
    paths: Iterable[Path],
    require_substrs: Iterable[str] | None = None,
) -> List[Tuple[str, str]]:
    """
    Extract training names from the text documents.

    Heuristic: keep lines that
      - start with "Numu_CC_Train_"
      - contain all required substrings (if provided)
      - contain "_Topology_" (to avoid grabbing raw datasets / csv/root filenames)
      - do NOT end in .csv/.root/.json/.txt
    """
    out = []
    seen = set()
    required = [s for s in (require_substrs or []) if s]
    for p in paths:
        user_code = _user_code_from_path(p)
        for raw in p.read_text(errors="replace").splitlines():
            s = raw.strip()
            if not s:
                continue
            if not s.startswith("Numu_CC_Train_"):
                continue
            if required and not all(req in s for req in required):
                continue
            if s.endswith((".csv", ".root", ".json", ".txt")):
                continue
            if "_Topology_" not in s:
                continue
            key = (s, user_code)
            if key in seen:
                continue
            seen.add(key)
            out.append((s, user_code))
    return out

def _spectrum_kind(training_name: str) -> str:
    tn = training_name.lower()
    if "flat" in tn:
        return "flat"
    if "natural" in tn or "nat" in tn:
        return "natural"
    return "unknown"

def _beam_family(training_name: str) -> str:
    tn = training_name.lower()
    if "nova" in tn:
        return "nova"
    if "dunebeam" in tn or ("dune" in tn and "beam" in tn):
        return "dune"
    return "unknown"

def _filter_models(
    training_names: List[Tuple[str, str]],
    *,
    spectrum_kind: str | None = None,
    beam_family: str | None = None,
) -> List[Tuple[str, str]]:
    out = []
    for tn, user_code in training_names:
        if spectrum_kind and _spectrum_kind(tn) != spectrum_kind:
            continue
        if beam_family and _beam_family(tn) != beam_family:
            continue
        out.append((tn, user_code))
    return out

def _resolve_model_dirs(
    training_name: str,
    user_code: str,
    model_search_roots: List[str],
) -> List[Path]:
    user = _user_from_code(user_code)
    candidates: List[Path] = []
    for root in (Path(p) for p in model_search_roots):
        if user and user not in root.parts:
            continue
        if not root.exists():
            continue
        for path in root.rglob("input.json"):
            if training_name in path.parts and (path.parent / "best_model.zip").exists():
                candidates.append(path.parent.resolve())
    return sorted({path.resolve() for path in candidates}, key=lambda p: str(p))

def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak{ts}")
    path.rename(bak)

def _build_config(
    training_names: List[Tuple[str, str]],
    samples: List[Tuple[str, str, str]],
    model_search_roots: List[str],
) -> Dict:
    model_objs = []
    pairs = []

    sample_objs = [{"name": name, "path": path} for (_kind, name, path) in samples]
    sample_by_kind = {kind: name for (kind, name, _path) in samples}

    for tn, user_code in training_names:
        resolved_dirs = _resolve_model_dirs(tn, user_code, model_search_roots)
        if not resolved_dirs:
            resolved_dirs = [None]
        for idx, model_dir in enumerate(resolved_dirs, start=1):
            mn = f"{tn}_{user_code}{idx}"
            if model_dir is None:
                model_objs.append({"name": mn, "training_name": tn})
            else:
                model_objs.append({"name": mn, "path": str(model_dir)})

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
    ap.add_argument("--atm-files", nargs="*", default=[],
                    help="Optional atmospheric model lists (unused by this beam-only study)")
    ap.add_argument("--beam-files", nargs="+", default=[
        str(Path(__file__).with_name("Train_DUNEBeamND_Flat_Models_woNoise_SVs.txt")),
    ], help="Beam model lists; defaults to the bundled flat no-noise single-variable models")
    ap.add_argument("--atm-require-substrs", nargs="*", default=[],
                    help="Optional substrings required in atmospheric training names")
    ap.add_argument("--beam-require-substrs", nargs="*", default=[],
                    help="Optional substrings required in beam training names")
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

    beam_models = _extract_training_names(beam_paths, require_substrs=args.beam_require_substrs)

    beam_flat_models = _filter_models(beam_models, spectrum_kind="flat", beam_family="dune")
    if not beam_flat_models:
        print("ERROR: no DUNE flat beam models detected; check beam file contents or "
              "--beam-require-substrs filters.", file=sys.stderr)
        return 2

    specs = {outname: beam_flat_models for outname in SAMPLES}

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
