"""
Feature correlation + permutation feature importance diagnostics for the
recovered DUNE atmospheric transformer_EE checkpoints.

Per model this produces (CERN-PID-style, matching the July 11 beam plots):
  * pearson_pred_<target>.png     Pearson r: input features vs PREDICTED target
  * pfi_<target>.png              permutation feature importance per target
                                  (fractional MSE increase)
  * pfi_mean_over_targets.png     mean fractional importance over all targets
  * resolution_<target>.png/.csv  binned bias / resolution vs true target
                                  (skipped for Topology)
  * pearson_vs_predicted.csv, pearson_vs_true.csv,
    pfi_delta_mse.csv, pfi_fractional_increase.csv, run_info.json

Recovery-aware behaviour (matches what survived the truncated archive):
  * model dir missing trainset_stat.json:
      - "Natural" models: stats are recomputed from this CSV using the model's
        own seed/split config. Exact, because this CSV *is* their training CSV.
      - "Flat" models: stats are borrowed from the sibling Flat model's
        trainset_stat.json. Exact too: both Flat runs trained on the same Flat
        CSV with the same split parameters, and stats depend only on features.
  * model dir missing input.json (the Flat momentum model):
      config is synthesized = sibling Flat config with "target" and "loss"
      taken from the momentum-variant donor config. Architecture correctness is
      verified implicitly by strict state_dict loading; the synthesized config
      is written to the output dir for the record.
"""

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description="Atmo model feature diagnostics")
    p.add_argument("--repo", required=True, help="Path to transformer_EE repo root")
    p.add_argument("--models-root", required=True,
                   help=r"...\recovered_models\SomeAtmoModels")
    p.add_argument("--csv", required=True, help="Atmospheric inference CSV")
    p.add_argument("--out", required=True, help="Output root directory")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--max-events", type=int, default=50000)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    return p.parse_args()


PALETTE = ["#C9A227", "#2E6FB0", "#2AA198", "#3B78C3", "#8E6BAE", "#B0562E"]
GREY = "#555555"


def bar_plot(values, labels, title, ylabel, color, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(range(len(values)), values, color=color, width=0.72)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def resolution_plot(truth, pred, name, color, out_dir):
    """Binned bias / resolution vs true target value.

    Uses fractional residual (pred-true)/true when the target is safely
    nonzero, absolute residual otherwise (momenta, cos theta)."""
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    frac_ok = np.percentile(np.abs(truth), 1) > 1e-3
    r = (pred - truth) / truth if frac_ok else (pred - truth)
    unit = "(pred-true)/true" if frac_ok else "pred-true"
    edges = np.unique(np.quantile(truth, np.linspace(0.01, 0.99, 11)))
    if len(edges) < 3:
        return
    idx = np.digitize(truth, edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 20:
            continue
        rows.append((float(truth[m].mean()), float(r[m].mean()),
                     float(r[m].std()), int(m.sum())))
    if len(rows) < 2:
        return
    x, bias, res, cnt = map(np.asarray, zip(*rows))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, res, "o-", color=color, label=f"resolution (std of {unit})")
    ax.plot(x, bias, "s--", color="#444444", label=f"bias (mean of {unit})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(unit)
    ax.set_title(f"Binned bias / resolution vs true {name}", fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"resolution_{name}.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame({"true_bin_center": x, "bias": bias, "resolution": res,
                  "n_events": cnt}).to_csv(
        os.path.join(out_dir, f"resolution_{name}.csv"), index=False)


# --------------------------------------------------------------- fast train stats
def finite_mean_std(values, name):
    """Return finite-only mean/std and fail clearly if a feature has no data."""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise RuntimeError(f"feature {name!r} has no finite values in the train split")
    mean = float(np.mean(a))
    std = float(np.std(a))
    if not np.isfinite(std) or std < 1e-10:
        print(f"[warn] feature {name}: invalid/tiny std={std}; using 1.0")
        std = 1.0
    return [mean, std]


def compute_feature_stats(df, idx, vector_cols, scalar_cols):
    """Vectorized replica of the repo's statistic(), ignoring NaN/Inf values."""
    stat = {}
    for v in vector_cols:
        s = df[v].iloc[idx]
        ok = s.map(lambda x: isinstance(x, str) and len(x.strip()) > 0)
        ex = (
            s.where(ok, "")
             .str.split(",")
             .explode()
        )
        ex = pd.to_numeric(ex, errors="coerce").to_numpy(np.float64)
        stat[v] = finite_mean_std(ex, v)

    for c in scalar_cols:
        a = pd.to_numeric(df[c].iloc[idx], errors="coerce").to_numpy(np.float64)
        stat[c] = finite_mean_std(a, c)

    return stat


def validate_stats(stat, feature_names, tag):
    """Reject corrupt normalization statistics before model inference."""
    for name in feature_names:
        if name not in stat:
            raise RuntimeError(f"{tag}: normalization stats missing feature {name!r}")

        pair = stat[name]
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            raise RuntimeError(f"{tag}: malformed stats for {name!r}: {pair!r}")

        mean = float(pair[0])
        std = float(pair[1])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise RuntimeError(
                f"{tag}: invalid normalization stats for {name!r}: "
                f"mean={mean}, std={std}"
            )


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.repo))

    import torch  # noqa: E402  (after CLI so --help is fast)
    from transformer_ee.dataloader.load import get_sample_indices  # noqa: E402
    from transformer_ee.dataloader.pd_dataset import (  # noqa: E402
        Normalized_pandas_Dataset_with_cache,
    )
    from transformer_ee.model import create_model  # noqa: E402
    from transformer_ee.utils.weights import NullWeights  # noqa: E402

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] device: {device}")

    # ------------------------------------------------------------ discover models
    runs = []
    for ck in sorted(glob.glob(os.path.join(args.models_root, "*", "model_*",
                                            "best_model.zip"))):
        mdir = os.path.dirname(ck)
        parent = os.path.basename(os.path.dirname(mdir))
        if parent.startswith("__"):
            continue
        spectrum = "Flat" if "_Flat_" in parent else "Natural"
        variant = "CosTheta" if "CosTheta" in parent else "Mom"
        runs.append({
            "tag": f"{spectrum}_{variant}",
            "spectrum": spectrum,
            "variant": variant,
            "parent": parent,
            "mdir": mdir,
            "ckpt": ck,
            "cfg_path": os.path.join(mdir, "input.json"),
            "stat_path": os.path.join(mdir, "trainset_stat.json"),
        })
    if not runs:
        sys.exit(f"[fatal] no */model_*/best_model.zip under {args.models_root}")
    print(f"[info] found {len(runs)} checkpoints:")
    for r in runs:
        print(f"        {r['tag']:<18s} {r['parent']}")

    # ------------------------------------------------------- resolve configs/stats
    for r in runs:
        r["cfg"] = None
        r["cfg_src"] = None
        if os.path.isfile(r["cfg_path"]):
            with open(r["cfg_path"], encoding="utf-8") as f:
                r["cfg"] = json.load(f)
            r["cfg_src"] = "own input.json"

    for r in runs:
        if r["cfg"] is not None:
            continue
        sibling = next((o for o in runs if o is not r and o["cfg"] and
                        o["spectrum"] == r["spectrum"]), None)
        donor = next((o for o in runs if o is not r and o["cfg"] and
                      o["variant"] == r["variant"]), None)
        if sibling is None or donor is None:
            sys.exit(f"[fatal] cannot reconstruct config for {r['parent']}")
        cfg = copy.deepcopy(sibling["cfg"])
        cfg["target"] = copy.deepcopy(donor["cfg"]["target"])
        if "loss" in donor["cfg"]:
            cfg["loss"] = copy.deepcopy(donor["cfg"]["loss"])
        r["cfg"] = cfg
        r["cfg_src"] = (f"RECONSTRUCTED: {sibling['tag']} config + "
                        f"target/loss from {donor['tag']}")
        print(f"[warn] {r['tag']}: input.json missing -> {r['cfg_src']}")
        print(f"[warn] {r['tag']}: target order assumed "
              f"{cfg['target']} (from directory-name ordering)")

    # ------------------------------------------------------------------- load CSV
    print(f"[info] reading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"[info] {len(df)} events, {len(df.columns)} columns")

    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    for r in runs:
        cfg = dict(r["cfg"])
        cfg["data_path"] = args.csv
        cfg["num_workers"] = 0
        if r["cfg"].get("dataframe_type") == "polars":
            print(f"[warn] {r['tag']}: config trained with polars; split indices "
                  f"here use the pandas convention, so '{args.split}' may not "
                  f"reproduce the training-time split. Treat results as a random "
                  f"evaluation sample rather than a strict holdout.")
        cfg["dataframe_type"] = "pandas"
        cfg.setdefault("seed", 0)
        cfg.pop("noise", None)  # noise is a train-time collate feature only

        tag = r["tag"]
        out_dir = os.path.join(args.out, tag)
        os.makedirs(out_dir, exist_ok=True)
        if r["cfg_src"].startswith("RECONSTRUCTED"):
            with open(os.path.join(out_dir, "reconstructed_input.json"), "w",
                      encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)

        vec_names = list(cfg["vector"])
        sca_names = list(cfg["scalar"])
        tgt_names = list(cfg["target"])
        feat_labels = sca_names + vec_names

        # ------------------------------------------------------------- split rows
        idx_train, idx_valid, idx_test = get_sample_indices(len(df), cfg)
        sel = {"train": idx_train, "valid": idx_valid, "test": idx_test}[args.split]
        sel = sel[: args.max_events]
        print(f"[info] {tag}: {len(sel)} events from '{args.split}' split")

        # ------------------------------------------------------- training stats
        if os.path.isfile(r["stat_path"]):
            with open(r["stat_path"], encoding="utf-8") as f:
                stat = json.load(f)
            stat_src = "own trainset_stat.json"
        elif r["spectrum"] == "Natural":
            print(f"[warn] {tag}: trainset_stat.json missing -> recomputing "
                  f"from this CSV's train split (exact: this IS its training CSV)")
            stat = compute_feature_stats(df, idx_train, vec_names, sca_names)
            stat_src = "recomputed from CSV train split"
        else:
            sib = next((o for o in runs if o is not r and
                        o["spectrum"] == r["spectrum"] and
                        os.path.isfile(o["stat_path"])), None)
            if sib is None:
                sys.exit(f"[fatal] no stats source for {r['parent']}")
            for k in ("seed", "test_size", "valid_size"):
                if cfg.get(k) != sib["cfg"].get(k):
                    print(f"[warn] {tag}: split param {k} differs from sibling; "
                          f"borrowed stats may be approximate")
            with open(sib["stat_path"], encoding="utf-8") as f:
                stat = json.load(f)
            stat_src = f"borrowed from sibling {sib['tag']}"
            print(f"[warn] {tag}: trainset_stat.json missing -> {stat_src} "
                  f"(same Flat training CSV + split params => identical stats)")

        missing = [k for k in vec_names + sca_names if k not in stat]
        if missing:
            sys.exit(f"[fatal] {tag}: stats missing for {missing}")
        validate_stats(stat, vec_names + sca_names, tag)

        # ------------------------------------------------- dataset + tensors
        sub = df.iloc[sel].reset_index(drop=True).copy()
        ds = Normalized_pandas_Dataset_with_cache(cfg, sub, weighter=NullWeights())
        ds.normalize(stat)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
        Vs, Ss, Ms, Ts = [], [], [], []
        for vec, sca, mask, tgt, _w in loader:
            Vs.append(vec); Ss.append(sca); Ms.append(mask); Ts.append(tgt)
        V = torch.cat(Vs).float()
        S = torch.cat(Ss).float()
        M = torch.cat(Ms).bool()
        T = torch.cat(Ts).float()
        del Vs, Ss, Ms, Ts

        def nonfinite_count(x):
            return int((~torch.isfinite(x)).sum().item())

        print(
            f"[check] {tag}: non-finite before cleaning: "
            f"V={nonfinite_count(V)}, "
            f"S={nonfinite_count(S)}, "
            f"T={nonfinite_count(T)}"
        )

        # Invalid truth values cannot be used for MSE or correlation metrics.
        # Remove those events rather than replacing their targets with zero.
        good_rows = torch.isfinite(T).all(dim=1)
        if not good_rows.all():
            n_removed = int((~good_rows).sum().item())
            print(f"[warn] {tag}: removing {n_removed} rows with invalid targets")
            V = V[good_rows]
            S = S[good_rows]
            M = M[good_rows]
            T = T[good_rows]

        if V.shape[0] == 0:
            raise RuntimeError(f"{tag}: no valid events remain after target filtering")

        # Inputs are normalized, so zero means the training-set mean.
        # Replace NaN/Inf inputs with zero and force padded prongs to zero.
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        V = V.masked_fill(M.unsqueeze(-1), 0.0)

        # A Transformer receives NaNs when every token in a row is masked.
        # Insert one zero-valued dummy token for such events.
        all_masked = M.all(dim=1)
        if all_masked.any():
            n_all_masked = int(all_masked.sum().item())
            print(
                f"[warn] {tag}: {n_all_masked} events have every prong masked; "
                "unmasking one zero-valued dummy token"
            )
            M[all_masked, 0] = False
            V[all_masked, 0, :] = 0.0

        print(
            f"[check] {tag}: non-finite after cleaning: "
            f"V={nonfinite_count(V)}, "
            f"S={nonfinite_count(S)}, "
            f"T={nonfinite_count(T)}"
        )
        print(
            f"[info] {tag}: tensors V{tuple(V.shape)} "
            f"S{tuple(S.shape)} T{tuple(T.shape)}"
        )

        # ------------------------------------------------------------ model
        model = create_model(cfg)
        state = torch.load(r["ckpt"], map_location="cpu")
        model.load_state_dict(state)  # strict: verifies architecture
        model.to(device).eval()

        def predict(m, Vt, St, Mt):
            outs = []
            with torch.no_grad():
                for s in range(0, Vt.shape[0], args.batch_size):
                    e = s + args.batch_size
                    outs.append(m(Vt[s:e].to(device), St[s:e].to(device),
                                  Mt[s:e].to(device)).cpu())
            return torch.cat(outs)

        P = predict(model, V, S, M)

        if not torch.isfinite(P).all():
            bad_rows = ~torch.isfinite(P).all(dim=1)
            bad_outputs = (~torch.isfinite(P)).sum(dim=0)
            print(f"[fatal] {tag}: model produced non-finite predictions")
            print(f"[fatal] affected rows: {int(bad_rows.sum())}/{P.shape[0]}")
            print(f"[fatal] bad values per output: {bad_outputs.tolist()}")
            raise RuntimeError(
                f"{tag}: predictions contain NaN/Inf after input cleaning. "
                "Check normalization statistics and checkpoint compatibility."
            )

        Tn = T.numpy()
        Pn = P.numpy()

        def per_target_mse(pred):
            mse = ((pred - T) ** 2).mean(dim=0)
            if not torch.isfinite(mse).all():
                raise RuntimeError(f"{tag}: non-finite MSE encountered")
            return mse.numpy()

        base_mse = per_target_mse(P)
        for i, t in enumerate(tgt_names):
            print(f"[info] {tag}: baseline {t:>16s}  MSE {base_mse[i]:.6f}")

        # ------------------------------------- feature matrix for correlations
        valid = (~M).unsqueeze(-1).float()              # (N, L, 1) 1=real prong
        cnt = valid.sum(dim=1).clamp(min=1.0)           # (N, 1)
        vec_mean = (V * valid).sum(dim=1) / cnt         # (N, nvec)
        X = np.concatenate([S.numpy(), vec_mean.numpy()], axis=1)
        with np.errstate(invalid="ignore"):
            col_mean = np.nanmean(X, axis=0)

        all_bad_columns = ~np.isfinite(col_mean)
        if all_bad_columns.any():
            names = [feat_labels[i] for i in np.flatnonzero(all_bad_columns)]
            print(
                f"[warn] {tag}: completely non-finite correlation columns: "
                f"{names}; replacing them with zero"
            )
            col_mean[all_bad_columns] = 0.0

        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(col_mean, np.nonzero(bad)[1])

        def pearson_matrix(Y):
            out = np.zeros((X.shape[1], Y.shape[1]))
            Xc = X - X.mean(0); Xs = X.std(0) + 1e-12
            Yc = Y - Y.mean(0); Ys = Y.std(0) + 1e-12
            out = (Xc / Xs).T @ (Yc / Ys) / X.shape[0]
            return out

        corr_pred = pearson_matrix(Pn)
        corr_true = pearson_matrix(Tn)
        pd.DataFrame(corr_pred, index=feat_labels, columns=tgt_names).to_csv(
            os.path.join(out_dir, "pearson_vs_predicted.csv"))
        pd.DataFrame(corr_true, index=feat_labels, columns=tgt_names).to_csv(
            os.path.join(out_dir, "pearson_vs_true.csv"))
        for i, t in enumerate(tgt_names):
            bar_plot(corr_pred[:, i], feat_labels,
                     f"[{tag}] Pearson correlation: input features vs predicted {t}",
                     "Pearson r", PALETTE[i % len(PALETTE)],
                     os.path.join(out_dir, f"pearson_pred_{t}.png"))

        # --------------------------------------- permutation feature importance
        rng = np.random.default_rng(0)
        nfeat = len(feat_labels)
        delta = np.zeros((nfeat, len(tgt_names)))
        for fi, name in enumerate(feat_labels):
            accum = np.zeros(len(tgt_names))
            for _ in range(args.n_repeats):
                perm = rng.permutation(V.shape[0])
                if fi < len(sca_names):
                    Sp = S.clone(); Sp[:, fi] = S[perm, fi]
                    Pm = predict(model, V, Sp, M)
                else:
                    vi = fi - len(sca_names)
                    Vp = V.clone(); Vp[:, :, vi] = V[perm, :, vi]
                    Pm = predict(model, Vp, S, M)
                accum += per_target_mse(Pm)
            delta[fi] = accum / args.n_repeats - base_mse
            print(f"[pfi ] {tag}: {name:>36s}  mean dMSE {delta[fi].mean():.6f}")

        frac = delta / np.maximum(base_mse, 1e-12)[None, :]
        pd.DataFrame(delta, index=feat_labels, columns=tgt_names).to_csv(
            os.path.join(out_dir, "pfi_delta_mse.csv"))
        pd.DataFrame(frac, index=feat_labels, columns=tgt_names).to_csv(
            os.path.join(out_dir, "pfi_fractional_increase.csv"))
        for i, t in enumerate(tgt_names):
            bar_plot(frac[:, i], feat_labels,
                     f"[{tag}] Permutation feature importance for {t} "
                     f"(n_repeats={args.n_repeats})",
                     "fractional MSE increase", PALETTE[i % len(PALETTE)],
                     os.path.join(out_dir, f"pfi_{t}.png"))
        bar_plot(frac.mean(axis=1), feat_labels,
                 f"[{tag}] Permutation feature importance (mean over targets)",
                 "mean fractional MSE increase", GREY,
                 os.path.join(out_dir, "pfi_mean_over_targets.png"))

        # ------------------------------------------------- bias / resolution
        for i, t in enumerate(tgt_names):
            if "topolog" in t.lower():
                continue
            resolution_plot(Tn[:, i], Pn[:, i], t,
                            PALETTE[i % len(PALETTE)], out_dir)

        manifest[tag] = {
            "model_dir": r["mdir"],
            "checkpoint": r["ckpt"],
            "config_source": r["cfg_src"],
            "stats_source": stat_src,
            "split": args.split,
            "n_events": int(V.shape[0]),
            "n_repeats": args.n_repeats,
            "targets": tgt_names,
            "baseline_mse": {t: float(base_mse[i])
                             for i, t in enumerate(tgt_names)},
        }
        with open(os.path.join(out_dir, "run_info.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest[tag], f, indent=4)
        del model, V, S, M, T, P
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {tag} -> {out_dir}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print(f"\n[done] all models. Outputs under: {args.out}")


if __name__ == "__main__":
    main()