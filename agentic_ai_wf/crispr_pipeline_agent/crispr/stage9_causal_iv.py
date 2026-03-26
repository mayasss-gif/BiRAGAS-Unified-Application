#!/usr/bin/env python3
"""
Stage-9: Causal Inference (IV / 2SLS)

Runs robust 2-stage least squares:
  First stage:   T ~ Z + X
  Second stage:  Y ~ T_hat + X

Z = predicted perturbation label (instrument), one-hot encoded
T = treatment (default pca:0)
Y = outcome   (default pca:1 or gene_score:...)

Outputs:
  processed_stage9/tables/stage9_iv_results.tsv
  processed_stage9/tables/stage9_diagnostics.tsv
  processed_stage9/figures/stage9_first_stage_top_instruments.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def detect_prediction_cols(obs: pd.DataFrame) -> Tuple[str, Optional[str]]:
    label_candidates = ["pred_label_stage7", "pred_class_stage7", "pred_label", "pred_class"]
    conf_candidates = ["pred_confidence_stage7", "pred_conf_stage7", "pred_confidence", "pred_conf"]

    label_col = next((c for c in label_candidates if c in obs.columns), None)
    if label_col is None:
        raise ValueError(f"Could not find predicted label column in obs. Tried {label_candidates}")

    conf_col = next((c for c in conf_candidates if c in obs.columns), None)
    return label_col, conf_col


def detect_pca_key(adata: ad.AnnData) -> str:
    if "X_pca" in adata.obsm:
        return "X_pca"
    for k in adata.obsm.keys():
        if "pca" in k.lower():
            return k
    raise ValueError(f"No PCA found in adata.obsm. keys={list(adata.obsm.keys())}")


def parse_signal(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Bad spec: {spec} (use pca:0 | obs:col | gene_score:G1,G2)")
    k, v = spec.split(":", 1)
    return k.strip().lower(), v.strip()


def compute_gene_score(adata: ad.AnnData, genes: List[str], out_col: str) -> None:
    import scanpy as sc
    present = [g for g in genes if g in adata.var_names]
    if len(present) < max(3, int(0.3 * len(genes))):
        raise ValueError(f"Too few genes found in var_names for score_genes: {len(present)}/{len(genes)}")
    sc.tl.score_genes(adata, gene_list=present, score_name=out_col, use_raw=False)


def get_signal(adata: ad.AnnData, spec: str, name: str) -> pd.Series:
    kind, val = parse_signal(spec)

    if kind == "obs":
        if val not in adata.obs.columns:
            raise ValueError(f"{name}: obs column '{val}' not found")
        return pd.to_numeric(adata.obs[val], errors="coerce")

    if kind == "pca":
        pca_key = detect_pca_key(adata)
        pc = int(val)
        X = adata.obsm[pca_key]
        if pc < 0 or pc >= X.shape[1]:
            raise ValueError(f"{name}: PC{pc} out of range for {pca_key} shape={X.shape}")
        return pd.Series(X[:, pc], index=adata.obs_names)

    if kind == "gene_score":
        genes = [x.strip() for x in val.split(",") if x.strip()]
        out_col = f"{name}_gene_score"
        compute_gene_score(adata, genes, out_col)
        return pd.to_numeric(adata.obs[out_col], errors="coerce")

    raise ValueError(f"Unknown signal kind: {kind}")


def auto_covariates(obs: pd.DataFrame) -> List[str]:
    candidates = ["percent_mito", "pct_counts_mt", "total_counts", "n_genes_by_counts", "sample", "batch"]
    return [c for c in candidates if c in obs.columns]


def make_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force everything numeric float64.
    Converts bool->float, object->category->dummy if needed,
    then fills NaNs and drops inf.
    """
    # Convert bool to int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)

    # Force numeric where possible
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# -------------------------
# IV
# -------------------------
def one_hot_instruments(z: pd.Series, min_cells: int) -> pd.DataFrame:
    z = z.astype(str)
    counts = z.value_counts()
    keep = counts[counts >= min_cells].index.tolist()
    z2 = z.where(z.isin(keep), "__OTHER__")
    Z = pd.get_dummies(z2, prefix="Z", drop_first=True)
    return Z


def fit_ols(endog: pd.Series, exog: pd.DataFrame):
    exog = sm.add_constant(exog, has_constant="add")
    model = sm.OLS(endog.values, exog.values)
    res = model.fit(cov_type="HC1")
    res._exog_cols = exog.columns.tolist()
    return res


def top_instruments_plot(res1, out_png: Path, topk: int = 25):
    cols = res1._exog_cols
    params = pd.Series(res1.params, index=cols)

    z_params = params[[c for c in params.index if c.startswith("Z_")]]
    if z_params.empty:
        log("[WARN] No Z coefficients found to plot.")
        return

    top = z_params.reindex(z_params.abs().sort_values(ascending=False).head(topk).index)

    plt.figure(figsize=(10, max(4, 0.25 * len(top))))
    plt.barh(range(len(top)), top.values)
    plt.yticks(range(len(top)), top.index)
    plt.xlabel("First-stage coefficient")
    plt.title("Top instrument coefficients (|coef|)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--min_conf", type=float, default=0.40)
    ap.add_argument("--min_cells_per_pert", type=int, default=50)

    ap.add_argument("--treatment", default="pca:0")
    ap.add_argument("--outcome", default="pca:1")
    ap.add_argument("--covariates", default="auto")

    ap.add_argument("--topk_plot", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Reading h5ad: {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)

    if not adata.obs_names.is_unique:
        log("[WARN] obs_names not unique -> making unique")
        adata.obs_names_make_unique()

    label_col, conf_col = detect_prediction_cols(adata.obs)
    log(f"[INFO] Detected prediction columns: label='{label_col}' conf='{conf_col}'")

    # Filter by confidence
    if conf_col is not None:
        conf = pd.to_numeric(adata.obs[conf_col], errors="coerce")
        keep = conf >= args.min_conf
        log(f"[INFO] Filtering by confidence >= {args.min_conf}: keep {int(keep.sum())}/{adata.n_obs}")
        adata = adata[keep].copy()
    else:
        log("[WARN] No confidence column found -> skipping confidence filtering")

    # Signals
    log(f"[INFO] Treatment spec: {args.treatment}")
    T = get_signal(adata, args.treatment, "treatment")

    log(f"[INFO] Outcome spec: {args.outcome}")
    Y = get_signal(adata, args.outcome, "outcome")

    # Covariates
    if args.covariates.lower() == "auto":
        covs = auto_covariates(adata.obs)
    else:
        covs = [c.strip() for c in args.covariates.split(",") if c.strip()]

    log(f"[INFO] Covariates: {covs}")

    # Instruments
    Z = one_hot_instruments(adata.obs[label_col], min_cells=args.min_cells_per_pert)

    # Covariate design matrix (X)
    X_parts = []
    for c in covs:
        if c not in adata.obs.columns:
            continue
        s = adata.obs[c]
        if pd.api.types.is_numeric_dtype(s):
            X_parts.append(pd.to_numeric(s, errors="coerce").to_frame(c))
        else:
            d = pd.get_dummies(s.astype(str), prefix=f"X_{c}", drop_first=True)
            if d.shape[1] > 0:
                X_parts.append(d)
    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=adata.obs_names)

    # Build aligned df
    df = pd.DataFrame({"T": T, "Y": Y}, index=adata.obs_names)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["T", "Y"])
    common = df.index.intersection(Z.index).intersection(X.index)

    df = df.loc[common]
    Z = Z.loc[common]
    X = X.loc[common]

    log(f"[INFO] Final rows: {df.shape[0]}  Instruments={Z.shape[1]}  Covariates={X.shape[1]}")

    # FORCE NUMERIC FLOAT64 (fixes your crash)
    Z_num = make_numeric_df(Z).astype(np.float64)
    X_num = make_numeric_df(X).astype(np.float64)

    # Drop any remaining bad rows
    design1 = pd.concat([Z_num, X_num], axis=1)
    bad = ~np.isfinite(design1.values).all(axis=1)
    if bad.any():
        log(f"[WARN] Dropping {int(bad.sum())} rows with non-finite values in design matrix")
        keep = ~bad
        df = df.loc[keep]
        Z_num = Z_num.loc[keep]
        X_num = X_num.loc[keep]

    # -----------------
    # First stage
    # -----------------
    log("[INFO] First stage: T ~ Z + X")
    res1 = fit_ols(df["T"], pd.concat([Z_num, X_num], axis=1))

    # Predict T_hat
    exog1 = sm.add_constant(pd.concat([Z_num, X_num], axis=1), has_constant="add")
    That = pd.Series(res1.predict(exog1.values), index=df.index, name="T_hat")

    # -----------------
    # Second stage
    # -----------------
    log("[INFO] Second stage: Y ~ T_hat + X")
    res2 = fit_ols(df["Y"], pd.concat([That.to_frame(), X_num], axis=1))

    cols2 = res2._exog_cols
    idx = cols2.index("T_hat")
    beta = float(res2.params[idx])
    se = float(res2.bse[idx])
    p = float(res2.pvalues[idx])

    out_res = out_dir / "tables" / "stage9_iv_results.tsv"
    pd.DataFrame([{
        "n_rows": int(df.shape[0]),
        "instrument": label_col,
        "treatment": args.treatment,
        "outcome": args.outcome,
        "beta_iv": beta,
        "se_robust": se,
        "p_value": p
    }]).to_csv(out_res, sep="\t", index=False)

    log(f"[OK] IV results -> {out_res}")

    out_diag = out_dir / "tables" / "stage9_diagnostics.tsv"
    pd.DataFrame([{
        "first_stage_r2": float(res1.rsquared),
        "second_stage_r2": float(res2.rsquared),
        "n_instruments": int(Z_num.shape[1]),
        "n_covariates": int(X_num.shape[1]),
    }]).to_csv(out_diag, sep="\t", index=False)

    log(f"[OK] diagnostics -> {out_diag}")

    # Plot top instruments
    out_png = out_dir / "figures" / "stage9_first_stage_top_instruments.png"
    top_instruments_plot(res1, out_png, topk=args.topk_plot)
    if out_png.exists():
        log(f"[OK] plot -> {out_png}")

    log("[DONE] Stage-9 complete")


if __name__ == "__main__":
    main()

