#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

CONTROL_PATTERNS = ["NTC", "NON", "NON-TARGET", "CONTROL", "SCRAMBLE", "NEG"]

def is_control_guide(guide: str) -> bool:
    if guide is None:
        return False
    g = str(guide).upper()
    return any(pat in g for pat in CONTROL_PATTERNS)

def sanitize_obs_for_h5ad(adata: sc.AnnData) -> sc.AnnData:
    obs = adata.obs.copy()
    for col in obs.columns:
        s = obs[col]
        if pd.api.types.is_bool_dtype(s):
            obs[col] = s.fillna(False).astype(np.int8)
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        s = s.astype("object")
        s = s.where(~pd.isna(s), "")

        def to_safe_str(x):
            if x is None:
                return ""
            if isinstance(x, (dict, list, tuple, set)):
                try:
                    return json.dumps(list(x) if isinstance(x, set) else x)
                except Exception:
                    return str(x)
            return str(x)

        obs[col] = s.map(to_safe_str)
    adata.obs = obs
    return adata

def load_cell_identities_if_needed(adata: sc.AnnData, gse_dir: Path, gsm_id: str) -> sc.AnnData:
    """
    If stage0 already joined identities into obs, do nothing.
    Otherwise read from gse_dir/<gsm>_cell_identities.csv(.gz) and join by barcode.
    """
    # If these columns exist, assume metadata is already present.
    # GEO raw uses exact names like "cell BC", "guide identity", "UMI count", "coverage" etc.
    needed_any = ["guide identity", "cell BC", "UMI count", "coverage", "good coverage"]
    if any(c in adata.obs.columns for c in needed_any):
        return adata

    # fallback: try to load
    cand1 = gse_dir / f"{gsm_id}_cell_identities.csv.gz"
    cand2 = gse_dir / f"{gsm_id}_cell_identities.csv"
    path = cand1 if cand1.exists() else cand2 if cand2.exists() else None
    if path is None:
        # still allow pipeline to run; will mark everything unknown
        return adata

    meta = pd.read_csv(path)

    lower = {c.lower(): c for c in meta.columns}
    bc_col = lower.get("cell bc", None) or lower.get("barcode", None)
    if bc_col is None:
        return adata

    meta = meta.copy()
    meta[bc_col] = meta[bc_col].astype(str)
    meta = meta.set_index(bc_col)

    # join by barcode (adata.obs_names are barcodes in raw_<gsm>.h5ad)
    adata.obs = adata.obs.join(meta, how="left")
    return adata

def normalize_meta_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Converts GEO names -> canonical internal names
    """
    colmap = {
        "cell BC": "barcode",
        "guide identity": "guide_id",
        "UMI count": "umi_count",
        "read count": "read_count",
        "coverage": "coverage",
        "good coverage": "good_coverage",
        "number of cells": "n_cells_called",
    }
    out = obs.rename(columns=colmap).copy()

    if "barcode" not in out.columns:
        out["barcode"] = out.index.astype(str)

    out["barcode"] = out["barcode"].astype(str)

    if "guide_id" not in out.columns:
        out["guide_id"] = ""
    out["guide_id"] = out["guide_id"].fillna("").astype(str)

    if "good_coverage" in out.columns:
        out["good_coverage"] = (
            out["good_coverage"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
        )
    else:
        out["good_coverage"] = False

    return out

def compute_perturbation_confidence(obs: pd.DataFrame) -> pd.Series:
    cov = pd.to_numeric(obs.get("coverage", np.nan), errors="coerce")
    umi = pd.to_numeric(obs.get("umi_count", np.nan), errors="coerce")
    good = obs.get("good_coverage", False)
    good = good.fillna(False).astype(bool)

    cov_min = np.nanmin(cov) if np.isfinite(np.nanmin(cov)) else 0.0
    cov_max = np.nanmax(cov) if np.isfinite(np.nanmax(cov)) else 1.0
    cov_norm = (cov - cov_min) / (cov_max - cov_min + 1e-9)

    logumi = np.log1p(umi)
    logumi_max = np.nanmax(logumi) if np.isfinite(np.nanmax(logumi)) else 1.0
    umi_norm = logumi / (logumi_max + 1e-9)

    score = 0.55 * cov_norm + 0.35 * umi_norm + 0.10 * good.astype(float)
    return score.fillna(0.0).clip(0, 1)

def stage1_label_single(adata: sc.AnnData, out_dir: Path) -> sc.AnnData:
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    obs = normalize_meta_columns(adata.obs)

    obs["gsm_id"] = obs.get("gsm_id", "")
    gsm_id = str(obs["gsm_id"].iloc[0]) if "gsm_id" in obs.columns else "UNKNOWN"

    # target gene heuristic
    obs["target_gene"] = obs["guide_id"].map(lambda x: str(x).split("_")[0] if x else "")
    obs["perturbation"] = obs["guide_id"].astype(str)

    # multiplet
    if "n_cells_called" in obs.columns:
        obs["is_multiplet"] = (pd.to_numeric(obs["n_cells_called"], errors="coerce") > 1).fillna(False)
    else:
        obs["is_multiplet"] = False

    obs["n_guides_detected"] = 1
    obs.loc[obs["guide_id"] == "", "n_guides_detected"] = 0

    obs["condition_class"] = obs["guide_id"].map(lambda g: "control" if is_control_guide(g) else "perturbed").astype(str)
    obs.loc[obs["guide_id"].astype(str).fillna("") == "", "condition_class"] = "unknown"

    obs["control_type"] = ""
    obs.loc[obs["condition_class"] == "control", "control_type"] = "non_targeting_or_control"

    obs["perturbation_confidence"] = compute_perturbation_confidence(obs)

    obs["is_responder_candidate"] = (
        (obs["perturbation_confidence"] >= 0.60) &
        (~obs["is_multiplet"].astype(bool)) &
        (obs["condition_class"] == "perturbed")
    )

    adata.obs = obs

    # summary table
    total = obs.shape[0]
    n_ctrl = int((obs["condition_class"] == "control").sum())
    n_pert = int((obs["condition_class"] == "perturbed").sum())
    n_unk  = int((obs["condition_class"] == "unknown").sum())
    n_mult = int(obs["is_multiplet"].astype(bool).sum())
    mean_conf = float(np.nanmean(pd.to_numeric(obs["perturbation_confidence"], errors="coerce")))

    top_guides = (
        obs.loc[obs["guide_id"].astype(str) != "", "guide_id"]
        .value_counts().head(10).to_dict()
    )

    summary_df = pd.DataFrame([{
        "gsm_id": gsm_id,
        "total_cells": total,
        "control_cells": n_ctrl,
        "perturbed_cells": n_pert,
        "unknown_cells": n_unk,
        "multiplets": n_mult,
        "mean_perturbation_confidence": round(mean_conf, 4),
        "top10_guides": str(top_guides),
    }])

    summary_path = tables_dir / "sample_control_perturbation_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    return adata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True, help="processed/raw_<gsm>.h5ad from Stage0")
    ap.add_argument("--gse_dir", required=True, help="input_data/<GSE..._RAW> folder for fallback identities load")
    ap.add_argument("--out_dir", required=True, help="Output directory (per-sample recommended)")
    args = ap.parse_args()

    input_h5ad = Path(args.input_h5ad).resolve()
    gse_dir = Path(args.gse_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)

    gsm_id = None
    if "gsm_id" in adata.obs.columns:
        gsm_id = str(adata.obs["gsm_id"].iloc[0])
    else:
        gsm_id = input_h5ad.stem.replace("raw_", "")

    # Make stage1 independent:
    adata = load_cell_identities_if_needed(adata, gse_dir, gsm_id)

    print(f"[INFO] labeling perturbations for {gsm_id}")
    adata = stage1_label_single(adata, out_dir)

    adata = sanitize_obs_for_h5ad(adata)

    out_h5ad = out_dir / "stage1_labeled.h5ad"
    adata.write(out_h5ad)

    print(f"[OK] wrote {out_h5ad}")
    print(f"[OK] wrote {out_dir / 'tables' / 'sample_control_perturbation_summary.tsv'}")

if __name__ == "__main__":
    main()

