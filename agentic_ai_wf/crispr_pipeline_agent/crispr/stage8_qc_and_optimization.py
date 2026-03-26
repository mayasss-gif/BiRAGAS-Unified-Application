#!/usr/bin/env python3
"""
Stage-8: QC + Optimization on Stage-7 predictions

Inputs:
  - stage3 merged h5ad
  - stage7 outputs:
      stage7_cell_predictions.tsv
      stage7_perturbation_ranked_by_confidence.tsv

Outputs:
  - QC tables
  - figures (confidence hist, top perturbations, UMAP confidence overlay)
  - updated h5ad with prediction columns attached

Fixes:
  - Handles different Stage7 column naming ("pred_class" vs "pred_label")
  - Handles duplicate cell_ids by keeping highest-confidence prediction per cell_id
"""

import os
import sys
import argparse
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def detect_stage7_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Try to infer the columns for:
      - cell id
      - predicted label
      - predicted confidence
    """
    cols = set(df.columns)

    # cell id candidates
    cell_candidates = ["cell_id", "barcode", "obs_name", "cell", "cell_barcode"]
    cell_col = None
    for c in cell_candidates:
        if c in cols:
            cell_col = c
            break
    if cell_col is None:
        # fallback to first column if it looks like barcodes
        cell_col = df.columns[0]
        warn(f"cell_id column not found explicitly, using first column: {cell_col}")

    # pred label candidates
    label_candidates = ["pred_label", "pred_class", "predicted_label", "prediction", "y_pred"]
    label_col = None
    for c in label_candidates:
        if c in cols:
            label_col = c
            break

    # confidence candidates
    conf_candidates = ["pred_confidence", "confidence", "pred_prob", "max_prob", "p_max"]
    conf_col = None
    for c in conf_candidates:
        if c in cols:
            conf_col = c
            break

    missing = []
    if label_col is None:
        missing.append("pred_label/pred_class")
    if conf_col is None:
        missing.append("pred_confidence/confidence")

    if missing:
        raise ValueError(f"stage7_cell_predictions.tsv missing required prediction columns: {missing}. Found={list(df.columns)}")

    return cell_col, label_col, conf_col


# -----------------------------
# QC computation
# -----------------------------
def compute_cell_qc(pred_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Cell-level QC derived from prediction table
    """
    cell_col, label_col, conf_col = detect_stage7_columns(pred_cells)

    info(f"Detected columns -> cell_id='{cell_col}', pred_label='{label_col}', pred_confidence='{conf_col}'")

    df = pred_cells[[cell_col, label_col, conf_col]].copy()
    df.columns = ["cell_id", "pred_label", "pred_confidence"]

    # Ensure types
    df["cell_id"] = df["cell_id"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str)
    df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors="coerce")

    # Flags
    df["is_high_conf"] = df["pred_confidence"] >= 0.5
    df["is_low_conf"] = df["pred_confidence"] < 0.5

    return df


def compute_perturbation_qc(df_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Per-perturbation summary based on predicted labels
    """
    g = (
        df_cells.groupby("pred_label", as_index=False)
        .agg(
            n_cells=("cell_id", "count"),
            mean_conf=("pred_confidence", "mean"),
            median_conf=("pred_confidence", "median"),
            frac_high_conf=("is_high_conf", "mean"),
        )
        .sort_values(["n_cells", "mean_conf"], ascending=False)
    )
    g.rename(columns={"pred_label": "predicted_perturbation"}, inplace=True)
    return g


# -----------------------------
# Plotting
# -----------------------------
def plot_confidence_hist(df_cells: pd.DataFrame, out_png: str):
    plt.figure(figsize=(6, 4))
    x = df_cells["pred_confidence"].dropna().values
    plt.hist(x, bins=40)
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of cells")
    plt.title("Stage7 prediction confidence distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top_predicted_perts(df_pert_qc: pd.DataFrame, out_png: str, topn: int = 20):
    top = df_pert_qc.head(topn).copy()
    plt.figure(figsize=(9, 5))
    plt.bar(top["predicted_perturbation"].astype(str), top["n_cells"].astype(int))
    plt.xticks(rotation=90)
    plt.ylabel("n_cells")
    plt.title(f"Top {topn} predicted perturbations")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Attach predictions to AnnData + UMAP plot
# -----------------------------
def attach_to_adata_umap_plot(adata: sc.AnnData, df_cells: pd.DataFrame, figs_dir: str):
    """
    Attach prediction columns onto full Stage3 AnnData.
    Fix for duplicates:
      - df_cells may contain duplicate cell_id across merged samples
      - keep the highest confidence row per cell_id
    """
    if "X_umap" not in adata.obsm:
        warn("No UMAP found in adata.obsm['X_umap'], skipping UMAP overlay plot.")
        return

    ensure_dir(figs_dir)

    # ✅ FIX: deduplicate predictions by taking max confidence per cell_id
    df_best = (
        df_cells.sort_values("pred_confidence", ascending=False)
        .drop_duplicates(subset=["cell_id"], keep="first")
        .copy()
    )

    pred_map = df_best.set_index("cell_id")[["pred_label", "pred_confidence"]]

    # Find overlap
    common = adata.obs_names.intersection(pred_map.index)
    if len(common) == 0:
        warn("No overlap between adata.obs_names and stage7 cell predictions. Skipping attach.")
        return

    # Create columns
    adata.obs["pred_label_stage7"] = "NA"
    adata.obs["pred_confidence_stage7"] = np.nan

    # ✅ Use .values to avoid alignment issues
    adata.obs.loc[common, "pred_label_stage7"] = pred_map.loc[common, "pred_label"].astype(str).values
    adata.obs.loc[common, "pred_confidence_stage7"] = pred_map.loc[common, "pred_confidence"].astype(float).values

    # Plot UMAP
    sc.pl.umap(
        adata,
        color="pred_confidence_stage7",
        show=False,
    )
    out_png = os.path.join(figs_dir, "stage8_umap_pred_confidence.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    info(f"[OK] UMAP overlay -> {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3_h5ad", required=True, help="Stage3 merged h5ad")
    ap.add_argument("--stage7_dir", required=True, help="Directory containing stage7 outputs")
    ap.add_argument("--out_dir", required=True, help="Output directory for Stage8")
    ap.add_argument("--min_conf", type=float, default=0.4, help="High-confidence threshold")
    ap.add_argument("--min_cells_per_pert", type=int, default=50, help="Min predicted cells per perturbation to keep in filtered list")
    args = ap.parse_args()

    stage7_cells = os.path.join(args.stage7_dir, "stage7_cell_predictions.tsv")
    stage7_ranked = os.path.join(args.stage7_dir, "stage7_perturbation_ranked_by_confidence.tsv")

    if not os.path.exists(stage7_cells):
        raise FileNotFoundError(f"Missing: {stage7_cells}")
    if not os.path.exists(stage7_ranked):
        warn(f"Missing ranked perturbations file (will continue): {stage7_ranked}")

    ensure_dir(args.out_dir)
    tables_dir = os.path.join(args.out_dir, "tables")
    figs_dir = os.path.join(args.out_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    info(f"Loading predictions: {stage7_cells}")
    pred_cells = read_tsv(stage7_cells)

    df_cells = compute_cell_qc(pred_cells)

    # Save cell QC table
    out_cells_qc = os.path.join(tables_dir, "stage8_cells_qc.tsv")
    df_cells.to_csv(out_cells_qc, sep="\t", index=False)
    info(f"[OK] cell QC -> {out_cells_qc}")

    # Per-pert QC
    df_pert_qc = compute_perturbation_qc(df_cells)
    out_pert_qc = os.path.join(tables_dir, "stage8_predicted_perturbation_qc.tsv")
    df_pert_qc.to_csv(out_pert_qc, sep="\t", index=False)
    info(f"[OK] per-pert QC -> {out_pert_qc}")

    # High confidence filtering
    info(f"Filtering high-confidence cells >= {args.min_conf}")
    df_high = df_cells[df_cells["pred_confidence"] >= args.min_conf].copy()
    out_high = os.path.join(tables_dir, f"stage8_cells_highconf_ge_{args.min_conf:.2f}.tsv")
    df_high.to_csv(out_high, sep="\t", index=False)
    info(f"[OK] high-conf cells -> {out_high} n={len(df_high)}")

    # Filter perts with enough cells
    df_high_pert = (
        df_high.groupby("pred_label", as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
        .sort_values("n_cells", ascending=False)
    )
    df_high_pert = df_high_pert[df_high_pert["n_cells"] >= args.min_cells_per_pert]
    out_high_pert = os.path.join(tables_dir, f"stage8_highconf_perts_ge_{args.min_cells_per_pert}.tsv")
    df_high_pert.to_csv(out_high_pert, sep="\t", index=False)
    info(f"[OK] high-conf perts -> {out_high_pert} n={len(df_high_pert)}")

    # Plots
    info("Saving confidence plots")
    plot_confidence_hist(df_cells, os.path.join(figs_dir, "stage8_confidence_hist.png"))
    plot_top_predicted_perts(df_pert_qc, os.path.join(figs_dir, "stage8_top_predicted_perts.png"), topn=25)
    info(f"[OK] figures -> {figs_dir}")

    # Attach to AnnData
    info(f"Reading h5ad: {args.stage3_h5ad}")
    adata = sc.read_h5ad(args.stage3_h5ad)

    if not adata.obs_names.is_unique:
        warn("obs_names not unique -> making unique")
        adata.obs_names_make_unique()

    info("Attaching predictions to adata + plotting confidence UMAP")
    attach_to_adata_umap_plot(adata, df_cells, figs_dir)

    out_h5ad = os.path.join(args.out_dir, "stage8_qc_annotated.h5ad")
    adata.write(out_h5ad)
    info(f"[OK] wrote annotated h5ad -> {out_h5ad}")

    info("[DONE] Stage-8 complete")


if __name__ == "__main__":
    main()

