#!/usr/bin/env python3
"""
Stage-11: Latent AI + UI export (robust + auto scoring)

Inputs:
  - Stage8 qc annotated h5ad (pred_label + pred_confidence)
Outputs:
  processed_stage11/
    stage11_latent.h5ad
    tables/
      stage11_cluster_summary.tsv
      stage11_cluster_top_genes.tsv
      stage11_perturbation_cluster_enrichment.tsv
      stage11_ui_cards.tsv
    figures/
      stage11_umap_clusters.png
      stage11_umap_predlabel.png
      stage11_umap_program_<name>.png  (auto, if programs exist)
"""
from __future__ import annotations

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

from utils_gene_programs import auto_gene_programs, score_programs_scanpy


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_make_unique(adata: ad.AnnData):
    if not adata.obs_names.is_unique:
        log("[WARN] obs_names not unique -> making unique")
        adata.obs_names_make_unique()


def detect_stage8_pred_cols(obs: pd.DataFrame):
    label_candidates = ["pred_label_stage7", "pred_class_stage7", "pred_label", "pred_class"]
    conf_candidates = ["pred_confidence_stage7", "pred_conf_stage7", "pred_confidence", "pred_conf"]
    label_col = next((c for c in label_candidates if c in obs.columns), None)
    conf_col = next((c for c in conf_candidates if c in obs.columns), None)
    if label_col is None:
        raise ValueError(f"Missing prediction label column. Tried: {label_candidates}")
    return label_col, conf_col


def cluster_and_latent(
    adata: ad.AnnData,
    n_pcs: int = 30,
    neighbors_k: int = 15,
    resolution: float = 0.7,
):
    # PCA (if missing)
    if "X_pca" not in adata.obsm:
        log("[INFO] X_pca not found -> computing PCA")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat", subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=max(n_pcs, 10))

    log("[INFO] Neighbors + UMAP")
    sc.pp.neighbors(adata, n_neighbors=neighbors_k, n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata)

    log("[INFO] Leiden clustering")
    sc.tl.leiden(adata, resolution=resolution, key_added="latent_cluster")
    return adata


def rank_cluster_genes(adata: ad.AnnData, cluster_key="latent_cluster", top_n=30):
    log("[INFO] Ranking genes per cluster")
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")
    rg = adata.uns["rank_genes_groups"]

    clusters = rg["names"].dtype.names
    rows = []
    for cl in clusters:
        names = rg["names"][cl][:top_n]
        scores = rg["scores"][cl][:top_n]
        pvals = rg["pvals_adj"][cl][:top_n]
        for g, s, p in zip(names, scores, pvals):
            rows.append({"cluster": cl, "gene": g, "score": float(s), "pval_adj": float(p)})
    return pd.DataFrame(rows)


def cluster_summary(adata: ad.AnnData, cluster_key: str, label_col: str, conf_col: str | None):
    obs = adata.obs.copy()
    obs[cluster_key] = obs[cluster_key].astype(str)

    grp = obs.groupby(cluster_key)
    df = pd.DataFrame({"cluster": grp.size().index, "n_cells": grp.size().values})

    if conf_col is not None:
        df["mean_pred_conf"] = grp[conf_col].mean().values

    top_label = grp[label_col].agg(lambda x: x.value_counts().index[0])
    df["top_pred_label"] = df["cluster"].map(top_label)
    return df.sort_values("n_cells", ascending=False)


def perturbation_cluster_enrichment(adata: ad.AnnData, cluster_key: str, label_col: str):
    obs = adata.obs.copy()
    obs[cluster_key] = obs[cluster_key].astype(str)
    obs[label_col] = obs[label_col].astype(str)

    tab = pd.crosstab(obs[label_col], obs[cluster_key])
    frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)

    rows = []
    for pert in frac.index:
        best_cluster = frac.loc[pert].idxmax()
        best_frac = float(frac.loc[pert, best_cluster])
        rows.append({
            "perturbation": pert,
            "best_cluster": best_cluster,
            "fraction_in_best_cluster": best_frac,
            "n_cells": int(tab.loc[pert].sum()),
        })

    return pd.DataFrame(rows).sort_values(["fraction_in_best_cluster", "n_cells"], ascending=False)


def build_ui_cards(cluster_summary_df: pd.DataFrame, top_genes_df: pd.DataFrame, topk_genes=10):
    genes_map = (
        top_genes_df.sort_values(["cluster", "score"], ascending=[True, False])
        .groupby("cluster")["gene"]
        .apply(lambda s: ",".join(list(s.head(topk_genes))))
        .to_dict()
    )
    cards = cluster_summary_df.copy()
    cards["top_genes"] = cards["cluster"].map(genes_map).fillna("")
    cards["card_text"] = (
        "Cluster " + cards["cluster"].astype(str)
        + " | cells=" + cards["n_cells"].astype(str)
        + " | top_label=" + cards["top_pred_label"].astype(str)
        + " | genes=" + cards["top_genes"].astype(str)
    )
    return cards


def save_umap(adata: ad.AnnData, color: str, out_png: Path, title: str = ""):
    sc.pl.umap(adata, color=color, show=False, title=title)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--min_conf", type=float, default=0.40)
    ap.add_argument("--min_cells_per_pert", type=int, default=50)

    ap.add_argument("--include_pcs", type=int, default=30)
    ap.add_argument("--neighbors_k", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=0.7)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Reading h5ad: {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)
    safe_make_unique(adata)

    label_col, conf_col = detect_stage8_pred_cols(adata.obs)
    log(f"[INFO] Prediction columns: label='{label_col}' conf='{conf_col}'")

    # confidence filter
    if conf_col is not None:
        conf = pd.to_numeric(adata.obs[conf_col], errors="coerce")
        keep = conf >= args.min_conf
        log(f"[INFO] Filter conf >= {args.min_conf}: keep {int(keep.sum())}/{adata.n_obs}")
        adata = adata[keep].copy()
        safe_make_unique(adata)

    # keep perturbations with enough cells
    vc = adata.obs[label_col].astype(str).value_counts()
    keep_levels = vc[vc >= args.min_cells_per_pert].index
    adata = adata[adata.obs[label_col].astype(str).isin(keep_levels)].copy()
    safe_make_unique(adata)
    log(f"[INFO] After min_cells_per_pert {args.min_cells_per_pert}: {adata.n_obs} cells")

    # auto score programs (never crashes)
    programs = auto_gene_programs(adata, max_programs=3, genes_per_program=60, min_present=10)
    scored_cols = []
    if programs:
        scored_cols, skipped = score_programs_scanpy(adata, programs, prefix="PROG_", min_present=10)
        if skipped:
            log(f"[WARN] Skipped program scoring: {skipped}")
        if scored_cols:
            log(f"[INFO] Scored programs: {scored_cols}")
    else:
        log("[WARN] No programs discovered -> skipping program scoring")

    # latent + clusters
    adata = cluster_and_latent(
        adata,
        n_pcs=args.include_pcs,
        neighbors_k=args.neighbors_k,
        resolution=args.resolution,
    )

    # write latent
    out_h5ad = out_dir / "stage11_latent.h5ad"
    adata.write(out_h5ad)
    log(f"[OK] wrote latent h5ad -> {out_h5ad}")

    # tables
    df_cluster = cluster_summary(adata, "latent_cluster", label_col, conf_col)
    df_cluster.to_csv(out_dir / "tables" / "stage11_cluster_summary.tsv", sep="\t", index=False)
    log("[OK] cluster summary -> tables/stage11_cluster_summary.tsv")

    df_top = rank_cluster_genes(adata, "latent_cluster", top_n=30)
    df_top.to_csv(out_dir / "tables" / "stage11_cluster_top_genes.tsv", sep="\t", index=False)
    log("[OK] top genes -> tables/stage11_cluster_top_genes.tsv")

    df_enrich = perturbation_cluster_enrichment(adata, "latent_cluster", label_col)
    df_enrich.to_csv(out_dir / "tables" / "stage11_perturbation_cluster_enrichment.tsv", sep="\t", index=False)
    log("[OK] enrichment -> tables/stage11_perturbation_cluster_enrichment.tsv")

    df_cards = build_ui_cards(df_cluster, df_top, topk_genes=10)
    df_cards.to_csv(out_dir / "tables" / "stage11_ui_cards.tsv", sep="\t", index=False)
    log("[OK] ui cards -> tables/stage11_ui_cards.tsv")

    # figures
    log("[INFO] Saving UMAPs")
    save_umap(adata, "latent_cluster", out_dir / "figures" / "stage11_umap_clusters.png", "Latent clusters")
    save_umap(adata, label_col, out_dir / "figures" / "stage11_umap_predlabel.png", "Predicted label")

    for c in scored_cols:
        safe_name = c.replace("PROG_", "program_").replace("/", "_").replace(" ", "_")
        save_umap(adata, c, out_dir / "figures" / f"stage11_umap_{safe_name}.png", c)

    log("[OK] figures -> processed_stage11/figures")
    log("[DONE] Stage-11 complete")


if __name__ == "__main__":
    main()
