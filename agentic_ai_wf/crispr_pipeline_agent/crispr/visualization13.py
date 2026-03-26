#!/usr/bin/env python3
"""
Unified Visualization & Reporting Layer
(Stage 3–11 → HTML Dashboard)

Robust version:
- Safe DEG plotting
- Auto-detect groupby for heatmaps
- Never crashes on missing annotations
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from jinja2 import Template


# =========================
# Utilities
# =========================
def log(msg):
    print(msg, flush=True)


def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_groupby(adata):
    """Return a valid obs column for grouping, or None."""
    for c in ["pred_label_stage7", "latent_cluster", "leiden"]:
        if c in adata.obs.columns:
            return c
    return None


# =========================
# DEG VISUALIZATION
# =========================
def plot_deg_volcano(adata, out_png, top_n=50):
    if "rank_genes_groups" not in adata.uns:
        log("[WARN] No rank_genes_groups found — skipping volcano")
        return pd.DataFrame()

    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names
    g = groups[0]

    df = pd.DataFrame({
        "gene": rg["names"][g],
        "logFC": rg["logfoldchanges"][g],
        "pval_adj": rg["pvals_adj"][g],
    }).dropna()

    df["-log10p"] = -np.log10(df["pval_adj"].clip(1e-300))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="logFC", y="-log10p", s=15)

    top = df.sort_values("pval_adj").head(top_n)
    for _, r in top.iterrows():
        plt.text(r["logFC"], r["-log10p"], r["gene"], fontsize=6)

    plt.axvline(0, color="gray", lw=1)
    plt.title("DEG Volcano Plot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return top


def plot_deg_heatmap(adata, out_png, top_n=30):
    if "rank_genes_groups" not in adata.uns:
        log("[WARN] No rank_genes_groups — skipping heatmap")
        return

    groupby = detect_groupby(adata)
    if groupby is None:
        log("[WARN] No valid groupby found — skipping heatmap")
        return

    rg = adata.uns["rank_genes_groups"]
    genes = rg["names"].dtype.names
    top_genes = list(rg["names"][genes[0]][:top_n])

    sc.pl.heatmap(
        adata,
        var_names=top_genes,
        groupby=groupby,
        show=False,
    )
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# DAG VISUALIZATION
# =========================
def render_dag(stage10_dir: Path, out_png: Path):
    edges_path = stage10_dir / "tables" / "stage10_edges.tsv"
    if not edges_path.exists():
        log("[WARN] No DAG edges found — skipping DAG")
        return

    edges = pd.read_csv(edges_path, sep="\t")
    G = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=900, font_size=7)
    plt.title("Causal DAG (Stage 10)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# HTML TEMPLATE
# =========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>CRISPR Pipeline Report</title>
  <style>
    body { font-family: Arial; margin: 40px; }
    h1,h2 { color: #2c3e50; }
    img { max-width: 900px; margin-bottom: 30px; }
    table { border-collapse: collapse; }
    td,th { border: 1px solid #ccc; padding: 6px; }
  </style>
</head>
<body>

<h1>CRISPR Perturb-seq Analysis Report</h1>

<h2>DEG Analysis</h2>
<img src="figures/deg_volcano.png">
<img src="figures/deg_heatmap.png">

<h2>Latent Space</h2>
<img src="figures/umap_clusters.png">
<img src="figures/umap_predlabel.png">

<h2>Causal DAG</h2>
<img src="figures/dag_network.png">

<h2>Top Differential Genes</h2>
{{ deg_table }}

</body>
</html>
"""


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True,
                    help="processed/GSMxxxx directory")
    args = ap.parse_args()

    base = Path(args.sample_dir)
    vis = base / "visualization"
    figs = vis / "figures"
    tabs = vis / "tables"

    ensure(figs)
    ensure(tabs)

    # -------- DEGs --------
    deg_h5ad = base / "processed_stage4" / "stage4_deg_results.h5ad"
    adata_deg = sc.read_h5ad(deg_h5ad)

    deg_top = plot_deg_volcano(
        adata_deg,
        figs / "deg_volcano.png"
    )
    if not deg_top.empty:
        deg_top.to_csv(tabs / "deg_top.tsv", sep="\t", index=False)

    plot_deg_heatmap(
        adata_deg,
        figs / "deg_heatmap.png"
    )

    # -------- Latent --------
    latent_h5ad = base / "processed_stage11" / "stage11_latent.h5ad"
    if latent_h5ad.exists():
        adata_lat = sc.read_h5ad(latent_h5ad)

        if "latent_cluster" in adata_lat.obs:
            sc.pl.umap(adata_lat, color="latent_cluster", show=False)
            plt.savefig(figs / "umap_clusters.png", dpi=200)
            plt.close()

        if "pred_label_stage7" in adata_lat.obs:
            sc.pl.umap(adata_lat, color="pred_label_stage7", show=False)
            plt.savefig(figs / "umap_predlabel.png", dpi=200)
            plt.close()

    # -------- DAG --------
    render_dag(base / "processed_stage10", figs / "dag_network.png")

    # -------- HTML --------
    html = Template(HTML_TEMPLATE).render(
        deg_table=deg_top.head(20).to_html(index=False) if not deg_top.empty else "<p>No DEGs</p>"
    )

    with open(vis / "index.html", "w") as f:
        f.write(html)

    log(f"[DONE] Visualization written to {vis}")


if __name__ == "__main__":
    main()

