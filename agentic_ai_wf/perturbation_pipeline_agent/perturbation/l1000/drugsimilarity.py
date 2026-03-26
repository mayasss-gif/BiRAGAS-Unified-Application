"""
Drug similarity & clustering module for L1000 runs.

- Auto-detects latest L1000_Run_* folder under ./runs
- Uses selected_signatures_meta.csv from that run
- Loads Level5 GCTX (landmarks only) and computes per-drug profiles
- Supports similarity metrics: cosine, pearson, spearman, kendall
- Outputs similarity matrices, clusters, plots, and a text summary.

Usage (from run_drugsimilarity.py):

    # Default: run all metrics (cosine, pearson, spearman, kendall)
    from drugsimilarity import run_drug_similarity
    run_drug_similarity()

    # Single metric (strict, raises on error like original)
    run_drug_similarity(metric="cosine")
"""

import os
import sys
import math
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from scipy.stats import spearmanr, kendalltau, rankdata

from cmapPy.pandasGEXpress import parse as gct_parse  # for GCTX



from .constants import CORE_SIG_PATH, CORE_GENE_PATH, CORE_GCTX_PATH

# -------------------- Paths & logging --------------------


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Simple logger for this module."""
    print(f"{_timestamp()} | INFO | L1000-DRUGSIM | {msg}", flush=True)



# -------------------- Similarity helpers --------------------


def _row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (gene) across columns."""
    arr = df.values.astype(float)
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    std[std == 0] = 1.0
    z = (arr - mean) / std
    out = pd.DataFrame(z, index=df.index, columns=df.columns)
    return out


def _compute_similarity(
    drug_mat: pd.DataFrame,
    metric: Literal["cosine", "pearson", "spearman", "kendall"] = "cosine",
) -> pd.DataFrame:
    """
    Compute drug × drug similarity matrix for the chosen metric.
    drug_mat: genes × drugs
    """
    drugs = list(drug_mat.columns)
    X = drug_mat.values.astype(float)  # shape: genes × drugs

    if len(drugs) < 2:
        raise ValueError("Need at least 2 drugs to compute similarity.")

    if metric == "cosine":
        sim = cosine_similarity(X.T)  # drugs × drugs

    elif metric == "pearson":
        # np.corrcoef expects variables as rows → transpose
        sim = np.corrcoef(X.T)

    elif metric == "spearman":
        # rank each gene across drugs, then Pearson corr on ranks
        ranks = np.apply_along_axis(rankdata, 1, X)  # genes × drugs
        sim = np.corrcoef(ranks.T)

    elif metric == "kendall":
        n = len(drugs)
        sim = np.full((n, n), np.nan, float)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                tau, _ = kendalltau(X[:, i], X[:, j], nan_policy="omit")
                sim[i, j] = sim[j, i] = tau
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    sim_df = pd.DataFrame(sim, index=drugs, columns=drugs)
    return sim_df


def _run_kmeans(drug_mat: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Metric-specific clustering on drugs using similarity-derived distances.
    Returns DataFrame with columns: [Drug, Cluster].
    """
    drugs = list(drug_mat.columns)
    X = drug_mat.values.T  # drugs × genes

    n_drugs = X.shape[0]
    if n_drugs < 2:
        raise ValueError("Need at least 2 drugs for clustering.")

    # simple heuristic for clusters: min(10, max(2, n_drugs // 10))
    n_clusters = min(10, max(2, n_drugs // 10 if n_drugs >= 10 else 2))

    # compute similarity for this metric and convert to a distance matrix
    sim_df = _compute_similarity(drug_mat, metric=metric)
    sim = sim_df.values.astype(float)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    # metric-specific clustering on distances
    try:
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:  # sklearn < 1.2 uses "affinity"
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )

    labels = agg.fit_predict(dist)

    # 2D coordinates for plotting (metric-specific)
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(dist)
    except Exception:
        coords = None

    clusters = pd.DataFrame({"Drug": drugs, "Cluster": labels})
    clusters.attrs["coords"] = coords  # store for plotting (metric-specific)
    clusters.attrs["n_clusters"] = n_clusters
    return clusters


# -------------------- Plotting helpers --------------------


def _plot_heatmap(sim_df: pd.DataFrame, clusters: pd.DataFrame, out_png: Path, metric: str) -> None:
    """Pretty heatmap with drugs ordered by cluster and within-cluster similarity."""
    try:
        import seaborn as sns
    except ImportError:
        log("seaborn not installed; skipping heatmap.")
        return

    sim = sim_df.copy()

    # reorder drugs by cluster, then alphabetically
    order = (
        clusters.sort_values(["Cluster", "Drug"])
        .reset_index(drop=True)["Drug"]
        .tolist()
    )
    sim = sim.loc[order, order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim,
        cmap="coolwarm",
        vmin=-1.0 if metric != "cosine" else 0.0,
        vmax=1.0,
        center=0.0 if metric != "cosine" else 0.5,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": f"{metric.capitalize()} similarity"},
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Drug–Drug Similarity ({metric})", fontsize=14)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_cluster_scatter(clusters: pd.DataFrame, out_png: Path, metric: str) -> None:
    """2D scatter colored by cluster."""
    coords = clusters.attrs.get("coords")
    if coords is None or coords.shape[1] < 2:
        log("Not enough PCs for scatter; skipping.")
        return

    drugs = clusters["Drug"].tolist()
    labels = clusters["Cluster"].values

    plt.figure(figsize=(7.5, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=40, alpha=0.9)

    # annotate some drugs (up to ~30)
    step = max(1, len(drugs) // 30)
    for i, name in enumerate(drugs):
        if i % step == 0:
            plt.text(coords[i, 0], coords[i, 1], name, fontsize=7, ha="center", va="center")

    plt.xlabel("Dim 1", fontsize=12)
    plt.ylabel("Dim 2", fontsize=12)
    plt.title(f"Drug Clusters (MDS, {metric})", fontsize=14)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_cluster_bar(clusters: pd.DataFrame, out_png: Path, metric: str) -> None:
    """Bar plot of number of drugs per cluster."""
    counts = clusters["Cluster"].value_counts().sort_index()
    plt.figure(figsize=(6, 4.5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("# Drugs", fontsize=12)
    plt.title(f"Drugs per Cluster ({metric})", fontsize=14)
    for idx, val in zip(counts.index.astype(str), counts.values):
        plt.text(idx, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------- Summary helper --------------------


def _write_summary(
    sim_df: pd.DataFrame,
    clusters: pd.DataFrame,
    out_txt: Path,
    metric: str,
    high_threshold: float = 0.75,
) -> None:
    """Write a human-readable summary of similarity + clusters."""
    drugs = sim_df.index.tolist()
    n_drugs = len(drugs)
    n_clusters = clusters.attrs.get("n_clusters", clusters["Cluster"].nunique())

    # upper triangle without diagonal
    vals = []
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            v = sim_df.iloc[i, j]
            if np.isfinite(v):
                vals.append(v)
    vals = np.array(vals, float) if vals else np.array([], float)

    n_pairs = len(vals)
    n_high = int(np.sum(vals >= high_threshold)) if n_pairs else 0
    frac_high = (n_high / n_pairs) if n_pairs else 0.0

    # high-sim pairs per cluster
    cluster_map = dict(zip(clusters["Drug"], clusters["Cluster"]))
    clusters_with_high = set()
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            d1, d2 = drugs[i], drugs[j]
            c1, c2 = cluster_map.get(d1), cluster_map.get(d2)
            if c1 != c2:
                continue
            v = sim_df.iloc[i, j]
            if np.isfinite(v) and v >= high_threshold:
                clusters_with_high.add(c1)

    with out_txt.open("w") as f:
        f.write(f"Drug similarity summary ({metric})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"# drugs         : {n_drugs}\n")
        f.write(f"# clusters      : {n_clusters}\n")
        f.write(f"# pairs         : {n_pairs}\n")
        f.write(f"# pairs ≥ {high_threshold:.2f} : {n_high} "
                f"({frac_high*100:.1f}% of all pairs)\n")
        f.write("\n")
        f.write(f"Clusters with ≥ {high_threshold:.2f} internal similarity pairs: "
                f"{len(clusters_with_high)} / {n_clusters}\n")
        f.write(f"Cluster IDs (0-based): {sorted(clusters_with_high)}\n\n")

        f.write("Drugs per cluster:\n")
        for cid, group in clusters.groupby("Cluster"):
            names = sorted(group["Drug"].tolist())
            f.write(f"  Cluster {cid} (n={len(names)}): {', '.join(names)}\n")

    log(f"Summary written → {out_txt.name}")


# -------------------- MAIN ENTRYPOINT --------------------


def run_drug_similarity(output_dir: Path,
    metric: Literal["cosine", "pearson", "spearman", "kendall", "all"] = "all"
) -> None:
    """
    Main function: compute drug similarity & clustering for latest run.

    metric:
        - "cosine", "pearson", "spearman", "kendall" → run only that metric (strict)
        - "all" (default) → run all four metrics in one pass.
          In "all" mode, if a metric fails it is logged and skipped, others still run.
    """
    metric = metric.lower()
    valid_metrics = {"cosine", "pearson", "spearman", "kendall", "all"}
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    # Decide which metrics to run
    if metric == "all":
        metrics_to_run = ["cosine", "pearson", "spearman", "kendall"]
    else:
        metrics_to_run = [metric]

    # ---------- shared setup (same structure as original) ----------

    run_dir = output_dir
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"Using latest run directory: {run_dir}")

    # files
    sig_path = CORE_SIG_PATH
    gene_path = CORE_GENE_PATH
    gctx = CORE_GCTX_PATH

    for p in [sig_path, gene_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")


    sel_meta_path = run_dir / "selected_signatures_meta.csv"
    if not sel_meta_path.exists():
        raise FileNotFoundError(f"Missing selection meta: {sel_meta_path}")

    # ---------------- load metadata ----------------
    log("Loading selected signatures …")
    small_meta = pd.read_csv(sel_meta_path)
    if "sig_id" not in small_meta.columns or "pert_iname" not in small_meta.columns:
        raise ValueError("selected_signatures_meta.csv must contain 'sig_id' and 'pert_iname'.")

    cid_list = small_meta["sig_id"].astype(str).tolist()
    if len(cid_list) < 2:
        raise ValueError("Need at least 2 signatures to build drug profiles.")

    log("Loading gene annotations …")
    gene = pd.read_csv(gene_path, sep="\t", low_memory=False)
    gene["pr_gene_id"] = gene["pr_gene_id"].astype(str)
    gene["pr_is_lm"] = pd.to_numeric(gene["pr_is_lm"], errors="coerce").fillna(0).astype(int)
    lm_ids = gene.loc[gene["pr_is_lm"] > 0, "pr_gene_id"].astype(str).tolist()

    log("Reading Level5 GCTX (landmarks × selected sigs) …")
    gctoo = gct_parse.parse(str(gctx), rid=lm_ids, cid=cid_list)
    expr = gctoo.data_df.copy()
    expr.index = expr.index.astype(str)  # genes × sigs

    if expr.shape[1] != len(cid_list):
        log("Warning: expression columns do not match expected signature count.")

    # ---------------- build per-drug matrix ----------------
    log("Building per-drug profiles …")

    sig2drug = dict(zip(small_meta["sig_id"].astype(str), small_meta["pert_iname"].astype(str)))

    expr_norm = _row_zscore(expr)
    expr_norm.columns = [sig2drug.get(c, c) for c in expr_norm.columns]

    # group by drug name (columns) and average replicates
    drug_mat = expr_norm.groupby(expr_norm.columns, axis=1).mean()
    n_drugs = drug_mat.shape[1]

    log(f"Per-drug matrix: genes={drug_mat.shape[0]}, drugs={n_drugs}")

    if n_drugs < 2:
        raise ValueError("Not enough drugs for similarity/clustering.")

    # ---------------- run metrics ----------------
    successful_metrics = []

    for m in metrics_to_run:
        if metric == "all":
            # tolerant mode: skip failing metrics
            try:
                log(f"Computing {m} similarity …")
                sim_df = _compute_similarity(drug_mat, metric=m)

                sim_out = run_dir / f"perturbation_similarity_{m}.csv"
                sim_df.to_csv(sim_out)
                log(f"Similarity matrix written → {sim_out.name}")

                # legacy name for cosine
                if m == "cosine":
                    legacy = run_dir / "perturbation_similarity.csv"
                    sim_df.to_csv(legacy)
                    log(f"Legacy similarity written → {legacy.name}")

                # clustering
                log("Running KMeans clustering on drug profiles …")
                clusters = _run_kmeans(drug_mat, metric=m)
                clusters_out = run_dir / f"drug_clusters_{m}.csv"
                clusters.to_csv(clusters_out, index=False)
                log(f"Clusters written → {clusters_out.name} (k={clusters.attrs['n_clusters']})")

                if m == "cosine":
                    legacy_clu = run_dir / "drug_clusters.csv"
                    clusters.to_csv(legacy_clu, index=False)
                    log(f"Legacy clusters written → {legacy_clu.name}")

                # plots
                heat_png = fig_dir / f"drug_similarity_heatmap_{m}.png"
                scatter_png = fig_dir / f"drug_cluster_scatter_{m}.png"
                bar_png = fig_dir / f"drug_cluster_bar_{m}.png"

                log(f"Creating heatmap ({m}) …")
                _plot_heatmap(sim_df, clusters, heat_png, m)

                log(f"Creating PCA scatter ({m}) …")
                _plot_cluster_scatter(clusters, scatter_png, m)

                log(f"Creating cluster bar plot ({m}) …")
                _plot_cluster_bar(clusters, bar_png, m)

                # summary
                summary_txt = run_dir / f"drug_similarity_summary_{m}.txt"
                log(f"Writing summary ({m}) …")
                _write_summary(sim_df, clusters, summary_txt, m)

                successful_metrics.append(m)

            except Exception as e:
                log(f"❗ Error while processing metric '{m}', skipping it. Details: {e}")

        else:
            # single metric mode: behave like original (raise errors)
            log(f"Computing {m} similarity …")
            sim_df = _compute_similarity(drug_mat, metric=m)

            sim_out = run_dir / f"perturbation_similarity_{m}.csv"
            sim_df.to_csv(sim_out)
            log(f"Similarity matrix written → {sim_out.name}")

            if m == "cosine":
                legacy = run_dir / "perturbation_similarity.csv"
                sim_df.to_csv(legacy)
                log(f"Legacy similarity written → {legacy.name}")

            log("Running KMeans clustering on drug profiles …")
            clusters = _run_kmeans(drug_mat, metric=m)
            clusters_out = run_dir / f"drug_clusters_{m}.csv"
            clusters.to_csv(clusters_out, index=False)
            log(f"Clusters written → {clusters_out.name} (k={clusters.attrs['n_clusters']})")

            if m == "cosine":
                legacy_clu = run_dir / "drug_clusters.csv"
                clusters.to_csv(legacy_clu, index=False)
                log(f"Legacy clusters written → {legacy_clu.name}")

            heat_png = fig_dir / f"drug_similarity_heatmap_{m}.png"
            scatter_png = fig_dir / f"drug_cluster_scatter_{m}.png"
            bar_png = fig_dir / f"drug_cluster_bar_{m}.png"

            log("Creating heatmap …")
            _plot_heatmap(sim_df, clusters, heat_png, m)

            log("Creating PCA scatter …")
            _plot_cluster_scatter(clusters, scatter_png, m)

            log("Creating cluster bar plot …")
            _plot_cluster_bar(clusters, bar_png, m)

            summary_txt = run_dir / f"drug_similarity_summary_{m}.txt"
            log("Writing summary …")
            _write_summary(sim_df, clusters, summary_txt, m)

            successful_metrics.append(m)

    if not successful_metrics:
        # In "all" mode this means everything failed; in single mode,
        # we never get here because errors raise earlier.
        raise RuntimeError("Drug similarity analysis failed for all requested metrics.")

    log("✅ Drug similarity analysis complete for metrics: " + ", ".join(successful_metrics))
