from __future__ import annotations
# src/perturbation_analysis_uif.py
# src/perturbation_analysis_uif.py

import numpy as np
import pandas as pd
from pathlib import Path
from logging import Logger
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import linkage, fcluster

from ..plotly_mpl_export import save_plotly_png_with_mpl


def _save_plotly(fig, png_path: Path, html_path: Path, logger=None, scale: int = 2):
    """
    Helper to save Plotly figures as PNG + HTML.
    If static image export fails (no kaleido), we only save HTML.
    """
    from ..plotting_utils import safe_plot_and_export
    
    png_ok, html_ok = safe_plot_and_export(
        fig,
        png_path,
        html_path=html_path,
        fig_type="plotly",
        scale=scale,
        logger_instance=logger
    )
    
    if logger:
        if html_ok:
            logger.info("💾 Saved figure HTML: %s", html_path)
        if png_ok:
            logger.info("💾 Saved figure PNG: %s", png_path)


def run_crispr_perturbation(logger: Logger, output_dir: Path):
    """
    Step 4: Perturbation Analysis – CRISPR + Similarity

    Reads (from RUN_ROOT):
      - DepMap_Genes/InputGenes_selected.csv
      - DepMap_Dependencies / DepMap_GuideAnalysis per-model tables

    Writes (into RUN_ROOT/DepMap_GuideAnalysis):
      - CRISPR_Perturbation_*.csv
      - PerturbationSimilarity_*.csv
      - figs/, html/
    """
    # RUN_ROOT is the run-specific folder, e.g. DepMap_output/DepMap_20251201_xxxxx
    RUN_ROOT = output_dir

    # All perturbation outputs (CSVs + plots) go under DepMap_GuideAnalysis
    OUTDIR = output_dir / "DepMap_GuideAnalysis"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    out_figs = OUTDIR / "figs"
    out_html = OUTDIR / "html"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_html.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 4: Perturbation Analysis – CRISPR + Similarity ===")
    logger.info("Output directory: %s", output_dir)
    logger.info("OUTDIR  (guide outputs)      : %s", OUTDIR)

    # -----------------------------
    # CONFIG DEFAULTS
    # -----------------------------
    
    PV_DEP_THRESH = 0.01
    FC_DEP_THRESH = -2.0
    SIM_THRESH = 0.75
    N_CLUSTERS = 15
    DENDRO_MAX_GENES = 150

    logger.info("Using thresholds:")
    logger.info("  PV_DEP_THRESH (p-value cutoff): %s", PV_DEP_THRESH)
    logger.info("  FC_DEP_THRESH (Chronos FC threshold): %s", FC_DEP_THRESH)
    logger.info("  SIM_THRESH (similarity threshold): %s", SIM_THRESH)
    logger.info("  N_CLUSTERS (k-means / hierarchical clusters): %s", N_CLUSTERS)
    logger.info("  DENDRO_MAX_GENES (for dendrogram): %s", DENDRO_MAX_GENES)

    # ------------------------------------------------
    # 1. Load selected genes (transcriptomic side)
    # ------------------------------------------------
    # NOTE: genes live under RUN_ROOT, not inside DepMap_GuideAnalysis
    genes_path = RUN_ROOT / "DepMap_Genes" / "InputGenes_selected.csv"
    if not genes_path.exists():
        raise FileNotFoundError(f"Selected gene list not found: {genes_path}")

    genes_sel = pd.read_csv(genes_path)
    genes_sel.columns = [c.lower() for c in genes_sel.columns]
    genes_sel["gene"] = genes_sel["gene"].astype(str).str.upper().str.strip()

    if "direction" in genes_sel.columns:
        genes_sel = genes_sel.drop(columns=["direction"])

    for col in ["log2fc", "pvalue"]:
        if col not in genes_sel.columns:
            genes_sel[col] = np.nan

    logger.info("🧬 Selected genes from input: %d", len(genes_sel))

    # ------------------------------------------------
    # 2. Load per-model DepMap perturbation table
    # ------------------------------------------------
    deps_dir  = RUN_ROOT / "DepMap_Dependencies"
    guide_dir = RUN_ROOT / "DepMap_GuideAnalysis"

    cand_files = [
        # 1) Guide-level / screen-context aware for THIS run
        guide_dir / "GeneEssentiality_PerModel_withPerturbagen_and_ScreenType.csv",
        guide_dir / "GeneEssentiality_PerModel_withScreenContext.csv",

        # 2) Fallback in DepMap_Dependencies
        deps_dir / "GeneEssentiality_PerModel_withPerturbagen_and_ScreenType.csv",
        deps_dir / "GeneEssentiality_PerModel_withScreenContext.csv",
        deps_dir / "GeneEssentiality_PerModel.csv",
    ]

    per_path: Path | None = None
    for fp in cand_files:
        if fp.exists():
            per_path = fp
            break

    if per_path is None:
        raise FileNotFoundError(
            "Could not find per-model DepMap table.\n"
            "Expected one of:\n" + "\n".join(str(p) for p in cand_files)
        )

    logger.info("📂 Using per-model DepMap file: %s", per_path)
    per_full = pd.read_csv(per_path)

    if "Gene" not in per_full.columns:
        raise KeyError(
            "Per-model DepMap table is missing 'Gene' column. "
            f"Columns available: {list(per_full.columns)}"
        )

    per_full["Gene"] = per_full["Gene"].astype(str).str.upper().str.strip()
    per_sel = per_full[per_full["Gene"].isin(genes_sel["gene"])].copy()
    logger.info("DepMap rows for selected genes: %d", len(per_sel))

    if per_sel.empty:
        logger.warning(
            "No overlapping DepMap rows found for selected genes – "
            "downstream CRISPR perturbation analysis will be empty."
        )
    # ------------------------------------------------
    # 3. CRISPR Screen Analysis – Gene-level stats, p-values, confidence
    # ------------------------------------------------
    null_vals = per_full["ChronosGeneEffect"].dropna().values
    if len(null_vals) == 0:
        raise ValueError("No non-null ChronosGeneEffect values in DepMap table.")

    null_sorted = np.sort(null_vals)
    n_null = null_sorted.size

    per_nn = per_sel.dropna(subset=["ChronosGeneEffect"]).copy()
    grp = per_nn.groupby("Gene")

    gene_stats = grp["ChronosGeneEffect"].agg(
        best_effect="min",
        median_effect="median",
        q10=lambda s: s.quantile(0.10),
        q90=lambda s: s.quantile(0.90),
        n_models="count",
    ).reset_index()

    dep_prob = grp["DependencyProbability"].agg(
        best_depprob="max",
        mean_depprob="mean",
    ).reset_index()

    gene_stats = gene_stats.merge(dep_prob, on="Gene", how="left")

    def empirical_p(val):
        count = np.searchsorted(null_sorted, val, side="right")
        return (1.0 + count) / (1.0 + n_null)

    gene_stats["p_empirical"] = gene_stats["best_effect"].apply(empirical_p)

    # BH-FDR
    pvals = gene_stats["p_empirical"].values
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, m + 1)
    p_sorted = pvals[order]
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    q = np.clip(q, 0, 1)

    gene_stats["q_empirical"] = q
    gene_stats["neglog10_p"] = -np.log10(gene_stats["p_empirical"].clip(lower=1e-300))

    # Best essentiality tag
    tag_order = [
        "Core essential",
        "Strong dependency",
        "Moderate dependency",
        "Weak/Contextual",
        "Non-essential / growth-suppressive",
    ]
    tag_rank = {t: i for i, t in enumerate(tag_order)}

    per_sel["EssRank"] = (
        per_sel["EssentialityTag"].map(tag_rank).fillna(len(tag_order)).astype(int)
    )

    best_tag = (
        per_sel.groupby("Gene")
        .apply(lambda df: df.loc[df["EssRank"].idxmin(), "EssentialityTag"])
        .reset_index(name="BestEssentialityTag")
    )

    gene_stats = gene_stats.merge(best_tag, on="Gene", how="left")

    # Confidence tag + composite %
    def depmap_conf_tag(p):
        if pd.isna(p):
            return "No DepMap data"
        if p >= 0.90:
            return "Very high confidence"
        if p >= 0.70:
            return "High confidence"
        if p >= 0.50:
            return "Moderate confidence"
        if p >= 0.25:
            return "Low confidence"
        return "Very low confidence"

    gene_stats["DepMapConfTag"] = gene_stats["best_depprob"].apply(depmap_conf_tag)

    prob_component = gene_stats["best_depprob"].fillna(0.0)
    coverage_factor = 1.0 - np.exp(-gene_stats["n_models"] / 3.0)
    strength_factor = np.clip(-gene_stats["best_effect"] / 2.0, 0.0, 1.0)

    conf_raw = 0.6 * prob_component + 0.2 * coverage_factor + 0.2 * strength_factor
    gene_stats["DepMapConfidencePct"] = (conf_raw * 100.0).round(1)

    gene_merged = (
        genes_sel.merge(gene_stats, left_on="gene", right_on="Gene", how="left")
        .drop(columns=["Gene"])
    )

    out_gene = OUTDIR / "CRISPR_Perturbation_GeneStats.csv"
    gene_merged.to_csv(out_gene, index=False)
    logger.info("💾 Saved CRISPR gene-level perturbation stats: %s", out_gene)

    # Essential gene classification
    essential_tag_set = {"Core essential", "Strong dependency"}

    mask_tag = gene_merged["BestEssentialityTag"].isin(essential_tag_set)
    mask_p = gene_merged["p_empirical"].notna() & (
        gene_merged["p_empirical"] <= PV_DEP_THRESH
    )
    mask_fc = gene_merged["best_effect"].notna() & (
        gene_merged["best_effect"] <= FC_DEP_THRESH
    )

    mask_all = mask_tag & mask_p & mask_fc

    n_ess = int(mask_all.sum())
    n_total = len(gene_merged)
    mean_conf = (
        gene_merged.loc[mask_all, "DepMapConfidencePct"].dropna().mean().round(1)
        if n_ess > 0
        else np.nan
    )

    # Save essential genes table (for convenience)
    essential_df = gene_merged.loc[mask_all].copy()
    out_ess = OUTDIR / "CRISPR_Perturbation_EssentialGenes.csv"
    essential_df.to_csv(out_ess, index=False)
    logger.info("💾 Saved essential gene table: %s", out_ess)

    logger.info("\n===== CRISPR Screen Results – Essential Gene Classification =====")
    logger.info("Essential Genes           : %d", n_ess)
    logger.info("Total input genes         : %d", n_total)
    logger.info("FC Threshold (Chronos)    : %s", FC_DEP_THRESH)
    logger.info("P-value Cutoff (empirical): %s", PV_DEP_THRESH)
    logger.info("Mean Confidence           : %s%%", mean_conf)
    logger.info("==================================================================")

    # ------------------------------------------------
    # 5. Volcano plot
    # ------------------------------------------------
    volcano = gene_merged.copy()
    if volcano["log2fc"].notna().any():
        x_col = "log2fc"
        x_title = "Log2 Fold Change (transcriptomic)"
    else:
        x_col = "best_effect"
        x_title = "CRISPR Effect (Chronos best)"

    volcano["neglog10_p"] = -np.log10(
        volcano["p_empirical"].astype(float).clip(lower=1e-300)
    )

    fig_volcano = px.scatter(
        volcano,
        x=x_col,
        y="neglog10_p",
        color="BestEssentialityTag",
        hover_data=[
            "gene",
            "best_effect",
            "median_effect",
            "p_empirical",
            "q_empirical",
            "best_depprob",
            "DepMapConfidencePct",
            "DepMapConfTag",
        ],
        title=f"CRISPR Screen Results – Volcano plot: {x_title} vs -log10(p-value)",
        labels={x_col: x_title, "neglog10_p": "-log10(p_empirical)"},
    )

    fig_volcano.add_vline(
        x=FC_DEP_THRESH,
        line_dash="dash",
        annotation_text=f"FC≤{FC_DEP_THRESH}",
        annotation_position="bottom left",
    )
    fig_volcano.add_hline(
        y=-np.log10(PV_DEP_THRESH),
        line_dash="dash",
        annotation_text=f"p≤{PV_DEP_THRESH}",
        annotation_position="top right",
    )

    fig_volcano.update_layout(
        height=550,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # volcano table + figure
    out_volcano_tbl = OUTDIR / "CRISPR_Perturbation_VolcanoTable.csv"
    volcano.to_csv(out_volcano_tbl, index=False)
    logger.info("💾 Saved volcano table CSV: %s", out_volcano_tbl)

    _save_plotly(
        fig_volcano,
        out_figs / "chronos_volcano_medianEffect_vs_neglog10_pLike.png",
        out_html / "chronos_volcano_medianEffect_vs_neglog10_pLike.html",
        logger=logger,
    )

    # ------------------------------------------------
    # 6. Heatmaps
    # ------------------------------------------------
    N_MODELS_HEAT = 25

    # 6a. Chronos effect
    heat_df = per_sel.copy()
    heat_df = heat_df[~heat_df["ChronosGeneEffect"].isna()]

    model_counts = (
        heat_df.groupby("ModelID")["Gene"].count().sort_values(ascending=False)
    )
    top_models = model_counts.head(N_MODELS_HEAT).index.tolist()
    heat_df = heat_df[heat_df["ModelID"].isin(top_models)]

    heat_mat = heat_df.pivot_table(
        index="Gene",
        columns="CellLineName",
        values="ChronosGeneEffect",
        aggfunc="median",
    )
    heat_mat = heat_mat.reindex(index=genes_sel["gene"])

    out_heat_mat = OUTDIR / "CRISPR_ChronosEffect_HeatmapMatrix.csv"
    heat_mat.to_csv(out_heat_mat)
    logger.info("💾 Saved Chronos effect heatmap matrix: %s", out_heat_mat)

    fig_heat = px.imshow(
        heat_mat,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower",
        labels=dict(x="Cell line", y="Gene", color="Chronos effect"),
        title="CRISPR Screen – Chronos gene effect heatmap (selected genes × top cell lines)",
    )
    fig_heat.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _save_plotly(
        fig_heat,
        out_figs / "chronos_effect_heatmap.png",
        out_html / "chronos_effect_heatmap.html",
        logger=logger,
    )

    # 6b. Dependency probability
    dep_heat_df = per_sel.copy()
    dep_heat_df = dep_heat_df[~dep_heat_df["DependencyProbability"].isna()]
    dep_heat_df = dep_heat_df[dep_heat_df["ModelID"].isin(top_models)]

    dep_heat_mat = dep_heat_df.pivot_table(
        index="Gene",
        columns="CellLineName",
        values="DependencyProbability",
        aggfunc="median",
    )
    dep_heat_mat = dep_heat_mat.reindex(index=genes_sel["gene"])

    out_dep_mat = OUTDIR / "CRISPR_DependencyProbability_HeatmapMatrix.csv"
    dep_heat_mat.to_csv(out_dep_mat)
    logger.info(
        "💾 Saved dependency probability heatmap matrix: %s", out_dep_mat
    )

    fig_dep_heat = px.imshow(
        dep_heat_mat,
        aspect="auto",
        color_continuous_scale="Viridis",
        origin="lower",
        labels=dict(x="Cell line", y="Gene", color="Dependency probability"),
        title="CRISPR Screen – dependency score distribution heatmap (selected genes × top cell lines)",
    )
    fig_dep_heat.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _save_plotly(
        fig_dep_heat,
        out_figs / "dependency_probability_heatmap.png",
        out_html / "dependency_probability_heatmap.html",
        logger=logger,
    )

    # ------------------------------------------------
    # 7. Perturbation Similarity – Connectivity mapping
    # ------------------------------------------------
    mat_df = per_sel.copy()
    mat_df = mat_df[~mat_df["ChronosGeneEffect"].isna()]

    chronos_mat = mat_df.pivot_table(
        index="Gene",
        columns="ModelID",
        values="ChronosGeneEffect",
        aggfunc="median",
    )
    chronos_mat = chronos_mat.reindex(index=genes_sel["gene"]).dropna(how="all")

    out_raw = OUTDIR / "PerturbationSimilarity_Chronos_MatrixRaw.csv"
    chronos_mat.to_csv(out_raw)
    logger.info("💾 Saved raw Chronos matrix: %s", out_raw)

    col_medians = chronos_mat.median(axis=0)
    chronos_filled = chronos_mat.fillna(col_medians)

    out_imp = OUTDIR / "PerturbationSimilarity_Chronos_MatrixImputed.csv"
    chronos_filled.to_csv(out_imp)
    logger.info("💾 Saved imputed Chronos matrix: %s", out_imp)

    cos_sim_mat = cosine_similarity(chronos_filled.values)
    cos_sim_df = pd.DataFrame(
        cos_sim_mat, index=chronos_filled.index, columns=chronos_filled.index
    )

    pairs = []
    genes_list = list(chronos_filled.index)
    n_g = len(genes_list)

    for i in range(n_g):
        for j in range(i + 1, n_g):
            s = cos_sim_mat[i, j]
            if s >= SIM_THRESH:
                pairs.append(
                    {"Gene1": genes_list[i], "Gene2": genes_list[j], "CosineSimilarity": s}
                )

    pairs_df = pd.DataFrame(pairs)
    n_pairs = len(pairs_df)

    sim_mat_out = OUTDIR / "PerturbationSimilarity_Chronos_CosineMatrix.csv"
    cos_sim_df.to_csv(sim_mat_out)
    logger.info("💾 Saved gene × gene cosine similarity matrix: %s", sim_mat_out)

    edges_out = OUTDIR / "PerturbationSimilarity_Chronos_EdgesAboveThreshold.csv"
    pairs_df.to_csv(edges_out, index=False)
    logger.info("💾 Saved similar pairs (edges) table: %s", edges_out)

    # ------------------------------------------------
    # 8. Clustering – K-means + Hierarchical + PCA
    # ------------------------------------------------
    logger.info("\n📊 Clustering genes based on Chronos perturbation signatures ...")
    logger.info("  • Number of genes in matrix: %d", n_g)

    if n_g >= N_CLUSTERS:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(chronos_filled.values)
    else:
        kmeans_labels = np.arange(n_g)

    if n_g > 1:
        Z = linkage(chronos_filled.values, method="ward", metric="euclidean")
        if n_g >= N_CLUSTERS:
            hier_labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
        else:
            hier_labels = np.arange(1, n_g + 1)
    else:
        Z = None
        hier_labels = np.array([1])

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(chronos_filled.values)

    cluster_df = pd.DataFrame(
        {
            "Gene": chronos_filled.index,
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "KMeansCluster": kmeans_labels,
            "HierCluster": hier_labels,
        }
    )

    cluster_out = OUTDIR / "PerturbationSimilarity_Chronos_Clusters.csv"
    cluster_df.to_csv(cluster_out, index=False)
    logger.info("💾 Saved clustering results (K-means + hierarchical): %s", cluster_out)

    fig_clust_k = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="KMeansCluster",
        hover_data=["Gene"],
        title="Perturbation Similarity – PCA + K-means clusters (Chronos signatures)",
    )
    fig_clust_k.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _save_plotly(
        fig_clust_k,
        out_figs / "perturbation_similarity_pca_kmeans.png",
        out_html / "perturbation_similarity_pca_kmeans.html",
        logger=logger,
    )

    fig_clust_h = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="HierCluster",
        hover_data=["Gene"],
        title="Perturbation Similarity – PCA + hierarchical clusters (Chronos signatures)",
    )
    fig_clust_h.update_layout(
        height=600,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _save_plotly(
        fig_clust_h,
        out_figs / "perturbation_similarity_pca_hierarchical.png",
        out_html / "perturbation_similarity_pca_hierarchical.html",
        logger=logger,
    )

    # 8.6 Dendrogram
    if Z is not None:
        if n_g > DENDRO_MAX_GENES:
            gene_var = chronos_filled.var(axis=1)
            top_genes_for_dendro = (
                gene_var.sort_values(ascending=False)
                .head(DENDRO_MAX_GENES)
                .index
            )
            dendro_mat = chronos_filled.loc[top_genes_for_dendro]
            dendro_labels = dendro_mat.index.tolist()
            Z_d = linkage(dendro_mat.values, method="ward", metric="euclidean")
            logger.info(
                "📉 Dendrogram built on top %d most variable genes.", DENDRO_MAX_GENES
            )
        else:
            Z_d = Z
            dendro_labels = chronos_filled.index.tolist()

        fig_dendro = ff.create_dendrogram(
            chronos_filled.loc[dendro_labels].values,
            labels=dendro_labels,
            linkagefun=lambda x: linkage(x, method="ward", metric="euclidean"),
        )
        fig_dendro.update_layout(
            title="Hierarchical clustering dendrogram (perturbation signatures)",
            height=700,
            xaxis_title="Genes",
            yaxis_title="Distance",
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        _save_plotly(
            fig_dendro,
            out_figs / "perturbation_similarity_hierarchical_dendrogram.png",
            out_html / "perturbation_similarity_hierarchical_dendrogram.html",
            logger=logger,
        )

    # ------------------------------------------------
    # 9. Dashboard-style summary
    # ------------------------------------------------
    n_kmeans_clusters_identified = len(np.unique(kmeans_labels))
    n_hier_clusters_identified = len(np.unique(hier_labels))

    logger.info("\n===== Perturbation Similarity Analysis – Summary =====")
    logger.info("Similarity Threshold (cosine)      : %s", SIM_THRESH)
    logger.info("K-means clusters identified        : %d", n_kmeans_clusters_identified)
    logger.info("Hierarchical clusters identified   : %d", n_hier_clusters_identified)
    logger.info("Similar Pairs (edges ≥ threshold)  : %d", n_pairs)
    logger.info("======================================================")
