"""High-level visualisations for DepMap run outputs.

This module currently focuses on:
    • Visualising the selected DepMap models (DepMap_CellLines/Selected_Models.csv)
    • Visualising dependency summaries and per-gene behaviour
      (DepMap_Dependencies/*.csv)
    • Visualising guide-level behaviour and gene-from-guides summaries
      (DepMap_GuideAnalysis/*.csv)
    • Visualising Chronos differential gene stats (CRISPR_Perturbation_GeneStats.csv)

It uses Plotly for modern, interactive figures. For each figure we save:
    • An interactive HTML file
    • A static PNG (if kaleido / Chrome is available)

Typical output locations:
    output_dir / "DepMap_CellLines"      / "figs" / "html"
    output_dir / "DepMap_Dependencies"   / "figs" / "html"
    output_dir / "DepMap_GuideAnalysis"  / "figs" / "html"
    output_dir                           / "figs" / "html"   (Chronos volcano)

Example CLI usage (from project root):

    # 🔹 Default: run EVERYTHING with sensible defaults
    #    - All selected_models, dependencies, guides plots
    #    - Top 10 genes for per-gene plots
    #    - Top 30 genes for combined heatmaps / grids
    python run_visualization.py

    # Selected models only
    python run_visualization.py --target selected_models --charts all

    # Dependencies overview + per-gene plots for ASB4 (override defaults)
    python run_visualization.py \
      --target dependencies \
      --charts all \
      --top-n 30 \
      --sort-by median_effect_abs \
      --gene ASB4

    # Essentiality heatmap for top 30 genes (all in one panel)
    python run_visualization.py \
      --target dependencies \
      --charts top_genes_heatmap \
      --top-n 30 \
      --sort-by median_effect_abs

    # Guide-level QC / enrichment plots (all)
    python run_visualization.py --target guides --charts all

    # Guide-level plots focusing on one gene (e.g. FLG2)
    python run_visualization.py \
      --target guides \
      --charts gene_guide_rank \
      --gene FLG2
"""
from __future__ import annotations

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os

from ..plotly_mpl_export import save_plotly_png_with_mpl


logger = logging.getLogger("DepMap")

# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    """
    Configure logging to output ONLY to the console (no .log file).

    If the environment variable DEPMAP_COMPACT_LOG is set (to anything),
    we reduce log verbosity to WARNING and above. This keeps the log output
    much smaller, while still recording all important warnings/errors.

        Normal run (default):
            log level = INFO  (same behaviour as before)

        Compact run:
            DEPMAP_COMPACT_LOG=1 python run_full_pipeline.py
            -> log level = WARNING
    """

    # Decide log level based on environment
    compact_flag = os.getenv("DEPMAP_COMPACT_LOG", "").strip()
    if compact_flag:
        level = logging.WARNING
    else:
        level = logging.INFO  # original behaviour

    # Clean old handlers (avoid duplicate logging when re-running in the same session)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Only a console handler (no file)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    console_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console_handler)

    # Our project logger
    logger = logging.getLogger("DepMap")

    # Let it propagate to root so handlers are used
    logger.setLevel(level)
    logger.propagate = True

    # Quiet down some very chatty third-party loggers a bit
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger.info(f"Log level: {logging.getLevelName(level)} (DEPMAP_COMPACT_LOG={compact_flag})")
    return logger

def _ensure_logger() -> logging.Logger:
    """
    Ensure the DepMap logger is configured via setup_logger().
    This is safe to call multiple times.
    """
    global logger
    if not logger.handlers:
        logger = setup_logger()
    return logger


def _ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path]:
    """
    Create /figs and /html folders under the given base directory.
    Returns (fig_dir, html_dir).
    """
    figs = base_dir / "figs"
    html = base_dir / "html"
    figs.mkdir(parents=True, exist_ok=True)
    html.mkdir(parents=True, exist_ok=True)
    return figs, html


def _save_plot(
    fig,
    html_path: Path,
    png_path: Path,
    width: int = 1200,
    height: int = 700,
) -> None:
    """
    Save a Plotly figure as HTML and, if possible, as PNG.

    We keep the style consistent: white background, grid, nice margins.
    Thread-safe implementation using plotting_utils.
    """
    from ..plotting_utils import safe_plot_and_export
    
    log = _ensure_logger()
    
    # Use thread-safe plotting utility
    png_ok, html_ok = safe_plot_and_export(
        fig,
        png_path,
        html_path=html_path,
        fig_type="plotly",
        width=width,
        height=height,
        scale=2,
        logger_instance=log
    )
    
    # Logging is handled inside safe_plot_and_export, but we log here for consistency
    if html_ok:
        log.info("Saved HTML figure → %s", html_path)
    if png_ok:
        log.info("Saved PNG figure → %s", png_path)


# =====================================================================
# 1) Selected_Models visualisations
# =====================================================================


def load_selected_models(output_dir: Path) -> pd.DataFrame:
    """
    Load DepMap_CellLines/Selected_Models.csv from output_dir.

    If the file is missing, return an empty DataFrame and log a warning.
    """
    log = _ensure_logger()

    cellline_dir = output_dir / "DepMap_CellLines"
    csv_path = cellline_dir / "Selected_Models.csv"

    if not csv_path.exists():
        log.warning("Selected_Models.csv not found at: %s; skipping selected_models plots.", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    log.info("Loaded Selected_Models.csv → %s (rows=%d)", csv_path, len(df))
    return df


def plot_selected_models_by_primary_disease(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart: number of selected models per OncotreePrimaryDisease."""
    log = _ensure_logger()

    if df.empty:
        log.warning("Selected models table is empty; skipping primary disease bar plot.")
        return

    base_dir = output_dir / "DepMap_CellLines"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    counts = (
        df.groupby("OncotreePrimaryDisease", dropna=False)["ModelID"]
        .nunique()
        .reset_index(name="n_models")
        .sort_values("n_models", ascending=True)
    )

    if counts.empty:
        log.warning("No rows to plot for primary disease distribution.")
        return

    fig = px.bar(
        counts,
        x="n_models",
        y="OncotreePrimaryDisease",
        orientation="h",
        color="n_models",
        color_continuous_scale="Viridis",
        labels={
            "n_models": "Number of selected models",
            "OncotreePrimaryDisease": "Oncotree primary disease",
        },
        title="Selected DepMap models by Oncotree primary disease",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=160, r=40, t=80, b=60),
        coloraxis_colorbar=dict(title="# models"),
        yaxis=dict(categoryorder="total ascending"),
    )

    _save_plot(
        fig,
        html_path=html_dir / "SelectedModels_ByPrimaryDisease_bar.html",
        png_path=figs_dir / "SelectedModels_ByPrimaryDisease_bar.png",
    )


def plot_selected_models_by_lineage(df: pd.DataFrame, output_dir: Path) -> None:
    """Vertical bar chart: number of selected models per OncotreeLineage."""
    log = _ensure_logger()

    if df.empty:
        log.warning("Selected models table is empty; skipping lineage bar plot.")
        return

    base_dir = output_dir / "DepMap_CellLines"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    counts = (
        df.groupby("OncotreeLineage", dropna=False)["ModelID"]
        .nunique()
        .reset_index(name="n_models")
        .sort_values("n_models", ascending=False)
    )

    if counts.empty:
        log.warning("No rows to plot for lineage distribution.")
        return

    fig = px.bar(
        counts,
        x="OncotreeLineage",
        y="n_models",
        color="OncotreeLineage",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={
            "OncotreeLineage": "Oncotree lineage",
            "n_models": "Number of selected models",
        },
        title="Selected DepMap models by Oncotree lineage",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=120),
        showlegend=False,
        xaxis_tickangle=-40,
    )

    _save_plot(
        fig,
        html_path=html_dir / "SelectedModels_ByLineage_bar.html",
        png_path=figs_dir / "SelectedModels_ByLineage_bar.png",
    )


def plot_selected_models_bubble_lineage_vs_disease(df: pd.DataFrame, output_dir: Path) -> None:
    """Bubble scatter: (PrimaryDisease, Lineage) with bubble size = #models."""
    log = _ensure_logger()

    if df.empty:
        log.warning("Selected models table is empty; skipping bubble plot.")
        return

    base_dir = output_dir / "DepMap_CellLines"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    combo = (
        df.groupby(
            ["OncotreeLineage", "OncotreePrimaryDisease"], dropna=False
        )["ModelID"]
        .nunique()
        .reset_index(name="n_models")
    )

    if combo.empty:
        log.warning("No rows to plot for lineage vs disease bubble chart.")
        return

    fig = px.scatter(
        combo,
        x="OncotreePrimaryDisease",
        y="OncotreeLineage",
        size="n_models",
        color="OncotreeLineage",
        size_max=45,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={
            "OncotreePrimaryDisease": "Oncotree primary disease",
            "OncotreeLineage": "Oncotree lineage",
            "n_models": "# selected models",
        },
        title="Selected models: lineage × primary disease (bubble size = #models)",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=120, r=40, t=80, b=160),
        xaxis_tickangle=-40,
    )

    _save_plot(
        fig,
        html_path=html_dir / "SelectedModels_LineageDisease_bubble.html",
        png_path=figs_dir / "SelectedModels_LineageDisease_bubble.png",
    )


def plot_selected_models_sunburst(df: pd.DataFrame, output_dir: Path) -> None:
    """Sunburst: inner ring = lineage, outer ring = primary disease."""
    log = _ensure_logger()

    if df.empty:
        log.warning("Selected models table is empty; skipping sunburst.")
        return

    base_dir = output_dir / "DepMap_CellLines"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    fig = px.sunburst(
        df,
        path=["OncotreeLineage", "OncotreePrimaryDisease"],
        values=None,  # count rows
        color="OncotreeLineage",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Selected models: lineage → primary disease (sunburst)",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=80, b=10),
    )

    _save_plot(
        fig,
        html_path=html_dir / "SelectedModels_LineageDisease_sunburst.html",
        png_path=figs_dir / "SelectedModels_LineageDisease_sunburst.png",
        width=800,
        height=800,
    )


def run_selected_models_visualisations(output_dir: Path, charts: Iterable[str] | None = None) -> None:
    """
    Orchestrate all Selected_Models visualisations.
    """
    log = _ensure_logger()

    if charts is None:
        charts = ["all"]
    charts = [c.strip().lower() for c in charts]

    df = load_selected_models(output_dir)
    if df.empty:
        log.warning("Selected_Models.csv missing or empty; skipping selected_models visualisations.")
        return

    if "all" in charts:
        charts = ["disease_bar", "lineage_bar", "bubble", "sunburst"]

    log.info("Selected_Models visualisations requested: %s", charts)

    if "disease_bar" in charts:
        plot_selected_models_by_primary_disease(df, output_dir)
    if "lineage_bar" in charts:
        plot_selected_models_by_lineage(df, output_dir)
    if "bubble" in charts:
        plot_selected_models_bubble_lineage_vs_disease(df, output_dir)
    if "sunburst" in charts:
        plot_selected_models_sunburst(df, output_dir)


# =====================================================================
# 2) Dependencies visualisations
# =====================================================================


def load_dependencies_summary(output_dir: Path) -> pd.DataFrame:
    """
    Load DepMap_Dependencies/Dependencies_GeneSummary_SelectedModels.csv.

    If the file is missing, return an empty DataFrame and log a warning.
    """
    log = _ensure_logger()
    dep_dir = output_dir / "DepMap_Dependencies"
    csv_path = dep_dir / "Dependencies_GeneSummary_SelectedModels.csv"
    if not csv_path.exists():
        log.warning(
            "Dependencies_GeneSummary_SelectedModels.csv not found at: %s; "
            "skipping dependencies summary–based plots.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded gene summary → %s (rows=%d)", csv_path, len(df))
    return df


def load_dependencies_tidy(output_dir: Path) -> pd.DataFrame:
    """Load DepMap_Dependencies/Dependencies_Tidy_SelectedGenes_SelectedModels.csv."""
    log = _ensure_logger()
    dep_dir = output_dir / "DepMap_Dependencies"
    csv_path = dep_dir / "Dependencies_Tidy_SelectedGenes_SelectedModels.csv"
    if not csv_path.exists():
        log.warning(
            "Dependencies_Tidy_SelectedGenes_SelectedModels.csv not found at: %s; "
            "skipping tidy per-gene plots.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded tidy dependencies → %s (rows=%d)", csv_path, len(df))
    return df


def load_top_dependents(output_dir: Path) -> pd.DataFrame:
    """Load DepMap_Dependencies/PerGene_TopDependents_long.csv."""
    log = _ensure_logger()
    dep_dir = output_dir / "DepMap_Dependencies"
    csv_path = dep_dir / "PerGene_TopDependents_long.csv"
    if not csv_path.exists():
        log.warning(
            "PerGene_TopDependents_long.csv not found at: %s; "
            "skipping top dependents plots.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded per-gene top dependents → %s (rows=%d)", csv_path, len(df))
    return df


# ---- Chronos differential gene stats (volcano) ----------------------


def load_chronos_gene_stats(output_dir: Path) -> pd.DataFrame:
    """
    Load CRISPR_Perturbation_GeneStats.csv from output_dir.

    This is the Chronos differential gene stats table used for a volcano-style plot.
    """
    log = _ensure_logger()
    csv_path = output_dir / "DepMap_GuideAnalysis" / "CRISPR_Perturbation_GeneStats.csv"
    if not csv_path.exists():
        log.warning(
            "CRISPR_Perturbation_GeneStats.csv not found at: %s; "
            "skipping Chronos volcano plot.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded Chronos gene stats → %s (rows=%d)", csv_path, len(df))
    return df


def plot_chronos_volcano(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Volcano-style plot from Chronos differential gene model.

    Mirrors the logic of:

        Step 5: Volcano-style plot from differential model (Chronos)

    Expected columns:
        gene, median_effect, BestEssentialityTag,
        and either neglog10_p or pvalue (to derive -log10(p)).
    """
    log = _ensure_logger()

    if df.empty:
        log.warning("Chronos gene stats table is empty; skipping volcano plot.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    # Standardise and check required columns
    df = df.copy()
    df = df.rename(columns={"gene": "Gene", "BestEssentialityTag": "BiologicalTag"})

    required = ["Gene", "median_effect", "BiologicalTag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(
            "Chronos gene stats missing required columns %s; skipping volcano plot.",
            missing,
        )
        return

    # Decide on y-axis: use neglog10_p if present; otherwise compute from pvalue
    if "neglog10_p" in df.columns:
        y_col = "neglog10_p"
        log.info("Using existing 'neglog10_p' as volcano y-axis.")
    elif "pvalue" in df.columns:
        log.info("Column 'neglog10_p' not found – computing from 'pvalue'.")
        eps = 1e-300
        df["neglog10_p"] = -np.log10(np.clip(df["pvalue"].astype(float), eps, 1.0))
        y_col = "neglog10_p"
    else:
        log.warning(
            "Could not find 'neglog10_p' or 'pvalue' in Chronos gene stats; "
            "skipping volcano plot."
        )
        return

    y_label = "-log10(p-value)"

    # Hover data: only include columns that actually exist
    hover_candidates = [
        "Gene",
        "median_effect",
        y_col,
        "pvalue",
        "DepMapConfidencePct",
        "best_depprob",
        "mean_depprob",
        "p_empirical",
        "q_empirical",
    ]
    hover_cols = [c for c in hover_candidates if c in df.columns]

    # Optional: color mapping for typical tags (others will use default colors)
    color_map = {
        "Core-essential": "#d62728",
        "Strongly selective": "#ff7f0e",
        "Weak/Contextual": "#1f77b4",
        "Non-essential / growth-suppressive": "#2ca02c",
    }

    fig_volcano = px.scatter(
        df,
        x="median_effect",
        y=y_col,
        color="BiologicalTag",
        hover_data=hover_cols,
        title="CRISPR Chronos volcano: median gene effect vs -log10(p-value)",
        labels={
            "median_effect": "Median Chronos gene effect",
            y_col: y_label,
            "BiologicalTag": "Essentiality category",
        },
        color_discrete_map=color_map,
    )

    fig_volcano.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=0.4, color="black"),
        )
    )
    fig_volcano.update_layout(
        height=700,
        template="plotly_white",
        legend_title_text="BiologicalTag",
    )

    html_path = html_dir / "chronos_volcano_medianEffect_vs_neglog10_pLike.html"
    png_path = figs_dir / "chronos_volcano_medianEffect_vs_neglog10_pLike.png"

    _save_plot(
        fig_volcano,
        html_path=html_path,
        png_path=png_path,
        width=1200,
        height=700,
    )


# ---------------------------------------------------------------------
# 2a) Overview grid (sectors rising / falling style)
# ---------------------------------------------------------------------


def plot_dependency_overview_grid(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 30,
    sort_by: str = "median_effect_abs",
) -> None:
    """
    Grid / panel overview of gene dependencies, inspired by
    the 'sectors rising / falling' style.

    • Genes with median_effect < 0 are shown in a lower band
      ("Stronger dependency").
    • Genes with median_effect ≥ 0 are shown in an upper band
      ("Weaker / non-essential").
    • Each gene is a small square; colour = median_effect (diverging scale).
    • Top-N genes (by `sort_by`) are outlined with a black circle.
      Gene names are visible on HOVER (no text labels).
    """
    log = _ensure_logger()

    if df.empty:
        log.warning("Dependencies summary table is empty; skipping overview grid.")
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    df = df.copy()
    df["median_effect_abs"] = df["median_effect"].abs()

    if sort_by not in df.columns:
        log.warning(
            "sort_by=%s not in columns; falling back to 'median_effect_abs'", sort_by
        )
        sort_by = "median_effect_abs"

    # two bands: up (>=0) vs down (<0)
    df["DirectionBand"] = np.where(
        df["median_effect"] < 0,
        "Stronger dependency (median_effect < 0)",
        "Weaker / non-essential (median_effect ≥ 0)",
    )

    # x-position within each band to form a horizontal strip of squares
    df["x_index"] = (
        df.sort_values(["DirectionBand", "median_effect"])
        .groupby("DirectionBand")
        .cumcount()
    )

    # base scatter of coloured squares
    fig = px.scatter(
        df,
        x="x_index",
        y="DirectionBand",
        color="median_effect",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        size="n_prob50",
        size_max=12,
        hover_data=[
            "Gene",
            "median_effect",
            "q10",
            "q90",
            "n_models",
            "n_prob50",
            "n_strong_lt_1",
            "BiologicalTag",
        ],
        labels={
            "x_index": "Genes (ordered within each band)",
            "DirectionBand": "",
            "median_effect": "Median Chronos effect",
            "n_prob50": "# models with DepProb ≥ 0.5",
        },
        title=(
            "Gene dependency landscape<br>"
            "Colour = median Chronos effect (RdYlGn, centred at 0); "
            "upper band = weaker/non-essential, lower band = stronger dependency "
            "— hover squares for gene names and details"
        ),
    )

    fig.update_traces(
        marker=dict(symbol="square", opacity=0.95, line=dict(width=0)),
        selector=dict(mode="markers"),
    )

    # highlight top-N genes with a black circle outline (no text)
    df_sorted = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(top_n).copy()

    if not top_df.empty:
        fig.add_trace(
            go.Scatter(
                x=top_df["x_index"],
                y=top_df["DirectionBand"],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=18,
                    line=dict(color="black", width=1.6),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=80, r=80, t=110, b=60),
        coloraxis_colorbar=dict(
            title="Median effect",
            ticks="outside",
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=[
                "Stronger dependency (median_effect < 0)",
                "Weaker / non-essential (median_effect ≥ 0)",
            ],
        ),
    )

    _save_plot(
        fig,
        html_path=html_dir / "Dependencies_GeneOverview_grid.html",
        png_path=figs_dir / "Dependencies_GeneOverview_grid.png",
        width=1600,
        height=500,
    )


# ---------------------------------------------------------------------
# 2b) Other dependency plots (biotag, per-gene, essentiality)
# ---------------------------------------------------------------------


def plot_dependency_biological_tag_bar(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: count of genes per BiologicalTag."""
    log = _ensure_logger()

    if df.empty:
        log.warning("Dependencies summary table is empty; skipping BiologicalTag bar.")
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    counts = (
        df.groupby("BiologicalTag", dropna=False)["Gene"]
        .nunique()
        .reset_index(name="n_genes")
        .sort_values("n_genes", ascending=False)
    )

    if counts.empty:
        log.warning("No rows to plot for BiologicalTag distribution.")
        return

    fig = px.bar(
        counts,
        x="BiologicalTag",
        y="n_genes",
        color="BiologicalTag",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"BiologicalTag": "Biological tag", "n_genes": "# genes"},
        title="Genes by biological dependency tag",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=160),
        showlegend=False,
        xaxis_tickangle=-30,
    )

    _save_plot(
        fig,
        html_path=html_dir / "Dependencies_BiologicalTag_bar.html",
        png_path=figs_dir / "Dependencies_BiologicalTag_bar.png",
    )


def plot_gene_dependency_across_models(df_tidy: pd.DataFrame, gene: str, output_dir: Path) -> None:
    """
    For a single gene, show ChronosGeneEffect and DependencyProbability
    across models.

    • x-axis: ChronosGeneEffect
    • y-axis: ModelLabel (CellLineName + primary disease)
    • bubble size: DependencyProbability (NaN -> 0 so Plotly doesn't crash)
    """
    log = _ensure_logger()

    if df_tidy.empty:
        log.warning("Tidy dependencies table is empty; skipping per-model scatter.")
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    gene_mask = df_tidy["Gene"].astype(str) == str(gene)
    df_gene = df_tidy.loc[gene_mask].copy()

    if df_gene.empty:
        log.warning("No rows found in tidy dependencies for gene=%s; skipping.", gene)
        return

    # Order by Chronos effect (strongest dependency to the left/right)
    df_gene = df_gene.sort_values("ChronosGeneEffect", ascending=True)

    df_gene["ModelLabel"] = (
        df_gene["CellLineName"].astype(str)
        + " (" + df_gene["OncotreePrimaryDisease"].astype(str) + ")"
    )

    # -----------------------------
    # Handle DependencyProbability
    # -----------------------------
    if "DependencyProbability" not in df_gene.columns:
        log.warning(
            "Column 'DependencyProbability' missing for gene=%s; "
            "using constant size for all points.",
            gene,
        )
        df_gene["DepProb_for_plot"] = 1.0
    else:
        dep_prob = pd.to_numeric(
            df_gene["DependencyProbability"], errors="coerce"
        ).fillna(0.0)  # NaN -> 0 for plotting
        df_gene["DepProb_for_plot"] = dep_prob

    fig = px.scatter(
        df_gene,
        x="ChronosGeneEffect",
        y="ModelLabel",
        size="DepProb_for_plot",
        color="EssentialityTag",
        hover_data=[
            "ModelID",
            "CellLineName",
            "OncotreeLineage",
            "OncotreePrimaryDisease",
            "DependencyProbability"
            if "DependencyProbability" in df_gene.columns
            else None,
        ],
        labels={
            "ChronosGeneEffect": f"{gene} Chronos effect (per model)",
            "ModelLabel": "Model / disease context",
            "DepProb_for_plot": "DepProb (NaN → 0 for plotting)",
        },
        title=f"{gene}: Chronos effect vs model (bubble size = DepProb)",
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=20,
        orientation="h",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=220, r=40, t=80, b=60),
        legend_title_text="Essentiality tag",
    )

    _save_plot(
        fig,
        html_path=html_dir / f"{gene}_PerModel_Effect_DepProb_scatter.html",
        png_path=figs_dir / f"{gene}_PerModel_Effect_DepProb_scatter.png",
        width=1300,
        height=max(600, 20 * len(df_gene)),
    )


def _encode_essentiality(series: pd.Series) -> Tuple[pd.Series, list[str]]:
    """
    Encode EssentialityTag strings as integers 0..K-1 and
    return (coded_series, ordered_tag_list).
    """
    tags = [t for t in series.dropna().unique().tolist()]
    # stable order but group similar tags roughly:
    preferred_order = [
        "Moderate dependency",
        "Strong dependency",
        "Weak/Contextual",
        "Non-essential / growth-suppressive",
    ]
    ordered = [t for t in preferred_order if t in tags] + [
        t for t in tags if t not in preferred_order
    ]
    tag_to_int = {t: i for i, t in enumerate(ordered)}
    coded = series.map(tag_to_int)
    return coded, ordered


def plot_gene_essentiality_heatmap(df_tidy: pd.DataFrame, gene: str, output_dir: Path) -> None:
    """
    Single-gene heatmap:
    y = gene, x = models ordered by Chronos effect,
    colour = EssentialityTag (categorical).
    """
    log = _ensure_logger()

    if df_tidy.empty:
        log.warning("Tidy dependencies table is empty; skipping single-gene heatmap.")
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    df_gene = df_tidy[df_tidy["Gene"].astype(str) == str(gene)].copy()
    if df_gene.empty:
        log.warning("No rows found in tidy dependencies for gene=%s; skipping heatmap.", gene)
        return

    # order models by Chronos effect (strongest dependency on right)
    df_gene = df_gene.sort_values("ChronosGeneEffect", ascending=True)

    df_gene["ModelLabel"] = df_gene["CellLineName"].astype(str)

    codes, tags = _encode_essentiality(df_gene["EssentialityTag"])

    z = np.array([codes.values])  # 1 × N
    x_labels = df_gene["ModelLabel"].tolist()
    y_labels = [gene]

    if z.size == 0 or len(x_labels) == 0:
        log.warning("No data points for essentiality heatmap of gene=%s.", gene)
        return

    # colour palette for categories
    base_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    colors = base_palette[: max(1, len(tags))]
    color_scale = [(i / max(1, len(colors) - 1), c) for i, c in enumerate(colors)]

    fig = px.imshow(
        z,
        x=x_labels,
        y=y_labels,
        color_continuous_scale=color_scale,
        aspect="auto",
        zmin=-0.5,
        zmax=len(tags) - 0.5,
    )

    fig.update_layout(
        template="plotly_white",
        title=f"{gene}: essentiality landscape across models",
        xaxis_title="Models (ordered by Chronos effect)",
        yaxis_title="",
        coloraxis_colorbar=dict(
            title="Essentiality tag",
            tickmode="array",
            tickvals=list(range(len(tags))),
            ticktext=tags,
        ),
        margin=dict(l=120, r=80, t=80, b=120),
    )

    _save_plot(
        fig,
        html_path=html_dir / f"{gene}_Essentiality_heatmap.html",
        png_path=figs_dir / f"{gene}_Essentiality_heatmap.png",
        width=max(1200, 40 * len(x_labels)),
        height=500,
    )


def plot_top_genes_essentiality_heatmap(
    output_dir: Path,
    df_tidy: pd.DataFrame,
    df_summary: pd.DataFrame,
    top_n: int = 30,
    sort_by: str = "median_effect_abs",
) -> None:
    """
    Essentiality heatmap for the top-N genes in ONE panel.

    • Rows  = genes (top_n, chosen by `sort_by` on the summary table)
    • Cols  = models (all models for those genes)
    • Color = EssentialityTag (categorical; same legend as single-gene plot)
    """
    log = _ensure_logger()

    if df_tidy.empty or df_summary.empty:
        log.warning(
            "Either tidy or summary dependencies table empty; skipping top genes heatmap."
        )
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    # --- choose top-N genes from summary --------------------------------
    df_sum = df_summary.copy()
    if "median_effect_abs" not in df_sum.columns:
        df_sum["median_effect_abs"] = df_sum["median_effect"].abs()

    if sort_by not in df_sum.columns:
        log.warning("sort_by=%s not found in summary; using 'median_effect_abs'", sort_by)
        sort_by = "median_effect_abs"

    df_sum = df_sum.sort_values(sort_by, ascending=False)
    top_genes = df_sum["Gene"].dropna().astype(str).unique().tolist()[:top_n]
    if not top_genes:
        log.warning("No genes available for top_genes_heatmap.")
        return

    log.info(
        "Top %d genes for essentiality heatmap (by %s): %s",
        len(top_genes),
        sort_by,
        top_genes,
    )

    # --- filter tidy table to those genes --------------------------------
    df_tidy = df_tidy[df_tidy["Gene"].astype(str).isin(top_genes)].copy()
    if df_tidy.empty:
        log.warning("No tidy rows found for selected top genes.")
        return

    # order models by global median Chronos effect across these genes
    model_order = (
        df_tidy.groupby("CellLineName")["ChronosGeneEffect"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    gene_order = top_genes  # already in ranking order

    # encode essentiality tags globally for this subset
    codes, tags = _encode_essentiality(df_tidy["EssentialityTag"])
    df_tidy["EssCode"] = codes

    # build pivot: genes × models → EssCode
    pivot = (
        df_tidy.pivot_table(
            index="Gene",
            columns="CellLineName",
            values="EssCode",
            aggfunc="first",
        )
        .reindex(index=gene_order)
        .reindex(columns=model_order)
    )

    z = pivot.values
    y_labels = pivot.index.tolist()
    x_labels = pivot.columns.tolist()

    if z.size == 0:
        log.warning("Pivot for top_genes_heatmap is empty.")
        return

    base_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    colors = base_palette[: max(1, len(tags))]
    color_scale = [(i / max(1, len(colors) - 1), c) for i, c in enumerate(colors)]

    fig = px.imshow(
        z,
        x=x_labels,
        y=y_labels,
        color_continuous_scale=color_scale,
        aspect="auto",
        zmin=-0.5,
        zmax=len(tags) - 0.5,
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            f"Top {len(y_labels)} genes: essentiality landscape across models<br>"
            f"(ranked by {sort_by})"
        ),
        xaxis_title="Models (ordered by median Chronos effect across top genes)",
        yaxis_title="Genes",
        coloraxis_colorbar=dict(
            title="Essentiality tag",
            tickmode="array",
            tickvals=list(range(len(tags))),
            ticktext=tags,
        ),
        margin=dict(l=160, r=80, t=90, b=160),
    )

    fig.update_xaxes(tickangle=-60)

    _save_plot(
        fig,
        html_path=html_dir
        / f"TopGenes_Essentiality_heatmap_top{len(y_labels)}.html",
        png_path=figs_dir
        / f"TopGenes_Essentiality_heatmap_top{len(y_labels)}.png",
        width=max(1400, 30 * len(x_labels)),
        height=max(700, 30 * len(y_labels) + 200),
    )


def plot_top_dependents_for_gene(
    output_dir: Path,
    df_top: pd.DataFrame,
    gene: str,
    max_rank: int = 5,
) -> None:
    """For a single gene, show its top dependent models."""
    log = _ensure_logger()

    if df_top.empty:
        log.warning("Top dependents table is empty; skipping gene=%s.", gene)
        return

    base_dir = output_dir / "DepMap_Dependencies"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    df_gene = df_top[df_top["Gene"].astype(str) == str(gene)].copy()
    if df_gene.empty:
        log.warning("No top dependents found for gene=%s; skipping.", gene)
        return

    df_gene = df_gene[df_gene["TopRank"] <= max_rank].copy()
    df_gene = df_gene.sort_values("GeneEffect", ascending=True)

    df_gene["ModelLabel"] = (
        df_gene["CellLineName"].astype(str)
        + " (" + df_gene["OncotreePrimaryDisease"].astype(str) + ")"
    )

    fig = px.bar(
        df_gene,
        x="GeneEffect",
        y="ModelLabel",
        color="EssentialityTag",
        orientation="h",
        labels={
            "GeneEffect": f"{gene} Chronos effect",
            "ModelLabel": "Top dependent models",
        },
        title=f"{gene}: top dependent models (TopRank ≤ {max_rank})",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=220, r=40, t=80, b=60),
        legend_title_text="Essentiality tag",
    )

    _save_plot(
        fig,
        html_path=html_dir / f"{gene}_TopDependents_bar.html",
        png_path=figs_dir / f"{gene}_TopDependents_bar.png",
        width=1200,
        height=max(500, 40 * len(df_gene)),
    )


def run_dependencies_visualisations(
    output_dir: Path,
    charts: Iterable[str] | None = None,
    top_n: int = 30,  # combined (heatmaps / overview)
    sort_by: str = "median_effect_abs",
    gene: Optional[str] = None,
) -> None:
    """
    Orchestrate dependencies visualisations.

    Defaults:
        • top_n = 30 for combined heatmaps / overview grids
        • Top 10 genes are used for per-gene plots when --gene is not provided.

    charts: names from:
        {
          'overview_scatter',       # grid / sectors plot
          'biotag_bar',             # biological-tag counts
          'per_gene',               # per-gene scatter vs models
          'per_gene_heatmap',       # single-gene essentiality heatmap
          'top_genes_heatmap',      # top-N genes essentiality heatmap
          'top_dependents',         # bar of top dependent models for gene(s)
          'chronos_volcano',        # volcano from CRISPR_Perturbation_GeneStats
          'all'
        }
    """
    log = _ensure_logger()

    if charts is None:
        charts = ["all"]
    charts = [c.strip().lower() for c in charts]

    if "all" in charts:
        charts = [
            "overview_scatter",
            "biotag_bar",
            "per_gene",
            "per_gene_heatmap",
            "top_dependents",
            "top_genes_heatmap",
            "chronos_volcano",
        ]

    log.info(
        "Dependencies visualisations requested: charts=%s, top_n=%d, sort_by=%s, gene=%s",
        charts,
        top_n,
        sort_by,
        gene,
    )

    # Load summary once if needed
    df_summary = None
    if any(
        c in charts
        for c in (
            "overview_scatter",
            "biotag_bar",
            "top_genes_heatmap",
            "per_gene",
            "per_gene_heatmap",
            "top_dependents",
        )
    ):
        df_summary = load_dependencies_summary(output_dir=output_dir)
        if df_summary.empty:
            log.warning(
                "Dependencies summary missing/empty; skipping all summary-based charts."
            )
            df_summary = None

    # Overview scatter
    if "overview_scatter" in charts and df_summary is not None:
        plot_dependency_overview_grid(output_dir=output_dir, df=df_summary, top_n=top_n, sort_by=sort_by)

    # Biotag bar
    if "biotag_bar" in charts and df_summary is not None:
        plot_dependency_biological_tag_bar(output_dir=output_dir, df=df_summary)

    # Combined essentiality heatmap of top-N genes
    if "top_genes_heatmap" in charts and df_summary is not None:
        df_tidy_for_heatmap = load_dependencies_tidy(output_dir=output_dir)
        if not df_tidy_for_heatmap.empty:
            plot_top_genes_essentiality_heatmap(
                output_dir=output_dir,
                df_tidy=df_tidy_for_heatmap,
                df_summary=df_summary,
                top_n=top_n,
                sort_by=sort_by,
            )

    # Chronos volcano (from CRISPR_Perturbation_GeneStats.csv)
    if "chronos_volcano" in charts:
        df_chronos = load_chronos_gene_stats(output_dir=output_dir)
        if not df_chronos.empty:
            plot_chronos_volcano(output_dir=output_dir, df=df_chronos)

    # Per-gene style plots:
    #   - if gene is provided: only that gene
    #   - if gene is None: top 10 genes by sort_by from summary
    n_per_gene = 10
    genes_for_per_gene: list[str] = []

    if any(c in charts for c in ("per_gene", "per_gene_heatmap", "top_dependents")):
        if gene:
            genes_for_per_gene = [gene]
        else:
            if df_summary is None:
                log.warning(
                    "Per-gene charts requested but no summary / gene list available; skipping."
                )
            else:
                df_sum = df_summary.copy()
                if "median_effect_abs" not in df_sum.columns:
                    df_sum["median_effect_abs"] = df_sum["median_effect"].abs()
                if sort_by not in df_sum.columns:
                    log.warning(
                        "sort_by=%s not found for per-gene; using 'median_effect_abs'",
                        sort_by,
                    )
                    sort_by = "median_effect_abs"
                df_sum = df_sum.sort_values(sort_by, ascending=False)
                genes_for_per_gene = (
                    df_sum["Gene"].dropna().astype(str).unique().tolist()[:n_per_gene]
                )
                if not genes_for_per_gene:
                    log.warning(
                        "No genes available for per-gene charts; skipping those."
                    )

    # Per-gene scatter and heatmap (dependencies_tidy)
    if genes_for_per_gene:
        df_tidy = load_dependencies_tidy(output_dir=output_dir)
        if not df_tidy.empty:
            if "per_gene" in charts:
                for g in genes_for_per_gene:
                    plot_gene_dependency_across_models(output_dir=output_dir, df_tidy=df_tidy, gene=g)
            if "per_gene_heatmap" in charts:
                for g in genes_for_per_gene:
                    plot_gene_essentiality_heatmap(output_dir=output_dir, df_tidy=df_tidy, gene=g)

    # Per-gene top dependents (PerGene_TopDependents_long.csv)
    if genes_for_per_gene and "top_dependents" in charts:
        df_top = load_top_dependents(output_dir=output_dir)
        if not df_top.empty:
            for g in genes_for_per_gene:
                plot_top_dependents_for_gene(output_dir=output_dir, df_top=df_top, gene=g, max_rank=5)


# =====================================================================
# 3) Guide-level visualisations (DepMap_GuideAnalysis)
# =====================================================================

# Thresholds used upstream to classify guides and genes
GUIDE_DEPLETE_THRESH = -1.0
GUIDE_ENRICH_THRESH = 0.5
FC_DEP_THRESH = -2.0  # used for gene-level "strongly depleted" classification


def load_guides_long(output_dir: Path) -> pd.DataFrame:
    """
    Load DepMap_GuideAnalysis/CRISPR_GuideLevel_Avana_SelectedModels_long.csv.
    """
    log = _ensure_logger()
    guide_dir = output_dir / "DepMap_GuideAnalysis"
    csv_path = guide_dir / "CRISPR_GuideLevel_Avana_SelectedModels_long.csv"
    if not csv_path.exists():
        log.warning(
            "CRISPR_GuideLevel_Avana_SelectedModels_long.csv not found at: %s; "
            "skipping guide-level plots.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded guide-level table → %s (rows=%d)", csv_path, len(df))
    return df


def load_gene_from_guides_overall(output_dir: Path) -> pd.DataFrame:
    """
    Load gene-level summary aggregated from guides across all selected models.
    """
    log = _ensure_logger()
    guide_dir = output_dir / "DepMap_GuideAnalysis"
    csv_path = guide_dir / "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels.csv"
    if not csv_path.exists():
        log.warning(
            "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels.csv not found at: %s; "
            "skipping overall gene-from-guides volcano.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info("Loaded gene-from-guides (overall) → %s (rows=%d)", csv_path, len(df))
    return df


def load_gene_from_guides_by_model(output_dir: Path) -> pd.DataFrame:
    """
    Load per-model gene-level summary aggregated from guides.
    """
    log = _ensure_logger()
    guide_dir = output_dir / "DepMap_GuideAnalysis"
    csv_path = (
        guide_dir
        / "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels_byModel.csv"
    )
    if not csv_path.exists():
        log.warning(
            "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels_byModel.csv not found at: %s; "
            "skipping per-model gene-from-guides volcano plots.",
            csv_path,
        )
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    log.info(
        "Loaded gene-from-guides (by model) → %s (rows=%d)", csv_path, len(df)
    )
    return df


# ---- 3a) Global histogram + KDE-style curve -------------------------


def plot_guides_lfc_distribution(output_dir: Path, df_guides: pd.DataFrame) -> None:
    """
    Histogram + smoothed curve for GuideLFC across all guides,
    with an annotation box reporting key summary statistics.

    This is conceptually similar to the "Mean Error Distribution" figure.
    """
    log = _ensure_logger()

    if df_guides.empty:
        log.warning("No guide rows for LFC distribution; skipping.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    series = df_guides["GuideLFC"].dropna()
    if series.empty:
        log.warning("GuideLFC column is empty; skipping histogram.")
        return

    desc = series.describe(percentiles=[0.25, 0.5, 0.75])
    data_min = float(desc["min"])
    data_max = float(desc["max"])
    mean = float(desc["mean"])
    q1 = float(desc["25%"])
    median = float(desc["50%"])
    q3 = float(desc["75%"])

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=60,
            marker_color="rgba(31, 119, 180, 0.6)",
            name="Guide LFC",
        )
    )

    # Smooth curve using a simple Gaussian approximation
    xs = np.linspace(series.min(), series.max(), 200)
    std = series.std(ddof=1)
    if std > 0:
        pdf = (
            1.0
            / (std * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((xs - mean) / std) ** 2)
        )
        # scale PDF to histogram counts range
        pdf_scaled = pdf * (len(series) * (xs[1] - xs[0]))
    else:
        pdf_scaled = np.zeros_like(xs)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=pdf_scaled,
            mode="lines",
            name="Approx. density",
        )
    )

    # vertical line at mean
    fig.add_trace(
        go.Scatter(
            x=[mean, mean],
            y=[0, pdf_scaled.max() if len(pdf_scaled) else 0],
            mode="lines",
            line=dict(dash="dash"),
            name=f"Mean LFC: {mean:.3f}",
        )
    )

    stats_text = (
        f"Data range: {data_min:.3f} – {data_max:.3f}<br>"
        f"Mean: {mean:.3f}<br>"
        f"Q1 (25%): {q1:.3f}<br>"
        f"Median (50%): {median:.3f}<br>"
        f"Q3 (75%): {q3:.3f}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.95,
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        text=stats_text,
        showarrow=False,
    )

    fig.update_layout(
        template="plotly_white",
        title="Guide-level LFC distribution",
        xaxis_title="Guide log-fold change (GuideLFC)",
        yaxis_title="Frequency",
        bargap=0.05,
        margin=dict(l=80, r=80, t=80, b=80),
    )

    _save_plot(
        fig,
        html_path=html_dir / "Guides_LFC_distribution.html",
        png_path=figs_dir / "Guides_LFC_distribution.png",
        width=1200,
        height=700,
    )


# ---- 3b) ECDF of GuideLFC by GuideDirection -------------------------


def _compute_ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple empirical CDF (for plotting cumulative probability curves)."""
    values = np.sort(values)
    n = len(values)
    y = np.arange(1, n + 1) / n
    return values, y


def plot_guides_ecdf_by_direction(output_dir: Path, df_guides: pd.DataFrame) -> None:
    """
    ECDF curves of GuideLFC, stratified by GuideDirection
    (Neutral / Depleted / Enriched).
    """
    log = _ensure_logger()

    if df_guides.empty:
        log.warning("No guide rows for ECDF plot; skipping.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    fig = go.Figure()

    directions = (
        df_guides["GuideDirection"]
        .fillna("Unknown")
        .astype(str)
        .unique()
        .tolist()
    )

    for direction in directions:
        sub = df_guides[
            df_guides["GuideDirection"].astype(str).fillna("Unknown") == direction
        ]
        vals = sub["GuideLFC"].dropna().values
        if len(vals) == 0:
            continue
        xs, ys = _compute_ecdf(vals)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=direction,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="GuideLFC cumulative distribution by GuideDirection",
        xaxis_title="Guide log-fold change (GuideLFC)",
        yaxis_title="Cumulative probability",
        margin=dict(l=80, r=80, t=80, b=80),
        legend_title_text="GuideDirection",
    )

    _save_plot(
        fig,
        html_path=html_dir / "Guides_LFC_ECDF_byDirection.html",
        png_path=figs_dir / "Guides_LFC_ECDF_byDirection.png",
        width=900,
        height=600,
    )


# ---- 3c) Bar: counts of guide directions ----------------------------


def plot_guides_direction_counts(output_dir: Path, df_guides: pd.DataFrame) -> None:
    """
    Simple bar chart: counts of guides classified as
    Depleted / Enriched / Neutral / NA.
    """
    log = _ensure_logger()

    if df_guides.empty:
        log.warning("No guide rows for direction counts; skipping.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    dir_counts = df_guides["GuideDirection"].value_counts(dropna=False).reset_index()
    dir_counts.columns = ["GuideDirection", "count"]

    fig = px.bar(
        dir_counts,
        x="GuideDirection",
        y="count",
        text="count",
        title="Guide-level classification counts (Depleted / Enriched / Neutral)",
        labels={"GuideDirection": "Guide direction", "count": "# guides"},
        color="GuideDirection",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=80),
        showlegend=False,
    )

    _save_plot(
        fig,
        html_path=html_dir / "Guides_DirectionCounts_bar.html",
        png_path=figs_dir / "Guides_DirectionCounts_bar.png",
        width=800,
        height=450,
    )


# ---- 3d) Histogram coloured by GuideDirection -----------------------


def plot_guides_lfc_hist_by_direction(output_dir: Path, df_guides: pd.DataFrame) -> None:
    """
    Histogram of GuideLFC coloured by GuideDirection, overlayed.
    """
    log = _ensure_logger()

    if df_guides.empty:
        log.warning("No guide rows for coloured LFC histogram; skipping.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    fig = px.histogram(
        df_guides,
        x="GuideLFC",
        color="GuideDirection",
        nbins=60,
        barmode="overlay",
        title="Guide-level LFC distribution (depletion vs enrichment)",
        labels={"GuideLFC": "Guide log-fold change (GuideLFC)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=80, r=80, t=80, b=80),
    )

    _save_plot(
        fig,
        html_path=html_dir / "Guides_LFC_hist_byDirection.html",
        png_path=figs_dir / "Guides_LFC_hist_byDirection.png",
        width=1200,
        height=500,
    )


# ---- 3e) Per-gene ranked guide plot (guides present & coloured) -----


def plot_gene_guide_rank(output_dir: Path, df_guides: pd.DataFrame, gene: str) -> None:
    """
    For a single gene, show each sgRNA as a vertical bar ranked by GuideLFC,
    coloured by GuideDirection.

    This echoes "relative representation" style plots where all
    guides are ordered and specific categories are highlighted.
    """
    log = _ensure_logger()

    if df_guides.empty:
        log.warning("Guide-level table is empty; skipping gene_guide_rank for %s.", gene)
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    sub = df_guides[df_guides["Gene"].astype(str) == str(gene)].copy()
    if sub.empty:
        log.warning("No guide rows found for gene=%s; skipping gene_guide_rank.", gene)
        return

    # aggregate per sgRNA across screens/models (mean LFC; majority GuideDirection)
    agg = (
        sub.groupby("sgRNA", as_index=False)
        .agg(
            GuideLFC_mean=("GuideLFC", "mean"),
            GuideLFC_median=("GuideLFC", "median"),
        )
        .merge(
            sub.groupby("sgRNA")["GuideDirection"]
            .agg(lambda x: x.mode().iat[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index(),
            on="sgRNA",
            how="left",
        )
    )

    # order by median LFC (strongest depletion to the left)
    agg = agg.sort_values("GuideLFC_median", ascending=True).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    fig = px.bar(
        agg,
        x="rank",
        y="GuideLFC_median",
        color="GuideDirection",
        hover_data=[
            "sgRNA",
            "GuideLFC_mean",
            "GuideLFC_median",
            "GuideDirection",
        ],
        labels={
            "rank": "sgRNAs (ranked by median GuideLFC)",
            "GuideLFC_median": "Median GuideLFC",
        },
        title=f"{gene}: ranked guide-level effects (per sgRNA)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=80, r=40, t=80, b=80),
        xaxis=dict(showgrid=False),
    )

    _save_plot(
        fig,
        html_path=html_dir / f"{gene}_GuideRank_bar.html",
        png_path=figs_dir / f"{gene}_GuideRank_bar.png",
        width=max(900, 15 * len(agg)),
        height=500,
    )


# ---- 3f) Gene-from-guides "volcano" (overall) -----------------------


def plot_gene_from_guides_volcano_overall(output_dir: Path, df_gene: pd.DataFrame) -> None:
    """
    Gene-level summary from guides (across all selected models):
    mean_LFC vs fraction of depleted guides, coloured by DepletionClass.

    Expects columns:
        Gene, n_guides, mean_LFC, frac_depleted, frac_enriched,
        n_models_with_guides, DepletionClass
    """
    log = _ensure_logger()

    if df_gene.empty:
        log.warning("No rows in gene-from-guides overall table; skipping volcano.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    required_cols = {"Gene", "mean_LFC", "frac_depleted", "n_guides"}
    missing = required_cols.difference(df_gene.columns)
    if missing:
        log.warning(
            "Gene-from-guides overall table missing columns: %s; skipping volcano.",
            ",".join(sorted(missing)),
        )
        return

    # If DepletionClass not present, create a simple rule
    if "DepletionClass" not in df_gene.columns:

        def _classify(row):
            if (row["mean_LFC"] <= FC_DEP_THRESH) and (row["frac_depleted"] >= 0.5):
                return "Strongly depleted"
            elif (row["mean_LFC"] <= -0.5) and (row["frac_depleted"] >= 0.25):
                return "Moderately depleted"
            else:
                return "Other / weak"

        df_gene = df_gene.copy()
        df_gene["DepletionClass"] = df_gene.apply(_classify, axis=1)

    fig = px.scatter(
        df_gene,
        x="mean_LFC",
        y="frac_depleted",
        size="n_guides",
        color="DepletionClass",
        hover_data=[
            "Gene",
            "n_guides",
            "frac_enriched" if "frac_enriched" in df_gene.columns else None,
            "n_models_with_guides"
            if "n_models_with_guides" in df_gene.columns
            else None,
        ],
        title=(
            "Gene-level summary from guides (all selected models): "
            "mean LFC vs fraction of depleted sgRNAs"
        ),
        labels={
            "mean_LFC": "Mean guide LFC per gene",
            "frac_depleted": "Fraction of depleted guides",
            "DepletionClass": "Depletion class",
        },
        color_discrete_sequence=[
            "#1f77b4",  # blue
            "#2ca02c",  # green
            "#d62728",  # red
        ],
    )

    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=0.4, color="black"),
        )
    )
    fig.update_layout(
        height=550,
        template="plotly_white",
        legend_title_text="Depletion class",
    )

    _save_plot(
        fig,
        html_path=html_dir / "GeneFromGuides_volcano_overall.html",
        png_path=figs_dir / "GeneFromGuides_volcano_overall.png",
        width=1100,
        height=550,
    )


# ---- 3g) Gene-from-guides "volcano" (per model) ---------------------


def plot_gene_from_guides_volcano_by_model(output_dir: Path, df_gene_by_model: pd.DataFrame) -> None:
    """
    Per-model gene-level summary from guides:
    for each ModelID, mean_LFC vs frac_depleted, coloured by DepletionClass.

    Expects columns:
        ModelID, Gene, n_guides, mean_LFC, frac_depleted, frac_enriched,
        DepletionClass
    """
    log = _ensure_logger()

    if df_gene_by_model.empty:
        log.warning("No rows in gene-from-guides by-model table; skipping.")
        return

    base_dir = output_dir / "DepMap_GuideAnalysis"
    figs_dir, html_dir = _ensure_output_dirs(base_dir)

    required_cols = {"ModelID", "Gene", "mean_LFC", "frac_depleted", "n_guides"}
    missing = required_cols.difference(df_gene_by_model.columns)
    if missing:
        log.warning(
            "Gene-from-guides by-model table missing columns: %s; skipping per-model volcano plots.",
            ",".join(sorted(missing)),
        )
        return

    if "DepletionClass" not in df_gene_by_model.columns:

        def _classify(row):
            if (row["mean_LFC"] <= FC_DEP_THRESH) and (row["frac_depleted"] >= 0.5):
                return "Strongly depleted"
            elif (row["mean_LFC"] <= -0.5) and (row["frac_depleted"] >= 0.25):
                return "Moderately depleted"
            else:
                return "Other / weak"

        df_gene_by_model = df_gene_by_model.copy()
        df_gene_by_model["DepletionClass"] = df_gene_by_model.apply(_classify, axis=1)

    for model_id, df_sub in df_gene_by_model.groupby("ModelID"):
        if df_sub.empty:
            continue

        title = (
            "Gene-level summary from guides: "
            f"{model_id} – mean LFC vs fraction of depleted sgRNAs"
        )

        fig = px.scatter(
            df_sub,
            x="mean_LFC",
            y="frac_depleted",
            size="n_guides",
            color="DepletionClass",
            hover_data=[
                "Gene",
                "n_guides",
                "frac_enriched" if "frac_enriched" in df_sub.columns else None,
            ],
            title=title,
            labels={
                "mean_LFC": "Mean guide LFC per gene",
                "frac_depleted": "Fraction of depleted guides",
                "DepletionClass": "Depletion class",
            },
            color_discrete_sequence=[
                "#1f77b4",
                "#2ca02c",
                "#d62728",
            ],
        )

        fig.update_traces(
            marker=dict(opacity=0.85, line=dict(width=0.4, color="black"))
        )
        fig.update_layout(height=550, template="plotly_white")

        png_path = figs_dir / f"GeneFromGuides_volcano_{model_id}.png"
        html_path = html_dir / f"GeneFromGuides_volcano_{model_id}.html"
        _save_plot(
            fig,
            html_path=html_path,
            png_path=png_path,
            width=1100,
            height=550,
        )


def run_guides_visualisations(
    output_dir: Path,
    charts: Iterable[str] | None = None,
    gene: Optional[str] = None,
) -> None:
    """
    Orchestrate guide-level visualisations.

    charts: names from:
        {
          'lfc_hist',                 # global GuideLFC histogram + density curve + stats box
          'lfc_ecdf',                 # ECDF curves by GuideDirection
          'direction_bar',            # bar: counts of GuideDirection
          'lfc_hist_by_direction',    # histogram coloured by GuideDirection
          'gene_guide_rank',          # per-gene ranked sgRNA plot
          'gene_from_guides_volcano', # overall mean_LFC vs frac_depleted
          'gene_from_guides_volcano_by_model',  # per-model volcano
          'all'
        }
    """
    log = _ensure_logger()

    if charts is None:
        charts = ["all"]
    charts = [c.strip().lower() for c in charts]

    if "all" in charts:
        charts = [
            "lfc_hist",
            "lfc_ecdf",
            "direction_bar",
            "lfc_hist_by_direction",
            "gene_guide_rank",
            "gene_from_guides_volcano",
            "gene_from_guides_volcano_by_model",
        ]

    log.info("Guide visualisations requested: charts=%s, gene=%s", charts, gene)

    df_guides = load_guides_long(output_dir)
    if df_guides.empty:
        log.warning("Guide-level table missing/empty; skipping all guide plots.")
        return

    if "lfc_hist" in charts:
        plot_guides_lfc_distribution(output_dir=output_dir, df_guides=df_guides)

    if "lfc_ecdf" in charts:
        plot_guides_ecdf_by_direction(output_dir=output_dir, df_guides=df_guides)

    if "direction_bar" in charts:
        plot_guides_direction_counts(output_dir=output_dir, df_guides=df_guides)

    if "lfc_hist_by_direction" in charts:
        plot_guides_lfc_hist_by_direction(output_dir=output_dir, df_guides=df_guides)

    if "gene_guide_rank" in charts:
        if not gene:
            log.warning("gene_guide_rank requested but no --gene provided; skipping.")
        else:
            plot_gene_guide_rank(output_dir=output_dir, df_guides=df_guides, gene=gene)

    if "gene_from_guides_volcano" in charts:
        df_gene_overall = load_gene_from_guides_overall(output_dir=output_dir)
        if not df_gene_overall.empty:
            plot_gene_from_guides_volcano_overall(output_dir=output_dir, df_gene=df_gene_overall)

    if "gene_from_guides_volcano_by_model" in charts:
        df_gene_by_model = load_gene_from_guides_by_model(output_dir=output_dir)
        if not df_gene_by_model.empty:
            plot_gene_from_guides_volcano_by_model(output_dir=output_dir, df_gene_by_model=df_gene_by_model)


# =====================================================================
# CLI entry point
# =====================================================================


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DepMap visualisation utilities")

    parser.add_argument(
        "--target",
        choices=["all", "selected_models", "dependencies", "guides"],
        default="all",
        help=(
            "Which visualisation group to run:\n"
            "  all             – run all three groups (default)\n"
            "  selected_models – plots for DepMap_CellLines/Selected_Models.csv\n"
            "  dependencies    – plots for DepMap_Dependencies/*\n"
            "  guides          – plots for DepMap_GuideAnalysis/*"
        ),
    )
    parser.add_argument(
        "--charts",
        type=str,
        default="all",
        help=(
            "Comma-separated list of charts for the target.\n"
            "For 'selected_models':\n"
            "  disease_bar,lineage_bar,bubble,sunburst,all\n"
            "For 'dependencies':\n"
            "  overview_scatter,biotag_bar,per_gene,per_gene_heatmap,"
            "top_genes_heatmap,top_dependents,chronos_volcano,all\n"
            "For 'guides':\n"
            "  lfc_hist,lfc_ecdf,direction_bar,lfc_hist_by_direction,"
            "gene_guide_rank,gene_from_guides_volcano,"
            "gene_from_guides_volcano_by_model,all"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help=(
            "For dependencies overview / top_genes_heatmap: how many genes to "
            "highlight together (default: 30). Per-gene plots will automatically "
            "use the top 10 genes when --gene is not provided."
        ),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="median_effect_abs",
        help=(
            "For dependencies overview / top_genes_heatmap and per-gene ranking: "
            "column used to rank genes (e.g. median_effect_abs, n_prob50, n_strong_lt_1)."
        ),
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=None,
        help="For per-gene dependencies / guide-level plots: gene symbol (e.g. ASB4).",
    )

    return parser.parse_args(argv)


def run_visualization(output_dir: Path, charts: Iterable[str] | str = "all", top_n: int = 30, sort_by: str = "median_effect_abs", gene: Optional[str] = None) -> None:
    """
    Run the visualization pipeline for the given charts.

    charts: Iterable[str] | str = "all"
    top_n: int = 30
    sort_by: str = "median_effect_abs"
    gene: Optional[str] = None

    If charts == "all", we run all charts for each group.
    If charts is a list of strings, we run only the charts in the list.
    """

    # If target == all, we ignore --charts per-group and just run each group
    # with its own internal 'all' logic.
    if charts == "all":
        run_selected_models_visualisations(output_dir=output_dir, charts=None)
        run_dependencies_visualisations(output_dir=output_dir, charts=None, top_n=top_n, sort_by=sort_by, gene=gene)
        run_guides_visualisations(output_dir=output_dir, charts=None, gene=gene)
    elif charts == "selected_models":
        run_selected_models_visualisations(output_dir=output_dir, charts=charts)
    elif charts == "dependencies":
        run_dependencies_visualisations(output_dir=output_dir, charts=charts, top_n=top_n, sort_by=sort_by, gene=gene)
    elif charts == "guides":
        run_guides_visualisations(output_dir=output_dir, charts=charts, gene=gene)
    else:
        raise ValueError(f"Unsupported charts: {charts}")
