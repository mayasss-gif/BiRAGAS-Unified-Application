#!/usr/bin/env python3
"""
Full dataset summary: DEGs, pathways, regulators, and visual summaries.

This script is the "do everything" view for a single mdp_pipeline_3 dataset
(or many diseases at once).

It assumes mdp_pipeline_3 has already produced its usual outputs inside each
disease folder, including:

- degs_from_counts.csv
- combined_counts.csv
- core_enrich_up.csv / core_enrich_down.csv
- gsea_prerank.tsv
- tf_enrich_up.csv / tf_enrich_down.csv
- ulm_collectri_tf_scores.tsv
- viper_tf_scores.tsv (optional)
- immune_enrich_up.csv / immune_enrich_down.csv
- epigenetic_enrich_up.csv / epigenetic_enrich_down.csv
- metabolite_enrich_up.csv / metabolite_enrich_down.csv

For each disease, it produces:

TABLES
------
- DEG summary table (top unusual genes).
- Core pathway tables (up / down).
- GSEA summary table (major disrupted pathways).
- TF enrichment tables (up / down).
- CollectRI TF activity table.
- Immune / epigenetic / metabolite enrichment tables (up / down).

FIGURES
-------
- Volcano plot (DEGs).
- MA plot (DEGs).
- Heatmap of top DEGs across samples.
- Core pathway barplots (up/down).
- GSEA NES vs -log10(FDR) panel.
- CollectRI TF activity barplot.
- VIPER TF activity heatmap (if available).
- Immune / epigenetic / metabolite barplots (up/down).

All outputs are saved under:
    <root_dir>/agentic_analysis/full_dataset_summary/<disease>/

Everything is written so it can later be wrapped as OpenAI function tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse



PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def list_disease_folders(
    root_dir: PathLike,
    exclude: Sequence[str] = ("baseline_consensus", "comparison", "results", "agentic_analysis"),
) -> List[str]:
    """
    List disease subfolders in a given root directory.

    A disease folder is defined as any subdirectory that is not in `exclude`.
    """
    root = Path(root_dir)
    diseases: List[str] = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in exclude:
            diseases.append(entry.name)
    return sorted(diseases)


def get_disease_dir(root_dir: PathLike, disease: str) -> Path:
    """
    Locate a disease's output folder.
    """
    root = Path(root_dir)
    disease_dir = root / disease
    if not disease_dir.exists():
        raise FileNotFoundError(f"Disease folder not found: {disease_dir}")
    return disease_dir


def _load_enrich_table(path: Path) -> pd.DataFrame:
    """
    Load a standard enrichment table if it exists.

    Expected columns:
        [library, term, pval, qval, odds_ratio, combined_score, genes]
    """
    if not path.exists():
        raise FileNotFoundError(f"Enrichment file not found: {path}")
    df = pd.read_csv(path)
    expected = {"library", "term", "pval", "qval", "odds_ratio", "combined_score", "genes"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing expected columns: {missing}")
    return df


def _add_minus_log10_qval(df: pd.DataFrame, qcol: str = "qval") -> pd.DataFrame:
    """
    Add -log10(qval) column, avoiding log10(0) issues.
    """
    df = df.copy()
    q = df[qcol].replace(0, np.nan)
    min_non_zero = q[q > 0].min()
    if pd.isna(min_non_zero):
        min_non_zero = 1e-300
    df["-log10_qval"] = -np.log10(df[qcol].replace(0, min_non_zero))
    return df


def _barplot_enrichment(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    label_col: str = "term",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Simple horizontal barplot for enrichment results.
    """
    if df.empty:
        raise ValueError("No rows to plot in enrichment barplot.")

    fig, ax = plt.subplots(figsize=figsize)
    y = df[label_col]
    x = df[value_col]
    ax.barh(y, x)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_ylabel("Term")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DEG-level summaries and plots
# ---------------------------------------------------------------------------

def make_deg_summary_table(
    root_dir: PathLike,
    disease: str,
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
    base_mean_col: str = "baseMean",
    lfc_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Build a summary DEG table with top up- and down-regulated genes.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    df = pd.read_csv(deg_path)

    # Drop rows without log2FC
    df = df.dropna(subset=[lfc_col]).copy()

    df["direction"] = np.where(df[lfc_col] > 0, "UP", "DOWN")

    sig = df.copy()
    if padj_col in sig.columns:
        sig = sig.dropna(subset=[padj_col])
        sig = sig[(sig[padj_col] <= padj_cutoff) & (sig[lfc_col].abs() >= lfc_cutoff)]
        pcol_to_use = padj_col
    else:
        sig = sig.dropna(subset=["pvalue"])
        sig = sig[(sig["pvalue"] <= padj_cutoff) & (sig[lfc_col].abs() >= lfc_cutoff)]
        pcol_to_use = "pvalue"

    up = sig[sig["direction"] == "UP"].sort_values(lfc_col, ascending=False)
    down = sig[sig["direction"] == "DOWN"].sort_values(lfc_col, ascending=True)

    n_each = max(top_n // 2, 1)
    top_up = up.head(n_each)
    top_down = down.head(n_each)

    out = pd.concat([top_up, top_down], axis=0)
    out = out[["Gene", base_mean_col, lfc_col, pcol_to_use, "direction"]]

    return out.sort_values(lfc_col, ascending=False)


def plot_deg_volcano(
    root_dir: PathLike,
    disease: str,
    lfc_col: str = "log2FoldChange",
    p_col: str = "padj",
    lfc_thresh: float = 1.0,
    p_thresh: float = 0.05,
    top_n_labels: int = 15,
) -> plt.Figure:
    """
    Create a volcano plot for one disease.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    df = pd.read_csv(deg_path)

    if p_col not in df.columns:
        p_col = "pvalue"

    df = df.dropna(subset=[lfc_col, p_col]).copy()
    min_non_zero = df.loc[df[p_col] > 0, p_col].min()
    df["neg_log10_p"] = -np.log10(df[p_col].replace(0, min_non_zero))

    sig_mask = (df[p_col] <= p_thresh) & (df[lfc_col].abs() >= lfc_thresh)
    df["category"] = "Not significant"
    df.loc[sig_mask & (df[lfc_col] > 0), "category"] = "Up"
    df.loc[sig_mask & (df[lfc_col] < 0), "category"] = "Down"

    fig, ax = plt.subplots(figsize=(8, 6))

    for cat in ["Not significant", "Up", "Down"]:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub[lfc_col],
            sub["neg_log10_p"],
            s=10,
            alpha=0.6,
            label=cat,
        )

    ax.axvline(lfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, linestyle="--", linewidth=1)
    ax.axhline(-np.log10(p_thresh), linestyle="--", linewidth=1)

    df["rank_score"] = df[lfc_col].abs() * df["neg_log10_p"]
    top = df.sort_values("rank_score", ascending=False).head(top_n_labels)
    for _, row in top.iterrows():
        ax.text(
            row[lfc_col],
            row["neg_log10_p"],
            row["Gene"],
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_title(f"{disease}: Differential expression (volcano)")
    ax.set_xlabel("log2 fold-change")
    ax.set_ylabel("-log10 p-value")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_deg_ma(
    root_dir: PathLike,
    disease: str,
    lfc_col: str = "log2FoldChange",
    base_mean_col: str = "baseMean",
    padj_col: str = "padj",
    padj_thresh: float = 0.05,
) -> plt.Figure:
    """
    Create an MA plot for one disease.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    df = pd.read_csv(deg_path)

    df = df.dropna(subset=[base_mean_col, lfc_col]).copy()
    df["log10_baseMean"] = np.log10(df[base_mean_col] + 1.0)

    if padj_col in df.columns:
        sig = df[padj_col] <= padj_thresh
    else:
        sig = df["pvalue"] <= padj_thresh

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df.loc[~sig, "log10_baseMean"],
        df.loc[~sig, lfc_col],
        s=10,
        alpha=0.5,
        label="Not significant",
    )
    ax.scatter(
        df.loc[sig, "log10_baseMean"],
        df.loc[sig, lfc_col],
        s=10,
        alpha=0.7,
        label="Significant",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(f"{disease}: MA-plot (mean expression vs log2FC)")
    ax.set_xlabel("log10(baseMean + 1)")
    ax.set_ylabel("log2 fold-change")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_top_degs_heatmap(
    root_dir: PathLike,
    disease: str,
    top_n: int = 50,
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
    lfc_cutoff: float = 1.0,
    padj_cutoff: float = 0.05,
) -> plt.Figure:
    """
    Heatmap of expression for top DEGs across all samples.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    counts_path = disease_dir / "combined_counts.csv"

    deg = pd.read_csv(deg_path)
    counts = pd.read_csv(counts_path)

    if padj_col in deg.columns:
        deg_f = deg.dropna(subset=[lfc_col, padj_col]).copy()
        deg_f = deg_f[(deg_f[padj_col] <= padj_cutoff) & (deg_f[lfc_col].abs() >= lfc_cutoff)]
    else:
        deg_f = deg.dropna(subset=[lfc_col, "pvalue"]).copy()
        deg_f = deg_f[(deg_f["pvalue"] <= padj_cutoff) & (deg_f[lfc_col].abs() >= lfc_cutoff)]

    deg_f["abs_lfc"] = deg_f[lfc_col].abs()
    top = deg_f.sort_values("abs_lfc", ascending=False).head(top_n)

    genes = top["Gene"].unique()
    expr = counts[counts["Gene"].isin(genes)].set_index("Gene")

    if expr.empty:
        raise ValueError(f"No expression rows found for selected DEGs in {disease}.")

    expr_z = expr.apply(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, max(4, expr_z.shape[0] * 0.12)))
    im = ax.imshow(expr_z.values, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(expr_z.shape[0]))
    ax.set_yticklabels(expr_z.index, fontsize=6)
    ax.set_xticks(np.arange(expr_z.shape[1]))
    ax.set_xticklabels(expr_z.columns, rotation=90, fontsize=6)

    ax.set_title(f"{disease}: Top {expr_z.shape[0]} DEGs (z-scored expression)")
    fig.colorbar(im, ax=ax, label="Z-score per gene")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pathway-level summaries (core enrichment + GSEA)
# ---------------------------------------------------------------------------

def make_core_pathway_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Build summary tables for core pathway enrichment (up/down).
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "core_enrich_up.csv"
    down_path = disease_dir / "core_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        up_df = _add_minus_log10_qval(up_df)
        up_df = up_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["up"] = up_df
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        down_df = _add_minus_log10_qval(down_df)
        down_df = down_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["down"] = down_df
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def plot_core_pathways_barplot(
    root_dir: PathLike,
    disease: str,
    direction: str = "up",
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of core pathways for one direction (up or down).
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "core_enrich_up.csv"
        label = "Up-regulated core pathways"
    else:
        path = disease_dir / "core_enrich_down.csv"
        label = "Down-regulated core pathways"

    df = _load_enrich_table(path)
    df = _add_minus_log10_qval(df)
    df = df.sort_values("combined_score", ascending=False).head(top_n)
    title = f"{disease}: {label} (top {len(df)})"
    return _barplot_enrichment(df, title=title, value_col="-log10_qval")


def make_gsea_summary_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Summarize GSEA prerank results for one disease.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    gsea_path = disease_dir / "gsea_prerank.tsv"
    if not gsea_path.exists():
        raise FileNotFoundError(f"GSEA prerank file not found for {disease}: {gsea_path}")

    df = pd.read_csv(gsea_path, sep="\t")
    needed = {"term", "NES", "FDR q-val"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"GSEA file {gsea_path} missing columns: {missing}")

    df = df.copy()
    df["direction"] = np.where(df["NES"] > 0, "UP", "DOWN")

    q = df["FDR q-val"].replace(0, np.nan)
    min_non_zero = q[q > 0].min()
    if pd.isna(min_non_zero):
        min_non_zero = 1e-300
    df["-log10_FDR"] = -np.log10(df["FDR q-val"].replace(0, min_non_zero))

    df["impact_score"] = df["NES"] * df["-log10_FDR"]
    df["abs_impact"] = df["impact_score"].abs()
    df = df.sort_values("abs_impact", ascending=False).head(top_n)

    return df[["term", "NES", "FDR q-val", "-log10_FDR", "direction", "impact_score"]]


def plot_gsea_nes_scatter(
    gsea_df: pd.DataFrame,
    disease: str,
) -> plt.Figure:
    """
    Scatter plot of NES vs -log10(FDR) for GSEA pathways.
    """
    if gsea_df.empty:
        raise ValueError("GSEA summary table is empty.")

    fig, ax = plt.subplots(figsize=(8, 6))

    for direction in ["UP", "DOWN"]:
        sub = gsea_df[gsea_df["direction"] == direction]
        if sub.empty:
            continue
        ax.scatter(
            sub["NES"],
            sub["-log10_FDR"],
            s=30,
            alpha=0.7,
            label=direction,
        )

    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("NES")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"{disease}: GSEA disrupted pathways (NES vs -log10(FDR))")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regulator-level summaries (TF enrichment, CollectRI, VIPER)
# ---------------------------------------------------------------------------

def make_tf_enrich_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Summarize TF enrichment (up/down) for one disease.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "tf_enrich_up.csv"
    down_path = disease_dir / "tf_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        up_df = _add_minus_log10_qval(up_df)
        up_df = up_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["up"] = up_df
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        down_df = _add_minus_log10_qval(down_df)
        down_df = down_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["down"] = down_df
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def plot_tf_enrich_barplot(
    root_dir: PathLike,
    disease: str,
    direction: str = "up",
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of TF enrichment terms for one direction (up or down).
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "tf_enrich_up.csv"
        label = "TF enrichment (UP genes)"
    else:
        path = disease_dir / "tf_enrich_down.csv"
        label = "TF enrichment (DOWN genes)"

    df = _load_enrich_table(path)
    df = _add_minus_log10_qval(df)
    df = df.sort_values("combined_score", ascending=False).head(top_n)
    title = f"{disease}: {label} (top {len(df)})"
    return _barplot_enrichment(df, title=title, value_col="-log10_qval")


def make_collectri_tf_activity_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Summarize TF activity from ULM CollectRI scores.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    tf_path = disease_dir / "ulm_collectri_tf_scores.tsv"
    if not tf_path.exists():
        raise FileNotFoundError(f"CollectRI TF scores not found for {disease}: {tf_path}")

    df = pd.read_csv(tf_path, sep="\t")
    row = df.iloc[0].drop(labels=["Unnamed: 0"], errors="ignore")

    scores = row.astype(float)
    tbl = scores.to_frame(name="score")
    tbl["TF"] = tbl.index
    tbl["abs_score"] = tbl["score"].abs()

    return tbl.sort_values("abs_score", ascending=False).head(top_n)[["TF", "score"]]


def plot_collectri_tf_activity_barplot(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of top TF activities from CollectRI ULM scores.
    """
    tbl = make_collectri_tf_activity_table(root_dir=root_dir, disease=disease, top_n=top_n)
    if tbl.empty:
        raise ValueError(f"No CollectRI TF activity rows for {disease}.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(tbl["TF"], tbl["score"])
    ax.invert_yaxis()
    ax.set_xlabel("ULM TF activity score")
    ax.set_ylabel("TF")
    ax.set_title(f"{disease}: CollectRI TF activity (top {len(tbl)})")
    fig.tight_layout()
    return fig


def plot_viper_tf_heatmap(
    root_dir: PathLike,
    disease: str,
    top_n_tfs: int = 50,
) -> plt.Figure:
    """
    Heatmap of sample-specific TF activities from VIPER.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    viper_path = disease_dir / "viper_tf_scores.tsv"
    if not viper_path.exists():
        raise FileNotFoundError(f"VIPER TF scores not found for {disease}: {viper_path}")

    df = pd.read_csv(viper_path, sep="\t")
    if "sample" not in df.columns:
        raise ValueError(f"VIPER file {viper_path} missing 'sample' column.")

    df = df.set_index("sample")
    if df.empty:
        raise ValueError(f"Empty VIPER TF matrix for {disease}.")

    var = df.var(axis=0)
    top_tfs = var.sort_values(ascending=False).head(top_n_tfs).index
    sub = df[top_tfs]

    sub_z = sub.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8), axis=0)

    fig, ax = plt.subplots(figsize=(10, max(4, sub_z.shape[1] * 0.12)))
    im = ax.imshow(sub_z.values.T, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(sub_z.shape[1]))
    ax.set_yticklabels(sub_z.columns, fontsize=6)
    ax.set_xticks(np.arange(sub_z.shape[0]))
    ax.set_xticklabels(sub_z.index, rotation=90, fontsize=6)

    ax.set_title(f"{disease}: VIPER TF activity (top {sub_z.shape[1]} TFs)")
    fig.colorbar(im, ax=ax, label="Z-score per TF")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Immune, epigenetic, metabolite summaries
# ---------------------------------------------------------------------------

def _make_enrich_up_down_tables(
    root_dir: PathLike,
    disease: str,
    base_name: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Generic helper: summarize up/down enrichment for immune/epi/metabolite.
    base_name is one of: 'immune', 'epigenetic', 'metabolite'.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / f"{base_name}_enrich_up.csv"
    down_path = disease_dir / f"{base_name}_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        up_df = _add_minus_log10_qval(up_df)
        up_df = up_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["up"] = up_df
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        down_df = _add_minus_log10_qval(down_df)
        down_df = down_df.sort_values("combined_score", ascending=False).head(top_n)
        tables["down"] = down_df
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def _plot_enrich_up_down_barplot(
    root_dir: PathLike,
    disease: str,
    base_name: str,
    direction: str,
    label_prefix: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Generic helper: barplot for immune/epi/metabolite enrichment (up or down).
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / f"{base_name}_enrich_up.csv"
        label = f"{label_prefix} (UP genes)"
    else:
        path = disease_dir / f"{base_name}_enrich_down.csv"
        label = f"{label_prefix} (DOWN genes)"

    df = _load_enrich_table(path)
    df = _add_minus_log10_qval(df)
    df = df.sort_values("combined_score", ascending=False).head(top_n)
    title = f"{disease}: {label} (top {len(df)})"
    return _barplot_enrichment(df, title=title, value_col="-log10_qval")


# ---------------------------------------------------------------------------
# Wrappers: one disease / many diseases
# ---------------------------------------------------------------------------

def analyze_full_dataset_one_disease(
    root_dir: PathLike,
    disease: str,
    top_deg_n: int = 50,
    top_pathways_n: int = 30,
    top_gsea_n: int = 50,
    top_tf_enrich_n: int = 30,
    top_tf_activity_n: int = 30,
    top_viper_tfs: int = 50,
    top_immune_n: int = 30,
    top_epigenetic_n: int = 30,
    top_metabolite_n: int = 30,
) -> Dict[str, object]:
    """
    Run the full "process raw dataset" summary for one disease.

    Returns
    -------
    dict of tables and figures.
    """
    results: Dict[str, object] = {}

    # --- DEGs ---
    deg_table = make_deg_summary_table(root_dir=root_dir, disease=disease, top_n=top_deg_n)
    results["deg_table"] = deg_table

    try:
        fig_volcano = plot_deg_volcano(root_dir=root_dir, disease=disease)
    except Exception:
        fig_volcano = None
    results["fig_volcano"] = fig_volcano

    try:
        fig_ma = plot_deg_ma(root_dir=root_dir, disease=disease)
    except Exception:
        fig_ma = None
    results["fig_ma"] = fig_ma

    try:
        fig_heatmap = plot_top_degs_heatmap(root_dir=root_dir, disease=disease, top_n=top_deg_n)
    except Exception:
        fig_heatmap = None
    results["fig_heatmap"] = fig_heatmap

    # --- Core pathways ---
    core_tables = make_core_pathway_tables(root_dir=root_dir, disease=disease, top_n=top_pathways_n)
    results["core_pathways_up"] = core_tables.get("up")
    results["core_pathways_down"] = core_tables.get("down")

    try:
        fig_core_up = plot_core_pathways_barplot(root_dir=root_dir, disease=disease,
                                                 direction="up", top_n=top_pathways_n)
    except Exception:
        fig_core_up = None
    results["fig_core_up"] = fig_core_up

    try:
        fig_core_down = plot_core_pathways_barplot(root_dir=root_dir, disease=disease,
                                                   direction="down", top_n=top_pathways_n)
    except Exception:
        fig_core_down = None
    results["fig_core_down"] = fig_core_down

    # --- GSEA ---
    try:
        gsea_table = make_gsea_summary_table(root_dir=root_dir, disease=disease, top_n=top_gsea_n)
    except FileNotFoundError:
        gsea_table = None
    results["gsea_table"] = gsea_table

    if gsea_table is not None and not gsea_table.empty:
        try:
            fig_gsea = plot_gsea_nes_scatter(gsea_table, disease=disease)
        except Exception:
            fig_gsea = None
    else:
        fig_gsea = None
    results["fig_gsea"] = fig_gsea

    # --- TF enrichment ---
    tf_tables = make_tf_enrich_tables(root_dir=root_dir, disease=disease, top_n=top_tf_enrich_n)
    results["tf_enrich_up"] = tf_tables.get("up")
    results["tf_enrich_down"] = tf_tables.get("down")

    try:
        fig_tf_up = plot_tf_enrich_barplot(root_dir=root_dir, disease=disease,
                                           direction="up", top_n=top_tf_enrich_n)
    except Exception:
        fig_tf_up = None
    results["fig_tf_up"] = fig_tf_up

    try:
        fig_tf_down = plot_tf_enrich_barplot(root_dir=root_dir, disease=disease,
                                             direction="down", top_n=top_tf_enrich_n)
    except Exception:
        fig_tf_down = None
    results["fig_tf_down"] = fig_tf_down

    # --- CollectRI TF activity ---
    try:
        collectri_table = make_collectri_tf_activity_table(
            root_dir=root_dir, disease=disease, top_n=top_tf_activity_n
        )
        fig_collectri = plot_collectri_tf_activity_barplot(
            root_dir=root_dir, disease=disease, top_n=top_tf_activity_n
        )
    except FileNotFoundError:
        collectri_table = None
        fig_collectri = None
    results["collectri_tf_table"] = collectri_table
    results["fig_collectri_tf"] = fig_collectri

    # --- VIPER ---
    try:
        fig_viper = plot_viper_tf_heatmap(root_dir=root_dir, disease=disease, top_n_tfs=top_viper_tfs)
    except FileNotFoundError:
        fig_viper = None
    except Exception:
        fig_viper = None
    results["fig_viper_tf"] = fig_viper

    # --- Immune ---
    immune_tables = _make_enrich_up_down_tables(root_dir=root_dir, disease=disease,
                                                base_name="immune", top_n=top_immune_n)
    results["immune_up"] = immune_tables.get("up")
    results["immune_down"] = immune_tables.get("down")

    try:
        fig_immune_up = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="immune", direction="up",
            label_prefix="Immune enrichment", top_n=top_immune_n,
        )
    except Exception:
        fig_immune_up = None
    results["fig_immune_up"] = fig_immune_up

    try:
        fig_immune_down = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="immune", direction="down",
            label_prefix="Immune enrichment", top_n=top_immune_n,
        )
    except Exception:
        fig_immune_down = None
    results["fig_immune_down"] = fig_immune_down

    # --- Epigenetic ---
    epi_tables = _make_enrich_up_down_tables(root_dir=root_dir, disease=disease,
                                             base_name="epigenetic", top_n=top_epigenetic_n)
    results["epigenetic_up"] = epi_tables.get("up")
    results["epigenetic_down"] = epi_tables.get("down")

    try:
        fig_epi_up = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="epigenetic", direction="up",
            label_prefix="Epigenetic enrichment", top_n=top_epigenetic_n,
        )
    except Exception:
        fig_epi_up = None
    results["fig_epigenetic_up"] = fig_epi_up

    try:
        fig_epi_down = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="epigenetic", direction="down",
            label_prefix="Epigenetic enrichment", top_n=top_epigenetic_n,
        )
    except Exception:
        fig_epi_down = None
    results["fig_epigenetic_down"] = fig_epi_down

    # --- Metabolite ---
    metab_tables = _make_enrich_up_down_tables(root_dir=root_dir, disease=disease,
                                               base_name="metabolite", top_n=top_metabolite_n)
    results["metabolite_up"] = metab_tables.get("up")
    results["metabolite_down"] = metab_tables.get("down")

    try:
        fig_metab_up = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="metabolite", direction="up",
            label_prefix="Metabolite enrichment", top_n=top_metabolite_n,
        )
    except Exception:
        fig_metab_up = None
    results["fig_metabolite_up"] = fig_metab_up

    try:
        fig_metab_down = _plot_enrich_up_down_barplot(
            root_dir=root_dir, disease=disease,
            base_name="metabolite", direction="down",
            label_prefix="Metabolite enrichment", top_n=top_metabolite_n,
        )
    except Exception:
        fig_metab_down = None
    results["fig_metabolite_down"] = fig_metab_down

    return results


def analyze_full_dataset_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, object]]:
    """
    Run the full dataset summary for multiple diseases.
    """
    root = Path(root_dir)

    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_full_dataset_one_disease(root_dir=root, disease=disease, **kwargs)
            all_results[disease] = res
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {disease}: {e}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Full dataset summary: DEGs, pathways, regulators, and visual summaries "
            "for one or many diseases from mdp_pipeline_3 outputs."
        )
    )
    parser.add_argument(
        "root_dir",
        help="Path to the mdp_pipeline_3 output root directory.",
    )
    parser.add_argument(
        "-d",
        "--diseases",
        nargs="*",
        help=(
            "Optional disease folder names. If omitted, all disease folders under "
            "root_dir (excluding 'baseline_consensus', 'comparison', 'results', "
            "'agentic_analysis') will be analyzed."
        ),
    )
    parser.add_argument(
        "--top-deg-n",
        type=int,
        default=50,
        help="Number of DEGs to summarize/plot per disease.",
    )
    parser.add_argument(
        "--top-pathways-n",
        type=int,
        default=30,
        help="Number of core pathway terms to show per direction.",
    )
    parser.add_argument(
        "--top-gsea-n",
        type=int,
        default=50,
        help="Number of GSEA pathways to summarize.",
    )
    parser.add_argument(
        "--top-tf-enrich-n",
        type=int,
        default=30,
        help="Number of TF enrichment terms to show per direction.",
    )
    parser.add_argument(
        "--top-tf-activity-n",
        type=int,
        default=30,
        help="Number of TFs to show from CollectRI activity.",
    )
    parser.add_argument(
        "--top-viper-tfs",
        type=int,
        default=50,
        help="Number of TFs to show in VIPER heatmap.",
    )
    parser.add_argument(
        "--top-immune-n",
        type=int,
        default=30,
        help="Number of immune terms to show per direction.",
    )
    parser.add_argument(
        "--top-epigenetic-n",
        type=int,
        default=30,
        help="Number of epigenetic terms to show per direction.",
    )
    parser.add_argument(
        "--top-metabolite-n",
        type=int,
        default=30,
        help="Number of metabolite terms to show per direction.",
    )

    args = parser.parse_args()

    results = analyze_full_dataset_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_deg_n=args.top_deg_n,
        top_pathways_n=args.top_pathways_n,
        top_gsea_n=args.top_gsea_n,
        top_tf_enrich_n=args.top_tf_enrich_n,
        top_tf_activity_n=args.top_tf_activity_n,
        top_viper_tfs=args.top_viper_tfs,
        top_immune_n=args.top_immune_n,
        top_epigenetic_n=args.top_epigenetic_n,
        top_metabolite_n=args.top_metabolite_n,
    )

    # Save outputs under root_dir/agentic_analysis/full_dataset_summary/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "full_dataset_summary"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- DEG table ---
        if isinstance(res.get("deg_table"), pd.DataFrame):
            res["deg_table"].to_csv(out_dir / f"{disease}_DEG_summary.csv", index=False)

        # --- Core pathways ---
        if isinstance(res.get("core_pathways_up"), pd.DataFrame):
            res["core_pathways_up"].to_csv(
                out_dir / f"{disease}_core_pathways_up.csv", index=False
            )
        if isinstance(res.get("core_pathways_down"), pd.DataFrame):
            res["core_pathways_down"].to_csv(
                out_dir / f"{disease}_core_pathways_down.csv", index=False
            )

        # --- GSEA ---
        if isinstance(res.get("gsea_table"), pd.DataFrame):
            res["gsea_table"].to_csv(
                out_dir / f"{disease}_gsea_summary.csv", index=False
            )

        # --- TF enrichment ---
        if isinstance(res.get("tf_enrich_up"), pd.DataFrame):
            res["tf_enrich_up"].to_csv(
                out_dir / f"{disease}_tf_enrich_up_summary.csv", index=False
            )
        if isinstance(res.get("tf_enrich_down"), pd.DataFrame):
            res["tf_enrich_down"].to_csv(
                out_dir / f"{disease}_tf_enrich_down_summary.csv", index=False
            )

        # --- CollectRI TF table ---
        if isinstance(res.get("collectri_tf_table"), pd.DataFrame):
            res["collectri_tf_table"].to_csv(
                out_dir / f"{disease}_collectri_tf_activity.csv", index=False
            )

        # --- Immune / epigenetic / metabolite tables ---
        if isinstance(res.get("immune_up"), pd.DataFrame):
            res["immune_up"].to_csv(
                out_dir / f"{disease}_immune_enrich_up_summary.csv", index=False
            )
        if isinstance(res.get("immune_down"), pd.DataFrame):
            res["immune_down"].to_csv(
                out_dir / f"{disease}_immune_enrich_down_summary.csv", index=False
            )

        if isinstance(res.get("epigenetic_up"), pd.DataFrame):
            res["epigenetic_up"].to_csv(
                out_dir / f"{disease}_epigenetic_enrich_up_summary.csv", index=False
            )
        if isinstance(res.get("epigenetic_down"), pd.DataFrame):
            res["epigenetic_down"].to_csv(
                out_dir / f"{disease}_epigenetic_enrich_down_summary.csv", index=False
            )

        if isinstance(res.get("metabolite_up"), pd.DataFrame):
            res["metabolite_up"].to_csv(
                out_dir / f"{disease}_metabolite_enrich_up_summary.csv", index=False
            )
        if isinstance(res.get("metabolite_down"), pd.DataFrame):
            res["metabolite_down"].to_csv(
                out_dir / f"{disease}_metabolite_enrich_down_summary.csv", index=False
            )

        # --- Figures ---
        if res.get("fig_volcano") is not None:
            res["fig_volcano"].savefig(
                out_dir / f"{disease}_volcano.png", dpi=300, bbox_inches="tight"
            )
        if res.get("fig_ma") is not None:
            res["fig_ma"].savefig(
                out_dir / f"{disease}_MA.png", dpi=300, bbox_inches="tight"
            )
        if res.get("fig_heatmap") is not None:
            res["fig_heatmap"].savefig(
                out_dir / f"{disease}_topDEG_heatmap.png", dpi=300, bbox_inches="tight"
            )

        if res.get("fig_core_up") is not None:
            res["fig_core_up"].savefig(
                out_dir / f"{disease}_core_pathways_up_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )
        if res.get("fig_core_down") is not None:
            res["fig_core_down"].savefig(
                out_dir / f"{disease}_core_pathways_down_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_gsea") is not None:
            res["fig_gsea"].savefig(
                out_dir / f"{disease}_gsea_nes_panel.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_tf_up") is not None:
            res["fig_tf_up"].savefig(
                out_dir / f"{disease}_tf_enrich_up_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )
        if res.get("fig_tf_down") is not None:
            res["fig_tf_down"].savefig(
                out_dir / f"{disease}_tf_enrich_down_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_collectri_tf") is not None:
            res["fig_collectri_tf"].savefig(
                out_dir / f"{disease}_collectri_tf_activity_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_viper_tf") is not None:
            res["fig_viper_tf"].savefig(
                out_dir / f"{disease}_viper_tf_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_immune_up") is not None:
            res["fig_immune_up"].savefig(
                out_dir / f"{disease}_immune_enrich_up_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )
        if res.get("fig_immune_down") is not None:
            res["fig_immune_down"].savefig(
                out_dir / f"{disease}_immune_enrich_down_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_epigenetic_up") is not None:
            res["fig_epigenetic_up"].savefig(
                out_dir / f"{disease}_epigenetic_enrich_up_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )
        if res.get("fig_epigenetic_down") is not None:
            res["fig_epigenetic_down"].savefig(
                out_dir / f"{disease}_epigenetic_enrich_down_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_metabolite_up") is not None:
            res["fig_metabolite_up"].savefig(
                out_dir / f"{disease}_metabolite_enrich_up_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )
        if res.get("fig_metabolite_down") is not None:
            res["fig_metabolite_down"].savefig(
                out_dir / f"{disease}_metabolite_enrich_down_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

    print(
        f"Finished full dataset summary for {len(results)} disease(s): "
        f"{', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
