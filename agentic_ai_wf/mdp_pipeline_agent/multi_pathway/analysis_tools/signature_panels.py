#!/usr/bin/env python3
"""
Signature panels: pathway, regulator, immune, epigenetic, and metabolic signatures
for one or many diseases based on mdp_pipeline_3 outputs.

Given a root output folder from mdp_pipeline_3 and one or more disease
folders inside it, this script provides small, reusable functions to:

- Summarize pathway enrichment (core_enrich_up/down + GSEA prerank).
- Summarize TF regulators (tf_enrich_up/down + CollectRI + optional VIPER).
- Summarize immune signatures (immune_enrich_up/down).
- Summarize epigenetic signatures (epigenetic_enrich_up/down).
- Summarize metabolic signatures (metabolite_enrich_up/down).

All generated outputs are saved under:
    <root_dir>/agentic_analysis/signature_panels/<disease>/

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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3 (user-provided).
    exclude : sequence of str
        Folder names to ignore (e.g. global folders).

    Returns
    -------
    list of str
        Names of disease folders.
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Name of the disease folder (e.g. "Lupus", "Addison_Disease").

    Returns
    -------
    Path
        Path to the disease-specific output directory.

    Raises
    ------
    FileNotFoundError
        If the disease directory does not exist.
    """
    root = Path(root_dir)
    disease_dir = root / disease
    if not disease_dir.exists():
        raise FileNotFoundError(f"Disease folder not found: {disease_dir}")
    return disease_dir


def _load_enrich_table(path: Path) -> pd.DataFrame:
    """
    Load a standard 7-column enrichment table if it exists.

    Parameters
    ----------
    path : Path
        Path to a CSV file with columns:
        [library, term, pval, qval, odds_ratio, combined_score, genes].

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Enrichment file not found: {path}")
    df = pd.read_csv(path)
    # Basic sanity check for expected columns
    expected = {"library", "term", "pval", "qval", "odds_ratio", "combined_score", "genes"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing expected columns: {missing}")
    return df


def _summarize_enrich_table(
    df: pd.DataFrame,
    top_n: int = 30,
    score_col: str = "combined_score",
) -> pd.DataFrame:
    """
    Summarize an enrichment table by adding -log10(qval) and taking top N.

    Parameters
    ----------
    df : DataFrame
        Enrichment results.
    top_n : int
        Number of terms to keep.
    score_col : str
        Column to sort by (typically 'combined_score').

    Returns
    -------
    DataFrame
        Subset with ['library', 'term', 'pval', 'qval', 'odds_ratio',
        'combined_score', 'genes', '-log10_qval'], top_n rows.
    """
    df = df.copy()
    # Avoid log10(0)
    q = df["qval"].replace(0, np.nan)
    min_non_zero = q[q > 0].min()
    if pd.isna(min_non_zero):
        # If all qvals are zero or NaN, set fallback
        min_non_zero = 1e-300
    df["-log10_qval"] = -np.log10(df["qval"].replace(0, min_non_zero))
    df = df.sort_values(score_col, ascending=False).head(top_n)
    return df


def _barplot_enrichment(
    df: pd.DataFrame,
    title: str,
    value_col: str = "-log10_qval",
    label_col: str = "term",
    figsize=(10, 6),
) -> plt.Figure:
    """
    Simple horizontal barplot for enrichment results.

    Parameters
    ----------
    df : DataFrame
        Enrichment results with columns label_col and value_col.
    title : str
        Plot title.
    value_col : str
        Column for bar length.
    label_col : str
        Column with term labels.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
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
# PATHWAY SIGNATURES
# ---------------------------------------------------------------------------

def make_core_pathway_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Build summary tables for core pathway enrichment (up/down).

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    top_n : int
        Number of pathways to keep per direction.

    Returns
    -------
    dict
        {
          "up": DataFrame,
          "down": DataFrame
        }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "core_enrich_up.csv"
    down_path = disease_dir / "core_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        tables["up"] = _summarize_enrich_table(up_df, top_n=top_n)
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        tables["down"] = _summarize_enrich_table(down_df, top_n=top_n)
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    direction : {"up", "down"}
        Which enrichment file to use.
    top_n : int
        Number of pathways to show.

    Returns
    -------
    Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "core_enrich_up.csv"
        label = "Up-regulated core pathways"
    else:
        path = disease_dir / "core_enrich_down.csv"
        label = "Down-regulated core pathways"

    df = _load_enrich_table(path)
    df_top = _summarize_enrich_table(df, top_n=top_n)
    title = f"{disease}: {label} (top {len(df_top)})"
    return _barplot_enrichment(df_top, title=title)


def make_gsea_summary_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Summarize GSEA prerank results for one disease.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of strongest pathways to keep (by |NES|).

    Returns
    -------
    DataFrame
        Columns: [term, NES, FDR q-val, -log10_FDR, direction, ...]
    """
    disease_dir = get_disease_dir(root_dir, disease)
    gsea_path = disease_dir / "gsea_prerank.tsv"
    if not gsea_path.exists():
        raise FileNotFoundError(f"GSEA prerank file not found for {disease}: {gsea_path}")

    df = pd.read_csv(gsea_path, sep="\t")
    # Ensure expected columns
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

    df["abs_NES"] = df["NES"].abs()
    df = df.sort_values("abs_NES", ascending=False).head(top_n)

    return df[["term", "NES", "FDR q-val", "-log10_FDR", "direction"]]


def plot_gsea_nes_panel(
    root_dir: PathLike,
    disease: str,
    top_n: int = 50,
) -> plt.Figure:
    """
    Scatter plot of NES vs -log10(FDR) for top GSEA pathways.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of strongest pathways to display.

    Returns
    -------
    Figure
    """
    df = make_gsea_summary_table(root_dir=root_dir, disease=disease, top_n=top_n)

    fig, ax = plt.subplots(figsize=(8, 6))

    for direction in ["UP", "DOWN"]:
        sub = df[df["direction"] == direction]
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
    ax.set_title(f"{disease}: GSEA NES vs -log10(FDR) (top {len(df)})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# REGULATOR SIGNATURES (TF enrichment + CollectRI + VIPER)
# ---------------------------------------------------------------------------

def make_tf_enrich_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Summarize TF enrichment (up/down) for one disease.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of TF terms to keep per direction.

    Returns
    -------
    dict
        {
          "up": DataFrame,
          "down": DataFrame
        }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "tf_enrich_up.csv"
    down_path = disease_dir / "tf_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        tables["up"] = _summarize_enrich_table(up_df, top_n=top_n)
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        tables["down"] = _summarize_enrich_table(down_df, top_n=top_n)
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    direction : {"up", "down"}
        Which enrichment file to use.
    top_n : int
        Number of terms to show.

    Returns
    -------
    Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "tf_enrich_up.csv"
        label = "TF enrichment (UP genes)"
    else:
        path = disease_dir / "tf_enrich_down.csv"
        label = "TF enrichment (DOWN genes)"

    df = _load_enrich_table(path)
    df_top = _summarize_enrich_table(df, top_n=top_n)
    title = f"{disease}: {label} (top {len(df_top)})"
    return _barplot_enrichment(df_top, title=title)


def make_collectri_tf_activity_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Summarize TF activity from ULM CollectRI scores.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of TFs to report (by |score|).

    Returns
    -------
    DataFrame
        Columns: [TF, score], sorted by |score| descending.
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of TFs to show.

    Returns
    -------
    Figure
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n_tfs : int
        Number of TFs to show (top by variance across samples).

    Returns
    -------
    Figure

    Raises
    ------
    FileNotFoundError
        If viper_tf_scores.tsv is missing.
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

    # Select TFs with highest variance across samples
    var = df.var(axis=0)
    top_tfs = var.sort_values(ascending=False).head(top_n_tfs).index
    sub = df[top_tfs]

    # z-score per TF
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
# IMMUNE, EPIGENETIC, METABOLIC SIGNATURES
# ---------------------------------------------------------------------------

def make_immune_enrich_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Summarize immune enrichment (up/down) for one disease.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of terms to keep per direction.

    Returns
    -------
    dict
        {
          "up": DataFrame,
          "down": DataFrame
        }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "immune_enrich_up.csv"
    down_path = disease_dir / "immune_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        tables["up"] = _summarize_enrich_table(up_df, top_n=top_n)
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        tables["down"] = _summarize_enrich_table(down_df, top_n=top_n)
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def plot_immune_enrich_barplot(
    root_dir: PathLike,
    disease: str,
    direction: str = "up",
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of immune enrichment terms for one direction.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    direction : {"up", "down"}
        Which enrichment file to use.
    top_n : int
        Number of terms to show.

    Returns
    -------
    Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "immune_enrich_up.csv"
        label = "Immune enrichment (UP genes)"
    else:
        path = disease_dir / "immune_enrich_down.csv"
        label = "Immune enrichment (DOWN genes)"

    df = _load_enrich_table(path)
    df_top = _summarize_enrich_table(df, top_n=top_n)
    title = f"{disease}: {label} (top {len(df_top)})"
    return _barplot_enrichment(df_top, title=title)


def make_epigenetic_enrich_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Summarize epigenetic enrichment (up/down) for one disease.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of terms to keep per direction.

    Returns
    -------
    dict
        { "up": DataFrame, "down": DataFrame }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "epigenetic_enrich_up.csv"
    down_path = disease_dir / "epigenetic_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        tables["up"] = _summarize_enrich_table(up_df, top_n=top_n)
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        tables["down"] = _summarize_enrich_table(down_df, top_n=top_n)
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def plot_epigenetic_enrich_barplot(
    root_dir: PathLike,
    disease: str,
    direction: str = "up",
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of epigenetic enrichment terms for one direction.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    direction : {"up", "down"}
        Which enrichment file to use.
    top_n : int
        Number of terms to show.

    Returns
    -------
    Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "epigenetic_enrich_up.csv"
        label = "Epigenetic enrichment (UP genes)"
    else:
        path = disease_dir / "epigenetic_enrich_down.csv"
        label = "Epigenetic enrichment (DOWN genes)"

    df = _load_enrich_table(path)
    df_top = _summarize_enrich_table(df, top_n=top_n)
    title = f"{disease}: {label} (top {len(df_top)})"
    return _barplot_enrichment(df_top, title=title)


def make_metabolite_enrich_tables(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Summarize metabolite enrichment (up/down) for one disease.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of terms to keep per direction.

    Returns
    -------
    dict
        { "up": DataFrame, "down": DataFrame }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    up_path = disease_dir / "metabolite_enrich_up.csv"
    down_path = disease_dir / "metabolite_enrich_down.csv"

    tables: Dict[str, pd.DataFrame] = {}

    try:
        up_df = _load_enrich_table(up_path)
        tables["up"] = _summarize_enrich_table(up_df, top_n=top_n)
    except FileNotFoundError:
        tables["up"] = pd.DataFrame()

    try:
        down_df = _load_enrich_table(down_path)
        tables["down"] = _summarize_enrich_table(down_df, top_n=top_n)
    except FileNotFoundError:
        tables["down"] = pd.DataFrame()

    return tables


def plot_metabolite_enrich_barplot(
    root_dir: PathLike,
    disease: str,
    direction: str = "up",
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of metabolite enrichment terms for one direction.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    direction : {"up", "down"}
        Which enrichment file to use.
    top_n : int
        Number of terms to show.

    Returns
    -------
    Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    if direction.lower() == "up":
        path = disease_dir / "metabolite_enrich_up.csv"
        label = "Metabolite enrichment (UP genes)"
    else:
        path = disease_dir / "metabolite_enrich_down.csv"
        label = "Metabolite enrichment (DOWN genes)"

    df = _load_enrich_table(path)
    df_top = _summarize_enrich_table(df, top_n=top_n)
    title = f"{disease}: {label} (top {len(df_top)})"
    return _barplot_enrichment(df_top, title=title)


# ---------------------------------------------------------------------------
# WRAPPERS: one disease / many diseases
# ---------------------------------------------------------------------------

def analyze_signatures_one_disease(
    root_dir: PathLike,
    disease: str,
    top_n_pathways: int = 30,
    top_n_gsea: int = 50,
    top_n_tf_enrich: int = 30,
    top_n_tf_activity: int = 30,
    top_n_viper_tfs: int = 50,
    top_n_immune: int = 30,
    top_n_epigenetic: int = 30,
    top_n_metabolite: int = 30,
) -> Dict[str, object]:
    """
    Run full signature analysis for one disease and return tables/figures.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n_* : int
        Top-N parameters for various summaries.

    Returns
    -------
    dict
        Keys include:
        - core_pathways_up, core_pathways_down (tables)
        - gsea_table
        - tf_enrich_up, tf_enrich_down (tables)
        - collectri_tf_table
        - immune_up, immune_down (tables)
        - epigenetic_up, epigenetic_down (tables)
        - metabolite_up, metabolite_down (tables)
        - figs_* : various matplotlib Figure objects
    """
    results: Dict[str, object] = {}

    # --- Pathways (core) ---
    core_tables = make_core_pathway_tables(root_dir, disease, top_n=top_n_pathways)
    results["core_pathways_up"] = core_tables.get("up")
    results["core_pathways_down"] = core_tables.get("down")

    try:
        fig_core_up = plot_core_pathways_barplot(root_dir, disease, direction="up", top_n=top_n_pathways)
    except Exception:
        fig_core_up = None
    try:
        fig_core_down = plot_core_pathways_barplot(root_dir, disease, direction="down", top_n=top_n_pathways)
    except Exception:
        fig_core_down = None

    results["fig_core_up"] = fig_core_up
    results["fig_core_down"] = fig_core_down

    # --- GSEA ---
    try:
        gsea_table = make_gsea_summary_table(root_dir, disease, top_n=top_n_gsea)
        fig_gsea = plot_gsea_nes_panel(root_dir, disease, top_n=top_n_gsea)
    except FileNotFoundError:
        gsea_table = None
        fig_gsea = None
    results["gsea_table"] = gsea_table
    results["fig_gsea"] = fig_gsea

    # --- TF enrichment ---
    tf_tables = make_tf_enrich_tables(root_dir, disease, top_n=top_n_tf_enrich)
    results["tf_enrich_up"] = tf_tables.get("up")
    results["tf_enrich_down"] = tf_tables.get("down")

    try:
        fig_tf_up = plot_tf_enrich_barplot(root_dir, disease, direction="up", top_n=top_n_tf_enrich)
    except Exception:
        fig_tf_up = None
    try:
        fig_tf_down = plot_tf_enrich_barplot(root_dir, disease, direction="down", top_n=top_n_tf_enrich)
    except Exception:
        fig_tf_down = None

    results["fig_tf_up"] = fig_tf_up
    results["fig_tf_down"] = fig_tf_down

    # CollectRI TF activity
    try:
        collectri_table = make_collectri_tf_activity_table(root_dir, disease, top_n=top_n_tf_activity)
        fig_collectri = plot_collectri_tf_activity_barplot(root_dir, disease, top_n=top_n_tf_activity)
    except FileNotFoundError:
        collectri_table = None
        fig_collectri = None

    results["collectri_tf_table"] = collectri_table
    results["fig_collectri_tf"] = fig_collectri

    # VIPER heatmap (optional)
    try:
        fig_viper = plot_viper_tf_heatmap(root_dir, disease, top_n_tfs=top_n_viper_tfs)
    except FileNotFoundError:
        fig_viper = None
    except Exception:
        fig_viper = None
    results["fig_viper_tf"] = fig_viper

    # --- Immune ---
    immune_tables = make_immune_enrich_tables(root_dir, disease, top_n=top_n_immune)
    results["immune_up"] = immune_tables.get("up")
    results["immune_down"] = immune_tables.get("down")

    try:
        fig_immune_up = plot_immune_enrich_barplot(root_dir, disease, direction="up", top_n=top_n_immune)
    except Exception:
        fig_immune_up = None
    try:
        fig_immune_down = plot_immune_enrich_barplot(root_dir, disease, direction="down", top_n=top_n_immune)
    except Exception:
        fig_immune_down = None

    results["fig_immune_up"] = fig_immune_up
    results["fig_immune_down"] = fig_immune_down

    # --- Epigenetic ---
    epi_tables = make_epigenetic_enrich_tables(root_dir, disease, top_n=top_n_epigenetic)
    results["epigenetic_up"] = epi_tables.get("up")
    results["epigenetic_down"] = epi_tables.get("down")

    try:
        fig_epi_up = plot_epigenetic_enrich_barplot(root_dir, disease, direction="up", top_n=top_n_epigenetic)
    except Exception:
        fig_epi_up = None
    try:
        fig_epi_down = plot_epigenetic_enrich_barplot(root_dir, disease, direction="down", top_n=top_n_epigenetic)
    except Exception:
        fig_epi_down = None

    results["fig_epigenetic_up"] = fig_epi_up
    results["fig_epigenetic_down"] = fig_epi_down

    # --- Metabolite ---
    metab_tables = make_metabolite_enrich_tables(root_dir, disease, top_n=top_n_metabolite)
    results["metabolite_up"] = metab_tables.get("up")
    results["metabolite_down"] = metab_tables.get("down")

    try:
        fig_metab_up = plot_metabolite_enrich_barplot(root_dir, disease, direction="up", top_n=top_n_metabolite)
    except Exception:
        fig_metab_up = None
    try:
        fig_metab_down = plot_metabolite_enrich_barplot(root_dir, disease, direction="down", top_n=top_n_metabolite)
    except Exception:
        fig_metab_down = None

    results["fig_metabolite_up"] = fig_metab_up
    results["fig_metabolite_down"] = fig_metab_down

    return results


def analyze_signatures_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, object]]:
    """
    Run the signature analysis for multiple diseases.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    diseases : sequence of str, optional
        Disease folder names. If None, all disease folders found under
        root_dir (excluding global folders) will be used.
    **kwargs
        Additional keyword arguments passed to analyze_signatures_one_disease().

    Returns
    -------
    dict
        Mapping: disease_name -> results_dict.
    """
    root = Path(root_dir)

    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_signatures_one_disease(root_dir=root, disease=disease, **kwargs)
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
            "Signature panels: pathway, regulator, immune, epigenetic, "
            "and metabolic signatures for one or many diseases."
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
        "--top-pathways",
        type=int,
        default=30,
        help="Number of core pathway terms to show per direction.",
    )
    parser.add_argument(
        "--top-gsea",
        type=int,
        default=50,
        help="Number of GSEA pathways to show.",
    )
    parser.add_argument(
        "--top-tf-enrich",
        type=int,
        default=30,
        help="Number of TF enrichment terms to show per direction.",
    )
    parser.add_argument(
        "--top-tf-activity",
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
        "--top-immune",
        type=int,
        default=30,
        help="Number of immune terms to show per direction.",
    )
    parser.add_argument(
        "--top-epigenetic",
        type=int,
        default=30,
        help="Number of epigenetic terms to show per direction.",
    )
    parser.add_argument(
        "--top-metabolite",
        type=int,
        default=30,
        help="Number of metabolite terms to show per direction.",
    )

    args = parser.parse_args()

    results = analyze_signatures_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_n_pathways=args.top_pathways,
        top_n_gsea=args.top_gsea,
        top_n_tf_enrich=args.top_tf_enrich,
        top_n_tf_activity=args.top_tf_activity,
        top_n_viper_tfs=args.top_viper_tfs,
        top_n_immune=args.top_immune,
        top_n_epigenetic=args.top_epigenetic,
        top_n_metabolite=args.top_metabolite,
    )

    # Save outputs under root_dir/agentic_analysis/signature_panels/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "signature_panels"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Save pathway tables ---
        if isinstance(res.get("core_pathways_up"), pd.DataFrame):
            res["core_pathways_up"].to_csv(out_dir / f"{disease}_core_pathways_up.csv", index=False)
        if isinstance(res.get("core_pathways_down"), pd.DataFrame):
            res["core_pathways_down"].to_csv(out_dir / f"{disease}_core_pathways_down.csv", index=False)

        if isinstance(res.get("gsea_table"), pd.DataFrame):
            res["gsea_table"].to_csv(out_dir / f"{disease}_gsea_summary.csv", index=False)

        # --- Save TF tables ---
        if isinstance(res.get("tf_enrich_up"), pd.DataFrame):
            res["tf_enrich_up"].to_csv(out_dir / f"{disease}_tf_enrich_up_summary.csv", index=False)
        if isinstance(res.get("tf_enrich_down"), pd.DataFrame):
            res["tf_enrich_down"].to_csv(out_dir / f"{disease}_tf_enrich_down_summary.csv", index=False)

        if isinstance(res.get("collectri_tf_table"), pd.DataFrame):
            res["collectri_tf_table"].to_csv(out_dir / f"{disease}_collectri_tf_activity.csv", index=False)

        # --- Save immune / epigenetic / metabolite tables ---
        if isinstance(res.get("immune_up"), pd.DataFrame):
            res["immune_up"].to_csv(out_dir / f"{disease}_immune_enrich_up_summary.csv", index=False)
        if isinstance(res.get("immune_down"), pd.DataFrame):
            res["immune_down"].to_csv(out_dir / f"{disease}_immune_enrich_down_summary.csv", index=False)

        if isinstance(res.get("epigenetic_up"), pd.DataFrame):
            res["epigenetic_up"].to_csv(out_dir / f"{disease}_epigenetic_enrich_up_summary.csv", index=False)
        if isinstance(res.get("epigenetic_down"), pd.DataFrame):
            res["epigenetic_down"].to_csv(out_dir / f"{disease}_epigenetic_enrich_down_summary.csv", index=False)

        if isinstance(res.get("metabolite_up"), pd.DataFrame):
            res["metabolite_up"].to_csv(out_dir / f"{disease}_metabolite_enrich_up_summary.csv", index=False)
        if isinstance(res.get("metabolite_down"), pd.DataFrame):
            res["metabolite_down"].to_csv(out_dir / f"{disease}_metabolite_enrich_down_summary.csv", index=False)

        # --- Save figures ---
        # Pathways
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

        # TFs
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

        # Immune
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

        # Epigenetic
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

        # Metabolite
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
        f"Finished signature analysis for {len(results)} disease(s): {', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
