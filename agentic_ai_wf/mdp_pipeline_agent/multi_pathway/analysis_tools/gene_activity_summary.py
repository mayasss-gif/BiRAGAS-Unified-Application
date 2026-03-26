#!/usr/bin/env python3
"""
End-to-end gene activity summary for one or many diseases.

Given a root output folder from mdp_pipeline_3 and one or more disease
folders inside it, this script provides small, reusable functions to:

- Summarize unusually active/suppressed genes (DEGs table).
- Plot a volcano plot.
- Plot an MA plot.
- Plot a heatmap of top DEGs across samples.
- Plot a Hallmark signature panel (ULM scores).
- Summarize TF activity (CollectRI ULM table).

Everything is written so it can later be wrapped as OpenAI function tools.

All generated outputs are saved under:
    <root_dir>/agentic_analysis/gene_activity_summary/<disease>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

PathLike = Union[str, Path]


def list_disease_folders(
    root_dir: PathLike,
    exclude: Sequence[str] = ("baseline_consensus", "comparison", "results"),
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


# ---------------------------------------------------------------------------
# Single-disease: DEG summary table
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

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    lfc_col : str
        Name of log2 fold-change column in `degs_from_counts.csv`.
    padj_col : str
        Name of adjusted p-value column.
    base_mean_col : str
        Name of mean expression column.
    lfc_cutoff : float
        Absolute log2 fold-change threshold for calling genes "unusual".
    padj_cutoff : float
        Adjusted p-value threshold for significance.
    top_n : int
        Total number of genes to return (half up, half down).

    Returns
    -------
    pd.DataFrame
        Table with columns [Gene, baseMean, log2FoldChange, padj/pvalue, direction].
        Sorted by |log2FoldChange| descending.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    df = pd.read_csv(deg_path)

    # Drop rows without log2FC
    df = df.dropna(subset=[lfc_col]).copy()

    # Direction + significance filter
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

    # Split up/down and take top_n/2 from each side by |log2FC|
    up = sig[sig["direction"] == "UP"].sort_values(lfc_col, ascending=False)
    down = sig[sig["direction"] == "DOWN"].sort_values(lfc_col, ascending=True)

    n_each = max(top_n // 2, 1)
    top_up = up.head(n_each)
    top_down = down.head(n_each)

    out = pd.concat([top_up, top_down], axis=0)
    out = out[["Gene", base_mean_col, lfc_col, pcol_to_use, "direction"]]

    return out.sort_values(lfc_col, ascending=False)


# ---------------------------------------------------------------------------
# Single-disease: Volcano plot
# ---------------------------------------------------------------------------

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

    X-axis: log2 fold-change (activity vs suppression).
    Y-axis: -log10(p), highlighting unusually active/suppressed genes.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    lfc_col : str
        Log2 fold-change column name.
    p_col : str
        Column for adjusted p-values (falls back to 'pvalue' if missing).
    lfc_thresh : float
        Threshold for |log2FC| to define strong changes.
    p_thresh : float
        P-value threshold for significance.
    top_n_labels : int
        Maximum number of extreme points to label by gene name.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object for further customization or saving.
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    df = pd.read_csv(deg_path)

    if p_col not in df.columns:
        p_col = "pvalue"

    # Drop NAs and compute -log10 p
    df = df.dropna(subset=[lfc_col, p_col]).copy()
    # Replace zeros with smallest non-zero p to avoid inf
    min_non_zero = df.loc[df[p_col] > 0, p_col].min()
    df["neg_log10_p"] = -np.log10(df[p_col].replace(0, min_non_zero))

    # Categories: up/down/non-significant
    sig_mask = (df[p_col] <= p_thresh) & (df[lfc_col].abs() >= lfc_thresh)
    df["category"] = "Not significant"
    df.loc[sig_mask & (df[lfc_col] > 0), "category"] = "Up"
    df.loc[sig_mask & (df[lfc_col] < 0), "category"] = "Down"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter by category (you can customize colors later if desired)
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

    # Threshold lines
    ax.axvline(lfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, linestyle="--", linewidth=1)
    ax.axhline(-np.log10(p_thresh), linestyle="--", linewidth=1)

    # Label a few most extreme genes by |log2FC| * -log10(p)
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


# ---------------------------------------------------------------------------
# Single-disease: MA plot
# ---------------------------------------------------------------------------

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

    X-axis: log10(baseMean + 1)
    Y-axis: log2 fold-change.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    lfc_col : str
        Log2 fold-change column name.
    base_mean_col : str
        Mean expression column name.
    padj_col : str
        Adjusted p-value column name.
    padj_thresh : float
        Threshold to highlight significantly changed genes.

    Returns
    -------
    matplotlib.figure.Figure
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
    # Non-significant
    ax.scatter(
        df.loc[~sig, "log10_baseMean"],
        df.loc[~sig, lfc_col],
        s=10,
        alpha=0.5,
        label="Not significant",
    )
    # Significant
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


# ---------------------------------------------------------------------------
# Single-disease: Heatmap of top DEGs across samples
# ---------------------------------------------------------------------------

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

    Steps:
    - Select top DEGs by |log2FC| with significance filter.
    - Extract their counts from `combined_counts.csv`.
    - Z-score per gene and plot as a heatmap.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    top_n : int
        Number of genes to include in the heatmap.
    lfc_col : str
        Log2 fold-change column in DE table.
    padj_col : str
        Adjusted p-value column in DE table.
    lfc_cutoff : float
        Minimum |log2FC| for a gene to be considered.
    padj_cutoff : float
        Maximum adjusted p-value for significance.

    Returns
    -------
    matplotlib.figure.Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    deg_path = disease_dir / "degs_from_counts.csv"
    counts_path = disease_dir / "combined_counts.csv"

    deg = pd.read_csv(deg_path)
    counts = pd.read_csv(counts_path)

    # Filter DEGs
    if padj_col in deg.columns:
        deg_f = deg.dropna(subset=[lfc_col, padj_col]).copy()
        deg_f = deg_f[(deg_f[padj_col] <= padj_cutoff) & (deg_f[lfc_col].abs() >= lfc_cutoff)]
    else:
        deg_f = deg.dropna(subset=[lfc_col, "pvalue"]).copy()
        deg_f = deg_f[(deg_f["pvalue"] <= padj_cutoff) & (deg_f[lfc_col].abs() >= lfc_cutoff)]

    # Take top_n by |log2FC|
    deg_f["abs_lfc"] = deg_f[lfc_col].abs()
    top = deg_f.sort_values("abs_lfc", ascending=False).head(top_n)

    # Subset counts matrix
    genes = top["Gene"].unique()
    expr = counts[counts["Gene"].isin(genes)].set_index("Gene")

    if expr.empty:
        raise ValueError(f"No expression rows found for selected DEGs in {disease}.")

    # Z-score per gene (row-wise)
    expr_z = expr.apply(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8),
        axis=1,
    )

    # Simple heatmap using imshow
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
# Single-disease: Hallmark signature panel
# ---------------------------------------------------------------------------

def plot_hallmark_signature_panel(
    root_dir: PathLike,
    disease: str,
    top_n: int = 15,
) -> plt.Figure:
    """
    Barplot of top Hallmark pathway activity scores (ULM).

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    top_n : int
        Number of strongest pathways to display (by |score|).

    Returns
    -------
    matplotlib.figure.Figure
    """
    disease_dir = get_disease_dir(root_dir, disease)
    hl_path = disease_dir / "ulm_hallmark_scores.tsv"
    if not hl_path.exists():
        raise FileNotFoundError(f"Hallmark scores not found for {disease}: {hl_path}")

    df = pd.read_csv(hl_path, sep="\t")
    # Assume single row with 'Unnamed: 0' as label
    row = df.iloc[0].drop(labels=["Unnamed: 0"], errors="ignore")
    scores = row.astype(float)

    tbl = scores.to_frame(name="score")
    tbl["hallmark"] = tbl.index
    tbl["abs_score"] = tbl["score"].abs()

    tbl = tbl.sort_values("abs_score", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(tbl["hallmark"], tbl["score"])
    ax.invert_yaxis()

    ax.set_title(f"{disease}: Hallmark activity panel (top {top_n})")
    ax.set_xlabel("ULM score (activation vs suppression)")
    ax.set_ylabel("Hallmark")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Single-disease: TF activity table (CollectRI ULM)
# ---------------------------------------------------------------------------

def make_tf_activity_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Summarize TF activity from ULM CollectRI scores.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    top_n : int
        Number of TFs to report (by |score|).

    Returns
    -------
    pd.DataFrame
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


# ---------------------------------------------------------------------------
# Wrappers: analyze one or many diseases
# ---------------------------------------------------------------------------

def analyze_one_disease(
    root_dir: PathLike,
    disease: str,
    top_deg_n: int = 50,
    top_hallmarks_n: int = 15,
    top_tf_n: int = 30,
) -> Dict[str, object]:
    """
    Run the gene activity summary for a single disease.

    Returns a dictionary of tables and figures, which you can later use
    in an agentic workflow or a notebook.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    disease : str
        Disease folder name.
    top_deg_n : int
        Number of DEGs to summarize/plot in detail.
    top_hallmarks_n : int
        Number of Hallmarks to show.
    top_tf_n : int
        Number of TFs to show in the TF activity table.

    Returns
    -------
    dict
        {
          "deg_table": DataFrame,
          "tf_table": DataFrame (or None if missing),
          "fig_volcano": Figure,
          "fig_ma": Figure,
          "fig_heatmap": Figure,
          "fig_hallmark": Figure (or None if missing)
        }
    """
    results: Dict[str, object] = {}

    # DEG table
    deg_table = make_deg_summary_table(
        root_dir=root_dir,
        disease=disease,
        top_n=top_deg_n,
    )
    results["deg_table"] = deg_table

    # TF table (may fail if file missing)
    try:
        tf_table = make_tf_activity_table(
            root_dir=root_dir,
            disease=disease,
            top_n=top_tf_n,
        )
    except FileNotFoundError:
        tf_table = None
    results["tf_table"] = tf_table

    # Plots
    fig_volcano = plot_deg_volcano(root_dir=root_dir, disease=disease)
    fig_ma = plot_deg_ma(root_dir=root_dir, disease=disease)
    fig_heatmap = plot_top_degs_heatmap(
        root_dir=root_dir,
        disease=disease,
        top_n=top_deg_n,
    )

    # Hallmark panel (optional)
    try:
        fig_hallmark = plot_hallmark_signature_panel(
            root_dir=root_dir,
            disease=disease,
            top_n=top_hallmarks_n,
        )
    except FileNotFoundError:
        fig_hallmark = None

    results["fig_volcano"] = fig_volcano
    results["fig_ma"] = fig_ma
    results["fig_heatmap"] = fig_heatmap
    results["fig_hallmark"] = fig_hallmark

    return results


def analyze_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    top_deg_n: int = 50,
    top_hallmarks_n: int = 15,
    top_tf_n: int = 30,
) -> Dict[str, Dict[str, object]]:
    """
    Run the gene activity summary for multiple diseases.

    Parameters
    ----------
    root_dir : str or Path
        Base output directory from mdp_pipeline_3.
    diseases : sequence of str, optional
        Disease folder names. If None, all disease folders found under
        root_dir (excluding global folders) will be used.
    top_deg_n : int
        Number of DEGs to summarize/plot per disease.
    top_hallmarks_n : int
        Number of Hallmarks to show per disease.
    top_tf_n : int
        Number of TFs in the TF activity table per disease.

    Returns
    -------
    dict
        Mapping: disease_name -> results_dict (same structure as
        `analyze_one_disease`).
    """
    root = Path(root_dir)

    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_one_disease(
                root_dir=root,
                disease=disease,
                top_deg_n=top_deg_n,
                top_hallmarks_n=top_hallmarks_n,
                top_tf_n=top_tf_n,
            )
            all_results[disease] = res
        except FileNotFoundError as e:
            # Gracefully skip diseases with missing core files
            print(f"[WARN] Skipping {disease}: {e}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Gene activity summary: summarize unusually active/suppressed genes "
            "and signature panels for one or many diseases."
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
            "root_dir (excluding 'baseline_consensus', 'comparison', 'results') "
            "will be analyzed."
        ),
    )
    parser.add_argument(
        "--top-deg-n",
        type=int,
        default=50,
        help="Number of DEGs to summarize/plot per disease.",
    )
    parser.add_argument(
        "--top-hallmarks-n",
        type=int,
        default=15,
        help="Number of Hallmarks to show per disease.",
    )
    parser.add_argument(
        "--top-tf-n",
        type=int,
        default=30,
        help="Number of TFs to show in the TF activity table per disease.",
    )

    args = parser.parse_args()

    results = analyze_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_deg_n=args.top_deg_n,
        top_hallmarks_n=args.top_hallmarks_n,
        top_tf_n=args.top_tf_n,
    )

    # Save outputs under root_dir/agentic_analysis/gene_activity_summary/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "gene_activity_summary"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Save plots ---
        fig_volcano = res.get("fig_volcano")
        if fig_volcano is not None:
            fig_volcano.savefig(
                out_dir / f"{disease}_volcano.png",
                dpi=300,
                bbox_inches="tight",
            )

        fig_ma = res.get("fig_ma")
        if fig_ma is not None:
            fig_ma.savefig(
                out_dir / f"{disease}_MA.png",
                dpi=300,
                bbox_inches="tight",
            )

        fig_heatmap = res.get("fig_heatmap")
        if fig_heatmap is not None:
            fig_heatmap.savefig(
                out_dir / f"{disease}_topDEG_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )

        fig_hallmark = res.get("fig_hallmark")
        if fig_hallmark is not None:
            fig_hallmark.savefig(
                out_dir / f"{disease}_hallmark_panel.png",
                dpi=300,
                bbox_inches="tight",
            )

        # --- Save tables ---
        deg_table = res.get("deg_table")
        if deg_table is not None:
            deg_table.to_csv(
                out_dir / f"{disease}_DEG_summary.csv",
                index=False,
            )

        tf_table = res.get("tf_table")
        if tf_table is not None:
            tf_table.to_csv(
                out_dir / f"{disease}_TF_activity_summary.csv",
                index=False,
            )

    print(
        f"Finished analysis for {len(results)} disease(s): {', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
