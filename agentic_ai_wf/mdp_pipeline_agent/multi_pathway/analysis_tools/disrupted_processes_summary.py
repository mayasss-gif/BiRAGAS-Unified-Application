#!/usr/bin/env python3
"""
Disrupted biological processes summary for one or many diseases.

Given a root output folder from mdp_pipeline_3 and one or more disease
folders inside it, this script:

- Uses DE-derived pathway enrichment outputs:
    * core_enrich_up.csv / core_enrich_down.csv
    * gsea_prerank.tsv
- Builds a unified "major disrupted pathways" table with signed impact scores.
- Summarizes which libraries (KEGG/Reactome/WikiPathways/GO, etc.) contribute most.
- Produces publication-ready plots:
    * Lollipop plot of major disrupted pathways.
    * Barplot of library × direction counts.
    * Barplot of library total disruption impact.
    * Scatter of NES vs -log10(FDR) (GSEA).

All outputs are saved under:
    <root_dir>/agentic_analysis/disrupted_processes/<disease>/

Everything is written so it can later be wrapped as OpenAI function tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence, Dict, List, Optional
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _add_minus_log10_qval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add -log10(qval) column, avoiding log10(0) issues.
    """
    df = df.copy()
    q = df["qval"].replace(0, np.nan)
    min_non_zero = q[q > 0].min()
    if pd.isna(min_non_zero):
        min_non_zero = 1e-300
    df["-log10_qval"] = -np.log10(df["qval"].replace(0, min_non_zero))
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
# GSEA helpers
# ---------------------------------------------------------------------------

def make_gsea_summary_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Summarize GSEA prerank results for one disease.

    Returns columns:
        [term, NES, FDR q-val, -log10_FDR, direction, impact_score]
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

    # Signed impact: NES * -log10(FDR)
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
# Major disrupted pathway table
# ---------------------------------------------------------------------------

def build_major_pathway_table(
    root_dir: PathLike,
    disease: str,
    top_n: int = 40,
) -> pd.DataFrame:
    """
    Build a unified 'major disrupted pathways' table by combining:
    - core_enrich_up.csv
    - core_enrich_down.csv
    - gsea_prerank.tsv (if available)

    Each row has a signed impact_score:
      * core_enrich: sign(direction) * -log10(qval)
      * GSEA: NES * -log10(FDR)

    Parameters
    ----------
    root_dir : str or Path
        Base output directory.
    disease : str
        Disease folder name.
    top_n : int
        Number of strongest pathways to keep (by |impact_score|).

    Returns
    -------
    DataFrame
        Columns include:
            [source, library, term, direction, impact_score,
             -log10_qval / -log10_FDR, etc.]
    """
    disease_dir = get_disease_dir(root_dir, disease)

    rows: List[pd.DataFrame] = []

    # Core enrichment: UP
    up_path = disease_dir / "core_enrich_up.csv"
    if up_path.exists():
        up_df = _load_enrich_table(up_path)
        up_df = _add_minus_log10_qval(up_df)
        up_df["direction"] = "UP"
        up_df["source"] = "core_enrich"
        up_df["impact_score"] = up_df["-log10_qval"]  # positive for UP
        rows.append(up_df)

    # Core enrichment: DOWN
    down_path = disease_dir / "core_enrich_down.csv"
    if down_path.exists():
        down_df = _load_enrich_table(down_path)
        down_df = _add_minus_log10_qval(down_df)
        down_df["direction"] = "DOWN"
        down_df["source"] = "core_enrich"
        down_df["impact_score"] = -down_df["-log10_qval"]  # negative for DOWN
        rows.append(down_df)

    # GSEA
    try:
        gsea_df = make_gsea_summary_table(root_dir=root_dir, disease=disease, top_n=1000)
        gsea_df = gsea_df.copy()
        gsea_df["library"] = "GSEA"
        gsea_df["source"] = "gsea"
        # already has impact_score
        gsea_df = gsea_df.rename(
            columns={"FDR q-val": "qval", "-log10_FDR": "-log10_qval"}
        )
        # Keep same columns as core where possible
        rows.append(gsea_df)
    except FileNotFoundError:
        gsea_df = pd.DataFrame()

    if not rows:
        raise FileNotFoundError(
            f"No pathway enrichment files found for {disease} "
            f"(expected core_enrich_* and/or gsea_prerank.tsv)."
        )

    combined = pd.concat(rows, ignore_index=True, sort=False)

    # Keep essential columns; others are carried along if present
    essential_cols = ["source", "library", "term", "direction", "impact_score"]
    extra_cols = [c for c in combined.columns if c not in essential_cols]
    combined = combined[essential_cols + extra_cols]

    combined["abs_impact"] = combined["impact_score"].abs()
    combined = combined.sort_values("abs_impact", ascending=False).head(top_n)

    return combined


def make_library_direction_summary(
    major_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how many disrupted pathways and total impact each library contributes.

    Parameters
    ----------
    major_df : DataFrame
        Output from build_major_pathway_table().

    Returns
    -------
    DataFrame
        Columns:
            [library, direction, n_pathways, total_abs_impact, mean_abs_impact]
    """
    if major_df.empty:
        return pd.DataFrame(columns=["library", "direction", "n_pathways",
                                     "total_abs_impact", "mean_abs_impact"])

    df = major_df.copy()
    df["abs_impact"] = df["impact_score"].abs()

    summary = (
        df.groupby(["library", "direction"])["abs_impact"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "n_pathways",
                "sum": "total_abs_impact",
                "mean": "mean_abs_impact",
            }
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Plotting: major disrupted processes
# ---------------------------------------------------------------------------

def plot_major_pathways_lollipop(
    major_df: pd.DataFrame,
    disease: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Lollipop plot of the most disrupted pathways (signed impact_score).

    Positive = more activated (UP); negative = more suppressed (DOWN).
    """
    if major_df.empty:
        raise ValueError("Major pathway table is empty.")

    df = major_df.copy()
    df = df.sort_values("abs_impact", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, df.shape[0] * 0.25)))

    y = np.arange(df.shape[0])
    x = df["impact_score"]

    # Stems
    for i, val in enumerate(x):
        ax.hlines(y=i, xmin=0, xmax=val, linewidth=1)

    # Points
    for i, (val, direction) in enumerate(zip(x, df["direction"])):
        ax.scatter(val, i, s=30, alpha=0.8, label=direction if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(df["term"], fontsize=7)
    ax.axvline(0.0, linestyle="--", linewidth=1)

    ax.set_xlabel("Signed impact score")
    ax.set_title(f"{disease}: Major disrupted pathways (top {len(df)})")
    fig.tight_layout()
    return fig


def plot_library_counts_barplot(
    summary_df: pd.DataFrame,
    disease: str,
) -> plt.Figure:
    """
    Barplot of n_pathways per library × direction.
    """
    if summary_df.empty:
        raise ValueError("Library summary table is empty.")

    df = summary_df.copy()
    # Pivot to library x direction
    pivot = df.pivot(index="library", columns="direction", values="n_pathways").fillna(0)

    libraries = pivot.index.tolist()
    directions = pivot.columns.tolist()

    x = np.arange(len(libraries))
    width = 0.35 if len(directions) == 2 else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, direction in enumerate(directions):
        offset = (idx - (len(directions) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[direction].values,
            width=width / max(len(directions), 1),
            label=direction,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(libraries, rotation=45, ha="right")
    ax.set_ylabel("Number of disrupted pathways")
    ax.set_title(f"{disease}: Disrupted pathways per library and direction")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_library_impact_barplot(
    summary_df: pd.DataFrame,
    disease: str,
) -> plt.Figure:
    """
    Barplot of total_abs_impact per library (summing across directions).
    """
    if summary_df.empty:
        raise ValueError("Library summary table is empty.")

    df = summary_df.copy()
    agg = (
        df.groupby("library")["total_abs_impact"]
        .sum()
        .reset_index()
        .sort_values("total_abs_impact", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg["library"], agg["total_abs_impact"])
    ax.set_ylabel("Total disruption impact (sum |impact_score|)")
    ax.set_xlabel("Library")
    ax.set_title(f"{disease}: Total disruption impact per library")
    ax.set_xticklabels(agg["library"], rotation=45, ha="right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wrappers: one disease / many diseases
# ---------------------------------------------------------------------------

def analyze_disrupted_processes_one_disease(
    root_dir: PathLike,
    disease: str,
    top_n_major: int = 40,
    top_n_gsea: int = 50,
) -> Dict[str, object]:
    """
    Run the disrupted-processes analysis for one disease.

    Returns
    -------
    dict
        {
          "major_pathways_table": DataFrame,
          "library_summary_table": DataFrame,
          "gsea_table": DataFrame or None,
          "fig_lollipop": Figure or None,
          "fig_library_counts": Figure or None,
          "fig_library_impact": Figure or None,
          "fig_gsea_scatter": Figure or None,
        }
    """
    results: Dict[str, object] = {}

    # Major pathway table
    major_table = build_major_pathway_table(root_dir=root_dir, disease=disease, top_n=top_n_major)
    results["major_pathways_table"] = major_table

    # Library summary
    lib_summary = make_library_direction_summary(major_table)
    results["library_summary_table"] = lib_summary

    # GSEA table (may already be partly included)
    try:
        gsea_table = make_gsea_summary_table(root_dir=root_dir, disease=disease, top_n=top_n_gsea)
    except FileNotFoundError:
        gsea_table = None
    results["gsea_table"] = gsea_table

    # Plots
    try:
        fig_lollipop = plot_major_pathways_lollipop(major_table, disease=disease, top_n=top_n_major)
    except Exception:
        fig_lollipop = None

    try:
        fig_library_counts = plot_library_counts_barplot(lib_summary, disease=disease)
    except Exception:
        fig_library_counts = None

    try:
        fig_library_impact = plot_library_impact_barplot(lib_summary, disease=disease)
    except Exception:
        fig_library_impact = None

    if gsea_table is not None and not gsea_table.empty:
        try:
            fig_gsea_scatter = plot_gsea_nes_scatter(gsea_table, disease=disease)
        except Exception:
            fig_gsea_scatter = None
    else:
        fig_gsea_scatter = None

    results["fig_lollipop"] = fig_lollipop
    results["fig_library_counts"] = fig_library_counts
    results["fig_library_impact"] = fig_library_impact
    results["fig_gsea_scatter"] = fig_gsea_scatter

    return results


def analyze_disrupted_processes_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    top_n_major: int = 40,
    top_n_gsea: int = 50,
) -> Dict[str, Dict[str, object]]:
    """
    Run disrupted-processes analysis for multiple diseases.
    """
    root = Path(root_dir)

    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_disrupted_processes_one_disease(
                root_dir=root,
                disease=disease,
                top_n_major=top_n_major,
                top_n_gsea=top_n_gsea,
            )
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
            "Disrupted biological processes: summarize major pathway changes "
            "from core enrichment and GSEA for one or many diseases."
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
        "--top-major",
        type=int,
        default=40,
        help="Number of major disrupted pathways to keep.",
    )
    parser.add_argument(
        "--top-gsea",
        type=int,
        default=50,
        help="Number of GSEA pathways to summarize.",
    )

    args = parser.parse_args()

    results = analyze_disrupted_processes_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_n_major=args.top_major,
        top_n_gsea=args.top_gsea,
    )

    # Save outputs under root_dir/agentic_analysis/disrupted_processes/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "disrupted_processes"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        major_tbl = res.get("major_pathways_table")
        if isinstance(major_tbl, pd.DataFrame):
            major_tbl.to_csv(out_dir / f"{disease}_major_disrupted_pathways.csv", index=False)

        lib_tbl = res.get("library_summary_table")
        if isinstance(lib_tbl, pd.DataFrame):
            lib_tbl.to_csv(out_dir / f"{disease}_library_direction_summary.csv", index=False)

        gsea_tbl = res.get("gsea_table")
        if isinstance(gsea_tbl, pd.DataFrame):
            gsea_tbl.to_csv(out_dir / f"{disease}_gsea_disrupted_pathways.csv", index=False)

        # Figures
        if res.get("fig_lollipop") is not None:
            res["fig_lollipop"].savefig(
                out_dir / f"{disease}_major_pathways_lollipop.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_library_counts") is not None:
            res["fig_library_counts"].savefig(
                out_dir / f"{disease}_library_counts_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_library_impact") is not None:
            res["fig_library_impact"].savefig(
                out_dir / f"{disease}_library_impact_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_gsea_scatter") is not None:
            res["fig_gsea_scatter"].savefig(
                out_dir / f"{disease}_gsea_nes_scatter.png",
                dpi=300,
                bbox_inches="tight",
            )

    print(
        f"Finished disrupted-processes analysis for {len(results)} disease(s): "
        f"{', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
