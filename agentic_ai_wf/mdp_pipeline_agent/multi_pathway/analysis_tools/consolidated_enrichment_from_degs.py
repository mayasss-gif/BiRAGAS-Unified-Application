#!/usr/bin/env python3
"""
Consolidated pathway enrichment from DEGs: key biological modules.

This script assumes mdp_pipeline_3 has already run and produced, per disease:

- core_enrich_up.csv / core_enrich_down.csv
- immune_enrich_up.csv / immune_enrich_down.csv
- epigenetic_enrich_up.csv / epigenetic_enrich_down.csv
- metabolite_enrich_up.csv / metabolite_enrich_down.csv
- tf_enrich_up.csv / tf_enrich_down.csv
- gsea_prerank.tsv

It **does not** re-run enrichment; instead, it:

1. Collects all enrichment tables driven by the DEG list.
2. Builds consolidated pathway tables (UP / DOWN) by combining evidence
   across libraries and sources.
3. Assigns each pathway to a coarse "biological module" based on keywords
   (immune, metabolism, cell cycle, etc.).
4. Summarizes modules and directions (UP vs DOWN).
5. Produces visual summaries.

Outputs per disease (saved under
    <root_dir>/agentic_analysis/consolidated_enrichment/<disease>/):

TABLES
------
- <disease>_consolidated_pathways_UP.csv
- <disease>_consolidated_pathways_DOWN.csv
- <disease>_module_summary.csv
- <disease>_gsea_consolidated_table.csv (if GSEA is present)

FIGURES
-------
- <disease>_consolidated_pathways_UP_barplot.png
- <disease>_consolidated_pathways_DOWN_barplot.png
- <disease>_modules_total_impact_barplot.png
- <disease>_modules_direction_barplot.png
- <disease>_gsea_nes_scatter.png (if GSEA is present)

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


def _add_minus_log10_qval(df: pd.DataFrame, qcol: str = "qval") -> pd.DataFrame:
    """
    Add -log10(qcol) column, avoiding log10(0) issues.
    """
    df = df.copy()
    q = df[qcol].replace(0, np.nan)
    min_non_zero = q[q > 0].min()
    if pd.isna(min_non_zero):
        min_non_zero = 1e-300
    df["-log10_qval"] = -np.log10(df[qcol].replace(0, min_non_zero))
    return df


def _barplot_generic(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    label_col: str,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Simple horizontal barplot utility.
    """
    if df.empty:
        raise ValueError("No rows to plot in barplot.")

    fig, ax = plt.subplots(figsize=figsize)
    y = df[label_col]
    x = df[value_col]
    ax.barh(y, x)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_ylabel(label_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GSEA integration
# ---------------------------------------------------------------------------

def _load_gsea_contributions(disease_dir: Path) -> pd.DataFrame:
    """
    Load GSEA prerank results and convert into a unified format with
    impact_score and direction.

    Returns a possibly empty DataFrame with columns at least:
        ['term', 'NES', 'FDR q-val', '-log10_qval', 'impact_score',
         'direction', 'library', 'source']
    """
    gsea_path = disease_dir / "gsea_prerank.tsv"
    if not gsea_path.exists():
        return pd.DataFrame()

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

    # For consolidation, align qval/-log10_qval names
    df["qval"] = df["FDR q-val"]
    df["-log10_qval"] = df["-log10_FDR"]

    df["library"] = "GSEA"
    df["source"] = "gsea"

    return df


def _extract_gsea_summary_from_combined(
    combined: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Extract a GSEA-only summary from the combined enrichment table.
    """
    if combined.empty:
        return pd.DataFrame()

    df = combined[combined["source"] == "gsea"].copy()
    if df.empty:
        return df

    # Expect NES / FDR q-val / -log10_FDR present from _load_gsea_contributions
    if "NES" not in df.columns or "FDR q-val" not in df.columns or "-log10_FDR" not in df.columns:
        # This might happen if combined was heavily trimmed; fall back to impact_score
        df["abs_impact"] = df["impact_score"].abs()
        df = df.sort_values("abs_impact", ascending=False).head(top_n)
        return df[["term", "impact_score"]]

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
    ax.set_title(f"{disease}: GSEA pathways (NES vs -log10(FDR))")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Collect all DEG-driven enrichment rows for a disease
# ---------------------------------------------------------------------------

def collect_all_enrichment_rows(
    root_dir: PathLike,
    disease: str,
) -> pd.DataFrame:
    """
    Collect all DEG-driven enrichment rows for a disease from:

    - core_enrich_up/down.csv
    - immune_enrich_up/down.csv
    - epigenetic_enrich_up/down.csv
    - metabolite_enrich_up/down.csv
    - tf_enrich_up/down.csv
    - gsea_prerank.tsv

    Returns a unified DataFrame where each row is a (library, term, direction,
    source) entry with a signed impact_score.

    impact_score is:
        * for Enrichr-like results: sign(direction) * -log10(qval)
        * for GSEA: NES * -log10(FDR)
    """
    disease_dir = get_disease_dir(root_dir, disease)

    rows: List[pd.DataFrame] = []

    # Enrichr-style enrichment tables
    enrich_configs = [
        ("core", "core_enrich"),
        ("immune", "immune_enrich"),
        ("epigenetic", "epigenetic_enrich"),
        ("metabolite", "metabolite_enrich"),
        ("tf", "tf_enrich"),
    ]

    for source_label, base_name in enrich_configs:
        for direction, suffix in [("UP", "up"), ("DOWN", "down")]:
            path = disease_dir / f"{base_name}_{suffix}.csv"
            if not path.exists():
                continue
            df = _load_enrich_table(path)
            df = _add_minus_log10_qval(df, qcol="qval")
            df["direction"] = direction
            df["source"] = source_label

            sign = 1.0 if direction == "UP" else -1.0
            df["impact_score"] = sign * df["-log10_qval"]
            rows.append(df)

    # GSEA contributions
    gsea_df = _load_gsea_contributions(disease_dir)
    if not gsea_df.empty:
        rows.append(gsea_df)

    if not rows:
        raise FileNotFoundError(
            f"No enrichment files found for {disease}. "
            f"Expected core/immune/epigenetic/metabolite/tf enrichments and/or gsea_prerank.tsv."
        )

    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined


# ---------------------------------------------------------------------------
# Biological module assignment (keyword-based)
# ---------------------------------------------------------------------------

MODULE_KEYWORDS: List[tuple] = [
    ("Immune / Inflammation", [
        "immune", "interferon", "cytokine", "chemokine", "leukocyte",
        "lymphocyte", "inflammatory", "inflammasome", "innate", "adaptive",
    ]),
    ("Cell Cycle / Proliferation", [
        "cell cycle", "mitotic", "mitosis", "g1", "g2", "s phase",
        "proliferation", "checkpoint",
    ]),
    ("Apoptosis / Cell Death", [
        "apoptosis", "cell death", "caspase", "anoikis", "necrosis",
        "programmed cell death",
    ]),
    ("Metabolism / Bioenergetics", [
        "metabolic", "metabolism", "glycolysis", "oxidative phosphorylation",
        "tca cycle", "tricarboxylic", "beta oxidation", "lipid", "fatty acid",
        "amino acid", "glucose", "gluconeogenesis", "pentose phosphate",
    ]),
    ("Signal Transduction", [
        "signaling", "signal transduction", "receptor", "kinase", "mapk",
        "jak-stat", "pi3k", "akt", "nf-kappa", "tyrosine kinase",
    ]),
    ("Stress / DNA Damage", [
        "stress", "dna damage", "repair", "unfolded protein", "er stress",
        "oxidative stress", "uv response",
    ]),
    ("Extracellular Matrix / Adhesion", [
        "extracellular matrix", "ecm", "integrin", "focal adhesion",
        "cell-cell adhesion", "cell adhesion",
    ]),
    ("Development / Differentiation", [
        "development", "morphogenesis", "differentiation", "patterning",
        "embryo", "organogenesis",
    ]),
    ("Translation / Ribosome", [
        "ribosome", "translation", "protein synthesis", "mrna translation",
    ]),
    ("Chromatin / Epigenetic", [
        "chromatin", "histone", "methylation", "acetylation", "epigenetic",
    ]),
]


def assign_module(term: str) -> str:
    """
    Assign a coarse biological module to a pathway term based on keywords.
    """
    t = str(term).lower()
    for module, keywords in MODULE_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return module
    return "Other / General"


# ---------------------------------------------------------------------------
# Consolidated pathway tables
# ---------------------------------------------------------------------------

def build_consolidated_pathways_for_direction(
    combined: pd.DataFrame,
    direction: str,
    top_n: int = 40,
) -> pd.DataFrame:
    """
    Build a consolidated pathway table for one direction (UP or DOWN).

    For each pathway term, we aggregate across libraries/sources and compute:

    - n_hits: number of enrichment rows supporting this term.
    - sum_abs_impact: sum of |impact_score| across all rows.
    - max_abs_impact: max |impact_score|.
    - impact_sum: sum of signed impact_score.
    - impact_mean: mean signed impact_score.
    - libraries: semicolon-joined list of libraries where this term appears.
    - sources: semicolon-joined list of sources (core/immune/epi/metabo/tf/gsea).
    - module: coarse biological module (keyword-based).
    """
    df = combined[combined["direction"] == direction].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "term",
                "direction",
                "module",
                "n_hits",
                "sum_abs_impact",
                "max_abs_impact",
                "impact_sum",
                "impact_mean",
                "libraries",
                "sources",
            ]
        )

    df["abs_impact"] = df["impact_score"].abs()

    # Basic statistics per term
    grouped_stats = df.groupby("term")["impact_score"].agg(
        impact_sum="sum",
        impact_mean="mean",
    )
    grouped_abs = df.groupby("term")["abs_impact"].agg(
        sum_abs_impact="sum",
        max_abs_impact="max",
    )
    n_hits = df.groupby("term").size().rename("n_hits")

    # Rename aggregated series so they become 'libraries' and 'sources'
    libraries = df.groupby("term")["library"].apply(
        lambda x: "; ".join(sorted(set(map(str, x))))
    ).rename("libraries")
    sources = df.groupby("term")["source"].apply(
        lambda x: "; ".join(sorted(set(map(str, x))))
    ).rename("sources")

    summary = pd.concat(
        [n_hits, grouped_abs, grouped_stats, libraries, sources],
        axis=1,
    ).reset_index()

    summary["direction"] = direction
    summary["module"] = summary["term"].apply(assign_module)

    # Rank by sum_abs_impact
    summary = summary.sort_values("sum_abs_impact", ascending=False).head(top_n)

    # Reorder columns
    summary = summary[
        [
            "term",
            "direction",
            "module",
            "n_hits",
            "sum_abs_impact",
            "max_abs_impact",
            "impact_sum",
            "impact_mean",
            "libraries",
            "sources",
        ]
    ]

    return summary


def make_module_summary(
    consolidated_up: pd.DataFrame,
    consolidated_down: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize modules across directions.

    Input: consolidated UP/DOWN tables from build_consolidated_pathways_for_direction.

    Output columns:
        [module, direction, n_pathways, total_abs_impact]
    """
    frames = []
    if consolidated_up is not None and not consolidated_up.empty:
        frames.append(consolidated_up)
    if consolidated_down is not None and not consolidated_down.empty:
        frames.append(consolidated_down)

    if not frames:
        return pd.DataFrame(
            columns=["module", "direction", "n_pathways", "total_abs_impact"]
        )

    df = pd.concat(frames, ignore_index=True)
    if "sum_abs_impact" not in df.columns:
        df["sum_abs_impact"] = df.get("sum_abs_impact", 0.0)

    grouped = (
        df.groupby(["module", "direction"])["sum_abs_impact"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(
            columns={
                "count": "n_pathways",
                "sum": "total_abs_impact",
            }
        )
    )
    return grouped


# ---------------------------------------------------------------------------
# Plotting: consolidated pathways and modules
# ---------------------------------------------------------------------------

def plot_consolidated_pathways_bar(
    consolidated_df: pd.DataFrame,
    disease: str,
    direction: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of top consolidated pathways for one direction.
    """
    if consolidated_df.empty:
        raise ValueError("Consolidated pathway table is empty.")

    df = consolidated_df.copy()
    df = df.sort_values("sum_abs_impact", ascending=False).head(top_n)

    title = f"{disease}: Consolidated pathways ({direction}, top {len(df)})"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="sum_abs_impact",
        label_col="term",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


def plot_module_total_impact(
    module_summary: pd.DataFrame,
    disease: str,
) -> plt.Figure:
    """
    Barplot of total_abs_impact per module, summing across directions.
    """
    if module_summary.empty:
        raise ValueError("Module summary table is empty.")

    df = module_summary.copy()
    agg = (
        df.groupby("module")["total_abs_impact"]
        .sum()
        .reset_index()
        .sort_values("total_abs_impact", ascending=False)
    )

    title = f"{disease}: Total disruption impact per module"
    return _barplot_generic(
        df=agg,
        title=title,
        value_col="total_abs_impact",
        label_col="module",
        figsize=(10, max(4, agg.shape[0] * 0.3)),
    )


def plot_module_direction_bar(
    module_summary: pd.DataFrame,
    disease: str,
) -> plt.Figure:
    """
    Grouped barplot of total_abs_impact per module × direction.
    """
    if module_summary.empty:
        raise ValueError("Module summary table is empty.")

    df = module_summary.copy()
    pivot = df.pivot(index="module", columns="direction", values="total_abs_impact").fillna(0.0)

    modules = pivot.index.tolist()
    directions = pivot.columns.tolist()

    x = np.arange(len(modules))
    width = 0.4 if len(directions) == 2 else 0.5

    fig, ax = plt.subplots(figsize=(10, max(4, len(modules) * 0.3)))

    for idx, direction in enumerate(directions):
        offset = (idx - (len(directions) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[direction].values,
            width=width / max(len(directions), 1),
            label=direction,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=45, ha="right")
    ax.set_ylabel("Total disruption impact (sum |impact_score|)")
    ax.set_title(f"{disease}: Module impact by direction")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wrappers: one disease / many diseases
# ---------------------------------------------------------------------------

def analyze_consolidated_enrichment_one_disease(
    root_dir: PathLike,
    disease: str,
    top_n_pathways: int = 40,
    top_n_gsea: int = 50,
) -> Dict[str, object]:
    """
    Run the consolidated enrichment analysis for one disease.

    Returns
    -------
    dict
        {
          "combined_rows": DataFrame,
          "consolidated_up": DataFrame,
          "consolidated_down": DataFrame,
          "module_summary": DataFrame,
          "gsea_table": DataFrame or None,
          "fig_consolidated_up": Figure or None,
          "fig_consolidated_down": Figure or None,
          "fig_modules_total": Figure or None,
          "fig_modules_direction": Figure or None,
          "fig_gsea_scatter": Figure or None,
        }
    """
    results: Dict[str, object] = {}

    # Collect all enrichment rows
    combined = collect_all_enrichment_rows(root_dir=root_dir, disease=disease)
    results["combined_rows"] = combined

    # Consolidated tables
    consolidated_up = build_consolidated_pathways_for_direction(
        combined=combined,
        direction="UP",
        top_n=top_n_pathways,
    )
    consolidated_down = build_consolidated_pathways_for_direction(
        combined=combined,
        direction="DOWN",
        top_n=top_n_pathways,
    )

    results["consolidated_up"] = consolidated_up
    results["consolidated_down"] = consolidated_down

    # Module summary
    module_summary = make_module_summary(consolidated_up, consolidated_down)
    results["module_summary"] = module_summary

    # GSEA summary (if present)
    gsea_table = _extract_gsea_summary_from_combined(combined, top_n=top_n_gsea)
    if gsea_table is None or gsea_table.empty:
        gsea_table = None
    results["gsea_table"] = gsea_table

    # Figures
    try:
        fig_up = plot_consolidated_pathways_bar(
            consolidated_df=consolidated_up,
            disease=disease,
            direction="UP",
            top_n=top_n_pathways,
        )
    except Exception:
        fig_up = None

    try:
        fig_down = plot_consolidated_pathways_bar(
            consolidated_df=consolidated_down,
            disease=disease,
            direction="DOWN",
            top_n=top_n_pathways,
        )
    except Exception:
        fig_down = None

    try:
        fig_modules_total = plot_module_total_impact(module_summary, disease=disease)
    except Exception:
        fig_modules_total = None

    try:
        fig_modules_direction = plot_module_direction_bar(module_summary, disease=disease)
    except Exception:
        fig_modules_direction = None

    if gsea_table is not None and not gsea_table.empty:
        try:
            fig_gsea = plot_gsea_nes_scatter(gsea_table, disease=disease)
        except Exception:
            fig_gsea = None
    else:
        fig_gsea = None

    results["fig_consolidated_up"] = fig_up
    results["fig_consolidated_down"] = fig_down
    results["fig_modules_total"] = fig_modules_total
    results["fig_modules_direction"] = fig_modules_direction
    results["fig_gsea_scatter"] = fig_gsea

    return results


def analyze_consolidated_enrichment_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    top_n_pathways: int = 40,
    top_n_gsea: int = 50,
) -> Dict[str, Dict[str, object]]:
    """
    Run consolidated enrichment analysis for multiple diseases.
    """
    root = Path(root_dir)
    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_consolidated_enrichment_one_disease(
                root_dir=root,
                disease=disease,
                top_n_pathways=top_n_pathways,
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
            "Use DEG-driven enrichment outputs to build consolidated pathway "
            "enrichment tables and highlight key biological modules."
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
        default=40,
        help="Number of consolidated pathways to keep per direction.",
    )
    parser.add_argument(
        "--top-gsea",
        type=int,
        default=50,
        help="Number of GSEA pathways to summarize (if present).",
    )

    args = parser.parse_args()

    results = analyze_consolidated_enrichment_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_n_pathways=args.top_pathways,
        top_n_gsea=args.top_gsea,
    )

    # Save outputs under root_dir/agentic_analysis/consolidated_enrichment/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "consolidated_enrichment"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # Consolidated tables
        up_tbl = res.get("consolidated_up")
        if isinstance(up_tbl, pd.DataFrame):
            up_tbl.to_csv(out_dir / f"{disease}_consolidated_pathways_UP.csv", index=False)

        down_tbl = res.get("consolidated_down")
        if isinstance(down_tbl, pd.DataFrame):
            down_tbl.to_csv(out_dir / f"{disease}_consolidated_pathways_DOWN.csv", index=False)

        module_tbl = res.get("module_summary")
        if isinstance(module_tbl, pd.DataFrame):
            module_tbl.to_csv(out_dir / f"{disease}_module_summary.csv", index=False)

        gsea_tbl = res.get("gsea_table")
        if isinstance(gsea_tbl, pd.DataFrame):
            gsea_tbl.to_csv(out_dir / f"{disease}_gsea_consolidated_table.csv", index=False)

        # Figures
        if res.get("fig_consolidated_up") is not None:
            res["fig_consolidated_up"].savefig(
                out_dir / f"{disease}_consolidated_pathways_UP_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_consolidated_down") is not None:
            res["fig_consolidated_down"].savefig(
                out_dir / f"{disease}_consolidated_pathways_DOWN_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_modules_total") is not None:
            res["fig_modules_total"].savefig(
                out_dir / f"{disease}_modules_total_impact_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_modules_direction") is not None:
            res["fig_modules_direction"].savefig(
                out_dir / f"{disease}_modules_direction_barplot.png",
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
        f"Finished consolidated enrichment analysis for {len(results)} disease(s): "
        f"{', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
