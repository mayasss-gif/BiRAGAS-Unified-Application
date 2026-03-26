#!/usr/bin/env python3
"""
Query: Use this list of condition-associated genes (here: disease-associated
genes) to infer pathways, regulators, and major biological mechanisms.

This script assumes mdp_pipeline_3 has already run and produced, per disease:

ENRICHMENT (gene-list–driven)
------------------------------
- core_enrich_up.csv / core_enrich_down.csv
- immune_enrich_up.csv / immune_enrich_down.csv
- epigenetic_enrich_up.csv / epigenetic_enrich_down.csv
- metabolite_enrich_up.csv / metabolite_enrich_down.csv
- tf_enrich_up.csv / tf_enrich_down.csv
- (optional) gsea_prerank.tsv

It does NOT re-run enrichment or DEG calling. Instead, for each disease it:

1. Collects all enrichment results that were driven by a condition/disease-
   associated gene list (DEGs or other).
2. Consolidates pathways across libraries into UP / DOWN tables and assigns
   coarse biological modules (immune, metabolism, etc.).
3. Parses the `genes` column to build a gene-centric "condition gene
   centrality" table: which genes appear in many high-impact pathways and
   which modules they touch.
4. Produces publication-style barplots summarizing:
   - the most affected pathways (UP and DOWN),
   - the most affected biological modules,
   - the most central condition genes (by pathway coverage and impact).

Outputs are written to a single folder:
    <root_dir>/agentic_analysis/condition_gene_mechanisms/

Each file is prefixed by the disease name, e.g.:

TABLES
------
- <disease>_condition_gene_centrality.csv
- <disease>_top_pathways_from_condition_genes_UP.csv
- <disease>_top_pathways_from_condition_genes_DOWN.csv
- <disease>_modules_from_condition_genes.csv

FIGURES
-------
- <disease>_top_pathways_from_condition_genes_UP_barplot.png
- <disease>_top_pathways_from_condition_genes_DOWN_barplot.png
- <disease>_modules_from_condition_genes_barplot.png
- <disease>_top_condition_genes_by_pathway_count_barplot.png
- <disease>_top_condition_genes_by_impact_barplot.png

Everything is structured so it can later be turned into OpenAI function tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence, Dict, List, Optional
import re
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
    Simple horizontal barplot utility (single chart, no subplots).
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
# GSEA integration (optional)
# ---------------------------------------------------------------------------

def _load_gsea_contributions(disease_dir: Path) -> pd.DataFrame:
    """
    Load GSEA prerank results and convert into a unified format with
    impact_score and direction.

    Returns a possibly empty DataFrame with at least:
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


# ---------------------------------------------------------------------------
# Collect all gene-list–driven enrichment rows for a disease
# ---------------------------------------------------------------------------

def collect_all_enrichment_rows(
    root_dir: PathLike,
    disease: str,
) -> pd.DataFrame:
    """
    Collect all enrichment rows for a disease from:

    - core_enrich_up/down.csv
    - immune_enrich_up/down.csv
    - epigenetic_enrich_up/down.csv
    - metabolite_enrich_up/down.csv
    - tf_enrich_up/down.csv
    - gsea_prerank.tsv (optional)

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

    # GSEA contributions (optional). Note: no `genes` column here, but useful for pathways.
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
    top_n: int = 30,
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
# Condition-gene–centric analysis
# ---------------------------------------------------------------------------

def _split_genes_field(genes_str: str) -> List[str]:
    """
    Split the 'genes' field into individual gene symbols.

    Handles separators like ';', ',', and whitespace.
    """
    if pd.isna(genes_str):
        return []
    genes_str = str(genes_str).strip()
    if not genes_str:
        return []
    # Split on semicolon, comma, or whitespace
    parts = re.split(r"[;, ]+", genes_str)
    return [p for p in parts if p]


def build_condition_gene_centrality_table(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a condition-gene centrality table from the combined enrichment rows.

    For each gene that appears in the 'genes' column of enrichment tables
    (i.e. a condition/disease-associated gene that overlaps with enriched
    pathways), we compute:

    - n_pathways: number of unique enriched terms containing this gene.
    - total_abs_impact: sum of |impact_score| across those pathways.
    - mean_abs_impact: mean |impact_score| across those pathways.
    - modules: semicolon-joined list of modules touched by those pathways.
    - top_pathways: up to 3 highest-impact pathways for this gene.
    """
    if "genes" not in combined.columns:
        # No explicit overlap gene lists present (e.g. pure GSEA only)
        return pd.DataFrame(
            columns=[
                "condition_gene",
                "n_pathways",
                "total_abs_impact",
                "mean_abs_impact",
                "modules",
                "top_pathways",
            ]
        )

    rows: List[Dict[str, object]] = []

    for _, row in combined.iterrows():
        if "genes" not in row or pd.isna(row["genes"]):
            continue
        term = row.get("term", "")
        impact = float(row.get("impact_score", 0.0))
        abs_impact = abs(impact)
        module = assign_module(term)

        genes = _split_genes_field(row["genes"])
        for g in genes:
            rows.append(
                {
                    "gene": g,
                    "term": term,
                    "module": module,
                    "abs_impact": abs_impact,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition_gene",
                "n_pathways",
                "total_abs_impact",
                "mean_abs_impact",
                "modules",
                "top_pathways",
            ]
        )

    dfg = pd.DataFrame(rows)

    # Basic stats per gene
    gene_stats = dfg.groupby("gene").agg(
        n_pathways=("term", "nunique"),
        total_abs_impact=("abs_impact", "sum"),
        mean_abs_impact=("abs_impact", "mean"),
    )

    # Modules per gene
    modules = dfg.groupby("gene")["module"].apply(
        lambda x: "; ".join(sorted(set(map(str, x))))
    ).rename("modules")

    # Top pathways per gene (by abs_impact)
    dfg_sorted = dfg.sort_values("abs_impact", ascending=False)
    top_terms = dfg_sorted.groupby("gene")["term"].apply(
        lambda x: "; ".join(list(dict.fromkeys(x))[:3])
    ).rename("top_pathways")

    gene_summary = pd.concat(
        [gene_stats, modules, top_terms],
        axis=1,
    ).reset_index()

    gene_summary = gene_summary.rename(columns={"gene": "condition_gene"})

    # Order by combined centrality (total_abs_impact then n_pathways)
    gene_summary = gene_summary.sort_values(
        ["total_abs_impact", "n_pathways"],
        ascending=[False, False],
    )

    return gene_summary


# ---------------------------------------------------------------------------
# Plotting: pathways, modules, condition genes
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

    title = f"{disease}: Pathways from condition genes ({direction})"
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

    title = f"{disease}: Major biological mechanisms from condition genes"
    return _barplot_generic(
        df=agg,
        title=title,
        value_col="total_abs_impact",
        label_col="module",
        figsize=(10, max(4, agg.shape[0] * 0.3)),
    )


def plot_top_condition_genes_by_pathway_count(
    gene_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of condition-associated genes ranked by n_pathways.
    """
    if gene_tbl.empty:
        raise ValueError("Condition-gene table is empty.")

    df = gene_tbl.copy()
    df = df.sort_values("n_pathways", ascending=False).head(top_n)

    title = f"{disease}: Condition genes with broad pathway coverage"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="n_pathways",
        label_col="condition_gene",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


def plot_top_condition_genes_by_impact(
    gene_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 30,
) -> plt.Figure:
    """
    Barplot of condition-associated genes ranked by total_abs_impact.
    """
    if gene_tbl.empty:
        raise ValueError("Condition-gene table is empty.")

    df = gene_tbl.copy()
    df = df.sort_values("total_abs_impact", ascending=False).head(top_n)

    title = f"{disease}: Condition genes with highest pathway impact"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="total_abs_impact",
        label_col="condition_gene",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


# ---------------------------------------------------------------------------
# Main per-disease analysis
# ---------------------------------------------------------------------------

def analyze_condition_genes_one_disease(
    root_dir: PathLike,
    disease: str,
    top_n_pathways: int = 30,
    top_n_genes: int = 30,
) -> Dict[str, object]:
    """
    Run the “condition gene mechanisms” analysis for one disease.

    Returns
    -------
    dict
        {
          "combined_enrichment": DataFrame,
          "pathways_up": DataFrame,
          "pathways_down": DataFrame,
          "module_summary": DataFrame,
          "gene_centrality": DataFrame,
          "fig_pathways_up": Figure or None,
          "fig_pathways_down": Figure or None,
          "fig_modules": Figure or None,
          "fig_genes_by_count": Figure or None,
          "fig_genes_by_impact": Figure or None,
        }
    """
    results: Dict[str, object] = {}

    # 1) Collect enrichment and build consolidated pathways
    combined_enrich = collect_all_enrichment_rows(root_dir, disease)
    results["combined_enrichment"] = combined_enrich

    pathways_up = build_consolidated_pathways_for_direction(
        combined=combined_enrich,
        direction="UP",
        top_n=top_n_pathways,
    )
    pathways_down = build_consolidated_pathways_for_direction(
        combined=combined_enrich,
        direction="DOWN",
        top_n=top_n_pathways,
    )
    results["pathways_up"] = pathways_up
    results["pathways_down"] = pathways_down

    # 2) Module summary
    module_summary = make_module_summary(pathways_up, pathways_down)
    results["module_summary"] = module_summary

    # 3) Condition-gene centrality
    gene_centrality = build_condition_gene_centrality_table(combined_enrich)
    results["gene_centrality"] = gene_centrality

    # 4) Plots
    try:
        fig_pathways_up = plot_consolidated_pathways_bar(
            pathways_up, disease=disease, direction="UP", top_n=top_n_pathways
        )
    except Exception:
        fig_pathways_up = None

    try:
        fig_pathways_down = plot_consolidated_pathways_bar(
            pathways_down, disease=disease, direction="DOWN", top_n=top_n_pathways
        )
    except Exception:
        fig_pathways_down = None

    try:
        fig_modules = plot_module_total_impact(module_summary, disease=disease)
    except Exception:
        fig_modules = None

    try:
        fig_genes_by_count = plot_top_condition_genes_by_pathway_count(
            gene_centrality, disease=disease, top_n=top_n_genes
        )
    except Exception:
        fig_genes_by_count = None

    try:
        fig_genes_by_impact = plot_top_condition_genes_by_impact(
            gene_centrality, disease=disease, top_n=top_n_genes
        )
    except Exception:
        fig_genes_by_impact = None

    results["fig_pathways_up"] = fig_pathways_up
    results["fig_pathways_down"] = fig_pathways_down
    results["fig_modules"] = fig_modules
    results["fig_genes_by_count"] = fig_genes_by_count
    results["fig_genes_by_impact"] = fig_genes_by_impact

    return results


def analyze_condition_genes_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    top_n_pathways: int = 30,
    top_n_genes: int = 30,
) -> Dict[str, Dict[str, object]]:
    """
    Run condition-gene mechanism analysis for multiple diseases.
    """
    root = Path(root_dir)
    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_condition_genes_one_disease(
                root_dir=root,
                disease=disease,
                top_n_pathways=top_n_pathways,
                top_n_genes=top_n_genes,
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
            "Use a condition/disease-associated gene list (implicit in enrichment "
            "results) to infer key pathways, biological modules, and central "
            "genes."
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
        help="Number of consolidated pathways to keep per direction.",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=30,
        help="Number of condition genes to highlight in gene-centric plots.",
    )

    args = parser.parse_args()

    results = analyze_condition_genes_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_n_pathways=args.top_pathways,
        top_n_genes=args.top_genes,
    )

    # Save outputs under a SINGLE folder:
    #   root_dir/agentic_analysis/condition_gene_mechanisms/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "condition_gene_mechanisms"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        # Tables
        up_tbl = res.get("pathways_up")
        if isinstance(up_tbl, pd.DataFrame):
            up_tbl.to_csv(
                agentic_root / f"{disease}_top_pathways_from_condition_genes_UP.csv",
                index=False,
            )

        down_tbl = res.get("pathways_down")
        if isinstance(down_tbl, pd.DataFrame):
            down_tbl.to_csv(
                agentic_root / f"{disease}_top_pathways_from_condition_genes_DOWN.csv",
                index=False,
            )

        module_tbl = res.get("module_summary")
        if isinstance(module_tbl, pd.DataFrame):
            module_tbl.to_csv(
                agentic_root / f"{disease}_modules_from_condition_genes.csv",
                index=False,
            )

        gene_tbl = res.get("gene_centrality")
        if isinstance(gene_tbl, pd.DataFrame):
            gene_tbl.to_csv(
                agentic_root / f"{disease}_condition_gene_centrality.csv",
                index=False,
            )

        # Figures
        if res.get("fig_pathways_up") is not None:
            res["fig_pathways_up"].savefig(
                agentic_root / f"{disease}_top_pathways_from_condition_genes_UP_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_pathways_down") is not None:
            res["fig_pathways_down"].savefig(
                agentic_root / f"{disease}_top_pathways_from_condition_genes_DOWN_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_modules") is not None:
            res["fig_modules"].savefig(
                agentic_root / f"{disease}_modules_from_condition_genes_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_genes_by_count") is not None:
            res["fig_genes_by_count"].savefig(
                agentic_root / f"{disease}_top_condition_genes_by_pathway_count_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_genes_by_impact") is not None:
            res["fig_genes_by_impact"].savefig(
                agentic_root / f"{disease}_top_condition_genes_by_impact_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

    print(
        f"Finished condition-gene mechanism analysis for {len(results)} disease(s): "
        f"{', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
