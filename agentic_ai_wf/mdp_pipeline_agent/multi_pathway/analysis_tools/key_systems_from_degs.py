#!/usr/bin/env python3
"""
Query: Take this DEG list and identify the most affected pathways,
regulators, and interacting biological systems.

This script assumes mdp_pipeline_3 has already run and produced, per disease:

ENRICHMENT (pathways)
----------------------
- core_enrich_up.csv / core_enrich_down.csv
- immune_enrich_up.csv / immune_enrich_down.csv
- epigenetic_enrich_up.csv / epigenetic_enrich_down.csv
- metabolite_enrich_up.csv / metabolite_enrich_down.csv
- tf_enrich_up.csv / tf_enrich_down.csv
- (optional) gsea_prerank.tsv

REGULATORS
----------
- viper_tf_scores.tsv                 (sample-level TF activity)
- ulm_collectri_tf_scores.tsv         (contrast-level TF activity)
- Enzyme_and_Signaling/*PTM_kinase_activity.csv
  or NEW_omnipath_outputs/*PTM_kinase_activity.csv (PTM kinase NES)

SYSTEM-LEVEL SIGNATURES
------------------------
- ulm_hallmark_scores.tsv             (Hallmark ULM)
- ulm_progeny_pathway_scores.tsv      (PROGENy ULM)   [optional]

It does NOT re-run enrichment or DEG calling. Instead, for each disease it:

1. Consolidates enrichment across libraries to get top pathways (UP/DOWN)
   and assigns coarse biological modules (immune, metabolism, etc).
2. Summarizes regulators: TFs (VIPER + CollectRI) and kinases (PTM NES).
3. Summarizes system-level signatures (Hallmark + PROGENy).
4. Produces publication-style barplots for pathways, regulators, and modules.

All outputs are written to:
    <root_dir>/agentic_analysis/key_systems_from_degs/<Disease>/

TABLES
------
- <disease>_top_pathways_UP.csv
- <disease>_top_pathways_DOWN.csv
- <disease>_module_summary.csv
- <disease>_top_regulators_combined.csv
- <disease>_top_hallmark_signatures.csv
- <disease>_top_progeny_signatures.csv   (if PROGENy is present)

FIGURES
-------
- <disease>_top_pathways_UP_barplot.png
- <disease>_top_pathways_DOWN_barplot.png
- <disease>_modules_impact_barplot.png
- <disease>_top_TF_regulators_barplot.png
- <disease>_top_kinase_regulators_barplot.png
- <disease>_hallmark_signatures_barplot.png
- <disease>_progeny_signatures_barplot.png   (if PROGENy is present)

Everything is structured so it can later be turned into OpenAI function tools.
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

    # GSEA contributions (optional)
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
# Regulators: TFs (VIPER + CollectRI) and kinases (PTM NES)
# ---------------------------------------------------------------------------

def load_collectri_tf_table(disease_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """
    Load CollectRI ULM TF scores and return top TFs by |score|.
    """
    tf_path = disease_dir / "ulm_collectri_tf_scores.tsv"
    if not tf_path.exists():
        raise FileNotFoundError(f"CollectRI TF scores not found: {tf_path}")

    df = pd.read_csv(tf_path, sep="\t")
    if df.empty:
        raise ValueError(f"CollectRI TF scores file is empty: {tf_path}")

    row = df.iloc[0].drop(labels=["Unnamed: 0"], errors="ignore")
    scores = pd.to_numeric(row, errors="coerce")

    tbl = pd.DataFrame({
        "TF": scores.index,
        "ulm_score": scores.values,
    })
    tbl["abs_ulm_score"] = tbl["ulm_score"].abs()
    tbl = tbl.dropna(subset=["ulm_score"])
    tbl = tbl.sort_values("abs_ulm_score", ascending=False).head(top_n)

    return tbl.reset_index(drop=True)


def load_viper_tf_table(disease_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """
    Load VIPER TF activity matrix and return mean NES per TF (top by |NES|).
    """
    viper_path = disease_dir / "viper_tf_scores.tsv"
    if not viper_path.exists():
        raise FileNotFoundError(f"VIPER TF scores not found: {viper_path}")

    df = pd.read_csv(viper_path, sep="\t")
    if "sample" in df.columns:
        tf_cols = [c for c in df.columns if c != "sample"]
    else:
        tf_cols = list(df.columns)

    if not tf_cols:
        raise ValueError(f"No TF columns found in VIPER file: {viper_path}")

    means = df[tf_cols].mean(axis=0, numeric_only=True)
    tbl = means.to_frame(name="viper_mean_NES").reset_index()
    tbl = tbl.rename(columns={"index": "TF_raw"})

    # Strip leading "TF:" if present
    tbl["TF"] = tbl["TF_raw"].astype(str).str.replace(r"^TF:", "", regex=True)
    tbl["abs_viper_mean_NES"] = tbl["viper_mean_NES"].abs()
    tbl = tbl.dropna(subset=["viper_mean_NES"])
    tbl = tbl.sort_values("abs_viper_mean_NES", ascending=False).head(top_n)

    return tbl[["TF", "viper_mean_NES", "abs_viper_mean_NES"]].reset_index(drop=True)


def _find_ptm_kinase_file(disease_dir: Path) -> Optional[Path]:
    """
    Try to discover a PTM kinase activity file for this disease.

    Expected locations:
      - <disease_dir>/Enzyme_and_Signaling/*PTM_kinase_activity.csv
      - <disease_dir>/NEW_omnipath_outputs/*PTM_kinase_activity.csv
      - <disease_dir>/*PTM_kinase_activity.csv
    """
    candidates: List[Path] = []

    for sub in ["Enzyme_and_Signaling", "NEW_omnipath_outputs", "."]:
        subdir = disease_dir / sub
        if not subdir.exists():
            continue
        candidates.extend(subdir.glob("*PTM_kinase_activity.csv"))

    if not candidates:
        return None
    # Just take the first match
    return sorted(candidates)[0]


def load_ptm_kinase_table(disease_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """
    Load PTM kinase activity (NES) and return top kinases by |NES|.
    """
    ptm_path = _find_ptm_kinase_file(disease_dir)
    if ptm_path is None:
        raise FileNotFoundError(
            f"No PTM kinase activity file found under {disease_dir} "
            f"(searched Enzyme_and_Signaling/ and NEW_omnipath_outputs/)."
        )

    df = pd.read_csv(ptm_path)
    needed = {"enzyme", "NES"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"PTM file {ptm_path} missing columns: {missing}")

    # In case of duplicates, average NES per enzyme
    grouped = df.groupby("enzyme")["NES"].mean()

    tbl = grouped.to_frame(name="NES").reset_index()
    tbl["abs_NES"] = tbl["NES"].abs()
    tbl = tbl.dropna(subset=["NES"])
    tbl = tbl.sort_values("abs_NES", ascending=False).head(top_n)

    return tbl.reset_index(drop=True)


def make_regulator_summary(
    collectri_tbl: Optional[pd.DataFrame],
    viper_tbl: Optional[pd.DataFrame],
    ptm_tbl: Optional[pd.DataFrame],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Combine TF (CollectRI + VIPER) and kinase (PTM) information into one table.

    Output columns:
        entity_type: "TF_ULM", "TF_VIPER", "Kinase_PTM"
        name:        TF or enzyme name
        score:       underlying score
        abs_score:   |score|
        source:      "CollectRI", "VIPER", "PTM_kinase"
    """
    frames: List[pd.DataFrame] = []

    if collectri_tbl is not None and not collectri_tbl.empty:
        c = collectri_tbl.copy()
        c["entity_type"] = "TF_ULM"
        c["name"] = c["TF"]
        c["score"] = c["ulm_score"]
        c["abs_score"] = c["abs_ulm_score"]
        c["source"] = "CollectRI"
        frames.append(c[["entity_type", "name", "score", "abs_score", "source"]])

    if viper_tbl is not None and not viper_tbl.empty:
        v = viper_tbl.copy()
        v["entity_type"] = "TF_VIPER"
        v["name"] = v["TF"]
        v["score"] = v["viper_mean_NES"]
        v["abs_score"] = v["abs_viper_mean_NES"]
        v["source"] = "VIPER"
        frames.append(v[["entity_type", "name", "score", "abs_score", "source"]])

    if ptm_tbl is not None and not ptm_tbl.empty:
        p = ptm_tbl.copy()
        p["entity_type"] = "Kinase_PTM"
        p["name"] = p["enzyme"]
        p["score"] = p["NES"]
        p["abs_score"] = p["abs_NES"]
        p["source"] = "PTM_kinase"
        frames.append(p[["entity_type", "name", "score", "abs_score", "source"]])

    if not frames:
        return pd.DataFrame(
            columns=["entity_type", "name", "score", "abs_score", "source"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("abs_score", ascending=False).head(top_n)
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# System-level signatures: Hallmarks & PROGENy
# ---------------------------------------------------------------------------

def load_hallmark_scores(disease_dir: Path, top_n: int = 20) -> pd.DataFrame:
    """
    Load Hallmark ULM scores and return top signatures by |score|.
    """
    hl_path = disease_dir / "ulm_hallmark_scores.tsv"
    if not hl_path.exists():
        raise FileNotFoundError(f"Hallmark scores not found: {hl_path}")

    df = pd.read_csv(hl_path, sep="\t")
    if df.empty:
        raise ValueError(f"Hallmark scores file is empty: {hl_path}")

    row = df.iloc[0].drop(labels=["Unnamed: 0"], errors="ignore")
    scores = pd.to_numeric(row, errors="coerce")

    tbl = pd.DataFrame({
        "hallmark": scores.index,
        "score": scores.values,
    })
    tbl["abs_score"] = tbl["score"].abs()
    tbl = tbl.dropna(subset=["score"])
    tbl = tbl.sort_values("abs_score", ascending=False).head(top_n)

    return tbl.reset_index(drop=True)


def load_progeny_scores(disease_dir: Path, top_n: int = 20) -> pd.DataFrame:
    """
    Load PROGENy ULM scores and return top pathways by |score|.

    File: ulm_progeny_pathway_scores.tsv
    """
    pr_path = disease_dir / "ulm_progeny_pathway_scores.tsv"
    if not pr_path.exists():
        raise FileNotFoundError(f"PROGENy scores not found: {pr_path}")

    df = pd.read_csv(pr_path, sep="\t")
    if df.empty:
        raise ValueError(f"PROGENy scores file is empty: {pr_path}")

    row = df.iloc[0].drop(labels=["Unnamed: 0"], errors="ignore")
    scores = pd.to_numeric(row, errors="coerce")

    tbl = pd.DataFrame({
        "progeny_pathway": scores.index,
        "score": scores.values,
    })
    tbl["abs_score"] = tbl["score"].abs()
    tbl = tbl.dropna(subset=["score"])
    tbl = tbl.sort_values("abs_score", ascending=False).head(top_n)

    return tbl.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers: pathways, modules, regulators, signatures
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

    title = f"{disease}: Top pathways ({direction}, consolidated)"
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

    title = f"{disease}: Most affected biological modules"
    return _barplot_generic(
        df=agg,
        title=title,
        value_col="total_abs_impact",
        label_col="module",
        figsize=(10, max(4, agg.shape[0] * 0.3)),
    )


def plot_top_tf_regulators(
    regulator_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 25,
) -> plt.Figure:
    """
    Horizontal barplot of top TF regulators (both ULM + VIPER).
    """
    if regulator_tbl.empty:
        raise ValueError("Regulator table is empty.")

    df = regulator_tbl[regulator_tbl["entity_type"].isin(["TF_ULM", "TF_VIPER"])].copy()
    if df.empty:
        raise ValueError("No TF entries in regulator table.")

    df = df.sort_values("abs_score", ascending=False).head(top_n)
    df["label"] = df["name"] + " (" + df["source"] + ")"

    title = f"{disease}: Top TF regulators (ULM + VIPER)"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="abs_score",
        label_col="label",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


def plot_top_kinase_regulators(
    regulator_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 25,
) -> plt.Figure:
    """
    Horizontal barplot of top kinase regulators (PTM NES).
    """
    if regulator_tbl.empty:
        raise ValueError("Regulator table is empty.")

    df = regulator_tbl[regulator_tbl["entity_type"] == "Kinase_PTM"].copy()
    if df.empty:
        raise ValueError("No kinase entries in regulator table.")

    df = df.sort_values("abs_score", ascending=False).head(top_n)

    title = f"{disease}: Top kinases (PTM NES)"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="abs_score",
        label_col="name",
        figsize=(8, max(4, df.shape[0] * 0.25)),
    )


def plot_hallmark_signatures(
    hall_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 20,
) -> plt.Figure:
    """
    Horizontal barplot of top Hallmark signatures by |score|.
    """
    if hall_tbl.empty:
        raise ValueError("Hallmark table is empty.")

    df = hall_tbl.copy()
    df = df.sort_values("abs_score", ascending=False).head(top_n)

    title = f"{disease}: Hallmark activity (top {len(df)})"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="score",
        label_col="hallmark",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


def plot_progeny_signatures(
    progeny_tbl: pd.DataFrame,
    disease: str,
    top_n: int = 20,
) -> plt.Figure:
    """
    Horizontal barplot of top PROGENy signatures by |score|.
    """
    if progeny_tbl.empty:
        raise ValueError("PROGENy table is empty.")

    df = progeny_tbl.copy()
    df = df.sort_values("abs_score", ascending=False).head(top_n)

    title = f"{disease}: PROGENy pathway activity (top {len(df)})"
    return _barplot_generic(
        df=df,
        title=title,
        value_col="score",
        label_col="progeny_pathway",
        figsize=(10, max(4, df.shape[0] * 0.25)),
    )


# ---------------------------------------------------------------------------
# Main per-disease analysis
# ---------------------------------------------------------------------------

def analyze_key_systems_one_disease(
    root_dir: PathLike,
    disease: str,
    top_n_pathways: int = 30,
    top_n_regulators: int = 50,
    top_n_signatures: int = 20,
) -> Dict[str, object]:
    """
    Run the "key systems from DEGs" analysis for one disease.

    Returns
    -------
    dict
        {
          "combined_enrichment": DataFrame,
          "pathways_up": DataFrame,
          "pathways_down": DataFrame,
          "module_summary": DataFrame,
          "regulators_table": DataFrame,
          "hallmark_table": DataFrame or None,
          "progeny_table": DataFrame or None,
          "fig_pathways_up": Figure or None,
          "fig_pathways_down": Figure or None,
          "fig_modules": Figure or None,
          "fig_tf_regulators": Figure or None,
          "fig_kinase_regulators": Figure or None,
          "fig_hallmark": Figure or None,
          "fig_progeny": Figure or None,
        }
    """
    disease_dir = get_disease_dir(root_dir, disease)
    results: Dict[str, object] = {}

    # 1) Consolidated pathways + modules
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

    module_summary = make_module_summary(pathways_up, pathways_down)

    results["pathways_up"] = pathways_up
    results["pathways_down"] = pathways_down
    results["module_summary"] = module_summary

    # 2) Regulators: TFs + kinases
    try:
        collectri_tbl = load_collectri_tf_table(disease_dir, top_n=top_n_regulators)
    except Exception:
        collectri_tbl = None

    try:
        viper_tbl = load_viper_tf_table(disease_dir, top_n=top_n_regulators)
    except Exception:
        viper_tbl = None

    try:
        ptm_tbl = load_ptm_kinase_table(disease_dir, top_n=top_n_regulators)
    except Exception:
        ptm_tbl = None

    regulators_table = make_regulator_summary(
        collectri_tbl=collectri_tbl,
        viper_tbl=viper_tbl,
        ptm_tbl=ptm_tbl,
        top_n=top_n_regulators,
    )
    results["regulators_table"] = regulators_table

    # 3) System-level signatures
    try:
        hallmark_table = load_hallmark_scores(disease_dir, top_n=top_n_signatures)
    except Exception:
        hallmark_table = None

    try:
        progeny_table = load_progeny_scores(disease_dir, top_n=top_n_signatures)
    except Exception:
        progeny_table = None

    results["hallmark_table"] = hallmark_table
    results["progeny_table"] = progeny_table

    # 4) Plots
    # Pathways
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

    # Modules
    try:
        fig_modules = plot_module_total_impact(module_summary, disease=disease)
    except Exception:
        fig_modules = None

    # TFs
    try:
        fig_tf_regulators = plot_top_tf_regulators(
            regulators_table, disease=disease, top_n=min(25, top_n_regulators)
        )
    except Exception:
        fig_tf_regulators = None

    # Kinases
    try:
        fig_kinase_regulators = plot_top_kinase_regulators(
            regulators_table, disease=disease, top_n=min(25, top_n_regulators)
        )
    except Exception:
        fig_kinase_regulators = None

    # Hallmarks
    if hallmark_table is not None and not hallmark_table.empty:
        try:
            fig_hallmark = plot_hallmark_signatures(
                hallmark_table, disease=disease, top_n=top_n_signatures
            )
        except Exception:
            fig_hallmark = None
    else:
        fig_hallmark = None

    # PROGENy
    if progeny_table is not None and not progeny_table.empty:
        try:
            fig_progeny = plot_progeny_signatures(
                progeny_table, disease=disease, top_n=top_n_signatures
            )
        except Exception:
            fig_progeny = None
    else:
        fig_progeny = None

    results["fig_pathways_up"] = fig_pathways_up
    results["fig_pathways_down"] = fig_pathways_down
    results["fig_modules"] = fig_modules
    results["fig_tf_regulators"] = fig_tf_regulators
    results["fig_kinase_regulators"] = fig_kinase_regulators
    results["fig_hallmark"] = fig_hallmark
    results["fig_progeny"] = fig_progeny

    return results


def analyze_key_systems_multiple_diseases(
    root_dir: PathLike,
    diseases: Optional[Sequence[str]] = None,
    top_n_pathways: int = 30,
    top_n_regulators: int = 50,
    top_n_signatures: int = 20,
) -> Dict[str, Dict[str, object]]:
    """
    Run key-systems analysis for multiple diseases.
    """
    root = Path(root_dir)
    if diseases is None:
        diseases = list_disease_folders(root)

    all_results: Dict[str, Dict[str, object]] = {}

    for disease in diseases:
        try:
            res = analyze_key_systems_one_disease(
                root_dir=root,
                disease=disease,
                top_n_pathways=top_n_pathways,
                top_n_regulators=top_n_regulators,
                top_n_signatures=top_n_signatures,
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
            "Take a DEG-derived enrichment set and identify the most affected "
            "pathways, regulators (TFs/kinases), and biological systems."
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
        "--top-regulators",
        type=int,
        default=50,
        help="Number of regulators (TFs + kinases) to keep.",
    )
    parser.add_argument(
        "--top-signatures",
        type=int,
        default=20,
        help="Number of Hallmark/PROGENy signatures to keep.",
    )

    args = parser.parse_args()

    results = analyze_key_systems_multiple_diseases(
        root_dir=args.root_dir,
        diseases=args.diseases,
        top_n_pathways=args.top_pathways,
        top_n_regulators=args.top_regulators,
        top_n_signatures=args.top_signatures,
    )

    # Save outputs under root_dir/agentic_analysis/key_systems_from_degs/<disease>/
    root_path = Path(args.root_dir)
    agentic_root = root_path / "agentic_analysis" / "key_systems_from_degs"
    agentic_root.mkdir(parents=True, exist_ok=True)

    for disease, res in results.items():
        out_dir = agentic_root / disease
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pathway tables
        up_tbl = res.get("pathways_up")
        if isinstance(up_tbl, pd.DataFrame):
            up_tbl.to_csv(out_dir / f"{disease}_top_pathways_UP.csv", index=False)

        down_tbl = res.get("pathways_down")
        if isinstance(down_tbl, pd.DataFrame):
            down_tbl.to_csv(out_dir / f"{disease}_top_pathways_DOWN.csv", index=False)

        module_tbl = res.get("module_summary")
        if isinstance(module_tbl, pd.DataFrame):
            module_tbl.to_csv(out_dir / f"{disease}_module_summary.csv", index=False)

        # Regulators
        reg_tbl = res.get("regulators_table")
        if isinstance(reg_tbl, pd.DataFrame):
            reg_tbl.to_csv(out_dir / f"{disease}_top_regulators_combined.csv", index=False)

        # Signatures
        hall_tbl = res.get("hallmark_table")
        if isinstance(hall_tbl, pd.DataFrame):
            hall_tbl.to_csv(out_dir / f"{disease}_top_hallmark_signatures.csv", index=False)

        prog_tbl = res.get("progeny_table")
        if isinstance(prog_tbl, pd.DataFrame):
            prog_tbl.to_csv(out_dir / f"{disease}_top_progeny_signatures.csv", index=False)

        # Figures
        if res.get("fig_pathways_up") is not None:
            res["fig_pathways_up"].savefig(
                out_dir / f"{disease}_top_pathways_UP_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_pathways_down") is not None:
            res["fig_pathways_down"].savefig(
                out_dir / f"{disease}_top_pathways_DOWN_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_modules") is not None:
            res["fig_modules"].savefig(
                out_dir / f"{disease}_modules_impact_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_tf_regulators") is not None:
            res["fig_tf_regulators"].savefig(
                out_dir / f"{disease}_top_TF_regulators_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_kinase_regulators") is not None:
            res["fig_kinase_regulators"].savefig(
                out_dir / f"{disease}_top_kinase_regulators_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_hallmark") is not None:
            res["fig_hallmark"].savefig(
                out_dir / f"{disease}_hallmark_signatures_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

        if res.get("fig_progeny") is not None:
            res["fig_progeny"].savefig(
                out_dir / f"{disease}_progeny_signatures_barplot.png",
                dpi=300,
                bbox_inches="tight",
            )

    print(
        f"Finished key-systems analysis for {len(results)} disease(s): "
        f"{', '.join(results.keys())}\n"
        f"Outputs saved under: {agentic_root}"
    )
