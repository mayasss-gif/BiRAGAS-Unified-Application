# src/depmap_dependencies_uif.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from logging import Logger

# Try display() if in notebook; otherwise fallback to print
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)


# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------
from .constants import DATA_DIR
from ..plotly_mpl_export import save_plotly_png_with_mpl

@dataclass
class DependencyTidyResult:
    mapping_path: Path
    tidy_path: Path
    n_selected_genes: int
    n_selected_models: int
    n_rows: int
    thresholds: Dict[str, float]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def save_plotly(fig, png_path: Path, html_path: Path, scale: int = 2) -> None:
    """
    Save a Plotly figure as PNG and HTML.
    PNG export requires kaleido; if not available, only HTML is saved.
    """
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))

    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        ok = save_plotly_png_with_mpl(fig, png_path, scale=scale)
        if not ok:
            fig.write_image(str(png_path), scale=scale)
    except Exception as e:
        print(f"⚠️ Could not save PNG for {png_path.name}: {e}")


def _load_gene_reference(path_gene: Path, logger: Optional[Logger] = None) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Load DepMap Gene.csv and construct:
      - alias_map: alias/previous symbol -> approved symbol
      - sym2eid  : approved symbol -> entrez_id (string)
    """
    if logger:
        logger.info("Loading gene reference from %s", path_gene)

    header = pd.read_csv(path_gene, nrows=0).columns.tolist()
    usecols = [c for c in ["symbol", "entrez_id", "alias_symbol", "prev_symbol"] if c in header]

    gref = pd.read_csv(path_gene, usecols=usecols)

    if "symbol" not in gref.columns:
        raise ValueError("Gene reference must contain 'symbol' column.")

    approved = set(gref["symbol"].dropna().astype(str))

    # Fill alias/prev columns
    for c in ["alias_symbol", "prev_symbol"]:
        if c in gref.columns:
            gref[c] = gref[c].fillna("").astype(str)

    # alias_map: alias → approved symbol
    alias_map: Dict[str, str] = {}
    for _, r in gref.dropna(subset=["symbol"]).iterrows():
        base = str(r["symbol"]).strip()
        for col in ["alias_symbol", "prev_symbol"]:
            if col in gref.columns:
                for tok in re.split(r"[|;,]", str(r[col])):
                    tok = tok.strip()
                    if tok and tok not in alias_map:
                        alias_map[tok] = base

    # sym2eid: approved symbol → entrez_id
    g2 = gref.dropna(subset=["symbol", "entrez_id"]).copy()
    g2["entrez_id"] = pd.to_numeric(g2["entrez_id"], errors="coerce").astype("Int64")
    g2["entrez_id"] = g2["entrez_id"].astype(str).str.replace("<NA>", "", regex=False)
    sym2eid: Dict[str, str] = dict(zip(g2["symbol"], g2["entrez_id"]))

    if logger:
        logger.info(
            "Gene reference loaded: %d symbols, %d alias entries.",
            len(approved),
            len(alias_map),
        )

    return gref, alias_map, sym2eid


def _to_approved(sym: str, approved: set, alias_map: Dict[str, str]) -> str:
    s = str(sym).strip()
    return s if s in approved else alias_map.get(s, s)


def _to_matrix_col(sym: str, approved: set, alias_map: Dict[str, str], sym2eid: Dict[str, str]) -> Optional[str]:
    s2 = _to_approved(sym, approved, alias_map)
    eid = sym2eid.get(s2)
    return f"{s2} ({eid})" if eid else None


def _load_selected_genes(genes_path: Path, logger: Optional[Logger] = None) -> List[str]:
    """
    Load selected genes (column 'gene') from InputGenes_selected.csv.
    """
    if logger:
        logger.info("Loading selected genes from %s", genes_path)

    if not genes_path.exists():
        raise FileNotFoundError(f"Selected genes file not found: {genes_path}")

    df = pd.read_csv(genes_path)
    if "gene" not in df.columns:
        for c in df.columns:
            if c.lower() in ["gene", "genes", "symbol"]:
                df = df.rename(columns={c: "gene"})
                break
    if "gene" not in df.columns:
        raise ValueError(f"Selected genes file must contain a 'gene' column. Found: {list(df.columns)}")

    return df["gene"].astype(str).tolist()


def _load_selected_models(models_path: Path, logger: Optional[Logger] = None) -> List[str]:
    """
    Load selected models (column 'ModelID') from Selected_Models.csv.
    """
    if logger:
        logger.info("Loading selected models from %s", models_path)

    if not models_path.exists():
        raise FileNotFoundError(f"Selected models file not found: {models_path}")

    df = pd.read_csv(models_path)
    if "ModelID" not in df.columns:
        raise ValueError(f"Selected models file must contain 'ModelID'. Found: {list(df.columns)}")

    return df["ModelID"].astype(str).tolist()


# ---------------------------------------------------------------------
# Essentiality tagging helper
# ---------------------------------------------------------------------
def interpret_essentiality(x: float, TH_CORE: float, TH_STRONG: float, TH_MOD: float) -> str:
    """
    Map a single ChronosGeneEffect value to a qualitative tag.
    """
    if pd.isna(x):
        return "No data"
    if x < TH_CORE:
        return "Core essential"
    if x < TH_STRONG:
        return "Strong dependency"
    if x < TH_MOD:
        return "Moderate dependency"
    if x < 0:
        return "Weak/Contextual"
    return "Non-essential / growth-suppressive"


# ---------------------------------------------------------------------
# Core: build tidy dependencies
# ---------------------------------------------------------------------
def build_tidy_dependencies(
    output_dir: Path,
    selected_genes_path: Path,
    selected_models_path: Path,
    logger: Optional[Logger] = None,
    TH_CORE: float = -1.0,
    TH_STRONG: float = -0.7,
    TH_MOD: float = -0.3,
) -> DependencyTidyResult:
    """
    Build a tidy dependency + Chronos table for SELECTED genes × SELECTED models.

    Uses:
      - DepMap_Repository/Gene.csv
      - DepMap_Repository/CRISPRGeneEffect.csv
      - DepMap_Repository/CRISPRGeneDependency.csv
      - DepMap_Repository/Model.csv

      - DepMap_Genes/InputGenes_selected.csv
      - DepMap_CellLines/Selected_Models.csv

    Saves:
      - DepMap_Dependencies/Gene_Mapping_InMatrices.csv
      - DepMap_Dependencies/Dependencies_Tidy_SelectedGenes_SelectedModels.csv
    """
    DEPS_OUTDIR = output_dir / "DepMap_Dependencies"
    DEPS_OUTDIR.mkdir(parents=True, exist_ok=True)

    path_gene = DATA_DIR / "Gene.csv"
    path_effect = DATA_DIR / "CRISPRGeneEffect.csv"
    path_dep = DATA_DIR / "CRISPRGeneDependency.csv"
    path_model = DATA_DIR / "Model.csv"

    # 1) Gene ref + alias mapping
    gref, alias_map, sym2eid = _load_gene_reference(path_gene, logger=logger)
    approved = set(gref["symbol"].dropna().astype(str))

    # 2) Selected genes & models
    selected_symbols = _load_selected_genes(selected_genes_path, logger=logger)
    selected_models = _load_selected_models(selected_models_path, logger=logger)
    selected_symbols_norm = [str(s).strip() for s in selected_symbols]

    # 3) Matrix headers
    eff_cols = pd.read_csv(path_effect, nrows=0).columns.tolist()
    dep_cols = pd.read_csv(path_dep, nrows=0).columns.tolist()
    model_col_effect = eff_cols[0]
    model_col_dep = dep_cols[0]

    print(f"✅ Headers: {len(eff_cols)} (effect), {len(dep_cols)} (dependency)")
    if logger:
        logger.info(
            "Effect matrix has %d columns, dependency matrix has %d columns.",
            len(eff_cols),
            len(dep_cols),
        )

    # 4) Map symbols -> approved -> matrix columns
    map_df = pd.DataFrame({"Gene": selected_symbols_norm})
    map_df["Approved"] = map_df["Gene"].map(lambda s: _to_approved(s, approved, alias_map))
    map_df["MatrixCol"] = map_df["Approved"].map(lambda s: _to_matrix_col(s, approved, alias_map, sym2eid))
    map_df["in_effect"] = map_df["MatrixCol"].isin(eff_cols)
    map_df["in_dep"] = map_df["MatrixCol"].isin(dep_cols)

    usable_cols = sorted(
        map_df.loc[map_df["in_effect"], "MatrixCol"].dropna().unique().tolist()
    )
    if len(usable_cols) == 0:
        raise AssertionError("None of the selected genes were found in CRISPRGeneEffect.csv headers.")

    print(f"✅ Mapped {len(usable_cols)} gene columns in Chronos matrix.")
    if logger:
        logger.info("Mapped %d usable gene columns in effect matrix.", len(usable_cols))

    mapping_path = DEPS_OUTDIR / "Gene_Mapping_InMatrices.csv"
    map_df.to_csv(mapping_path, index=False)

    # 5) Read subset of effect & dependency matrices
    dtype_eff = {model_col_effect: "string"} | {c: "float32" for c in usable_cols}
    effect = pd.read_csv(path_effect, usecols=[model_col_effect] + usable_cols, dtype=dtype_eff)
    effect = effect.rename(columns={model_col_effect: "ModelID"})
    effect = effect[effect["ModelID"].isin(selected_models)].copy()

    use_dep_cols = [model_col_dep] + [c for c in usable_cols if c in dep_cols]
    dtype_dep = {model_col_dep: "string"} | {c: "float32" for c in use_dep_cols if c != model_col_dep}
    dep = pd.read_csv(path_dep, usecols=use_dep_cols, dtype=dtype_dep)
    dep = dep.rename(columns={model_col_dep: "ModelID"})
    dep = dep[dep["ModelID"].isin(selected_models)].copy()

    # 6) Melt to long format
    eff_long = effect.melt(id_vars="ModelID", var_name="GeneCol", value_name="ChronosGeneEffect")
    dep_long = dep.melt(id_vars="ModelID", var_name="GeneCol", value_name="DependencyProbability")
    eff_long["Gene"] = eff_long["GeneCol"].str.replace(r" \(\d+\)$", "", regex=True)
    dep_long["Gene"] = dep_long["GeneCol"].str.replace(r" \(\d+\)$", "", regex=True)

    # 7) Model metadata
    model_meta = pd.read_csv(
        path_model,
        usecols=["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"],
        low_memory=False,
    )
    tidy = (
        eff_long
        .merge(
            dep_long[["ModelID", "GeneCol", "DependencyProbability"]],
            on=["ModelID", "GeneCol"],
            how="left",
        )
        .merge(model_meta, on="ModelID", how="left")
    )

    print(f"✅ Tidy shape: {tidy.shape}")
    display(tidy.head(10))
    if logger:
        logger.info("Tidy dependency table shape: %s", tidy.shape)

    # 8) Essentiality tagging (per row)
    tidy["EssentialityTag"] = tidy["ChronosGeneEffect"].apply(
        lambda x: interpret_essentiality(x, TH_CORE, TH_STRONG, TH_MOD)
    )

    tidy_path = DEPS_OUTDIR / "Dependencies_Tidy_SelectedGenes_SelectedModels.csv"
    tidy.to_csv(tidy_path, index=False)
    print("💾 Saved tidy:", tidy_path)

    if logger:
        logger.info(
            "Saved tidy dependency table to %s for %d genes × %d models.",
            tidy_path,
            len(selected_symbols_norm),
            len(selected_models),
        )

    return DependencyTidyResult(
        mapping_path=mapping_path,
        tidy_path=tidy_path,
        n_selected_genes=len(selected_symbols_norm),
        n_selected_models=len(selected_models),
        n_rows=len(tidy),
        thresholds={"TH_CORE": TH_CORE, "TH_STRONG": TH_STRONG, "TH_MOD": TH_MOD},
    )


# ---------------------------------------------------------------------
# Plotting helpers (boxplot + TOPN barplots)
# ---------------------------------------------------------------------
def boxplot_by_lineage(
    df: pd.DataFrame,
    gene: str,
    TH_STRONG: float = -0.7,
    min_n: int = 5,
    show: bool = True,
) -> None:
    """
    Boxplot of Chronos gene effect by lineage for a given gene.
    Mirrors your original Colab code.
    """
    gg = df[(df["Gene"] == gene) & df["ChronosGeneEffect"].notna()].copy()
    if gg.empty:
        print(f"No data points for {gene}.")
        return

    counts = gg["OncotreeLineage"].value_counts()
    keep = counts[counts >= int(min_n)].index
    gg = gg[gg["OncotreeLineage"].isin(keep)]
    if gg.empty:
        print(f"No lineage with ≥{min_n} models for {gene}.")
        return

    order = (
        gg.groupby("OncotreeLineage")["ChronosGeneEffect"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    gg["OncotreeLineage"] = pd.Categorical(gg["OncotreeLineage"], categories=order, ordered=True)
    gg = gg.sort_values(["OncotreeLineage"])

    plt.figure(figsize=(10, 6))
    gg.boxplot(column="ChronosGeneEffect", by="OncotreeLineage", grid=False, vert=True)
    plt.gca().set_xticklabels(order, rotation=60, ha="right")
    plt.axhline(TH_STRONG, linestyle="--", linewidth=1)
    plt.title(f"{gene} — Chronos gene effect by lineage")
    plt.suptitle("")
    plt.ylabel("Chronos effect (more negative = more essential)")
    plt.xlabel("Oncotree lineage")
    plt.tight_layout()

    if show:
        plt.show()
    plt.close()


def plot_top_dependents(
    output_dir: Path,
    tidy: pd.DataFrame,
    top_n: int = 20,
    TH_STRONG: float = -0.7,
    show: bool = False,
) -> None:
    """
    For each gene, plot top N most dependent cell lines (ChronosGeneEffect lowest).
    Saves PNG + HTML; optionally shows figures.
    """
    DEPS_OUTDIR = output_dir / "DepMap_Dependencies"
    genes = sorted(tidy["Gene"].dropna().unique())
    for g in genes:
        gg = tidy[(tidy["Gene"] == g) & tidy["ChronosGeneEffect"].notna()].copy()
        if gg.empty:
            continue
        gg = gg.sort_values("ChronosGeneEffect").head(int(top_n))
        if gg.empty:
            continue

        gg["label"] = gg["CellLineName"] + " (" + gg["OncotreeLineage"].astype(str) + ")"

        fig_top = px.bar(
            gg.iloc[::-1],
            x="ChronosGeneEffect",
            y="label",
            orientation="h",
            title=f"Top {len(gg)} {g}-dependent cell lines (lower = stronger dependency)",
            labels={"label": "Cell line", "ChronosGeneEffect": "Chronos Gene Effect"},
            height=520,
        )
        fig_top.add_vline(x=TH_STRONG, line_dash="dash")
        fig_top.update_layout(margin=dict(l=10, r=10, t=50, b=10))

        if show:
            fig_top.show()

        png_path = DEPS_OUTDIR / "figs" / f"top_dependents_{g}.png"
        html_path = DEPS_OUTDIR / "html" / f"top_dependents_{g}.html"
        save_plotly(fig_top, png_path, html_path, scale=2)

        print(f"💾 Saved top-dependent plot for {g} -> {png_path}")


# ---------------------------------------------------------------------
# Gene-level summary & per-gene top dependents
# ---------------------------------------------------------------------
def summarize_gene_essentiality(
    tidy: pd.DataFrame,
    output_dir: Path,
    TH_CORE: float,
    TH_STRONG: float,
    TH_MOD: float,
    top_per_gene: int = 2,
    min_prob: Optional[float] = None,
) -> Dict[str, object]:
    """
    Implements your "gene-level summary + top dependents" block:

      - Dependencies_GeneSummary_SelectedModels.csv
      - gene_summary_median_effect (PNG + HTML)
      - PerGene_TopDependents_long.csv
      - PerGene_TopDependents_wide.csv
    """
    DEPS_OUTDIR = output_dir / "DepMap_Dependencies"
    # --- Gene-level summary (median-based essentiality) ---
    summary = (
        tidy.dropna(subset=["ChronosGeneEffect"])
            .groupby("Gene")
            .agg(
                n_models=("ModelID", "nunique"),
                median_effect=("ChronosGeneEffect", "median"),
                q10=("ChronosGeneEffect", lambda s: s.quantile(0.10)),
                q90=("ChronosGeneEffect", lambda s: s.quantile(0.90)),
                n_prob50=("DependencyProbability", lambda s: int((s.fillna(0) >= 0.5).sum())),
                n_strong_lt_1=("ChronosGeneEffect", lambda s: int((s < TH_STRONG).sum())),
            )
            .reset_index()
    )

    summary["BiologicalTag"] = summary["median_effect"].apply(
        lambda m: interpret_essentiality(m, TH_CORE, TH_STRONG, TH_MOD)
    )

    display(summary.sort_values("median_effect").head(20))

    summary_csv_path = DEPS_OUTDIR / "Dependencies_GeneSummary_SelectedModels.csv"
    summary.to_csv(summary_csv_path, index=False)
    print("💾 Saved:", summary_csv_path)

    # Bar plot of median essentiality per gene
    fig_sum = px.bar(
        summary.sort_values("median_effect"),
        x="median_effect",
        y="Gene",
        orientation="h",
        title="Median essentiality across selected models",
        labels={"median_effect": "Median Chronos gene effect"},
    )
    fig_sum.add_vline(x=TH_STRONG, line_dash="dash")
    fig_sum.update_layout(height=700, margin=dict(l=10, r=10, t=50, b=10))

    # Save figure
    png_sum = DEPS_OUTDIR / "figs" / "gene_summary_median_effect.png"
    html_sum = DEPS_OUTDIR / "html" / "gene_summary_median_effect.html"
    save_plotly(fig_sum, png_sum, html_sum, scale=2)

    # --- Per-gene top dependents tables (LONG & WIDE) ---
    df_top_source = tidy.dropna(subset=["ChronosGeneEffect"]).copy()

    if min_prob is not None and "DependencyProbability" in df_top_source.columns:
        before = len(df_top_source)
        df_top_source = df_top_source[df_top_source["DependencyProbability"].fillna(0) >= float(min_prob)]
        print(f"ℹ️ MIN_PROB ≥ {min_prob}: {before} → {len(df_top_source)} rows retained")

    df_top_source["rank_within_gene"] = (
        df_top_source.groupby("Gene")["ChronosGeneEffect"]
                     .rank(method="first", ascending=True)
    )

    top_hits_long = (
        df_top_source.sort_values(["Gene", "rank_within_gene"])
                     .groupby("Gene", as_index=False)
                     .head(int(top_per_gene))
                     .loc[:, [
                         "Gene", "rank_within_gene", "ModelID", "CellLineName",
                         "OncotreeLineage", "OncotreePrimaryDisease",
                         "ChronosGeneEffect", "DependencyProbability",
                     ]]
                     .rename(columns={
                         "rank_within_gene": "TopRank",
                         "ChronosGeneEffect": "GeneEffect",
                         "DependencyProbability": "DepProb",
                     })
    )
    top_hits_long["EssentialityTag"] = top_hits_long["GeneEffect"].apply(
        lambda x: interpret_essentiality(x, TH_CORE, TH_STRONG, TH_MOD)
    )

    def _flatten_cols(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("TopRank").head(int(top_per_gene))
        out = {"Gene": g["Gene"].iloc[0]}
        for i, (_, r) in enumerate(g.iterrows(), start=1):
            out.update({
                f"Top{i}_ModelID": r["ModelID"],
                f"Top{i}_CellLine": r["CellLineName"],
                f"Top{i}_Lineage": r["OncotreeLineage"],
                f"Top{i}_Disease": r["OncotreePrimaryDisease"],
                f"Top{i}_GeneEffect": r["GeneEffect"],
                f"Top{i}_DepProb": r["DepProb"],
                f"Top{i}_EssentialityTag": interpret_essentiality(
                    r["GeneEffect"], TH_CORE, TH_STRONG, TH_MOD
                ),
            })
        return pd.Series(out)

    top_hits_wide = (
        top_hits_long.groupby("Gene", as_index=False)
                     .apply(_flatten_cols)
                     .reset_index(drop=True)
    )

    display(top_hits_long.head(20))
    display(top_hits_wide.head(20))

    top_long_csv = DEPS_OUTDIR / "PerGene_TopDependents_long.csv"
    top_wide_csv = DEPS_OUTDIR / "PerGene_TopDependents_wide.csv"
    top_hits_long.to_csv(top_long_csv, index=False)
    top_hits_wide.to_csv(top_wide_csv, index=False)
    print("💾 Saved:", top_long_csv)
    print("💾 Saved:", top_wide_csv)

    return {
        "summary_df": summary,
        "summary_csv": summary_csv_path,
        "top_long": top_hits_long,
        "top_long_csv": top_long_csv,
        "top_wide": top_hits_wide,
        "top_wide_csv": top_wide_csv,
    }


# ---------------------------------------------------------------------
# Essentiality tables (per-model, by median, by any-model)
# ---------------------------------------------------------------------
def build_essentiality_tables(
    tidy: pd.DataFrame,
    output_dir: Path,
    TH_CORE: float,
    TH_STRONG: float,
    TH_MOD: float,
    min_prob: Optional[float] = None,
    enforce_min_prob_per_model: bool = False,
    essential_rule: str = "strong_or_better",
    summary_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Implements your STEP 8b block:
      1) GeneEssentiality_PerModel.csv
      2) GeneEssentiality_ByMedian.csv
      3) GeneEssentiality_ByAnyModel.csv
      + tag counts and essential gene lists.
    """
    DEPS_OUTDIR = output_dir / "DepMap_Dependencies"
    # -----------------------------
    # Required columns
    # -----------------------------
    per_cols = [
        "Gene", "ModelID", "CellLineName", "OncotreeLineage",
        "OncotreePrimaryDisease", "ChronosGeneEffect", "DependencyProbability",
    ]
    missing_cols = [c for c in per_cols if c not in tidy.columns]
    if missing_cols:
        raise KeyError(f"tidy is missing expected columns: {missing_cols}")

    per_model = tidy.loc[:, per_cols].copy()

    # Optional probability threshold
    if min_prob is not None:
        per_model["DepProbPass"] = per_model["DependencyProbability"].fillna(0) >= float(min_prob)
        if enforce_min_prob_per_model:
            before = len(per_model)
            per_model = per_model[per_model["DepProbPass"]].copy()
            print(f"ℹ️ Enforced DepProb ≥ {min_prob} in per-model table: {before} → {len(per_model)} rows")
    else:
        per_model["DepProbPass"] = True

    # Per-row essentiality tag
    per_model["EssentialityTag"] = per_model["ChronosGeneEffect"].apply(
        lambda x: interpret_essentiality(x, TH_CORE, TH_STRONG, TH_MOD)
    )

    per_model_csv = DEPS_OUTDIR / "GeneEssentiality_PerModel.csv"
    per_model.to_csv(per_model_csv, index=False)
    print("💾 Saved per-model essentiality:", per_model_csv)

    # -----------------------------
    # 2) Gene-level (median) essentiality
    # -----------------------------
    if summary_df is None:
        summary_df = (
            tidy.dropna(subset=["ChronosGeneEffect"])
                .groupby("Gene")
                .agg(
                    n_models=("ModelID", "nunique"),
                    median_effect=("ChronosGeneEffect", "median"),
                    q10=("ChronosGeneEffect", lambda s: s.quantile(0.10)),
                    q90=("ChronosGeneEffect", lambda s: s.quantile(0.90)),
                    n_prob50=("DependencyProbability", lambda s: int((s.fillna(0) >= 0.5).sum())),
                    n_strong_lt_1=("ChronosGeneEffect", lambda s: int((s < TH_STRONG).sum())),
                )
                .reset_index()
        )
        summary_df["BiologicalTag"] = summary_df["median_effect"].apply(
            lambda m: interpret_essentiality(m, TH_CORE, TH_STRONG, TH_MOD)
        )

    # Choose rule
    if essential_rule == "strong_or_better":
        essential_mask_median = summary_df["median_effect"] < TH_STRONG
        essential_label_median = "Essential by median (Core+Strong)"
    elif essential_rule == "moderate_or_better":
        essential_mask_median = summary_df["median_effect"] < TH_MOD
        essential_label_median = "Essential by median (Core+Strong+Moderate)"
    elif essential_rule == "any_negative":
        essential_mask_median = summary_df["median_effect"] < 0
        essential_label_median = "Essential by median (any negative median)"
    else:
        raise ValueError(f"Unknown essential_rule: {essential_rule}")

    gene_by_median = summary_df.copy()
    gene_by_median["IsEssential_byMedianRule"] = essential_mask_median.values

    gene_by_median_csv = DEPS_OUTDIR / "GeneEssentiality_ByMedian.csv"
    gene_by_median.to_csv(gene_by_median_csv, index=False)
    print("💾 Saved gene-level (median) essentiality:", gene_by_median_csv)
    print(f"📌 {essential_label_median}: {int(essential_mask_median.sum())}/{len(gene_by_median)} genes")

    # -----------------------------
    # 3) Gene-level “any-model” roll-up
    # -----------------------------
    any_strong = (
        per_model.assign(is_strong=lambda d: d["ChronosGeneEffect"] < TH_STRONG)
                 .groupby("Gene", as_index=False)["is_strong"].any()
                 .rename(columns={"is_strong": "IsEssential_anyStrong"})
    )

    min_effect = (
        per_model.groupby("Gene", as_index=False)["ChronosGeneEffect"].min()
                 .rename(columns={"ChronosGeneEffect": "min_effect"})
    )
    min_effect["MinEffectTag"] = min_effect["min_effect"].apply(
        lambda x: interpret_essentiality(x, TH_CORE, TH_STRONG, TH_MOD)
    )

    gene_any = (
        any_strong
          .merge(min_effect, on="Gene", how="left")
          .merge(
              summary_df.loc[:, ["Gene", "n_models", "median_effect", "BiologicalTag"]],
              on="Gene", how="left"
          )
    )

    gene_any_csv = DEPS_OUTDIR / "GeneEssentiality_ByAnyModel.csv"
    gene_any.to_csv(gene_any_csv, index=False)
    print("💾 Saved gene-level (any-model roll-up):", gene_any_csv)

    # -----------------------------
    # 4) Category counts + essential rollups
    # -----------------------------
    tag_counts = (
        summary_df["BiologicalTag"]
               .value_counts(dropna=False)
               .rename_axis("BiologicalTag")
               .reset_index(name="n_genes")
               .sort_values("n_genes", ascending=False)
    )

    counts_csv = DEPS_OUTDIR / "Essentiality_Tag_Counts.csv"
    tag_counts.to_csv(counts_csv, index=False)

    essential_genes_median = gene_by_median.loc[
        gene_by_median["IsEssential_byMedianRule"], "Gene"
    ].sort_values().tolist()
    essential_list_csv = DEPS_OUTDIR / "Essential_Genes_List_byMedian.csv"
    pd.DataFrame({"Gene": essential_genes_median}).to_csv(essential_list_csv, index=False)

    summary_flags = summary_df.copy()
    summary_flags["IsEssential_byMedianRule"] = gene_by_median["IsEssential_byMedianRule"].values
    summary_flags_csv = DEPS_OUTDIR / "Dependencies_GeneSummary_withEssentialFlag.csv"
    summary_flags.to_csv(summary_flags_csv, index=False)

    print("\n================ Essentiality summary ================")
    print(f"Total genes evaluated: {len(summary_df)}")
    print("\nPer-category (BiologicalTag) counts:")
    for _, row in tag_counts.iterrows():
        print(f"  • {row['BiologicalTag']:<34} : {row['n_genes']}")
    print(f"\n{essential_label_median}: {len(essential_genes_median)}/{len(summary_df)} genes")
    print(f"Essential by ANY model (Chronos < TH_STRONG): {int(gene_any['IsEssential_anyStrong'].sum())}/{len(gene_any)} genes")
    print("======================================================\n")

    print("💾 Saved tag counts:", counts_csv)
    print("💾 Saved essential (median-rule) list:", essential_list_csv)
    print("💾 Saved flagged summary:", summary_flags_csv)

    return {
        "per_model_csv": per_model_csv,
        "gene_by_median_csv": gene_by_median_csv,
        "gene_any_csv": gene_any_csv,
        "tag_counts_csv": counts_csv,
        "essential_list_csv": essential_list_csv,
        "summary_flags_csv": summary_flags_csv,
    }
