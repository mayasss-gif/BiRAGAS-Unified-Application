# src/depmap_celllines_uif.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
from logging import Logger




# Try to get a nice display() if running in notebook; otherwise fallback
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)


# -------------------------------------------------------------------
# PATH SETUP (DepMap repository style)
# -------------------------------------------------------------------
from .constants import DATA_DIR




@dataclass
class DepMapCelllineContext:
    """
    Container for core DepMap objects from this pipeline.
    """
    paths: Dict[str, Path]
    models_raw: pd.DataFrame
    models_core: pd.DataFrame
    lineage_counts: Optional[pd.DataFrame]
    disease_counts: Optional[pd.DataFrame]
    master: pd.DataFrame
    effect_df: pd.DataFrame
    dep_df: pd.DataFrame
    expr_df: pd.DataFrame
    cnv_df: pd.DataFrame


def _build_paths(output_dir: Path) -> Dict[str, Path]:
    """
    Build the canonical set of input paths for DepMap metadata + matrices.
    """
    USER_GENE_LIST = output_dir / "DepMap_Genes" / "InputGenes_selected.csv"
    paths = {
        "Model.csv": DATA_DIR / "Model.csv",
        "Gene.csv": DATA_DIR / "Gene.csv",
        "CRISPRGeneEffect.csv": DATA_DIR / "CRISPRGeneEffect.csv",      # Chronos
        "CRISPRGeneDependency.csv": DATA_DIR / "CRISPRGeneDependency.csv",  # dep prob
        "Expression (TPM log1p)": DATA_DIR
        / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "Copy Number (WGS)": DATA_DIR / "OmicsCNGeneWGS.csv",
        # Optional user gene list produced by Step 2
        "Input gene list": USER_GENE_LIST,
    }
    return paths


def _check_files(paths: Dict[str, Path], logger: Optional[Logger] = None) -> None:
    """
    STEP 1 – Check that all required files exist.
    Raises FileNotFoundError if any are missing (except optional gene list).
    """
    missing = []
    print("\n📁 Checking required DepMap files:")
    for label, p in paths.items():
        status = "OK" if p.exists() else "MISSING"
        print(f"{label:<28} -> {p}  [{status}]")
        if logger:
            logger.info("%s -> %s [%s]", label, p, status)

        # Input gene list is optional; others are required
        if not p.exists() and label not in {"Input gene list"}:
            missing.append(label)

    if missing:
        msg = f"Missing required files: {missing}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError("❌ " + msg)
    else:
        print("\n✅ All required core DepMap files present.")
        if logger:
            logger.info("All required core DepMap files present.")


def _read_matrix_with_model_index(path: Path) -> pd.DataFrame:
    """
    STEP 4 – Read a matrix where rows correspond to cell lines.
    Tries ModelID / DepMap_ID, otherwise uses the first column as ID.
    Returns a DataFrame with index name 'ModelID'.
    """
    df = pd.read_csv(path, low_memory=False)
    candidate_id_cols = ["ModelID", "DepMap_ID"]

    chosen = None
    for col in candidate_id_cols:
        if col in df.columns:
            df = df.set_index(col)
            chosen = col
            break

    if chosen is None:
        first_col = df.columns[0]
        df = df.set_index(first_col)
        chosen = first_col

    df.index.name = "ModelID"
    print(
        f"📂 Loaded {path.name} with index column '{chosen}' "
        f"({df.shape[0]} cell lines × {df.shape[1]} columns)"
    )
    return df


def run_depmap_cellline_setup(
    output_dir: Path,
    logger: Optional[Logger] = None,
) -> DepMapCelllineContext:
    """
    Main entrypoint for DepMap cell line setup.

    0) Build paths and ensure output dirs
    1) Check presence of Model, Gene, Effect, Dependency, Expression, CNV
    2) Load full Model.csv, build core metadata, save
    3) Build lineage & primary disease summaries (+ plots & CSVs)
    4) Load effect/dep/expression/CNV matrices with ModelID index
    5) Build master cell-line table with flags for each data layer

    Returns a DepMapCelllineContext with all relevant DataFrames.
    """
    OUTDIR = output_dir / "DepMap_CellLines"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("=== Step 03: DepMap cell line setup ===")
        logger.info("Cell line output directory: %s", OUTDIR)

    paths = _build_paths(output_dir)
    _check_files(paths, logger=logger)

    # ------------------------------------------------------------------
    # STEP 2 – Load ALL model metadata (cell line centric)
    # ------------------------------------------------------------------
    path_model = paths["Model.csv"]
    if logger:
        logger.info("Loading Model.csv: %s", path_model)
    models_raw = pd.read_csv(path_model, low_memory=False)

    core_cols = [
        "ModelID",
        "CellLineName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
    ]
    core_cols_exist = [c for c in core_cols if c in models_raw.columns]
    models_core = models_raw[core_cols_exist].copy()

    # Clean missing values
    if "OncotreeLineage" in models_core.columns:
        models_core["OncotreeLineage"] = models_core["OncotreeLineage"].fillna(
            "(missing)"
        )
    if "OncotreePrimaryDisease" in models_core.columns:
        models_core["OncotreePrimaryDisease"] = models_core[
            "OncotreePrimaryDisease"
        ].fillna("(missing)")

    print(f"\n🧬 Total models (rows in Model.csv): {len(models_core)}")
    if "OncotreeLineage" in models_core.columns:
        print(
            f"🧬 Unique Oncotree lineages: "
            f"{models_core['OncotreeLineage'].nunique()}"
        )
    if "OncotreePrimaryDisease" in models_core.columns:
        print(
            f"🧬 Unique primary diseases: "
            f"{models_core['OncotreePrimaryDisease'].nunique()}"
        )

    if logger:
        logger.info("Total models: %d", len(models_core))

    # Save metadata versions
    models_raw.to_csv(
        OUTDIR / "CellLines_Metadata_Full_ModelTable.csv", index=False
    )
    models_core.to_csv(OUTDIR / "CellLines_Metadata_Core.csv", index=False)

    # Use ModelID as index if available
    if "ModelID" in models_core.columns:
        models_core = models_core.set_index("ModelID")
        models_core.index.name = "ModelID"
    else:
        print("⚠️ 'ModelID' column missing in Model.csv – using row index as ID.")
        models_core.index.name = "ModelID"

    # ------------------------------------------------------------------
    # STEP 3 – High-level lineage / disease summaries
    # ------------------------------------------------------------------
    lineage_counts = None
    disease_counts = None

    if "OncotreeLineage" in models_core.columns:
        lineage_counts = (
            models_core["OncotreeLineage"]
            .value_counts()
            .rename_axis("OncotreeLineage")
            .reset_index(name="n_models")
        )
        fig1 = px.bar(
            lineage_counts,
            x="n_models",
            y="OncotreeLineage",
            orientation="h",
            title="Oncotree Lineages by model count",
            height=700,
            text="n_models",
        )
        fig1.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=120, r=20, t=60, b=40),
        )
        fig1.update_traces(textposition="auto")
        fig1.write_html(str(OUTDIR / "Lineage_Counts_BarPlot.html"))
        lineage_counts.to_csv(
            OUTDIR / "Summary_Lineage_Counts.csv", index=False
        )
        print("💾 Saved lineage counts (CSV + HTML bar plot).")
        if logger:
            logger.info("Saved lineage counts and bar plot.")

    if "OncotreePrimaryDisease" in models_core.columns:
        disease_counts = (
            models_core["OncotreePrimaryDisease"]
            .value_counts()
            .rename_axis("OncotreePrimaryDisease")
            .reset_index(name="n_models")
        )
        fig2 = px.bar(
            disease_counts,
            x="n_models",
            y="OncotreePrimaryDisease",
            orientation="h",
            title="Primary Diseases by model count",
            height=700,
            text="n_models",
        )
        fig2.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=120, r=20, t=60, b=40),
        )
        fig2.update_traces(textposition="auto")
        fig2.write_html(str(OUTDIR / "PrimaryDisease_Counts_BarPlot.html"))
        disease_counts.to_csv(
            OUTDIR / "Summary_PrimaryDisease_Counts.csv", index=False
        )
        print("💾 Saved primary disease counts (CSV + HTML bar plot).")
        if logger:
            logger.info("Saved primary disease counts and bar plot.")

    print("💾 Saved lineage & primary disease summaries.")

    # ------------------------------------------------------------------
    # STEP 4 – Load matrices with ModelID index
    # ------------------------------------------------------------------
    if logger:
        logger.info("Loading CRISPR effect, dependency, expression, CNV matrices.")

    effect_df = _read_matrix_with_model_index(
        paths["CRISPRGeneEffect.csv"]
    )
    dep_df = _read_matrix_with_model_index(paths["CRISPRGeneDependency.csv"])
    expr_df = _read_matrix_with_model_index(
        paths["Expression (TPM log1p)"]
    )
    cnv_df = _read_matrix_with_model_index(paths["Copy Number (WGS)"])

    # ------------------------------------------------------------------
    # STEP 5 – MASTER table with ALL cell lines (no subsetting)
    # ------------------------------------------------------------------
    ids_model = set(models_core.index)
    ids_effect = set(effect_df.index)
    ids_dep = set(dep_df.index)
    ids_expr = set(expr_df.index)
    ids_cnv = set(cnv_df.index)

    common_ids = ids_model & ids_effect & ids_dep & ids_expr & ids_cnv

    print(f"\n📊 Cell lines in Model.csv:          {len(ids_model)}")
    print(f"   Cell lines with CRISPR effect:    {len(ids_effect)}")
    print(f"   Cell lines with CRISPR dependency:{len(ids_dep)}")
    print(f"   Cell lines with expression:       {len(ids_expr)}")
    print(f"   Cell lines with CNV:              {len(ids_cnv)}")
    print(f"   ➜ Cell lines with ALL datasets:   {len(common_ids)}")

    if logger:
        logger.info("Cell lines in Model.csv: %d", len(ids_model))
        logger.info("Cell lines with CRISPR effect: %d", len(ids_effect))
        logger.info(
            "Cell lines with CRISPR dependency: %d", len(ids_dep)
        )
        logger.info("Cell lines with expression: %d", len(ids_expr))
        logger.info("Cell lines with CNV: %d", len(ids_cnv))
        logger.info(
            "Cell lines with ALL datasets (intersection): %d",
            len(common_ids),
        )

    # Keep ALL models; just add flags
    master = models_core.copy()
    master["has_CRISPR_effect"] = master.index.isin(ids_effect)
    master["has_CRISPR_dependency"] = master.index.isin(ids_dep)
    master["has_expression_TPMlog1p"] = master.index.isin(ids_expr)
    master["has_CNV_WGS"] = master.index.isin(ids_cnv)
    master["in_all_layers"] = master.index.isin(common_ids)

    # Ensure CellLineName is present
    if (
        "CellLineName" not in master.columns
        and "CellLineName" in models_raw.columns
        and "ModelID" in models_raw.columns
    ):
        name_map = models_raw.set_index("ModelID")["CellLineName"].to_dict()
        master["CellLineName"] = master.index.map(name_map)

    master_path = OUTDIR / "CellLines_Master_AllModels.csv"
    master.to_csv(master_path)
    print("\n✅ Built and saved master cell line table with ALL models.")
    print(f"   -> {master_path}")
    if logger:
        logger.info("Saved master cell line table: %s", master_path)

    # ID–name list
    if "CellLineName" in master.columns:
        id_name = master.reset_index()[["ModelID", "CellLineName"]]
    else:
        id_name = master.reset_index()[["ModelID"]]
    id_name_path = OUTDIR / "CellLines_ID_Name_List_All.csv"
    id_name.to_csv(id_name_path, index=False)
    print(f"💾 Saved ID–name list -> {id_name_path}")
    if logger:
        logger.info("Saved ID–name list: %s", id_name_path)

    # Return context for further analysis
    return DepMapCelllineContext(
        paths=paths,
        models_raw=models_raw,
        models_core=models_core,
        lineage_counts=lineage_counts,
        disease_counts=disease_counts,
        master=master,
        effect_df=effect_df,
        dep_df=dep_df,
        expr_df=expr_df,
        cnv_df=cnv_df,
    )


# -------------------------------------------------------------------
# SEARCH HELPERS (you can call them from scripts/notebooks)
# -------------------------------------------------------------------
def search_cell_lines(
    master: pd.DataFrame,
    name_substring: Optional[str] = None,
    lineage: Optional[str] = None,
    primary_disease: Optional[str] = None,
    require_effect: Optional[bool] = None,
    require_dep: Optional[bool] = None,
    require_expr: Optional[bool] = None,
    require_cnv: Optional[bool] = None,
    require_all_layers: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Flexible filter over ALL cell lines in `master`.
    Returns a filtered DataFrame (no row limit).
    """
    df = master.copy()

    # Name substring (case-insensitive)
    if name_substring:
        if "CellLineName" in df.columns:
            df = df[
                df["CellLineName"]
                .astype(str)
                .str.contains(name_substring, case=False, na=False)
            ]

    # Lineage filter
    if lineage is not None and "OncotreeLineage" in df.columns:
        df = df[df["OncotreeLineage"] == lineage]

    # Primary disease filter
    if primary_disease is not None and "OncotreePrimaryDisease" in df.columns:
        df = df[df["OncotreePrimaryDisease"] == primary_disease]

    # Data-layer flags
    if require_effect is True and "has_CRISPR_effect" in df.columns:
        df = df[df["has_CRISPR_effect"]]
    if require_dep is True and "has_CRISPR_dependency" in df.columns:
        df = df[df["has_CRISPR_dependency"]]
    if require_expr is True and "has_expression_TPMlog1p" in df.columns:
        df = df[df["has_expression_TPMlog1p"]]
    if require_cnv is True and "has_CNV_WGS" in df.columns:
        df = df[df["has_CNV_WGS"]]
    if require_all_layers is True and "in_all_layers" in df.columns:
        df = df[df["in_all_layers"]]

    sort_cols = [
        c
        for c in ["OncotreeLineage", "OncotreePrimaryDisease", "CellLineName"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)

    print(f"🔎 Found {len(df)} matching cell lines.")
    display(df)
    return df


def search_cell_line_anywhere(master: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Simple search over ModelID and CellLineName.
    """
    df = master.copy()

    mask = pd.Series(False, index=df.index)

    # Match ModelID substring
    mask = mask | df.index.astype(str).str.contains(query, case=False, na=False)

    # Match CellLineName substring (if available)
    if "CellLineName" in df.columns:
        mask = mask | df["CellLineName"].astype(str).str.contains(
            query, case=False, na=False
        )

    hits = df[mask].copy()

    if hits.empty:
        print(f"❌ No cell lines found matching '{query}'.")
    else:
        print(f"✅ Found {len(hits)} cell line(s) matching '{query}':")
        display(hits)

    return hits
