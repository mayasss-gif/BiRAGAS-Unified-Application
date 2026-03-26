# src/depmap_guides_uif.py

from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from logging import Logger


from .constants import DATA_DIR
# ---------------------------------------------------------------------
# Alias map helpers
# ---------------------------------------------------------------------
def _build_alias_map(gene_csv: Path, logger):
    """
    Build HGNC alias → approved symbol mapping from DepMap Gene.csv.
    """
    if not gene_csv.exists():
        raise FileNotFoundError(f"Missing HGNC gene reference file: {gene_csv}")

    logger.info("Building HGNC alias mapping from %s ...", gene_csv)

    header = pd.read_csv(gene_csv, nrows=0).columns.tolist()
    use_cols = [c for c in ["symbol", "entrez_id", "alias_symbol", "prev_symbol"] if c in header]
    genes_ref = pd.read_csv(gene_csv, usecols=use_cols)

    if "symbol" not in genes_ref.columns:
        raise ValueError("Gene.csv must contain a 'symbol' column.")

    approved = set(genes_ref["symbol"].dropna().astype(str))

    for c in ["alias_symbol", "prev_symbol"]:
        if c in genes_ref.columns:
            genes_ref[c] = genes_ref[c].fillna("").astype(str)

    alias_map: dict[str, str] = {}
    for _, r in genes_ref.dropna(subset=["symbol"]).iterrows():
        base = str(r["symbol"]).strip()
        for col in ["alias_symbol", "prev_symbol"]:
            if col in genes_ref.columns:
                for tok in re.split(r"[|;,]", str(r[col])):
                    tok = tok.strip()
                    if tok and tok not in alias_map:
                        alias_map[tok] = base

    logger.info("Alias map entries: %d", len(alias_map))
    return approved, alias_map


def _to_approved(sym: str, approved: set[str], alias_map: dict[str, str]) -> str:
    s = str(sym).strip()
    if s in approved:
        return s
    return alias_map.get(s, s)


# ---------------------------------------------------------------------
# Main: guide-level Avana enrichment
# ---------------------------------------------------------------------
def run_guide_level_enrichment(
    output_dir: Path,
    logger: Logger,
):
    """
    Guide-level Avana enrichment/depletion analysis for the selected genes & models
    from your DepMap pipeline.

    When called from the run driver, `out_root` should be RUN_OUTPUT_DIR, so all
    outputs go under the current run:

        RUN_OUTPUT_DIR / "DepMap_GuideAnalysis"

    Inputs (under out_root):
      - DepMap_Genes/InputGenes_selected.csv
      - DepMap_CellLines/Selected_Models.csv

    DepMap repository (data_dir):
      - Gene.csv
      - ScreenSequenceMap.csv
      - ModelCondition.csv
      - Model.csv
      - AvanaGuideMap.csv
      - AvanaLogfoldChange.csv

    Outputs (under out_root / "DepMap_GuideAnalysis"):
      - CRISPR_GuideLevel_Avana_SelectedModels_long.csv
      - CRISPR_GeneLevel_FromGuides_Avana_SelectedModels.csv
      - CRISPR_GeneLevel_FromGuides_Avana_SelectedModels_byModel.csv
    """

    # --------------------------
    # 0. Setup paths & logger
    # --------------------------


    guides_outdir = output_dir / "DepMap_GuideAnalysis"
    guides_outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Guide-level Avana enrichment started ===")
    logger.info("Output (guides) directory : %s", guides_outdir)
    logger.info("DepMap repository folder  : %s", DATA_DIR)

    # thresholds (optionally from settings.txt)
    GUIDE_DEPLETE_THRESH = float(-1.0)
    GUIDE_ENRICH_THRESH = float(0.5)

    logger.info("Guide depletion threshold  (LFC <=): %s", GUIDE_DEPLETE_THRESH)
    logger.info("Guide enrichment threshold (LFC >=): %s", GUIDE_ENRICH_THRESH)

    # --------------------------
    # 1. Load selected genes & models
    # --------------------------
    genes_sel_path = output_dir / "DepMap_Genes" / "InputGenes_selected.csv"
    models_sel_path = output_dir / "DepMap_CellLines" / "Selected_Models.csv"

    if not genes_sel_path.exists():
        raise FileNotFoundError(f"Missing selected genes file: {genes_sel_path}")
    if not models_sel_path.exists():
        raise FileNotFoundError(f"Missing selected models file: {models_sel_path}")

    genes_sel = pd.read_csv(genes_sel_path)
    genes_sel.columns = [c.lower() for c in genes_sel.columns]
    if "gene" not in genes_sel.columns:
        raise ValueError(
            f"{genes_sel_path} must contain a 'gene' column. Found: {list(genes_sel.columns)}"
        )
    genes_sel["gene"] = genes_sel["gene"].astype(str).str.strip()

    models_sel = pd.read_csv(models_sel_path)
    if "ModelID" not in models_sel.columns:
        raise ValueError("Selected_Models.csv must contain a 'ModelID' column.")
    selected_models = set(models_sel["ModelID"].astype(str).tolist())

    logger.info("Selected genes (rows): %d", len(genes_sel))
    logger.info("Selected models       : %d", len(selected_models))

    # --------------------------
    # 2. Build HGNC alias map
    # --------------------------
    PATH_GENE = DATA_DIR / "Gene.csv"
    approved, alias_map = _build_alias_map(PATH_GENE, logger)

    genes_sel["gene_raw_upper"] = genes_sel["gene"].str.upper().str.strip()
    genes_sel["gene_approved"] = genes_sel["gene"].map(
        lambda x: _to_approved(x, approved, alias_map)
    )
    genes_sel["gene_approved_upper"] = (
        genes_sel["gene_approved"].astype(str).str.upper().str.strip()
    )

    selected_genes_raw = set(genes_sel["gene_raw_upper"].tolist())
    selected_genes_approved = set(genes_sel["gene_approved_upper"].tolist())
    selected_genes_union = selected_genes_raw.union(selected_genes_approved)

    logger.info("Selected genes (raw, unique)      : %d", len(selected_genes_raw))
    logger.info("Selected genes (approved, unique) : %d", len(selected_genes_approved))

    # --------------------------
    # 3. DepMap file paths
    # --------------------------
    PATH_SEQMAP = DATA_DIR / "ScreenSequenceMap.csv"
    PATH_MCOND = DATA_DIR / "ModelCondition.csv"
    PATH_MODEL = DATA_DIR / "Model.csv"
    PATH_GUIDEMAP = DATA_DIR / "AvanaGuideMap.csv"
    PATH_LFC = DATA_DIR / "AvanaLogfoldChange.csv"

    for p in [PATH_SEQMAP, PATH_MCOND, PATH_MODEL, PATH_GUIDEMAP, PATH_LFC]:
        status = "OK" if p.exists() else "MISSING"
        logger.info("%s -> [%s]", p, status)
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # --------------------------
    # 4. Load screen / model metadata
    # --------------------------
    logger.info("Loading ScreenSequenceMap / ModelCondition / Model metadata ...")
    seqmap = pd.read_csv(PATH_SEQMAP, low_memory=False)
    mcond = pd.read_csv(PATH_MCOND, low_memory=False)
    model = pd.read_csv(PATH_MODEL, low_memory=False)

    # cast IDs as strings
    for col in ["SequenceID", "ScreenID", "ModelConditionID", "ModelID"]:
        if col in seqmap.columns:
            seqmap[col] = seqmap[col].astype(str)
    for col in ["ModelConditionID", "ModelID"]:
        if col in mcond.columns:
            mcond[col] = mcond[col].astype(str)
    if "ModelID" in model.columns:
        model["ModelID"] = model["ModelID"].astype(str)

    # filter to your selected models + Avana + 2D screens + QC
    seq_filt = seqmap[seqmap["ModelID"].isin(selected_models)].copy()
    seq_filt = seq_filt[
        seq_filt["Library"].astype(str).str.contains("avana", case=False, na=False)
    ]
    if "ScreenType" in seq_filt.columns:
        seq_filt = seq_filt[seq_filt["ScreenType"].astype(str).str.upper() == "2DS"]
    if "PassesQC" in seq_filt.columns:
        seq_filt = seq_filt[seq_filt["PassesQC"] == True]

    logger.info("Sequences after model/library/QC filters: %d", len(seq_filt))

    # attach model-condition metadata
    mcond_cols = [
        "ModelConditionID",
        "ModelID",
        "CellFormat",
        "GrowthMedia",
        "Morphology",
        "PrescreenTreatmentDrug",
        "PrescreenTreatmentDrugDays",
        "AnchorDrug",
        "AnchorDrugConcentration",
        "AnchorDaysWithDrug",
    ]
    mcond_small = mcond[[c for c in mcond_cols if c in mcond.columns]].copy()

    seq_annot = seq_filt.merge(mcond_small, on=["ModelConditionID", "ModelID"], how="left")

    # attach model-level metadata
    model_cols = [
        "ModelID",
        "CellLineName",
        "StrippedCellLineName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "GrowthPattern",
    ]
    model_small = model[[c for c in model_cols if c in model.columns]].copy()
    seq_annot = seq_annot.merge(model_small, on="ModelID", how="left")

    logger.info("Annotated sequences: %s", seq_annot.shape)

    # --------------------------
    # 5. Load AvanaLogfoldChange subset
    # --------------------------
    needed_seq_ids = sorted(seq_annot["SequenceID"].unique().tolist())
    logger.info(
        "Unique SequenceIDs needed from AvanaLogfoldChange: %d", len(needed_seq_ids)
    )

    lfc_header = pd.read_csv(PATH_LFC, nrows=0)
    lfc_cols = list(lfc_header.columns)

    if "sgRNA" in lfc_cols:
        sg_col = "sgRNA"
    else:
        sg_col = lfc_cols[0]
        logger.info(
            "Column 'sgRNA' not found; treating '%s' as sgRNA ID column in AvanaLogfoldChange.",
            sg_col,
        )

    available_seq_ids = [c for c in lfc_cols if c in needed_seq_ids]
    logger.info(
        "SequenceIDs found in AvanaLogfoldChange: %d", len(available_seq_ids)
    )

    if not available_seq_ids:
        raise ValueError(
            "No overlapping SequenceIDs between ScreenSequenceMap and AvanaLogfoldChange.csv"
        )

    usecols = [sg_col] + available_seq_ids
    logger.info("Loading AvanaLogfoldChange subset (columns=%d) ...", len(usecols))
    avana_lfc = pd.read_csv(PATH_LFC, usecols=usecols, low_memory=False)
    avana_lfc.rename(columns={sg_col: "sgRNA"}, inplace=True)
    avana_lfc["sgRNA"] = avana_lfc["sgRNA"].astype(str)

    lfc_long = avana_lfc.melt(
        id_vars="sgRNA",
        var_name="SequenceID",
        value_name="GuideLFC",
    )
    lfc_long["SequenceID"] = lfc_long["SequenceID"].astype(str)
    logger.info(
        "Long-format guide rows (before NaN filter): %d", len(lfc_long)
    )
    lfc_long = lfc_long[~lfc_long["GuideLFC"].isna()].copy()
    logger.info("Long-format guide rows (non-null LFC): %d", len(lfc_long))

    # --------------------------
    # 6. Attach guide → gene mapping
    # --------------------------
    logger.info("Loading AvanaGuideMap (sgRNA → Gene) ...")
    guide_map = pd.read_csv(PATH_GUIDEMAP, low_memory=False)
    guide_map["sgRNA"] = guide_map["sgRNA"].astype(str)

    guide_map["Gene_raw_full"] = guide_map["Gene"].astype(str).str.strip()
    guide_map["Gene_symbol_only"] = guide_map["Gene_raw_full"].str.replace(
        r"\s*\(\d+\)$", "", regex=True
    )
    guide_map["Gene_symbol_only_upper"] = (
        guide_map["Gene_symbol_only"].str.upper().str.strip()
    )
    guide_map["Gene_approved"] = guide_map["Gene_symbol_only"].map(
        lambda x: _to_approved(x, approved, alias_map)
    )
    guide_map["Gene_approved_upper"] = (
        guide_map["Gene_approved"].astype(str).str.upper().str.strip()
    )

    genes_from_guide_raw = set(guide_map["Gene_symbol_only_upper"].unique())
    genes_from_guide_approved = set(guide_map["Gene_approved_upper"].unique())

    overlap_raw = selected_genes_union.intersection(genes_from_guide_raw)
    overlap_approved = selected_genes_union.intersection(genes_from_guide_approved)

    logger.info("Overlap stats after stripping '(id)' from guide genes:")
    logger.info("  Overlap with raw symbols      : %d", len(overlap_raw))
    logger.info("  Overlap with approved symbols : %d", len(overlap_approved))

    def choose_gene(row):
        ga = row["Gene_approved_upper"]
        gr = row["Gene_symbol_only_upper"]
        if ga in selected_genes_union:
            return ga
        if gr in selected_genes_union:
            return gr
        return ga or gr

    guide_map["GeneUnified"] = guide_map.apply(choose_gene, axis=1)

    lfc_guides = (
        lfc_long.merge(
            guide_map[["sgRNA", "GeneUnified"]],
            on="sgRNA",
            how="left",
        ).rename(columns={"GeneUnified": "Gene"})
    )

    lfc_guides = lfc_guides[lfc_guides["Gene"].isin(selected_genes_union)].copy()
    logger.info(
        "Guide rows after restricting to selected genes: %d", len(lfc_guides)
    )

    # --------------------------
    # 7. Merge with sequence/model metadata
    # --------------------------
    guide_meta = lfc_guides.merge(seq_annot, on="SequenceID", how="left")
    guide_meta = guide_meta[guide_meta["ModelID"].notna()].copy()
    logger.info(
        "Final guide-level rows (selected genes × selected models): %d",
        len(guide_meta),
    )

    guide_out = (
        guides_outdir / "CRISPR_GuideLevel_Avana_SelectedModels_long.csv"
    )
    gene_from_guides_out = (
        guides_outdir / "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels.csv"
    )
    gene_from_guides_by_model_out = (
        guides_outdir
        / "CRISPR_GeneLevel_FromGuides_Avana_SelectedModels_byModel.csv"
    )

    if guide_meta.empty:
        logger.warning(
            "No guide-level rows found for your selected genes in Avana. "
            "Writing empty outputs."
        )
        pd.DataFrame(columns=["sgRNA", "Gene", "GuideLFC"]).to_csv(
            guide_out, index=False
        )
        pd.DataFrame(columns=["Gene"]).to_csv(
            gene_from_guides_out, index=False
        )
        pd.DataFrame(columns=["ModelID", "Gene"]).to_csv(
            gene_from_guides_by_model_out, index=False
        )
        logger.info("Saved EMPTY guide-level table: %s", guide_out)
        logger.info(
            "Saved EMPTY gene-from-guides tables: %s, %s",
            gene_from_guides_out,
            gene_from_guides_by_model_out,
        )
        logger.info(
            "=== Guide-level Avana enrichment finished (no overlapping guides) ==="
        )
        return

    # --------------------------
    # 8. Classify guides + save long table
    # --------------------------
    def classify_guide(lfc):
        if pd.isna(lfc):
            return "NA"
        if lfc <= GUIDE_DEPLETE_THRESH:
            return "Depleted"
        if lfc >= GUIDE_ENRICH_THRESH:
            return "Enriched"
        return "Neutral"

    guide_meta["GuideDirection"] = guide_meta["GuideLFC"].apply(classify_guide)
    dir_counts = guide_meta["GuideDirection"].value_counts(dropna=False)
    logger.info("Guide-level direction counts:\n%s", dir_counts.to_string())

    guide_meta.to_csv(guide_out, index=False)
    logger.info("Saved guide-level table: %s", guide_out)

    # --------------------------
    # 9. Gene-level (overall) summary from guides
    # --------------------------
    gene_from_guides = (
        guide_meta.groupby("Gene", as_index=False)
        .agg(
            n_guides=("sgRNA", "count"),
            mean_LFC=("GuideLFC", "mean"),
            median_LFC=("GuideLFC", "median"),
            min_LFC=("GuideLFC", "min"),
            max_LFC=("GuideLFC", "max"),
        )
    )

    gd_counts = (
        guide_meta.groupby(["Gene", "GuideDirection"])
        .size()
        .reset_index(name="n")
    )
    total_per_gene = (
        gd_counts.groupby("Gene")["n"]
        .sum()
        .reset_index(name="n_total")
    )
    gd_counts = gd_counts.merge(total_per_gene, on="Gene", how="left")
    gd_counts["frac"] = gd_counts["n"] / gd_counts["n_total"].replace(0, np.nan)

    frac_pivot = (
        gd_counts.pivot(index="Gene", columns="GuideDirection", values="frac")
        .fillna(0.0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    gene_from_guides = gene_from_guides.merge(frac_pivot, on="Gene", how="left")

    # rename fraction columns to frac_depleted / frac_enriched
    if "Depleted" in gene_from_guides.columns:
        gene_from_guides["frac_depleted"] = gene_from_guides["Depleted"]
        gene_from_guides = gene_from_guides.drop(columns=["Depleted"])
    else:
        gene_from_guides["frac_depleted"] = 0.0

    if "Enriched" in gene_from_guides.columns:
        gene_from_guides["frac_enriched"] = gene_from_guides["Enriched"]
        gene_from_guides = gene_from_guides.drop(columns=["Enriched"])
    else:
        gene_from_guides["frac_enriched"] = 0.0

    gene_models = (
        guide_meta.groupby("Gene")["ModelID"]
        .nunique()
        .reset_index(name="n_models_with_guides")
    )
    gene_from_guides = gene_from_guides.merge(
        gene_models, on="Gene", how="left"
    )

    gene_from_guides.to_csv(gene_from_guides_out, index=False)
    logger.info(
        "Saved gene-level summary from guides (overall): %s",
        gene_from_guides_out,
    )

    # --------------------------
    # 10. Gene-level per model
    # --------------------------
    gene_from_guides_by_model = (
        guide_meta.groupby(["ModelID", "Gene"], as_index=False)
        .agg(
            n_guides=("sgRNA", "count"),
            mean_LFC=("GuideLFC", "mean"),
            median_LFC=("GuideLFC", "median"),
            min_LFC=("GuideLFC", "min"),
            max_LFC=("GuideLFC", "max"),
        )
    )

    gd_counts_model = (
        guide_meta.groupby(["ModelID", "Gene", "GuideDirection"])
        .size()
        .reset_index(name="n")
    )
    total_per_gene_model = (
        gd_counts_model.groupby(["ModelID", "Gene"])["n"]
        .sum()
        .reset_index(name="n_total")
    )
    gd_counts_model = gd_counts_model.merge(
        total_per_gene_model, on=["ModelID", "Gene"], how="left"
    )
    gd_counts_model["frac"] = (
        gd_counts_model["n"] / gd_counts_model["n_total"].replace(0, np.nan)
    )

    frac_pivot_model = (
        gd_counts_model.pivot(
            index=["ModelID", "Gene"],
            columns="GuideDirection",
            values="frac",
        )
        .fillna(0.0)
        .reset_index()
    )

    gene_from_guides_by_model = gene_from_guides_by_model.merge(
        frac_pivot_model, on=["ModelID", "Gene"], how="left"
    )

    if "Depleted" in gene_from_guides_by_model.columns:
        gene_from_guides_by_model["frac_depleted"] = gene_from_guides_by_model[
            "Depleted"
        ]
        gene_from_guides_by_model = gene_from_guides_by_model.drop(
            columns=["Depleted"]
        )
    else:
        gene_from_guides_by_model["frac_depleted"] = 0.0

    if "Enriched" in gene_from_guides_by_model.columns:
        gene_from_guides_by_model["frac_enriched"] = gene_from_guides_by_model[
            "Enriched"
        ]
        gene_from_guides_by_model = gene_from_guides_by_model.drop(
            columns=["Enriched"]
        )
    else:
        gene_from_guides_by_model["frac_enriched"] = 0.0

    gene_from_guides_by_model.to_csv(
        gene_from_guides_by_model_out, index=False
    )
    logger.info(
        "Saved gene-level summary from guides (by model): %s",
        gene_from_guides_by_model_out,
    )

    # --------------------------
    # 11. Tiny textual summary
    # --------------------------
    total_guides = int(guide_meta["sgRNA"].nunique())
    total_genes = int(gene_from_guides["Gene"].nunique())
    n_depleted_g = int(
        (gene_from_guides["frac_depleted"].fillna(0.0) > 0).sum()
    )

    logger.info("===== Guide-Level Enrichment Analysis – Summary =====")
    logger.info("  Guides processed             : %d", total_guides)
    logger.info("  Genes covered                : %d", total_genes)
    logger.info(
        "  Genes with any depleted sgRNA: %d",
        n_depleted_g,
    )
    logger.info("  Output (guide-level)         : %s", guide_out.name)
    logger.info(
        "  Output (gene-from-guides)    : %s",
        gene_from_guides_out.name,
    )
    logger.info(
        "  Output (by-model guides)     : %s",
        gene_from_guides_by_model_out.name,
    )
    logger.info("====================================================")
    logger.info("=== Guide-level Avana enrichment finished ===")
