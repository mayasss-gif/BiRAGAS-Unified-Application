# src/screen_context_uif.py

from pathlib import Path
from typing import Optional, Dict, Any

import logging
import numpy as np
import pandas as pd


def attach_screen_context_to_per_model(
    per_model_path: Path,
    data_dir: Path,
    outdir: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Guide-level enrichment / screen context analysis:
    Attach DepMap screen / ModelCondition metadata per ModelID to the
    per-model essentiality table.

    Inputs
    ------
    per_model_path : Path
        Path to GeneEssentiality_PerModel.csv
        (output of your essentiality tables step).
    data_dir : Path
        Path to DepMap repository directory (must contain:
        ModelCondition.csv, CRISPRScreenMap.csv, ScreenSequenceMap.csv).
    outdir : Path
        Output directory where final tables will be written.
        (e.g., RUN_OUTPUT_DIR / "DepMap_GuideAnalysis")
    logger : logging.Logger, optional
        If None, a module-level logger is used.

    Outputs
    -------
    Returns a dict with:
        {
          "per_model_with_context": Path,
          "model_screen_context": Path,
          "n_models_with_metadata": int,
          "shape_per_model": (rows, cols),
          "shape_per_model_with_context": (rows, cols),
        }
    """

    log = logger or logging.getLogger(__name__)

    per_model_path = Path(per_model_path)
    data_dir = Path(data_dir)
    outdir = Path(outdir)

    log.info("=== STEP 9: Guide level enrichment – attach DepMap screen context per ModelID ===")

    # -------------------------------------------------
    # 0. Check all required paths
    # -------------------------------------------------
    path_mcond = data_dir / "ModelCondition.csv"
    path_scrmap = data_dir / "CRISPRScreenMap.csv"
    path_seqmap = data_dir / "ScreenSequenceMap.csv"

    for p in [per_model_path, path_mcond, path_scrmap, path_seqmap]:
        status = "OK" if p.exists() else "MISSING"
        log.info(f"{p} -> [{status}]")
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for screen-context step: {p}")

    # -------------------------------------------------
    # 1. Load per-model essentiality base table
    # -------------------------------------------------
    per_model = pd.read_csv(per_model_path)
    log.info(f"Loaded per-model essentiality: shape={per_model.shape}")

    # -------------------------------------------------
    # 2. Load DepMap metadata tables
    # -------------------------------------------------
    mcond = pd.read_csv(path_mcond, low_memory=False)
    scrmap = pd.read_csv(path_scrmap, low_memory=False)
    seqmap = pd.read_csv(path_seqmap, low_memory=False)

    log.info(f"ModelCondition.csv rows      : {mcond.shape}")
    log.info(f"CRISPRScreenMap.csv rows     : {scrmap.shape}")
    log.info(f"ScreenSequenceMap.csv rows   : {seqmap.shape}")

    # -------------------------------------------------
    # 3. Build Screen + ModelCondition metadata per ModelID
    # -------------------------------------------------
    seq_cols = [
        "ScreenID", "ModelConditionID", "ModelID",
        "ScreenType", "Library", "Days",
    ]
    seq_small = seqmap[seq_cols].drop_duplicates()

    # restrict to ScreenIDs that appear in CRISPRScreenMap
    scr_seq = scrmap.merge(seq_small, on=["ScreenID", "ModelID"], how="left")
    log.info(f"Rows after Screen ↔ Sequence merge: {scr_seq.shape}")

    mcond_cols = [
        "ModelConditionID", "ModelID",
        "CellFormat", "GrowthMedia",
    ]
    mcond_small = mcond[mcond_cols].copy()

    scr_mc = scr_seq.merge(
        mcond_small,
        on=["ModelConditionID", "ModelID"],
        how="left",
    )
    log.info(f"Rows after adding ModelCondition: {scr_mc.shape}")

    # -------------------------------------------------
    # 4. Aggregate context to one row per ModelID
    # -------------------------------------------------
    def uniq_join(series: pd.Series) -> str:
        vals = [str(v).strip() for v in series.dropna().astype(str) if str(v).strip() != ""]
        vals = sorted(set(vals))
        return ";".join(vals) if vals else np.nan

    agg = (
        scr_mc
        .groupby("ModelID", as_index=False)
        .agg(
            n_screens    = ("ScreenID", "nunique"),
            n_modelconds = ("ModelConditionID", "nunique"),
            ScreenTypes  = ("ScreenType", uniq_join),
            Libraries    = ("Library", uniq_join),
            AssayDays    = ("Days", uniq_join),
            CellFormats  = ("CellFormat", uniq_join),
            GrowthMedia  = ("GrowthMedia", uniq_join),
        )
    )

    log.info(f"Models with any screen/MC metadata: {len(agg)}")

    # -------------------------------------------------
    # 5. Merge aggregated context into per-model table
    # -------------------------------------------------
    per_full = per_model.merge(agg, on="ModelID", how="left")
    log.info(f"Merged table shape (per-model + context): {per_full.shape}")

    base_cols = [
        "Gene", "ModelID", "CellLineName",
        "OncotreeLineage", "OncotreePrimaryDisease",
        "ChronosGeneEffect", "DependencyProbability",
        "DepProbPass", "EssentialityTag",
    ]
    context_cols = [
        "n_screens", "n_modelconds",
        "Libraries", "ScreenTypes", "AssayDays",
        "CellFormats", "GrowthMedia",
    ]

    # Keep only columns that actually exist
    keep_cols = [c for c in base_cols + context_cols if c in per_full.columns]
    missing_base = [c for c in base_cols if c not in per_full.columns]
    if missing_base:
        log.warning(f"Some expected base columns are missing in per_model: {missing_base}")

    per_slim = per_full[keep_cols].copy()

    # -------------------------------------------------
    # 6. Save outputs
    # -------------------------------------------------
    outdir.mkdir(parents=True, exist_ok=True)

    per_with_ctx_path = outdir / "GeneEssentiality_PerModel_withScreenContext.csv"
    per_slim.to_csv(per_with_ctx_path, index=False)
    log.info(f"Saved slim per-model + screen context table -> {per_with_ctx_path}")

    # Model-level metadata table
    meta_cols = [
        "ModelID",
        "Libraries", "ScreenTypes", "AssayDays",
        "CellFormats", "GrowthMedia",
    ]
    meta_cols = [c for c in meta_cols if c in per_slim.columns]

    model_meta = (
        per_slim[meta_cols]
        .drop_duplicates(subset=["ModelID"])
        .reset_index(drop=True)
    )

    model_ctx_path = outdir / "Model_ScreenContext_Metadata.csv"
    model_meta.to_csv(model_ctx_path, index=False)
    log.info(f"Saved ModelID-level screen context metadata -> {model_ctx_path}")

    return {
        "per_model_with_context": per_with_ctx_path,
        "model_screen_context": model_ctx_path,
        "n_models_with_metadata": int(len(agg)),
        "shape_per_model": tuple(per_model.shape),
        "shape_per_model_with_context": tuple(per_slim.shape),
    }
