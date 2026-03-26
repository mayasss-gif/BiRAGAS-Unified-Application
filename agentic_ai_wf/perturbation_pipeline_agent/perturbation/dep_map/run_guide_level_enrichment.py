# run_guide_level_enrichment.py

import sys
from pathlib import Path
from logging import Logger

from .constants import DATA_DIR

from .screen_context_uif import attach_screen_context_to_per_model
from .depmap_guides_uif import run_guide_level_enrichment as run_avana_guides
from .perturbation_analysis_uif import run_crispr_perturbation


def run_guide_level_enrichment(output_dir: Path, logger: Logger):

    logger.info("=== Guide level Enrichment analysis – full pipeline ===")

    # -----------------------------
    # STEP 1: Screen context per model
    # -----------------------------
    logger.info(">>> STEP 1: Attach DepMap screen context to per-model essentiality table")

    # DepMap repository directory for screen metadata
    data_dir = DATA_DIR  # points to DepMap_Repository

    # Where essentiality outputs live for THIS run
    deps_outdir = output_dir / "DepMap_Dependencies"
    per_model_path = deps_outdir / "GeneEssentiality_PerModel.csv"

    # Where to save screen-context outputs for THIS run
    guide_outdir = output_dir / "DepMap_GuideAnalysis"
    guide_outdir.mkdir(parents=True, exist_ok=True)

    if not per_model_path.exists():
        raise FileNotFoundError(
            f"Cannot find {per_model_path}.\n"
            "Make sure you ran the essentiality / DepMap dependencies step before this."
        )

    result_ctx = attach_screen_context_to_per_model(
        per_model_path=per_model_path,
        data_dir=data_dir,
        outdir=guide_outdir,
        logger=logger,
    )

    logger.info("=== Screen context attachment completed ===")
    logger.info(f" Per-model base shape          : {result_ctx['shape_per_model']}")
    logger.info(f" Per-model with context shape  : {result_ctx['shape_per_model_with_context']}")
    logger.info(f" Models with any metadata      : {result_ctx['n_models_with_metadata']}")
    logger.info(f" Per-model+context CSV         : {result_ctx['per_model_with_context']}")
    logger.info(f" Model screen-context metadata : {result_ctx['model_screen_context']}")

    # -----------------------------
    # STEP 2: Guide-level Avana enrichment
    # -----------------------------
    logger.info(">>> STEP 2: Guide-level Avana enrichment for selected genes/models")

    run_avana_guides(
        output_dir=output_dir,   # per-run root
        logger=logger,
    )

    logger.info("=== Guide-level Avana enrichment completed ===")

    # -----------------------------
    # STEP 3: CRISPR perturbation analysis + similarity
    # -----------------------------
    logger.info(">>> STEP 3: CRISPR perturbation analysis & similarity mapping")

    run_crispr_perturbation(
        logger=logger,
        output_dir=output_dir,  # per-run outputs, figs, html
    )

    logger.info("=== CRISPR perturbation analysis completed ===")
    logger.info("=== Guide level Enrichment pipeline (all sub-steps) finished ===")