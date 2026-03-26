import argparse
import logging
import os

from .config import load_config
from .utils import setup_logging
from . import (
    step1_ingestion,
    step2_preprocessing,
    step3_integration,
    step4_ml_biomarkers as step4_ml,
    step5_crossomics,
    step6_literature,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multiomics pipeline from the command line."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file (e.g. config.yaml)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all steps (1–6) in sequence.",
    )
    parser.add_argument(
        "--up-to-step",
        type=int,
        choices=range(1, 7),
        help=(
            "Run steps up to the given step number (1–6). "
            "If provided together with --run-all, this acts as a cutoff."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---------------- Load config ----------------
    cfg = load_config(args.config)

    # Make sure base_dir exists
    os.makedirs(cfg.base_dir, exist_ok=True)

    # ---------------- Logging ----------------
    # This will typically create a logs/ directory under base_dir
    setup_logging(os.path.join(cfg.base_dir, "logs"))
    logger = logging.getLogger("multiomics.cli")

    logger.info("Loaded config from %s", args.config)
    logger.info("Base output directory: %s", cfg.base_dir)

    # Helper: check whether we should stop after a given step
    def stop_here(step_number: int) -> bool:
        if args.up_to_step is not None and args.up_to_step == step_number:
            logger.info("Stopping after Step %d as requested.", step_number)
            return True
        return False

    # If neither --run-all nor --up-to-step is given, default to run-all
    run_all = args.run_all or (args.up_to_step is None)

    # ---------------- STEP 1 — Ingestion ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 1):
        logger.info("=== STEP 1: Ingestion ===")
        s1 = step1_ingestion.run_step1(
            base_dir=cfg.base_dir,
            layer_paths=cfg.layers,      # LayerPaths mapping from config
            metadata_path=cfg.metadata,
        )
        logger.info("Step 1 completed: %s", s1)
        if stop_here(1):
            return

    # ---------------- STEP 2 — Preprocessing & QC ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 2):
        logger.info("=== STEP 2: Preprocessing & QC ===")
        s2 = step2_preprocessing.run_step2(base_dir=cfg.base_dir)
        logger.info("Step 2 completed: %s", s2)
        if stop_here(2):
            return

    # ---------------- STEP 3 — Integration ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 3):
        logger.info("=== STEP 3: Integration ===")
        # Call Step 3 with only base_dir; it uses its own defaults
        s3 = step3_integration.run_step3(base_dir=cfg.base_dir)
        logger.info("Step 3 completed: %s", s3)
        if stop_here(3):
            return

    # ---------------- STEP 4 — ML biomarker ranking ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 4):
        logger.info("=== STEP 4: ML biomarkers ===")
        # Step 4 internally decides supervised vs unsupervised
        # depending on the presence of labels/metadata.
        s4 = step4_ml.run_step4(base_dir=cfg.base_dir)
        logger.info("Step 4 completed: %s", s4)
        if stop_here(4):
            return

    # ---------------- STEP 5 — Cross-omics / network biology ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 5):
        logger.info("=== STEP 5: Cross-omics / network biology ===")
        s5 = step5_crossomics.run_step5(base_dir=cfg.base_dir)
        logger.info("Step 5 completed: %s", s5)
        if stop_here(5):
            return

    # ---------------- STEP 6 — Literature mining + report (optional) ----------------
    if run_all or (args.up_to_step and args.up_to_step >= 6):
        logger.info("=== STEP 6: Literature mining ===")
        s6 = step6_literature.run_step6(
            base_dir=cfg.base_dir,
            query_term=cfg.query_term,
            email=cfg.email_for_ncbi,
            disease_term=getattr(cfg, "disease_term", None),
            top_n=cfg.top_n_results,
        )
        logger.info("Step 6 completed: %s", s6)

    logger.info("Multiomics pipeline completed.")


if __name__ == "__main__":
    main()
