# run_depmap_codependency.py

from pathlib import Path
import pandas as pd
from logging import Logger

from .depmap_codependency_uif import run_codependency_correlation


def run_depmap_codependency(output_dir: Path, logger: Logger):
    # Make sure timestamped run dirs exist

    logger.info("=== DepMap co-dependency correlation (selected genes/models) ===")
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # 1) Input: tidy dependency table from the existing dependencies step
    # ------------------------------------------------------------------
    deps_dir = output_dir / "DepMap_Dependencies"
    tidy_path = deps_dir / "Dependencies_Tidy_SelectedGenes_SelectedModels.csv"

    if not tidy_path.exists():
        raise FileNotFoundError(
            f"Could not find tidy dependency table:\n  {tidy_path}\n"
            "Make sure you have already run the DepMap dependencies step "
            "(run_depmap_dependencies.py) for this timestamped run."
        )

    logger.info("Loading tidy dependencies from: %s", tidy_path)
    tidy = pd.read_csv(tidy_path)

    # ------------------------------------------------------------------
    # 2) Output folder for co-dependency results
    # ------------------------------------------------------------------
    outdir = output_dir / "DepMap_GuideAnalysis" / "Codependency"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Co-dependency outputs will be written under: %s", outdir)

    # ------------------------------------------------------------------
    # 3) FIXED, NON-INTERACTIVE CONFIG
    #
    #    - Uses ALL models present in `tidy`
    #      (whatever was selected upstream via Selected_Models.csv)
    #    - Default method: Spearman
    #      -> to run multiple: change to ["spearman", "pearson", "cosine"], etc.
    # ------------------------------------------------------------------
    config = {
        "MIN_OVERLAP": 3,          # min shared models per gene pair
        "MAX_CORR_GENES": 200,     # cap number of genes for runtime
        "MIN_PROB": None,          # e.g. set to 0.5 to enforce DepProb ≥ 0.5
        "CORR_METHODS": ["spearman", "pearson", "kendall", "cosine"],  # default methods (no user input)
    }

    logger.info("Co-dependency config: %s", config)

    # ------------------------------------------------------------------
    # 4) Run co-dependency module (NO user interaction)
    # ------------------------------------------------------------------
    run_codependency_correlation(
        tidy=tidy,
        outdir=outdir,
        config=config,
        logger=logger,
    )

    logger.info("=== DepMap co-dependency correlation completed ===")

