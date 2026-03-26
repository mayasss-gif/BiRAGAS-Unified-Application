from pathlib import Path
from .logging_utils import setup_logger

from .integration import (
    build_connectivity_score,
    compute_effect_strength,
    compute_essentiality,
    compute_connectivity,
    compute_druggability_and_safety,
    build_final_prioritization,
    generate_integration_report,
)
from .constants import left_logo, right_logo



def run_integration_pipeline(
    deg_path: Path,
    output_dir: Path,
    l1000_path: Path,
    depmap_path: Path,
):

    """
    Runs the integration pipeline.

    """

    logger = setup_logger(log_dir=output_dir, name="Integration")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running Integration pipeline with deg_path: {deg_path}, output_dir: {output_dir}")
    
    # DEBUG: Log received path values
    logger.info(f"[DEBUG] Received paths in integration pipeline:")
    logger.info(f"[DEBUG]   - l1000_path: {l1000_path} (type: {type(l1000_path)}, is None: {l1000_path is None})")
    logger.info(f"[DEBUG]   - depmap_path: {depmap_path} (type: {type(depmap_path)}, is None: {depmap_path is None})")

    build_connectivity_score(
        deg_path=deg_path,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info(f"[DEBUG] About to call compute_effect_strength with l1000_path={l1000_path}")
    effect_df = compute_effect_strength(
        l1000_path=l1000_path,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info(f"[DEBUG] compute_effect_strength completed successfully")

    ess_df = compute_essentiality(
        output_dir=output_dir,
        logger=logger,
        depmap_path=depmap_path,
    )
    conn_df = compute_connectivity(
        output_dir=output_dir,
        logger=logger,
    )
    drug_df = compute_druggability_and_safety(
        output_dir=output_dir,
        logger=logger,
    )



    build_final_prioritization(
        output_dir=output_dir,
        logger=logger,
        l1000_path=l1000_path,
        effect_df=effect_df,
        ess_df=ess_df,
        conn_df=conn_df,
        drug_df=drug_df,
    )



    generate_integration_report(
        output_dir=output_dir,
        logo_left_path=left_logo,
        logo_right_path=right_logo,
    )


    logger.info(f"Integration pipeline completed successfully")


# if __name__ == "__main__":
#     output_dir = Path("Integration_Output")
#     l1000_path = Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\L1000_Output")
#     depmap_path = Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\Output")
#     run_integration_pipeline(
#         deg_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\leukemia_DEGs_prioritized.csv"),
#         output_dir=output_dir,
#         l1000_path=l1000_path,
#         depmap_path=depmap_path,
#     )
