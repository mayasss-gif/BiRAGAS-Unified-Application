from .dep_map import (
    prepare_deg_files,
    build_gene_list_from_prepared_simple,
    build_selection_files,
    run_check_cell_essentiality,
    run_guide_level_enrichment,
    run_depmap_codependency,
    run_ace_analysis,
    run_visualization,
    generate_depmap_report,
)
from .logging_utils import setup_logger
from pathlib import Path
from .constants import left_logo, right_logo




def run_depmap_pipeline(
    raw_deg_path: Path,
    output_dir: Path,
    disease: str,
    mode_model: str = None,
    genes_selection: str = "all",
    top_up: int = None,
    top_down: int = None,
    run_ace: bool = True,
    ace_top_n: int = 40,
):

    """
    Runs the DEPMAP pipeline.

    Args:
        raw_deg_path (Path): The path to the raw DEG file.
        output_dir (Path): The path to the output directory.
        disease (str): The disease to model.
        mode_model (str): The mode to use for selecting models. Example: "by_disease", "by_lineage", "by_ids", "by_names", "keyword". Default: None.
        genes_selection (str): The mode to use for selecting genes. Example: "all", "top". Default: "all".
        top_up (int): The number of top up genes to select. Example: 20. Default: None.
        top_down (int): The number of top down genes to select. Example: 20. Default: None.
        run_ace (bool): Whether to run ACE causality analysis. Default: True.
        ace_top_n (int): Top N genes for ACE graphs and figures. Default: 40.
    """


    logger = setup_logger(log_dir=output_dir, name="DepMap")


    logger.info(f"Running DEPMAP pipeline with raw DEG file: {raw_deg_path}")

    logger.info(f"Running Step 0/5: Setting up output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    build_selection_files(
        output_dir=output_dir,
        disease=disease,
        mode_model=mode_model,
        genes_selection=genes_selection,
        top_up=top_up,
        top_down=top_down,
    )

    # ==========================================================================
    # Step 1: Prepare DEG files
    # ==========================================================================
    logger.info(f"Running Step 1/5: Processing DEG files from {raw_deg_path}")
    result = prepare_deg_files(
        raw_deg_path=raw_deg_path,
        output_dir=output_dir,
        logger=logger,
    )
    build_gene_list_from_prepared_simple(
        prepared_simple_path=result.prepared_simple,
        output_dir=output_dir,
        logger=logger,
    )

    # ==========================================================================
    # Step 2: Run Check Cell Essentiality
    # ==========================================================================
    logger.info(f"Running Step 2/5: Running Check Cell Essentiality")
    run_check_cell_essentiality(
        output_dir=output_dir,
        deg_simple_path=result.prepared_simple,
        logger=logger,
    )

    # ==========================================================================
    # Step 3: Run Guide Level Enrichment
    # ==========================================================================
    logger.info(f"Running Step 3/5: Running Guide Level Enrichment")
    run_guide_level_enrichment(
        output_dir=output_dir,
        logger=logger,
    )

    # ==========================================================================
    # Step 4: Run DepMap Codependency
    # ==========================================================================
    logger.info(f"Running Step 4/5: Running DepMap Codependency")
    run_depmap_codependency(
        output_dir=output_dir,
        logger=logger,
    )

    # ==========================================================================
    # Step 5: Run ACE Causality Analysis
    # ==========================================================================
    if run_ace:
        logger.info(f"Running Step 5/7: Running ACE Causality Analysis")
        try:
            run_ace_analysis(
                output_dir=output_dir,
                top_n=ace_top_n,
                logger=logger,
            )
        except FileNotFoundError as e:
            logger.warning(f"ACE analysis skipped: {str(e)}")
        except Exception as e:
            logger.error(f"ACE analysis failed: {str(e)}")
            logger.warning("Continuing pipeline without ACE results")
    else:
        logger.info(f"Step 5/7: ACE Causality Analysis skipped (run_ace=False)")

    # ==========================================================================
    # Step 6: Run Visualization
    # ==========================================================================
    logger.info(f"Running Step 6/7: Running Visualization")
    run_visualization(
        output_dir=output_dir
    )
    logger.info(f"[DEBUG] Step 6/7 Visualization completed successfully")

    # ==========================================================================
    # Step 7: Generate DepMap Report
    # ==========================================================================
    logger.info(f"Running Step 7/7: Generating DepMap Report")
    generate_depmap_report(
        depmap_root=output_dir,
        out_path=output_dir / "DepMap_Report.html",
        ayass_logo_left=left_logo,
        ayass_logo_right=right_logo,
        disease=disease,
    )
    logger.info(f"[DEBUG] Step 7/7 Report generation completed successfully")

    logger.info(f"DEPMAP pipeline completed successfully")
    logger.info(f"[DEBUG] DepMap pipeline returning successfully - all steps completed")



# if __name__ == "__main__":
#     run_depmap_pipeline(
#         raw_deg_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\leukemia_DEGs_prioritized.csv"),
#         output_dir=Path("Output"),
#         disease="Bladder Urothelial Carcinoma",
#     )