from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Thread-safe non-interactive backend - MUST be set before importing pyplot
from .logging_utils import setup_logger
from .l1000.build_user_input import DEFAULT_MAX_SIGS

from .l1000 import (
    run_input_sanity_check,
    build_user_input,
    run_prepare_files,
    run_l1000_eda,
    run_drug_perturbation_from_userinput,
    run_l1000_extra,
    run_drug_similarity,
    run_visualizations,
    generate_reversal_plots,
    build_report,
)
from .constants import left_logo, right_logo



def run_l1000_pipeline(
    deg_path: Path,
    pathway_path: Path,
    output_dir: Path,
    disease: str,
    tissue: str = None,
    drug: str = None,
    time_points: str = None,
    cell_lines: str = None,
    max_sigs: int | None = DEFAULT_MAX_SIGS,
):

    """
    Runs the L1000 pipeline.
    
    Args:
        deg_path: Path, (e.g. "lupus_DEGs_prioritized.csv")
        pathway_path: Path, (e.g. "lupus_Pathways_Consolidated.csv")
        output_dir: Path, (e.g. "L1000_Output")
        disease: str, (e.g. "Lupus")
        tissue: str = None, (e.g. "Breast", "Lung", "Bone")
        drug: str = None, (e.g. "Doxorubicin")
        time_points: str = None, (e.g. "6,24")
        cell_lines: str = None, (e.g. "A375, BT20")
        max_sigs: int | None = DEFAULT_MAX_SIGS (forces this value if provided)
    """


    logger = setup_logger(log_dir=output_dir, name="L1000")
    
    # Check plotting dependencies at startup
    from .plotting_utils import check_plotting_dependencies
    deps = check_plotting_dependencies()
    logger.info(f"Plotting dependencies: matplotlib={deps['matplotlib']}, "
                f"backend_agg={deps['matplotlib_backend_agg']}, "
                f"plotly={deps['plotly']}, kaleido={deps['kaleido']}")



    logger.info(
        f"Running L1000 pipeline with deg_path: {deg_path}, pathway_path: {pathway_path}, "
        f"output_dir: {output_dir}, disease: {disease}, tissue: {tissue}, drug: {drug}, "
        f"time_points: {time_points}, cell_lines: {cell_lines}, max_sigs: {max_sigs}"
    )

    ###########################################################################
    # Running Input Sanity Check
    ###########################################################################
    logger.info(f"Running Input Sanity Check")
    output_dir.mkdir(parents=True, exist_ok=True)


    build_user_input(
        output_dir=output_dir,
        disease=disease,
        tissue=tissue,
        drug=drug,
        time_points=time_points,
        cell_lines=cell_lines,
        max_sigs_override=max_sigs,
    )

    logger.info(f"User input built successfully")


    run_input_sanity_check(
        output_dir=output_dir,
        deg_src=deg_path,
        path_src=pathway_path,
        logger=logger,
    )
    logger.info(f"Input sanity check completed successfully")

    ###########################################################################
    # Preparing Files
    ###########################################################################
    logger.info(f"Preparing Files")
    run_prepare_files(
        output_dir=output_dir,
        deg_src=deg_path,
        pathway_src=pathway_path,
        logger=logger,
    )
    logger.info(f"Files prepared successfully")


    ###########################################################################
    # Running L1000 EDA
    ###########################################################################
    logger.info(f"Running L1000 EDA")
    run_l1000_eda(
        output_dir=output_dir,
        logger=logger,
    )
    logger.info(f"L1000 EDA completed successfully")


    ###########################################################################
    # Running L1000 Drug Perturbation
    ###########################################################################
    logger.info(f"Running L1000 Drug Perturbation")
    run_drug_perturbation_from_userinput(
        output_dir=output_dir,
        logger=logger,
    )
    logger.info(f"L1000 Drug Perturbation completed successfully")

    ###########################################################################
    # Running L1000 Extra
    ###########################################################################
    logger.info(f"Running L1000 Extra")
    run_l1000_extra(
        output_dir=output_dir,
    )
    logger.info(f"L1000 Extra completed successfully")

    ###########################################################################
    # Running Drug Similarity
    ###########################################################################
    logger.info(f"Running Drug Similarity")
    run_drug_similarity(
        output_dir=output_dir,
    )
    logger.info(f"Drug Similarity completed successfully")


    ###########################################################################
    # Running Visualizations
    ###########################################################################
    logger.info(f"Running Visualizations")
    run_visualizations(
        output_dir=output_dir,
    )
    logger.info(f"Visualizations completed successfully")

    ###########################################################################
    # Running Reversal Plots
    ###########################################################################
    logger.info(f"Running Reversal Plots")
    generate_reversal_plots(
        output_dir=output_dir,
        logger=logger,
    )
    logger.info(f"Reversal Plots completed successfully")

    ###########################################################################
    # Generating L1000 Report
    ###########################################################################
    logger.info(f"Generating L1000 Report")
    build_report(
        run_dir=output_dir,
        disease_path=output_dir / "userinput.txt",
        out_html=output_dir / "l1000_report.html",
        logo_left_path=left_logo,
        logo_right_path=right_logo,
    )
    logger.info(f"L1000 Report completed successfully")


    logger.info(f"L1000 Drug Perturbation completed successfully")


# if __name__ == "__main__":
#     run_l1000_pipeline(
#         deg_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\lupus_DEGs_prioritized.csv"),
#         pathway_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\lupus_Pathways_Consolidated.csv"),
#         output_dir=Path("L1000_Output"),
#         disease="Lupus",
#     )
