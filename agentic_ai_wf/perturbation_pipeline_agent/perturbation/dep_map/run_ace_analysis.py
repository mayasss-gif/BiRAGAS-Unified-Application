"""
ACE Analysis Orchestrator

Coordinates the complete ACE causality analysis pipeline:
1. Compute ACE (causality_analysis)
2. Generate figures (causality_figures)
3. Generate graph (causality_graph)
"""

from pathlib import Path
from typing import Optional

from .ace import (
    compute_ace_analysis,
    generate_causality_figures,
    generate_ace_graph,
    ACEConfig,
)


def run_ace_analysis(
    output_dir: Path,
    top_n: int = 40,
    ace_config: Optional[ACEConfig] = None,
    logger=None,
) -> dict:
    """
    Run complete ACE causality analysis pipeline.
    
    This function orchestrates three sub-steps:
    1. ACE computation with therapeutic alignment (causality_analysis)
    2. Interactive HTML figure generation (causality_figures)
    3. Signed ACE graph visualization (causality_graph)
    
    Args:
        output_dir: Root DepMap output directory
        top_n: Top N genes for graphs/figures (default: 40)
        ace_config: ACEConfig instance for customizing ACE computation (optional)
        logger: Logger instance for progress tracking (optional)
    
    Expected Input Files (from previous pipeline steps):
        - DepMap_Dependencies/Dependencies_Tidy_SelectedGenes_SelectedModels.csv
        - DepMap_GuideAnalysis/CRISPR_Perturbation_GeneStats.csv
    
    Output Files:
        - DepMap_Causality/CausalEffects_ACE.csv
        - DepMap_Causality/CausalDrivers_Ranked.csv
        - DepMap_Causality/figs_html/index.html (and individual figure HTMLs)
        - DepMap_Causality/graphs_extra/signed_gene_viability_alignment.html
        
    Returns:
        Dictionary with paths to generated outputs:
            - causality_dir: Path to DepMap_Causality directory
            - ace_csv: Path to CausalEffects_ACE.csv
            - ranked_csv: Path to CausalDrivers_Ranked.csv
            - figures_index: Path to figures index.html
            - graph_html: Path to signed ACE graph
    
    Raises:
        FileNotFoundError: If required input files are missing
        ValueError: If input files have incorrect format
    """
    
    if logger:
        logger.info("=" * 70)
        logger.info("Starting ACE Causality Analysis Pipeline")
        logger.info("=" * 70)
    
    # Define directory structure
    causality_dir = output_dir / "DepMap_Causality"
    figs_dir = causality_dir / "figs_html"
    graphs_dir = causality_dir / "graphs_extra"
    
    # Create output directories
    causality_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Define input file paths
    tidy_path = output_dir / "DepMap_Dependencies" / "Dependencies_Tidy_SelectedGenes_SelectedModels.csv"
    gene_stats_path = output_dir / "DepMap_GuideAnalysis" / "CRISPR_Perturbation_GeneStats.csv"
    
    # Validate input files exist
    if not tidy_path.exists():
        error_msg = (
            f"Missing required input file: {tidy_path}\n"
            f"ACE analysis requires DepMap Codependency step to run first."
        )
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not gene_stats_path.exists():
        error_msg = (
            f"Missing required input file: {gene_stats_path}\n"
            f"ACE analysis requires Guide Level Enrichment step to run first."
        )
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # =========================================================================
    # Step 1: Compute ACE + Therapeutic Alignment
    # =========================================================================
    if logger:
        logger.info("-" * 70)
        logger.info("Step 1/3: Computing ACE with therapeutic alignment")
        logger.info("-" * 70)
    
    try:
        ace_results = compute_ace_analysis(
            tidy_csv=str(tidy_path),
            gene_stats_csv=str(gene_stats_path),
            output_dir=causality_dir,
            config=ace_config,
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.error(f"ACE computation failed: {str(e)}")
        raise
    
    # =========================================================================
    # Step 2: Generate Standard HTML Figures
    # =========================================================================
    if logger:
        logger.info("-" * 70)
        logger.info("Step 2/3: Generating interactive HTML figures")
        logger.info("-" * 70)
    
    try:
        figures_index = generate_causality_figures(
            causality_dir=causality_dir,
            output_dir=figs_dir,
            tidy_path=tidy_path,  # Optional: enables Program→Viability edges
            top_n=top_n,
            top_gene_viability=30,
            max_gene_program_edges=600,
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.error(f"Figure generation failed: {str(e)}")
        raise
    
    # =========================================================================
    # Step 3: Generate Signed ACE Graph
    # =========================================================================
    if logger:
        logger.info("-" * 70)
        logger.info("Step 3/3: Generating signed ACE graph")
        logger.info("-" * 70)
    
    try:
        graph_html = generate_ace_graph(
            causality_dir=causality_dir,
            output_dir=graphs_dir,
            top_n=top_n,
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.error(f"Graph generation failed: {str(e)}")
        raise
    
    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    if logger:
        logger.info("=" * 70)
        logger.info("ACE CAUSALITY ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Output Summary:")
        logger.info(f"  - Causality CSVs : {causality_dir}")
        logger.info(f"  - ACE Effects    : {ace_results['ace_csv']}")
        logger.info(f"  - Ranked Drivers : {ace_results['ranked_csv']}")
        logger.info(f"  - Figures        : {figures_index}")
        logger.info(f"  - Signed Graph   : {graph_html}")
        logger.info("=" * 70)
    
    return {
        "causality_dir": causality_dir,
        "ace_csv": ace_results["ace_csv"],
        "ranked_csv": ace_results["ranked_csv"],
        "manifest": ace_results["manifest"],
        "figures_index": figures_index,
        "graph_html": graph_html,
    }
