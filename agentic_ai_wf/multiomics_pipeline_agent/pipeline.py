"""
Programmatic interface for running the multiomics pipeline.

This module provides a function-based API to run the pipeline without requiring
a config file, making it easy to integrate into Python scripts or notebooks.
"""

import json
import logging
import os
from typing import Dict, Optional, Union

from .config import LayerPaths
from .utils import setup_logging
from . import (
    step1_ingestion,
    step2_preprocessing,
    step3_integration,
    step4_ml_biomarkers as step4_ml,
    step5_crossomics,
    step6_literature,
)

email_for_ncbi = "haseeb.manzoor@ayassbioscience.com"

def run_pipeline(
    output_dir: str,
    layers: Optional[Union[Dict[str, str], LayerPaths]] = None,
    metadata_path: Optional[str] = None,
    label_column: Optional[str] = None,
    n_pcs_per_layer: int = 20,
    integrated_dim: int = 50,
    query_term: Optional[str] = None,
    disease_term: Optional[str] = None,
    top_n_results: int = 20,
    up_to_step: Optional[int] = None,
    enable_logging: bool = True,
) -> Dict[str, any]:
    """
    Run the complete multiomics pipeline programmatically.

    This function provides a programmatic interface to run the pipeline without
    requiring a YAML config file. All parameters can be passed directly as function
    arguments.

    Parameters
    ----------
    output_dir : str
        Root output directory for all pipeline outputs. Will be created if it
        doesn't exist.
    layers : dict[str, str] or LayerPaths, optional
        Dictionary mapping layer names to file paths, or a LayerPaths object.
        Supported layer names: 'genomics', 'transcriptomics', 'epigenomics',
        'proteomics', 'metabolomics'.
        Example: {'genomics': 'path/to/genomics.csv', 'transcriptomics': 'path/to/transcriptomics.csv'}
    metadata_path : str, optional
        Path to metadata CSV file with sample information and labels.
    label_column : str, optional
        Name of the column in metadata CSV that contains sample labels/classes.
        If not provided, the system will automatically detect the label column using
        LLM-based analysis (if OpenAI API key is available) or intelligent rule-based detection.
        If metadata_path is provided, auto-detection happens before Step 1.
    n_pcs_per_layer : int, default=20
        Number of principal components to extract per omics layer.
        Saved to pipeline_config.json for use by steps that need it.
    integrated_dim : int, default=50
        Dimensionality for the integrated multiomics representation.
        Saved to pipeline_config.json for use by steps that need it.
    query_term : str, optional
        Query term for literature mining (Step 6). Required if running Step 6.

    disease_term : str, optional
        Disease term for literature search context. If not provided, uses query_term.
    top_n_results : int, default=20
        Number of top biomarkers to include in literature mining (Step 6).
    up_to_step : int, optional
        Run pipeline up to and including this step number (1-6). If None, runs all steps.
    enable_logging : bool, default=True
        Whether to set up file and console logging.

    Returns
    -------
    dict
        Dictionary containing:
        - 'output_dir': The output directory used
        - 'steps_completed': List of step numbers that were executed
        - 'step_results': Dictionary mapping step numbers to their return values
        - 'final_status': 'completed' or 'stopped_at_step_N'

    Examples
    --------
    >>> from multiomics.pipeline import run_pipeline
    >>>
    >>> # Run all steps
    >>> results = run_pipeline(
    ...     output_dir="./output/my_analysis",
    ...     layers={
    ...         "genomics": "data/genomics.csv",
    ...         "transcriptomics": "data/transcriptomics.csv"
    ...     },
    ...     metadata_path="data/metadata.csv",
    ...     query_term="cancer biomarkers",
    ...     disease_term="breast cancer",

    ... )
    >>>
    >>> # Run only up to step 3
    >>> results = run_pipeline(
    ...     output_dir="./output/my_analysis",
    ...     layers={"genomics": "data/genomics.csv"},
    ...     up_to_step=3
    ... )
    """
    # Normalize layers parameter
    if layers is None:
        layers = LayerPaths()
    elif isinstance(layers, dict):
        # Convert dict to LayerPaths object
        layers = LayerPaths(**layers)

    # Map output_dir to base_dir for internal step functions (they use base_dir)
    base_dir = output_dir

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Save pipeline configuration parameters to a JSON file
    # This allows steps to access these parameters if needed
    pipeline_config_path = os.path.join(output_dir, "pipeline_config.json")
    pipeline_config = {
        "label_column": label_column,
        "n_pcs_per_layer": n_pcs_per_layer,
        "integrated_dim": integrated_dim,
        "query_term": query_term,
        "email_for_ncbi": email_for_ncbi,
        "disease_term": disease_term,
        "top_n_results": top_n_results,
    }
    with open(pipeline_config_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2)

    # Set up logging
    logger = None
    if enable_logging:
        setup_logging(os.path.join(output_dir, "logs"))
        logger = logging.getLogger("multiomics.pipeline")
        logger.info("Starting multiomics pipeline")
        logger.info("Output directory: %s", output_dir)
    else:
        # Create a minimal logger that doesn't output anything
        logger = logging.getLogger("multiomics.pipeline")
        logger.setLevel(logging.WARNING)

    # Helper: check whether we should stop after a given step
    def stop_here(step_number: int) -> bool:
        if up_to_step is not None and up_to_step == step_number:
            if enable_logging:
                logger.info("Stopping after Step %d as requested.", step_number)
            return True
        return False

    steps_completed = []
    step_results = {}

    # Auto-detect label_column if not provided but metadata_path is given
    if label_column is None and metadata_path is not None and os.path.exists(metadata_path):
        if enable_logging:
            logger.info("Auto-detecting label column from metadata...")
        try:
            import pandas as pd
            from .metadata_analyzer import detect_label_column_smart
            
            meta_df = pd.read_csv(metadata_path)
            detected_label = detect_label_column_smart(meta_df, use_llm=True)
            if detected_label:
                label_column = detected_label
                if enable_logging:
                    logger.info(f"Auto-detected label column: {label_column}")
            else:
                if enable_logging:
                    logger.warning("Could not auto-detect label column. Will use default or auto-detection in Step 1.")
        except Exception as e:
            if enable_logging:
                logger.warning(f"Error during label column auto-detection: {e}. Will proceed with Step 1 auto-detection.")

    # ---------------- STEP 1 — Ingestion ----------------
    if up_to_step is None or up_to_step >= 1:
        if enable_logging:
            logger.info("=== STEP 1: Ingestion ===")
        s1 = step1_ingestion.run_step1(
            base_dir=base_dir,
            layer_paths=layers,
            metadata_path=metadata_path,
            label_column=label_column,
        )
        steps_completed.append(1)
        step_results[1] = s1
        if enable_logging:
            logger.info("Step 1 completed: %s", s1)
        if stop_here(1):
            return {
                "output_dir": output_dir,
                "steps_completed": steps_completed,
                "step_results": step_results,
                "final_status": f"stopped_at_step_1",
            }

    # ---------------- STEP 2 — Preprocessing & QC ----------------
    if up_to_step is None or up_to_step >= 2:
        if enable_logging:
            logger.info("=== STEP 2: Preprocessing & QC ===")
        s2 = step2_preprocessing.run_step2(base_dir=base_dir)
        steps_completed.append(2)
        step_results[2] = s2
        if enable_logging:
            logger.info("Step 2 completed: %s", s2)
        if stop_here(2):
            return {
                "output_dir": output_dir,
                "steps_completed": steps_completed,
                "step_results": step_results,
                "final_status": f"stopped_at_step_2",
            }

    # ---------------- STEP 3 — Integration ----------------
    if up_to_step is None or up_to_step >= 3:
        if enable_logging:
            logger.info("=== STEP 3: Integration ===")
        s3 = step3_integration.run_step3(base_dir=base_dir)
        steps_completed.append(3)
        step_results[3] = s3
        if enable_logging:
            logger.info("Step 3 completed: %s", s3)
        if stop_here(3):
            return {
                "output_dir": output_dir,
                "steps_completed": steps_completed,
                "step_results": step_results,
                "final_status": f"stopped_at_step_3",
            }

    # ---------------- STEP 4 — ML biomarker ranking ----------------
    if up_to_step is None or up_to_step >= 4:
        if enable_logging:
            logger.info("=== STEP 4: ML biomarkers ===")
        s4 = step4_ml.run_step4(base_dir=base_dir)
        steps_completed.append(4)
        step_results[4] = s4
        if enable_logging:
            logger.info("Step 4 completed: %s", s4)
        if stop_here(4):
            return {
                "output_dir": output_dir,
                "steps_completed": steps_completed,
                "step_results": step_results,
                "final_status": f"stopped_at_step_4",
            }

    # ---------------- STEP 5 — Cross-omics / network biology ----------------
    if up_to_step is None or up_to_step >= 5:
        if enable_logging:
            logger.info("=== STEP 5: Cross-omics / network biology ===")
        s5 = step5_crossomics.run_step5(base_dir=base_dir)
        steps_completed.append(5)
        step_results[5] = s5
        if enable_logging:
            logger.info("Step 5 completed: %s", s5)
        if stop_here(5):
            return {
                "output_dir": output_dir,
                "steps_completed": steps_completed,
                "step_results": step_results,
                "final_status": f"stopped_at_step_5",
            }

    # ---------------- STEP 6 — Literature mining + report (optional) ----------------
    if up_to_step is None or up_to_step >= 6:
        if enable_logging:
            logger.info("=== STEP 6: Literature mining ===")
        s6 = step6_literature.run_step6(
            base_dir=base_dir,
            query_term=query_term,
            email=email_for_ncbi,
            disease_term=disease_term,
            top_n=top_n_results,
        )
        steps_completed.append(6)
        step_results[6] = s6
        if enable_logging:
            logger.info("Step 6 completed: %s", s6)

    if enable_logging:
        logger.info("Multiomics pipeline completed.")

    return {
        "output_dir": output_dir,
        "steps_completed": steps_completed,
        "step_results": step_results,
        "final_status": "completed",
    }

