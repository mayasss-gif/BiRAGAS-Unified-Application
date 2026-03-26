"""
CRISPR Perturb-seq Pipeline Package

This package provides a complete pipeline for analyzing CRISPR perturbation 
single-cell RNA-seq data.

Main Functions:
- run_pipeline: Execute the full CRISPR pipeline (Stages 0a-12 + scRNA + report)
- validate_dataset: Validate input data before running the pipeline
- generate_report: Generate HTML report from pipeline outputs
- get_available_samples: Discover samples in a GSE directory
- get_metadata_groups: Extract group information from metadata.csv
- get_scrna_config_options: Get all available scRNA configuration options
- discover_pipeline_inputs: Comprehensive discovery of all inputs and options
"""

from .run_pipeline import (
    run_pipeline,
    get_available_samples,
    get_metadata_groups,
    get_scrna_config_options,
    discover_pipeline_inputs,
)
from .reporting import generate_report
from .stage0a_dataset_validator import validate_dataset

__all__ = [
    "run_pipeline",
    "validate_dataset",
    "generate_report",
    "get_available_samples",
    "get_metadata_groups",
    "get_scrna_config_options",
    "discover_pipeline_inputs",
]
__version__ = "1.0.0"
