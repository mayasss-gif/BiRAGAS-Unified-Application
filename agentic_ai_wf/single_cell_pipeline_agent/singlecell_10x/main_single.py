# main_single.py
from pathlib import Path
import logging
import sys
from typing import Optional

from .loader_10x import load_10x_feature_barcode_matrix
from .pipeline import run_scanpy_pipeline
from .sc_to_bisq import process_h5ad_file


def _find_geo_json(single_10x_dir: Path) -> Optional[Path]:
    """
    Automatically find GEO metadata JSON file in the single_10x_dir.
    Looks for files matching patterns like GSE*_metadata.json or *_metadata.json
    """
    if not single_10x_dir.exists():
        return None
    
    # Common patterns for GEO metadata JSON files
    patterns = [
        "GSE*_metadata.json",
        "*_metadata.json",
        "GSE*.json",
    ]
    
    for pattern in patterns:
        matches = list(single_10x_dir.glob(pattern))
        if matches:
            # Prefer files with "metadata" in the name
            metadata_files = [f for f in matches if "metadata" in f.name.lower()]
            if metadata_files:
                return metadata_files[0]
            return matches[0]
    
    return None


def run_pipeline(
    single_10x_dir: str | Path,
    sample_label: Optional[str] = None,
    group_label: Optional[str] = None,
    out_name: str = "SC_RESULTS",
    do_pathway_clustering: bool = True,
    do_groupwise_de: bool = False,
    do_dpt: bool = False,
    batch_key: Optional[str] = None,
    integration_method: Optional[str] = None,
    geo_json_path: Optional[str | Path] = None,
    logos_dir: Optional[str | Path] = None,
    generate_report: bool = True,
    prepare_for_bisque: bool = True,
) -> Path:
    """
    Run the single-cell 10x pipeline programmatically.
    
    Parameters
    ----------
    single_10x_dir : str | Path
        Path to the 10x Genomics data folder (containing matrix.mtx, barcodes.tsv, features.tsv)
    sample_label : str, optional
        Sample label to store in obs['sample']. If None, will be extracted from single_10x_dir name.
    group_label : str, optional
        Group label to store in obs['group'] (e.g., "CASE", "CONTROL", "TUMOR", "NORMAL")
    out_name : str, default "SC_RESULTS"
        Name of the output folder (will be created inside single_10x_dir)
    do_pathway_clustering : bool, default True
        Whether to run pathway enrichment analysis
    do_groupwise_de : bool, default False
        Whether to run group-wise differential expression analysis
    do_dpt : bool, default False
        Whether to compute diffusion pseudotime
    analysis_name : str, default "single_dataset"
        Name for this analysis (used in output file names)
    batch_key : str, optional
        Batch key for integration (if None, no batch correction)
    integration_method : str, optional
        Integration method ("bbknn" or None)
    geo_json_path : str | Path, optional
        Path to GEO metadata JSON file (e.g., GSE208653_metadata.json). 
        If None, will automatically search for GEO JSON files in single_10x_dir.
    logos_dir : str | Path, optional
        Directory containing logo files for the report. If None, uses default logos directory.
    generate_report : bool, default True
        Whether to generate the HTML/PDF report after pipeline completion.
        Report will be generated even if GEO JSON is not found (with limited metadata).
    prepare_for_bisque : bool, default True
        Whether to automatically prepare the output h5ad file for Bisque deconvolution.
        If True, will create a Bisque-ready version of the processed h5ad file.
    
    Returns
    -------
    Path
        Path to the output directory
        
    Examples
    --------
    >>> from singlecell_10x import run_pipeline
    >>> output_dir = run_pipeline(
    ...     single_10x_dir="/path/to/10x/data",
    ...     sample_label="Sample1",  # Optional: will use directory name if not provided
    ...     group_label="CASE",
    ...     out_name="my_results",
    ...     logos_dir="/path/to/logos"  # Optional
    ... )
    """
    # Convert to Path
    single_10x_dir = Path(single_10x_dir)
    
    # Extract sample_label from directory name if not provided
    if sample_label is None:
        sample_label = single_10x_dir.name

    analysis_name = sample_label

    if not single_10x_dir.exists():
        raise FileNotFoundError(f"10x data directory not found: {single_10x_dir}")
    
    # Set up output directory
    combined_out_dir = Path(out_name)
    combined_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = combined_out_dir / "pipeline.log"
    
    # Remove existing handlers to avoid duplicates
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t[%(levelname)s]\t%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    pipeline_logger = logging.getLogger()
    pipeline_logger.info(f"Logging to file: {log_file}")
    pipeline_logger.info(f"Input directory: {single_10x_dir}")
    pipeline_logger.info(f"Output directory: {combined_out_dir}")
    pipeline_logger.info(f"Sample label: {sample_label}")
    if group_label:
        pipeline_logger.info(f"Group label: {group_label}")
    
    # Load data
    adata_single_raw = load_10x_feature_barcode_matrix(single_10x_dir)
    
    # Add metadata
    adata_single_raw.obs["sample"] = sample_label
    if group_label is not None:
        adata_single_raw.obs["group"] = group_label
    
    # Run pipeline
    run_scanpy_pipeline(
        adata_single_raw,
        combined_out_dir,
        analysis_name=analysis_name,
        batch_key=batch_key,
        integration_method=integration_method,
        do_groupwise_de=do_groupwise_de,
        do_dpt=do_dpt,
        do_pathway_clustering=do_pathway_clustering,
    )
    
    pipeline_logger.info(
        "DONE — Full single-cell pipeline (10x-only, single dataset) finished with structured outputs."
    )
    
    # Prepare for Bisque deconvolution if requested
    if prepare_for_bisque:
        try:
            processed_h5ad = combined_out_dir / f"{analysis_name}_processed_scanpy_output.h5ad"
            if processed_h5ad.exists():
                pipeline_logger.info("Preparing h5ad file for Bisque deconvolution...")
                process_h5ad_file(processed_h5ad)
                pipeline_logger.info("Bisque preparation completed successfully.")
            else:
                pipeline_logger.warning(
                    f"Expected h5ad file not found: {processed_h5ad}. "
                    "Skipping Bisque preparation."
                )
        except Exception as e:
            pipeline_logger.error(f"Error preparing for Bisque: {e}", exc_info=True)
            pipeline_logger.warning("Continuing without Bisque preparation.")
    
    # Generate report if requested
    if generate_report:
        try:
            from .singlecell_sc_report_generation import build_singlecell_report
            
            # Auto-detect GEO JSON if not provided
            detected_geo_json = None
            if geo_json_path is None:
                detected_geo_json = _find_geo_json(single_10x_dir)
                if detected_geo_json:
                    pipeline_logger.info(f"Auto-detected GEO JSON file: {detected_geo_json}")
                else:
                    pipeline_logger.info(
                        "No GEO JSON file found in input directory. Report will be generated without GEO metadata."
                    )
            else:
                detected_geo_json = Path(geo_json_path)
                if not detected_geo_json.exists():
                    pipeline_logger.warning(
                        f"Specified GEO JSON file not found: {detected_geo_json}. "
                        "Attempting to auto-detect..."
                    )
                    detected_geo_json = _find_geo_json(single_10x_dir)
                    if detected_geo_json:
                        pipeline_logger.info(f"Auto-detected GEO JSON file: {detected_geo_json}")
                    else:
                        pipeline_logger.info(
                            "No GEO JSON file found. Report will be generated without GEO metadata."
                        )
            
            pipeline_logger.info("Generating single-cell report...")
            logos_path = Path(logos_dir) if logos_dir else None
            
            build_singlecell_report(
                sc_root=combined_out_dir,
                geo_json_path=detected_geo_json,
                case_id=sample_label,
                logos_dir=logos_path,
            )
            pipeline_logger.info("Report generation completed successfully.")
        except Exception as e:
            pipeline_logger.error(f"Error generating report: {e}", exc_info=True)
            pipeline_logger.warning("Continuing without report generation.")
    
    return combined_out_dir


# def main():
#     """CLI entry point - parses CLI arguments and runs pipeline"""
#     from .config_cli import parse_cli_args
    
#     # Parse CLI arguments
#     args = parse_cli_args()
    
#     # Set up logging
#     single_10x_dir = Path(args.single_10x_dir)
#     combined_out_dir = single_10x_dir / args.out_name
#     combined_out_dir.mkdir(parents=True, exist_ok=True)
    
#     log_file = combined_out_dir / "pipeline.log"
    
#     # Remove existing handlers to avoid duplicates
#     for h in logging.root.handlers[:]:
#         logging.root.removeHandler(h)
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s\t[%(levelname)s]\t%(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode="w", encoding="utf-8"),
#             logging.StreamHandler(sys.stdout),
#         ],
#     )
    
#     cli_logger = logging.getLogger()
#     cli_logger.info(f"Logging to file: {log_file}")
#     cli_logger.info(f"Input directory: {single_10x_dir}")
#     cli_logger.info(f"Output directory: {combined_out_dir}")
#     cli_logger.info(f"Sample label: {args.single_sample_label}")
#     if args.single_group_label:
#         cli_logger.info(f"Group label: {args.single_group_label}")
    
#     # Run pipeline using the programmatic function
#     run_pipeline(
#         single_10x_dir=single_10x_dir,
#         sample_label=args.single_sample_label,
#         group_label=args.single_group_label,
#         out_name=args.out_name,
#         do_pathway_clustering=not args.no_pathway_clustering,
#         do_groupwise_de=False,
#         do_dpt=False,
#         analysis_name="single_dataset",
#         batch_key=None,
#         integration_method=None,
#     )
    
#     cli_logger.info(
#         "DONE — Full single-cell pipeline (10x-only, single dataset) finished with structured outputs."
#     )


# if __name__ == "__main__":
#     main()
