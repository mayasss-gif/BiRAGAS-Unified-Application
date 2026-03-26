"""
Gene Prioritization Pipeline
Entry point for DEGs processing pipeline with efficient error handling and performance monitoring.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List

from .helper_tools.file_utils import resolve_and_validate_directory
from .agent import autonomous_filter_patient_only
from .tools.filter_tools import build_filter_agent
from .tools.merge_tools import combine_degs_matrix, annotate_with_hgnc
from .helpers import logger

# Removed asyncio imports for Celery compatibility


class PerformanceMonitor:
    """Context manager for timing operations and logging performance metrics."""
    
    def __init__(self, operation_name: str, logger_instance=None):
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {elapsed:.2f} seconds")
        else:
            self.logger.error(f"Failed {self.operation_name} after {elapsed:.2f} seconds: {exc_val}")


def find_deg_files(directory: Path) -> List[Path]:
    """
    Find DEG files in a directory that end with "DEGs" (before file extension).
    Only files ending with "DEGs" will be returned, all other files are ignored.
    Files without a "Gene" or "gene" column will be skipped.
    Files with empty dataframes or all NaN values will be skipped.
    
    Args:
        directory: Directory to search for DEG files
        
    Returns:
        List of paths to DEG files with valid gene columns and non-empty data
        
    Raises:
        ValueError: If no valid DEG files are found
    """
    import pandas as pd
    
    # Get all files in the directory
    all_files = [f for f in directory.iterdir() if f.is_file()]
    logger.info(f"All files in {directory.name}: {[f.name for f in all_files]}")
    
    # Filter for only files ending with "DEGs" (before extension)
    deg_files = [f for f in all_files if f.stem.endswith("DEGs")]
    
    if not deg_files:
        raise ValueError(f"No DEG files found in {directory}. Expected files with names ending with 'DEGs' (before extension)")
    
    # Validate that each DEG file has a "Gene" or "gene" column and contains valid data
    valid_deg_files = []
    skipped_files = []
    
    for deg_file in deg_files:
        try:
            # Read the entire file to check for data quality
            df = pd.read_csv(deg_file)
            
            # Check if any column name contains "gene" (case insensitive)
            has_gene_column = any('gene' in col.lower() for col in df.columns)
            
            if not has_gene_column:
                skipped_files.append(deg_file.name)
                logger.warning(f"Skipping {deg_file.name}: no 'Gene' or 'gene' column found. Available columns: {list(df.columns)}")
                continue
            
            # Check if dataframe is empty
            if df.empty:
                skipped_files.append(deg_file.name)
                logger.warning(f"Skipping {deg_file.name}: dataframe is empty (0 rows)")
                continue
            
            # Check if all values in the dataframe are NaN (excluding the gene column)
            gene_col = None
            for col in df.columns:
                if 'gene' in col.lower():
                    gene_col = col
                    break
            
            if gene_col is None:
                skipped_files.append(deg_file.name)
                logger.warning(f"Skipping {deg_file.name}: gene column not found despite previous check")
                continue
            
            # Check if all rows have NaN values in all columns except the gene column
            data_columns = [col for col in df.columns if col != gene_col]
            if data_columns:
                # Check if all data columns are all NaN
                all_nan_mask = df[data_columns].isna().all(axis=1)
                if all_nan_mask.all():
                    skipped_files.append(deg_file.name)
                    logger.warning(f"Skipping {deg_file.name}: all data columns contain only NaN values")
                    continue
                
                # Check if there are any non-NaN values in the dataframe (excluding gene column)
                if df[data_columns].isna().all().all():
                    skipped_files.append(deg_file.name)
                    logger.warning(f"Skipping {deg_file.name}: all data columns are completely NaN")
                    continue
            
            # If we reach here, the file is valid
            valid_deg_files.append(deg_file)
            logger.debug(f"Validated {deg_file.name}: contains gene column and valid data ({len(df)} rows)")
                
        except Exception as e:
            skipped_files.append(deg_file.name)
            logger.warning(f"Skipping {deg_file.name}: error reading file - {e}")
    
    if not valid_deg_files:
        raise ValueError(f"No valid DEG files found in {directory}. All files either missing 'Gene' column, unreadable, empty, or contain only NaN values. Skipped files: {skipped_files}")
    
    # Log which files are being picked and which are being ignored
    ignored_files = [f.name for f in all_files if not f.stem.endswith("DEGs")]
    if ignored_files:
        logger.info(f"Ignored non-DEG files in {directory.name}: {ignored_files}")
    
    if skipped_files:
        logger.info(f"Skipped invalid DEG files in {directory.name}: {skipped_files}")
    
    logger.info(f"Selected valid DEG file(s) in {directory.name}: {[f.name for f in valid_deg_files]}")
    return valid_deg_files


def discover_directories(base_dir: Path, analysis_id: str) -> Tuple[Path, List[Path]]:
    """
    Discover patient and cohort directories from a base directory.
    
    Args:
        base_dir: Base directory containing patient and cohort subdirectories
        
    Returns:
        Tuple of (analysis_dir, list_of_cohort_dirs) - cohort_dirs can be empty list
        
    Raises:
        ValueError: If required patient directory is not found
    """
    try:
        # Resolve and validate base directory
        base_dir = resolve_and_validate_directory(base_dir)
        logger.info(f"Base directory validated: {base_dir}")
        
        # Find patient directory (starts with analysis_id)
        aid = (analysis_id or "").strip().lower()
        if aid:
            analysis_dirs = [
                d for d in base_dir.iterdir()
                if d.is_dir() and d.name.lower().startswith(aid)
            ]
        else:
            analysis_dirs = []
        
        if not analysis_dirs:
            # Fallback: check if base_dir itself contains DEG files (flat structure)
            logger.info(f"No subdirectory starting with '{analysis_id}' found in {base_dir}")
            deg_files_in_base = find_deg_files(base_dir)
            if deg_files_in_base:
                logger.info(f"Found {len(deg_files_in_base)} DEG files directly in base_dir. Using base_dir as analysis_dir.")
                analysis_dir = base_dir
                patient_deg_files = deg_files_in_base
            else:
                raise ValueError(
                    f"No patient directory found in {base_dir}. "
                    f"Expected either:\n"
                    f"  1. A subdirectory starting with '{analysis_id}' containing DEG CSV files\n"
                    f"  2. DEG CSV files directly in {base_dir}"
                )
        else:
            if len(analysis_dirs) > 1:
                logger.warning(f"Multiple patient directories found: {[d.name for d in analysis_dirs]}. Using the first one: {analysis_dirs[0].name}")
            
            analysis_dir = analysis_dirs[0]
            logger.info(f"Found patient directory: {analysis_dir}")
            
            # Validate that patient directory contains DEG files
            patient_deg_files = find_deg_files(analysis_dir)
        
        # Cohort directories no longer used - always run patient data only
        logger.info("Running pipeline with patient data only (cohort comparison disabled)")
        return analysis_dir, []
        
    except Exception as exc:
        logger.error(f"Directory discovery failed: {exc}")
        raise ValueError(f"Directory discovery failed: {exc}")


def validate_directories(analysis_dir: Path, cohort_dirs: List[Path]) -> Tuple[Path, List[Path]]:
    """
    Validate patient and cohort directories.
    
    Args:
        analysis_dir: Path to patient directory
        cohort_dirs: List of paths to cohort directories
        
    Returns:
        Tuple of (validated_analysis_dir, validated_cohort_dirs)
        
    Raises:
        ValueError: If any directory validation fails
    """
    try:
        # Validate patient directory
        validated_analysis_dir = resolve_and_validate_directory(analysis_dir)
        
        # Validate all cohort directories
        validated_cohort_dirs = []
        for cohort_dir in cohort_dirs:
            validated_cohort_dir = resolve_and_validate_directory(cohort_dir)
            validated_cohort_dirs.append(validated_cohort_dir)
        
        logger.info(f"Directory validation completed successfully - Patient: {validated_analysis_dir}, Cohorts: {len(validated_cohort_dirs)}")
        return validated_analysis_dir, validated_cohort_dirs
        
    except Exception as exc:
        logger.error(f"Directory validation failed: {exc}")
        raise ValueError(f"Directory validation failed: {exc}")


def setup_output_directory(output_dir: Path) -> Path:
    """
    Setup and validate output directory with error handling.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Resolved output directory path
        
    Raises:
        OSError: If directory creation fails
    """
    try:
        resolved_output_dir = output_dir.expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify write permissions
        test_file = resolved_output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        logger.info(f"Output directory ready: {resolved_output_dir}")
        return resolved_output_dir
        
    except Exception as exc:
        logger.error(f"Failed to setup output directory {output_dir}: {exc}")
        raise


def execute_merge_and_annotate(output_dir: Path) -> str:
    """
    Execute merge and annotation steps with error handling and monitoring.
    
    Args:
        output_dir: Output directory containing filtered results
        
    Returns:
        Path to the final annotated file
    """
    # Define file paths
    combined_csv = output_dir / "combined_DEGs_matrix.csv"
    annotated_csv = output_dir / "combined_DEGs_matrix_annotated.csv"
    
    try:
        # Step 1: Combine DEGs matrix
        with PerformanceMonitor("DEGs matrix combination"):
            result_message = combine_degs_matrix(filtered_folder=output_dir, output_path=combined_csv)
            logger.info(f"Matrix combination result: {result_message}")
            
            # Verify output file exists and has content
            if not combined_csv.exists():
                raise FileNotFoundError(f"Expected combined matrix file not created: {combined_csv}")
            
            file_size = combined_csv.stat().st_size
            if file_size == 0:
                raise ValueError(f"Combined matrix file is empty: {combined_csv}")
                
            logger.info(f"Combined matrix file created successfully ({file_size:,} bytes)")
        
        # Step 2: HGNC annotation
        with PerformanceMonitor("HGNC annotation"):
            annotation_result = annotate_with_hgnc(combined_input_path=combined_csv, output_path=annotated_csv)
            logger.info(f"Annotation result saved to: {annotation_result}")
            
            # Verify annotated output
            if not annotated_csv.exists():
                raise FileNotFoundError(f"Expected annotated file not created: {annotated_csv}")
                
            annotated_size = annotated_csv.stat().st_size
            logger.info(f"Annotated file created successfully ({annotated_size:,} bytes)")
            
        return annotation_result
    except Exception as exc:
        logger.error(f"Merge and annotation failed: {exc}")
        raise


def run_hormonizer(base_dir: Path, output_dir: Path = None, analysis_id: str = None, causal: bool = False) -> str:
    """
    Main function with efficient error handling and performance monitoring.
    
    Args:
        base_dir: Base directory containing patient DEG files
        output_dir: Output directory path (optional, defaults to base_dir/output)
        analysis_id: Analysis ID to match directory name (optional)
        causal: If True, skip DEG thresholding inside filtering
        
    Returns:
        Path to the final annotated output file
    """
    start_time = time.perf_counter()
    
    try:
        logger.info("Starting DEGs processing pipeline (patient data only)...")
        
        # Discover and validate patient directory
        with PerformanceMonitor("Directory discovery and validation"):
            analysis_dir, _ = discover_directories(base_dir, analysis_id=analysis_id)
            analysis_dir, _ = validate_directories(analysis_dir, [])
        
        # Setup output directory (default to base_dir/output if not specified)
        if output_dir is None:
            output_dir = base_dir / "output"
        
        with PerformanceMonitor("Output directory setup"):
            output_dir = setup_output_directory(output_dir)

        # Build filter agent
        with PerformanceMonitor("Agent initialization"):
            filter_agent = build_filter_agent()

        # Execute patient-only filtering (no cohort comparison)
        with PerformanceMonitor("Filter operations"):
            logger.info("Processing patient data only (cohort comparison disabled)")
            autonomous_filter_patient_only(analysis_dir, output_dir, filter_agent, causal=causal)

        # Execute merge and annotation
        with PerformanceMonitor("Merge and annotation operations"):
            final_output_path = execute_merge_and_annotate(output_dir)
        
        # Final success message
        total_time = time.perf_counter() - start_time
        logger.info(f"✅ DEGs processing pipeline completed successfully in {total_time:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Processed patient data only (cohort comparison disabled)")
        logger.info(f"Final output path: {final_output_path}")

        return final_output_path
        
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        print(f"❌ Pipeline failed: {exc}")
        raise


# if __name__ == "__main__":
#     # Example usage - replace with actual path as needed
#     base_directory = Path(r"C:\Ayass Bio Work\Hormonizer Test run")
#     run_hormonizer(base_directory, Path("ANALYSIS_DIR"))