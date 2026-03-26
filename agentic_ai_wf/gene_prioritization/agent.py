import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from agents import Runner
from .tools.filter_tools import build_filter_agent, FilterContext
# Removed plotting_tools import as plotting is no longer needed
from .tools.merge_tools import combine_degs_matrix, annotate_with_hgnc

from .helpers import logger

# Safety valve if queue bookkeeping ever desyncs from disk (prevents infinite Celery burn).
_MAX_FILTER_QUEUE_ROUNDS = 5


def _matching_filtered_csvs(output_dir: Path, deg_file: Path) -> List[Path]:
    """
    Find filtered output CSVs for this raw DEG file.

    save_filtered_degs may use folder casing from context (mixed case); process_single_filter
    historically expected parent.name.upper(). Linux is case-sensitive, so match case-insensitively.
    """
    if not output_dir.is_dir():
        return []
    needle = f"filtered_{deg_file.parent.name}_{deg_file.stem}_".casefold()
    return sorted(
        p
        for p in output_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and p.name.casefold().startswith(needle)
    )


@dataclass
class ProcessingStats:
    """Track processing statistics for monitoring"""
    files_filtered: int = 0
    errors: int = 0
    total_time: float = 0.0

def get_files_to_process(raw_dirs: List[Path], filtered_done: Set[str], output_dir: Path) -> List[Path]:
    """Get all files that need filtering in batch. Only processes files ending with 'DEGs'."""
    logger.info(f"Getting files to process from {raw_dirs}")
    files_to_filter = []
    valid_extensions = {".csv", ".tsv", ".xls", ".xlsx"}
    
    for rd in raw_dirs:
        if not rd.exists():
            logger.warning(f"Directory {rd} does not exist, skipping")
            continue
            
        # Get all files in the directory
        # Ensure rd is absolute and resolve all file paths to absolute
        rd = rd.resolve()
        all_files = [f.resolve() for f in sorted(rd.iterdir()) if f.is_file()]
        logger.info(f"All files in {rd.name}: {[f.name for f in all_files]}")
        logger.debug(f"Full paths: {[str(f) for f in all_files]}")
        
        for f in all_files:
            # Check file extension
            if f.suffix.lower() not in valid_extensions:
                logger.info(f"Skipping {f.name} - invalid extension")
                continue
            
            # Check if filename ends with "DEGs" (before extension)
            if not f.stem.endswith("DEGs"):
                logger.info(f"Skipping {f.name} - does not end with 'DEGs'")
                continue

            # Build the prefix to match the saved output naming (folder uppercased)
            folder_upper = f.parent.name.upper()
            prefix = f"filtered_{folder_upper}_{f.stem}"
            matches_disk = _matching_filtered_csvs(output_dir, f)

            already_file = bool(matches_disk)
            already_done = (
                any(
                    fn.is_file()
                    and fn.name.casefold().startswith(prefix.casefold())
                    and fn.name.endswith(".csv.done")
                    for fn in output_dir.iterdir()
                )
                if output_dir.exists()
                else False
            )

            in_memory_done = any(
                str(n).casefold().startswith(prefix.casefold()) for n in filtered_done
            )
            if not in_memory_done and not (already_file or already_done):
                files_to_filter.append(f)
                logger.info(f"Added {f.name} to processing queue")
            else:
                logger.info(f"Skipping {f.name} - already processed")
     
    logger.info(f"Total files to process: {len(files_to_filter)}")
    return files_to_filter


# Removed get_plots_to_process function as plotting is no longer needed

    
# ── agent.py ──
def process_single_filter(
    args: Tuple[Path, Path, object, bool]
) -> Tuple[bool, str, str]:
    file_path, output_dir, filter_agent, causal = args

    # Ensure file_path is absolute and exists
    file_path = file_path.resolve()
    if not file_path.exists():
        error_msg = f"File not found: {file_path} (parent directory exists: {file_path.parent.exists()})"
        logger.error(error_msg)
        return False, file_path.name, error_msg

    # ensure an asyncio loop exists in this thread; track if we created it
    created_loop = False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        created_loop = True

    try:
        # Use absolute path as string to ensure full path is preserved
        file_path_str = str(file_path.resolve())
        logger.debug(f"Processing file: {file_path_str}")
        
        out_dir = Path(output_dir).resolve()
        result = Runner.run_sync(
            filter_agent,
            input=(
                "Load, filter, and save DEGs.\n"
                "IMPORTANT: Call load_expression_file using file_path EXACTLY equal to:\n"
                f"{file_path_str}\n"
                "Do not alter this path (no extra characters).\n"
                f"Call save_filtered_degs with output_dir EXACTLY:\n{out_dir}\n"
            ),
            context=FilterContext(
                causal=causal,
                canonical_input_path=file_path_str,
                canonical_output_dir=str(out_dir),
            ),
        )
        folder_name = file_path.parent.name.upper()
        prefix = f"filtered_{folder_name}_{file_path.stem}"
        matches = _matching_filtered_csvs(out_dir, file_path)
        if not matches or matches[-1].stat().st_size == 0:
            logger.error(
                "Filter agent did not write expected CSV (prefix %r, case-insensitive) under %s",
                prefix,
                out_dir,
            )
            return (
                False,
                file_path.name,
                result.final_output
                or f"Expected filtered output with prefix {prefix} not created under {out_dir}",
            )
        return True, matches[-1].name, result.final_output
    except Exception as e:
        logger.error("Error filtering %s: %s", file_path, e)
        return False, file_path.name, str(e)
    finally:
        # Clean up the loop we created to avoid Windows overlapped future warnings
        if created_loop:
            try:
                asyncio.get_event_loop().close()
                asyncio.set_event_loop(None)
            except Exception as cleanup_err:
                logger.debug("Event loop cleanup warning: %s", cleanup_err)


# Removed process_single_plot function as plotting is no longer needed


def batch_process_files(files: List[Path], output_dir: Path, agent, process_func, 
                       max_workers: int = 4, causal: bool = False) -> Tuple[Set[str], List[str]]:
    """Process multiple files in parallel using ThreadPoolExecutor"""
    if not files:
        return set(), []
    
    completed_files = set()
    results = []
    
    # Prepare arguments for parallel processing
    args_list = [(f, output_dir, agent, causal) for f in files]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_func, args): args[0] 
            for args in args_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                success, output_name, result_output = future.result()
                if success:
                    completed_files.add(output_name)
                    results.append(result_output)
                    logger.info(f"Successfully processed: {file_path.name}")
                else:
                    logger.error(f"Failed to process: {file_path.name}")
            except Exception as e:
                logger.error(f"Exception processing {file_path.name}: {str(e)}")
    
    return completed_files, results

# Removed autonomous_filter_and_plot function as plotting is no longer needed

def autonomous_filter_only(
    patient_dir: Path,
    cohort_dir: Path,
    output_dir: Path,
    filter_agent,
    max_workers: int = 4,
    batch_size: int = 10,
    causal: bool = False
) -> ProcessingStats:
    """
    Simplified version: Only run FilterAgent on raw files, no plotting.
    
    Args:
        patient_dir: Directory containing patient data files
        cohort_dir: Directory containing cohort data files  
        output_dir: Directory to save processed files
        filter_agent: Agent for filtering operations
        max_workers: Maximum number of parallel workers (default: 4)
        batch_size: Maximum files to process in each batch (default: 10)
    
    Returns:
        ProcessingStats: Statistics about the processing run
    """
    import time
    start_time = time.time()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dirs = [patient_dir, cohort_dir]
    filtered_done = set()
    stats = ProcessingStats()
    
    logger.info(f"Starting filter-only processing with {max_workers} workers, batch size {batch_size}")

    queue_round = 0
    while True:
        queue_round += 1
        if queue_round > _MAX_FILTER_QUEUE_ROUNDS:
            logger.error(
                "autonomous_filter_only: exceeded %s queue rounds; aborting",
                _MAX_FILTER_QUEUE_ROUNDS,
            )
            break
        files_to_filter = get_files_to_process(raw_dirs, filtered_done, output_dir)
        
        if files_to_filter:
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(files_to_filter), batch_size):
                batch = files_to_filter[i:i + batch_size]
                logger.info(f"Processing filter batch {i//batch_size + 1}: {len(batch)} files")
                
                completed, results = batch_process_files(
                    batch, output_dir, filter_agent, process_single_filter, max_workers, causal=causal
                )
                
                filtered_done.update(completed)
                stats.files_filtered += len(completed)
                # Write .done markers for each completed CSV to prevent re-queuing when no rows saved downstream
                for out_name in completed:
                    done_marker = output_dir / f"{out_name}.done"
                    try:
                        done_marker.touch(exist_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to write done marker {done_marker.name}: {e}")
                
                # Print results for compatibility with original function
                for result in results:
                    print(result)
            
            # Continue to next iteration to check for more files
            continue
        
        # No more files to process
        break
    
    stats.total_time = time.time() - start_time
    
    logger.info(f"Filter processing complete! Filtered: {stats.files_filtered}, Time: {stats.total_time:.2f}s")
    
    return stats

def autonomous_filter_patient_only(
    patient_dir: Path,
    output_dir: Path,
    filter_agent,
    max_workers: int = 4,
    batch_size: int = 10,
    causal: bool = False
) -> ProcessingStats:
    """
    Process only patient data without cohort comparison - filter only.
    
    Args:
        patient_dir: Directory containing patient data files
        output_dir: Directory to save processed files
        filter_agent: Agent for filtering operations
        max_workers: Maximum number of parallel workers (default: 4)
        batch_size: Maximum files to process in each batch (default: 10)
    
    Returns:
        ProcessingStats: Statistics about the processing run
    """
    import time
    start_time = time.time()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dirs = [patient_dir]  # Only patient directory
    filtered_done = set()
    stats = ProcessingStats()
    
    logger.info(f"Starting patient-only filter processing with {max_workers} workers, batch size {batch_size}")

    queue_round = 0
    while True:
        queue_round += 1
        if queue_round > _MAX_FILTER_QUEUE_ROUNDS:
            logger.error(
                "autonomous_filter_patient_only: exceeded %s queue rounds; aborting",
                _MAX_FILTER_QUEUE_ROUNDS,
            )
            break
        files_to_filter = get_files_to_process(raw_dirs, filtered_done, output_dir)
        
        if files_to_filter:
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(files_to_filter), batch_size):
                batch = files_to_filter[i:i + batch_size]
                logger.info(f"Processing filter batch {i//batch_size + 1}: {len(batch)} files")
                
                completed, results = batch_process_files(
                    batch, output_dir, filter_agent, process_single_filter, max_workers, causal=causal
                )
                
                filtered_done.update(completed)
                stats.files_filtered += len(completed)
                for out_name in completed:
                    done_marker = output_dir / f"{out_name}.done"
                    try:
                        done_marker.touch(exist_ok=True)
                    except Exception as e:
                        logger.warning(
                            "Failed to write done marker %s: %s", done_marker.name, e
                        )

                for result in results:
                    print(result)

            continue

        break

    stats.total_time = time.time() - start_time

    logger.info(
        "Patient-only filter processing complete! Filtered: %s, Time: %.2fs",
        stats.files_filtered,
        stats.total_time,
    )
    
    return stats

# Removed autonomous_filter_and_plot_with_merge and autonomous_filter_and_plot_original functions as plotting is no longer needed

# Removed autonomous_filter_and_plot_patient_only function as plotting is no longer needed