import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import glob, os
from ..helper_tools.data_utils import drop_na_in_columns, validate_numeric_column
from ..helper_tools.file_utils import resolve_and_validate_file

from ..helpers import logger


def _process_single_file(file_path: Path) -> Optional[Tuple[str, pd.DataFrame, Dict[str, int]]]:
    """
    Process a single filtered CSV file in parallel.
    Returns (tag, processed_dataframe, stats) or None if processing fails.
    """
    try:
        start_time = time.time()
        tag = file_path.stem.replace("filtered_", "")
        
        # Optimized CSV reading with specific dtypes for better performance
        try:
            df = pd.read_csv(
                file_path, 
                low_memory=False,
                dtype={'Gene': 'string'},  # Use string dtype for gene names
                na_values=['', 'NA', 'N/A', 'null', 'NULL', 'nan']
            )
        except Exception as read_exc:
            logger.warning("Failed to read %s: %s", file_path.name, read_exc)
            return None
        
        initial_rows = len(df)
        
        # Enhanced column requirement checking
        req_cols = {"Gene", "log2FC", "p-value"}
        available_cols = set(df.columns)
        missing_cols = req_cols - available_cols
        
        if missing_cols:
            logger.warning("Skipping %s: missing columns %s", file_path.name, missing_cols)
            return None
        
        # Optimized data cleaning pipeline
        stats = {
            'initial_rows': initial_rows,
            'after_duplicate_removal': 0,
            'after_na_removal': 0,
            'after_validation': 0,
            'final_rows': 0,
            'processing_time': 0
        }
        
        # Remove duplicates early to reduce processing load
        df = df.drop_duplicates("Gene", keep='first')  # Keep first occurrence
        stats['after_duplicate_removal'] = len(df)
        
        if stats['after_duplicate_removal'] < initial_rows:
            duplicates_removed = initial_rows - stats['after_duplicate_removal']
            logger.debug("Removed %d duplicate genes from %s", duplicates_removed, file_path.name)
        
        # Enhanced NA handling with targeted column cleaning
        df = drop_na_in_columns(df, ["Gene", "log2FC", "p-value"])
        stats['after_na_removal'] = len(df)
        
        if df.empty:
            logger.warning("No valid rows remaining in %s after NA removal", file_path.name)
            return None
        
        # Vectorized numeric validation (more efficient than individual validation)
        try:
            # Convert to numeric with error handling
            df['log2FC'] = pd.to_numeric(df['log2FC'], errors='coerce')
            df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')
            
            # Single vectorized finite check for both columns
            finite_mask = (
                np.isfinite(df['log2FC']) & 
                np.isfinite(df['p-value']) &
                df['log2FC'].notna() &
                df['p-value'].notna()
            )
            
            df = df[finite_mask].reset_index(drop=True)
            stats['after_validation'] = len(df)
            
        except Exception as val_exc:
            logger.warning("Validation failed for %s: %s", file_path.name, val_exc)
            return None
        
        if df.empty:
            logger.warning("No valid numeric data in %s", file_path.name)
            return None
        
        # Additional data quality checks
        # Check for reasonable p-value ranges
        invalid_pvals = ((df['p-value'] < 0) | (df['p-value'] > 1)).sum()
        if invalid_pvals > 0:
            logger.warning("%s has %d invalid p-values outside [0,1] range", 
                          file_path.name, invalid_pvals)
            df = df[(df['p-value'] >= 0) & (df['p-value'] <= 1)]
        
        # Check for extreme log2FC values (potential outliers)
        extreme_fc = (np.abs(df['log2FC']) > 20).sum()
        if extreme_fc > 0:
            logger.info("%s has %d genes with |log2FC| > 20", file_path.name, extreme_fc)
        
        #column renaming
        # --- column renaming with parent-folder tag in [brackets] ---
        # Extract the raw basename and its parent folder name
        basename = file_path.stem.replace("filtered_", "")
        tag = basename

    # Rename metrics to {tag}_{col}, so merge/status code can find them
        rename_map = {}
        for col in ("log2FC", "p-value", "adj-p-value"):
           if col in df.columns:
                rename_map[col] = f"{tag}_{col}"
        if rename_map:
            df.rename(columns=rename_map, inplace=True, copy=False)


        
        stats['final_rows'] = len(df)
        stats['processing_time'] = time.time() - start_time
        
        logger.debug("Processed %s: %d → %d rows in %.3fs", 
                    file_path.name, initial_rows, stats['final_rows'], stats['processing_time'])
        
        return tag, df, stats
        
    except Exception as exc:
        logger.error("Error processing %s: %s", file_path.name, exc)
        return None


def _compute_status_vectorized(merged_df: pd.DataFrame, tags: List[str]) -> pd.Series:
    """
    Vectorized computation of gene status column for better performance.
    """
    # Create a matrix of presence indicators for all tags
    presence_matrix = pd.DataFrame(index=merged_df.index)
    
    for tag in tags:
        col_name = f"{tag}_log2FC"
        if col_name in merged_df.columns:
            presence_matrix[tag] = ~merged_df[col_name].isna()
    
    # Vectorized status computation
    def compute_status_row(row_idx: int) -> str:
        present_tags = [tag for tag in tags if presence_matrix.iloc[row_idx][tag]]
        
        if len(present_tags) == 0:
            return "No Gene"
        elif len(present_tags) == 1:
            return f"Unique to {present_tags[0]}"
        else:
            return "Shared by " + ", ".join(sorted(present_tags))
    
    # Use list comprehension for better performance than apply
    status_list = [compute_status_row(i) for i in range(len(merged_df))]
    return pd.Series(status_list, index=merged_df.index, name='Status')


def combine_degs_matrix(
    filtered_folder: Union[str, Path],
    output_path: Union[str, Path],
    max_workers: Optional[int] = None,
    chunk_size: int = 1000
) -> str:
    """
    Merge all filtered_*.csv into combined_DEGs_matrix.csv with a 'Status' column.
    Enhanced with parallel processing, vectorized operations, and comprehensive statistics.
    
    Args:
        filtered_folder: Directory containing filtered CSV files
        output_path: Output path for combined matrix
        max_workers: Maximum number of parallel workers (None = auto-detect)
        chunk_size: Chunk size for batch operations
    """
    start_time = time.time()
    
    pattern = os.path.join(filtered_folder, "filtered_*.csv")
    filtered_files = glob.glob(pattern)
    print("⎯⎯⎯ merging these files:", filtered_files)
    try:
        # Enhanced input validation
        folder = Path(filtered_folder).expanduser().resolve()
        if not folder.exists():
            logger.error("Filtered folder does not exist: %s", folder)
            return f"Error: Filtered folder not found: {folder}"
            
        if not folder.is_dir():
            logger.error("Path is not a directory: %s", folder)
            return "Error: Provided path is not a directory."

        # Enhanced file discovery with better filtering
        files = sorted([
            p for p in folder.iterdir() 
            if (p.name.startswith("filtered_") and 
                p.suffix.lower() == ".csv" and 
                p.is_file() and 
                p.stat().st_size > 0)  # Skip empty files
        ], key=lambda p: p.name)
        
        if not files:
            logger.warning("No filtered CSV files found in %s", folder)
            return "No filtered files to merge."
        
        logger.info("Found %d filtered files to process in %s", len(files), folder)
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(files), mp.cpu_count())
        
        logger.info("Using %d parallel workers for file processing", max_workers)
        
        # Parallel file processing
        processed_data: Dict[str, pd.DataFrame] = {}
        processing_stats = []
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(_process_single_file, file_path): file_path 
                for file_path in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        tag, df, stats = result
                        processed_data[tag] = df
                        processing_stats.append({
                            'file': file_path.name,
                            'tag': tag,
                            **stats
                        })
                        logger.debug("Successfully processed %s (%d genes)", 
                                   file_path.name, stats['final_rows'])
                    else:
                        failed_files.append(file_path.name)
                except Exception as exc:
                    logger.error("Failed to process %s: %s", file_path.name, exc)
                    failed_files.append(file_path.name)
        
        # Report processing results
        successful_files = len(processed_data)
        if failed_files:
            logger.warning("Failed to process %d files: %s", len(failed_files), failed_files)
        
        if not processed_data:
            return "No valid filtered files for merging."
        
        logger.info("Successfully processed %d/%d files", successful_files, len(files))
        
        # Enhanced merging with memory optimization
        tags = list(processed_data.keys())
        logger.info("Merging data from tags: %s", tags)
        
        merge_start = time.time()
        merged = None
        
        # Progressive merging to handle large datasets efficiently
        for i, (tag, df) in enumerate(processed_data.items()):
            if merged is None:
                merged = df.copy()
                logger.debug("Initialized merged dataset with %s (%d genes)", tag, len(df))
            else:
                pre_merge_size = len(merged)
                merged = merged.merge(df, on="Gene", how="outer", suffixes=('', '_dup'))
                post_merge_size = len(merged)
                logger.debug("Merged %s: %d → %d genes", tag, pre_merge_size, post_merge_size)
                
                # Memory cleanup for large datasets
                if len(merged) > 100000:  # For large datasets
                    merged = merged.copy()  # Consolidate memory
        
        merge_time = time.time() - merge_start
        logger.info("Data merging completed in %.3fs, final dataset: %d genes", 
                   merge_time, len(merged))
        
        # Optimized sorting with memory efficiency
        sort_start = time.time()
        merged = merged.sort_values("Gene", kind='mergesort').reset_index(drop=True)
        sort_time = time.time() - sort_start
        logger.debug("Gene sorting completed in %.3fs", sort_time)
        
        # Vectorized status computation
        status_start = time.time()
        merged["Status"] = _compute_status_vectorized(merged, tags)
        status_time = time.time() - status_start
        logger.debug("Status computation completed in %.3fs", status_time)
        
        # Generate comprehensive statistics
        total_genes = len(merged)
        status_counts = merged["Status"].value_counts()
        
        unique_genes = sum(count for status, count in status_counts.items() 
                          if "Unique to" in status)
        shared_genes = sum(count for status, count in status_counts.items() 
                          if "Shared by" in status)
        
        logger.info("Matrix statistics - Total: %d, Unique: %d, Shared: %d", 
                   total_genes, unique_genes, shared_genes)
        
        # Enhanced output with error handling
        try:
            output_file = Path(output_path).expanduser().resolve()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Optimized CSV writing
            merged.to_csv(
                output_file, 
                index=False,
                float_format='%.6g',  # Compact float representation
                encoding='utf-8'
            )
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            total_time = time.time() - start_time
            
            logger.info("Combined matrix saved: %s (%.2f MB) in %.3fs", 
                       output_file.name, file_size_mb, total_time)
            
            # Generate comprehensive summary
            summary_lines = [
                f"Combined matrix saved to {output_file.name}:",
                f"  • {total_genes:,} total genes from {successful_files} datasets",
                f"  • {unique_genes:,} unique genes, {shared_genes:,} shared genes",
                f"  • Processing time: {total_time:.2f}s",
                f"  • File size: {file_size_mb:.2f} MB"
            ]
            
            return "\n".join(summary_lines)
            
        except PermissionError:
            error_msg = f"Error: Permission denied writing to {output_path}"
            logger.error(error_msg)
            return error_msg
        except Exception as save_exc:
            error_msg = f"Error saving combined matrix: {save_exc}"
            logger.error(error_msg)
            return error_msg
            
    except Exception as exc:
        logger.exception("Unexpected error in combine_degs_matrix: %s", exc)
        return f"Error during combine_degs_matrix: {exc}"


def annotate_with_hgnc(
    combined_input_path: Union[str, Path],
    output_path: Union[str, Path],
    retry_attempts: int = 3,
    retry_delay: int = 5,
) -> str:
    """
    Annotate the merged matrix with HGNC symbols via MyGeneInfo.
    Enhanced with batch processing, caching, and improved error handling.
    """
    import time
    import concurrent.futures
    import mygene
    from functools import lru_cache
    import logging
    from typing import Dict, List, Tuple, Optional
    import numpy as np
    
    # Set up logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        path = resolve_and_validate_file(combined_input_path, allowed_extensions={".csv"})
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Successfully loaded dataset with {len(df)} rows")
    except Exception as exc:
        logger.error(f"Error reading {combined_input_path}: {exc}")
        return f"Error reading {combined_input_path}: {exc}"

    if "Gene" not in df.columns:
        logger.error("Input missing 'Gene' column")
        return "Input missing 'Gene' column."

    # Efficient unique gene extraction with pandas
    genes = df["Gene"].astype(str).tolist()
    unique_genes = list(df["Gene"].astype(str).drop_duplicates())
    logger.info(f"Processing {len(unique_genes)} unique genes from {len(genes)} total entries")
    
    # Initialize MyGene with session reuse for better performance
    mg = mygene.MyGeneInfo()
    
    # Enhanced batch processing with optimal chunk sizes
    def process_gene_batch(gene_list: List[str], batch_size: int = 1000) -> Tuple[Dict[str, Optional[str]], Dict[str, str]]:
        """Process genes in optimized batches to avoid API limits"""
        norm_map, alias_map = {}, {}
        
        # Process in chunks to avoid overwhelming the API
        for i in range(0, len(gene_list), batch_size):
            chunk = gene_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(gene_list)-1)//batch_size + 1} ({len(chunk)} genes)")
            
            # Primary batch query with enhanced error handling
            for attempt in range(retry_attempts):
                try:
                    results = mg.querymany(
                        chunk, 
                        scopes=["symbol", "alias", "entrezgene", "ensemblgene"],
                        fields=["symbol", "alias", "entrezgene"], 
                        species="human", 
                        as_dataframe=False, 
                        returnall=True,
                        verbose=False  # Reduce API verbosity
                    )
                    break
                except Exception as e:
                    logger.warning(f"Batch query attempt {attempt + 1} failed: {e}")
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All batch query attempts failed for chunk starting at index {i}")
                        # Initialize with original gene names as fallback
                        for gene in chunk:
                            norm_map[gene] = gene
                            alias_map[gene] = ""
                        continue
            
            if 'results' in locals():
                # Process successful results
                out_hits = results.get("out", [])
                dup_genes = results.get("dup", [])
                notfound_genes = results.get("notfound", [])
                
                # Process main hits
                for entry in out_hits:
                    query = entry.get("query")
                    symbol = entry.get("symbol")
                    aliases = entry.get("alias", [])
                    
                    norm_map[query] = symbol if symbol else query
                    alias_map[query] = ";".join(aliases) if isinstance(aliases, list) else (str(aliases) if aliases else "")
                
                # Handle not found genes
                for gene in notfound_genes:
                    norm_map[gene] = gene  # Keep original name
                    alias_map[gene] = ""
                
                # Handle duplicates with refined search
                if dup_genes:
                    logger.info(f"Resolving {len(dup_genes)} duplicate entries")
                    try:
                        dup_results = mg.querymany(
                            dup_genes, 
                            scopes=["symbol", "ensemblgene"], 
                            fields=["symbol", "alias"], 
                            species="human", 
                            as_dataframe=False,
                            verbose=False
                        )
                        for entry in dup_results:
                            query = entry.get("query")
                            symbol = entry.get("symbol")
                            aliases = entry.get("alias", [])
                            
                            norm_map[query] = symbol if symbol else query
                            alias_map[query] = ";".join(aliases) if isinstance(aliases, list) else (str(aliases) if aliases else "")
                    except Exception as e:
                        logger.warning(f"Failed to resolve duplicates: {e}")
                        for gene in dup_genes:
                            norm_map[gene] = gene
                            alias_map[gene] = ""
            
            # Add small delay between batches to be respectful to API
            if i + batch_size < len(gene_list):
                time.sleep(0.1)
        
        return norm_map, alias_map
    
    # Process all unique genes in batches
    norm_map, alias_map = process_gene_batch(unique_genes)
    
    # Handle any remaining unresolved genes with parallel individual queries
    unresolved = [g for g in unique_genes if norm_map.get(g) is None or norm_map.get(g) == g and g not in alias_map]
    
    if unresolved:
        logger.info(f"Performing individual lookups for {len(unresolved)} unresolved genes")
        
        @lru_cache(maxsize=1000)
        def cached_gene_lookup(gene: str) -> Tuple[str, str]:
            """Cached individual gene lookup with multiple scope attempts"""
            for scope in ["symbol", "entrezgene", "ensemblgene", "alias"]:
                try:
                    result = mg.query(
                        gene, 
                        scopes=scope, 
                        fields=["symbol", "alias", "entrezgene"], 
                        species="human",
                        verbose=False
                    )
                    if isinstance(result, dict) and result.get("symbol"):
                        symbol = result["symbol"]
                        aliases = result.get("alias", [])
                        alias_str = ";".join(aliases) if isinstance(aliases, list) else (str(aliases) if aliases else "")
                        return symbol, alias_str
                except Exception as e:
                    logger.debug(f"Scope {scope} failed for gene {gene}: {e}")
                    continue
            return gene, ""  # Fallback to original gene name
        
        # Parallel processing for individual lookups with controlled concurrency
        max_workers = min(8, len(unresolved))  # Limit concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_gene = {executor.submit(cached_gene_lookup, gene): gene for gene in unresolved}
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_gene):
                gene = future_to_gene[future]
                try:
                    symbol, aliases = future.result(timeout=30)  # Add timeout
                    norm_map[gene] = symbol
                    alias_map[gene] = aliases
                except Exception as e:
                    logger.warning(f"Individual lookup failed for {gene}: {e}")
                    norm_map[gene] = gene
                    alias_map[gene] = ""
    
    # Efficiently map results back to original dataframe
    logger.info("Mapping results back to original dataset")
    df["Normalized_Gene"] = df["Gene"].astype(str).map(norm_map).fillna(df["Gene"])
    df["HGNC_Synonyms"] = df["Gene"].astype(str).map(alias_map).fillna("")
    
    # Save results with error handling
    try:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use efficient CSV writing
        df.to_csv(output_path, index=False, chunksize=10000)
        
        logger.info(f"Successfully saved annotated matrix to {output_path}")
        
        # Log statistics
        normalized_count = (df["Normalized_Gene"] != df["Gene"].astype(str)).sum()
        synonym_count = (df["HGNC_Synonyms"] != "").sum()
        logger.info(f"Annotation complete: {normalized_count} genes normalized, {synonym_count} genes with synonyms")
        
        return output_path
        
    except Exception as exc:
        logger.error(f"Error saving annotated file: {exc}")
        return f"Error saving annotated matrix: {exc}"