"""
Autonomous H5AD File Size Reduction Agent

This module provides intelligent preprocessing of h5ad files to reduce file size
while preserving essential data for Bisque deconvolution analysis.

The agent dynamically adjusts filtering thresholds based on file size and applies:
- Removal of unnecessary metadata (PCA, UMAP, neighbor graphs)
- Gene filtering (min_cells threshold)
- Cell sampling per cell type (max cells per type)
- Gene symbol filtering
- Sparse matrix conversion
- Compression optimization
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Default thresholds (will be adjusted based on file size)
DEFAULT_MIN_CELLS_PER_GENE = 5
DEFAULT_MIN_CELLS_PER_GENE_STRICT = 20
DEFAULT_MAX_CELLS_PER_CELL_TYPE = 5000
DEFAULT_TARGET_SIZE_MB = 200  # Target size after reduction (MB)

# File size thresholds (GB) for dynamic parameter adjustment
SIZE_THRESHOLD_SMALL = 0.5   # < 0.5 GB: minimal filtering
SIZE_THRESHOLD_MEDIUM = 2.0  # 0.5-2 GB: moderate filtering  
SIZE_THRESHOLD_LARGE = 4.0   # 2-4 GB: aggressive filtering
SIZE_THRESHOLD_VERY_LARGE = 4.0  # > 4 GB: very aggressive filtering


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def get_file_size_gb(file_path: str) -> float:
    """Get file size in gigabytes."""
    return get_file_size_mb(file_path) / 1024


def determine_filtering_strategy(file_size_gb: float) -> Dict[str, Any]:
    """
    Determine filtering strategy based on file size.
    
    Returns a dictionary with filtering parameters:
    - min_cells_per_gene: Minimum cells a gene must be expressed in
    - max_cells_per_cell_type: Maximum cells to sample per cell type
    - filter_genes_twice: Whether to apply two-stage gene filtering
    - remove_metadata: Whether to remove analysis metadata
    """
    if file_size_gb < SIZE_THRESHOLD_SMALL:
        # Small files: minimal filtering
        return {
            "min_cells_per_gene": DEFAULT_MIN_CELLS_PER_GENE,
            "max_cells_per_cell_type": None,  # No cell sampling
            "filter_genes_twice": False,
            "remove_metadata": True,  # Still remove metadata to save space
            "compression": "gzip",
        }
    elif file_size_gb < SIZE_THRESHOLD_MEDIUM:
        # Medium files: moderate filtering
        return {
            "min_cells_per_gene": DEFAULT_MIN_CELLS_PER_GENE,
            "max_cells_per_cell_type": DEFAULT_MAX_CELLS_PER_CELL_TYPE,
            "filter_genes_twice": False,
            "remove_metadata": True,
            "compression": "gzip",
        }
    elif file_size_gb < SIZE_THRESHOLD_LARGE:
        # Large files: aggressive filtering
        return {
            "min_cells_per_gene": DEFAULT_MIN_CELLS_PER_GENE_STRICT,
            "max_cells_per_cell_type": DEFAULT_MAX_CELLS_PER_CELL_TYPE,
            "filter_genes_twice": True,  # Two-stage filtering
            "remove_metadata": True,
            "compression": "gzip",
        }
    else:
        # Very large files: very aggressive filtering
        return {
            "min_cells_per_gene": DEFAULT_MIN_CELLS_PER_GENE_STRICT,
            "max_cells_per_cell_type": DEFAULT_MAX_CELLS_PER_CELL_TYPE,
            "filter_genes_twice": True,
            "remove_metadata": True,
            "compression": "gzip",
            "aggressive_sampling": True,  # May reduce max_cells_per_cell_type further
        }


def reduce_h5ad_file(
    input_path: str,
    output_path: Optional[str] = None,
    min_cells_per_gene: Optional[int] = None,
    max_cells_per_cell_type: Optional[int] = None,
    remove_metadata: bool = True,
    filter_genes_twice: bool = False,
    compression: str = "gzip",
    random_state: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Reduce h5ad file size through intelligent filtering and preprocessing.
    
    Parameters
    ----------
    input_path : str
        Path to input h5ad file
    output_path : Optional[str]
        Path to output reduced h5ad file. If None, creates a new file with
        '_bisque_reduced' suffix in the same directory.
    min_cells_per_gene : Optional[int]
        Minimum number of cells a gene must be expressed in. If None, uses
        strategy-based default.
    max_cells_per_cell_type : Optional[int]
        Maximum cells to sample per cell type. If None, uses strategy-based default.
    remove_metadata : bool
        Whether to remove analysis metadata (PCA, UMAP, neighbor graphs)
    filter_genes_twice : bool
        Whether to apply two-stage gene filtering (first with min_cells=5, then stricter)
    compression : str
        Compression method for output file ('gzip', 'lzf', or None)
    random_state : int
        Random seed for reproducible sampling
    
    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Path to reduced h5ad file and summary statistics dictionary
    """
    try:
        import scanpy as sc
        import anndata as ad
        from scipy import sparse
    except ImportError as e:
        raise RuntimeError(
            f"Required packages not installed: {e}. "
            "Install with: pip install scanpy anndata scipy"
        )
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(
            input_path_obj.parent / f"{input_path_obj.stem}_bisque_reduced.h5ad"
        )
    
    # Get file size and determine strategy
    file_size_gb = get_file_size_gb(input_path)
    file_size_mb = get_file_size_mb(input_path)
    
    logger.info(f"[H5AD Reducer] Processing file: {input_path}")
    logger.info(f"[H5AD Reducer] Original size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    
    # Determine strategy if parameters not explicitly provided
    if min_cells_per_gene is None or max_cells_per_cell_type is None:
        strategy = determine_filtering_strategy(file_size_gb)
        min_cells_per_gene = min_cells_per_gene or strategy["min_cells_per_gene"]
        max_cells_per_cell_type = max_cells_per_cell_type or strategy.get("max_cells_per_cell_type")
        remove_metadata = strategy.get("remove_metadata", remove_metadata)
        filter_genes_twice = strategy.get("filter_genes_twice", filter_genes_twice)
        compression = strategy.get("compression", compression)
    
    logger.info(f"[H5AD Reducer] Strategy: min_cells={min_cells_per_gene}, "
                f"max_cells_per_ct={max_cells_per_cell_type}, "
                f"remove_metadata={remove_metadata}, "
                f"filter_twice={filter_genes_twice}")
    
    # Load the h5ad file
    logger.info("[H5AD Reducer] Loading h5ad file...")
    adata = ad.read_h5ad(input_path)
    
    initial_stats = {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "original_size_mb": file_size_mb,
        "original_size_gb": file_size_gb,
    }
    
    logger.info(f"[H5AD Reducer] Initial: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Step 1: Remove analysis metadata (PCA, UMAP, neighbor graphs, etc.)
    if remove_metadata:
        logger.info("[H5AD Reducer] Removing analysis metadata...")
        adata.obsm.clear()  # PCA, UMAP embeddings, etc.
        adata.obsp.clear()  # Neighbor graphs
        adata.uns.clear()   # Analysis metadata
        adata.layers.clear()  # Additional layers (e.g., normalized counts)
        logger.info("[H5AD Reducer] ✓ Metadata removed")
    
    # Step 2: Ensure we're using counts (if available in layers)
    # If not, keep X as-is
    if "counts" in adata.layers:
        logger.info("[H5AD Reducer] Using counts layer for X...")
        adata.X = adata.layers["counts"]
        adata.layers.clear()  # Clear layers after extracting counts
    
    # Step 3: Convert to sparse matrix if not already
    if not sparse.issparse(adata.X):
        logger.info("[H5AD Reducer] Converting to sparse matrix...")
        adata.X = sparse.csr_matrix(adata.X)
    
    # Step 4: Filter genes (two-stage if requested)
    if filter_genes_twice:
        logger.info(f"[H5AD Reducer] Stage 1: Filtering genes (min_cells={DEFAULT_MIN_CELLS_PER_GENE})...")
        sc.pp.filter_genes(adata, min_cells=DEFAULT_MIN_CELLS_PER_GENE)
        logger.info(f"[H5AD Reducer] After stage 1: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    logger.info(f"[H5AD Reducer] Filtering genes (min_cells={min_cells_per_gene})...")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    logger.info(f"[H5AD Reducer] After gene filtering: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    # Step 5: Sample cells per cell type (if cell_type column exists)
    if max_cells_per_cell_type is not None:
        cell_type_col = None
        # Try to find cell type column
        for col in ["cell_type", "celltype", "cluster", "annotation", "celltypes"]:
            if col in adata.obs.columns:
                cell_type_col = col
                break
        
        if cell_type_col:
            logger.info(f"[H5AD Reducer] Sampling max {max_cells_per_cell_type} cells per {cell_type_col}...")
            
            # Group by cell type and sample
            try:
                cells_to_keep = (
                    adata.obs
                    .groupby(cell_type_col, observed=True, group_keys=False)
                    .apply(
                        lambda x: x.sample(min(len(x), max_cells_per_cell_type), random_state=random_state),
                        include_groups=False
                    )
                    .index
                )
                adata = adata[cells_to_keep].copy()
                logger.info(f"[H5AD Reducer] After cell sampling: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
            except Exception as e:
                logger.warning(f"[H5AD Reducer] Cell sampling failed: {e}. Continuing without sampling.")
        else:
            logger.info("[H5AD Reducer] No cell_type column found. Skipping cell sampling.")
    
    # Step 6: Filter genes without symbol (if symbol column exists)
    if "symbol" in adata.var.columns:
        logger.info("[H5AD Reducer] Filtering genes without symbol...")
        n_before = adata.n_vars
        adata = adata[:, adata.var["symbol"].notna()].copy()
        n_after = adata.n_vars
        logger.info(f"[H5AD Reducer] Removed {n_before - n_after} genes without symbol. "
                   f"Remaining: {n_after:,} genes")
    
    # Step 7: Make var_names unique
    logger.info("[H5AD Reducer] Making var_names unique...")
    adata.var_names_make_unique()
    
    # Step 8: Optimize categorical columns
    if "cell_type" in adata.obs.columns:
        logger.info("[H5AD Reducer] Converting cell_type to category...")
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    
    # Step 9: Save with compression
    logger.info(f"[H5AD Reducer] Saving reduced file to: {output_path}")
    adata.write(output_path, compression=compression)
    
    # Calculate final size
    final_size_mb = get_file_size_mb(output_path)
    final_size_gb = get_file_size_gb(output_path)
    reduction_ratio = (1 - final_size_mb / file_size_mb) * 100
    
    final_stats = {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "final_size_mb": final_size_mb,
        "final_size_gb": final_size_gb,
        "reduction_ratio": reduction_ratio,
    }
    
    summary = {
        **initial_stats,
        **final_stats,
        "output_path": output_path,
        "strategy": {
            "min_cells_per_gene": min_cells_per_gene,
            "max_cells_per_cell_type": max_cells_per_cell_type,
            "remove_metadata": remove_metadata,
            "filter_genes_twice": filter_genes_twice,
            "compression": compression,
        }
    }
    
    logger.info(f"[H5AD Reducer] ✓ Reduction complete!")
    logger.info(f"[H5AD Reducer] Final: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    logger.info(f"[H5AD Reducer] Final size: {final_size_mb:.2f} MB ({final_size_gb:.2f} GB)")
    logger.info(f"[H5AD Reducer] Size reduction: {reduction_ratio:.1f}%")
    
    return output_path, summary


def auto_reduce_h5ad(
    input_path: str,
    output_path: Optional[str] = None,
    target_size_mb: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Automatically reduce h5ad file size with dynamic strategy selection.
    
    This is the main entry point for the autonomous preprocessing agent.
    It automatically determines the best filtering strategy based on file size
    and applies it to reduce the file while preserving essential data.
    
    Parameters
    ----------
    input_path : str
        Path to input h5ad file
    output_path : Optional[str]
        Path to output reduced h5ad file. If None, creates a new file with
        '_bisque_reduced' suffix.
    target_size_mb : Optional[float]
        Target file size in MB. If provided, the agent will iteratively
        apply more aggressive filtering until target is reached.
    
    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Path to reduced h5ad file and summary statistics dictionary
    """
    file_size_gb = get_file_size_gb(input_path)
    strategy = determine_filtering_strategy(file_size_gb)
    
    # Apply initial reduction
    output_path, summary = reduce_h5ad_file(
        input_path=input_path,
        output_path=output_path,
        min_cells_per_gene=strategy["min_cells_per_gene"],
        max_cells_per_cell_type=strategy.get("max_cells_per_cell_type"),
        remove_metadata=strategy.get("remove_metadata", True),
        filter_genes_twice=strategy.get("filter_genes_twice", False),
        compression=strategy.get("compression", "gzip"),
    )
    
    # If target size specified and not reached, apply more aggressive filtering
    if target_size_mb is not None:
        current_size_mb = summary["final_size_mb"]
        iteration = 1
        max_iterations = 3
        
        while current_size_mb > target_size_mb and iteration <= max_iterations:
            logger.info(f"[H5AD Reducer] Iteration {iteration}: "
                       f"Current size {current_size_mb:.2f} MB > target {target_size_mb:.2f} MB. "
                       "Applying more aggressive filtering...")
            
            # Increase min_cells_per_gene
            new_min_cells = strategy["min_cells_per_gene"] + (5 * iteration)
            
            # Reduce max_cells_per_cell_type if it exists
            new_max_cells = None
            if strategy.get("max_cells_per_cell_type"):
                new_max_cells = max(1000, strategy["max_cells_per_cell_type"] - (1000 * iteration))
            
            output_path, summary = reduce_h5ad_file(
                input_path=output_path,  # Use previous output as input
                output_path=output_path,  # Overwrite
                min_cells_per_gene=new_min_cells,
                max_cells_per_cell_type=new_max_cells,
                remove_metadata=True,
                filter_genes_twice=True,
                compression="gzip",
            )
            
            current_size_mb = summary["final_size_mb"]
            iteration += 1
        
        if current_size_mb > target_size_mb:
            logger.warning(f"[H5AD Reducer] Could not reach target size {target_size_mb} MB. "
                          f"Final size: {current_size_mb:.2f} MB")
    
    return output_path, summary
