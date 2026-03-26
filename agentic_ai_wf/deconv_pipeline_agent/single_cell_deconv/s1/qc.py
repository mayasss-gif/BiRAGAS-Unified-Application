#!/usr/bin/env python3
"""
S1 (prep) — QC utilities
calc_qc, mad_thresholds, run_scrublet, qc_filter
"""

from __future__ import annotations
import warnings
from typing import Dict, Optional, Tuple
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse

# Reuse the dense-conversion helper from s1/io.py
try:
    from .io import to_dense  # type: ignore
except Exception:
    # Fallback (keeps this module usable standalone)
    def to_dense(X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)

# -------------------- Defaults --------------------
DEF_USE_SCRUBLET: bool = False
DEF_MIN_GENES_PER_CELL: int = 200
DEF_MIN_CELLS_PER_GENE: int = 3
DEF_MITO_PCT_MAX: Optional[float] = 15.0

# Optional dependency: scrublet
try:
    import scrublet as scr  # type: ignore
    HAVE_SCRUBLET = True
except Exception:
    HAVE_SCRUBLET = False

warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------- QC metrics --------------------
def calc_qc(adata: ad.AnnData, species: str = "human") -> Dict:
    """
    Compute standard QC metrics and return a compact summary.
    Adds `pct_counts_is_mito` if a gene-symbol column is present.
    """
    gene_symbol_cols = [
        c for c in adata.var.columns
        if c.lower() in ["feature_name", "gene_symbol", "gene_symbols",
                         "gene", "gene_name", "symbol", "hgnc_symbol"]
    ]

    if gene_symbol_cols:
        gs = adata.var[gene_symbol_cols[0]].astype(str)
        mito_prefix = "MT-" if species == "human" else "mt-"
        adata.var["is_mito"] = gs.str.startswith(mito_prefix, na=False) | gs.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["is_mito"], percent_top=None, log1p=False, inplace=True)
    else:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    qc = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "median_total_counts": float(np.median(adata.obs["total_counts"])) if "total_counts" in adata.obs else None,
        "median_n_genes_by_counts": float(np.median(adata.obs["n_genes_by_counts"])) if "n_genes_by_counts" in adata.obs else None,
        "median_pct_counts_is_mito": float(np.median(adata.obs["pct_counts_is_mito"])) if "pct_counts_is_mito" in adata.obs else None,
    }
    return qc


# -------------------- Robust thresholds --------------------
def mad_thresholds(
    x: np.ndarray,
    low: bool = True,
    high: bool = True,
    k_low: float = 3.0,
    k_high: float = 3.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (low, high) robust cutoffs using MAD around the median.
    """
    x = x[~np.isnan(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    lo = med - k_low * 1.4826 * mad if low else None
    hi = med + k_high * 1.4826 * mad if high else None
    return lo, hi


# -------------------- Doublet detection --------------------
def run_scrublet(adata: ad.AnnData, expected_doublet_rate: float = 0.06) -> np.ndarray:
    """
    Run Scrublet and annotate `doublet_score` and `predicted_doublet` in `obs`.
    Returns the boolean predicted_doublet vector.
    """
    if not HAVE_SCRUBLET:
        raise RuntimeError("Scrublet not installed. Install with `pip install scrublet`.")

    X = to_dense(adata.X)
    s = scr.Scrublet(X, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = s.scrub_doublets()
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets
    return predicted_doublets


# -------------------- Main QC filter --------------------
def qc_filter(
    adata: ad.AnnData,
    min_genes: int = DEF_MIN_GENES_PER_CELL,
    min_cells_per_gene: int = DEF_MIN_CELLS_PER_GENE,
    mito_pct_max: Optional[float] = DEF_MITO_PCT_MAX,
    use_scrublet: bool = DEF_USE_SCRUBLET
) -> Tuple[ad.AnnData, Dict]:
    """
    Apply standard QC filters:
      - Remove cells with very low/high n_genes_by_counts (robust high cutoff via MAD)
      - Remove high mitochondrial fraction (> mito_pct_max) if available
      - Optional: remove predicted doublets via Scrublet
      - Filter genes expressed in < min_cells_per_gene cells

    Returns filtered AnnData and a summary dict with removal counts.
    """
    summary = {
        "initial_cells": int(adata.n_obs),
        "initial_genes": int(adata.n_vars),
        "remove_low_genes": 0,
        "remove_high_genes": 0,
        "remove_high_mito": 0,
        "remove_doublets": 0,
        "remove_rare_genes": 0,
    }

    keep = np.ones(adata.n_obs, dtype=bool)

    # Filter by n_genes_by_counts
    if "n_genes_by_counts" in adata.obs:
        low = adata.obs["n_genes_by_counts"] < min_genes
        keep &= ~low
        summary["remove_low_genes"] = int(low.sum())

        # Robust high cutoff (right tail)
        _, hi = mad_thresholds(
            adata.obs["n_genes_by_counts"].values,
            low=False, high=True, k_high=4.0
        )
        if hi is not None and np.isfinite(hi):
            high = adata.obs["n_genes_by_counts"] > hi
            keep &= ~high
            summary["remove_high_genes"] = int(high.sum())

    # Mito fraction cutoff (auto if None and metric available)
    if mito_pct_max is None and "pct_counts_is_mito" in adata.obs:
        p95 = np.percentile(adata.obs["pct_counts_is_mito"], 95)
        mito_pct_max = float(min(max(p95, 5.0), 25.0))
    if mito_pct_max is not None and "pct_counts_is_mito" in adata.obs:
        high_mito = adata.obs["pct_counts_is_mito"] > mito_pct_max
        keep &= ~high_mito
        summary["remove_high_mito"] = int(high_mito.sum())

    # Scrublet doublet removal (on the kept subset so far)
    if use_scrublet:
        try:
            predicted_doublets = run_scrublet(adata[keep].copy())
            dmask = np.zeros(adata.n_obs, dtype=bool)
            dmask[np.where(keep)[0]] = predicted_doublets
            keep &= ~dmask
            summary["remove_doublets"] = int(predicted_doublets.sum())
        except Exception as e:
            print("[WARN] Scrublet skipped:", e)

    # Apply cell mask
    adata = adata[keep].copy()

    # Filter genes by detection across cells
    if min_cells_per_gene > 1:
        before = int(adata.n_vars)
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        summary["remove_rare_genes"] = int(before - adata.n_vars)

    summary["final_cells"] = int(adata.n_obs)
    summary["final_genes"] = int(adata.n_vars)
    return adata, summary
