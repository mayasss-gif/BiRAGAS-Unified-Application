#!/usr/bin/env python3
"""
S1 (prep) — Marker computation
compute_markers
"""

from __future__ import annotations

from typing import Optional
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def compute_markers(
    adata: ad.AnnData,
    cell_type_key: str,
    top_n: int = 100,
    min_cells_per_group: int = 10,
) -> pd.DataFrame:
    """
    Compute per–cell-type marker genes using Scanpy's rank_genes_groups (Wilcoxon).

    Parameters
    ----------
    adata : AnnData
        QC-filtered, analysis-ready AnnData.
    cell_type_key : str
        Column in `adata.obs` containing cell-type labels.
    top_n : int
        Number of top markers per group to keep.
    min_cells_per_group : int
        Skip groups with fewer than this many cells.

    Returns
    -------
    DataFrame with columns: ["gene", "cluster", "weight"]
      - gene: gene name
      - cluster: cell-type label
      - weight: ranking score (Scanpy's 'scores')
    """
    if cell_type_key not in adata.obs.columns:
        return pd.DataFrame(columns=["gene", "cluster", "weight"])

    # Keep only sufficiently sized groups
    counts = adata.obs[cell_type_key].value_counts()
    valid_groups = counts[counts >= min_cells_per_group].index.tolist()
    if len(valid_groups) < 2:
        return pd.DataFrame(columns=["gene", "cluster", "weight"])

    tmp = adata[adata.obs[cell_type_key].isin(valid_groups)].copy()

    # Work on processed (non-raw) data
    tmp.raw = None
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)

    sc.tl.rank_genes_groups(
        tmp,
        groupby=cell_type_key,
        method="wilcoxon",
        n_genes=top_n,
        use_raw=False,
        layer=None,
    )

    rgg = tmp.uns["rank_genes_groups"]
    names, scores = rgg["names"], rgg["scores"]

    rows = []
    # Handle both structured-array and dict-like outputs from Scanpy
    if hasattr(names, "dtype") and getattr(names.dtype, "names", None):
        # Structured array: iterate over group fields
        for grp in names.dtype.names:
            if grp not in valid_groups:
                continue
            gnames = np.array(names[grp]).astype(str)[:top_n]
            gweights = np.array(scores[grp])[:top_n]
            rows.extend((n, grp, float(w)) for n, w in zip(gnames, gweights))
    else:
        # Dict-like mapping
        for grp in scores.dtype.names:
            if grp not in valid_groups:
                continue
            gnames = np.array(names[grp]).astype(str)[:top_n]
            gweights = np.array(scores[grp])[:top_n]
            rows.extend((n, grp, float(w)) for n, w in zip(gnames, gweights))

    df = pd.DataFrame(rows, columns=["gene", "cluster", "weight"]).drop_duplicates()
    return df
