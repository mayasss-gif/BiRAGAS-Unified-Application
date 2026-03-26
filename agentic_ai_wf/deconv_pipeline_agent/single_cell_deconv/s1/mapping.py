#!/usr/bin/env python3
"""
S1 (prep) — Gene ID mapping utilities
best_var_key_for_overlap, harmonize_to_bulk_ids
"""
from __future__ import annotations

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import anndata as ad


def _strip_ens_version_series(s: pd.Series) -> pd.Series:
    """Remove Ensembl version suffixes like 'ENSG000001.12' → 'ENSG000001'."""
    return s.astype(str).str.replace(r"\.\d+$", "", regex=True)


def best_var_key_for_overlap(adata: ad.AnnData, bulk_genes: pd.Series) -> Optional[str]:
    """
    Heuristically pick the `adata.var[...]` column whose values overlap the most
    with the provided bulk gene identifiers.

    Parameters
    ----------
    adata : AnnData
        Single-cell AnnData object.
    bulk_genes : pd.Series
        Vector of bulk gene identifiers (already cleaned/uppercased if desired).

    Returns
    -------
    str | None
        The chosen column name from `adata.var` or None if no candidate columns exist.
    """
    candidate_keys = [
        "feature_id", "feature_name",
        "gene_symbol", "gene_symbols", "gene", "gene_name", "symbol", "hgnc_symbol",
        "ensembl", "ensembl_id", "gene_id",
    ]
    keys = [k for k in candidate_keys if k in adata.var.columns]
    if not keys:
        return None

    bulk_set = set(bulk_genes.astype(str))
    best_key, best_overlap = None, -1

    for k in keys:
        vals = adata.var[k].astype(str)
        if any(t in k.lower() for t in ["ensembl", "feature_id", "gene_id"]):
            vals = _strip_ens_version_series(vals).str.upper()
        else:
            vals = vals.str.upper()
        ov = len(bulk_set & set(vals))
        if ov > best_overlap:
            best_overlap, best_key = ov, k

    return best_key


def harmonize_to_bulk_ids(adata: ad.AnnData, bulk_genes: pd.Series) -> Tuple[ad.AnnData, Dict]:
    """
    Align single-cell gene identifiers to bulk gene IDs by choosing the best
    available identifier space, then subset to the intersecting genes.

    Strategy
    --------
    1) Check current `var_names` overlap with bulk genes.
    2) If overlap is already good (≥60% of bulk) or no better key exists, keep var_names.
       Otherwise, switch to the best‐overlap column (e.g., Ensembl) and normalize.
    3) Subset to intersection and record a summary.

    Returns
    -------
    (AnnData, dict) :
        - New AnnData with `var_names` set to chosen ID space and subset to overlap.
        - Summary dict: {"var_key_used", "n_bulk_genes", "n_overlap"}.
    """
    # Normalize current var_names for overlap check
    vn = pd.Series(adata.var_names.astype(str)).pipe(_strip_ens_version_series).str.upper()
    bulk_set = set(bulk_genes.astype(str))

    overlap_current = len(bulk_set & set(vn))
    key = best_var_key_for_overlap(adata, bulk_genes)

    if overlap_current >= (0.6 * len(bulk_set)) or key is None:
        used_key = "var_names"
        new_names = vn
    else:
        used_key = key
        series = adata.var[key].astype(str)
        if any(t in key.lower() for t in ["ensembl", "feature_id", "gene_id"]):
            series = _strip_ens_version_series(series).str.upper()
        else:
            series = series.str.upper()
        new_names = series

    out = adata.copy()
    out.var["original_var_name"] = out.var_names
    out.var_names = new_names.values
    out.var_names_make_unique()

    inter = sorted(list(set(map(str, out.var_names)) & bulk_set))

    # Subset to the overlap
    out = out[:, inter].copy()
    out.var["in_bulk"] = True

    # Summary and light logging hints (caller can print if desired)
    summary = {
        "var_key_used": used_key,
        "n_bulk_genes": int(len(bulk_genes)),
        "n_overlap": int(len(inter)),
    }

    if summary["n_overlap"] < max(100, int(0.05 * len(bulk_genes))):
        print("[WARN] Small overlap between single-cell and bulk genes:",
              summary["n_overlap"], "of", summary["n_bulk_genes"])

    return out, summary
