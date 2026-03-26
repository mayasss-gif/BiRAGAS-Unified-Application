#!/usr/bin/env python3
"""
S1 (prep) — Tissue filtering helpers
guess_tissue_keys, _norm_str, _make_allowed_set, make_tissue_mask, filter_adatas_by_tissue
"""
from __future__ import annotations

from __future__ import annotations
from typing import List, Optional
import numpy as np
import anndata as ad

def guess_tissue_keys(adata: ad.AnnData) -> List[str]:
    """
    Try to discover tissue/organ columns in `adata.obs`.
    Priority list first; then any column containing 'tissue' or 'organ' (excluding 'organism').
    """
    pri = [
        "tissue", "organ", "tissue_general", "tissue_label",
        "tissue_ontology_term", "tissue_ontology_term_id",
        "anatomical_structure", "anatomical_region",
    ]
    keys = [k for k in pri if k in adata.obs.columns]
    if not keys:
        keys = [
            c for c in adata.obs.columns
            if ("tissue" in c.lower()) or ("organ" in c.lower() and "organism" not in c.lower())
        ]
    return keys

def _norm_str(x: str, case_sensitive: bool) -> str:
    """Normalize a string for comparison (optional case-insensitive)."""
    s = "" if x is None else str(x)
    return s if case_sensitive else s.lower()

def _make_allowed_set(allowed: List[str], case_sensitive: bool) -> set:
    """Pre-normalize allowed tissue names to a set for fast membership tests."""
    return set(_norm_str(a, case_sensitive) for a in allowed)

def make_tissue_mask(
    adata: ad.AnnData,
    allowed: List[str],
    keys: Optional[List[str]] = None,
    mode: str = "exact",
    case_sensitive: bool = False,
) -> np.ndarray:
    """
    Build a boolean mask over cells selecting those whose tissue annotations
    match `allowed`.

    Parameters
    ----------
    allowed : list[str]
        Tissue names/keywords to keep. Empty → keep all cells.
    keys : list[str] | None
        Columns in `obs` to search. If None/empty, auto-guess via `guess_tissue_keys`.
    mode : {'exact','substring'}
        Exact matching or substring containment.
    case_sensitive : bool
        Whether to treat comparisons as case-sensitive.

    Returns
    -------
    np.ndarray (bool) : mask of length `adata.n_obs`
    """
    if not allowed:
        return np.ones(adata.n_obs, dtype=bool)

    if keys is None or len(keys) == 0:
        keys = guess_tissue_keys(adata)
    if len(keys) == 0:
        print("[WARN] No tissue-like columns found; tissue filtering skipped for this file.")
        return np.ones(adata.n_obs, dtype=bool)

    allowed_set = _make_allowed_set(allowed, case_sensitive)
    mask = np.zeros(adata.n_obs, dtype=bool)

    for k in keys:
        if k not in adata.obs.columns:
            continue
        col = adata.obs[k].astype(str).fillna("")
        if not case_sensitive:
            col = col.str.lower()

        if mode == "exact":
            mask |= col.isin(allowed_set).values
        elif mode == "substring":
            part = np.zeros(len(col), dtype=bool)
            for a in allowed_set:
                part |= col.str.contains(a, na=False)
            mask |= part
        else:
            raise ValueError("tissue_match_mode must be 'exact' or 'substring'")

    return mask

def filter_adatas_by_tissue(
    adatas: List[ad.AnnData],
    include: List[str],
    keys: Optional[List[str]],
    mode: str,
    case_sens: bool,
) -> List[ad.AnnData]:
    """
    Apply a tissue filter to a list of AnnData objects, returning only the kept cells.
    If no tissues are provided, returns the inputs unchanged.
    """
    if not include:
        return adatas

    kept: List[ad.AnnData] = []
    total_before, total_after = 0, 0

    for i, a in enumerate(adatas):
        total_before += int(a.n_obs)

        keys_i = keys if keys else guess_tissue_keys(a)
        if len(keys_i) == 0:
            print(f"[WARN] File {i}: no tissue keys found; keeping all cells (no filter).")
            kept.append(a)
            total_after += int(a.n_obs)
            continue

        m = make_tissue_mask(
            a, allowed=include, keys=keys_i, mode=mode, case_sensitive=case_sens
        )
        n_keep = int(m.sum())
        n_drop = int(a.n_obs) - n_keep

        if n_keep == 0:
            print(f"[INFO] File {i}: no cells match tissues {include} in keys {keys_i}; skipping this file.")
            continue

        print(f"[INFO] File {i}: kept {n_keep:,} cells, dropped {n_drop:,} (tissue keys={keys_i}, allowed={include})")
        kept.append(a[m].copy())
        total_after += n_keep

    print(f"[INFO] Tissue filter summary: kept {total_after:,} / {total_before:,} cells across files.")
    if len(kept) == 0:
        raise ValueError("Tissue filter removed all cells. Check tissue_include or column names.")

    return kept
