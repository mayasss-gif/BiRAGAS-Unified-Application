#!/usr/bin/env python3
"""
utils_gene_programs.py

Auto gene-program discovery + robust scoring utilities.

Goal:
- No hard-coded gene lists
- Programs adapt per dataset
- Never crash if genes are missing
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def _present_genes(var_names, genes: List[str], min_present: int) -> List[str]:
    vn = set(var_names)
    present = [g for g in genes if g in vn]
    if len(present) >= min_present:
        return present
    return []


def auto_gene_programs(
    adata,
    max_programs: int = 3,
    genes_per_program: int = 50,
    min_present: int = 10,
) -> Dict[str, List[str]]:
    """
    Discover gene programs in this priority:
      1) rank_genes_groups in adata.uns (best)
      2) PCA loadings in adata.varm['PCs'] (scanpy convention)
      3) HVGs in adata.var['highly_variable'] (fallback)

    Returns:
      dict program_name -> list of genes (present in adata.var_names)
    """
    programs: Dict[str, List[str]] = {}

    # -------------------------
    # 1) DEG programs (best)
    # -------------------------
    if isinstance(getattr(adata, "uns", None), dict) and "rank_genes_groups" in adata.uns:
        rg = adata.uns["rank_genes_groups"]
        names = rg.get("names", None)
        if names is not None and hasattr(names, "dtype") and getattr(names.dtype, "names", None):
            groups = list(names.dtype.names)[:max_programs]
            for g in groups:
                genes = list(names[g][:genes_per_program])
                present = _present_genes(adata.var_names, genes, min_present=min_present)
                if present:
                    programs[f"DEG_{g}"] = present
            if programs:
                return programs

    # -------------------------
    # 2) PCA programs
    # -------------------------
    varm = getattr(adata, "varm", {})
    if isinstance(varm, dict) and "PCs" in varm:
        pcs = varm["PCs"]
        try:
            loadings = pd.DataFrame(pcs, index=adata.var_names)
            n_pc = min(max_programs, loadings.shape[1])
            for i in range(n_pc):
                top = (
                    loadings.iloc[:, i]
                    .abs()
                    .sort_values(ascending=False)
                    .head(genes_per_program)
                    .index
                    .tolist()
                )
                present = _present_genes(adata.var_names, top, min_present=min_present)
                if present:
                    programs[f"PC{i}_PROGRAM"] = present
            if programs:
                return programs
        except Exception:
            pass

    # -------------------------
    # 3) HVG fallback
    # -------------------------
    var = getattr(adata, "var", None)
    if var is not None and "highly_variable" in var.columns:
        try:
            hvgs = adata.var.query("highly_variable").index.tolist()
            hvgs = hvgs[:genes_per_program]
            present = _present_genes(adata.var_names, hvgs, min_present=min_present)
            if present:
                programs["HVG_PROGRAM"] = present
        except Exception:
            pass

    return programs


def score_programs_scanpy(
    adata,
    programs: Dict[str, List[str]],
    prefix: str = "PROG_",
    min_present: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Computes scanpy score_genes for each program, safely.

    Returns:
      (scored_cols, skipped_programs)
    """
    scored_cols: List[str] = []
    skipped: List[str] = []

    try:
        import scanpy as sc
    except Exception as e:
        log(f"[WARN] scanpy not available -> cannot score programs. Error={repr(e)}")
        return [], list(programs.keys())

    for name, genes in programs.items():
        genes = _present_genes(adata.var_names, genes, min_present=min_present)
        if not genes:
            skipped.append(name)
            continue

        col = f"{prefix}{name}"
        try:
            sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=False)
            scored_cols.append(col)
        except Exception:
            skipped.append(name)

    return scored_cols, skipped

