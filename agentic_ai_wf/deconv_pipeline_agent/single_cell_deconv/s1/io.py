# src/bisque_deconv/s1/io.py
from __future__ import annotations
import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np

def _read_bulk_any(path: str, gene_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Read a bulk expression table (CSV/TSV). Return (dataframe, detected_gene_col).
    The dataframe includes all columns from disk; detection is robust to common gene col names.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bulk file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    candidates = [
        "ensembl_gene_id","ensembl","ensembl_id","gene_id",
        "gene","genes","gene_symbol","gene_symbols","symbol","gene_name","hgnc_symbol",
        "ID","Name","Gene","GeneID","Gene_Symbol","Gene.Name"
    ]

    if gene_col and gene_col in df.columns:
        gcol = gene_col
    else:
        gcol = next((c for c in candidates if c in df.columns), None)
        if gcol is None:
            # Fallback: first non-numeric column; else first column.
            nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            gcol = nonnum[0] if nonnum else df.columns[0]

    return df, gcol

def _strip_ens_version_series(s: pd.Series) -> pd.Series:
    """Drop Ensembl version suffix (.10 etc.) and normalize to upper-case strings."""
    return s.astype(str).str.replace(r"\.\d+$", "", regex=True).str.upper().str.strip()

def read_bulk_genes(path: str, gene_col: Optional[str] = None) -> pd.Series:
    """
    Return a clean, de-duplicated Series of gene identifiers from a bulk file.
    - Strips Ensembl version suffixes.
    - Upper-cases symbols/IDs.
    - Drops blanks/NaN and duplicates.
    """
    df, gcol = _read_bulk_any(path, gene_col=gene_col)
    genes = _strip_ens_version_series(df[gcol])
    genes = genes[(genes != "") & genes.notna()].drop_duplicates()
    return genes

def read_bulk_counts(path: str, gene_col: Optional[str] = None) -> pd.DataFrame:
    """
    Return a numeric counts matrix indexed by cleaned gene IDs.
    - Non-gene columns must be numeric; non-numeric will be coerced to NaN then dropped.
    - Duplicate genes are de-duplicated by first occurrence.
    """
    df, gcol = _read_bulk_any(path, gene_col=gene_col)

    # Ensure numeric samples (coerce non-numeric to NaN then drop fully-non-numeric cols)
    for c in df.columns:
        if c != gcol:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num = df.drop(columns=[gcol], errors="ignore").select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("No numeric sample columns found in bulk file.")

    idx = _strip_ens_version_series(df[gcol])
    counts = num.copy()
    counts.index = idx.values
    counts = counts[~counts.index.duplicated(keep="first")]
    return counts
