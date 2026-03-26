#!/usr/bin/env python3
"""
S3 (analysis) — Loader helpers
load_table_auto
"""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_table_auto(path: str, index_col: Optional[int] = None) -> pd.DataFrame:
    """
    Load a delimited table with smart delimiter detection and light robustness.

    Supports
    --------
    - .csv → comma
    - .tsv / .txt → tab
    - Other extensions → sniff between [',', '\\t', ';', '|']
    - Lines starting with '#' are treated as comments
    - UTF-8 by default; falls back to latin-1 if needed
    - Gzip files via pandas ('.gz' suffix)

    Parameters
    ----------
    path : str
        File path to load.
    index_col : int | None
        Optional 0-based index column to set.

    Returns
    -------
    pandas.DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    # Choose delimiter from extension
    ext = p.suffix.lower()
    sep: Optional[str]
    if ext == ".csv":
        sep = ","
    elif ext in {".tsv", ".txt"}:
        sep = "\t"
    else:
        # Sniff delimiter from the first 8KB
        import csv
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(8192)
            try:
                sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                sep = sniffed.delimiter
            except Exception:
                sep = ","  # sane default

    # Try UTF-8, then latin-1 as fallback (keeps weird characters instead of failing)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            index_col=index_col,
            comment="#",
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=sep,
            index_col=index_col,
            comment="#",
            low_memory=False,
            encoding="latin-1",
        )

    return df
