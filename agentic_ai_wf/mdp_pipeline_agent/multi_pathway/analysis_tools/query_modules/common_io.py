#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(p: Path) -> Path:
    """
    Ensure directory exists (mkdir -p).
    Returns the resolved Path (as Path object).
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_write_csv(df: pd.DataFrame, path: Path, index: bool = False, **to_csv_kwargs: Any) -> str:
    """
    Write CSV safely (atomic replace).
    - Creates parent dirs
    - Writes to <file>.tmp then os.replace() to avoid partial files
    """
    path = Path(path)
    ensure_dir(path.parent)

    tmp = path.with_suffix(path.suffix + ".tmp")

    # Sensible defaults; allow callers to override via kwargs
    if "encoding" not in to_csv_kwargs:
        to_csv_kwargs["encoding"] = "utf-8"
    if "lineterminator" not in to_csv_kwargs and "line_terminator" not in to_csv_kwargs:
        # pandas uses 'lineterminator' (no underscore)
        to_csv_kwargs["lineterminator"] = "\n"

    df.to_csv(tmp, index=index, **to_csv_kwargs)
    os.replace(tmp, path)
    return str(path)


def safe_read_csv(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Read CSV with clear errors.
    Accepts pandas.read_csv kwargs for flexibility (drop-in safe).
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    if "encoding" not in read_csv_kwargs:
        # utf-8 works for most; caller can override
        read_csv_kwargs["encoding"] = "utf-8"

    return pd.read_csv(path, **read_csv_kwargs)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> str:
    """
    Atomic text write (write tmp then replace).
    """
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)
    return str(path)
