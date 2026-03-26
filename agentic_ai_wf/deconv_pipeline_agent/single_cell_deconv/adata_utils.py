# src/bisque_deconv/adata_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

def find_h5ad_files(root_or_file: str) -> List[str]:
    """
    If given a .h5ad file, return [that file].
    If given a directory, return all *.h5ad files under it (non-recursive).
    """
    p = Path(root_or_file)
    if p.is_file() and p.suffix.lower() == ".h5ad":
        return [str(p.resolve())]
    if p.is_dir():
        return [str(q.resolve()) for q in sorted(p.glob("*.h5ad"))]
    raise FileNotFoundError(f"Not a file/dir or missing: {root_or_file}")

def read_h5ad(path: str):
    """Read a single .h5ad and return an AnnData."""
    try:
        import anndata as ad
    except ImportError:
        raise RuntimeError("anndata is required. Install with: pip install anndata")
    return ad.read_h5ad(path)

def concat_adatas(adatas: Iterable):
    """Concatenate multiple AnnData objects along observations (outer join on genes)."""
    try:
        import anndata as ad
    except ImportError:
        raise RuntimeError("anndata is required. Install with: pip install anndata")

    adatas = list(adatas)
    if not adatas:
        raise ValueError("concat_adatas() received no AnnData objects.")
    if len(adatas) == 1:
        return adatas[0]
    return ad.concat(adatas, axis=0, join="outer", label=None, merge="unique")
