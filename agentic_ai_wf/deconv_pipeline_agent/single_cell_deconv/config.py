#!/usr/bin/env python3
"""
bisque_deconv.config

Contains:
- USER_DEFAULTS: baseline defaults for CLI & drivers
- resolve_paths(args): expand user paths, normalize relative ones
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


# -------------------------------------------------------------
# Default user-configurable parameters (used by CLI & drivers)
# -------------------------------------------------------------
# Central defaults used by the CLI. Make sure EVERY key the CLI reads exists here.
USER_DEFAULTS = {
    # Inputs
    "ref_h5ad":  r"D:\AyassBio_Workspace_Downloads\ORGAN_WISE\spleen.h5ad",
    "root":      "",  # used only if ref_h5ad is empty
    "bulk":      r"C:\Users\shahr\Downloads\Bisque-Deconv-py-pkg\Lupus.csv",
    "bulk_gene_col": "gene",      # "None" or "" means auto-detect
    "metadata":  r"C:\Users\shahr\Downloads\Bisque-Deconv-py-pkg\Lupus_meta.csv",
    "species":   "human",         # human | mouse

    # Outputs
    "outdir":     r"D:\AyassBio_Workspace_Downloads\ORGAN_WISE\spleen-test\Bisque-Preprocessing",
    "s2_outdir":  r"D:\AyassBio_Workspace_Downloads\ORGAN_WISE\spleen-test\Bisque_deconvolution_results",
    "enh_outdir": r"D:\AyassBio_Workspace_Downloads\ORGAN_WISE\spleen-test\Enhanced-deconvolution-Reports",

    # Tissue filter
    "tissue_include": [],
    "tissue_keys": ["tissue"],
    "tissue_match_mode": "substring",  # exact | substring

    # S2 tuning (strings on purpose; CLI passes through)
    "drop_mito": "true",
    "allow_duplicate_genes": "true",
    "warn_min_overlap": "200",
    "max_prop_deviation": "0.05",
    "top_markers_per_ct": "25",
    "top_genes_per_sample": "40",
    "auto_install": "true",

    # S3 labels
    "sample_col": "sample_id",
    "condition_col": "condition",
    "control_label": "Control",
    "patient_label": "Disease",
}

# -------------------------------------------------------------
# Helper: normalize and expand paths
# -------------------------------------------------------------
def _expand_path(p: Optional[str]) -> Optional[str]:
    """Expand ~ and make absolute, or None if blank."""
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return None
    path = Path(p).expanduser()
    return str(path if path.is_absolute() else path.resolve())


def resolve_paths(args: Any) -> Dict[str, Optional[str]]:
    """
    Normalize all input/output paths in a CLI namespace or dict.

    Works with argparse.Namespace or plain dict.
    Returns a dict of resolved absolute paths.

    Examples
    --------
    >>> from argparse import Namespace
    >>> ns = Namespace(root='.', bulk='data.csv', outdir='out')
    >>> resolve_paths(ns)
    {'root': 'C:/project', 'bulk': 'C:/project/data.csv', 'outdir': 'C:/project/out'}
    """
    if hasattr(args, "__dict__"):
        items = vars(args)
    elif isinstance(args, dict):
        items = args
    else:
        raise TypeError("resolve_paths() expects dict or argparse.Namespace")

    keys = [
        "root", "ref_h5ad", "bulk", "metadata",
        "outdir", "s2_outdir", "enh_outdir",
    ]
    resolved = {}
    for k in keys:
        v = items.get(k)
        resolved[k] = _expand_path(v)
    return resolved


# -------------------------------------------------------------
# Optional: run as script to print defaults
# -------------------------------------------------------------
if __name__ == "__main__":
    import json
    print(json.dumps(USER_DEFAULTS, indent=2))
