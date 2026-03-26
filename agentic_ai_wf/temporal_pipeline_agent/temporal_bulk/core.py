#!/usr/bin/env python3
"""
Temporal Bulk CLI (TDP Step‑46 style) — v3.1 (rigid, end‑to‑end)
=================================================================

This script takes bulk RNA‑seq counts + metadata, derives a temporal axis
(true time or pseudotime), fits per‑gene impulse models, summarizes results,
optionally runs ssGSEA, exports standardized trajectory tables (for causal UIs),
and produces publication‑style figures. It is hardened to:

- Write early artifacts (pseudotime.csv) and frequent checkpoints
- Handle missing/odd column names and reorient matrices safely
- Fall back gracefully if phenopath/rpy2/gseapy are unavailable
- Provide parallel fits with joblib, or serial with progress
- **Always** do a final write with FDR (overwriting checkpoints)
- Accept MSigDB shorthand alias ("H" -> best matching Hallmark library) or local .gmt path
- Headless plotting (Agg), Matplotlib 3.9 deprecation fixes
- Robust **deconvolution** ingestion (auto-detect sample_id column)

Outputs (in --output_dir):
- pseudotime.csv
- temporal_gene_fits.tsv (gene_id, time_of_peak, time_of_valley, p_value, p_adj, pattern, aic, r2_impulse)
- temporal_qc.tsv (autocorr_lag1, r2_impulse, r2_linear, cooks_max)
- ssgsea/ (if --run_ssgsea)
- pathway_temporal_summary.csv (if ssGSEA succeeds)
- pathway_profiles.csv (tidy: entity,t,score,t_norm; if ssGSEA succeeds)
- cellmix_profiles.csv (tidy; if --deconv_csv provided)
- temporal_pack/manifest.json (per‑entity PNGs)
- Figures: pseudotime_by_covariate.png, de_vs_beta.png (+ CSV),
           pseudotime_vs_pathways_top15.png (if ssGSEA),
           temporal_pathways_top12.png / temporal_cellmix_top12.png,
           elbo.png (if phenopath exposes ELBO)
- report.html (links & run summary)

Example:
    python -u temporal_bulk_cli_v3.py \
      -i data/Deconvulution_data/ \
      -o temporal/Deconvolution-1 \
      --counts GSE139061_counts_clean.csv \
      --metadata GSE139061_metadata_clean.csv \
      --covariate condition --treatment_level Treatment --extra_covariates age \
      --run_ssgsea --gene_sets Hallmark \
      --deconv_csv data/deconvulution/bisque_bulk_proportions_clean.csv \
      --make_figures --render_report --n_jobs 6
"""

from __future__ import annotations
import argparse, warnings, json, os, sys, time, re, zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Headless plotting first
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# stats / models
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import OLSInfluence
    from statsmodels.stats.multitest import multipletests
except Exception:
    sm = None; smf = None; OLSInfluence = None; multipletests = None

# ML & utils
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
except Exception:
    PCA = None; StandardScaler = None; Ridge = None

# numerics
try:
    from scipy.optimize import curve_fit
    from scipy.stats import chi2
except Exception:
    curve_fit = None; chi2 = None

# pathway
try:
    import gseapy as gp
except Exception:
    gp = None

# rpy2 / phenopath (optional) - lazy initialization
R_AVAILABLE = None  # None = not checked yet, True/False = checked
_ro = None
_pandas2ri = None
_importr = None

def _lazy_init_rpy2():
    """Lazy initialize rpy2 - only when actually needed"""
    global R_AVAILABLE, _ro, _pandas2ri, _importr
    if R_AVAILABLE is True and _ro is not None:
        return _ro, _pandas2ri, _importr
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        _ro = ro
        _pandas2ri = pandas2ri
        _importr = importr
        R_AVAILABLE = True
        return ro, pandas2ri, importr
    except Exception:
        R_AVAILABLE = False
        return None, None, None

def check_r_available():
    """Check if R is available (lazy check)"""
    if R_AVAILABLE is None:
        _lazy_init_rpy2()
    return R_AVAILABLE is True

# Provide lazy accessors for backward compatibility
def get_ro():
    """Get rpy2.robjects - lazy initialized"""
    ro, _, _ = _lazy_init_rpy2()
    return ro

def get_pandas2ri():
    """Get pandas2ri - lazy initialized"""
    _, pandas2ri, _ = _lazy_init_rpy2()
    return pandas2ri

def get_importr():
    """Get importr - lazy initialized"""
    _, _, importr = _lazy_init_rpy2()
    return importr

# Lazy proxy classes for backward compatibility with `from .core import *`
class _LazyRO:
    """Lazy proxy for ro (rpy2.robjects)"""
    def __getattr__(self, name):
        ro, _, _ = _lazy_init_rpy2()
        if ro is None:
            raise RuntimeError("rpy2 not available")
        return getattr(ro, name)
    
    def __call__(self, *args, **kwargs):
        ro, _, _ = _lazy_init_rpy2()
        if ro is None:
            raise RuntimeError("rpy2 not available")
        return ro(*args, **kwargs)

class _LazyPandas2RI:
    """Lazy proxy for pandas2ri"""
    def __getattr__(self, name):
        _, pandas2ri, _ = _lazy_init_rpy2()
        if pandas2ri is None:
            raise RuntimeError("rpy2 not available")
        return getattr(pandas2ri, name)

class _LazyImportR:
    """Lazy proxy for importr"""
    def __call__(self, *args, **kwargs):
        _, _, importr = _lazy_init_rpy2()
        if importr is None:
            raise RuntimeError("rpy2 not available")
        return importr(*args, **kwargs)

# Export for `from .core import *` compatibility
ro = _LazyRO()
pandas2ri = _LazyPandas2RI()
importr = _LazyImportR()

# R_AVAILABLE check function for backward compatibility
def _get_r_available():
    """Lazy check for R availability"""
    return check_r_available()

# Make R_AVAILABLE work as a property for backward compatibility
class _RAvailableProperty:
    """Property-like access to R_AVAILABLE"""
    def __bool__(self):
        return check_r_available()
    
    def __repr__(self):
        return str(check_r_available())

# Replace R_AVAILABLE with lazy property
R_AVAILABLE = _RAvailableProperty()

# optional parallel
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None; delayed = None

# -------------------------------
# CLI
# -------------------------------

