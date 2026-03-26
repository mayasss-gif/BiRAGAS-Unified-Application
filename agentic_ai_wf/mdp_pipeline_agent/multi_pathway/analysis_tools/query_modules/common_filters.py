#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd

BAD_STRINGS = {"", "NA", "N/A", "NULL", "NONE", "UNKNOWN", "UNK", "?"}

GO_BP_PREFIX = "GO_Biological_Process"
GO_MF_PREFIX = "GO_Molecular_Function"
GO_CC_PREFIX = "GO_Cellular_Component"
KEGG_PREFIX = "KEGG"
REACTOME_PREFIX = "Reactome"
WIKIPW_PREFIX = "WikiPathways"
HALLMARK_PREFIX = "Hallmark"

TYPE_ORDER = [
    "GO_Biological_Process",
    "GO_Molecular_Function",
    "GO_Cellular_Component",
    "Pathways",
    "Hallmarks",
    "Other",
]


def normalize_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_bad_label(x: object) -> bool:
    s = normalize_str(x).upper()
    return (s in BAD_STRINGS) or (s == "")


def pick_best_source_label_col(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer a column that best describes the ontology/library source:
    chooses the one with fewest UNKNOWN/NaN.
    """
    candidates = []
    for c in ["Ontology_Source", "Library", "library", "gene_set", "Gene_set", "Gene Set", "GeneSet"]:
        if c in df.columns:
            candidates.append(c)
    if not candidates:
        return None
    n = len(df)
    if n == 0:
        return candidates[0]
    best = candidates[0]
    best_bad = float("inf")
    for c in candidates:
        bad = df[c].apply(is_bad_label).sum()
        bad_rate = bad / max(1, n)
        if bad_rate < best_bad:
            best_bad = bad_rate
            best = c
    return best


def classify_type(source_label: str) -> str:
    s = normalize_str(source_label)
    if not s:
        return "Other"
    if s.startswith(GO_BP_PREFIX):
        return "GO_Biological_Process"
    if s.startswith(GO_MF_PREFIX):
        return "GO_Molecular_Function"
    if s.startswith(GO_CC_PREFIX):
        return "GO_Cellular_Component"
    if s.startswith(KEGG_PREFIX) or s.startswith(REACTOME_PREFIX) or s.startswith(WIKIPW_PREFIX):
        return "Pathways"
    if s.startswith(HALLMARK_PREFIX):
        return "Hallmarks"
    return "Other"


def safe_neglog10(q: object) -> float:
    try:
        if q is None or pd.isna(q):
            return 0.0
        qv = float(q)
        if qv <= 0:
            return 50.0
        return -math.log10(qv)
    except Exception:
        return 0.0


def pick_significance_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try common significance columns across GSEA/ORA.
    Returns the best available column name in df.
    """
    # Most common
    for c in ["FDR q-val", "FDR", "qval", "q_value", "padj", "adj_p", "FDR_q", "Q-value", "QValue"]:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in ["fdr q-val", "fdr", "qval", "q_value", "padj", "adj_p", "q-value", "qvalue"]:
        if key in lower_map:
            return lower_map[key]
    # p-value fallback
    for c in ["pval", "p-value", "P-value", "NOM p-val"]:
        if c in df.columns:
            return c
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in ["pval", "p-value", "p-value", "nom p-val"]:
        if key in lower_map:
            return lower_map[key]
    return None


def apply_sig_and_cap(df: pd.DataFrame, sig_col: str, sig: float, cap: int) -> pd.DataFrame:
    out = df.copy()
    out[sig_col] = pd.to_numeric(out[sig_col], errors="coerce")
    out = out.dropna(subset=[sig_col])
    out = out[out[sig_col] <= sig].copy()
    if out.empty:
        return out
    out = out.sort_values(sig_col, ascending=True).head(cap)
    return out


def ensure_class_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    out = df.copy()
    main = "Main_Class"
    sub = "Sub_Class"
    if main not in out.columns:
        out[main] = "Unclassified"
    if sub not in out.columns:
        out[sub] = "Unclassified"
    return out, main, sub
