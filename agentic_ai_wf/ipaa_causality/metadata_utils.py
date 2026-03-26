#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metadata_helper.py

Purpose
-------
A tiny, dependency-light helper module used by IPAA downstream stages
(overlap JSONs / reporting) to apply metadata labels.

This module is intentionally defensive:
- Auto-detects sample/group columns if user-provided names don't exist.
- Canonicalizes sample IDs so they match expression/sample names robustly.
- Normalizes group labels to {case, control} with a safe 2-class rule.
- Can enrich overlap JSONs with metadata (counts + sample lists).

Drop-in location for your run style:
- If you run: python IPAA_test/IPAA/main_ipaa_best.py ...
  then put this file in: IPAA_test/IPAA/metadata_helper.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


# -----------------------------
# Canonicalization / detection
# -----------------------------
_SAMPLE_PATTERNS = [
    r"^sample$",
    r"sample[_\s]*id",
    r"^id$",
    r"^sid$",
    r"^run$",
    r"^name$",
]
_GROUP_PATTERNS = [
    r"^group$",
    r"condition",
    r"phenotype",
    r"status",
    r"class",
    r"label",
    r"case[_\s-]*control",
]


def canonical_sample_id(x: object) -> str:
    """
    Make sample IDs match across sources.
    - strips quotes/space
    - strips directory prefix
    - strips common sequencing suffixes
    - lowercases
    """
    s = str(x).strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.split("/")[-1]

    suffixes = [
        ".fastq.gz", ".fq.gz", ".fastq", ".fq",
        ".bam", ".sam", ".cram",
    ]
    low = s.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if low.endswith(suf):
                s = s[: -len(suf)]
                low = s.lower()
                changed = True
                break
    return s.strip().lower()


def _pick_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None


def autodetect_meta_columns(df: pd.DataFrame) -> Tuple[str, str]:
    s = _pick_col(df, _SAMPLE_PATTERNS)
    g = _pick_col(df, _GROUP_PATTERNS)
    if not s or not g:
        raise ValueError(
            "Could not auto-detect metadata columns.\n"
            f"Columns: {list(df.columns)}\n"
            "Fix: provide explicit sample_col and group_col."
        )
    return s, g


def normalize_group_label(v: object) -> str:
    """
    Normalize to case/control when possible.
    """
    if pd.isna(v):
        return ""
    s = str(v).strip().strip('"').strip("'").lower()

    # boolean-ish
    if s in {"0", "0.0", "false", "f", "no"}:
        return "control"
    if s in {"1", "1.0", "true", "t", "yes"}:
        return "case"

    if s in {"case", "control"}:
        return s

    control_hits = ["control", "ctrl", "healthy", "normal", "baseline", "untreated", "vehicle", "wt", "wildtype"]
    case_hits = ["case", "patient", "disease", "tumor", "treated", "mutant", "ko", "knockout", "affected"]

    if any(tok in s for tok in control_hits):
        return "control"
    if any(tok in s for tok in case_hits):
        return "case"

    return s


def normalize_groups_two_class(series: pd.Series, group_map: Optional[Dict[str, str]] = None) -> pd.Series:
    """
    Enforces exactly 2 groups and resolves to {case, control}.
    If group_map is provided, it's applied first (case-insensitive).
    """
    s = series.astype(str)

    if group_map:
        gm = {str(k).strip().lower(): str(v).strip().lower() for k, v in group_map.items()}
        s = s.map(lambda x: gm.get(str(x).strip().lower(), x))

    norm = s.map(normalize_group_label).astype(str)
    norm = norm.replace({"": pd.NA}).dropna()

    uniq = sorted(set(norm.unique()))
    if set(uniq).issubset({"case", "control"}):
        return norm.rename("group")

    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]

        def looks_control(x: str) -> bool:
            x = x.lower()
            return ("control" in x) or ("healthy" in x) or ("normal" in x) or (x in {"ctrl", "baseline", "wt"})

        def looks_case(x: str) -> bool:
            x = x.lower()
            return ("patient" in x) or ("disease" in x) or ("tumor" in x) or ("treated" in x) or (x in {"ko", "case"})

        if looks_control(a) and not looks_control(b):
            norm = norm.replace({a: "control", b: "case"})
        elif looks_control(b) and not looks_control(a):
            norm = norm.replace({b: "control", a: "case"})
        elif looks_case(a) and not looks_case(b):
            norm = norm.replace({a: "case", b: "control"})
        elif looks_case(b) and not looks_case(a):
            norm = norm.replace({b: "case", a: "control"})
        else:
            raise ValueError(
                "Two-class metadata found but ambiguous.\n"
                f"Labels: {uniq}\n"
                "Fix: provide group_map like {'healthy':'control','sle':'case'}."
            )

        uniq2 = sorted(set(norm.unique()))
        if not set(uniq2).issubset({"case", "control"}):
            raise ValueError(f"Failed to normalize metadata groups. Final labels: {uniq2}")
        return norm.rename("group")

    raise ValueError(
        "Metadata group column must resolve to exactly 2 groups.\n"
        f"Found after normalization: {uniq}\n"
        "Fix: clean metadata or provide group_map."
    )


# -----------------------------
# Main API
# -----------------------------
def load_meta_labels(
    meta_path: Union[str, Path],
    sample_col: Optional[str] = None,
    group_col: Optional[str] = None,
    group_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Returns a standardized metadata table with:
      raw_sample, canonical_sample, group_raw, group
    """
    p = Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    # delimiter heuristic
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep)

    if sample_col and group_col and (sample_col in df.columns) and (group_col in df.columns):
        s_col, g_col = sample_col, group_col
    else:
        s_col, g_col = autodetect_meta_columns(df)

    raw_s = df[s_col]
    raw_g = df[g_col]

    g_norm = normalize_groups_two_class(raw_g, group_map=group_map)

    out = pd.DataFrame(
        {
            "raw_sample": raw_s.astype(str),
            "canonical_sample": raw_s.map(canonical_sample_id).astype(str),
            "group_raw": raw_g.astype(str),
            "group": g_norm.astype(str),
        }
    )

    # drop exact duplicates on canonical_sample (keep first)
    out = out.drop_duplicates(subset=["canonical_sample"], keep="first").reset_index(drop=True)
    return out


def build_sample_to_group_map(meta_df: pd.DataFrame) -> Dict[str, str]:
    """
    Map canonical_sample -> group
    """
    if "canonical_sample" not in meta_df.columns or "group" not in meta_df.columns:
        raise ValueError("meta_df must contain columns: canonical_sample, group")
    return dict(zip(meta_df["canonical_sample"].astype(str), meta_df["group"].astype(str)))


def apply_group_labels_to_index(
    df: pd.DataFrame,
    sample_to_group: Dict[str, str],
    joiner: str = "__",
) -> pd.DataFrame:
    """
    Renames index values as: <original><joiner><group> when meta is available.
    """
    new_index = []
    for s in df.index.astype(str):
        key = canonical_sample_id(s)
        g = sample_to_group.get(key)
        new_index.append(f"{s}{joiner}{g}" if g else s)
    out = df.copy()
    out.index = pd.Index(new_index)
    return out


def summarize_groups(meta_df: pd.DataFrame) -> Dict[str, int]:
    """
    Returns {'n_case':..., 'n_control':...}
    """
    g = meta_df["group"].astype(str)
    return {"n_case": int((g == "case").sum()), "n_control": int((g == "control").sum())}


def enrich_overlap_json_with_meta(
    overlap_json_path: Union[str, Path],
    meta_path: Union[str, Path],
    sample_col: Optional[str] = None,
    group_col: Optional[str] = None,
    group_map: Optional[Dict[str, str]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Adds a top-level key 'meta_labels' to the overlap JSON:
      {
        "meta_labels": {
          "n_case": ...,
          "n_control": ...,
          "case_samples": [...canonical...],
          "control_samples": [...canonical...]
        }
      }

    Does NOT modify existing pathway/entity payload.
    """
    oj = Path(overlap_json_path)
    if not oj.exists():
        raise FileNotFoundError(str(oj))

    meta_df = load_meta_labels(meta_path, sample_col=sample_col, group_col=group_col, group_map=group_map)
    counts = summarize_groups(meta_df)

    case_samples = meta_df.loc[meta_df["group"] == "case", "canonical_sample"].astype(str).tolist()
    ctrl_samples = meta_df.loc[meta_df["group"] == "control", "canonical_sample"].astype(str).tolist()

    payload = json.loads(oj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Overlap JSON must be a JSON object at top-level.")

    payload["meta_labels"] = {
        **counts,
        "case_samples": case_samples,
        "control_samples": ctrl_samples,
    }

    outp = Path(out_path) if out_path else oj
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return outp


# -----------------------------
# Compatibility aliases
# -----------------------------
# Many pipelines do: from metadata_helper import apply_meta_labels
def apply_meta_labels(*args, **kwargs):
    """
    Back-compat alias.
    Most commonly used to enrich overlap JSONs with meta context.
    """
    return enrich_overlap_json_with_meta(*args, **kwargs)
