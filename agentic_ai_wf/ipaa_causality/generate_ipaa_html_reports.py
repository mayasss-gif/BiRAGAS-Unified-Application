#!/usr/bin/env python3
"""
Generate elegant, interactive HTML reports from IPAA pathway_summary outputs.

Expected input structure:
  OUT_ROOT/
    engines/pathway_summary/<DISEASE>/pathway_summary.tsv   (primary)
    engines/pathway_summary/<DISEASE>/summary.tsv           (also accepted if present)

Outputs:
  OUT_ROOT/engines/pathway_summary/<DISEASE>/report.html
  OUT_ROOT/engines/pathway_summary/index.html

Key behavior (as requested):
- Uses pathway_summary.tsv / summary.tsv (NOT filtered_summary.tsv)
- Category comes from stats_Main_Class (Classification removed)
- Statistical significance for ranking/plots/“sig ≤ 0.05” uses stats_p (NOT FDR)
  - Falls back to p-value-like columns only if stats_p is missing
- Loads only a minimal, relevant subset of columns (fast + robust)
- Lead genes count is derived from lead genes list if missing
- LLM narrative produces a high-insight “story” per disease + per-pathway notes
  - Structured outputs schema, strict-safe, retries, never crashes the report build
- Plots are responsive and resize to container; bar plot height adapts to Top-N
- UI polish: global search, expand/collapse all, row click jumps to pathway note, TSV download
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import plotly.express as px
import plotly.io as pio

from openai import OpenAI  # type: ignore
import requests  # type: ignore

DEFAULT_CDN = {
    "plotly": "https://cdn.plot.ly/plotly-2.30.0.min.js",
    "tabulator_css": "https://unpkg.com/tabulator-tables@6.2.1/dist/css/tabulator.min.css",
    "tabulator_js": "https://unpkg.com/tabulator-tables@6.2.1/dist/js/tabulator.min.js",
    "fonts": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
}


# ----------------------------
# utils
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _fmt_p(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _fmt_f(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return ""
    return f"{v:.3f}"


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", "", str(c or "").strip().lower())


def _canon_alnum(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c or "").lower())


def _numeric_coverage(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    if len(x) == 0:
        return 0.0
    return float(x.notna().mean())


def _pick_pathway_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    norm = {_norm_col(c): c for c in cols}
    for key in ["pathway", "pathway_id", "term", "term_name", "name", "id"]:
        if key in norm:
            return norm[key]
    return cols[0] if cols else None


def _ensure_pathway_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pc = _pick_pathway_column(df)
    if pc is None:
        df["pathway"] = ""
        return df
    df["pathway"] = df[pc].astype(str).str.strip()
    df = df[df["pathway"] != ""]
    return df


def _neglog10_series(p: pd.Series) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce")
    eps = 1e-300
    x = x.where(x > 0, pd.NA).clip(lower=eps)
    return x.apply(lambda v: -math.log10(v) if pd.notna(v) else float("nan"))


def _find_first_norm(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    nmap = {_norm_col(c): c for c in df.columns}
    for k in keys:
        if k in nmap:
            return nmap[k]
    return None


def _slug(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:maxlen] if s else "item"


def _strip_unsafe_html(fragment: str) -> str:
    """
    We trust our own HTML, but LLM HTML could include scripts.
    Remove <script> blocks and inline event handlers (onload=, onclick=, etc).
    """
    if not isinstance(fragment, str):
        return ""
    x = fragment
    x = re.sub(r"<script\b[^>]*>.*?</script>", "", x, flags=re.IGNORECASE | re.DOTALL)
    x = re.sub(r"\son\w+\s*=\s*(['\"]).*?\1", "", x, flags=re.IGNORECASE | re.DOTALL)
    return x.strip()


def _count_lead_genes(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "na", "none", "null"):
        return None
    parts = re.split(r"[;,|]\s*", s)
    genes = [p.strip() for p in parts if p.strip() and p.strip().lower() not in ("nan", "na", "none", "null")]
    return len(genes) if genes else None


# ----------------------------
# column picking from pathway_summary.tsv
# ----------------------------

def choose_best_fdr_column(df: pd.DataFrame) -> Optional[str]:
    candidates: List[str] = []
    for c in df.columns:
        cn = _canon_alnum(c)
        if any(k in cn for k in ["fdr", "qval", "qvalue", "padj", "adjp"]):
            candidates.append(c)

    def prio(c: str) -> int:
        cl = str(c).lower()
        if cl.startswith("stats_"):
            return 0
        if cl.startswith("gsea_"):
            return 1
        if "evidence_" in cl:
            return 2
        return 3

    for c in sorted(candidates, key=prio):
        if _numeric_coverage(df[c]) >= 0.05:
            return c
    return None


def choose_best_pval_column(df: pd.DataFrame) -> Optional[str]:
    candidates: List[str] = []
    for c in df.columns:
        cn = _canon_alnum(c)
        if ("pval" in cn) or ("pvalue" in cn):
            candidates.append(c)

    def prio(c: str) -> int:
        cl = str(c).lower()
        if cl.startswith("stats_"):
            return 0
        if cl.startswith("gsea_"):
            return 1
        if "evidence_" in cl:
            return 2
        return 3

    for c in sorted(candidates, key=prio):
        if _numeric_coverage(df[c]) >= 0.05:
            return c
    return None


def choose_best_nes_column(df: pd.DataFrame) -> Optional[str]:
    candidates: List[str] = []
    for c in df.columns:
        if "nes" in _canon_alnum(c):
            candidates.append(c)

    def prio(c: str) -> int:
        cl = str(c).lower()
        if cl.startswith("stats_"):
            return 0
        if cl.startswith("gsea_"):
            return 1
        if "evidence_" in cl:
            return 2
        return 3

    for c in sorted(candidates, key=prio):
        if _numeric_coverage(df[c]) >= 0.05:
            return c
    return None


def choose_best_delta_column(df: pd.DataFrame) -> Optional[str]:
    candidates: List[str] = []
    for c in df.columns:
        cn = _canon_alnum(c)
        if ("delta" in cn) or ("effect" in cn):
            candidates.append(c)

    def prio(c: str) -> int:
        cl = str(c).lower()
        cn = _canon_alnum(c)
        if cl.startswith("stats_") and "delta" in cn and ("baseline" in cn or "expect" in cn):
            return 0
        if cl.startswith("stats_") and "delta" in cn:
            return 1
        if cl.startswith("activity_") and "delta" in cn:
            return 2
        return 3

    for c in sorted(candidates, key=prio):
        if _numeric_coverage(df[c]) >= 0.05:
            return c
    return None


def choose_stats_p_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer stats_p as the primary significance column.
    Handles variants like stats_p, stats_P, stats_pvalue, stats_p_val, etc.
    """
    exact: List[str] = []
    loose: List[str] = []

    for c in df.columns:
        cn = _canon_alnum(c)
        if cn in ("statsp", "statspvalue", "statspval"):
            exact.append(c)
        elif cn.startswith("statsp"):
            loose.append(c)

    # best coverage wins
    for group in (exact, loose):
        best = None
        best_cov = 0.0
        for c in group:
            cov = _numeric_coverage(df[c])
            if cov > best_cov:
                best_cov = cov
                best = c
        if best is not None and best_cov >= 0.05:
            return best

    return None


def select_usecols_from_header(tsv_path: Path) -> Optional[List[str]]:
    """
    Keep report-relevant subset to avoid loading huge wide tables.
    IMPORTANT: include stats_p explicitly (it doesn't contain 'pval'/'pvalue').
    """
    try:
        header = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        cols = header.split("\t")
    except Exception:
        return None

    keep_exact = {
        "disease", "selected_tissue", "pathway",
        "stats_Main_Class", "stats_main_class",
        "activity_label_a", "activity_label_b",
        "activity_mean", "activity_sd", "activity_mean_a", "activity_mean_b",
        "activity_delta_b_minus_a",
    }

    def want(c: str) -> bool:
        if c in keep_exact:
            return True

        cl = c.lower()
        cn = _canon_alnum(c)

        # Always include stats_p and variants
        if cn.startswith("statsp"):
            return True

        # Always include anything that looks like NES (stats_NES / gsea_NES / NES)
        if "nes" in cn:
            return True

        # Include any pvalue-ish columns too (fallbacks)
        if "pval" in cn or "pvalue" in cn:
            return True

        # Useful stats for ranking/plots
        if cl.startswith("stats_") and any(k in cn for k in ["fdr", "qval", "padj", "delta", "effect", "tstat", "score", "z"]):
            return True

        # Lead genes list/count variants
        if any(k in cn for k in ["leadingedge", "leadgenes", "edgegenes", "leadingedgen", "leadgenesn", "edgegenesn"]):
            return True

        # Some pipelines store lead genes under gsea/evidence fields
        if cl.startswith("gsea_") and any(k in cn for k in ["leadingedge", "edgegene", "leadgene"]):
            return True
        if "evidence_" in cl and any(k in cn for k in ["lead", "edge", "topgene"]):
            return True

        return False

    kept = [c for c in cols if want(c)]
    for must in ["pathway", "disease", "selected_tissue"]:
        if must in cols and must not in kept:
            kept.append(must)
    return kept


def build_canonical_view(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_pathway_key(df_raw)

    if "disease" not in df.columns:
        df["disease"] = ""
    if "selected_tissue" not in df.columns:
        df["selected_tissue"] = ""

    # Category from stats_Main_Class
    main_class_col = None
    for cand in ["stats_Main_Class", "stats_main_class"]:
        if cand in df.columns:
            main_class_col = cand
            break

    stats_p_col = choose_stats_p_column(df)
    fdr_col = choose_best_fdr_column(df)
    pval_col = choose_best_pval_column(df)
    nes_col = choose_best_nes_column(df)
    delta_col = choose_best_delta_column(df)

    # activity columns
    activity_mean_col = _find_first_norm(df, ["activity_mean"])
    activity_sd_col = _find_first_norm(df, ["activity_sd"])
    activity_label_a_col = _find_first_norm(df, ["activity_label_a"])
    activity_label_b_col = _find_first_norm(df, ["activity_label_b"])
    activity_mean_a_col = _find_first_norm(df, ["activity_mean_a"])
    activity_mean_b_col = _find_first_norm(df, ["activity_mean_b"])
    activity_delta_col = _find_first_norm(df, ["activity_delta_b_minus_a", "activity_deltabminusa"])

    # lead genes (optional)
    lead_genes_col = None
    lead_genes_n_col = None
    for c in df.columns:
        cn = _canon_alnum(c)
        cl = str(c).lower()
        if lead_genes_col is None and (("leadingedge" in cn) or ("leadgenes" in cn) or ("edgegene" in cn) or ("edgegenes" in cn)):
            lead_genes_col = c
        if lead_genes_n_col is None and ("leadgenesn" in cn or "leadingedgen" in cn or "edgegenesn" in cn):
            lead_genes_n_col = c
        if lead_genes_col is None and cl.startswith("gsea_") and ("edge" in cn and "gene" in cn):
            lead_genes_col = c

    out = pd.DataFrame(index=df.index)
    out["disease"] = df["disease"].astype(str)
    out["selected_tissue"] = df["selected_tissue"].astype(str)
    out["pathway"] = df["pathway"].astype(str)

    # Keep these (display only). Significance uses stats_p.
    out["fdr"] = pd.to_numeric(df[fdr_col], errors="coerce") if fdr_col else float("nan")

    # stats_p is the canonical significance column
    out["stats_p"] = pd.to_numeric(df[stats_p_col], errors="coerce") if stats_p_col else float("nan")

    # optional pvalue-like fallback column (can show in table if you want)
    out["pvalue"] = pd.to_numeric(df[pval_col], errors="coerce") if pval_col else float("nan")

    out["nes"] = pd.to_numeric(df[nes_col], errors="coerce") if nes_col else float("nan")

    out["activity_label_a"] = df[activity_label_a_col].astype(str) if activity_label_a_col else "A"
    out["activity_label_b"] = df[activity_label_b_col].astype(str) if activity_label_b_col else "B"
    out["activity_mean"] = pd.to_numeric(df[activity_mean_col], errors="coerce") if activity_mean_col else float("nan")
    out["activity_sd"] = pd.to_numeric(df[activity_sd_col], errors="coerce") if activity_sd_col else float("nan")
    out["activity_mean_a"] = pd.to_numeric(df[activity_mean_a_col], errors="coerce") if activity_mean_a_col else float("nan")
    out["activity_mean_b"] = pd.to_numeric(df[activity_mean_b_col], errors="coerce") if activity_mean_b_col else float("nan")

    # DELTA standardization: delta_ab = (A - B)
    if activity_mean_a_col and activity_mean_b_col and out["activity_mean_a"].notna().any() and out["activity_mean_b"].notna().any():
        out["delta_ab"] = out["activity_mean_a"] - out["activity_mean_b"]
    elif activity_delta_col:
        out["delta_ab"] = -pd.to_numeric(df[activity_delta_col], errors="coerce")
    elif delta_col:
        out["delta_ab"] = pd.to_numeric(df[delta_col], errors="coerce")
    else:
        out["delta_ab"] = float("nan")

    # significance used for ranking/plots: stats_p first (as requested)
    sig = out["stats_p"].copy()
    sig_source = "stats_p"
    if not pd.to_numeric(sig, errors="coerce").notna().any():
        # fallback: pvalue-like columns
        if pd.to_numeric(out["pvalue"], errors="coerce").notna().any():
            sig = out["pvalue"].copy()
            sig_source = "pvalue"
        elif pd.to_numeric(out["fdr"], errors="coerce").notna().any():
            sig = out["fdr"].copy()
            sig_source = "fdr"
        else:
            sig_source = "missing"

    out["sig_value"] = pd.to_numeric(sig, errors="coerce")
    out["sig_source"] = sig_source  # string column for UI/warnings
    out["neglog10_sig"] = _neglog10_series(out["sig_value"])

    # Category populated from stats_Main_Class
    out["category"] = df[main_class_col].astype(str) if main_class_col else ""

    out["lead_genes"] = df[lead_genes_col].astype(str) if lead_genes_col else ""
    out["lead_genes_n"] = pd.to_numeric(df[lead_genes_n_col], errors="coerce") if lead_genes_n_col else float("nan")

    # derive lead_genes_n if missing
    derived = out["lead_genes"].apply(_count_lead_genes)
    out["lead_genes_n"] = pd.to_numeric(out["lead_genes_n"], errors="coerce")
    out["lead_genes_n"] = out["lead_genes_n"].fillna(derived)

    # optional interpretive stats
    tstat_col = None
    score_col = None
    for c in df.columns:
        cn = _canon_alnum(c)
        cl = str(c).lower()
        if tstat_col is None and cl.startswith("stats_") and ("tstat" in cn):
            tstat_col = c
        if score_col is None and cl.startswith("stats_") and ("score" in cn or cn.endswith("z")):
            score_col = c
    out["tstat"] = pd.to_numeric(df[tstat_col], errors="coerce") if tstat_col else float("nan")
    out["score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else float("nan")

    return out


def read_summary_table(tsv_path: Path) -> pd.DataFrame:
    usecols = select_usecols_from_header(tsv_path)
    df_raw = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "NaN", "nan", "NULL", "null", "None", "none"],
        usecols=usecols,
    )
    return build_canonical_view(df_raw)


def _summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    sig = pd.to_numeric(df["sig_value"], errors="coerce")
    sig05 = int((sig <= 0.05).sum()) if sig.notna().any() else None

    delta = pd.to_numeric(df["delta_ab"], errors="coerce")
    up = int((delta > 0).sum()) if delta.notna().any() else None
    down = int((delta < 0).sum()) if delta.notna().any() else None

    tissue = df["selected_tissue"].dropna().astype(str).iloc[0] if df["selected_tissue"].notna().any() else None
    disease = df["disease"].dropna().astype(str).iloc[0] if df["disease"].notna().any() else None

    sig_source = None
    if "sig_source" in df.columns and df["sig_source"].astype(str).str.strip().replace({"nan": ""}).ne("").any():
        sig_source = str(df["sig_source"].astype(str).dropna().iloc[0])

    has_stats_p = bool(pd.to_numeric(df.get("stats_p", pd.Series([], dtype=float)), errors="coerce").notna().any()) if "stats_p" in df.columns else False
    has_fdr = bool(pd.to_numeric(df.get("fdr", pd.Series([], dtype=float)), errors="coerce").notna().any()) if "fdr" in df.columns else False

    return {
        "disease": disease,
        "tissue": tissue,
        "n_pathways": int(len(df)),
        "n_sig_0p05": sig05,
        "n_up": up,
        "n_down": down,
        "sig_source": sig_source or ("stats_p" if has_stats_p else ("fdr" if has_fdr else "missing")),
        "has_stats_p": has_stats_p,
        "has_fdr": has_fdr,
    }


def _top_categories(df: pd.DataFrame, n: int = 6) -> List[Dict[str, Any]]:
    if df.empty or "category" not in df.columns:
        return []
    tmp = df.copy()
    tmp["category"] = tmp["category"].astype(str).replace({"nan": ""}).fillna("")
    tmp["category"] = tmp["category"].where(tmp["category"] != "", other="Uncategorized")
    g = tmp.groupby("category", dropna=False).agg(
        n=("pathway", "count"),
        mean_abs_delta=("delta_ab", lambda x: float(pd.to_numeric(x, errors="coerce").abs().mean()) if len(x) else float("nan")),
        mean_sig=("sig_value", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else float("nan")),
    ).reset_index()
    g = g.sort_values(["n", "mean_abs_delta"], ascending=[False, False]).head(n)
    out = []
    for _, r in g.iterrows():
        out.append({
            "category": str(r["category"]),
            "n": int(r["n"]) if pd.notna(r["n"]) else 0,
            "mean_abs_delta": float(r["mean_abs_delta"]) if pd.notna(r["mean_abs_delta"]) else None,
            "mean_sig": float(r["mean_sig"]) if pd.notna(r["mean_sig"]) else None,
        })
    return out


# ----------------------------
# Selection
# ----------------------------

@dataclass
class SelectionConfig:
    topn: int = 10
    fdr_cutoff: float = 0.10  # keep arg name, but it applies to sig_value (stats_p) by default
    candidate_pool: int = 30
    use_llm: bool = False
    llm_model: str = "gpt-4.1"
    llm_api_key_env: str = "OPENAI_API_KEY"


def rank_pathways(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-300
    sig = pd.to_numeric(df["sig_value"], errors="coerce")
    sig_pos = sig.where(sig > 0, pd.NA).clip(lower=eps)
    neglog = sig_pos.apply(lambda v: -math.log10(v) if pd.notna(v) else float("nan"))

    delta = pd.to_numeric(df["delta_ab"], errors="coerce").abs().fillna(0.0)
    nes = pd.to_numeric(df["nes"], errors="coerce").abs().fillna(0.0)

    # core composite: significance * (effect + NES)
    rank_score = neglog.fillna(0.0) * (delta + 0.25 * nes)

    # mild bonus if a usable stats score exists
    if "score" in df.columns and pd.to_numeric(df["score"], errors="coerce").notna().any():
        rank_score = rank_score + 0.25 * pd.to_numeric(df["score"], errors="coerce").abs().fillna(0.0)

    out = df.copy()
    out["rank_score"] = rank_score
    return out.sort_values("rank_score", ascending=False, na_position="last")


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ----------------------------
# OpenAI: strict schema enforcement + resilient calls
# ----------------------------

def enforce_openai_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce strict JSON schema subset expectations:
      - Every object has: additionalProperties=false
      - Every object has: required listing EVERY key in properties (no extras)
      - Recurse into arrays/items and nested objects
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dict")

    def _fix(node: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(node, dict):
            return node

        if "anyOf" in node and isinstance(node["anyOf"], list):
            node["anyOf"] = [_fix(x) if isinstance(x, dict) else x for x in node["anyOf"]]

        t = node.get("type")

        if t == "object":
            props = node.get("properties")
            if props is None or not isinstance(props, dict):
                props = {}
                node["properties"] = props

            for k, v in list(props.items()):
                if isinstance(v, dict):
                    props[k] = _fix(v)
                else:
                    props[k] = {"type": "string"}

            node["additionalProperties"] = False
            node["required"] = list(props.keys())

        elif t == "array":
            items = node.get("items")
            if isinstance(items, dict):
                node["items"] = _fix(items)

        return node

    return _fix(json.loads(json.dumps(schema)))  # deep-copy


def _fix_structured_outputs_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = json.loads(json.dumps(payload))  # deep-copy
    fmt = (p.get("text") or {}).get("format")
    if not isinstance(fmt, dict):
        return p
    if fmt.get("type") != "json_schema":
        return p
    schema = fmt.get("schema")
    if isinstance(schema, dict):
        fmt["schema"] = enforce_openai_strict_schema(schema)
        fmt["strict"] = True
        p["text"]["format"] = fmt
    return p


def response_to_output_text(resp: dict) -> str:
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
        return resp["output_text"]

    texts: List[str] = []
    for item in resp.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                texts.append(c["text"])
    return "\n".join(texts).strip()


def openai_responses_create(payload: dict, api_key: Optional[str] = None, timeout_s: int = 120, max_retries: int = 5) -> dict:
    """
    Robust Responses API call:
      - fixes strict schema automatically
      - retries transient failures (429/5xx)
      - if LLM fails, raises (caller decides), but higher-level code never crashes the report
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set, but LLM features were requested.")

    fixed_payload = _fix_structured_outputs_payload(payload)

    def _sleep(attempt: int) -> None:
        base = min(8.0, 2.0 ** attempt)
        time.sleep(base + random.random())

    try:
        
        client = OpenAI(api_key=api_key)

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = client.responses.create(**fixed_payload)
                if hasattr(resp, "model_dump"):
                    return resp.model_dump()
                if isinstance(resp, dict):
                    return resp
                return json.loads(str(resp))
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "invalid_json_schema" in msg or "invalid schema" in msg:
                    raise
                if any(x in msg for x in ["429", "rate", "timeout", "temporarily", "502", "503", "504", "server error"]):
                    _sleep(attempt)
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("OpenAI request failed (unknown).")
    except ImportError:
        pass

    

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=fixed_payload,
                timeout=timeout_s,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep(attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            _sleep(attempt)
    raise last_err if last_err else RuntimeError("OpenAI request failed (unknown).")


# ----------------------------
# LLM: selection (optional)
# ----------------------------

def llm_select_top_pathways(
    disease: str,
    tissue: Optional[str],
    candidates: pd.DataFrame,
    cfg: SelectionConfig,
) -> Tuple[List[str], Dict[str, Any]]:
    cols = ["pathway", "category", "stats_p", "pvalue", "fdr", "nes", "delta_ab", "lead_genes_n"]
    rows = candidates[[c for c in cols if c in candidates.columns]].head(cfg.candidate_pool).fillna("").to_dict(orient="records")

    schema = {
        "type": "object",
        "properties": {
            "selected_pathways": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": cfg.topn,
                "maxItems": cfg.topn,
            },
            "rationale": {"type": "string"},
            "highlights": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6},
        },
    }
    schema = enforce_openai_strict_schema(schema)

    payload = {
        "model": cfg.llm_model,
        "input": [
            {"role": "system", "content": (
                "You are a senior biomedical analyst. Select the most disease-relevant pathways from a candidate list.\n"
                "Pick EXACTLY topn pathways by name (verbatim). Prefer mechanistic plausibility + interpretability.\n"
                "Use stats_p as the primary significance signal when present (lower is better)."
            )},
            {"role": "user", "content": json.dumps({
                "disease": disease,
                "selected_tissue": tissue,
                "topn": cfg.topn,
                "candidates": rows,
            }, ensure_ascii=False)},
        ],
        "text": {"format": {"type": "json_schema", "name": "pathway_selection", "schema": schema, "strict": True}},
    }

    resp = openai_responses_create(payload, api_key=os.getenv(cfg.llm_api_key_env))
    obj = _extract_json_object(response_to_output_text(resp)) or {}
    selected = [str(x) for x in obj.get("selected_pathways", []) if x]
    return selected, {"used_llm": True, "llm_obj": obj}


def select_top_pathways(df: pd.DataFrame, cfg: SelectionConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ranked = rank_pathways(df)

    sig = pd.to_numeric(ranked["sig_value"], errors="coerce")
    if sig.notna().any():
        ranked_sig = ranked[sig <= cfg.fdr_cutoff].copy()
        if len(ranked_sig) >= cfg.topn:
            ranked = ranked_sig

    debug: Dict[str, Any] = {"used_llm": False}

    if cfg.use_llm:
        disease = str(df["disease"].dropna().iloc[0]) if df["disease"].notna().any() else "Disease"
        tissue = str(df["selected_tissue"].dropna().iloc[0]) if df["selected_tissue"].notna().any() else None
        try:
            chosen, llm_debug = llm_select_top_pathways(disease, tissue, ranked, cfg)
            debug.update(llm_debug)

            chosen_set = set(chosen)
            picked = ranked[ranked["pathway"].astype(str).isin(chosen_set)].copy()
            if len(picked) < cfg.topn:
                filler = ranked[~ranked["pathway"].astype(str).isin(chosen_set)].head(cfg.topn - len(picked))
                picked = pd.concat([picked, filler], ignore_index=True)
            return picked.head(cfg.topn), debug
        except Exception as e:
            debug["llm_error"] = str(e)

    return ranked.head(cfg.topn).copy(), debug


# ----------------------------
# Narrative (LLM) — robust + strict-safe schema
# ----------------------------

@dataclass
class NarrativeConfig:
    enabled: bool = False
    model: str = "gpt-4.1"
    api_key_env: str = "OPENAI_API_KEY"


def llm_generate_narrative(
    disease: str,
    tissue: Optional[str],
    top_df: pd.DataFrame,
    dataset_summary: Dict[str, Any],
    cfg: NarrativeConfig,
) -> Dict[str, Any]:
    if not cfg.enabled:
        return {}

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        return {"_error": f"{cfg.api_key_env} not set; narrative disabled."}

    cols = [
        "pathway", "category",
        "stats_p", "fdr", "pvalue",
        "nes", "delta_ab",
        "lead_genes", "lead_genes_n", "tstat", "score",
        "activity_label_a", "activity_label_b",
        "activity_mean_a", "activity_mean_b", "activity_mean", "activity_sd",
    ]
    rows = top_df[[c for c in cols if c in top_df.columns]].fillna("").to_dict(orient="records")

    schema = {
        "type": "object",
        "properties": {
            "executive_summary_html": {"type": "string"},
            "story_arc_html": {"type": "string"},
            "key_takeaways": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6},
            "pathway_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pathway": {"type": "string"},
                        "note_html": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "confounders": {"type": "string"},
                        "next_experiments": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
                    },
                },
            },
            "cross_pathway_themes": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 7},
            "interpretation_guidance_html": {"type": "string"},
            "methods_note_html": {"type": "string"},
        },
    }
    schema = enforce_openai_strict_schema(schema)

    system = (
        "You are a senior biomedical scientist writing a crisp, high-insight report for a disease cohort.\n"
        "Goal: turn Top-N pathways into a coherent mechanistic story + actionable next steps.\n"
        "Rules:\n"
        "- Output MUST be valid JSON matching the schema. No extra keys.\n"
        "- Write compact HTML fragments only (no full document).\n"
        "- Make it feel like a real internal analysis: connect pathways to disease biology, suggest confounders, propose experiments.\n"
        "- Direction: delta_ab = (A - B). Positive means higher in label A.\n"
        "- Significance: use stats_p when present (lower = stronger). If missing, say it's missing.\n"
        "- Do NOT invent citations or claim web search.\n"
        "- If data is weak (high stats_p), explain that plainly and shift to hypothesis framing.\n"
    )

    payload = {
        "model": cfg.model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "disease": disease,
                "selected_tissue": tissue,
                "dataset_summary": dataset_summary,
                "top_pathways": rows,
                "format_notes": {
                    "tone": "sharp, insightful, confident but honest about uncertainty",
                    "length": "about 1-2 pages worth of content",
                    "style": "modern minimal HTML; short paragraphs; occasional bullets",
                },
            }, ensure_ascii=False)},
        ],
        "text": {"format": {"type": "json_schema", "name": "ipaa_report_narrative_v3", "schema": schema, "strict": True}},
    }

    try:
        resp = openai_responses_create(payload, api_key=api_key)
        obj = _extract_json_object(response_to_output_text(resp))
        if not isinstance(obj, dict):
            return {}

        for k in ["executive_summary_html", "story_arc_html", "interpretation_guidance_html", "methods_note_html"]:
            if k in obj and isinstance(obj[k], str):
                obj[k] = _strip_unsafe_html(obj[k])

        if isinstance(obj.get("pathway_notes"), list):
            for it in obj["pathway_notes"]:
                if isinstance(it, dict) and isinstance(it.get("note_html"), str):
                    it["note_html"] = _strip_unsafe_html(it["note_html"])

        return obj or {}
    except Exception as e:
        return {"_error": str(e)}


# ----------------------------
# plots (responsive + dynamic)
# ----------------------------

def make_plots(top_df: pd.DataFrame) -> Dict[str, str]:
    df = top_df.copy()
    df["pathway_short"] = df["pathway"].astype(str)

    n = max(1, len(df))
    bar_h = max(320, min(980, 180 + 46 * n))
    scat_h = 380
    cat_h = 320

    delta = pd.to_numeric(df.get("delta_ab", pd.Series([], dtype=float)), errors="coerce")
    df["direction"] = delta.apply(lambda v: "Higher in A" if pd.notna(v) and v > 0 else ("Higher in B" if pd.notna(v) and v < 0 else "Flat/NA"))

    fig_bar = px.bar(
        df.sort_values("delta_ab", ascending=True),
        x="delta_ab", y="pathway_short", orientation="h",
        color="direction",
        hover_data={c: True for c in df.columns if c != "pathway_short"},
        title="Top pathways by effect size (Δ activity = A − B)",
    )
    fig_bar.update_layout(height=bar_h, autosize=True, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="Direction")

    xcol = "nes" if "nes" in df.columns and pd.to_numeric(df["nes"], errors="coerce").notna().any() else "delta_ab"
    fig_scatter = px.scatter(
        df,
        x=xcol,
        y=pd.to_numeric(df["neglog10_sig"], errors="coerce"),
        hover_name="pathway_short",
        size=pd.to_numeric(df["delta_ab"], errors="coerce").abs().fillna(0.0) + 1e-9,
        color="category" if "category" in df.columns else None,
        title="Signal overview (NES vs stats_p)" if xcol == "nes" else "Signal overview (Δ vs stats_p)",
    )
    fig_scatter.update_layout(
        height=scat_h,
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="-log10(stats_p)",
    )

    if "category" in df.columns and df["category"].astype(str).str.strip().replace({"nan": ""}).ne("").any():
        cat = df.copy()
        cat["category"] = cat["category"].astype(str).replace({"nan": ""}).fillna("")
        cat["category"] = cat["category"].where(cat["category"] != "", other="Uncategorized")
        g = cat.groupby("category").agg(n=("pathway", "count")).reset_index().sort_values("n", ascending=True)
        fig_cat = px.bar(g, x="n", y="category", orientation="h", title="Top-N category composition")
        fig_cat.update_layout(height=cat_h, autosize=True, margin=dict(l=20, r=20, t=60, b=20))
        cat_html = pio.to_html(fig_cat, include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True})
    else:
        cat_html = "<div class='empty'>No Category values found (stats_Main_Class missing).</div>"

    cfg = {"displayModeBar": False, "responsive": True}
    return {
        "bar": pio.to_html(fig_bar, include_plotlyjs=False, full_html=False, config=cfg),
        "scatter": pio.to_html(fig_scatter, include_plotlyjs=False, full_html=False, config=cfg),
        "category": cat_html,
    }


# ----------------------------
# HTML rendering (beautified + interactive)
# ----------------------------

def render_report_html(
    disease: str,
    tissue: Optional[str],
    dataset_summary: Dict[str, Any],
    top_df: pd.DataFrame,
    narrative: Dict[str, Any],
    selection_debug: Dict[str, Any],
    input_relpath: str,
    cdn: Dict[str, str] = DEFAULT_CDN,
) -> str:
    plots = make_plots(top_df)

    exec_html = narrative.get("executive_summary_html", "") if narrative else ""
    story_arc_html = narrative.get("story_arc_html", "") if narrative else ""
    takeaways = narrative.get("key_takeaways", []) if narrative else []
    interpret_html = narrative.get("interpretation_guidance_html", "") if narrative else ""
    methods_html = narrative.get("methods_note_html", "") if narrative else ""
    narr_error = narrative.get("_error") if narrative else None

    pathway_notes_map: Dict[str, Dict[str, Any]] = {}
    pn = narrative.get("pathway_notes", []) if narrative else []
    if isinstance(pn, list):
        for it in pn:
            if isinstance(it, dict):
                k = str(it.get("pathway", "")).strip()
                if k:
                    pathway_notes_map[k] = it

    if not exec_html:
        exec_html = (
            f"<p>This report summarizes the top pathways from <code>{html.escape(input_relpath)}</code> "
            f"for <b>{html.escape(disease)}</b>{' ('+html.escape(tissue)+')' if tissue else ''}. "
            "Pathways are ranked by a composite of <b>stats_p</b> significance and effect size (Δ). "
            "Category is derived from <code>stats_Main_Class</code> when available.</p>"
        )
        if narr_error:
            exec_html += f"<p class='warn'>Narrative generation failed: <code>{html.escape(str(narr_error))}</code></p>"

    takeaways_html = ""
    if isinstance(takeaways, list) and takeaways:
        takeaways_html = "<ul class='takeaways'>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in takeaways) + "</ul>"

    label_a = str(top_df["activity_label_a"].dropna().iloc[0]) if top_df["activity_label_a"].notna().any() else "A"
    label_b = str(top_df["activity_label_b"].dropna().iloc[0]) if top_df["activity_label_b"].notna().any() else "B"

    cat_summary = _top_categories(top_df, n=6)
    cat_pills = "".join(
        f"<span class='pill' title='n={c['n']} · mean|Δ|={_fmt_f(c.get('mean_abs_delta'))}'>"
        f"{html.escape(c['category'])} · {c['n']}</span>"
        for c in cat_summary
    ) if cat_summary else "<span class='pill'>No categories</span>"

    sig_source = str(dataset_summary.get("sig_source") or "stats_p")
    warn_sig = ""
    if sig_source != "stats_p":
        warn_sig = (
            "<div class='warn'>"
            f"<b>Warning:</b> <code>stats_p</code> not found/usable. Significance is falling back to <code>{html.escape(sig_source)}</code>. "
            "If you expect stats_p, confirm the column exists in pathway_summary.tsv."
            "</div>"
        )

    detail_blocks = []
    for _, r in top_df.iterrows():
        pw = str(r.get("pathway", ""))
        pid = f"pw-{_slug(pw)}"

        delta = _safe_float(r.get("delta_ab"))
        stats_p = _safe_float(r.get("stats_p"))
        fdr = _safe_float(r.get("fdr"))
        nes = _safe_float(r.get("nes"))

        sig_show = _safe_float(r.get("sig_value"))  # stats_p (or fallback)
        direction = (
            f"Higher in <b>{html.escape(label_a)}</b>" if (delta is not None and delta > 0) else
            (f"Higher in <b>{html.escape(label_b)}</b>" if (delta is not None and delta < 0) else "No clear direction")
        )

        cat = str(r.get("category", "") or "")
        lead_n = _safe_float(r.get("lead_genes_n"))
        lead_genes = str(r.get("lead_genes", "") or "")

        llm_note = pathway_notes_map.get(pw, {})
        note_html = llm_note.get("note_html", "") if isinstance(llm_note, dict) else ""
        why = llm_note.get("why_it_matters", "") if isinstance(llm_note, dict) else ""
        conf = llm_note.get("confounders", "") if isinstance(llm_note, dict) else ""
        nxt = llm_note.get("next_experiments", []) if isinstance(llm_note, dict) else []

        if not note_html:
            note_html = (
                f"<p><b>{html.escape(pw)}</b>: {direction}. "
                f"(Δ={_fmt_f(delta)}, stats_p={_fmt_p(stats_p)}, NES={_fmt_f(nes)})</p>"
            )

        nxt_html = ""
        if isinstance(nxt, list) and nxt:
            nxt_html = "<ul class='mini'>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in nxt[:7]) + "</ul>"

        why_html = f"<div class='mini-block'><div class='mini-h'>Why it matters</div><div class='mini-t'>{html.escape(str(why))}</div></div>" if why else ""
        conf_html = f"<div class='mini-block'><div class='mini-h'>Confounders to check</div><div class='mini-t'>{html.escape(str(conf))}</div></div>" if conf else ""

        lead_html = ""
        if lead_genes or lead_n is not None:
            lead_html = (
                "<div class='mini-block'>"
                f"<div class='mini-h'>Lead genes</div>"
                f"<div class='mini-t'>{html.escape(str(int(lead_n)) if lead_n is not None and not math.isnan(lead_n) else '')}"
                f"{(' · ' + html.escape(lead_genes[:220]) + ('…' if len(lead_genes) > 220 else '')) if lead_genes else ''}</div>"
                "</div>"
            )

        badge = "pos" if (delta is not None and delta > 0) else ("neg" if (delta is not None and delta < 0) else "neu")

        detail_blocks.append(f"""
          <details class="accordion pathway-card" id="{pid}">
            <summary>
              <div class="row">
                <span class="pw">{html.escape(pw)}</span>
                <span class="pill">{html.escape(cat[:60]) if cat and cat.lower() != "nan" else "Uncategorized"}</span>
                <span class="badge {badge}">{("A↑" if badge=="pos" else ("B↑" if badge=="neg" else "—"))}</span>
              </div>
              <span class="meta">{direction} · Δ {_fmt_f(delta)} · stats_p {_fmt_p(sig_show)} · NES {_fmt_f(nes)}</span>
            </summary>
            <div class="detail-body">
              <div class="note">{note_html}</div>
              <div class="mini-grid">
                {why_html}
                {conf_html}
                {lead_html}
              </div>
              {("<div class='mini-block'><div class='mini-h'>Next experiments</div>" + nxt_html + "</div>") if nxt_html else ""}
              {f"<div class='mini-block'><div class='mini-h'>Other stats</div><div class='mini-t'>FDR={_fmt_p(fdr)} · raw p={_fmt_p(_safe_float(r.get('pvalue')))}</div></div>" if (fdr is not None or _safe_float(r.get('pvalue')) is not None) else ""}
            </div>
          </details>
        """)

    # TABLE (Category only; show stats_p, plus optional FDR)
    table_cols = ["pathway", "category", "stats_p", "fdr", "nes", "delta_ab", "lead_genes_n", "lead_genes"]
    table_df = top_df[[c for c in table_cols if c in top_df.columns]].copy()

    if "stats_p" in table_df.columns:
        table_df["stats_p"] = table_df["stats_p"].apply(_fmt_p)
    if "fdr" in table_df.columns:
        table_df["fdr"] = table_df["fdr"].apply(_fmt_p)
    if "nes" in table_df.columns:
        table_df["nes"] = table_df["nes"].apply(_fmt_f)
    if "delta_ab" in table_df.columns:
        table_df["delta_ab"] = table_df["delta_ab"].apply(_fmt_f)

    # clean nan strings
    for c in table_df.columns:
        table_df[c] = table_df[c].astype(str).replace({"nan": ""}).fillna("")

    table_json = table_df.to_dict(orient="records")

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    n_pathways = dataset_summary.get("n_pathways")
    n_sig = dataset_summary.get("n_sig_0p05")
    n_up = dataset_summary.get("n_up")
    n_down = dataset_summary.get("n_down")

    debug_blob = html.escape(json.dumps(selection_debug, default=str)[:15000])

    story_block = f"<div class='card' style='margin-top:14px;'><h2>Story arc</h2>{story_arc_html}</div>" if story_arc_html else ""
    interpret_block = f"<div class='card' style='margin-top:14px;'><h2>Interpretation guidance</h2>{interpret_html}</div>" if interpret_html else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(disease)} · IPAA Pathway Report</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{cdn["fonts"]}" rel="stylesheet">

  <script src="{cdn["plotly"]}"></script>
  <link href="{cdn["tabulator_css"]}" rel="stylesheet">
  <script src="{cdn["tabulator_js"]}"></script>

  <style>
    :root {{
      --bg: #060A16;
      --bg2:#0a1026;
      --card: rgba(255,255,255,0.06);
      --stroke: rgba(255,255,255,0.10);
      --stroke2: rgba(255,255,255,0.14);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.72);
      --muted2: rgba(255,255,255,0.55);
      --shadow: 0 18px 60px rgba(0,0,0,0.40);
      --radius: 18px;
      --ok: #38bdf8;
      --warn: #fbbf24;
      --bad: #fb7185;
      --good: #34d399;
      --chip: rgba(0,0,0,0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background:
        radial-gradient(1200px 800px at 18% 8%, #1b2a7a 0%, var(--bg2) 55%),
        radial-gradient(1000px 700px at 82% 0%, #51215f 0%, var(--bg2) 60%),
        linear-gradient(180deg, var(--bg), var(--bg2));
      color: var(--text);
    }}
    .page {{ max-width: 1160px; margin: 18px auto; padding: 0 18px 34px 18px; }}
    .topbar {{
      position: sticky; top: 0; z-index: 50;
      backdrop-filter: blur(12px);
      background: linear-gradient(135deg, rgba(8,10,18,0.92), rgba(10,16,38,0.75));
      border-bottom: 1px solid rgba(255,255,255,0.10);
    }}
    .topbar-inner {{
      max-width: 1160px; margin: 0 auto; padding: 10px 18px;
      display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
    }}
    .brand {{ display:flex; gap:10px; align-items:center; }}
    .dot {{ width:10px; height:10px; border-radius:999px; background: var(--ok); box-shadow: 0 0 22px rgba(56,189,248,0.65); }}
    .title {{ font-weight: 700; letter-spacing:-0.02em; }}
    .nav {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    .nav a {{
      color: var(--muted); text-decoration:none; font-size:12px;
      padding:6px 10px; border:1px solid rgba(255,255,255,0.12); border-radius:999px;
      background: rgba(0,0,0,0.12);
    }}
    .nav a:hover {{ color: var(--text); border-color: rgba(255,255,255,0.20); }}
    header {{
      margin-top: 14px;
      background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
      border: 1px solid var(--stroke2);
      border-radius: calc(var(--radius) + 2px);
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .title-row {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; flex-wrap:wrap; }}
    h1 {{ margin: 0; font-size: 26px; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top:6px; }}
    .chips {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .chip {{
      padding:7px 10px;
      border-radius:999px;
      border:1px solid var(--stroke2);
      background:var(--chip);
      color:var(--muted);
      font-size:12px;
      white-space:nowrap;
    }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:14px; }}
    @media (max-width: 920px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .card {{
      background:var(--card);
      border:1px solid var(--stroke2);
      border-radius:var(--radius);
      padding:16px;
      box-shadow:var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .card h2 {{ margin:0 0 10px 0; font-size:14px; text-transform:uppercase; color:var(--muted); letter-spacing:0.08em; }}
    .metrics {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; }}
    @media (max-width: 920px) {{ .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .metric {{ padding:12px; border-radius:14px; border:1px solid var(--stroke2); background:rgba(255,255,255,0.04); }}
    .metric .k {{ font-size:12px; color:var(--muted2); margin-bottom:6px; }}
    .metric .v {{ font-size:18px; font-weight:700; letter-spacing:-0.01em; }}
    .takeaways {{ margin:10px 0 0 18px; color:var(--muted); line-height:1.5; }}
    .section-title {{ margin:18px 2px 10px 2px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.10em; }}
    .plot {{
      background:rgba(255,255,255,0.04);
      border:1px solid var(--stroke2);
      border-radius:var(--radius);
      padding:8px 8px 0 8px;
      overflow:hidden;
      width:100%;
      min-height: 320px;
    }}
    .plot > div {{ width: 100% !important; }}
    .three {{ display:grid; grid-template-columns: 1.25fr 0.75fr; gap:14px; }}
    @media (max-width: 920px) {{ .three {{ grid-template-columns: 1fr; }} }}
    .accordion {{
      border:1px solid var(--stroke2);
      border-radius:14px;
      background:rgba(255,255,255,0.04);
      padding:12px 12px;
      margin:10px 0;
    }}
    details > summary {{ cursor:pointer; outline:none; list-style:none; }}
    details > summary::-webkit-details-marker {{ display:none; }}
    .row {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
    .pw {{ font-weight:700; letter-spacing:-0.01em; }}
    .pill {{
      padding:4px 8px;
      border-radius:999px;
      border:1px solid var(--stroke2);
      background:rgba(0,0,0,0.18);
      color:var(--muted);
      font-size:11px;
    }}
    .badge {{
      padding:4px 8px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,0.14);
      font-size:11px;
      background: rgba(255,255,255,0.06);
      color: var(--muted);
    }}
    .badge.pos {{ border-color: rgba(52,211,153,0.35); color: rgba(167,243,208,0.95); }}
    .badge.neg {{ border-color: rgba(251,113,133,0.35); color: rgba(254,202,202,0.95); }}
    .badge.neu {{ border-color: rgba(255,255,255,0.18); color: rgba(255,255,255,0.75); }}
    .meta {{ color:var(--muted); font-size:12px; margin-top:6px; }}
    .detail-body {{ margin-top:10px; color:var(--muted); line-height:1.55; }}
    .note p {{ margin: 0 0 8px 0; }}
    .mini-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px; }}
    @media (max-width: 920px) {{ .mini-grid {{ grid-template-columns: 1fr; }} }}
    .mini-block {{ border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:10px; background: rgba(255,255,255,0.03); }}
    .mini-h {{ font-size:12px; color: var(--muted2); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.08em; }}
    .mini-t {{ font-size:13px; color: var(--muted); }}
    .mini {{ margin: 8px 0 0 18px; color: var(--muted); }}
    .toolbar {{
      display:flex; gap:10px; flex-wrap:wrap; align-items:center; justify-content:space-between;
      margin: 8px 0 0 0;
    }}
    .search {{
      width: min(560px, 100%);
      padding:10px 12px;
      border:1px solid rgba(255,255,255,0.16);
      border-radius:12px;
      background: rgba(0,0,0,0.18);
      color: var(--text);
      outline:none;
    }}
    .btn {{
      padding:9px 12px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.18);
      color: var(--muted);
      cursor:pointer;
      font-size:12px;
    }}
    .btn:hover {{ color: var(--text); border-color: rgba(255,255,255,0.24); }}
    .warn {{
      margin-top:10px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid rgba(251,191,36,0.28);
      background: rgba(251,191,36,0.10);
      color: rgba(255, 236, 179, 0.98);
      font-size: 12px;
    }}
    .footer {{ margin-top:18px; color:var(--muted2); font-size:12px; line-height:1.5; }}
    .small {{ font-size:12px; color:var(--muted2); }}
    .page-break {{ break-after: page; page-break-after: always; }}
    .empty {{ color: var(--muted2); padding: 10px; }}
    @media print {{
      @page {{
        size: A4;
        margin: 1.5cm 1.2cm;
      }}
      body {{
        background: white !important;
        color: #111 !important;
        margin: 0;
        padding: 0;
      }}
      .topbar {{ display: none !important; }}
      .page {{
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      header, .card, .plot, .accordion {{
        box-shadow: none !important;
        backdrop-filter: none !important;
        background: white !important;
        border-color: #ddd !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .grid {{
        grid-template-columns: 1fr !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .three {{
        grid-template-columns: 1fr !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .chip, .pill, .badge {{
        border-color: #ddd !important;
        background: white !important;
        color: #333 !important;
      }}
      .sub, .section-title, .small, .meta, .takeaways, .footer, .mini-t {{
        color: #333 !important;
      }}
      .plot {{
        max-width: 100% !important;
        overflow: visible !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .accordion {{
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      table {{
        font-size: 10px !important;
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .toolbar, .search, .btn {{
        display: none !important;
      }}
      h1, h2, h3 {{
        page-break-after: avoid;
        break-after: avoid;
      }}
    }}
  </style>
</head>

<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="dot"></div>
        <div class="title">{html.escape(disease)} · IPAA report</div>
      </div>
      <div class="nav">
        <a href="#summary">Summary</a>
        <a href="#plots">Plots</a>
        <a href="#table">Table</a>
        <a href="#notes">Notes</a>
        <a href="#methods">Methods</a>
      </div>
    </div>
  </div>

  <div class="page">
    <header id="summary">
      <div class="title-row">
        <div>
          <h1>{html.escape(disease)} · Pathway signal report</h1>
          <div class="sub">Generated {today} · input <code>{html.escape(input_relpath)}</code></div>
        </div>
        <div class="chips">
          <span class="chip">Tissue: {html.escape(tissue) if tissue else "n/a"}</span>
          <span class="chip">Labels: {html.escape(label_a)} vs {html.escape(label_b)}</span>
          <span class="chip">Top N: {len(top_df)}</span>
          <span class="chip">Sig: <code>stats_p</code></span>
        </div>
      </div>
      <div class="toolbar">
        <input id="globalSearch" class="search" placeholder="Search pathways, genes, notes, categories..."
               oninput="applyGlobalSearch()" />
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          <button class="btn" onclick="expandAll(true)">Expand all</button>
          <button class="btn" onclick="expandAll(false)">Collapse all</button>
          <button class="btn" onclick="downloadTable('tsv')">Download TSV</button>
        </div>
      </div>
      {warn_sig}
    </header>

    <div class="grid">
      <div class="card">
        <h2>Executive summary</h2>
        {exec_html}
        {takeaways_html}
        <div style="margin-top:12px;">
          <div class="small" style="margin-bottom:6px;">Top categories (in Top-N)</div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">{cat_pills}</div>
        </div>
      </div>

      <div class="card">
        <h2>Snapshot</h2>
        <div class="metrics">
          <div class="metric"><div class="k">Pathways in table</div><div class="v">{n_pathways if n_pathways is not None else "—"}</div></div>
          <div class="metric"><div class="k">stats_p ≤ 0.05</div><div class="v">{n_sig if n_sig is not None else "—"}</div></div>
          <div class="metric"><div class="k">Δ &gt; 0</div><div class="v">{n_up if n_up is not None else "—"}</div></div>
          <div class="metric"><div class="k">Δ &lt; 0</div><div class="v">{n_down if n_down is not None else "—"}</div></div>
        </div>
        <div class="small" style="margin-top:10px;">
          Δ is <b>{html.escape(label_a)} − {html.escape(label_b)}</b>. Positive means higher in {html.escape(label_a)}.
        </div>
      </div>
    </div>

    {story_block}
    {interpret_block}

    <div class="section-title" id="plots">Top pathways</div>
    <div class="three">
      <div class="plot" id="plot_bar">{plots["bar"]}</div>
      <div class="plot" id="plot_scatter">{plots["scatter"]}</div>
    </div>
    <div class="plot" style="margin-top:14px;" id="plot_category">{plots["category"]}</div>

    <div class="page-break"></div>

    <div class="section-title" id="table">Details</div>
    <div class="card">
      <h2>Top pathways table</h2>
      <div id="tabulator"></div>
      <div class="small" style="margin-top:10px;">
        Tip: filter in headers, sort columns, click a row to jump to its pathway note.
      </div>
    </div>

    <div class="section-title" id="notes">Pathway notes</div>
    <div class="card">
      {''.join(detail_blocks)}
    </div>

    <div class="card footer" id="methods">
      <h2>Methods & provenance</h2>
      {methods_html if methods_html else "<p>Inputs come from <code>pathway_summary.tsv</code>. Significance uses <code>stats_p</code> when present (otherwise falls back to other p-value-like columns). Category uses <code>stats_Main_Class</code>. Interpretations are hypothesis-generating.</p>"}
      <details style="margin-top:10px;">
        <summary class="small">Debug (selection)</summary>
        <pre class="small" style="white-space: pre-wrap;">{debug_blob}</pre>
      </details>
    </div>
  </div>

<script>
  // ----------------------------
  // Global Search (filters: cards + Tabulator)
  // ----------------------------
  function applyGlobalSearch(){{
    const q = (document.getElementById("globalSearch").value || "").toLowerCase();

    document.querySelectorAll(".pathway-card").forEach(el => {{
      const t = (el.innerText || "").toLowerCase();
      el.style.display = t.includes(q) ? "" : "none";
    }});

    if (window._tabulator) {{
      if (!q) {{
        window._tabulator.clearFilter(true);
      }} else {{
        window._tabulator.setFilter((data) => {{
          const joined = Object.values(data || {{}}).join(" ").toLowerCase();
          return joined.includes(q);
        }});
      }}
    }}
  }}

  function expandAll(openIt){{
    document.querySelectorAll(".pathway-card").forEach(el => {{
      el.open = !!openIt;
    }});
  }}

  // ----------------------------
  // Tabulator table
  // ----------------------------
  const tableData = {json.dumps(table_json, ensure_ascii=False)};
  const tableCols = [
      {{title:"Pathway", field:"pathway", headerFilter:"input", widthGrow: 2}},
      {{title:"Category (stats_Main_Class)", field:"category", headerFilter:"input"}},
      {{title:"stats_p", field:"stats_p"}},
      {{title:"FDR", field:"fdr"}},
      {{title:"NES", field:"nes"}},
      {{title:"Δ (A−B)", field:"delta_ab"}},
      {{title:"Lead genes", field:"lead_genes_n"}},
      {{title:"Lead genes list", field:"lead_genes", widthGrow: 2}},
  ];

  window._tabulator = new Tabulator("#tabulator", {{
    data: tableData,
    layout: "fitColumns",
    height: "560px",
    movableColumns: true,
    columns: (tableData.length ? tableCols.filter(c => Object.prototype.hasOwnProperty.call(tableData[0], c.field)) : tableCols),
    rowClick: function(e, row) {{
      const data = row.getData();
      const pw = (data.pathway || "").toString();
      const id = "pw-" + pw.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,"").slice(0,80);
      const el = document.getElementById(id);
      if (el) {{
        el.open = true;
        el.scrollIntoView({{behavior:"smooth", block:"start"}});
      }}
    }}
  }});

  function downloadTable(fmt){{
    if (!window._tabulator) return;
    if (fmt === "tsv") {{
      window._tabulator.download("csv", "{html.escape(disease)}_top_pathways.tsv", {{delimiter:"\\t"}});
    }} else {{
      window._tabulator.download("csv", "{html.escape(disease)}_top_pathways.csv");
    }}
  }}

  // ----------------------------
  // Plotly responsiveness
  // ----------------------------
  function resizePlots() {{
    document.querySelectorAll('.js-plotly-plot').forEach(el => {{
      try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
    }});
  }}
  window.addEventListener('resize', () => resizePlots());
  setTimeout(resizePlots, 250);
  setTimeout(resizePlots, 900);
</script>
</body>
</html>
"""


# ----------------------------
# driver
# ----------------------------

def discover_disease_folders(outdir: Path) -> List[Tuple[str, Path]]:
    base = outdir / "engines" / "pathway_summary"
    if not base.exists():
        raise FileNotFoundError(f"Expected folder not found: {base}")

    found: List[Tuple[str, Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        candidates = [
            child / "pathway_summary.tsv",
            child / "summary.tsv",
        ]
        pick = next((p for p in candidates if p.exists()), None)
        if pick is not None:
            found.append((child.name, pick))
    return found


def write_index(base: Path, diseases: List[str]) -> None:
    links = "\n".join(
        f'<li class="li"><a class="a" href="{html.escape(d)}/report.html">{html.escape(d)}</a></li>'
        for d in diseases
    )
    idx = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IPAA Pathway Reports</title>
  <link href="{DEFAULT_CDN["fonts"]}" rel="stylesheet">
  <style>
    body {{
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      padding: 24px;
      background: #0b1020;
      color: rgba(255,255,255,0.92);
    }}
    h1 {{ margin: 0 0 10px 0; letter-spacing:-0.02em; }}
    .muted {{ color: rgba(255,255,255,0.70); margin-bottom: 14px; }}
    ul {{ padding-left: 18px; }}
    .li {{ margin: 8px 0; }}
    .a {{
      color: rgba(255,255,255,0.90);
      text-decoration:none;
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 12px;
      display: inline-block;
      background: rgba(255,255,255,0.04);
    }}
    .a:hover {{ border-color: rgba(255,255,255,0.24); }}
  </style>
</head>
<body>
  <h1>IPAA Pathway Reports</h1>
  <div class="muted">Generated reports per disease.</div>
  <ul>{links}</ul>
</body>
</html>"""
    (base / "index.html").write_text(idx, encoding="utf-8")


def _html_to_pdf(html_content: str, out_pdf: Path) -> bool:
    """
    Convert HTML to PDF using weasyprint. Returns True if successful, False otherwise.
    """
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=html_content).write_pdf(str(out_pdf))
        return True
    except ImportError:
        print(f"[WARN] weasyprint not installed. Skipping PDF generation. Install with: pip install weasyprint", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[WARN] PDF generation failed: {e}", file=sys.stderr)
        return False


def _config_to_args(cfg) -> "argparse.Namespace":
    """Build args-like object from ReportPhaseConfig or dict."""
    import argparse
    def _g(k, d):
        if hasattr(cfg, k):
            return getattr(cfg, k)
        return cfg.get(k, d) if isinstance(cfg, dict) else d
    return argparse.Namespace(
        outdir=_g("outdir", ""),
        topn=_g("topn", 20),
        fdr_cutoff=_g("fdr_cutoff", 0.10),
        candidate_pool=_g("candidate_pool", 30),
        llm_selector=_g("llm_selector", "auto"),
        llm_narrative=_g("llm_narrative", "auto"),
        llm_model=_g("llm_model", "gpt-4o-mini"),
        api_key_env=_g("api_key_env", "OPENAI_API_KEY"),
        generate_pdf=_g("generate_pdf", False),
    )


def run_reports_from_config(config) -> int:
    """Run HTML report generation from config. No argparse. Returns exit code."""
    args = _config_to_args(config)
    return _main_impl(args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Pipeline OUT_ROOT directory")
    ap.add_argument("--topn", type=int, default=10, help="Top N pathways to include")
    ap.add_argument("--fdr-cutoff", type=float, default=0.10, help="Prefer significant pathways before ranking (applies to stats_p if present)")
    ap.add_argument("--candidate-pool", type=int, default=30, help="Candidate pool size for LLM selection (if enabled)")
    ap.add_argument("--llm-selector", choices=["off", "on", "auto"], default="auto", help="LLM top-pathway selection")
    ap.add_argument("--llm-narrative", choices=["off", "on", "auto"], default="auto", help="LLM narrative")
    ap.add_argument("--llm-model", default="gpt-4.1", help="LLM model id")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var containing API key")
    ap.add_argument("--generate-pdf", action="store_true", help="Generate PDF reports alongside HTML (requires weasyprint)")
    args = ap.parse_args()
    return _main_impl(args)


def _main_impl(args) -> int:
    outdir = Path(args.outdir).expanduser().resolve()
    diseases = discover_disease_folders(outdir)
    if not diseases:
        base = outdir / "engines" / "pathway_summary"
        msg = (
            f"No diseases found under {base}/*/(pathway_summary.tsv or summary.tsv)\n"
            f"Check: ls -lah {base}\n"
            f"And inside a disease folder: ls -lah {base}/<DISEASE>/\n"
        )
        print(msg, file=sys.stderr)
        return 2

    have_key = bool(os.getenv(args.api_key_env))
    use_llm_selector = (args.llm_selector == "on") or (args.llm_selector == "auto" and have_key)
    use_llm_narr = (args.llm_narrative == "on") or (args.llm_narrative == "auto" and have_key)

    sel_cfg = SelectionConfig(
        topn=args.topn,
        fdr_cutoff=args.fdr_cutoff,
        candidate_pool=args.candidate_pool,
        use_llm=use_llm_selector,
        llm_model=args.llm_model,
        llm_api_key_env=args.api_key_env,
    )
    nar_cfg = NarrativeConfig(
        enabled=use_llm_narr,
        model=args.llm_model,
        api_key_env=args.api_key_env,
    )

    base = outdir / "engines" / "pathway_summary"
    made: List[str] = []

    for disease, tsv_path in diseases:
        df = read_summary_table(tsv_path)
        summary = _summarize_dataset(df)
        tissue = summary.get("tissue")

        top_df, debug = select_top_pathways(df, sel_cfg)
        narrative = llm_generate_narrative(disease, tissue, top_df, summary, nar_cfg) if nar_cfg.enabled else {}

        try:
            input_relpath = str(tsv_path.relative_to(outdir))
        except Exception:
            input_relpath = str(tsv_path)

        report_html = render_report_html(
            disease, tissue, summary, top_df, narrative, debug, input_relpath=input_relpath
        )

        out_html = base / disease / "report.html"
        out_html.write_text(report_html, encoding="utf-8")
        made.append(disease)
        print(f"Wrote: {out_html}")

        if args.generate_pdf:
            out_pdf = base / disease / "report.pdf"
            if _html_to_pdf(report_html, out_pdf):
                print(f"Wrote: {out_pdf}")

    write_index(base, made)
    print(f"Wrote: {base / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
