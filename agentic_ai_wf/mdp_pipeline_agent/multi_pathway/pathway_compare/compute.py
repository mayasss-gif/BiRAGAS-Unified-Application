from __future__ import annotations
import logging, re, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PCConfig
from .tables_figs import run_multi_pathway_outputs  # local import to avoid cycles
from .tables_figs import run_one_pathway_all_outputs  # local import to avoid cycles
# ---------------- robust aggregations (helpers) ----------------

def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric, preserving NaNs."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return pd.to_numeric(s, errors="coerce")

def _nanmedian_numeric(s: pd.Series) -> float:
    """Safe nanmedian over a possibly non-numeric/empty series; returns NaN if no data."""
    s_num = _to_numeric_series(s).dropna()
    if s_num.empty:
        return float("nan")
    return float(np.nanmedian(s_num.values))

def _nanmean_numeric(s: pd.Series) -> float:
    s_num = _to_numeric_series(s).dropna()
    if s_num.empty:
        return float("nan")
    return float(np.nanmean(s_num.values))

def _nanmax_numeric(s: pd.Series) -> float:
    s_num = _to_numeric_series(s).dropna()
    if s_num.empty:
        return float("nan")
    return float(np.nanmax(s_num.values))

def _pick_longest_list(lists: pd.Series) -> List[str]:
    """Return the longest list/tuple/set (by length). If none, []"""
    best = []
    for z in lists:
        if isinstance(z, (list, tuple, set)):
            z = list(z)
            if len(z) > len(best):
                best = z
    return best

# ---------------- entity cleaning ----------------

_MOUSE_TAG_RE = re.compile(r"(^|\b)(mouse|murine|mmu)(\b|$)|[\[\(]\s*mouse\s*[\]\)]", re.I)
_MM_SUFFIX_RE = re.compile(r"(?:[_\-\s]*(mm|mm9|mm10|mm39|mmu))$", re.I)

def _clean_entity_name(entity: str, etype: str, cfg: PCConfig) -> str:
    s = str(entity or "").strip()
    if not s:
        return s
    if etype == "epigenetic" and cfg.clean_mm_suffix_in_epigenetic:
        s = _MM_SUFFIX_RE.sub("", s).strip()
    if etype == "tf" and cfg.clean_mouse_tags_in_tf:
        s = _MOUSE_TAG_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------- tidy builder (directional/non-directional) ----------------

def _as_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _sig_from_qp(q, p, cap: float):
    try:
        if q is not None and np.isfinite(q):
            return min(-np.log10(max(q, 1e-300)), cap)
        if p is not None and np.isfinite(p):
            return min(-np.log10(max(p, 1e-300)), cap)
    except Exception:
        pass
    return np.nan

def _is_directional_schema(js_obj: dict) -> bool:
    try:
        return all(isinstance(v, dict) and ("UP" in v or "DOWN" in v) for v in js_obj.values())
    except Exception:
        return False

# NOTE: cfg is provided at runtime by run_pathway_compare (module-level injection)
cfg: PCConfig

def build_tidy_entities(all_json: Dict[str, dict], sig_cap: float) -> pd.DataFrame:
    """
    Columns:
      disease, pathway, direction (ANY/UP/DOWN), entity_type, entity, overlap_genes,
      OR, qval, pval, sig, k, a, b, N, Jaccard
    """
    recs: List[dict] = []

    def push(disease: str, pathway: str, direction: str, etype: str, item: dict):
        ent_raw = str(item.get("entity", "")).strip()
        etype = str(etype).strip()
        ent = _clean_entity_name(ent_raw, etype, cfg)  # <- uses module-level cfg
        og = item.get("overlap_genes", [])
        og = [g for g in og if isinstance(g, str)]
        qv = _as_float(item.get("qval"))
        pv = _as_float(item.get("pval"))
        recs.append({
            "disease": disease,
            "pathway": pathway,
            "direction": direction,
            "entity_type": etype,
            "entity": ent,
            "overlap_genes": og,
            "OR": _as_float(item.get("OR")),
            "qval": qv,
            "pval": pv,
            "sig": _sig_from_qp(qv, pv, sig_cap),
            "k": _as_float(item.get("k")),
            "a": _as_float(item.get("a")),
            "b": _as_float(item.get("b")),
            "N": _as_float(item.get("N")),
            "Jaccard": _as_float(item.get("Jaccard")),
        })

    for disease, obj in all_json.items():
        if not isinstance(obj, dict):
            continue
        if _is_directional_schema(obj):
            for pathway, blk in obj.items():
                blk = blk or {}
                for d in ("UP", "DOWN"):
                    etypes = blk.get(d, {}) or {}
                    if not isinstance(etypes, dict):
                        continue
                    for et, arr in etypes.items():
                        if isinstance(arr, list):
                            for it in arr:
                                if isinstance(it, dict):
                                    push(disease, str(pathway), d, str(et), it)
        else:
            for pathway, etypes in obj.items():
                etypes = etypes or {}
                if not isinstance(etypes, dict):
                    continue
                for et, arr in etypes.items():
                    if isinstance(arr, list):
                        for it in arr:
                            if isinstance(it, dict):
                                push(disease, str(pathway), "ANY", str(et), it)

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return pd.DataFrame(columns=[
            "disease","pathway","direction","entity_type","entity","overlap_genes",
            "OR","qval","pval","sig","k","a","b","N","Jaccard"
        ])
    df = (df.sort_values(["qval","pval","sig"], ascending=[True, True, False], na_position="last")
            .drop_duplicates(["disease","pathway","direction","entity_type","entity"], keep="first"))
    return df


def _collapse_any(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    dir_df = df[df["direction"].isin(["UP","DOWN"])]
    if dir_df.empty:
        return df[df["direction"].eq("ANY")].copy()

    any_from_dir = (
        dir_df
        .groupby(["disease","pathway","entity_type","entity"], as_index=False)
        .agg(
            sig=("sig", _nanmax_numeric),                 # best significance across directions
            OR=("OR", _nanmedian_numeric),                # robust median OR (safe on all-NaN)
            qval=("qval", _nanmin := (lambda s: _to_numeric_series(s).min(skipna=True))),
            pval=("pval", _nanmin),
            Jaccard=("Jaccard", _nanmax_numeric),
            k=("k", _nanmax_numeric),
            a=("a", _nanmax_numeric),
            b=("b", _nanmax_numeric),
            N=("N", _nanmax_numeric),
            overlap_genes=("overlap_genes", _pick_longest_list),
        )
    )

    base_any = df[df["direction"].eq("ANY")]
    merged = pd.concat([base_any, any_from_dir], ignore_index=True)
    merged = (
        merged
        .sort_values(["qval","pval","sig"], ascending=[True, True, False], na_position="last")
        .drop_duplicates(["disease","pathway","entity_type","entity"], keep="first")
        .assign(direction="ANY")
    )
    return merged


def build_views_from_entities(df: pd.DataFrame, borrow: str = "up") -> Dict[str, pd.DataFrame]:
    borrow = (borrow or "none").lower()
    any_view = _collapse_any(df)
    up_view = df[df["direction"].eq("UP")].copy()
    down_view = df[df["direction"].eq("DOWN")].copy()

    def _borrow_into(view: pd.DataFrame, d: str) -> pd.DataFrame:
        borrowed = any_view.assign(direction=d, borrowed=True)
        merged = pd.concat([view, borrowed], ignore_index=True)
        return merged.drop_duplicates(["disease","pathway","entity_type","entity","direction"], keep="first")

    if borrow in {"up","both"}:
        up_view = _borrow_into(up_view, "UP")
    if borrow in {"down","both"}:
        down_view = _borrow_into(down_view, "DOWN")

    return {"ANY": any_view, "UP": up_view, "DOWN": down_view}


# ---------------- analysis runners (now call the new tables_figs helpers) ----------------

def run_individual_pathway_analysis(requested_pathways: List[str], views: Dict[str, pd.DataFrame], out_root: Path, cfg: PCConfig) -> None:
    
    out_root.mkdir(parents=True, exist_ok=True)
    for pw in requested_pathways:
        try:
            run_one_pathway_all_outputs(pw, views, out_root, cfg)
        except Exception as e:
            logging.error(f"individual analysis failed for '{pw}': {e}")

def run_multi_pathway_comparison(requested_pathways: List[str], views: Dict[str, pd.DataFrame], out_root: Path, cfg: PCConfig) -> None:
    
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        run_multi_pathway_outputs(requested_pathways, views, out_root, cfg)
    except Exception as e:
        logging.error(f"multi-pathway comparison failed: {e}")
