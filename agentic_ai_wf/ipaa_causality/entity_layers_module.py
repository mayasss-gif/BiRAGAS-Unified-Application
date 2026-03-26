# entity_layers_module.py
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

# ---- pandas (optional at runtime; clean for type-checkers) ----
import pandas as pd
from pandas import DataFrame



ENTITY_TYPES = ("tf", "epigenetic", "metabolite")


# -----------------------------
# Normalization helpers
# -----------------------------
_norm_re = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def norm_key(s: str) -> str:
    return _norm_re.sub("", (s or "").strip().lower())


def looks_non_human(label: str) -> bool:
    x = (label or "").lower()
    # conservative: only obvious species tags
    return any(tok in x for tok in ["mus musculus", "murine", "rattus", "rat ", " mouse", " (mouse", "(rat"])


def safe_label(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, (int, float)):
        return str(label)
    return str(label).strip()


def canonicalize_entity(entity: str, etype: str) -> str:
    s = safe_label(entity)
    if not s:
        return ""
    # TFs often gene symbols: keep uppercase and strip junk
    if etype == "tf":
        s = re.sub(r"\s+", "", s).upper()
        s = re.sub(r"[^A-Z0-9_\-\.]", "", s)
    else:
        s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# JSON schema-flex extraction
# -----------------------------
def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _pick_entity_and_weight(item: Any) -> Tuple[str, float]:
    """
    Accepts:
      - "TP53"
      - {"name": "TP53", "score": 3.1}
      - {"entity": "TP53", "weight": 2}
      - {"term": "H3K27ac", "p": 1e-6}
    Returns: (label, weight)
    """
    if isinstance(item, str):
        return item, 1.0
    if isinstance(item, (int, float)):
        return str(item), 1.0

    if isinstance(item, dict):
        label = (
            item.get("entity")
            or item.get("name")
            or item.get("term")
            or item.get("id")
            or item.get("label")
            or item.get("Entity")
            or item.get("Name")
        )
        # heuristic weight fields
        w = (
            item.get("score")
            or item.get("weight")
            or item.get("w")
            or item.get("evidence")
            or item.get("odds_ratio")
            or item.get("OR")
        )
        try:
            wv = float(w) if w is not None else 1.0
        except Exception:
            wv = 1.0
        return safe_label(label), wv

    return safe_label(item), 1.0


def _iter_pathway_maps(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Tries to locate dicts that look like:
      { "KEGG_...": {...}, "REACTOME_...": {...}, ... }
    including nested at a few common keys.
    """
    if not isinstance(obj, dict):
        return

    # common nest keys
    for k in ["pathways", "Pathways", "data", "results", "overlap", "payload", "entitiesByPathway"]:
        v = obj.get(k)
        if isinstance(v, dict):
            yield v

    # if obj itself looks like a pathway map (many keys, many look like pathway IDs)
    keys = list(obj.keys())
    if len(keys) >= 5:
        hit = 0
        for kk in keys[: min(50, len(keys))]:
            if isinstance(kk, str) and (kk.startswith(("KEGG_", "REACTOME_", "WP_", "HALLMARK_")) or "PATHWAY" in kk):
                hit += 1
        if hit >= 2:
            yield obj


def _find_pathway_block(djson: Dict[str, Any], pathway: str) -> Optional[Tuple[str, Any]]:
    """
    Returns (actual_key, block) for a best-effort match.
    """
    target = norm_key(pathway)

    best_key = None
    best_block = None

    for mp in _iter_pathway_maps(djson):
        if pathway in mp:
            return pathway, mp[pathway]

        norm_to_key = {norm_key(k): k for k in mp.keys() if isinstance(k, str)}
        if target in norm_to_key:
            k0 = norm_to_key[target]
            return k0, mp[k0]

        for nk, orig in norm_to_key.items():
            if target and (target in nk or nk in target):
                best_key, best_block = orig, mp[orig]
                break

        if best_key is not None:
            return best_key, best_block

    return None


def _extract_entities_from_container(container: Any, etype: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    if container is None:
        return out

    # Case: already type-keyed dict, e.g. {"tf":[...], "epigenetic":[...]}
    if isinstance(container, dict):
        if etype in container:
            for it in _as_list(container.get(etype)):
                lbl, w = _pick_entity_and_weight(it)
                out.append((lbl, w))
            return out

        if etype + "s" in container:
            for it in _as_list(container.get(etype + "s")):
                lbl, w = _pick_entity_and_weight(it)
                out.append((lbl, w))
            return out

        # Case: "entities" list with Source field
        ents = container.get("entities") or container.get("Entities")
        if isinstance(ents, list):
            for it in ents:
                if isinstance(it, dict):
                    src = safe_label(it.get("Source") or it.get("source") or it.get("type")).lower()
                    src2 = "tf" if src in ["tf", "transcription_factor", "transcription factors"] else src
                    src2 = "epigenetic" if src2 in ["epi", "epigenetic", "chromatin", "histone"] else src2
                    src2 = "metabolite" if src2 in ["met", "metabolite", "metabolites"] else src2
                    if src2 != etype:
                        continue
                lbl, w = _pick_entity_and_weight(it)
                out.append((lbl, w))
            return out

    # Case: plain list
    if isinstance(container, list):
        for it in container:
            lbl, w = _pick_entity_and_weight(it)
            out.append((lbl, w))
        return out

    # fallback scalar
    lbl, w = _pick_entity_and_weight(container)
    out.append((lbl, w))
    return out


def extract_entities_for_pathway(djson: Dict[str, Any], pathway: str) -> Dict[str, List[str]]:
    """
    Returns:
      {"tf":[...], "epigenetic":[...], "metabolite":[...]}
    """
    found = _find_pathway_block(djson, pathway)
    if not found:
        return {t: [] for t in ENTITY_TYPES}

    _, block = found

    # handle directional blocks: {"UP": {...}, "DOWN": {...}}
    blocks: List[Any] = []
    if isinstance(block, dict) and any(k in block for k in ["UP", "DOWN", "up", "down"]):
        for k in ["UP", "up", "DOWN", "down"]:
            if k in block and isinstance(block[k], (dict, list)):
                blocks.append(block[k])
    else:
        blocks.append(block)

    out: Dict[str, Counter] = {t: Counter() for t in ENTITY_TYPES}

    for b in blocks:
        for t in ENTITY_TYPES:
            pairs = _extract_entities_from_container(b, t)
            for raw_lbl, w in pairs:
                lbl = canonicalize_entity(raw_lbl, t)
                if not lbl or looks_non_human(lbl):
                    continue
                ww = float(w) if w is not None else 1.0
                out[t][lbl] += ww if math.isfinite(ww) else 1.0

    return {t: [k for k, _ in out[t].most_common()] for t in ENTITY_TYPES}


def aggregate_entity_drivers(djson: Dict[str, Any], top_n: int = 10) -> Dict[str, List[str]]:
    """
    Aggregates entity weights across *all* pathways for each type.
    """
    totals: Dict[str, Counter] = {t: Counter() for t in ENTITY_TYPES}

    for mp in _iter_pathway_maps(djson):
        for pw, block in mp.items():
            if not isinstance(pw, str):
                continue
            per_pw = extract_entities_for_pathway({"pathways": {pw: block}}, pw)
            for t in ENTITY_TYPES:
                for ent in per_pw[t]:
                    totals[t][ent] += 1.0

    return {t: [k for k, _ in totals[t].most_common(top_n)] for t in ENTITY_TYPES}


# -----------------------------
# Common entities across diseases
# -----------------------------
def common_entities_across_diseases(
    disease_to_entities: Dict[str, Dict[str, List[str]]],
    etype: str,
    min_support_frac: float = 1.0,
    top_n: int = 10,
) -> List[str]:
    """
    If min_support_frac=1.0 => strict intersection across all diseases.
    If intersection empty, you can call again with e.g. 0.6.
    """
    diseases = list(disease_to_entities.keys())
    if not diseases:
        return []

    support = max(1, int(math.ceil(len(diseases) * min_support_frac)))
    counts = Counter()

    for d in diseases:
        for e in set(disease_to_entities[d].get(etype, []) or []):
            counts[e] += 1

    keep = [e for e, c in counts.items() if c >= support]
    keep.sort(key=lambda x: (-counts[x], x))
    return keep[:top_n]


# -----------------------------
# Top-3 pathways by activity score
# -----------------------------
@dataclass(frozen=True)
class ActivityPick:
    score_col: str
    kind: str  # "differential" or "whole"


def _pick_best_activity_column(dfs: List[DataFrame]) -> Optional[ActivityPick]:
    if pd is None:
        return None

    diff_candidates = [
        "differential_score", "diff_score", "delta_score", "activity_diff",
        "differential", "delta", "z_diff", "t"
    ]
    whole_candidates = [
        "whole_score", "whole", "activity_score", "activity", "score", "z"
    ]

    def score_col_quality(col: str) -> float:
        vals: List[float] = []
        for df in dfs:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    vals.extend(list(s.abs().values))
        if len(vals) < 10:
            return 0.0
        return float(pd.Series(vals).std())

    best: Optional[ActivityPick] = None
    best_q = 0.0

    for col in diff_candidates:
        q = score_col_quality(col)
        if q > best_q:
            best_q = q
            best = ActivityPick(score_col=col, kind="differential")

    if best is None or best_q == 0.0:
        for col in whole_candidates:
            q = score_col_quality(col)
            if q > best_q:
                best_q = q
                best = ActivityPick(score_col=col, kind="whole")

    return best


def select_top3_pathways_by_activity(
    disease_activity_tables: Dict[str, DataFrame],
    n: int = 3,
) -> Tuple[List[str], Optional[ActivityPick]]:
    """
    Returns (top_pathways, activity_pick_used)
    """
    if pd is None:
        return [], None

    dfs = [df for df in disease_activity_tables.values() if isinstance(df, pd.DataFrame) and len(df) > 0]
    if not dfs:
        return [], None

    pick = _pick_best_activity_column(dfs)
    if pick is None:
        return [], None

    rows = []
    for disease, df in disease_activity_tables.items():
        if "Pathway" not in df.columns or pick.score_col not in df.columns:
            continue
        tmp = df[["Pathway", pick.score_col]].copy()
        tmp["disease"] = disease
        tmp["score"] = pd.to_numeric(tmp[pick.score_col], errors="coerce")
        tmp = tmp.dropna(subset=["score"])
        rows.append(tmp[["Pathway", "disease", "score"]])

    if not rows:
        return [], pick

    big = pd.concat(rows, ignore_index=True)

    bad = re.compile(r"(olfact|odor|taste|phototransduction|sensory)", re.IGNORECASE)
    big = big[~big["Pathway"].astype(str).str.contains(bad)]

    agg = (
        big.groupby("Pathway")
        .agg(n_diseases=("disease", "nunique"), mean_abs=("score", lambda s: float(pd.Series(s).abs().mean())))
        .reset_index()
        .sort_values(["n_diseases", "mean_abs"], ascending=[False, False])
    )

    cand = agg[agg["n_diseases"] >= 2]
    if len(cand) < n:
        cand = agg

    top = list(cand["Pathway"].head(n).astype(str).values)
    return top, pick


# -----------------------------
# Public "build artifacts" API
# -----------------------------
def load_jsons_from_dir(json_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Expects per-disease JSON files in json_dir.
    Disease label = file stem.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(json_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out[p.stem] = obj
        except Exception:
            continue
    return out


def build_driver_tables(
    disease_jsons: Dict[str, Dict[str, Any]],
    diseases: List[str],
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns mapping etype -> list of (Section, "A, B, C").
    Includes a 'Shared (common)' row when non-empty.
    """
    disease_entities: Dict[str, Dict[str, List[str]]] = {}
    for d in diseases:
        djson = disease_jsons.get(d)
        if not djson:
            disease_entities[d] = {t: [] for t in ENTITY_TYPES}
            continue
        disease_entities[d] = aggregate_entity_drivers(djson, top_n=200)

    tables: Dict[str, List[Tuple[str, str]]] = {}
    for t in ENTITY_TYPES:
        rows: List[Tuple[str, str]] = []

        shared = common_entities_across_diseases(disease_entities, t, min_support_frac=1.0, top_n=top_n)
        if not shared:
            shared = common_entities_across_diseases(disease_entities, t, min_support_frac=0.6, top_n=top_n)

        if shared:
            rows.append(("Shared (common)", ", ".join(shared)))

        for d in diseases:
            top = disease_entities[d].get(t, [])[:top_n]
            rows.append((d, ", ".join(top) if top else "—"))

        tables[t] = rows

    return tables


def build_case_data(
    disease_jsons: Dict[str, Dict[str, Any]],
    diseases: List[str],
    pathways: List[str],
    top_n: int = 200,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    CASE_DATA format expected by JS:
      CASE_DATA[pathway][disease]["tf"/"epigenetic"/"metabolite"] = [...]
    """
    out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for pw in pathways:
        out[pw] = {}
        for d in diseases:
            djson = disease_jsons.get(d, {})
            per = extract_entities_for_pathway(djson, pw) if djson else {t: [] for t in ENTITY_TYPES}
            out[pw][d] = {t: (per.get(t, [])[:top_n]) for t in ENTITY_TYPES}
    return out
