#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ayass Bioscience — Cross-Disease Report (ENTITIES)
==================================================

File: report_entities.py  (CODE 2 / 3)

This module is responsible for:
- Loading disease JSONs (prefer OUT_ROOT/results/all_jsons or OUT_ROOT/jsons_all*)
- Falling back to overlap JSON inside each disease folder (pathway ↔ entity links)
- Final fallback for *disease-level* entity lists from ALL_COMBINED.csv (not pathway-specific)
- Normalizing many possible JSON schemas into ONE internal representation
- Producing:
    1) disease_to_json (normalized)
    2) json_sources (per-disease provenance note)
    3) fallback_entities (per-disease TF/epi/met lists from ALL_COMBINED.csv)
    4) entity rankings (across all pathways) per disease
    5) CASE_DATA for top pathways (per pathway → per disease → per entity-type list)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


# =========================
# === Filters
# =========================

NON_HUMAN_PAT = re.compile(r"\b(mouse|mice|rat|mm9|mm10|murine|c2c12)\b", flags=re.IGNORECASE)


def is_human_entity(name: str) -> bool:
    if not name:
        return False
    return NON_HUMAN_PAT.search(str(name)) is None


def clean_name(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return np.nan


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        suf = path.suffix.lower()
        if suf in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suf == ".csv":
            return pd.read_csv(path)
        if suf == ".xlsx":
            return pd.read_excel(path)
    except Exception:
        return None
    return None


# =========================
# === JSON discovery
# =========================

def _candidate_json_roots(out_root: Path) -> List[Path]:
    return [
        out_root / "results" / "all_jsons",         # canonical counts/degs
        out_root / "jsons_all",                     # legacy
        out_root / "jsons_all_folder",              # legacy genelist/gc copy target
        out_root / "results" / "jsons_all_folder",  # sometimes used
        out_root / "GL_enrich" / "jsons_all_folder",
        out_root / "GC_enrich" / "jsons_all_folder",
    ]


def _normalize_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\.json$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _find_disease_json_file(out_root: Path, disease: str) -> Optional[Path]:
    dkey = _normalize_key(disease)
    for root in _candidate_json_roots(out_root):
        if not root.exists() or not root.is_dir():
            continue

        exact = root / f"{disease}.json"
        if exact.exists():
            return exact

        exact2 = root / f"{dkey}.json"
        if exact2.exists():
            return exact2

        # shallow scan
        try:
            for p in root.glob("*.json"):
                if _normalize_key(p.stem) == dkey:
                    return p
        except Exception:
            continue
    return None


def _find_overlap_json(disease_dir: Path) -> Optional[Path]:
    """
    Fallback: disease folder overlap JSON that *does* encode pathway ↔ entity links.
    We accept several common filenames.
    """
    candidates = [
        disease_dir / "overlap" / "pathway_entity_overlap.json",
        disease_dir / "overlap" / "pathway_entity_overlap_directional.json",
        disease_dir / "overlap" / "pathway_entity_overlap_UPDOWN.json",
        disease_dir / "results" / "overlap" / "pathway_entity_overlap.json",
        disease_dir / "results" / "overlap" / "pathway_entity_overlap_directional.json",
        disease_dir / "pathway_entity_overlap.json",
        disease_dir / "pathway_entity_overlap_directional.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # last resort: any file containing pathway_entity_overlap in name within 2 levels
    try:
        hits = list(disease_dir.glob("**/*pathway_entity_overlap*.json"))
        for p in hits[:5]:
            if p.exists():
                return p
    except Exception:
        pass
    return None


def _find_all_combined_csv(disease_dir: Path) -> Optional[Path]:
    """
    Last fallback: ALL_COMBINED.csv (disease-level TF/epi/met entities; not pathway-specific).
    """
    candidates = [
        disease_dir / "ALL_COMBINED.csv",
        disease_dir / "results" / "ALL_COMBINED.csv",
        disease_dir / "GC_out" / "ALL_COMBINED.csv",
        disease_dir / "outputs" / "ALL_COMBINED.csv",
        disease_dir / "enrichr" / "ALL_COMBINED.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    try:
        hits = list(disease_dir.glob("**/ALL_COMBINED.csv"))
        for p in hits[:5]:
            if p.exists():
                return p
    except Exception:
        pass
    return None


# =========================
# === JSON normalization
# =========================

_ENTITY_KEYS = {"tf", "tfs", "transcription_factor", "transcription_factors",
                "epigenetic", "epigenetics", "histone", "chromatin",
                "metabolite", "metabolites"}

_CANON_TYPE = {
    "tf": "tf",
    "tfs": "tf",
    "transcription_factor": "tf",
    "transcription_factors": "tf",
    "epigenetic": "epigenetic",
    "epigenetics": "epigenetic",
    "histone": "epigenetic",
    "chromatin": "epigenetic",
    "metabolite": "metabolite",
    "metabolites": "metabolite",
}

_DIR_KEYS = {"ANY", "ALL", "UP", "DOWN"}


def _looks_like_pathway_name(k: str) -> bool:
    s = str(k or "")
    if len(s) < 6:
        return False
    # common pathway prefixes in your outputs
    return bool(re.match(r"^(KEGG|REACTOME|WP|HALLMARK|BIOCARTA|PID|GO_|GOBP|GOMF|GOCC|MSIGDB)_", s, flags=re.IGNORECASE))


def _to_entity_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Converts one of:
      - ["STAT1", "IRF1"]
      - [{"name":"STAT1","score":2.1}, ...]
      - {"STAT1": 2.1, "IRF1": 1.7}
    into a list of dicts: [{"name":..., "score":...}, ...] with best-effort scoring.
    """
    out: List[Dict[str, Any]] = []

    if obj is None:
        return out

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, str):
                nm = clean_name(it)
                if nm:
                    out.append({"name": nm, "score": np.nan})
            elif isinstance(it, (int, float)):
                nm = clean_name(it)
                if nm:
                    out.append({"name": nm, "score": np.nan})
            elif isinstance(it, dict):
                nm = clean_name(it.get("name") or it.get("entity") or it.get("id") or it.get("symbol") or it.get("term"))
                sc = safe_float(it.get("score") or it.get("weight") or it.get("value") or it.get("strength") or it.get("odds_ratio"))
                if nm:
                    out.append({"name": nm, "score": sc})
        return out

    if isinstance(obj, dict):
        # dict of entity -> score OR nested with {"items":[...]} etc.
        if "items" in obj and isinstance(obj["items"], list):
            return _to_entity_list(obj["items"])
        if "entities" in obj and isinstance(obj["entities"], (dict, list)):
            return _to_entity_list(obj["entities"])

        # likely mapping entity->score
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and k in {"items", "entities"}:
                continue
            nm = clean_name(k)
            sc = safe_float(v)
            if nm:
                out.append({"name": nm, "score": sc})
        return out

    # unknown scalar
    nm = clean_name(obj)
    if nm:
        out.append({"name": nm, "score": np.nan})
    return out


def _sort_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(d: Dict[str, Any]):
        sc = d.get("score", np.nan)
        sc = safe_float(sc)
        # if score missing, push down but keep stable-ish
        return (0 if np.isfinite(sc) else 1, -abs(sc) if np.isfinite(sc) else 0.0, d.get("name", ""))

    items2 = [x for x in items if x.get("name")]
    items2.sort(key=key_fn)
    # de-dup by name preserving order
    seen = set()
    out = []
    for it in items2:
        nm = clean_name(it.get("name"))
        if not nm:
            continue
        if nm in seen:
            continue
        seen.add(nm)
        out.append({"name": nm, "score": safe_float(it.get("score"))})
    return out


def _extract_entities_block(node: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Given a node that may include entities in many shapes,
    return dict: {"tf":[...], "epigenetic":[...], "metabolite":[...]}
    """
    out: Dict[str, List[Dict[str, Any]]] = {"tf": [], "epigenetic": [], "metabolite": []}
    if node is None:
        return out

    # Common: node["entities"] = {tf:..., epigenetic:..., metabolite:...}
    if isinstance(node, dict) and "entities" in node and isinstance(node["entities"], (dict, list)):
        node = node["entities"]

    # node directly keyed by types
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).strip().lower()
            if lk in _ENTITY_KEYS:
                typ = _CANON_TYPE.get(lk, lk)
                out[typ] = _sort_entities(_to_entity_list(v))
        # also: nested "drivers" / "entity_drivers"
        for dk in ["drivers", "entity_drivers", "layers", "layer_entities"]:
            if dk in node and isinstance(node[dk], (dict, list)):
                sub = _extract_entities_block(node[dk])
                for t in out:
                    if sub.get(t):
                        out[t] = _sort_entities(out[t] + sub[t])

    # node is list containing dicts with type fields
    if isinstance(node, list):
        for it in node:
            if not isinstance(it, dict):
                continue
            typ = clean_name(it.get("type") or it.get("layer") or it.get("source")).lower()
            if typ in _ENTITY_KEYS:
                canon = _CANON_TYPE.get(typ, typ)
                out[canon] = _sort_entities(out[canon] + _to_entity_list(it.get("entities") or it.get("items") or it.get("values") or it))

    # human-only filter for TF/EPI (metabolites usually fine but keep same filter)
    for t in ["tf", "epigenetic", "metabolite"]:
        filt = []
        for it in out[t]:
            nm = clean_name(it.get("name"))
            if not nm:
                continue
            if is_human_entity(nm):
                filt.append(it)
        out[t] = filt

    return out


def normalize_any_json_to_internal(obj: Any) -> Dict[str, Any]:
    """
    Internal normalized schema (per disease):
      {
        "pathways": {
          "<PATHWAY>": {
            "ANY": {"tf":[{name,score}...], "epigenetic":[...], "metabolite":[...]},
            "UP":  {...},
            "DOWN":{...}
          }, ...
        }
      }

    This function tries hard to map many JSON layouts into this.
    """
    norm: Dict[str, Any] = {"pathways": {}}
    if obj is None:
        return norm

    # Identify pathway map
    pathway_map = None

    if isinstance(obj, dict):
        if "pathways" in obj and isinstance(obj["pathways"], dict):
            pathway_map = obj["pathways"]
        elif "data" in obj and isinstance(obj["data"], dict) and "pathways" in obj["data"]:
            if isinstance(obj["data"]["pathways"], dict):
                pathway_map = obj["data"]["pathways"]
        elif "results" in obj and isinstance(obj["results"], dict) and "pathways" in obj["results"]:
            if isinstance(obj["results"]["pathways"], dict):
                pathway_map = obj["results"]["pathways"]
        else:
            # maybe keys are pathway names directly
            # only accept if many look like pathway names
            keys = list(obj.keys())
            looks = sum(1 for k in keys if _looks_like_pathway_name(str(k)))
            if looks >= max(3, int(0.2 * max(1, len(keys)))):
                pathway_map = obj

    if pathway_map is None:
        return norm

    for pw, node in pathway_map.items():
        pwn = clean_name(pw)
        if not pwn:
            continue

        # Detect per-direction children
        dirs: Dict[str, Any] = {}
        if isinstance(node, dict):
            # common direction keys exist
            found_dir = False
            for dk, dv in node.items():
                dk_u = str(dk).strip().upper()
                if dk_u in _DIR_KEYS and isinstance(dv, (dict, list)):
                    dirs[dk_u] = dv
                    found_dir = True
            if not found_dir:
                dirs["ANY"] = node
        else:
            dirs["ANY"] = node

        # Build canonical directions
        entry = {"ANY": {"tf": [], "epigenetic": [], "metabolite": []},
                 "UP": {"tf": [], "epigenetic": [], "metabolite": []},
                 "DOWN": {"tf": [], "epigenetic": [], "metabolite": []}}

        for dk_u, dv in dirs.items():
            dk_u = "ANY" if dk_u in {"ALL"} else dk_u
            if dk_u not in entry:
                dk_u = "ANY"
            ent_block = _extract_entities_block(dv)
            for typ in ["tf", "epigenetic", "metabolite"]:
                if ent_block.get(typ):
                    entry[dk_u][typ] = _sort_entities(entry[dk_u][typ] + ent_block[typ])

        # If ANY empty but UP/DOWN exist, merge them into ANY (for report)
        any_empty = all(len(entry["ANY"][t]) == 0 for t in ["tf", "epigenetic", "metabolite"])
        updown_has = any(len(entry["UP"][t]) > 0 or len(entry["DOWN"][t]) > 0 for t in ["tf", "epigenetic", "metabolite"])
        if any_empty and updown_has:
            for typ in ["tf", "epigenetic", "metabolite"]:
                merged = entry["UP"][typ] + entry["DOWN"][typ]
                entry["ANY"][typ] = _sort_entities(merged)

        norm["pathways"][pwn] = entry

    return norm


def _has_any_entities(norm: Dict[str, Any]) -> bool:
    pmap = (norm or {}).get("pathways", {}) or {}
    for _pw, dnode in pmap.items():
        for dk in ["ANY", "UP", "DOWN"]:
            layer = (dnode or {}).get(dk, {}) or {}
            if any((layer.get(t) or []) for t in ["tf", "epigenetic", "metabolite"]):
                return True
    return False


def merge_norms(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill missing entities in primary from secondary (pathway-by-pathway, direction-by-direction).
    Primary wins if already has entities.
    """
    out = {"pathways": {}}
    p1 = (primary or {}).get("pathways", {}) or {}
    p2 = (secondary or {}).get("pathways", {}) or {}

    all_pw = sorted(set(p1.keys()) | set(p2.keys()))
    for pw in all_pw:
        e1 = p1.get(pw)
        e2 = p2.get(pw)
        if e1 is None and e2 is None:
            continue
        entry = {"ANY": {"tf": [], "epigenetic": [], "metabolite": []},
                 "UP": {"tf": [], "epigenetic": [], "metabolite": []},
                 "DOWN": {"tf": [], "epigenetic": [], "metabolite": []}}

        for dk in ["ANY", "UP", "DOWN"]:
            for typ in ["tf", "epigenetic", "metabolite"]:
                a = (((e1 or {}).get(dk, {}) or {}).get(typ) or []) if e1 else []
                b = (((e2 or {}).get(dk, {}) or {}).get(typ) or []) if e2 else []
                if a:
                    entry[dk][typ] = _sort_entities(a)
                elif b:
                    entry[dk][typ] = _sort_entities(b)

        # ensure ANY filled if empty and UP/DOWN exist
        any_empty = all(len(entry["ANY"][t]) == 0 for t in ["tf", "epigenetic", "metabolite"])
        updown_has = any(len(entry["UP"][t]) > 0 or len(entry["DOWN"][t]) > 0 for t in ["tf", "epigenetic", "metabolite"])
        if any_empty and updown_has:
            for typ in ["tf", "epigenetic", "metabolite"]:
                entry["ANY"][typ] = _sort_entities(entry["UP"][typ] + entry["DOWN"][typ])

        out["pathways"][pw] = entry

    return out


# =========================
# === ALL_COMBINED fallback (disease-level only)
# =========================

def _extract_fallback_entities_from_all_combined(csv_path: Path, top_n: int = 50) -> Dict[str, List[str]]:
    """
    Returns dict: {"tf":[...], "epigenetic":[...], "metabolite":[...]}
    This is NOT pathway-specific; used only if JSONs are missing.
    """
    df = _read_table(csv_path)
    out = {"tf": [], "epigenetic": [], "metabolite": []}
    if df is None or df.empty:
        return out

    cols = {str(c).lower(): str(c) for c in df.columns}
    src_col = cols.get("source")
    term_col = cols.get("term") or cols.get("name") or cols.get("entity") or cols.get("label") or cols.get("pathway")

    if not src_col or not term_col:
        # try best effort: first two cols
        if df.shape[1] >= 2:
            src_col = df.columns[0]
            term_col = df.columns[1]
        else:
            return out

    for _, r in df.iterrows():
        src = clean_name(r.get(src_col)).lower()
        term = clean_name(r.get(term_col))
        if not term or not is_human_entity(term):
            continue
        if src in {"tf", "tfs", "transcription_factor"}:
            out["tf"].append(term)
        elif src in {"epigenetic", "epigenetics", "histone", "chromatin"}:
            out["epigenetic"].append(term)
        elif src in {"metabolite", "metabolites"}:
            out["metabolite"].append(term)

    # de-dup preserve order, cap
    for k in out:
        seen = set()
        uniq = []
        for x in out[k]:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
            if len(uniq) >= top_n:
                break
        out[k] = uniq

    return out


# =========================
# === Public API (used by core_report.py)
# =========================

def load_entities(
    out_root: Path,
    diseases: List[str],
    disease_dirs: Dict[str, Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, List[str]]]]:
    """
    Returns:
      disease_to_json: normalized per-disease JSON (pathways->directions->layers)
      json_sources: per-disease provenance string (what we loaded + fallbacks used)
      fallback_entities: per-disease disease-level entities from ALL_COMBINED.csv (last resort)
    """
    disease_to_json: Dict[str, Dict[str, Any]] = {}
    json_sources: Dict[str, str] = {}
    fallback_entities: Dict[str, Dict[str, List[str]]] = {}

    for d in diseases:
        ddir = disease_dirs.get(d, None)
        src_bits: List[str] = []

        # 1) jsons_all (preferred)
        p_main = _find_disease_json_file(out_root, d)
        obj_main = _read_json(p_main) if p_main else None
        norm_main = normalize_any_json_to_internal(obj_main) if obj_main is not None else {"pathways": {}}
        if p_main:
            src_bits.append(f"jsons_all={p_main}")

        # 2) overlap json (fallback + also used to fill missing pathway entities)
        p_ov = _find_overlap_json(ddir) if ddir else None
        obj_ov = _read_json(p_ov) if p_ov else None
        norm_ov = normalize_any_json_to_internal(obj_ov) if obj_ov is not None else {"pathways": {}}
        if p_ov:
            src_bits.append(f"overlap={p_ov}")

        # merge to ensure pathway-level entities exist for case studies
        if _has_any_entities(norm_main):
            merged = merge_norms(norm_main, norm_ov)
        else:
            # if jsons_all has no entities, prefer overlap (it usually does)
            merged = merge_norms(norm_ov, norm_main)

        disease_to_json[d] = merged

        # 3) ALL_COMBINED fallback (disease-level only)
        fb = {"tf": [], "epigenetic": [], "metabolite": []}
        p_ac = _find_all_combined_csv(ddir) if ddir else None
        if p_ac:
            fb = _extract_fallback_entities_from_all_combined(p_ac, top_n=80)
            src_bits.append(f"ALL_COMBINED={p_ac}")
        fallback_entities[d] = fb

        if not src_bits:
            src_bits = ["none_found"]
        json_sources[d] = " | ".join(src_bits)

    return disease_to_json, json_sources, fallback_entities


def compute_entity_rankings_from_json(
    disease_to_json: Dict[str, Dict[str, Any]],
    diseases: List[str],
    entity_type: str,
    direction: str = "ANY",
) -> Dict[str, pd.Series]:
    """
    Aggregate across all pathways for each disease.
    Scoring:
      - if entity has numeric 'score' → sum(abs(score)) across occurrences
      - else → count occurrences
    Returns per disease: pandas Series indexed by entity name (descending).
    """
    typ = str(entity_type).strip().lower()
    if typ not in {"tf", "epigenetic", "metabolite"}:
        raise ValueError(f"entity_type must be one of tf/epigenetic/metabolite; got {entity_type}")

    dir_key = str(direction or "ANY").strip().upper()
    if dir_key == "ALL":
        dir_key = "ANY"

    out: Dict[str, pd.Series] = {}
    for d in diseases:
        norm = disease_to_json.get(d, {}) or {}
        pmap = norm.get("pathways", {}) or {}

        agg: Dict[str, float] = {}
        cnt: Dict[str, int] = {}

        for _pw, dnode in pmap.items():
            node = (dnode or {}).get(dir_key) or None
            if node is None and dir_key == "ANY":
                # merge UP+DOWN if ANY absent
                node_up = (dnode or {}).get("UP", {}) or {}
                node_dn = (dnode or {}).get("DOWN", {}) or {}
                items = (node_up.get(typ) or []) + (node_dn.get(typ) or [])
            else:
                items = (node or {}).get(typ, []) if isinstance(node, dict) else []

            for it in items or []:
                nm = clean_name(it.get("name"))
                if not nm or not is_human_entity(nm):
                    continue
                sc = safe_float(it.get("score"))
                cnt[nm] = cnt.get(nm, 0) + 1
                if np.isfinite(sc):
                    agg[nm] = agg.get(nm, 0.0) + abs(sc)
                else:
                    agg[nm] = agg.get(nm, 0.0) + 1.0

        if not agg:
            out[d] = pd.Series(dtype=float)
            continue

        s = pd.Series(agg, dtype=float)
        # tie-break: higher count first, then score
        if cnt:
            s = s.sort_values(ascending=False)
            # stable-ish by count then score
            df = pd.DataFrame({"score": s, "count": pd.Series(cnt, dtype=float)})
            df = df.sort_values(["count", "score"], ascending=[False, False])
            s = df["score"]
        else:
            s = s.sort_values(ascending=False)

        out[d] = s

    return out


def shared_entities_summary(
    disease_to_series: Dict[str, pd.Series],
    diseases: List[str],
    min_diseases: int = 2,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Returns a table of entities shared across >= min_diseases diseases.
    Score = sum(scores) across diseases.
    """
    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}

    for d in diseases:
        s = disease_to_series.get(d)
        if s is None or s.empty:
            continue
        for nm, sc in s.items():
            nm = clean_name(nm)
            if not nm:
                continue
            counts[nm] = counts.get(nm, 0) + 1
            sums[nm] = sums.get(nm, 0.0) + float(sc)

    rows = []
    for nm, c in counts.items():
        if c >= int(min_diseases):
            rows.append({"Entity": nm, "#Diseases": int(c), "Score": float(sums.get(nm, 0.0))})

    if not rows:
        return pd.DataFrame(columns=["Entity", "#Diseases", "Score"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["#Diseases", "Score"], ascending=[False, False]).head(int(top_n))
    df["Score"] = df["Score"].map(lambda x: f"{x:.2f}")
    return df


def extract_entities_for_pathway(
    disease_norm: Dict[str, Any],
    pathway: str,
    direction: str = "ANY",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns dict: {"tf":[{name,score}...], "epigenetic":[...], "metabolite":[...]}
    """
    pmap = (disease_norm or {}).get("pathways", {}) or {}
    pw = clean_name(pathway)
    node = pmap.get(pw) or pmap.get(pathway) or None
    if node is None:
        # try case-insensitive match
        low = {str(k).lower(): k for k in pmap.keys()}
        key = low.get(str(pw).lower())
        if key:
            node = pmap.get(key)
    if node is None:
        return {"tf": [], "epigenetic": [], "metabolite": []}

    dk = str(direction or "ANY").strip().upper()
    if dk == "ALL":
        dk = "ANY"
    layer = (node or {}).get(dk, {}) or {}
    if dk == "ANY" and not any(layer.get(t) for t in ["tf", "epigenetic", "metabolite"]):
        # merge UP+DOWN
        up = (node or {}).get("UP", {}) or {}
        dn = (node or {}).get("DOWN", {}) or {}
        merged = {}
        for t in ["tf", "epigenetic", "metabolite"]:
            merged[t] = _sort_entities((up.get(t) or []) + (dn.get(t) or []))
        return merged

    return {
        "tf": _sort_entities(layer.get("tf") or []),
        "epigenetic": _sort_entities(layer.get("epigenetic") or []),
        "metabolite": _sort_entities(layer.get("metabolite") or []),
    }


def build_case_data_for_pathways(
    diseases: List[str],
    disease_to_json: Dict[str, Dict[str, Any]],
    pathways: List[str],
    top_k_per_type: int = 25,
) -> Dict[str, Any]:
    """
    CASE_DATA format (used by report_ui.js):
      {
        "<PATHWAY>": {
          "<DiseaseA>": {
             "tf": [{"name":"STAT1","score":2.1}, ...],
             "metabolite": [...],
             "epigenetic": [...]
          },
          "<DiseaseB>": {...}
        }, ...
      }

    This is pathway-specific and should NOT be empty if overlap JSON exists.
    """
    case_data: Dict[str, Any] = {}
    top_k = int(top_k_per_type)

    for pw in pathways:
        pw_clean = clean_name(pw)
        if not pw_clean:
            continue

        entry: Dict[str, Any] = {}
        for d in diseases:
            norm = disease_to_json.get(d, {}) or {}
            layers = extract_entities_for_pathway(norm, pw_clean, direction="ANY")

            # cap lists
            for t in ["tf", "metabolite", "epigenetic"]:
                items = layers.get(t) or []
                # ensure dict shape
                items2 = [{"name": clean_name(x.get("name")), "score": safe_float(x.get("score"))} for x in items if clean_name(x.get("name"))]
                items2 = _sort_entities(items2)[:top_k]
                layers[t] = items2

            entry[d] = layers

        case_data[pw_clean] = entry

    return case_data
