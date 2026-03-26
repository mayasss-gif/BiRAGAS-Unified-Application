#!/usr/bin/env python3
"""
analysis_tools/query_modules/common_json.py

Robust utilities for:
- locating JSON bundles under OUT_ROOT (strict layout)
- inferring disease name from JSON filename
- parsing pathway_entity_overlap JSONs into a normalized structure

HARD GUARANTEE (for downstream presence/similarity code):
  pathway -> {"ANY": set(...), "UP": set(...), "DOWN": set(...), "ALL": alias of ANY}

Rules:
- If ANY missing but UP/DOWN exist => ANY = UP ∪ DOWN
- If ANY missing but ALL exists => ANY = ALL
- If schema only has 'entities' => ANY = entities
- If payload is list/str/etc => ANY = payload
- ALL is always set(ANY) (alias / duplicate for compatibility)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple


# -----------------------------
# Strict JSON bundle resolution
# -----------------------------

def find_json_bundle_strict(root: str | Path) -> Path:
    """
    Given OUT_ROOT (or a direct json folder), return the json root folder.

    Strict candidates (first match wins):
      - <OUT_ROOT>/results/all_jsons
      - <OUT_ROOT>/GL_enrich/jsons_all_folder
      - <OUT_ROOT>/GC_enrich/jsons_all_folder
      - <root> itself if it contains *.json

    If `root` is a json file, its parent is treated as the root.

    Raises FileNotFoundError if none found or empty.
    """
    base = Path(root).expanduser().resolve()
    if base.is_file() and base.suffix.lower() == ".json":
        base = base.parent

    candidates = [
        base / "results" / "all_jsons",
        base / "GL_enrich" / "jsons_all_folder",
        base / "GC_enrich" / "jsons_all_folder",
        base,
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            if any(c.glob("*.json")):
                return c

    raise FileNotFoundError(
        f"No JSON bundle found under: {base}\n"
        "Expected one of:\n"
        f"  - {base / 'results' / 'all_jsons'}\n"
        f"  - {base / 'GL_enrich' / 'jsons_all_folder'}\n"
        f"  - {base / 'GC_enrich' / 'jsons_all_folder'}\n"
        f"  - or *.json directly in {base}\n"
    )


# -----------------------------
# Name normalization
# -----------------------------

_SUFFIX_PATTERNS: Tuple[str, ...] = (
    r"_pathway_entity_overlap$",
    r"-pathway_entity_overlap$",
    r"pathway_entity_overlap$",
    r"_pathway_overlap$",
    r"_overlap$",
)


def infer_disease_name_from_json_path(p: str | Path) -> str:
    """
    Infer a disease name from a JSON filename robustly.

    Handles:
      - <Disease>_pathway_entity_overlap.json
      - <Disease>.json
      - <Disease>_overlap.json
      - etc.
    """
    stem = Path(p).stem
    name = stem

    for pat in _SUFFIX_PATTERNS:
        name = re.sub(pat, "", name)

    name = name.strip(" _-")
    return name if name else stem


# -----------------------------
# Parsing helpers
# -----------------------------

def _clean_str(x: Any) -> str:
    s = str(x).strip()
    return s


def _as_set(x: Any) -> Set[str]:
    """
    Convert arbitrary payload into a set[str], stripping empties.
    - list/tuple/set => set of items
    - dict => flatten values recursively
    - str => singleton if non-empty
    - None => empty set
    """
    if x is None:
        return set()

    if isinstance(x, set):
        return {_clean_str(v) for v in x if _clean_str(v)}

    if isinstance(x, (list, tuple)):
        return {_clean_str(v) for v in x if _clean_str(v)}

    if isinstance(x, str):
        s = x.strip()
        return {s} if s else set()

    if isinstance(x, Mapping):
        out: Set[str] = set()
        for v in x.values():
            out |= _as_set(v)
        return out

    s = _clean_str(x)
    return {s} if s else set()


def _pick_key(d: Mapping[str, Any], *keys: str) -> Optional[Any]:
    """
    Return the first matching key's value from dict, case-insensitive.
    """
    if not isinstance(d, Mapping):
        return None
    lower = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower:
            return d[lower[kk]]
    return None


def _unwrap_top_level(data: Any) -> Any:
    """
    Unwrap common wrappers so we end up with pathway->payload mapping.
    """
    if not isinstance(data, Mapping):
        return data

    for wrapper_key in ("pathways", "results", "data", "overlap", "payload"):
        if wrapper_key in data and isinstance(data[wrapper_key], Mapping):
            return data[wrapper_key]

    # Some producers store the mapping under a single unknown key alongside metadata.
    # If there's exactly one dict-like value and the rest look like metadata, unwrap it.
    dict_values = [v for v in data.values() if isinstance(v, Mapping)]
    if len(dict_values) == 1 and len(data) <= 3:
        return dict_values[0]

    return data


def _normalize_pathway_payload(payload: Any) -> Dict[str, Set[str]]:
    """
    Convert one pathway's payload into {"ANY","UP","DOWN","ALL"} sets.
    Supports multiple schema variants.
    """
    up_set: Set[str] = set()
    down_set: Set[str] = set()
    any_set: Set[str] = set()

    if isinstance(payload, Mapping):
        up_set = _as_set(_pick_key(payload, "UP", "up", "Up"))
        down_set = _as_set(_pick_key(payload, "DOWN", "down", "Down"))

        # Prefer explicit ANY
        any_set = _as_set(_pick_key(payload, "ANY", "any", "Any"))

        # Fall back to ALL
        if not any_set:
            any_set = _as_set(_pick_key(payload, "ALL", "all", "All"))

        # Fall back to entities / items
        if not any_set:
            any_set = _as_set(_pick_key(payload, "entities", "entity", "entity_list", "items"))

        # If only UP/DOWN exist, define ANY = union
        if not any_set and (up_set or down_set):
            any_set = set(up_set) | set(down_set)

        # If schema is category->entities mapping (no UP/DOWN), flatten into ANY
        if not any_set and not up_set and not down_set:
            any_set = _as_set(payload)

    else:
        # payload is list/str/etc -> treat as direction-free
        any_set = _as_set(payload)

    # HARD guarantee
    any_set = set(any_set)
    up_set = set(up_set)
    down_set = set(down_set)

    # If ANY still empty but UP/DOWN exist, union them
    if not any_set and (up_set or down_set):
        any_set = set(up_set) | set(down_set)

    return {"ANY": any_set, "UP": up_set, "DOWN": down_set, "ALL": set(any_set)}


def parse_overlap_json(json_path: str | Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse a pathway_entity_overlap JSON into normalized structure:

      pathway -> {"ANY": set(...), "UP": set(...), "DOWN": set(...), "ALL": alias of ANY}

    This is intentionally permissive on schema shape:
    - Top-level may be pathway->payload OR wrapped under keys like 'pathways', 'results', 'data'.
    """
    p = Path(json_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"JSON not found: {p}")

    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = p.read_text(encoding="utf-8", errors="replace")

    data = json.loads(raw)
    data = _unwrap_top_level(data)

    if not isinstance(data, Mapping):
        raise ValueError(f"JSON does not contain a pathway mapping after unwrap: {p}")

    out: Dict[str, Dict[str, Set[str]]] = {}
    for pathway, payload in data.items():
        pname = str(pathway).strip()
        if not pname:
            continue
        out[pname] = _normalize_pathway_payload(payload)

    return out


def load_overlap_bundle(json_root: str | Path) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """
    Load all overlap JSONs in json_root into:
        disease -> (pathway -> {"ANY","UP","DOWN","ALL"} sets)
    """
    root = Path(json_root).expanduser().resolve()
    if root.is_file() and root.suffix.lower() == ".json":
        # treat as single-file bundle
        disease = infer_disease_name_from_json_path(root)
        return {disease: parse_overlap_json(root)}

    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json files in: {root}")

    bundle: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
    for f in files:
        disease = infer_disease_name_from_json_path(f)
        bundle[disease] = parse_overlap_json(f)

    return bundle
