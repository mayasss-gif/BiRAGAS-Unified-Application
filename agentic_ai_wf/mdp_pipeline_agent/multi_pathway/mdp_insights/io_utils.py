from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {p}: {e}") from e


# -----------------------------
# Name normalization (disease)
# -----------------------------

_SUFFIX_PATTERNS = (
    r"_pathway_entity_overlap$",
    r"-pathway_entity_overlap$",
    r"pathway_entity_overlap$",
    r"_pathway_overlap$",
    r"_overlap$",
)


def infer_disease_name_from_stem(stem: str) -> str:
    name = str(stem or "").strip()
    if not name:
        return stem
    for pat in _SUFFIX_PATTERNS:
        name = re.sub(pat, "", name)
    name = name.strip(" _-")
    return name if name else stem


def _pick_ci(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    if not isinstance(d, dict):
        return None
    lower = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower:
            return d[lower[kk]]
    return None


# -----------------------------
# JSON schema normalization
# -----------------------------

def _unwrap_common_wrappers(data: Any) -> Any:
    """
    Many producers wrap pathway maps under: pathways/results/data/overlap/payload
    Unwrap one level if found.
    """
    if not isinstance(data, dict):
        return data
    for k in ("pathways", "results", "data", "overlap", "payload"):
        v = data.get(k)
        if isinstance(v, dict) and v:
            return v
    return data


def _looks_like_updown_root(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    keys = {str(k).strip().upper() for k in data.keys()}
    return bool(keys.intersection({"UP", "DOWN"}))


def _merge_directional_roots(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert:
      {"UP": {pathway: payload}, "DOWN": {pathway: payload}, ...}
    into:
      {pathway: {"UP": payload, "DOWN": payload, "ANY": payload?}}
    Also supports top-level ANY/ALL/entities if present.
    """
    out: Dict[str, Any] = {}

    up_map = _pick_ci(data, "UP")
    down_map = _pick_ci(data, "DOWN")

    for direction, block in (("UP", up_map), ("DOWN", down_map)):
        if not isinstance(block, dict):
            continue
        for pathway, payload in block.items():
            pw = str(pathway).strip()
            if not pw:
                continue
            if pw not in out or not isinstance(out.get(pw), dict):
                out[pw] = {}
            out[pw][direction] = payload

    any_block = _pick_ci(data, "ANY", "ALL", "entities", "entity_list", "items")
    if isinstance(any_block, dict) and any_block:
        # This could be either:
        #  (A) pathway -> {etype:[...]}, or
        #  (B) etype -> [...], in which case it's not attachable per pathway.
        # We only attach if keys look like pathways (heuristic: values are dict/list).
        attachable = True
        for _, v in list(any_block.items())[:10]:
            if not isinstance(v, (dict, list, tuple)):
                attachable = False
                break
        if attachable:
            for pathway, payload in any_block.items():
                pw = str(pathway).strip()
                if not pw:
                    continue
                if pw not in out or not isinstance(out.get(pw), dict):
                    out[pw] = {}
                # store as ANY so downstream can parse it
                out[pw]["ANY"] = payload

    return out


def normalize_json_to_pathway_map(data: Any) -> Dict[str, Any]:
    """
    Guarantee that the return is a dict mapping:
        pathway -> payload
    by unwrapping common wrappers and normalizing UP/DOWN roots.
    """
    data = _unwrap_common_wrappers(data)

    # sometimes wrappers are nested twice
    data = _unwrap_common_wrappers(data)

    if not isinstance(data, dict) or not data:
        return {}

    # If top-level is UP/DOWN, merge into pathway->{"UP","DOWN",...}
    if _looks_like_updown_root(data):
        merged = _merge_directional_roots(data)
        return merged if merged else {}

    # Otherwise, assume it's already pathway->payload
    return data


def load_jsons(json_root: Path) -> Dict[str, dict]:
    """
    Load *.json under `json_root`.

    Key = normalized disease name (suffixes removed).
    Value = normalized pathway mapping dict.

    Robust to empty/malformed files; logs & skips them.
    """
    out: Dict[str, dict] = {}

    if not json_root.exists() or not json_root.is_dir():
        raise FileNotFoundError(f"json_root not found or not a directory: {json_root}")

    files = sorted(json_root.glob("*.json"))
    if not files:
        raise RuntimeError(f"No *.json files in {json_root}")

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            norm = normalize_json_to_pathway_map(data)
            if not isinstance(norm, dict) or not norm:
                logging.warning(f"Skipping {fp.name}: empty/unrecognized JSON schema after normalization")
                continue

            disease = infer_disease_name_from_stem(fp.stem)
            out[disease] = norm

        except json.JSONDecodeError as je:
            logging.error(f"JSON parse error in {fp.name}: {je}")
        except Exception as e:
            logging.error(f"Error reading {fp.name}: {e}")

    if not out:
        raise RuntimeError("No valid JSON files found after parsing/normalization")

    return out


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        logging.error(f"Failed writing CSV {path}: {e}")


def safe_write_json(obj: Any, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed writing JSON {path}: {e}")
