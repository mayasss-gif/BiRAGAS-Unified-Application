
# report/ipaa_report_data.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------
# Basic IO helpers
# ---------------------------
def is_file(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


def is_dir(p: Path) -> bool:
    try:
        return p.exists() and p.is_dir()
    except Exception:
        return False


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(read_text(p))
    except Exception:
        return None


def read_table_auto(p: Path) -> pd.DataFrame:
    # TSV then CSV fallback
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.read_csv(p, sep=",")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(s, encoding="utf-8")


# ---------------------------
# Manifest deep search
# ---------------------------
def deep_find_first(obj: Any, keys: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            out = deep_find_first(v, keys)
            if out is not None:
                return out
    if isinstance(obj, list):
        for v in obj:
            out = deep_find_first(v, keys)
            if out is not None:
                return out
    return None


# ---------------------------
# Column detection
# ---------------------------
def pick_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return str(c)
    return None


@dataclass
class PathwayCols:
    pathway: str
    fdr: Optional[str]
    p: Optional[str]
    t: Optional[str]
    delta: Optional[str]
    main_class: Optional[str]
    sub_class: Optional[str]


def detect_pathway_cols(df: pd.DataFrame) -> PathwayCols:
    cols = [str(c) for c in df.columns]

    pathway = pick_col(cols, [r"^pathway$", r"^term$", r"pathway", r"term"]) or cols[0]
    fdr = pick_col(cols, [r"^fdr$", r"padj", r"adj", r"q[\s_]*value", r"qval"])
    p = pick_col(cols, [r"^p$", r"p[\s_]*value", r"^pval$"])
    t = pick_col(cols, [r"^t$", r"t[\s_]*stat", r"t_like", r"stat", r"score"])
    delta = pick_col(cols, [r"delta", r"delta[_\s]*activity", r"effect", r"diff"])

    main_class = pick_col(cols, [r"^main[_\s]*class$", r"mainclass"])
    sub_class = pick_col(cols, [r"^sub[_\s]*class$", r"subclass"])

    return PathwayCols(
        pathway=pathway,
        fdr=fdr,
        p=p,
        t=t,
        delta=delta,
        main_class=main_class,
        sub_class=sub_class,
    )


def coerce_numeric(df: pd.DataFrame, col: Optional[str]) -> None:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------
# Cohort discovery + layout
# ---------------------------
def resolve_cohort_dir(out_root: Path, cohort: str) -> Optional[Path]:
    primary = out_root / cohort
    legacy = out_root / "cohorts" / cohort
    if is_dir(primary):
        return primary
    if is_dir(legacy):
        return legacy
    return None


def discover_cohorts(out_root: Path) -> List[str]:
    """
    Prefer PIPELINE_MANIFEST.json, else discover dirs.
    """
    out_root = out_root.resolve()
    manifest = read_json(out_root / "PIPELINE_MANIFEST.json") or {}
    names = deep_find_first(manifest, ["cohorts", "cohort_runs", "cohort_names"])

    cohorts: List[str] = []

    if isinstance(names, list) and names:
        if all(isinstance(x, str) for x in names):
            cohorts = [str(x) for x in names]
        elif all(isinstance(x, dict) for x in names):
            for d in names:
                if "name" in d:
                    cohorts.append(str(d["name"]))
    elif isinstance(names, dict):
        cohorts = [str(k) for k in names.keys()]

    if not cohorts:
        skip = {
            "compare", "report", "baseline_consensus", "jsons_all", "cohorts",
            ".git", "__pycache__", "data"
        }
        for p in out_root.iterdir():
            if p.is_dir() and p.name not in skip:
                cohorts.append(p.name)

    # keep only resolvable
    good = []
    for c in sorted(set(cohorts)):
        if resolve_cohort_dir(out_root, c) is not None:
            good.append(c)
    return good


# ---------------------------
# Pathway table selection
# ---------------------------
def pick_pathway_table(cohort_dir: Path) -> Tuple[Optional[Path], str]:
    candidates = [
        ("filtered+classified (final backbone)", "pathway_stats_with_baseline_filtered_classified.tsv"),
        ("filtered (final backbone)",            "pathway_stats_with_baseline_filtered.tsv"),
        ("baseline+classified",                  "pathway_stats_with_baseline_classified.tsv"),
        ("baseline fallback",                    "pathway_stats_with_baseline.tsv"),
        ("filtered fallback",                    "pathway_stats_filtered.tsv"),
        ("raw fallback",                         "pathway_stats.tsv"),
    ]
    for label, fname in candidates:
        p = cohort_dir / fname
        if is_file(p):
            return p, label
    return None, "missing"


# ---------------------------
# Overlap canonicalization
# Canonical format:
# canon[pathway]["ALL"][category][entity] = hit_count
# canon[pathway][direction][category][entity] = hit_count
# where direction in {"UP","DOWN","NA"} plus "ALL"
# ---------------------------
DIR_KEYS = {"UP", "DOWN", "1", "-1", "+1"}


def _norm_dir(v: Any) -> str:
    if isinstance(v, str):
        s = v.strip().upper()
        if s in {"UP", "DOWN"}:
            return s
        if s in {"1", "+1"}:
            return "UP"
        if s == "-1":
            return "DOWN"
    if isinstance(v, (int, float)):
        return "UP" if v >= 0 else "DOWN"
    return "NA"


def _norm_cat(cat: Any) -> str:
    s = str(cat).strip().lower()
    # normalize common aliases
    if s in {"tf", "tfs", "transcription_factor", "transcription_factors"}:
        return "tf"
    if s in {"epi", "epigenetic", "chromatin", "histone", "epigenetics"}:
        return "epigenetic"
    if s in {"met", "metabolite", "metabolites"}:
        return "metabolite"
    return s


def _hit_count_from_info(info: Any) -> int:
    """
    Try to infer evidence strength (#hits) for an entity.
    """
    if isinstance(info, dict):
        for k in ["hits", "Hits", "hit_list", "genes", "hit_genes", "leading_edge"]:
            v = info.get(k)
            if isinstance(v, list):
                return len(v)
        # if dict exists but no list, count as 1 evidence
        return 1
    if isinstance(info, list):
        return len(info)
    # scalar/unknown
    return 1


def _ingest_entity_dict(dst: Dict[str, int], ent_dict: Any) -> None:
    """
    ent_dict expected: {entity: {hits:[...]}} or {entity: ...} or list of records
    """
    if isinstance(ent_dict, dict):
        for ent, info in ent_dict.items():
            e = str(ent).strip()
            if not e:
                continue
            dst[e] = dst.get(e, 0) + max(0, _hit_count_from_info(info))
        return

    if isinstance(ent_dict, list):
        # possible list of dicts with keys like entity/name + hits
        for rec in ent_dict:
            if not isinstance(rec, dict):
                continue
            ent = rec.get("entity") or rec.get("name") or rec.get("Entity")
            if not ent:
                continue
            e = str(ent).strip()
            dst[e] = dst.get(e, 0) + max(0, _hit_count_from_info(rec))
        return


def canonicalize_overlap_json(js: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Handles schema variants:
    A) pathways[path] = {"direction":"UP","entities":{cat:{entity:{hits:[...]}}}}
    B) pathways[path] = {"UP":{cat:{entity:{hits}}}, "DOWN":{...}}
    C) pathways[path] = {cat:{entity:{hits}}} (directionless)
    """
    pathways = js.get("pathways") or js.get("Pathways") or js.get("data") or js.get("DATA") or {}
    canon: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}

    if not isinstance(pathways, dict):
        return canon

    for pth, block in pathways.items():
        pname = str(pth).strip()
        if not pname:
            continue

        if pname not in canon:
            canon[pname] = {"ALL": {}}

        if not isinstance(block, dict):
            continue

        # Case A: explicit "entities"
        if "entities" in block and isinstance(block["entities"], dict):
            d = _norm_dir(block.get("direction") or block.get("dir") or block.get("sign"))
            ent_root = block["entities"]
            _ingest_entities_for_direction(canon[pname], d, ent_root)
            continue

        # Case B: direction keys at top
        dir_like = [k for k in block.keys() if str(k).strip().upper() in DIR_KEYS]
        if dir_like:
            for dk in dir_like:
                d = _norm_dir(dk)
                sub = block.get(dk)
                if isinstance(sub, dict) and "entities" in sub and isinstance(sub["entities"], dict):
                    _ingest_entities_for_direction(canon[pname], d, sub["entities"])
                else:
                    _ingest_entities_for_direction(canon[pname], d, sub)
            continue

        # Case C: treat as directionless category dict
        _ingest_entities_for_direction(canon[pname], "NA", block)

    return canon


def _ensure_dir(canon_path: Dict[str, Dict[str, Dict[str, int]]], direction: str) -> None:
    if direction not in canon_path:
        canon_path[direction] = {}
    if "ALL" not in canon_path:
        canon_path["ALL"] = {}


def _ingest_entities_for_direction(
    canon_path: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    direction: str,
    ent_root: Any
) -> None:
    """
    ent_root expected: {cat: {entity: info}} but we keep robust.
    """
    if not isinstance(ent_root, dict):
        return

    _ensure_dir(canon_path, direction)

    for cat, ent_dict in ent_root.items():
        c = _norm_cat(cat)
        if c not in canon_path[direction]:
            canon_path[direction][c] = {}
        if c not in canon_path["ALL"]:
            canon_path["ALL"][c] = {}

        before = dict(canon_path[direction][c])

        _ingest_entity_dict(canon_path[direction][c], ent_dict)

        # also aggregate into ALL
        for ent, cnt in canon_path[direction][c].items():
            prev = before.get(ent, 0)
            added = cnt - prev
            if added > 0:
                canon_path["ALL"][c][ent] = canon_path["ALL"][c].get(ent, 0) + added


def load_overlap_canonical(out_root: Path, cohort_dir: Path, cohort: str) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Prefer:
      cohort_dir/overlap/pathway_entity_overlap.json
      out_root/jsons_all/<cohort>.json (case tolerant)
    """
    candidates = [
        cohort_dir / "overlap" / "pathway_entity_overlap.json",
        out_root / "jsons_all" / f"{cohort}.json",
        out_root / "jsons_all" / f"{cohort.lower()}.json",
    ]
    for p in candidates:
        if is_file(p):
            js = read_json(p)
            if isinstance(js, dict):
                return canonicalize_overlap_json(js), p
    return {}, None


def normalize_key(s: str) -> str:
    x = str(s).upper().strip()
    x = x.replace(" ", "_")
    x = re.sub(r"[^A-Z0-9_]+", "", x)
    x = re.sub(r"_+", "_", x)
    return x


def build_normalized_lookup(keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in keys:
        nk = normalize_key(k)
        if nk and nk not in out:
            out[nk] = k
    return out


def get_entities_for_pathway(
    canon: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    pathway_name: str,
    category: str,
    direction: str = "ALL",
    entity_limit: int = 200
) -> List[str]:
    """
    Returns entities sorted by hit-count desc, limited.
    Uses normalized lookup to tolerate naming mismatch.
    """
    if not canon:
        return []

    lookup = build_normalized_lookup(list(canon.keys()))
    pkey = lookup.get(normalize_key(pathway_name))
    if not pkey:
        return []

    dblock = canon[pkey].get(direction, {})
    c = _norm_cat(category)
    ent_map = dblock.get(c, {})
    if not isinstance(ent_map, dict) or not ent_map:
        return []

    items = sorted(ent_map.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[: max(1, int(entity_limit))]]


def total_hits_for_pathway(
    canon: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    pathway_name: str
) -> int:
    """
    Total evidence hits across ALL categories, direction ALL.
    """
    if not canon:
        return 0
    lookup = build_normalized_lookup(list(canon.keys()))
    pkey = lookup.get(normalize_key(pathway_name))
    if not pkey:
        return 0
    all_block = canon[pkey].get("ALL", {})
    total = 0
    if isinstance(all_block, dict):
        for ent_map in all_block.values():
            if isinstance(ent_map, dict):
                total += sum(int(v) for v in ent_map.values() if v is not None)
    return int(total)


# ---------------------------
# Cohort data container
# ---------------------------
@dataclass
class CohortData:
    name: str
    cohort_dir: Path
    pathway_table_path: Optional[Path]
    pathway_table_label: str
    pathway_df: pd.DataFrame
    cols: PathwayCols
    overlap_canon: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]
    overlap_path: Optional[Path]


def load_cohort(out_root: Path, cohort: str) -> CohortData:
    cdir = resolve_cohort_dir(out_root, cohort)
    if cdir is None:
        raise FileNotFoundError(f"Could not resolve cohort directory for: {cohort}")

    table_path, label = pick_pathway_table(cdir)
    df = pd.DataFrame()
    cols = PathwayCols("pathway", None, None, None, None, None, None)

    if table_path and is_file(table_path):
        df = read_table_auto(table_path)
        cols = detect_pathway_cols(df)

        # normalize pathway column name to 'pathway'
        if cols.pathway != "pathway" and cols.pathway in df.columns:
            df = df.rename(columns={cols.pathway: "pathway"})
            cols = PathwayCols(
                pathway="pathway",
                fdr=cols.fdr,
                p=cols.p,
                t=cols.t,
                delta=cols.delta,
                main_class=cols.main_class,
                sub_class=cols.sub_class,
            )

        if "pathway" in df.columns:
            df["pathway"] = df["pathway"].astype(str)

        coerce_numeric(df, cols.fdr)
        coerce_numeric(df, cols.p)
        coerce_numeric(df, cols.t)
        coerce_numeric(df, cols.delta)

    canon, opath = load_overlap_canonical(out_root, cdir, cohort)

    return CohortData(
        name=cohort,
        cohort_dir=cdir,
        pathway_table_path=table_path,
        pathway_table_label=label,
        pathway_df=df,
        cols=cols,
        overlap_canon=canon,
        overlap_path=opath,
    )


def safe_log10(x: float) -> float:
    if x is None or not (x > 0):
        return 0.0
    return math.log10(x)
