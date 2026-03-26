# mdp_insights/processing.py
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from .config import InsightsConfig

# ============================================================
# Helpers: significance / normalization
# ============================================================

def _sig_from_qp(q: Optional[float], p: Optional[float], cap: float) -> float:
    """
    Compute capped significance as -log10(q) else -log10(p).
    Returns np.nan if neither is valid.
    """
    try:
        if q is not None and np.isfinite(q):
            q = max(float(q), 1e-300)
            return min(-np.log10(q), cap)
        if p is not None and np.isfinite(p):
            p = max(float(p), 1e-300)
            return min(-np.log10(p), cap)
    except Exception:
        pass
    return np.nan


def _as_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _as_str_list(x: Any) -> List[str]:
    """
    Normalize overlap_genes (or similar) to a clean list[str].
    """
    if x is None:
        return []
    if isinstance(x, (set, tuple, list)):
        out = []
        for v in x:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return []


def _normalize_entity_type(etype: str) -> str:
    """
    Normalize entity_type values to match downstream expectations.
    """
    s = str(etype or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"tf", "tfs", "transcriptionfactor", "transcription_factor", "transcription factors"}:
        return "tf"
    if s in {"epigenetic", "epi", "epigenetics", "chromatin", "histone", "histone_mark", "histone marks"}:
        return "epigenetic"
    if s in {"metabolite", "metabolites", "metabolic", "hmdb"}:
        return "metabolites"
    # keep other types but lowercased
    return s


def _norm_entity_item(item: dict, etype: str, cap: float) -> Optional[dict]:
    """
    Convert one entity record (dict) into the canonical tidy record (without disease/pathway/direction).
    """
    if not isinstance(item, dict):
        return None

    ent = str(item.get("entity", "")).strip()
    if not ent:
        # Some schemas might use 'name' instead of 'entity'
        ent = str(item.get("name", "")).strip()
    if not ent:
        return None

    og = _as_str_list(item.get("overlap_genes", item.get("genes", item.get("overlap", []))))

    OR = _as_float(item.get("OR", item.get("odds_ratio")))
    qv = _as_float(item.get("qval", item.get("q_value", item.get("q", np.nan))))
    pv = _as_float(item.get("pval", item.get("p_value", item.get("p", np.nan))))

    sig = _sig_from_qp(qv if np.isfinite(qv) else None,
                       pv if np.isfinite(pv) else None,
                       cap)

    rec = {
        "entity_type": _normalize_entity_type(etype),
        "entity": ent,
        "overlap_genes": og,
        "OR": OR,
        "qval": qv,
        "pval": pv,
        "sig": sig,
        "k": _as_float(item.get("k")),
        "a": _as_float(item.get("a")),
        "b": _as_float(item.get("b")),
        "N": _as_float(item.get("N")),
        "Jaccard": _as_float(item.get("Jaccard", item.get("jaccard"))),
    }
    return rec


def _norm_entity_block(block: Any, etype: str, cap: float) -> List[dict]:
    """
    Normalize an entity block to a list of canonical entity records.
    Supports:
      - list[dict]
      - dict -> attempts to flatten dict values (if values are dicts or lists)
    """
    out: List[dict] = []
    if block is None:
        return out

    if isinstance(block, dict):
        # flatten values (could be entity->stats dict OR etype->list)
        # If dict looks like entity->stats:
        #   {"TP53": {"OR":..., "qval":...}, ...}
        # convert into list with 'entity' field set to key
        if block and all(isinstance(v, dict) for v in list(block.values())[:10]):
            for k, v in block.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("entity", str(k))
                    rec = _norm_entity_item(vv, etype, cap)
                    if rec:
                        out.append(rec)
            return out

        # else: flatten dict values recursively if they are lists
        for v in block.values():
            out.extend(_norm_entity_block(v, etype, cap))
        return out

    if not isinstance(block, list):
        return out

    for item in block:
        if not isinstance(item, dict):
            continue
        # If item itself has an entity_type field, prefer it
        itype = item.get("entity_type", etype)
        rec = _norm_entity_item(item, str(itype), cap)
        if rec:
            out.append(rec)
    return out


# ============================================================
# JSON schema handling (defensive even if io_utils normalized)
# ============================================================

def _unwrap_common_wrappers(data: Any) -> Any:
    """
    Unwrap common wrapper keys: pathways/results/data/overlap/payload (one or two levels).
    """
    if not isinstance(data, dict):
        return data
    for _ in range(2):
        if not isinstance(data, dict):
            break
        for k in ("pathways", "results", "data", "overlap", "payload"):
            v = data.get(k)
            if isinstance(v, dict) and v:
                data = v
                break
        else:
            break
    return data


def _looks_like_updown_root(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    keys = {str(k).strip().upper() for k in data.keys()}
    return bool(keys.intersection({"UP", "DOWN"}))


def _merge_updown_root_to_pathways(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert {"UP": {pathway: payload}, "DOWN": {pathway: payload}} into:
      pathway -> {"UP": payload, "DOWN": payload}
    """
    out: Dict[str, Any] = {}

    def pick_ci(d: Dict[str, Any], key: str) -> Any:
        lk = key.lower()
        for k in d.keys():
            if str(k).lower() == lk:
                return d[k]
        return None

    for direction in ("UP", "DOWN"):
        block = pick_ci(data, direction)
        if not isinstance(block, dict):
            continue
        for pw, payload in block.items():
            name = str(pw).strip()
            if not name:
                continue
            if name not in out or not isinstance(out.get(name), dict):
                out[name] = {}
            out[name][direction] = payload

    # Optional top-level ANY/ALL/entities mapping (attach per pathway if plausible)
    any_block = None
    for k in ("ANY", "ALL", "entities", "entity_list", "items"):
        any_block = pick_ci(data, k)
        if any_block is not None:
            break
    if isinstance(any_block, dict) and any_block:
        attachable = True
        for _, v in list(any_block.items())[:10]:
            if not isinstance(v, (dict, list, tuple)):
                attachable = False
                break
        if attachable:
            for pw, payload in any_block.items():
                name = str(pw).strip()
                if not name:
                    continue
                if name not in out or not isinstance(out.get(name), dict):
                    out[name] = {}
                out[name]["ANY"] = payload

    return out


def _normalize_root_to_pathway_map(js_obj: Any) -> Dict[str, Any]:
    """
    Ensure we operate on a mapping: pathway -> payload
    """
    js_obj = _unwrap_common_wrappers(js_obj)
    if not isinstance(js_obj, dict) or not js_obj:
        return {}
    if _looks_like_updown_root(js_obj):
        merged = _merge_updown_root_to_pathways(js_obj)
        return merged if merged else {}
    return js_obj


def _is_directional_payload(payload: Any) -> bool:
    """
    Pathway payload is directional if it has UP or DOWN keys (case-insensitive).
    """
    if not isinstance(payload, dict) or not payload:
        return False
    keys = {str(k).strip().upper() for k in payload.keys()}
    return "UP" in keys or "DOWN" in keys


def _pick_ci(payload: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(payload, dict):
        return None
    lower = {str(k).lower(): k for k in payload.keys()}
    for k in keys:
        kk = str(k).lower()
        if kk in lower:
            return payload[lower[kk]]
    return None


def _coerce_types_map(types_map: Any) -> Dict[str, Any]:
    """
    Ensure a types_map is a dict of etype -> block.
    If not possible, return empty dict.
    """
    if isinstance(types_map, dict):
        return types_map
    # If a list was provided directly, treat as unknown type list
    if isinstance(types_map, list):
        return {"unknown": types_map}
    return {}


# ============================================================
# Main tidy builder
# ============================================================

def tidy_from_json_obj(disease: str, js_obj: dict, cap: float) -> pd.DataFrame:
    """
    Build unified, tidy DF:
      columns:
        disease, pathway, direction, entity_type, entity, overlap_genes,
        OR, qval, pval, sig, k, a, b, N, Jaccard

    Supports schema variants:
      1) pathway -> {"UP": {etype: [..]}, "DOWN": {etype: [..]}}
      2) pathway -> {etype: [..]}  (direction-free)
      3) pathway -> {"ANY"/"ALL"/"entities": {etype: [..]}}  (direction-free wrapped per pathway)
      4) top-level {"UP": {pathway: ...}, "DOWN": {pathway: ...}} (defensive)
    """
    recs: List[dict] = []

    def emit(pathway: str, direction: str, etype: str, item: dict) -> None:
        recs.append({**item, "disease": disease, "pathway": pathway, "direction": direction})

    root = _normalize_root_to_pathway_map(js_obj)
    if not root:
        return pd.DataFrame(columns=[
            "disease", "pathway", "direction", "entity_type", "entity", "overlap_genes",
            "OR", "qval", "pval", "sig", "k", "a", "b", "N", "Jaccard"
        ])

    for pathway, payload in root.items():
        pw = str(pathway).strip()
        if not pw:
            continue

        # Directional payload
        if _is_directional_payload(payload):
            if not isinstance(payload, dict):
                continue

            for direction in ("UP", "DOWN", "ANY"):
                block = _pick_ci(payload, direction)
                if block is None:
                    continue

                # If direction block is itself wrapped under ANY/ALL/entities (rare), unwrap once
                if isinstance(block, dict):
                    maybe = _pick_ci(block, "ANY", "ALL", "entities", "entity_list", "items")
                    if isinstance(maybe, dict) or isinstance(maybe, list):
                        # Only take this unwrap if it looks like a types-map or list
                        block = maybe

                types_map = _coerce_types_map(block)
                if not types_map:
                    continue
                for et, arr in types_map.items():
                    for r in _norm_entity_block(arr, str(et), cap):
                        emit(pw, direction, str(et), r)

            continue

        # Direction-free payload
        # common: payload is {etype: [..]}
        # also: payload may be {"ANY"/"ALL"/"entities": {etype:[..]}} or {"ANY":[..]}
        any_block = None
        if isinstance(payload, dict):
            any_block = _pick_ci(payload, "ANY", "ALL", "entities", "entity_list", "items")
        if any_block is not None:
            types_map = _coerce_types_map(any_block)
        else:
            types_map = _coerce_types_map(payload)

        if not types_map:
            continue

        for et, arr in types_map.items():
            for r in _norm_entity_block(arr, str(et), cap):
                emit(pw, "ANY", str(et), r)

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return pd.DataFrame(columns=[
            "disease", "pathway", "direction", "entity_type", "entity", "overlap_genes",
            "OR", "qval", "pval", "sig", "k", "a", "b", "N", "Jaccard"
        ])

    # Canonicalize entity_type
    df["entity_type"] = df["entity_type"].apply(_normalize_entity_type)

    # Dedupe: keep best (lowest q, then lowest p), then highest sig
    for col in ("qval", "pval", "sig"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values(by=["qval", "pval", "sig"], ascending=[True, True, False], na_position="last")
          .drop_duplicates(["disease", "pathway", "direction", "entity_type", "entity"], keep="first")
          .reset_index(drop=True)
    )
    return df


# ============================================================
# Filters
# ============================================================

def apply_epigenetic_suffix_filter(df: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    """
    Drop epigenetic entities whose final token is mm/mm9/mm10… (mouse genome tags).
    Only applies when cfg.exclude_mouse_epigenetic_suffix is True.
    """
    if df is None or df.empty or not getattr(cfg, "exclude_mouse_epigenetic_suffix", False):
        return df

    pat = re.compile(getattr(cfg, "mouse_suffix_regex", r"(?i)(?:^|[\s_\-()])mm(?:\d{1,2})?$"))

    def is_mouse_suffix(name: str) -> bool:
        if not isinstance(name, str) or not name.strip():
            return False
        tokens = re.split(r"[^A-Za-z0-9]+", name.strip())
        last = tokens[-1].lower() if tokens else ""
        if re.fullmatch(r"mm(?:\d{1,2})?", last):
            return True
        return bool(pat.search(name))

    df = df.copy()
    mask_drop = (df["entity_type"].astype(str).str.lower().eq("epigenetic")) & (
        df["entity"].apply(is_mouse_suffix)
    )
    return df.loc[~mask_drop].reset_index(drop=True)


def apply_tf_mouse_filter(df: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    """
    Drop TF entities whose names suggest mouse origin anywhere:
      matches: 'mouse', 'murine', 'Mus musculus', '(mouse)', '[mouse]', case-insensitive.
    Controlled by cfg.exclude_mouse_tf and cfg.tf_mouse_regex.
    """
    if df is None or df.empty or not getattr(cfg, "exclude_mouse_tf", False):
        return df
    pat = re.compile(getattr(cfg, "tf_mouse_regex", r"(?i)\b(?:mouse|murine|mus\s*musculus)\b"))
    df = df.copy()
    pat_str = pat.pattern
    mask_drop = (df["entity_type"].astype(str).str.lower().eq("tf")) & (
        df["entity"].astype(str).str.contains(pat_str, regex=True, na=False)
    )
    return df.loc[~mask_drop].reset_index(drop=True)


# ============================================================
# Presence matrices & view builders
# ============================================================

def _collapse_to_any(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an ANY view even if input is only directional (UP/DOWN).
    - If ANY exists, keep it.
    - If UP/DOWN exists, synthesize ANY as union aggregation.
    """
    if df.empty:
        return df.copy()

    dir_df = df[df["direction"].isin(["UP", "DOWN"])]
    if dir_df.empty:
        return df[df["direction"].eq("ANY")].copy()

    # max sig; median OR; keep union/longest overlap_genes list
    def _longest_gene_list(lists: pd.Series) -> List[str]:
        best: List[str] = []
        for v in lists:
            if isinstance(v, (list, tuple, set)):
                vv = [x for x in v if isinstance(x, str)]
                if len(vv) > len(best):
                    best = vv
        return best

    any_from_dir = (
        dir_df.groupby(["disease", "pathway", "entity_type", "entity"], as_index=False)
              .agg(
                  sig=("sig", "max"),
                  OR=("OR", lambda x: float(np.nanmedian(pd.to_numeric(x, errors="coerce")))),
                  qval=("qval", "min"),
                  pval=("pval", "min"),
                  Jaccard=("Jaccard", "max"),
                  k=("k", "max"),
                  a=("a", "max"),
                  b=("b", "max"),
                  N=("N", "max"),
                  overlap_genes=("overlap_genes", _longest_gene_list),
              )
    )

    base_any = df[df["direction"].eq("ANY")]
    merged = pd.concat([base_any, any_from_dir], ignore_index=True)
    merged = (
        merged.sort_values(by=["qval", "pval", "sig"], ascending=[True, True, False], na_position="last")
              .drop_duplicates(["disease", "pathway", "entity_type", "entity"], keep="first")
              .assign(direction="ANY")
              .reset_index(drop=True)
    )
    return merged


def build_views(df: pd.DataFrame, borrow: str) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with ANY/UP/DOWN views. Optionally borrows ANY into UP/DOWN.
      borrow ∈ {"none","up","down","both"}
    """
    borrow = (borrow or "none").lower()
    any_view = _collapse_to_any(df)
    up_view = df[df["direction"].eq("UP")].copy()
    down_view = df[df["direction"].eq("DOWN")].copy()

    def _borrow_into(view: pd.DataFrame, direction: str) -> pd.DataFrame:
        borrowed = any_view.assign(direction=direction, borrowed=True)
        merged = pd.concat([view, borrowed], ignore_index=True)
        return merged.drop_duplicates(["disease", "pathway", "entity_type", "entity", "direction"], keep="first")

    if borrow in {"up", "both"}:
        up_view = _borrow_into(up_view, "UP")
    if borrow in {"down", "both"}:
        down_view = _borrow_into(down_view, "DOWN")

    return {"ANY": any_view, "UP": up_view, "DOWN": down_view}


def build_presence_matrix(presence_df: pd.DataFrame, mode: str = "ANY") -> pd.DataFrame:
    if presence_df.empty:
        return pd.DataFrame()
    mode = (mode or "ANY").upper()
    if mode not in {"UP", "DOWN", "ANY"}:
        mode = "ANY"
    try:
        mat = presence_df.pivot_table(
            index="pathway",
            columns="disease",
            values=mode,
            aggfunc="max",
            fill_value=0,
        )
        return mat.fillna(0).astype(int)
    except Exception as e:
        logging.error(f"build_presence_matrix failed: {e}")
        return pd.DataFrame()


def _normalize_direction_value(x: Any) -> str:
    """
    Robust normalization for direction labels.
    """
    try:
        s = str(x).strip().upper()
    except Exception:
        return "ANY"
    if s in {"UP", "UPREG", "UPREGULATED", "UP-REGULATED", "POS", "POSITIVE"}:
        return "UP"
    if s in {"DOWN", "DOWNREG", "DOWNREGULATED", "DOWN-REGULATED", "NEG", "NEGATIVE"}:
        return "DOWN"
    if s in {"ANY", "ALL", "BOTH", "NA", "NONE", ""}:
        return "ANY"
    return "ANY"


def presence_from_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build presence table with columns UP/DOWN/ANY from tidy entities DF.

    Critical rule:
      ANY = 1 if (ANY present) OR (UP present) OR (DOWN present)
    """
    if df.empty:
        return pd.DataFrame(columns=["disease", "pathway", "UP", "DOWN", "ANY"])

    work = df.copy()
    if "direction" in work.columns:
        work["direction"] = work["direction"].apply(_normalize_direction_value)
    else:
        work["direction"] = "ANY"

    g = (
        work.assign(v=1)
            .pivot_table(
                index=["disease", "pathway"],
                columns="direction",
                values="v",
                aggfunc="max",
                fill_value=0,
            )
    )

    for col in ("UP", "DOWN", "ANY"):
        if col not in g.columns:
            g[col] = 0

    try:
        up = pd.to_numeric(g["UP"], errors="coerce").fillna(0).astype(int)
        dn = pd.to_numeric(g["DOWN"], errors="coerce").fillna(0).astype(int)
        anyv = pd.to_numeric(g["ANY"], errors="coerce").fillna(0).astype(int)
        g["ANY"] = np.maximum(anyv, np.maximum(up, dn)).astype(int)
        g["UP"] = up
        g["DOWN"] = dn
    except Exception as e:
        logging.error(f"presence_from_entities ANY-union fix failed: {e}")

    g = g[["UP", "DOWN", "ANY"]].reset_index()
    return g


# ============================================================
# Shared & Individual summaries
# ============================================================

def compute_global_degrees(entities_df: pd.DataFrame) -> pd.DataFrame:
    if entities_df.empty:
        return pd.DataFrame(columns=["entity_type", "entity", "pathways_count", "diseases_count", "total_occurrences"])
    try:
        g1 = entities_df.groupby(["entity_type", "entity"])["pathway"].nunique().rename("pathways_count")
        g2 = entities_df.groupby(["entity_type", "entity"])["disease"].nunique().rename("diseases_count")
        g3 = entities_df.groupby(["entity_type", "entity"]).size().rename("total_occurrences")
        out = pd.concat([g1, g2, g3], axis=1).reset_index()
        return out.sort_values(["entity_type", "total_occurrences"], ascending=[True, False])
    except Exception as e:
        logging.error(f"compute_global_degrees failed: {e}")
        return pd.DataFrame(columns=["entity_type", "entity", "pathways_count", "diseases_count", "total_occurrences"])


def pathway_specificity_score(entities_df: pd.DataFrame) -> pd.DataFrame:
    if entities_df.empty:
        return pd.DataFrame(columns=["disease", "pathway", "pss", "n_entities"])
    try:
        per_dis_path = entities_df.groupby(["disease", "pathway"])["entity"].nunique().rename("n_entities").reset_index()
        global_prev = entities_df.groupby(["pathway"])["disease"].nunique().rename("global_prev").reset_index()
        df = per_dis_path.merge(global_prev, on="pathway", how="left")
        df["pss"] = df["n_entities"] * (1.0 / (1.0 + np.log(df["global_prev"].clip(lower=1))))
        return df
    except Exception as e:
        logging.error(f"pathway_specificity_score failed: {e}")
        return pd.DataFrame(columns=["disease", "pathway", "pss", "n_entities"])


def _filter_for_shared_views(entities_df: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    try:
        df = entities_df.copy()
        if getattr(cfg, "exclude_hmdb_in_shared", False):
            is_hmdb = df["entity_type"].eq("metabolites") & df["entity"].astype(str).str.contains(r"\(HMDB", na=False)
            df = df.loc[~is_hmdb].copy()
        return df
    except Exception as e:
        logging.debug(f"_filter_for_shared_views failed: {e}")
        return entities_df


def summarize_shared(entities_df: pd.DataFrame, presence_df: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    try:
        df = _filter_for_shared_views(entities_df, cfg)
        mat_any = build_presence_matrix(presence_df, "ANY")
        if df.empty or mat_any.empty:
            return pd.DataFrame(columns=[
                "pathway", "diseases", "n_diseases",
                "tf_names", "tf_n", "epigenetic_names", "epigenetic_n",
                "metabolite_names", "metabolite_n", "shared_genes", "shared_genes_n",
            ])

        def collect_names(pw: str, et: str) -> Tuple[str, int]:
            sub = df[(df["pathway"] == pw) & (df["entity_type"] == et)]
            names = sorted(set(sub["entity"].dropna().astype(str).tolist()))
            return "; ".join(names), len(names)

        # union per (pathway, disease)
        union_map: Dict[Tuple[str, str], Set[str]] = {}
        for _, r in df[["pathway", "disease", "overlap_genes"]].dropna().iterrows():
            pw = str(r["pathway"])
            ds = str(r["disease"])
            og = r["overlap_genes"]
            if not isinstance(og, (list, tuple, set)):
                continue
            s = union_map.get((pw, ds), set())
            s.update([g for g in og if isinstance(g, str)])
            union_map[(pw, ds)] = s

        rows = []
        for pw in mat_any.index:
            try:
                diseases = [d for d in mat_any.columns if int(mat_any.loc[pw, d]) == 1]
                if not diseases:
                    continue

                tf_names, tf_n = collect_names(pw, "tf")
                epi_names, epi_n = collect_names(pw, "epigenetic")
                met_names, met_n = collect_names(pw, "metabolites")

                gene_sets = [union_map.get((pw, d), set()) for d in diseases]
                if not gene_sets:
                    shared = set()
                elif len(gene_sets) == 1:
                    shared = set(gene_sets[0])
                else:
                    shared = set.intersection(*gene_sets)

                if getattr(cfg, "min_shared_gene_intersection", 0) > 1 and len(shared) < cfg.min_shared_gene_intersection:
                    shared = set()

                rows.append({
                    "pathway": pw,
                    "diseases": ", ".join(sorted(diseases)),
                    "n_diseases": len(diseases),
                    "tf_names": tf_names, "tf_n": tf_n,
                    "epigenetic_names": epi_names, "epigenetic_n": epi_n,
                    "metabolite_names": met_names, "metabolite_n": met_n,
                    "shared_genes": "; ".join(sorted(shared)), "shared_genes_n": len(shared),
                })
            except Exception as e:
                logging.debug(f"summarize_shared row fail for {pw}: {e}")

        res = pd.DataFrame(rows)
        if not res.empty:
            res = res.sort_values(["n_diseases", "pathway"], ascending=[False, True]).reset_index(drop=True)
        return res
    except Exception as e:
        logging.error(f"summarize_shared failed: {e}")
        return pd.DataFrame()


def summarize_individual(entities_df: pd.DataFrame, cfg: InsightsConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    outputs: Dict[str, Dict[str, pd.DataFrame]] = {}
    try:
        if entities_df.empty:
            return outputs

        pss_all = pathway_specificity_score(entities_df)
        per_dis = dict(tuple(entities_df.groupby("disease", sort=True)))
        deg = compute_global_degrees(entities_df)

        for disease, df in per_dis.items():
            try:
                pw_pss = pss_all.loc[pss_all["disease"] == disease, ["pathway", "pss", "n_entities"]].copy()
                pw_pss["n_entities"] = pd.to_numeric(pw_pss["n_entities"], errors="coerce").fillna(0).astype(int)
                pw_pss["pss"] = pd.to_numeric(pw_pss["pss"], errors="coerce").fillna(0.0)
                top_pw = pw_pss.sort_values(["pss", "n_entities"], ascending=[False, False]).head(cfg.top_k)

                def top_entities(etype: str) -> pd.DataFrame:
                    sub = df[df["entity_type"].eq(etype)].copy()
                    if sub.empty:
                        return pd.DataFrame(columns=["entity", "count", "median_qval", "is_hub"])
                    g = (
                        sub.groupby("entity")
                           .agg(
                               count=("entity", "size"),
                               median_qval=("qval", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce")))),
                           )
                           .reset_index()
                           .sort_values(["count", "median_qval"], ascending=[False, True])
                           .head(cfg.top_k)
                    )
                    if not deg.empty:
                        g = g.merge(
                            deg[deg["entity_type"].eq(etype)][["entity", "pathways_count"]],
                            on="entity",
                            how="left",
                        )
                        g["is_hub"] = (g["pathways_count"].fillna(0) > cfg.hub_cap).astype(int)
                        g.drop(columns=["pathways_count"], inplace=True, errors="ignore")
                    else:
                        g["is_hub"] = 0
                    return g

                mix = (
                    df.pivot_table(
                        index="pathway",
                        columns="entity_type",
                        values="entity",
                        aggfunc="nunique",
                        fill_value=0,
                    )
                    .reset_index()
                )
                for col in ("tf", "epigenetic", "metabolites"):
                    if col not in mix.columns:
                        mix[col] = 0
                mix = (
                    mix.rename(columns={"tf": "tf_n", "epigenetic": "epigenetic_n", "metabolites": "metabolite_n"})
                       .assign(total_n=lambda x: x.get("tf_n", 0) + x.get("epigenetic_n", 0) + x.get("metabolite_n", 0))
                       .sort_values(["total_n", "pathway"], ascending=[False, True])
                )

                qvals_per_type = df[["entity_type", "qval"]].dropna().copy()
                or_per_type = df[["entity_type", "OR"]].dropna().copy()

                gene_union_counts = []
                for pw, sub in df.groupby("pathway"):
                    genes = set()
                    for og in sub["overlap_genes"].dropna():
                        if isinstance(og, (list, tuple, set)):
                            genes.update([g for g in og if isinstance(g, str)])
                    gene_union_counts.append({"pathway": pw, "genes_n": len(genes)})
                gene_union_counts = (
                    pd.DataFrame(gene_union_counts)
                    if gene_union_counts
                    else pd.DataFrame(columns=["pathway", "genes_n"])
                ).sort_values("genes_n", ascending=False)

                gene_counts: Dict[str, int] = {}
                for og in df["overlap_genes"].dropna():
                    if isinstance(og, (list, tuple, set)):
                        for gname in og:
                            if isinstance(gname, str) and gname.strip():
                                gene_counts[gname.strip()] = gene_counts.get(gname.strip(), 0) + 1
                gc = (
                    pd.DataFrame({"gene": list(gene_counts.keys()), "count": list(gene_counts.values())})
                      .sort_values("count", ascending=False)
                      .head(cfg.top_k)
                )

                outputs[disease] = {
                    "top_pathways": top_pw.reset_index(drop=True),
                    "top_tf": top_entities("tf"),
                    "top_epigenetic": top_entities("epigenetic"),
                    "top_metabolites": top_entities("metabolites"),
                    "gene_leaders": gc,
                    "pathway_type_counts": mix,
                    "qvals_per_type": qvals_per_type,
                    "or_per_type": or_per_type,
                    "pathway_gene_counts": gene_union_counts,
                }

            except Exception as ee:
                logging.error(f"summarize_individual failed for {disease}: {ee}")
        return outputs
    except Exception as e:
        logging.error(f"summarize_individual failed: {e}")
        return outputs


# ============================================================
# Cross-disease Top-N (PIS/TPS/etc.)
# ============================================================

def top200_pathways_cross_disease(views_any: pd.DataFrame, entities_any: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    """
    Compute PIS (Pathway Intelligence Score) and return top N pathways.
    """
    if views_any.empty or entities_any.empty:
        return pd.DataFrame(columns=["pathway", "n_diseases", "max_sig", "median_log10OR", "total_genes", "median_k", "specificity", "PIS"])
    try:
        n_diseases = views_any.groupby(["pathway"])["disease"].nunique().rename("n_diseases").reset_index()

        agg = (
            entities_any.groupby(["pathway"])
                        .agg(
                            max_sig=("sig", lambda s: float(np.nanmax(pd.to_numeric(s, errors="coerce")))),
                            median_log10OR=("OR", lambda s: float(np.nanmedian(np.log10(np.clip(pd.to_numeric(s, errors="coerce").values, 1e-12, None))))),
                            median_k=("k", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce")))),
                        )
                        .reset_index()
        )

        global_prev = entities_any.groupby(["pathway"])["disease"].nunique().rename("global_prev").reset_index()

        genes_count = (
            entities_any.groupby(["pathway", "disease"])["overlap_genes"]
                        .apply(lambda col: len(set([g for og in col.dropna()
                                                    for g in (og if isinstance(og, (list, tuple, set)) else [])
                                                    if isinstance(g, str)])))
                        .groupby("pathway")
                        .sum()
                        .rename("total_genes")
                        .reset_index()
        )

        df = n_diseases.merge(agg, on="pathway", how="left") \
                       .merge(global_prev, on="pathway", how="left") \
                       .merge(genes_count, on="pathway", how="left")

        df["specificity"] = 1.0 / (1.0 + np.log(df["global_prev"].clip(lower=1)))

        w1, w2, w3, w4 = cfg.pis_w_prevalence, cfg.pis_w_evidence, cfg.pis_w_specificity, cfg.pis_w_support
        support = df["median_log10OR"].fillna(0.0) + (df["median_k"].fillna(0.0) / 10.0) + (np.log1p(df["total_genes"].fillna(0.0)) / 5.0)

        df["PIS"] = (
            w1 * df["n_diseases"].fillna(0.0) +
            w2 * df["max_sig"].fillna(0.0) +
            w3 * df["specificity"].fillna(0.0) +
            w4 * support
        )

        df = df.sort_values(["PIS", "n_diseases", "max_sig"], ascending=[False, False, False]).head(cfg.top_n)
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"top200_pathways_cross_disease failed: {e}")
        return pd.DataFrame()


def top200_targets_by_type(entities_any: pd.DataFrame, etype: str, cfg: InsightsConfig) -> pd.DataFrame:
    if entities_any.empty:
        return pd.DataFrame(columns=["entity", "type", "n_pathways", "n_diseases", "max_sig", "median_log10OR", "up_hits", "down_hits", "mixed_flag", "TPS"])
    try:
        etype = _normalize_entity_type(etype)
        sub = entities_any[entities_any["entity_type"].eq(etype)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["entity", "type", "n_pathways", "n_diseases", "max_sig", "median_log10OR", "up_hits", "down_hits", "mixed_flag", "TPS"])

        n_pathways = sub.groupby("entity")["pathway"].nunique().rename("n_pathways")
        n_diseases = sub.groupby("entity")["disease"].nunique().rename("n_diseases")
        max_sig = sub.groupby("entity")["sig"].max().rename("max_sig")
        med_or = sub.groupby("entity")["OR"].apply(
            lambda s: float(np.nanmedian(np.log10(np.clip(pd.to_numeric(s, errors="coerce").values, 1e-12, None))))
        ).rename("median_log10OR")

        df = pd.concat([n_pathways, n_diseases, max_sig, med_or], axis=1).reset_index()
        df["type"] = etype

        # Directionality placeholders (0 for ANY-only inputs)
        df["up_hits"] = 0
        df["down_hits"] = 0
        df["mixed_flag"] = 0

        v1, v2, v3, v4 = cfg.tps_w_cross_pathway, cfg.tps_w_cross_disease, cfg.tps_w_evidence, cfg.tps_w_directionality
        df["TPS"] = (
            v1 * df["n_pathways"].fillna(0.0) +
            v2 * df["n_diseases"].fillna(0.0) +
            v3 * df["max_sig"].fillna(0.0) +
            v4 * (df["up_hits"] - df["down_hits"])
        )

        df = df.sort_values(["TPS", "n_diseases", "max_sig"], ascending=[False, False, False]).head(cfg.top_n)
        return df[["entity", "type", "n_pathways", "n_diseases", "max_sig", "median_log10OR", "up_hits", "down_hits", "mixed_flag", "TPS"]]
    except Exception as e:
        logging.error(f"top200_targets_by_type failed for {etype}: {e}")
        return pd.DataFrame()


def directional_concordance_table(df_dir: pd.DataFrame, cfg: InsightsConfig) -> pd.DataFrame:
    """
    For directional rows only (UP/DOWN), compute concordance per pathway.
    """
    try:
        sub = df_dir[df_dir["direction"].isin(["UP", "DOWN"])].dropna(subset=["pathway"])
        if sub.empty:
            return pd.DataFrame(columns=["pathway", "n_diseases_with_direction", "n_UP", "n_DOWN", "concordance", "median_sig_UP", "median_sig_DOWN"])
        pres = (
            sub.assign(v=1)
               .groupby(["disease", "pathway", "direction"])["v"]
               .max()
               .unstack("direction", fill_value=0)
               .reset_index()
        )
        # ensure both columns exist to avoid KeyError
        for col in ("UP", "DOWN"):
            if col not in pres.columns:
                pres[col] = 0
        grp = pres.groupby("pathway")
        n_UP = grp["UP"].sum()
        n_DOWN = grp["DOWN"].sum()
        n_dir = n_UP.add(n_DOWN)
        concord = np.where(n_dir > 0, np.maximum(n_UP, n_DOWN) / n_dir, np.nan)
        med_up = sub[sub["direction"].eq("UP")].groupby("pathway")["sig"].median()
        med_dn = sub[sub["direction"].eq("DOWN")].groupby("pathway")["sig"].median()
        out = pd.DataFrame({
            "pathway": n_dir.index,
            "n_diseases_with_direction": n_dir.values.astype(int),
            "n_UP": n_UP.values.astype(int),
            "n_DOWN": n_DOWN.values.astype(int),
            "concordance": concord,
            "median_sig_UP": med_up.reindex(n_dir.index).values,
            "median_sig_DOWN": med_dn.reindex(n_dir.index).values,
        }).sort_values(["concordance", "n_diseases_with_direction"], ascending=[False, False])
        return out.reset_index(drop=True)
    except Exception as e:
        logging.error(f"directional_concordance_table failed: {e}")
        return pd.DataFrame()


def disease_interconnection_table(mat_any: pd.DataFrame) -> pd.DataFrame:
    try:
        if mat_any is None or mat_any.empty:
            return pd.DataFrame(columns=["disease_i", "disease_j", "jaccard_ANY", "shared_pathways"])
        X = mat_any.to_numpy().astype(int)
        inter = X.T @ X
        col_sums = X.sum(axis=0, keepdims=True)
        union = col_sums + col_sums.T - inter
        union[union == 0] = 1
        sim = inter / union
        diseases = mat_any.columns.tolist()
        rows = []
        for i in range(len(diseases)):
            for j in range(i + 1, len(diseases)):
                rows.append({
                    "disease_i": diseases[i],
                    "disease_j": diseases[j],
                    "jaccard_ANY": float(sim[i, j]),
                    "shared_pathways": int(inter[i, j]),
                })
        return pd.DataFrame(rows).sort_values(["jaccard_ANY", "shared_pathways"], ascending=[False, False]).reset_index(drop=True)
    except Exception as e:
        logging.error(f"disease_interconnection_table failed: {e}")
        return pd.DataFrame()
