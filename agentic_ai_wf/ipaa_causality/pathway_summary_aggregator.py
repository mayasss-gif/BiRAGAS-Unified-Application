#!/usr/bin/env python3
"""
pathway_summary_aggregator.py

Post-run aggregator: builds a single pathway-row table per disease from scattered pipeline outputs.

- Robust to missing files.
- Works for multiple diseases under a single out_root.
- Does NOT modify pipeline artifacts; only reads and writes summaries.

Outputs:
  OUT_ROOT/engines/pathway_summary/<disease>/pathway_summary.tsv
  OUT_ROOT/engines/pathway_summary/<disease>/pathway_summary_ranked.tsv
  OUT_ROOT/engines/pathway_summary/<disease>/PATHWAY_SUMMARY_MANIFEST.json
  OUT_ROOT/engines/pathway_summary/<disease>/filtered_summary.tsv     <-- filtered significant-only, important columns
  OUT_ROOT/engines/pathway_summary/all_diseases_pathway_summary.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# default significance threshold for filtered_summary.tsv
_DEFAULT_FILTER_FDR = 0.05


# ----------------------------
# small utilities
# ----------------------------

def _info(msg: str) -> None:
    print(msg)

def _warn(msg: str) -> None:
    print(msg, file=os.sys.stderr)

def _safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def _sniff_sep(sample: str) -> str:
    # simple: tsv vs csv
    if sample.count("\t") >= sample.count(","):
        return "\t"
    return ","

def read_table_auto(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()

    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suf in {".tsv", ".txt", ".csv"}:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        sep = _sniff_sep(sample)
        # python engine is more forgiving than C engine
        return pd.read_csv(path, sep=sep, dtype=str, engine="python")

    # fallback
    return pd.read_csv(path, dtype=str, engine="python")

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None

def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _norm_col(c: str) -> str:
    return re.sub(r"\s+", "", str(c or "").strip().lower())

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _pick_pathway_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    norm = {_norm_col(c): c for c in cols}
    for key in ["pathway", "pathway_id", "term", "term_name", "name", "id"]:
        if key in norm:
            return norm[key]
    # fallback: first column
    return cols[0] if cols else None

def _ensure_key(df: pd.DataFrame, key_name: str = "pathway") -> pd.DataFrame:
    df = df.copy()
    pc = _pick_pathway_column(df)
    if pc is None:
        return pd.DataFrame(columns=[key_name])
    df[key_name] = df[pc].astype(str).str.strip()
    df = df[df[key_name] != ""]
    return df

def _prefix_columns(df: pd.DataFrame, prefix: str, keep: List[str]) -> pd.DataFrame:
    """
    keep: list of columns that should not be prefixed (e.g., key column)
    """
    df = df.copy()
    ren = {}
    for c in df.columns:
        if c in keep:
            continue
        ren[c] = f"{prefix}{c}"
    return df.rename(columns=ren)

def _merge_left(base: pd.DataFrame, add: pd.DataFrame, on: str = "pathway") -> pd.DataFrame:
    if add is None or add.empty:
        return base
    # protect duplicates in add
    add = add.drop_duplicates(subset=[on], keep="first")
    return base.merge(add, on=on, how="left")


# ----------------------------
# tissue selection + baseline
# ----------------------------

def load_selected_tissue(out_root: Path, disease: str) -> Optional[str]:
    c1 = out_root / "cohorts" / disease / "TISSUE_SELECTION.json"
    c2 = out_root / disease / "TISSUE_SELECTION.json"
    p = _first_existing([c1, c2])
    if p is None:
        return None
    try:
        obj = json.loads(_safe_read_text(p))
        # try common keys
        for k in ["resolved_tissue", "tissue", "selected_tissue", "final_tissue"]:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # sometimes nested
        if isinstance(obj.get("selection"), dict):
            sel = obj["selection"]
            for k in ["resolved_tissue", "tissue", "selected_tissue", "final_tissue"]:
                v = sel.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        return None
    return None

def load_baseline_expectations(out_root: Path, tissue: str) -> Optional[pd.DataFrame]:
    """
    baseline_consensus/baseline_pathway_expectations_c2cp.tsv has ALL tissues.
    We filter to the selected tissue.
    """
    p = out_root / "baseline_consensus" / "baseline_pathway_expectations_c2cp.tsv"
    if not p.exists():
        return None
    try:
        df = read_table_auto(p)
        if df.empty:
            return None
        df = _ensure_key(df, key_name="pathway")
        # need tissue column
        cols = {_norm_col(c): c for c in df.columns}
        tcol = cols.get("tissue")
        if tcol is None:
            return None
        # filter
        t = (tissue or "").strip().lower()
        df = df[df[tcol].astype(str).str.strip().str.lower() == t]
        if df.empty:
            return None
        # keep only useful fields
        keep_cols = ["pathway"]
        for want in ["expectation", "n_genes", "tissue"]:
            c = cols.get(_norm_col(want))
            if c and c not in keep_cols:
                keep_cols.append(c)
        out = df[keep_cols].copy()
        out = out.rename(columns={tcol: "tissue"})
        # standardize names
        if "expectation" not in out.columns:
            for c in out.columns:
                if "expect" in _norm_col(c):
                    out = out.rename(columns={c: "expectation"})
                    break
        if "n_genes" not in out.columns:
            for c in out.columns:
                if _norm_col(c) in {"ngenes", "n_gene", "genes"}:
                    out = out.rename(columns={c: "n_genes"})
                    break
        return out
    except Exception as e:
        _warn(f"[baseline] failed reading {p}: {e}")
        return None


# ----------------------------
# labels + matrix summarization
# ----------------------------

def load_labels(out_root: Path, disease: str) -> Optional[pd.Series]:
    """
    Accepts:
      - index=sample_id with a label column
      - OR two-column: sample_id, label
    Returns Series indexed by sample_id, value=label.
    """
    c1 = out_root / disease / "labels_used.tsv"
    c2 = out_root / "cohorts" / disease / "labels_used.tsv"
    p = _first_existing([c1, c2])
    if p is None:
        return None
    try:
        df = read_table_auto(p)
        if df.empty:
            return None
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] >= 2:
            c0 = df.columns[0]
            c1 = df.columns[1]
            cols = {_norm_col(c): c for c in df.columns}
            if "label" in cols:
                labc = cols["label"]
                sampc = cols.get("sample_id") or cols.get("sample") or cols.get("id") or c0
                s = df[[sampc, labc]].dropna()
                s[sampc] = s[sampc].astype(str).str.strip()
                s[labc] = s[labc].astype(str).str.strip()
                s = s[s[sampc] != ""]
                if not s.empty:
                    return pd.Series(s[labc].values, index=s[sampc].values)
            s = df[[c0, c1]].dropna()
            s[c0] = s[c0].astype(str).str.strip()
            s[c1] = s[c1].astype(str).str.strip()
            s = s[s[c0] != ""]
            if not s.empty:
                return pd.Series(s[c1].values, index=s[c0].values)

        cols = {_norm_col(c): c for c in df.columns}
        if "label" in cols:
            labc = cols["label"]
            idx = df.index.astype(str).str.strip()
            vals = df[labc].astype(str).str.strip()
            return pd.Series(vals.values, index=idx.values)
    except Exception:
        return None
    return None

def summarize_matrix_by_pathway(
    matrix_path: Path,
    labels: Optional[pd.Series] = None,
) -> Optional[pd.DataFrame]:
    """
    Reads a sample x pathway matrix (index=samples, columns=pathways) and returns:
      pathway, mean, sd
      optionally: mean_labelA, mean_labelB, delta(B-A) if exactly 2 labels align
    """
    if matrix_path is None or not matrix_path.exists():
        return None
    try:
        df = read_table_auto(matrix_path)
        if df.empty:
            return None

        cols = {_norm_col(c): c for c in df.columns}
        idx_col = cols.get("sample") or cols.get("sample_id") or cols.get("id")
        if idx_col is not None:
            df = df.set_index(idx_col)

        if df.columns.size > 0 and df.index.name is None:
            first = df.columns[0]
            if first and first not in ("pathway", "pathway_id"):
                ser = df[first].astype(str)
                numeric_rate = pd.to_numeric(ser, errors="coerce").notna().mean()
                if numeric_rate < 0.2:
                    df = df.set_index(first)

        df.index = df.index.astype(str).str.strip()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")

        mean = df.mean(axis=0, skipna=True)
        sd = df.std(axis=0, skipna=True, ddof=0)

        out = pd.DataFrame({
            "pathway": mean.index.astype(str),
            "mean": mean.values,
            "sd": sd.values,
        })

        if labels is not None and not labels.empty:
            lab = labels.copy()
            lab.index = lab.index.astype(str).str.strip()
            common = df.index.intersection(lab.index)
            if len(common) >= 4:
                dfa = df.loc[common]
                laba = lab.loc[common]
                uniq = [u for u in pd.unique(laba.values) if str(u).strip() != ""]
                if len(uniq) == 2:
                    a, b = sorted([str(uniq[0]), str(uniq[1])])
                    ma = dfa[laba == a].mean(axis=0, skipna=True)
                    mb = dfa[laba == b].mean(axis=0, skipna=True)
                    out["label_a"] = a
                    out["label_b"] = b
                    out["mean_a"] = ma.values
                    out["mean_b"] = mb.values
                    out["delta_b_minus_a"] = (mb - ma).values

        return out
    except Exception as e:
        _warn(f"[matrix] failed reading {matrix_path}: {e}")
        return None


# ----------------------------
# disease file discovery
# ----------------------------

def discover_diseases(out_root: Path) -> List[str]:
    diseases: List[str] = []
    c = out_root / "cohorts"
    if c.exists() and c.is_dir():
        for p in sorted(c.iterdir()):
            if p.is_dir():
                diseases.append(p.name)

    if diseases:
        return diseases

    e = out_root / "engines" / "evidence_bundle"
    if e.exists() and e.is_dir():
        for p in sorted(e.iterdir()):
            if p.is_dir():
                diseases.append(p.name)

    return diseases


# ----------------------------
# signor summarization
# ----------------------------

def load_signor_matches(out_root: Path, disease: str) -> Optional[pd.DataFrame]:
    p = out_root / "engines" / "signor_pathways" / disease / "signor_pathway_matches.tsv"
    if not p.exists():
        return None
    try:
        df = read_table_auto(p)
        if df.empty:
            return None
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={
            "ipaa_pathway": "pathway",
        })
        df = _ensure_key(df, key_name="pathway")

        if "match_score" in df.columns:
            df["match_score_num"] = _to_num(df["match_score"])
            df = df.sort_values("match_score_num", ascending=False)
            df = df.drop_duplicates(subset=["pathway"], keep="first")
            df = df.drop(columns=["match_score_num"], errors="ignore")
        else:
            df = df.drop_duplicates(subset=["pathway"], keep="first")

        keep = ["pathway"]
        for c in ["signor_pathway_id", "signor_pathway_name", "match_score", "pathway_id"]:
            if c in df.columns and c not in keep:
                keep.append(c)
        return df[keep].copy()
    except Exception as e:
        _warn(f"[signor] failed reading matches: {e}")
        return None

def summarize_signor_relations(out_root: Path, disease: str) -> Optional[pd.DataFrame]:
    p = out_root / "engines" / "signor_pathways" / disease / "signor_relations.tsv"
    if not p.exists():
        return None
    try:
        df = read_table_auto(p)
        if df.empty:
            return None

        df.columns = [str(c).strip() for c in df.columns]
        cols = {_norm_col(c): c for c in df.columns}
        pid = cols.get("pathway_id") or cols.get("signor_pathway_id") or cols.get("pathway")
        if pid is None:
            return None
        df[pid] = df[pid].astype(str).str.strip()
        df = df[df[pid] != ""]

        action_col = cols.get("action") or cols.get("effect") or cols.get("interaction")
        sign_col = cols.get("sign")

        src = cols.get("source") or cols.get("src") or cols.get("a")
        dst = cols.get("target") or cols.get("dst") or cols.get("b")

        g = df.groupby(pid, dropna=True)

        out = pd.DataFrame({
            "signor_pathway_id": [k for k in g.groups.keys()],
            "signor_edges_n": g.size().values,
        })

        if src and dst:
            nodes_n = []
            for k, sub in g:
                s = set(sub[src].astype(str).str.strip()) | set(sub[dst].astype(str).str.strip())
                s.discard("")
                nodes_n.append(len(s))
            out["signor_nodes_n"] = nodes_n

        if action_col:
            act = []
            inh = []
            other = []
            for k, sub in g:
                a = sub[action_col].astype(str).str.lower()
                act.append(int(a.str.contains("activ").sum()))
                inh.append(int(a.str.contains("inhib").sum()))
                other.append(int(len(a) - act[-1] - inh[-1]))
            out["signor_activation_n"] = act
            out["signor_inhibition_n"] = inh
            out["signor_other_action_n"] = other
        elif sign_col:
            s = _to_num(df[sign_col]).fillna(0.0)
            df["_sign_num"] = s
            g2 = df.groupby(pid, dropna=True)
            out["signor_activation_n"] = [int((sub["_sign_num"] > 0).sum()) for _, sub in g2]
            out["signor_inhibition_n"] = [int((sub["_sign_num"] < 0).sum()) for _, sub in g2]
            out["signor_other_action_n"] = [int((sub["_sign_num"] == 0).sum()) for _, sub in g2]
            df = df.drop(columns=["_sign_num"], errors="ignore")

        return out
    except Exception as e:
        _warn(f"[signor] failed reading relations: {e}")
        return None


# ----------------------------
# per-disease build
# ----------------------------

@dataclass
class DiseaseInputsUsed:
    disease: str
    selected_tissue: Optional[str]
    files_used: Dict[str, Optional[str]]


def build_pathway_summary_for_disease(out_root: Path, disease: str) -> Tuple[pd.DataFrame, DiseaseInputsUsed]:
    out_root = Path(out_root).resolve()

    stats_candidates = [
        out_root / "cohorts" / disease / "pathway_stats_with_baseline_filtered_classified.tsv",
        out_root / disease / "pathway_stats_with_baseline_filtered_classified.tsv",
        out_root / "cohorts" / disease / "pathway_stats_with_baseline_filtered.tsv",
        out_root / disease / "pathway_stats_with_baseline_filtered.tsv",
        out_root / "cohorts" / disease / "pathway_stats_with_baseline.tsv",
        out_root / disease / "pathway_stats_with_baseline.tsv",
        out_root / "cohorts" / disease / "pathway_stats.tsv",
        out_root / disease / "pathway_stats.tsv",
    ]
    stats_path = _first_existing(stats_candidates)

    files_used: Dict[str, Optional[str]] = {
        "stats": str(stats_path) if stats_path else None,
        "baseline_expectations": None,
        "gsea": None,
        "pathway_activity_matrix": None,
        "footprint_activity_matrix": None,
        "evidence_bundle": None,
        "signor_matches": None,
        "signor_relations": None,
    }

    if stats_path is None:
        evp = out_root / "engines" / "evidence_bundle" / disease / "pathways_evidence.tsv"
        if evp.exists():
            ev = read_table_auto(evp)
            ev = _ensure_key(ev, key_name="pathway")
            base = ev[["pathway"]].drop_duplicates().copy()
        else:
            base = pd.DataFrame(columns=["pathway"])
    else:
        stats = read_table_auto(stats_path)
        stats = _ensure_key(stats, key_name="pathway")
        base = stats[["pathway"]].drop_duplicates().copy()
        stats = _prefix_columns(stats, "stats_", keep=["pathway"])
        base = _merge_left(base, stats, on="pathway")

    tissue = load_selected_tissue(out_root, disease)
    if tissue:
        b = load_baseline_expectations(out_root, tissue)
        if b is not None and not b.empty:
            files_used["baseline_expectations"] = str(out_root / "baseline_consensus" / "baseline_pathway_expectations_c2cp.tsv")
            b = _prefix_columns(b, "baseline_", keep=["pathway"])
            base = _merge_left(base, b, on="pathway")

    gsea_candidates = [
        out_root / disease / "gsea_c2cp" / "gsea_prerank_results.tsv",
        out_root / disease / "gsea_prerank_results.tsv",
    ]
    gsea_path = _first_existing(gsea_candidates)
    if gsea_path:
        files_used["gsea"] = str(gsea_path)
        gsea = read_table_auto(gsea_path)
        gsea = _ensure_key(gsea, key_name="pathway")
        gsea = _prefix_columns(gsea, "gsea_", keep=["pathway"])
        base = _merge_left(base, gsea, on="pathway")

    ev_path = out_root / "engines" / "evidence_bundle" / disease / "pathways_evidence.tsv"
    if ev_path.exists():
        files_used["evidence_bundle"] = str(ev_path)
        ev = read_table_auto(ev_path)
        ev = _ensure_key(ev, key_name="pathway")
        ev = _prefix_columns(ev, "evidence_", keep=["pathway"])
        base = _merge_left(base, ev, on="pathway")

    labels = load_labels(out_root, disease)

    act_path = _first_existing([out_root / disease / "pathway_activity.tsv"])
    if act_path:
        files_used["pathway_activity_matrix"] = str(act_path)
        act = summarize_matrix_by_pathway(act_path, labels=labels)
        if act is not None and not act.empty:
            act = _prefix_columns(act, "activity_", keep=["pathway"])
            base = _merge_left(base, act, on="pathway")

    # footprint matrix (still merged into full pathway_summary, but NOT used in filtered_summary)
    fp_path = out_root / "engines" / "causal_pathway_features" / disease / "pathway_footprint_activity.tsv"
    if fp_path.exists():
        files_used["footprint_activity_matrix"] = str(fp_path)
        fp = summarize_matrix_by_pathway(fp_path, labels=labels)
        if fp is not None and not fp.empty:
            fp = _prefix_columns(fp, "footprint_", keep=["pathway"])
            base = _merge_left(base, fp, on="pathway")

    sm = load_signor_matches(out_root, disease)
    if sm is not None and not sm.empty:
        files_used["signor_matches"] = str(out_root / "engines" / "signor_pathways" / disease / "signor_pathway_matches.tsv")
        sm2 = _prefix_columns(sm, "signor_", keep=["pathway"])
        base = _merge_left(base, sm2, on="pathway")

    sr = summarize_signor_relations(out_root, disease)
    if sr is not None and not sr.empty:
        files_used["signor_relations"] = str(out_root / "engines" / "signor_pathways" / disease / "signor_relations.tsv")
        join_col = None
        for c in base.columns:
            if c.endswith("signor_pathway_id"):
                join_col = c
                break
        if join_col:
            tmp = sr.copy()
            tmp = tmp.rename(columns={"signor_pathway_id": join_col})
            tmp = _prefix_columns(tmp, "signorrel_", keep=[join_col])
            base = base.merge(tmp, on=join_col, how="left")

    base.insert(0, "disease", disease)
    base.insert(1, "selected_tissue", tissue if tissue else "")

    source_flags = []
    for _, _r in base.iterrows():
        present = []
        for k, v in files_used.items():
            if v:
                present.append(k)
        source_flags.append(",".join(present))
    base["sources_present"] = source_flags

    inputs_used = DiseaseInputsUsed(
        disease=disease,
        selected_tissue=tissue,
        files_used=files_used,
    )
    return base, inputs_used


# ----------------------------
# ranking helpers
# ----------------------------

def choose_best_fdr_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic: pick a column containing 'fdr' or 'q' that has numeric values.
    Prefer stats_* over gsea_* over evidence_*.
    """
    candidates = []
    for c in df.columns:
        cn = _norm_col(c)
        if any(x in cn for x in ["fdr", "qval", "qvalue", "padj", "adjp"]):
            candidates.append(c)

    def prio(c: str) -> int:
        cn = c.lower()
        if cn.startswith("stats_"):
            return 0
        if cn.startswith("gsea_"):
            return 1
        if "evidence_" in cn:
            return 2
        return 3

    candidates = sorted(candidates, key=prio)
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.05 * len(df))):
            return c
    return None

def choose_best_pval_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic: pick a column containing 'pval' / 'pvalue' that has numeric values.
    Prefer stats_* over gsea_* over evidence_*.
    """
    candidates = []
    for c in df.columns:
        cn = _norm_col(c)
        if any(x in cn for x in ["pval", "pvalue", "p_value"]):
            candidates.append(c)

    def prio(c: str) -> int:
        cn = c.lower()
        if cn.startswith("stats_"):
            return 0
        if cn.startswith("gsea_"):
            return 1
        if "evidence_" in cn:
            return 2
        return 3

    candidates = sorted(candidates, key=prio)
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.05 * len(df))):
            return c
    return None


# ----------------------------
# filtered summary builder
# ----------------------------

def _find_col_by_norm(df: pd.DataFrame, norm_keys: List[str]) -> Optional[str]:
    """Return first df column whose normalized name matches any key in norm_keys."""
    if df is None or df.empty:
        return None
    norm_map = {_norm_col(c): c for c in df.columns}
    for k in norm_keys:
        if k in norm_map:
            return norm_map[k]
    return None

def _find_first_col_containing(
    df: pd.DataFrame,
    must_contain: List[str],
    *,
    prefix: Optional[str] = None,
    avoid: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Return first column where normalized name contains all substrings in must_contain.
    Optional prefix restricts to columns starting with that prefix (case-insensitive).
    Optional avoid excludes columns containing any avoid substring (normalized).
    """
    if df is None or df.empty:
        return None
    avoid = avoid or []
    for c in df.columns:
        cl = str(c).lower()
        if prefix is not None and not cl.startswith(prefix.lower()):
            continue
        cn = _norm_col(c)
        if any(a in cn for a in avoid):
            continue
        ok = True
        for s in must_contain:
            if s not in cn:
                ok = False
                break
        if ok:
            return c
    return None

def _numeric_coverage(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    if len(x) == 0:
        return 0.0
    return float(x.notna().mean())

def _choose_best_metric_column(df: pd.DataFrame, keys_any: List[str], prefer_prefix: List[str]) -> Optional[str]:
    """
    General selector for "NES", "ES", "score", etc.
    keys_any: list of substrings to look for in normalized column name (any match).
    prefer_prefix: ordered list of prefixes to prioritize (e.g. ["stats_", "gsea_", "evidence_"])
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)

    def has_key(c: str) -> bool:
        cn = _norm_col(c)
        return any(k in cn for k in keys_any)

    candidates = [c for c in cols if has_key(c)]

    # Filter out some obvious false positives
    # (e.g., "genes" columns that might include "nes" as part of another string is rare, but be safe)
    candidates = [c for c in candidates if "genes" not in _norm_col(c)]

    if not candidates:
        return None

    def prio(c: str) -> int:
        cl = c.lower()
        for i, pfx in enumerate(prefer_prefix):
            if cl.startswith(pfx.lower()):
                return i
        return len(prefer_prefix)

    candidates = sorted(candidates, key=prio)
    for c in candidates:
        if _numeric_coverage(df[c]) >= 0.05:
            return c
    return None

def _parse_gene_list(val: object) -> List[str]:
    """
    Best-effort parsing of lead/edge genes.
    Accepts strings like: "A,B,C" or "A; B; C" or "[A, B]".
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    # strip common wrappers
    s = s.strip("[](){}")
    # normalize separators to comma
    s = re.sub(r"[;\t|]+", ",", s)
    # split on comma or whitespace runs
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # if chunk still has spaces, treat as multiple genes only if it looks like a list
        if " " in chunk and not re.search(r"^[A-Za-z0-9_.-]+$", chunk):
            parts.extend([x for x in re.split(r"\s+", chunk) if x])
        else:
            parts.append(chunk)
    # de-dup keep order
    out = []
    seen = set()
    for g in parts:
        g2 = g.strip()
        if not g2:
            continue
        if g2 not in seen:
            seen.add(g2)
            out.append(g2)
    return out

def build_filtered_summary(df: pd.DataFrame, *, fdr_threshold: float = _DEFAULT_FILTER_FDR) -> pd.DataFrame:
    """
    Build filtered_summary.tsv (pipeline-friendly, stable schema):

      - ONLY significant pathways:
          prefer FDR <= threshold (best FDR col),
          else fallback pvalue <= threshold (best pvalue col),
          else returns empty with standard headers.

      - IMPORTANT columns only (no footprint_*):
          disease, selected_tissue, pathway,
          fdr, pvalue,
          nes (best available stats_/gsea_),
          baseline_expectation, delta_vs_baseline (derived if possible),
          activity_mean, activity_sd, activity_mean_a/mean_b + activity_delta (if present),
          n_genes (best available baseline/stats/evidence),
          lead_genes (leading edge / edge genes / top genes best-effort),
          lead_genes_n (derived),
          lead_genes_ratio (existing if present, else derived from n_genes),
          a minimal stats context: direction/class/category/classification/effect/tstat/score if present.

    This function does NOT modify other outputs.
    """
    # stable headers even if empty
    stable_headers = [
        "disease",
        "selected_tissue",
        "pathway",
        "fdr",
        "pvalue",
        "nes",
        "baseline_expectation",
        "delta_vs_baseline",
        "activity_mean",
        "activity_sd",
        "activity_label_a",
        "activity_label_b",
        "activity_mean_a",
        "activity_mean_b",
        "activity_delta",
        "n_genes",
        "lead_genes",
        "lead_genes_n",
        "lead_genes_ratio",
        "direction",
        "classification",
        "category",
        "effect",
        "tstat",
        "score",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=stable_headers)

    fdr_col = choose_best_fdr_column(df)
    pval_col = choose_best_pval_column(df)

    # determine significance mask
    sig_mask = None
    if fdr_col:
        f = pd.to_numeric(df[fdr_col], errors="coerce")
        sig_mask = (f.notna()) & (f <= float(fdr_threshold))
    elif pval_col:
        p = pd.to_numeric(df[pval_col], errors="coerce")
        sig_mask = (p.notna()) & (p <= float(fdr_threshold))
    else:
        # cannot determine significance robustly
        _warn("[filtered_summary] no FDR/pvalue column found; writing empty filtered_summary.tsv")
        return pd.DataFrame(columns=stable_headers)

    sub = df.loc[sig_mask].copy()
    if sub.empty:
        # fallback: if stats_FDR yields nothing, try GSEA FDR
        if fdr_col and _norm_col(fdr_col).startswith("stats_"):
            # prefer any GSEA FDR-like column (FDR/q-val/qvalue)
            gsea_fdr_col = (
                _find_first_col_containing(df, ["fdr"], prefix="gsea_")
                or _find_first_col_containing(df, ["qval"], prefix="gsea_")
                or _find_first_col_containing(df, ["qvalue"], prefix="gsea_")
            )
            if gsea_fdr_col:
                f2 = pd.to_numeric(df[gsea_fdr_col], errors="coerce")
                sig_mask2 = (f2.notna()) & (f2 <= float(fdr_threshold))
                sub = df.loc[sig_mask2].copy()
        if sub.empty:
            return pd.DataFrame(columns=stable_headers)

    # -------- select source columns robustly --------

    # baseline expectation + n_genes
    base_exp_col = _find_col_by_norm(sub, ["baseline_expectation", "baselineexpectation", "baseline_expect"])
    n_genes_col = _choose_best_metric_column(
        sub,
        keys_any=["n_genes", "ngenes", "geneset_size", "setsize", "genes"],
        prefer_prefix=["baseline_", "stats_", "evidence_", "gsea_"],
    )

    # activity (no footprint)
    activity_mean_col = _find_col_by_norm(sub, ["activity_mean"])
    activity_sd_col = _find_col_by_norm(sub, ["activity_sd"])
    activity_label_a_col = _find_col_by_norm(sub, ["activity_label_a"])
    activity_label_b_col = _find_col_by_norm(sub, ["activity_label_b"])
    activity_mean_a_col = _find_col_by_norm(sub, ["activity_mean_a"])
    activity_mean_b_col = _find_col_by_norm(sub, ["activity_mean_b"])
    activity_delta_col = _find_col_by_norm(sub, ["activity_delta_b_minus_a", "activity_deltabminusa"])

    # significance / NES
    nes_col = _choose_best_metric_column(
        sub,
        keys_any=["nes"],
        prefer_prefix=["stats_", "gsea_", "evidence_"],
    )
    # some tables use "normalizedenrichmentscore"
    if nes_col is None:
        nes_col = _find_first_col_containing(sub, ["normalized", "enrichment", "score"], prefix="gsea_")

    # delta vs baseline: prefer any explicit stats delta, else derive activity_mean - baseline_expectation
    delta_vs_base_col = (
        _find_first_col_containing(sub, ["delta", "baseline"], prefix="stats_")
        or _find_first_col_containing(sub, ["delta", "expect"], prefix="stats_")
        or _find_first_col_containing(sub, ["delta"], prefix="stats_", avoid=["b_minus_a", "bminusa"])
    )

    # lead/edge genes: prefer GSEA leading edge, else evidence-derived, else stats
    lead_genes_col = (
        _find_first_col_containing(sub, ["leading", "edge"], prefix="gsea_")
        or _find_first_col_containing(sub, ["edge", "gene"], prefix="gsea_")
        or _find_first_col_containing(sub, ["lead", "gene"], prefix="gsea_")
        or _find_first_col_containing(sub, ["lead", "gene"], prefix="evidence_")
        or _find_first_col_containing(sub, ["edge", "gene"], prefix="evidence_")
        or _find_first_col_containing(sub, ["top", "gene"], prefix="evidence_")
        or _find_first_col_containing(sub, ["lead", "gene"], prefix="stats_")
    )

    # an existing ratio column (if pipeline already computed it)
    lead_ratio_col = (
        _find_first_col_containing(sub, ["lead", "ratio"], prefix="gsea_")
        or _find_first_col_containing(sub, ["lead", "ratio"], prefix="evidence_")
        or _find_first_col_containing(sub, ["lead", "ratio"], prefix="stats_")
        or _find_first_col_containing(sub, ["leading", "ratio"], prefix="gsea_")
        or _find_first_col_containing(sub, ["edge", "ratio"], prefix="gsea_")
    )

    # minimal stats context
    direction_col = (
        _find_col_by_norm(sub, ["stats_direction"])
        or _find_first_col_containing(sub, ["direction"], prefix="stats_")
    )
    classification_col = (
        _find_col_by_norm(sub, ["stats_classification", "stats_class", "stats_classified"])
        or _find_first_col_containing(sub, ["class"], prefix="stats_")
    )
    category_col = (
        _find_col_by_norm(sub, ["stats_category"])
        or _find_first_col_containing(sub, ["category"], prefix="stats_")
    )
    effect_col = (
        _find_col_by_norm(sub, ["stats_effect"])
        or _find_first_col_containing(sub, ["effect"], prefix="stats_")
    )
    tstat_col = (
        _find_col_by_norm(sub, ["stats_tstat", "stats_t"])
        or _find_first_col_containing(sub, ["tstat"], prefix="stats_")
        or _find_first_col_containing(sub, ["t"], prefix="stats_", avoid=["tissue"])
    )
    score_col = _choose_best_metric_column(
        sub,
        keys_any=["score", "stat", "z", "effectsize"],
        prefer_prefix=["stats_", "gsea_", "evidence_"],
    )

    # -------- construct output with stable schema --------

    out = pd.DataFrame(index=sub.index)

    def _col_or_nan(name: str, src: Optional[str]) -> None:
        if src and src in sub.columns:
            out[name] = sub[src]
        else:
            out[name] = np.nan

    # ids
    out["disease"] = sub["disease"] if "disease" in sub.columns else ""
    out["selected_tissue"] = sub["selected_tissue"] if "selected_tissue" in sub.columns else ""
    out["pathway"] = sub["pathway"] if "pathway" in sub.columns else ""

    # significance
    out["fdr"] = pd.to_numeric(sub[fdr_col], errors="coerce") if fdr_col and fdr_col in sub.columns else np.nan
    out["pvalue"] = pd.to_numeric(sub[pval_col], errors="coerce") if pval_col and pval_col in sub.columns else np.nan

    # metrics
    out["nes"] = pd.to_numeric(sub[nes_col], errors="coerce") if nes_col and nes_col in sub.columns else np.nan

    # baseline expectation
    out["baseline_expectation"] = pd.to_numeric(sub[base_exp_col], errors="coerce") if base_exp_col and base_exp_col in sub.columns else np.nan

    # delta vs baseline
    if delta_vs_base_col and delta_vs_base_col in sub.columns:
        out["delta_vs_baseline"] = pd.to_numeric(sub[delta_vs_base_col], errors="coerce")
    else:
        # derive if possible: activity_mean - baseline_expectation
        if activity_mean_col and base_exp_col and activity_mean_col in sub.columns and base_exp_col in sub.columns:
            out["delta_vs_baseline"] = (
                pd.to_numeric(sub[activity_mean_col], errors="coerce")
                - pd.to_numeric(sub[base_exp_col], errors="coerce")
            )
        else:
            out["delta_vs_baseline"] = np.nan

    # activity (no footprint columns)
    out["activity_mean"] = pd.to_numeric(sub[activity_mean_col], errors="coerce") if activity_mean_col and activity_mean_col in sub.columns else np.nan
    out["activity_sd"] = pd.to_numeric(sub[activity_sd_col], errors="coerce") if activity_sd_col and activity_sd_col in sub.columns else np.nan
    out["activity_label_a"] = sub[activity_label_a_col] if activity_label_a_col and activity_label_a_col in sub.columns else ""
    out["activity_label_b"] = sub[activity_label_b_col] if activity_label_b_col and activity_label_b_col in sub.columns else ""
    out["activity_mean_a"] = pd.to_numeric(sub[activity_mean_a_col], errors="coerce") if activity_mean_a_col and activity_mean_a_col in sub.columns else np.nan
    out["activity_mean_b"] = pd.to_numeric(sub[activity_mean_b_col], errors="coerce") if activity_mean_b_col and activity_mean_b_col in sub.columns else np.nan
    out["activity_delta"] = pd.to_numeric(sub[activity_delta_col], errors="coerce") if activity_delta_col and activity_delta_col in sub.columns else np.nan

    # n_genes
    if n_genes_col and n_genes_col in sub.columns:
        out["n_genes"] = pd.to_numeric(sub[n_genes_col], errors="coerce")
    else:
        out["n_genes"] = np.nan

    # lead genes + derived count/ratio
    if lead_genes_col and lead_genes_col in sub.columns:
        out["lead_genes"] = sub[lead_genes_col].astype(str)
    else:
        out["lead_genes"] = ""

    lead_counts = []
    for v in out["lead_genes"].values:
        lead_counts.append(len(_parse_gene_list(v)))
    out["lead_genes_n"] = np.array(lead_counts, dtype=float)

    if lead_ratio_col and lead_ratio_col in sub.columns:
        out["lead_genes_ratio"] = pd.to_numeric(sub[lead_ratio_col], errors="coerce")
    else:
        # derive ratio if n_genes is available
        ng = pd.to_numeric(out["n_genes"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["lead_genes_ratio"] = np.where((ng.notna()) & (ng > 0), out["lead_genes_n"] / ng, np.nan)

    # minimal stats context (renamed to stable names)
    out["direction"] = sub[direction_col] if direction_col and direction_col in sub.columns else ""
    out["classification"] = sub[classification_col] if classification_col and classification_col in sub.columns else ""
    out["category"] = sub[category_col] if category_col and category_col in sub.columns else ""
    out["effect"] = pd.to_numeric(sub[effect_col], errors="coerce") if effect_col and effect_col in sub.columns else np.nan
    out["tstat"] = pd.to_numeric(sub[tstat_col], errors="coerce") if tstat_col and tstat_col in sub.columns else np.nan
    out["score"] = pd.to_numeric(sub[score_col], errors="coerce") if score_col and score_col in sub.columns else np.nan

    # enforce column order and drop any extras (no surprise extra columns)
    out = out[stable_headers].copy()

    # final: sort by fdr then pvalue where available (stable for pipeline)
    if out["fdr"].notna().any():
        out = out.sort_values(["fdr", "pvalue"], ascending=[True, True], na_position="last")
    elif out["pvalue"].notna().any():
        out = out.sort_values(["pvalue"], ascending=[True], na_position="last")

    return out.reset_index(drop=True)


# ----------------------------
# Config-based entry (used by aggregation_service)
# ----------------------------


def _config_to_args(cfg) -> "argparse.Namespace":
    """Build args-like object from AggregationPhaseConfig or dict."""
    import argparse
    def _g(k, default):
        v = getattr(cfg, k, None) if hasattr(cfg, k) else (cfg.get(k) if isinstance(cfg, dict) else None)
        return v if v is not None else default
    diseases_raw = _g("diseases", None)
    diseases_str = ",".join(diseases_raw) if isinstance(diseases_raw, list) else (str(diseases_raw) if diseases_raw else "")
    return argparse.Namespace(
        out_root=_g("out_root", ""),
        diseases=diseases_str,
        out_subdir=_g("out_subdir", "engines/pathway_summary"),
    )


def run_aggregation_from_config(config) -> int:
    """Run aggregator from config. No argparse. Returns exit code."""
    args = _config_to_args(config)
    return _main_impl(args)


# ----------------------------
# CLI main
# ----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="Pipeline output root (e.g. /mnt/d/temp/CUASALITY_5)")
    ap.add_argument("--diseases", default="", help="Comma-separated diseases. If empty, auto-discover under cohorts/")
    ap.add_argument("--out-subdir", default="engines/pathway_summary", help="Where to write summaries under out_root")
    args = ap.parse_args()
    return _main_impl(args)


def _main_impl(args) -> int:
    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        _warn(f"[error] out_root not found: {out_root}")
        return 2

    if args.diseases.strip():
        diseases = [d.strip() for d in args.diseases.split(",") if d.strip()]
    else:
        diseases = discover_diseases(out_root)

    if not diseases:
        _warn("[error] no diseases found (expected cohorts/<disease>/...)")
        return 3

    out_base = out_root / args.out_subdir
    _mkdir(out_base)

    all_rows: List[pd.DataFrame] = []
    manifest_all = {
        "out_root": str(out_root),
        "out_subdir": str(out_base),
        "diseases": [],
    }

    for disease in diseases:
        df, used = build_pathway_summary_for_disease(out_root, disease)

        ddir = out_base / disease
        _mkdir(ddir)

        out_tsv = ddir / "pathway_summary.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        _info(f"[ok] wrote: {out_tsv}")

        # ranked (unchanged)
        rank_col = choose_best_fdr_column(df)
        ranked = df.copy()
        if rank_col:
            ranked["_rank_fdr"] = pd.to_numeric(ranked[rank_col], errors="coerce")
            ranked = ranked.sort_values(["_rank_fdr"], ascending=[True])
            ranked = ranked.drop(columns=["_rank_fdr"], errors="ignore")
        else:
            score_cols = [c for c in df.columns if any(x in _norm_col(c) for x in ["nes", "score", "tstat", "effect", "delta"])]
            if score_cols:
                c = score_cols[0]
                ranked["_rank_score"] = pd.to_numeric(ranked[c], errors="coerce").abs()
                ranked = ranked.sort_values(["_rank_score"], ascending=[False])
                ranked = ranked.drop(columns=["_rank_score"], errors="ignore")

        out_rank = ddir / "pathway_summary_ranked.tsv"
        ranked.to_csv(out_rank, sep="\t", index=False)
        _info(f"[ok] wrote: {out_rank}")

        # filtered summary (UPDATED: includes NES + edge/lead genes; excludes footprint cols; stable schema)
        filtered = build_filtered_summary(df, fdr_threshold=_DEFAULT_FILTER_FDR)
        out_filt = ddir / "filtered_summary.tsv"
        filtered.to_csv(out_filt, sep="\t", index=False)
        _info(f"[ok] wrote: {out_filt}")

        # manifest per disease (unchanged)
        man = {
            "disease": used.disease,
            "selected_tissue": used.selected_tissue,
            "files_used": used.files_used,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
        }
        (ddir / "PATHWAY_SUMMARY_MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        _info(f"[ok] wrote: {ddir / 'PATHWAY_SUMMARY_MANIFEST.json'}")

        manifest_all["diseases"].append(man)
        all_rows.append(df)

    # combined across diseases (unchanged)
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_all = out_base / "all_diseases_pathway_summary.tsv"
        combined.to_csv(out_all, sep="\t", index=False)
        _info(f"[ok] wrote: {out_all}")

    (out_base / "PATHWAY_SUMMARY_ALL_MANIFEST.json").write_text(json.dumps(manifest_all, indent=2), encoding="utf-8")
    _info(f"[ok] wrote: {out_base / 'PATHWAY_SUMMARY_ALL_MANIFEST.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
