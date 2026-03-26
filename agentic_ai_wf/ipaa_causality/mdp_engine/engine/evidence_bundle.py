# mdp_engine/engines/evidence_bundle.py
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from scipy.stats import ttest_ind

from ..exceptions import DataError, ValidationError
from ..logging_utils import get_logger

log = get_logger("mdp_engine.evidence_bundle")


# ----------------------------
# Outputs
# ----------------------------
@dataclass(frozen=True)
class EvidenceBundlePaths:
    out_dir: Path
    genes_evidence: Path
    pathways_evidence: Path
    regulators_evidence: Path
    pathway_combined_evidence: Path
    all_evidence_long: Path
    mechanism_summary: Path
    manifest: Path


# ----------------------------
# Small helpers
# ----------------------------
def _cohort_dir(out_root: Path, disease: str) -> Path:
    d1 = out_root / disease
    d2 = out_root / "cohorts" / disease
    if d1.exists():
        return d1
    if d2.exists():
        return d2
    raise DataError(f"Cannot find cohort folder for '{disease}' in {out_root} (tried {d1} and {d2})")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise DataError(f"Missing file: {path}")
    suf = path.suffix.lower()
    try:
        if suf in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suf == ".csv":
            return pd.read_csv(path)
        # fallback: try tab then csv
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Failed reading table: {path} ({e})") from e


def _read_indexed_table(path: Path) -> pd.DataFrame:
    """Read table with index_col=0 (typical for activity matrices)."""
    if not path.exists() or not path.is_file():
        raise DataError(f"Missing file: {path}")
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep, index_col=0)
    except Exception as e:
        raise DataError(f"Failed reading indexed table: {path} ({e})") from e
    if df is None or df.empty:
        raise DataError(f"Empty table: {path}")
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _read_indexed_header_only(path: Path) -> Tuple[List[str], str]:
    """
    Read only the header of an indexed table (index_col=0) and return:
      - columns list
      - sep used ('\\t' or ',')
    This is used for expressed_flag without loading full expression matrices.
    """
    if not path.exists() or not path.is_file():
        raise DataError(f"Missing file: {path}")
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    try:
        df0 = pd.read_csv(path, sep=sep, index_col=0, nrows=0)
    except Exception:
        # fallback try the other separator
        sep2 = "," if sep == "\t" else "\t"
        df0 = pd.read_csv(path, sep=sep2, index_col=0, nrows=0)
        sep = sep2
    cols = [str(c).strip() for c in df0.columns.tolist()]
    return cols, sep


def _first_existing(base: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists() and p.is_file():
            return p
    return None


def _find_any_matching(base: Path, contains_any: Iterable[str]) -> Optional[Path]:
    if not base.exists() or not base.is_dir():
        return None
    keys = [str(x).lower() for x in contains_any]
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        n = p.name.lower()
        if any(k in n for k in keys):
            return p
    return None


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _as_str(x: Any) -> str:
    return str(x).strip()


def _is_nan(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _parse_list_str(x: Any) -> List[str]:
    """Parse gene/entity list strings like 'A;B;C' or 'A/B/C'."""
    if x is None or _is_nan(x):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    # common separators
    for sep in [";", "/", "|", ","]:
        if sep in s:
            parts = [p.strip() for p in s.replace("\n", " ").split(sep)]
            parts = [p for p in parts if p and p.lower() != "nan"]
            # de-dup keep order
            seen: Set[str] = set()
            out: List[str] = []
            for p in parts:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            return out
    # space separated (rare)
    if " " in s and len(s.split()) > 3:
        parts = [p.strip() for p in s.split() if p.strip()]
        return list(dict.fromkeys(parts))
    return [s]


def _direction_from_score(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "NA"
    if v > 0:
        return "UP"
    if v < 0:
        return "DOWN"
    return "ZERO"


# ----------------------------
# Overlap JSON parsing (robust)
# ----------------------------
def _walk_objs(obj: Any) -> Iterable[Any]:
    """Yield all nested dict/list nodes (including obj itself)."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_objs(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_objs(it)


def _normalize_entity_type(t: Any) -> str:
    s = str(t or "").strip().lower()
    if not s:
        return "unknown"
    # normalize a few common variants
    if s in {"tf", "tfs", "transcription_factor", "transcription factors"}:
        return "TF"
    if s in {"epigenetic", "epigenetics", "epimark", "histone", "methylation"}:
        return "epigenetic"
    if s in {"metabolite", "metabolites", "compound"}:
        return "metabolite"
    return str(t).strip()


def parse_overlap_edges(overlap_json: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Best-effort parser for many overlap JSON schemas.
    Returns edges DataFrame with columns:
      pathway, entity_type, entity, score
    """
    warnings: List[str] = []
    if overlap_json is None or not Path(overlap_json).exists():
        return pd.DataFrame(columns=["pathway", "entity_type", "entity", "score"]), ["overlap_json not found"]

    try:
        obj = json.loads(Path(overlap_json).read_text(encoding="utf-8"))
    except Exception as e:
        return pd.DataFrame(columns=["pathway", "entity_type", "entity", "score"]), [f"Failed parsing overlap JSON ({e})"]

    rows: List[Dict[str, Any]] = []

    # Pattern 1: {"pathways": {PW: {"entities": {"tf":[{name,score},...], ...}}}}
    if isinstance(obj, dict):
        pws = obj.get("pathways") or obj.get("Pathways") or obj.get("pathway_to_entities")
        if isinstance(pws, dict):
            for pw, pw_obj in pws.items():
                if not isinstance(pw_obj, dict):
                    continue
                ents = pw_obj.get("entities") or pw_obj.get("Entities") or pw_obj.get("drivers") or pw_obj.get("regulators")
                if not isinstance(ents, dict):
                    continue
                for et, et_val in ents.items():
                    et_norm = _normalize_entity_type(et)
                    if isinstance(et_val, dict):
                        # mapping entity -> score
                        for en, sc in et_val.items():
                            if en is None:
                                continue
                            rows.append({"pathway": _as_str(pw), "entity_type": et_norm, "entity": _as_str(en), "score": sc})
                    elif isinstance(et_val, list):
                        for it in et_val:
                            if isinstance(it, dict):
                                en = it.get("name") or it.get("entity") or it.get("id") or it.get("target")
                                sc = it.get("score") or it.get("weight") or it.get("value")
                                if en:
                                    rows.append({"pathway": _as_str(pw), "entity_type": et_norm, "entity": _as_str(en), "score": sc})
                            elif isinstance(it, str):
                                rows.append({"pathway": _as_str(pw), "entity_type": et_norm, "entity": it.strip(), "score": None})

    # Pattern 2: list edges under common keys
    if isinstance(obj, dict):
        for key in ["pathway_entity", "edges", "links", "relationships", "items"]:
            val = obj.get(key)
            if isinstance(val, list):
                for it in val:
                    if not isinstance(it, dict):
                        continue
                    pw = it.get("pathway") or it.get("source") or it.get("Pathway") or it.get("Term")
                    en = it.get("entity") or it.get("target") or it.get("name") or it.get("Entity")
                    et = it.get("entity_type") or it.get("type") or it.get("category") or it.get("layer")
                    sc = it.get("score") or it.get("weight") or it.get("value")
                    if pw and en:
                        rows.append(
                            {
                                "pathway": _as_str(pw),
                                "entity_type": _normalize_entity_type(et),
                                "entity": _as_str(en),
                                "score": sc,
                            }
                        )

    # Pattern 3: brute-force scan for dicts that look like an edge
    for node in _walk_objs(obj):
        if not isinstance(node, dict):
            continue

        # "edge-like" if it has a pathway-ish key and entity-ish key
        pw = node.get("pathway") or node.get("Pathway") or node.get("Term") or node.get("term") or node.get("source")
        en = node.get("entity") or node.get("Entity") or node.get("target") or node.get("name")
        et = node.get("entity_type") or node.get("EntityType") or node.get("type") or node.get("category") or node.get("layer")
        if pw and en and et:
            sc = node.get("score") or node.get("weight") or node.get("value")
            rows.append({"pathway": _as_str(pw), "entity_type": _normalize_entity_type(et), "entity": _as_str(en), "score": sc})

    if not rows:
        warnings.append("Could not extract overlap edges (unknown schema).")
        return pd.DataFrame(columns=["pathway", "entity_type", "entity", "score"]), warnings

    df = pd.DataFrame(rows)
    df["pathway"] = df["pathway"].astype(str).str.strip()
    df["entity_type"] = df["entity_type"].astype(str).str.strip()
    df["entity"] = df["entity"].astype(str).str.strip()

    # numeric score if possible
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # drop empties
    df = df[(df["pathway"] != "") & (df["entity"] != "")]
    df = df.drop_duplicates(subset=["pathway", "entity_type", "entity"], keep="first").reset_index(drop=True)
    return df, warnings


# ----------------------------
# Genes evidence normalization (+ direction + expressed_flag)
# ----------------------------
def normalize_genes_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_cols(df)
    rename: Dict[str, str] = {}

    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in {"gene", "genes", "symbol", "hgnc", "gene_symbol"}:
            rename[c] = "gene"
        elif lc in {"log2fc", "logfc", "lfc", "log2foldchange"}:
            rename[c] = "log2fc"
        elif lc in {"stat", "t", "t_stat", "tlike", "t_like", "wald_stat", "score"}:
            rename[c] = "stat"
        elif lc in {"p", "pval", "p_value", "pvalue", "p-value"}:
            rename[c] = "p_value"
        elif lc in {"fdr", "q", "q_value", "qvalue", "padj", "adj_p"}:
            rename[c] = "fdr"

    out = df.rename(columns=rename).copy()

    if "gene" not in out.columns:
        out = out.rename(columns={out.columns[0]: "gene"})

    out["gene"] = out["gene"].astype(str).str.strip()
    out = out[out["gene"] != ""].copy()

    for c in ["log2fc", "stat", "p_value", "fdr"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # define a single score column
    if "score" not in out.columns:
        if "stat" in out.columns:
            out["score"] = out["stat"]
        elif "log2fc" in out.columns:
            out["score"] = out["log2fc"]
        else:
            out["score"] = pd.NA

    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["__abs__"] = out["score"].abs()
    out = out.sort_values("__abs__", ascending=False, na_position="last").drop_duplicates("gene", keep="first")
    out = out.drop(columns=["__abs__"]).reset_index(drop=True)
    return out


def _add_gene_direction_and_expressed_flag(
    genes_df: pd.DataFrame,
    expressed_genes_upper: Optional[Set[str]],
) -> pd.DataFrame:
    """
    Additive-only: adds direction and expressed_flag if not present.
    expressed_flag is 1 if the gene is present in expression_used columns; else 0 (or NA if expression not available).
    """
    out = genes_df.copy()
    if "direction" not in out.columns:
        out["direction"] = out["score"].map(_direction_from_score) if "score" in out.columns else "NA"

    if "expressed_flag" not in out.columns:
        if expressed_genes_upper is None:
            out["expressed_flag"] = pd.NA
        else:
            g_upper = out["gene"].astype(str).str.upper().str.strip()
            out["expressed_flag"] = g_upper.isin(expressed_genes_upper).astype(int)

    return out


def _load_expressed_gene_set_from_expression_used(cohort_dir: Path) -> Tuple[Optional[Set[str]], List[str]]:
    """
    Best-effort: load *only* the gene column names from expression_used.* (no full matrix load).
    Returns (set_of_upper_genes or None, warnings).
    """
    warnings: List[str] = []
    p = _first_existing(
        cohort_dir,
        ["expression_used.tsv", "expression_used.txt", "expression_used.csv", "expr_used.tsv", "expr_used.csv"],
    )
    if p is None:
        return None, ["expression_used not found; expressed_flag will be NA"]

    try:
        cols, _ = _read_indexed_header_only(p)
        expressed = {str(c).strip().upper() for c in cols if str(c).strip()}
        if not expressed:
            warnings.append("expression_used header read but no gene columns found; expressed_flag will be NA")
            return None, warnings
        return expressed, warnings
    except Exception as e:
        warnings.append(f"Failed reading expression_used header: {p} ({e}); expressed_flag will be NA")
        return None, warnings


# ----------------------------
# Regulators evidence (TF deltas + n_pathways from overlap)
# ----------------------------
def _try_scipy_ttest(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    xa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    xb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)

    try:
        

        res = ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
        t = float(res.statistic) if res.statistic is not None and not math.isnan(float(res.statistic)) else 0.0
        p = float(res.pvalue) if res.pvalue is not None and not math.isnan(float(res.pvalue)) else 1.0
        return t, p
    except Exception:
        # fallback: crude t-like, p=1
        ma = float(pd.Series(xa).mean(skipna=True)) if xa.size else 0.0
        mb = float(pd.Series(xb).mean(skipna=True)) if xb.size else 0.0
        return float(ma - mb), 1.0


def _bh_fdr(pvals: List[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    idx = sorted(range(n), key=lambda i: float(pvals[i]))
    q = [0.0] * n
    for rank, i in enumerate(idx, start=1):
        p = float(pvals[i])
        if math.isnan(p) or p < 0 or p > 1:
            p = 1.0
        q[i] = (p * n) / rank
    # monotone
    prev = 1.0
    for i in reversed(idx):
        prev = min(prev, q[i])
        q[i] = prev
    return [min(max(float(v), 0.0), 1.0) for v in q]


def _load_labels(cohort_dir: Path) -> Optional[pd.Series]:
    """
    Robust label loader.

    Supports:
      - index-based single-column labels (common)
      - two-column sample_id/label tables (very common)
      - tables with explicit columns: sample/sample_id + label/group/condition

    Returns a Series: index=sample_id (str), value=label (str)
    """
    p = _first_existing(cohort_dir, ["labels_used.tsv", "labels_used.csv", "labels_used.txt"])
    if p is None:
        return None

    df = _read_table(p)
    df = _clean_cols(df)

    if df is None or df.empty:
        return None

    # Case 1: single column => use index as sample IDs
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        s.index = df.index.astype(str)
        s = s[~s.isna()]
        s.index = s.index.astype(str).str.strip()
        s = s.astype(str).str.strip()
        return s

    # Identify possible sample and label columns
    col_lc = {str(c).lower().strip(): c for c in df.columns}

    sample_col = None
    for key in ["sample", "sample_id", "sampleid", "id", "barcode", "run", "name"]:
        if key in col_lc:
            sample_col = col_lc[key]
            break

    label_col = None
    for key in ["label", "group", "condition", "phenotype", "class"]:
        if key in col_lc:
            label_col = col_lc[key]
            break

    # Case 2: explicit sample + label columns
    if sample_col is not None and label_col is not None:
        s = df[label_col].astype(str)
        idx = df[sample_col].astype(str)
        s.index = idx
        s.index = s.index.astype(str).str.strip()
        s = s.astype(str).str.strip()
        s = s[~s.index.isna()]
        return s

    # Case 3: two columns, assume first is sample and second is label (typical)
    if df.shape[1] >= 2:
        idx = df.iloc[:, 0].astype(str)
        s = df.iloc[:, 1].astype(str)
        s.index = idx
        s.index = s.index.astype(str).str.strip()
        s = s.astype(str).str.strip()
        s = s[~s.index.isna()]
        # guard: if labels look like sample IDs (too many uniques), fallback to column 0 only
        if s.nunique(dropna=True) >= max(3, int(0.9 * len(s))):
            # try opposite (maybe first col is label)
            idx2 = df.iloc[:, 1].astype(str)
            s2 = df.iloc[:, 0].astype(str)
            s2.index = idx2
            s2.index = s2.index.astype(str).str.strip()
            s2 = s2.astype(str).str.strip()
            if s2.nunique(dropna=True) < s.nunique(dropna=True):
                return s2
        return s

    # Fallback: return first column with index-based sample IDs
    s = df.iloc[:, 0].astype(str)
    s.index = df.index.astype(str)
    s.index = s.index.astype(str).str.strip()
    s = s.astype(str).str.strip()
    return s


def _load_engine1_provenance(out_root: Path, disease: str) -> Dict[str, Any]:
    """
    Best-effort: read Engine1 provenance to populate regulators evidence columns like method/regulon_source.
    Does not raise; returns {} on failure.
    """
    prov = out_root / "engines" / "causal_pathway_features" / disease / "feature_provenance.json"
    if not prov.exists():
        return {}
    try:
        return json.loads(prov.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_regulon_source_from_prov(prov: Dict[str, Any]) -> str:
    for k in ["regulon_source", "tf_regulon_source", "collectri_source", "dorothea_source", "net_source"]:
        v = prov.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # some provenance may store in nested dict
    for k in ["tf", "tf_activity", "decoupler"]:
        sub = prov.get(k)
        if isinstance(sub, dict):
            v = sub.get("regulon_source") or sub.get("source") or sub.get("net_source")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "unknown"


def _infer_method_from_prov(prov: Dict[str, Any]) -> str:
    for k in ["tf_method", "method", "tf_activity_method"]:
        v = prov.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["tf", "tf_activity", "decoupler"]:
        sub = prov.get(k)
        if isinstance(sub, dict):
            v = sub.get("method") or sub.get("tf_method")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "tf_activity"


def _load_tf_activity(out_root: Path, disease: str) -> Optional[pd.DataFrame]:
    # Engine1 folder location (canonical)
    p = out_root / "engines" / "causal_pathway_features" / disease / "tf_activity.tsv"
    if p.exists():
        return _read_indexed_table(p)

    # fallback: search under disease dir
    cohort = _cohort_dir(out_root, disease)
    cand = _find_any_matching(cohort, ["tf_activity"])
    if cand and cand.exists():
        try:
            return _read_indexed_table(cand)
        except Exception:
            return None
    return None


def build_regulators_evidence(
    out_root: Path,
    disease: str,
    overlap_edges: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preferred:
      - If tf_activity + labels exist -> compute TF deltas and stats; add n_pathways from overlap.
      - Else -> summarize overlap TF edges (n_pathways + mean edge score).

    Additive schema tightening:
      - adds 'method', 'regulon_source', 'activity', 'activity_metric', 'direction'
        while preserving existing columns.
    """
    warnings: List[str] = []
    cohort = _cohort_dir(out_root, disease)

    prov = _load_engine1_provenance(out_root, disease)
    method = _infer_method_from_prov(prov)
    regulon_source = _infer_regulon_source_from_prov(prov)

    tf_act = _load_tf_activity(out_root, disease)
    labels = _load_labels(cohort)

    # mapping TF -> n_pathways and mean_overlap_score
    tf_map = pd.DataFrame(columns=["entity", "n_pathways", "overlap_mean_score"])
    if overlap_edges is not None and not overlap_edges.empty:
        oe = overlap_edges.copy()
        oe["entity_type_norm"] = oe["entity_type"].astype(str).map(_normalize_entity_type)
        oe = oe[oe["entity_type_norm"] == "TF"].copy()
        if not oe.empty:
            oe["entity_norm"] = oe["entity"].astype(str).str.replace("^TF:", "", regex=True).str.upper().str.strip()
            agg = (
                oe.groupby("entity_norm", dropna=False)
                .agg(
                    n_pathways=("pathway", lambda x: int(pd.Series(x).nunique())),
                    overlap_mean_score=("score", "mean"),
                )
                .reset_index()
                .rename(columns={"entity_norm": "entity"})
            )
            tf_map = agg.copy()
            tf_map["overlap_mean_score"] = pd.to_numeric(tf_map["overlap_mean_score"], errors="coerce")

    # Case 1: TF delta evidence
    if tf_act is not None and labels is not None:
        y = labels.copy()
        y.index = y.index.astype(str).str.strip()
        tf_act = tf_act.copy()
        tf_act.index = tf_act.index.astype(str).str.strip()

        y = y.reindex(tf_act.index).dropna()
        if y.nunique() < 2:
            warnings.append("labels_used found but has <2 groups after alignment with tf_activity; falling back to overlap-only regulators evidence.")
        else:
            counts = y.value_counts()
            gA, gB = counts.index[0], counts.index[1]
            idxA = y.index[y == gA]
            idxB = y.index[y == gB]

            X = tf_act.loc[y.index].copy()
            # strip TF: prefix if present
            X.columns = [str(c).replace("TF:", "", 1).strip() for c in X.columns]
            X.columns = [c.upper() for c in X.columns]

            deltas: List[float] = []
            tvals: List[float] = []
            pvals: List[float] = []
            meansA: List[float] = []
            meansB: List[float] = []
            entities: List[str] = []

            for tf in list(X.columns):
                a = pd.to_numeric(X.loc[idxA, tf], errors="coerce")
                b = pd.to_numeric(X.loc[idxB, tf], errors="coerce")
                mA = float(a.mean(skipna=True)) if len(a) else 0.0
                mB = float(b.mean(skipna=True)) if len(b) else 0.0
                t, p = _try_scipy_ttest(a, b)

                entities.append(str(tf))
                meansA.append(mA)
                meansB.append(mB)
                deltas.append(mA - mB)
                tvals.append(float(t))
                pvals.append(float(p))

            q = _bh_fdr(pvals)
            out = pd.DataFrame(
                {
                    "entity_type": "TF",
                    "entity": entities,
                    "n_pathways": 0,  # will be filled from overlap if available
                    "mean_score": deltas,  # legacy name: delta(A-B)
                    "activity": deltas,  # schema-stable alias
                    "activity_metric": "delta_mean(A-B)",
                    "mean_A": meansA,
                    "mean_B": meansB,
                    "t": tvals,
                    "p_value": pvals,
                    "fdr": q,
                    "group_A": str(gA),
                    "group_B": str(gB),
                    "n_A": int(len(idxA)),
                    "n_B": int(len(idxB)),
                    "method": str(method),
                    "regulon_source": str(regulon_source),
                }
            )
            out["direction"] = out["activity"].map(_direction_from_score)

            # join overlap mapping
            if tf_map is not None and not tf_map.empty:
                out = out.merge(tf_map, on="entity", how="left")
                out["n_pathways"] = pd.to_numeric(out.get("n_pathways_y"), errors="coerce").fillna(0).astype(int)
                out = out.drop(columns=[c for c in ["n_pathways_x", "n_pathways_y"] if c in out.columns])
            else:
                out["overlap_mean_score"] = pd.NA

            out = out.sort_values(["fdr", "p_value"], ascending=[True, True]).reset_index(drop=True)
            return out, warnings

    # Case 2: overlap-only summary
    if tf_map is not None and not tf_map.empty:
        out2 = tf_map.copy()
        out2.insert(0, "entity_type", "TF")
        out2 = out2.rename(columns={"overlap_mean_score": "mean_score"})
        out2["mean_score"] = pd.to_numeric(out2["mean_score"], errors="coerce")
        # schema-stable fields
        out2["activity"] = out2["mean_score"]
        out2["activity_metric"] = "overlap_mean_score"
        out2["direction"] = out2["activity"].map(_direction_from_score)
        out2["p_value"] = pd.NA
        out2["fdr"] = pd.NA
        out2["method"] = "overlap_json"
        out2["regulon_source"] = "overlap_json"
        out2["source"] = "overlap_json"
        return out2, warnings + ["TF delta evidence not available; regulators evidence is overlap-only."]

    return pd.DataFrame(
        columns=[
            "entity_type",
            "entity",
            "n_pathways",
            "mean_score",
            "activity",
            "activity_metric",
            "direction",
            "method",
            "regulon_source",
            "p_value",
            "fdr",
        ]
    ), warnings + ["No tf_activity+labels evidence AND no overlap TF edges; regulators_evidence.tsv is empty."]


# ----------------------------
# Pathways evidence merge (+ stable columns + ORA + direction)
# ----------------------------
def _find_gseapy_report(cohort_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for sub in ["gsea", "gsea_c2cp", "GSEA", "gsea_outputs", "gseapy"]:
        d = cohort_dir / sub
        if d.exists() and d.is_dir():
            candidates += list(d.rglob("*report*.csv"))
            candidates += list(d.rglob("*report*.tsv"))
            candidates += list(d.rglob("gseapy.*.report.csv"))
            candidates += list(d.rglob("gseapy.*.report.tsv"))
    candidates += list(cohort_dir.rglob("*report*.csv"))
    candidates += list(cohort_dir.rglob("*report*.tsv"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_gseapy_report(p: Path) -> pd.DataFrame:
    df = _read_table(p)
    df = _clean_cols(df)
    if "Term" not in df.columns:
        for alt in ["term", "NAME", "Pathway", "pathway"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Term"})
                break
    return df


def _merge_pathways_and_gsea(pw_stats: pd.DataFrame, gsea_df: pd.DataFrame) -> pd.DataFrame:
    # normalize to "pathway" column on both sides
    out = pw_stats.copy() if pw_stats is not None else pd.DataFrame()

    if out is not None and not out.empty:
        id_col = None
        for c in out.columns:
            lc = str(c).lower().strip()
            if lc in {"pathway", "term", "name", "pid"}:
                id_col = c
                break
        if id_col is None:
            id_col = out.columns[0]
        out = out.rename(columns={id_col: "pathway"})
        out["pathway"] = out["pathway"].astype(str).str.strip()

    g = gsea_df.copy() if gsea_df is not None else pd.DataFrame()
    if not g.empty:
        if "Term" in g.columns:
            g = g.rename(columns={"Term": "pathway"})
        g["pathway"] = g["pathway"].astype(str).str.strip()
        g = g.add_prefix("gsea_")
        g = g.rename(columns={"gsea_pathway": "pathway"})

    if not out.empty and not g.empty:
        merged = out.merge(g, on="pathway", how="outer")
    elif not out.empty:
        merged = out
    elif not g.empty:
        merged = g
    else:
        merged = pd.DataFrame(columns=["pathway"])

    # flags (more robust than delta_activity-only)
    ipaa_cols = {"delta_activity", "t_like", "t", "stat", "p_value", "fdr", "q_value"}
    has_ipaa = pd.Series([False] * len(merged))
    for c in merged.columns:
        if str(c).strip() in ipaa_cols or str(c).lower().strip() in ipaa_cols:
            has_ipaa = has_ipaa | merged[c].notna()
    merged["has_ipaa_stats"] = has_ipaa.astype(int)

    merged["has_gsea"] = (merged.filter(like="gsea_").notna().any(axis=1)).astype(int)

    # canonical gsea_fdr
    qcol = None
    for c in merged.columns:
        lc = str(c).lower()
        if "gsea_" in lc and ("fdr q-val" in lc or lc.endswith("fdr q-val") or "fdr_q" in lc or "fdr" == lc.split("_")[-1]):
            qcol = c
            break
    if qcol and qcol in merged.columns:
        merged["gsea_fdr"] = pd.to_numeric(merged[qcol], errors="coerce")
    elif "gsea_FDR q-val" in merged.columns:
        merged["gsea_fdr"] = pd.to_numeric(merged["gsea_FDR q-val"], errors="coerce")
    else:
        merged["gsea_fdr"] = pd.NA

    # canonical gsea_nes
    nes_col = None
    for c in merged.columns:
        lc = str(c).lower()
        if lc.endswith("nes") and lc.startswith("gsea_"):
            nes_col = c
            break
        if lc == "gsea_nes":
            nes_col = c
            break
        if lc.startswith("gsea_") and lc.endswith("nes "):
            nes_col = c
            break
    if nes_col and nes_col in merged.columns:
        merged["gsea_nes"] = pd.to_numeric(merged[nes_col], errors="coerce")
    else:
        # common exact column name from gseapy report
        if "gsea_NES" in merged.columns:
            merged["gsea_nes"] = pd.to_numeric(merged["gsea_NES"], errors="coerce")
        else:
            merged["gsea_nes"] = pd.NA

    # canonical gsea_p_value (best-effort)
    pcol = None
    for c in merged.columns:
        lc = str(c).lower()
        if lc.startswith("gsea_") and (lc.endswith("p-value") or lc.endswith("pvalue") or lc.endswith("p_val") or lc.endswith("p")):
            pcol = c
            break
    merged["gsea_p_value"] = pd.to_numeric(merged[pcol], errors="coerce") if pcol and pcol in merged.columns else pd.NA

    return merged


def _looks_like_pathway_library_file(p: Path) -> bool:
    """
    Heuristic to avoid accidentally ingesting entity enrichment ORA as pathway ORA.
    We only accept if filename/path suggests pathway libraries.
    """
    s = (str(p).lower() + " " + str(p.parent).lower())
    tokens = ["kegg", "reactome", "hallmark", "msig", "wp", "wikipathways", "biocarta", "go_", "pathway"]
    return any(t in s for t in tokens)


def _find_ora_files(cohort_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Best-effort find ORA results for UP/DOWN.
    Returns (ora_up_path, ora_down_path), may be None.
    We keep this conservative to avoid mixing entity ORA into pathway ORA.
    """
    up_cands: List[Path] = []
    down_cands: List[Path] = []

    for p in cohort_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".tsv") or name.endswith(".csv") or name.endswith(".txt")):
            continue
        if "ora" not in name and "enrichr" not in name:
            continue
        if "up" in name and "down" not in name:
            if _looks_like_pathway_library_file(p):
                up_cands.append(p)
        if "down" in name:
            if _looks_like_pathway_library_file(p):
                down_cands.append(p)

    def pick_latest(cands: List[Path]) -> Optional[Path]:
        if not cands:
            return None
        cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]

    return pick_latest(up_cands), pick_latest(down_cands)


def _parse_ora_table(p: Path) -> pd.DataFrame:
    df = _read_table(p)
    df = _clean_cols(df)

    # normalize term column
    if "Term" not in df.columns:
        for alt in ["term", "NAME", "Pathway", "pathway"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Term"})
                break

    # normalize adjusted p / fdr
    adj_col = None
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in {"adjusted p-value", "adjusted p value", "adj_p", "fdr", "q-value", "q_value", "padj"}:
            adj_col = c
            break
        if "adjusted" in lc and "p" in lc:
            adj_col = c
            break
    if adj_col is None:
        # common gseapy enrichr column names
        for c in df.columns:
            if str(c).lower().strip() == "adjusted p-value":
                adj_col = c
                break

    if adj_col is not None and adj_col in df.columns:
        df["ora_fdr"] = pd.to_numeric(df[adj_col], errors="coerce")
    else:
        df["ora_fdr"] = pd.NA

    # keep only needed
    keep = ["Term", "ora_fdr"]
    for c in ["P-value", "p_value", "Pvalue", "pvalue"]:
        if c in df.columns:
            df["ora_p_value"] = pd.to_numeric(df[c], errors="coerce")
            keep.append("ora_p_value")
            break

    df = df[keep].copy()
    df["Term"] = df["Term"].astype(str).str.strip()
    df = df[df["Term"] != ""].copy()
    # collapse duplicates (min fdr)
    df = df.groupby("Term", as_index=False).agg({"ora_fdr": "min", **({"ora_p_value": "min"} if "ora_p_value" in df.columns else {})})
    return df


def _add_pathway_stable_columns(
    merged_pw: pd.DataFrame,
    ora_up: Optional[pd.DataFrame],
    ora_down: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Additive-only tightening:
      - ensure stable columns exist: ipaa_activity, gsea_nes, gsea_fdr, ora_up_fdr, ora_down_fdr, direction
    """
    out = merged_pw.copy()

    # ipaa_activity: prefer delta_activity, else t_like/stat (fallback)
    if "ipaa_activity" not in out.columns:
        if "delta_activity" in out.columns:
            out["ipaa_activity"] = pd.to_numeric(out["delta_activity"], errors="coerce")
        elif "t_like" in out.columns:
            out["ipaa_activity"] = pd.to_numeric(out["t_like"], errors="coerce")
        elif "stat" in out.columns:
            out["ipaa_activity"] = pd.to_numeric(out["stat"], errors="coerce")
        else:
            out["ipaa_activity"] = pd.NA
    else:
        out["ipaa_activity"] = pd.to_numeric(out["ipaa_activity"], errors="coerce")

    # canonical gsea_nes/gsea_fdr already created by merge; ensure numeric
    if "gsea_nes" not in out.columns:
        out["gsea_nes"] = pd.NA
    out["gsea_nes"] = pd.to_numeric(out["gsea_nes"], errors="coerce")

    if "gsea_fdr" not in out.columns:
        out["gsea_fdr"] = pd.NA
    out["gsea_fdr"] = pd.to_numeric(out["gsea_fdr"], errors="coerce")

    # ORA
    if "ora_up_fdr" not in out.columns:
        out["ora_up_fdr"] = pd.NA
    if "ora_down_fdr" not in out.columns:
        out["ora_down_fdr"] = pd.NA

    if ora_up is not None and not ora_up.empty:
        tmp = ora_up.rename(columns={"Term": "pathway"}).copy()
        tmp["pathway"] = tmp["pathway"].astype(str).str.strip()
        tmp = tmp[["pathway", "ora_fdr"]].rename(columns={"ora_fdr": "ora_up_fdr"})
        out = out.merge(tmp, on="pathway", how="left", suffixes=("", "_ora_up"))
        if "ora_up_fdr_ora_up" in out.columns:
            out["ora_up_fdr"] = out["ora_up_fdr"].combine_first(out["ora_up_fdr_ora_up"])
            out = out.drop(columns=["ora_up_fdr_ora_up"])

    if ora_down is not None and not ora_down.empty:
        tmp = ora_down.rename(columns={"Term": "pathway"}).copy()
        tmp["pathway"] = tmp["pathway"].astype(str).str.strip()
        tmp = tmp[["pathway", "ora_fdr"]].rename(columns={"ora_fdr": "ora_down_fdr"})
        out = out.merge(tmp, on="pathway", how="left", suffixes=("", "_ora_down"))
        if "ora_down_fdr_ora_down" in out.columns:
            out["ora_down_fdr"] = out["ora_down_fdr"].combine_first(out["ora_down_fdr_ora_down"])
            out = out.drop(columns=["ora_down_fdr_ora_down"])

    out["ora_up_fdr"] = pd.to_numeric(out["ora_up_fdr"], errors="coerce")
    out["ora_down_fdr"] = pd.to_numeric(out["ora_down_fdr"], errors="coerce")

    # direction (prefer ipaa_activity, else gsea_nes)
    if "direction" not in out.columns:
        out["direction"] = out["ipaa_activity"].map(_direction_from_score)

        # if ipaa_activity NA, try gsea_nes
        mask_na = out["direction"].isin(["NA"])
        if mask_na.any():
            out.loc[mask_na, "direction"] = out.loc[mask_na, "gsea_nes"].map(_direction_from_score)

    return out


# ----------------------------
# Confounding evidence (engine2 -> bundle)
# ----------------------------
def _load_confounding_report(out_root: Path, disease: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Best-effort: reads Engine2 confounding report and normalizes to a stable table.
    Returns (df_or_None, warnings).
    """
    warnings: List[str] = []
    p = out_root / "engines" / "confounding" / disease / "confounding_report.tsv"
    if not p.exists():
        return None, ["confounding_report.tsv not found; confounding_evidence.tsv will not be written"]

    try:
        df = _read_table(p)
        df = _clean_cols(df)
    except Exception as e:
        return None, [f"Failed reading confounding_report.tsv: {p} ({e})"]

    if df.empty:
        return None, ["confounding_report.tsv is empty; confounding_evidence.tsv will not be written"]

    # Normalize likely columns
    rename: Dict[str, str] = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in {"feature", "feature_id", "node", "variable"}:
            rename[c] = "feature"
        elif lc in {"cell_program", "program", "cell_type", "cell_signature"}:
            rename[c] = "cell_program"
        elif lc in {"corr", "correlation", "r", "spearman_r", "pearson_r"}:
            rename[c] = "corr"
        elif lc in {"abs_corr", "abs(r)", "abs_r"}:
            rename[c] = "abs_corr"
        elif lc in {"penalty", "penalty_score", "downweight"}:
            rename[c] = "penalty"
        elif lc in {"flagged", "is_flagged"}:
            rename[c] = "flagged"
    out = df.rename(columns=rename).copy()

    # ensure required columns exist
    if "feature" not in out.columns:
        out["feature"] = out.iloc[:, 0].astype(str)
        warnings.append("confounding_report.tsv missing 'feature' column; using first column as feature")

    if "corr" in out.columns:
        out["corr"] = pd.to_numeric(out["corr"], errors="coerce")
    else:
        out["corr"] = pd.NA

    if "abs_corr" in out.columns:
        out["abs_corr"] = pd.to_numeric(out["abs_corr"], errors="coerce")
    else:
        out["abs_corr"] = out["corr"].abs()

    if "penalty" in out.columns:
        out["penalty"] = pd.to_numeric(out["penalty"], errors="coerce")
    else:
        # if not provided, use abs_corr as a conservative penalty proxy
        out["penalty"] = out["abs_corr"]
        warnings.append("confounding_report.tsv missing 'penalty'; using abs_corr as penalty proxy")

    if "cell_program" not in out.columns:
        out["cell_program"] = pd.NA

    if "flagged" not in out.columns:
        out["flagged"] = (out["abs_corr"] >= 0.4).fillna(False).astype(int)

    # stable column order (keep extras at end)
    base_cols = ["feature", "cell_program", "corr", "abs_corr", "penalty", "flagged"]
    extras = [c for c in out.columns if c not in base_cols]
    out = out[base_cols + extras].copy()

    out["feature"] = out["feature"].astype(str).str.strip()
    out = out[out["feature"] != ""].copy()

    return out.reset_index(drop=True), warnings


# ----------------------------
# Combined evidence (wide + long)
# ----------------------------
def _infer_leading_genes_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if "lead_genes" in lc or "leading_edge" in lc:
            return c
    return None


def _infer_ora_genes_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc.endswith("ora_genes") or lc == "ora_genes" or lc.endswith("genes"):
            if "ora_" in lc:
                return c
    return None


def build_pathway_combined_evidence(
    pathways_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    overlap_edges: pd.DataFrame,
    regulators_df: pd.DataFrame,
) -> pd.DataFrame:
    pw = pathways_df.copy()
    pw = _clean_cols(pw)
    if "pathway" not in pw.columns:
        pw = pw.rename(columns={pw.columns[0]: "pathway"})
    pw["pathway"] = pw["pathway"].astype(str).str.strip()

    g = genes_df.copy() if genes_df is not None else pd.DataFrame(columns=["gene", "score", "fdr", "p_value"])
    g = _clean_cols(g)
    if "gene" not in g.columns and len(g.columns) > 0:
        g = g.rename(columns={g.columns[0]: "gene"})
    if "score" not in g.columns:
        if "stat" in g.columns:
            g["score"] = g["stat"]
        elif "log2fc" in g.columns:
            g["score"] = g["log2fc"]
        else:
            g["score"] = pd.NA
    g["gene"] = g["gene"].astype(str).str.strip()
    g["gene_norm"] = g["gene"].str.upper()
    g["score"] = pd.to_numeric(g["score"], errors="coerce")
    if "fdr" in g.columns:
        g["fdr"] = pd.to_numeric(g["fdr"], errors="coerce")
    else:
        g["fdr"] = pd.NA

    # overlap edges normalized
    oe = overlap_edges.copy() if overlap_edges is not None else pd.DataFrame(columns=["pathway", "entity_type", "entity", "score"])
    if not oe.empty:
        oe["pathway"] = oe["pathway"].astype(str).str.strip()
        oe["entity_type_norm"] = oe["entity_type"].astype(str).map(_normalize_entity_type)
        oe["entity_norm"] = oe["entity"].astype(str).str.replace("^TF:", "", regex=True).str.upper().str.strip()
        oe["score"] = pd.to_numeric(oe.get("score"), errors="coerce")

    # regulators normalized (TF delta table)
    reg = regulators_df.copy() if regulators_df is not None else pd.DataFrame(columns=["entity_type", "entity", "mean_score"])
    if not reg.empty:
        reg["entity_type_norm"] = reg["entity_type"].astype(str).map(_normalize_entity_type)
        reg = reg[reg["entity_type_norm"] == "TF"].copy()
        reg["entity_norm"] = reg["entity"].astype(str).str.replace("^TF:", "", regex=True).str.upper().str.strip()
        # prefer schema-stable "activity" if present, else legacy mean_score
        if "activity" in reg.columns:
            reg["__act__"] = pd.to_numeric(reg.get("activity"), errors="coerce")
        else:
            reg["__act__"] = pd.to_numeric(reg.get("mean_score"), errors="coerce")
        if "fdr" in reg.columns:
            reg["fdr"] = pd.to_numeric(reg["fdr"], errors="coerce")
        else:
            reg["fdr"] = pd.NA

    lead_col = _infer_leading_genes_col(pw)
    ora_genes_col = _infer_ora_genes_col(pw)

    gene_n: List[int] = []
    gene_mean: List[float] = []
    gene_mean_abs: List[float] = []
    gene_sig_n: List[int] = []
    gene_top: List[str] = []

    tf_n: List[int] = []
    tf_mean: List[float] = []
    tf_mean_abs: List[float] = []
    tf_sig_n: List[int] = []
    tf_top: List[str] = []

    ent_tf_n: List[int] = []
    ent_epi_n: List[int] = []
    ent_met_n: List[int] = []

    for _, row in pw.iterrows():
        pid = str(row.get("pathway", "")).strip()

        # genes in pathway from leading edge or ORA
        genes_list: List[str] = []
        if lead_col and lead_col in pw.columns:
            genes_list = _parse_list_str(row.get(lead_col))
        if not genes_list and ora_genes_col and ora_genes_col in pw.columns:
            genes_list = _parse_list_str(row.get(ora_genes_col))

        genes_norm = {x.upper().strip() for x in genes_list if x and str(x).strip()}
        subg = g[g["gene_norm"].isin(list(genes_norm))].copy() if genes_norm else g.iloc[0:0].copy()

        gene_n.append(int(len(subg)))
        if len(subg):
            gene_mean.append(float(subg["score"].mean(skipna=True)))
            gene_mean_abs.append(float(subg["score"].abs().mean(skipna=True)))
            gene_sig_n.append(int((subg["fdr"] <= 0.05).fillna(False).sum()) if "fdr" in subg.columns else 0)
            topg = subg.assign(__abs__=subg["score"].abs()).sort_values("__abs__", ascending=False)
            gene_top.append(";".join(topg["gene"].astype(str).head(10).tolist()))
        else:
            gene_mean.append(float("nan"))
            gene_mean_abs.append(float("nan"))
            gene_sig_n.append(0)
            gene_top.append("")

        # entity counts per pathway (from overlap)
        if not oe.empty and pid:
            sube = oe[oe["pathway"] == pid].copy()
            ent_tf_n.append(int((sube["entity_type_norm"] == "TF").sum()))
            ent_epi_n.append(int((sube["entity_type_norm"] == "epigenetic").sum()))
            ent_met_n.append(int((sube["entity_type_norm"] == "metabolite").sum()))
        else:
            ent_tf_n.append(0)
            ent_epi_n.append(0)
            ent_met_n.append(0)

        # TF aggregation per pathway: use overlap edges to pick TFs, then use reg deltas
        tfs_for_pw: Set[str] = set()
        if not oe.empty and pid:
            tmp = oe[(oe["pathway"] == pid) & (oe["entity_type_norm"] == "TF")]
            tfs_for_pw = set(tmp["entity_norm"].dropna().astype(str).tolist())

        if tfs_for_pw and not reg.empty:
            subr = reg[reg["entity_norm"].isin(list(tfs_for_pw))].copy()
        else:
            subr = reg.iloc[0:0].copy()

        tf_n.append(int(len(subr)))
        if len(subr):
            tf_mean.append(float(subr["__act__"].mean(skipna=True)))
            tf_mean_abs.append(float(subr["__act__"].abs().mean(skipna=True)))
            tf_sig_n.append(int((subr["fdr"] <= 0.05).fillna(False).sum()) if "fdr" in subr.columns else 0)
            topt = subr.assign(__abs__=subr["__act__"].abs()).sort_values("__abs__", ascending=False)
            # show raw entity IDs (already uppercase)
            tf_top.append(";".join(topt["entity"].astype(str).head(10).tolist()))
        else:
            tf_mean.append(float("nan"))
            tf_mean_abs.append(float("nan"))
            tf_sig_n.append(0)
            tf_top.append("")

    pw["gene_evidence_n"] = gene_n
    pw["gene_evidence_mean_score"] = gene_mean
    pw["gene_evidence_mean_abs_score"] = gene_mean_abs
    pw["gene_evidence_n_fdr05"] = gene_sig_n
    pw["gene_evidence_top10"] = gene_top

    pw["tf_evidence_n"] = tf_n
    pw["tf_evidence_mean_activity"] = tf_mean
    pw["tf_evidence_mean_abs_activity"] = tf_mean_abs
    pw["tf_evidence_n_fdr05"] = tf_sig_n
    pw["tf_evidence_top10"] = tf_top

    pw["overlap_tf_edges_n"] = ent_tf_n
    pw["overlap_epigenetic_edges_n"] = ent_epi_n
    pw["overlap_metabolite_edges_n"] = ent_met_n

    return pw


def build_all_evidence_long(
    pathways_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    regulators_df: pd.DataFrame,
    confounding_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    pw = pathways_df.copy()
    if "pathway" not in pw.columns:
        pw = pw.rename(columns={pw.columns[0]: "pathway"})
    pw["evidence_level"] = "pathway"
    pw["evidence_id"] = pw["pathway"].astype(str)

    g = genes_df.copy()
    if "gene" not in g.columns and len(g.columns) > 0:
        g = g.rename(columns={g.columns[0]: "gene"})
    g["evidence_level"] = "gene"
    g["evidence_id"] = g["gene"].astype(str)

    r = regulators_df.copy()
    if "entity" in r.columns:
        r["evidence_level"] = "regulator"
        r["evidence_id"] = r["entity"].astype(str)
    else:
        r["evidence_level"] = "regulator"
        r["evidence_id"] = ""

    frames = [pw, g, r]

    if confounding_df is not None and not confounding_df.empty:
        cdf = confounding_df.copy()
        cdf["evidence_level"] = "confounding"
        cdf["evidence_id"] = cdf.get("feature", pd.Series([""] * len(cdf))).astype(str)
        frames.append(cdf)

    # align columns
    cols: List[str] = sorted(set().union(*[set(f.columns) for f in frames]))
    frames2 = [f.reindex(columns=cols) for f in frames]
    out = pd.concat(frames2, axis=0, ignore_index=True)
    return out


# ----------------------------
# Engine entrypoints
# ----------------------------
def build_evidence_bundle_for_disease(
    out_root: Path,
    disease: str,
    *,
    q_cutoff: float = 0.25,
) -> EvidenceBundlePaths:
    """
    Drop-in-safe behavior:
      - preserves existing outputs and columns,
      - adds schema-tightening columns additively:
          genes: direction, expressed_flag
          pathways: ipaa_activity, gsea_nes, gsea_fdr, ora_up_fdr, ora_down_fdr, direction
          regulators: method, regulon_source, activity, activity_metric, direction
      - writes confounding_evidence.tsv if Engine2 report exists (additive new file)
    """
    out_root = Path(out_root)
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    cohort = _cohort_dir(out_root, disease)
    engine_dir = out_root / "engines" / "evidence_bundle" / disease
    engine_dir.mkdir(parents=True, exist_ok=True)

    paths = EvidenceBundlePaths(
        out_dir=engine_dir,
        genes_evidence=engine_dir / "genes_evidence.tsv",
        pathways_evidence=engine_dir / "pathways_evidence.tsv",
        regulators_evidence=engine_dir / "regulators_evidence.tsv",
        pathway_combined_evidence=engine_dir / "pathway_combined_evidence.tsv",
        all_evidence_long=engine_dir / "all_evidence_long.tsv",
        mechanism_summary=engine_dir / "mechanism_summary.json",
        manifest=engine_dir / "ENGINE_MANIFEST.json",
    )

    confounding_evidence_path = engine_dir / "confounding_evidence.tsv"

    warnings: List[str] = []

    # --- expressed genes set (header-only) ---
    expressed_genes_upper, exp_warn = _load_expressed_gene_set_from_expression_used(cohort)
    warnings.extend(exp_warn)

    # --- genes evidence ---
    genes_df: Optional[pd.DataFrame] = None
    for cand in [cohort / "de_gene_stats.tsv", cohort / "DE_gene_stats.tsv", cohort / "degs.tsv"]:
        if cand.exists():
            genes_df = normalize_genes_df(_read_table(cand))
            break
    if genes_df is None:
        genes_df = pd.DataFrame(columns=["gene", "score", "stat", "p_value", "fdr", "log2fc"])
        warnings.append("No de_gene_stats.tsv / DE_gene_stats.tsv / degs.tsv found; genes_evidence is empty.")

    genes_df = _add_gene_direction_and_expressed_flag(genes_df, expressed_genes_upper)
    genes_df.to_csv(paths.genes_evidence, sep="\t", index=False)

    # --- pathway stats + gseapy report merge ---
    pw_stats = None
    for cand in [cohort / "pathway_stats_with_baseline.tsv", cohort / "pathway_stats.tsv"]:
        if cand.exists():
            pw_stats = _read_table(cand)
            break
    if pw_stats is None:
        pw_stats = pd.DataFrame(columns=["pathway"])
        warnings.append("No pathway_stats(.tsv) found; pathways_evidence will be minimal.")

    gsea_report_path = _find_gseapy_report(cohort)
    gsea_df = pd.DataFrame()
    if gsea_report_path:
        try:
            gsea_df = _parse_gseapy_report(gsea_report_path)
        except Exception as e:
            warnings.append(f"Failed parsing gseapy report: {gsea_report_path} ({e})")

    merged_pw = _merge_pathways_and_gsea(pw_stats, gsea_df)

    # ORA ingestion (best-effort; conservative heuristic)
    ora_up_path, ora_down_path = _find_ora_files(cohort)
    ora_up_df: Optional[pd.DataFrame] = None
    ora_down_df: Optional[pd.DataFrame] = None
    if ora_up_path:
        try:
            ora_up_df = _parse_ora_table(ora_up_path)
        except Exception as e:
            warnings.append(f"Failed parsing ORA UP file: {ora_up_path} ({e})")
    if ora_down_path:
        try:
            ora_down_df = _parse_ora_table(ora_down_path)
        except Exception as e:
            warnings.append(f"Failed parsing ORA DOWN file: {ora_down_path} ({e})")

    merged_pw = _add_pathway_stable_columns(merged_pw, ora_up_df, ora_down_df)

    # mark gsea_sig if possible
    merged_pw["gsea_sig"] = (pd.to_numeric(merged_pw.get("gsea_fdr"), errors="coerce") <= float(q_cutoff)).fillna(False).astype(int)

    merged_pw.to_csv(paths.pathways_evidence, sep="\t", index=False)

    # --- overlap JSON (for TF->pathway mapping + pathway entity counts) ---
    overlap_json = cohort / "overlap" / "pathway_entity_overlap.json"
    if not overlap_json.exists():
        # try shallow search
        cand = _find_any_matching(cohort, ["pathway_entity_overlap.json"])
        if cand:
            overlap_json = cand

    overlap_edges, overlap_warn = parse_overlap_edges(overlap_json) if overlap_json.exists() else (pd.DataFrame(), ["overlap_json not found"])
    warnings.extend(overlap_warn)

    # --- regulators evidence (TF deltas + n_pathways) ---
    regulators_df, reg_warn = build_regulators_evidence(out_root, disease, overlap_edges)
    warnings.extend(reg_warn)
    regulators_df.to_csv(paths.regulators_evidence, sep="\t", index=False)

    # --- confounding evidence (Engine2) ---
    confounding_df, conf_warn = _load_confounding_report(out_root, disease)
    warnings.extend(conf_warn)
    if confounding_df is not None and not confounding_df.empty:
        confounding_df.to_csv(confounding_evidence_path, sep="\t", index=False)

    # --- combined evidence ---
    combined_pw = build_pathway_combined_evidence(
        pathways_df=merged_pw,
        genes_df=genes_df,
        overlap_edges=overlap_edges if overlap_edges is not None else pd.DataFrame(),
        regulators_df=regulators_df,
    )
    combined_pw.to_csv(paths.pathway_combined_evidence, sep="\t", index=False)

    # --- all evidence (one file: pathways + genes + regulators + confounding if present) ---
    all_long = build_all_evidence_long(merged_pw, genes_df, regulators_df, confounding_df)
    all_long.to_csv(paths.all_evidence_long, sep="\t", index=False)

    # --- summary + manifest ---
    summary = {
        "disease": disease,
        "cohort_dir": str(cohort),
        "gseapy_report": str(gsea_report_path) if gsea_report_path else None,
        "ora_up_file": str(ora_up_path) if ora_up_path else None,
        "ora_down_file": str(ora_down_path) if ora_down_path else None,
        "overlap_json": str(overlap_json) if overlap_json and overlap_json.exists() else None,
        "engine1_provenance": str(out_root / "engines" / "causal_pathway_features" / disease / "feature_provenance.json"),
        "confounding_report": str(out_root / "engines" / "confounding" / disease / "confounding_report.tsv"),
        "schema": {
            "genes_has_direction": True,
            "genes_has_expressed_flag": True,
            "pathways_has_stable_fields": True,
            "regulators_has_method_and_regulon_source": True,
            "confounding_written": bool(confounding_df is not None and not confounding_df.empty),
        },
        "counts": {
            "genes": int(len(genes_df)) if genes_df is not None else 0,
            "pathways": int(len(merged_pw)) if merged_pw is not None else 0,
            "regulators": int(len(regulators_df)) if regulators_df is not None else 0,
            "overlap_edges": int(len(overlap_edges)) if overlap_edges is not None else 0,
            "confounding_rows": int(len(confounding_df)) if confounding_df is not None else 0,
        },
        "notes": warnings[:],
    }
    paths.mechanism_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "engine": "evidence_bundle",
        "version": "2.1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "inputs": {"out_root": str(out_root), "disease": disease},
        "outputs": {
            "genes_evidence": str(paths.genes_evidence),
            "pathways_evidence": str(paths.pathways_evidence),
            "regulators_evidence": str(paths.regulators_evidence),
            "pathway_combined_evidence": str(paths.pathway_combined_evidence),
            "all_evidence_long": str(paths.all_evidence_long),
            "mechanism_summary": str(paths.mechanism_summary),
            # additive new output
            "confounding_evidence": str(confounding_evidence_path) if confounding_evidence_path.exists() else None,
        },
        "warnings": warnings,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return paths


def build_evidence_bundle_all(out_root: Path, *, q_cutoff: float = 0.25) -> List[EvidenceBundlePaths]:
    out_root = Path(out_root)
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    reserved = {"engines", "cohorts", "Report", "reports", "classification", "jsons_all", "results", "logs", "data"}
    diseases: List[str] = []

    for p in out_root.iterdir():
        if p.is_dir() and p.name not in reserved and not p.name.startswith("."):
            diseases.append(p.name)

    legacy_root = out_root / "cohorts"
    if legacy_root.exists():
        for p in legacy_root.iterdir():
            if p.is_dir() and p.name not in reserved and not p.name.startswith("."):
                if p.name not in diseases:
                    diseases.append(p.name)

    if not diseases:
        raise DataError(f"No disease/cohort folders discovered in {out_root}")

    out: List[EvidenceBundlePaths] = []
    for d in sorted(diseases):
        out.append(build_evidence_bundle_for_disease(out_root, d, q_cutoff=q_cutoff))
    return out


# ----------------------------
# Optional CLI
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MDP Engine 0: Evidence bundle (+ combined evidence outputs).")
    ap.add_argument("--out-root", required=True, help="IPAA OUT_ROOT")
    ap.add_argument("--disease", default=None, help="If set, build only for this disease. Otherwise build all.")
    ap.add_argument("--q-cutoff", type=float, default=0.25, help="GSEA FDR cutoff (flag only).")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = ap.parse_args()

    if args.verbose >= 1:
        logging.getLogger().setLevel(logging.DEBUG)

    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        raise SystemExit(f"out_root not found: {out_root}")

    if args.disease:
        build_evidence_bundle_for_disease(out_root, str(args.disease), q_cutoff=float(args.q_cutoff))
    else:
        build_evidence_bundle_all(out_root, q_cutoff=float(args.q_cutoff))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
