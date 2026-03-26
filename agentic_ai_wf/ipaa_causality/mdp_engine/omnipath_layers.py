#!/usr/bin/env python3
"""
mdp_engine/engine/omnipath_layers.py  (FULL DROP-IN REPLACEMENT)

This version is designed to STOP the "PTM TSV is empty" failure mode by enforcing:
1) PTM cache must be built from OmniPath PTMs endpoint (NOT enzsub), with genesymbols.
2) Scoring uses enzyme_genesymbol / substrate_genesymbol (HGNC symbols) when present.
3) Gene IDs are normalized (UPPER/strip) on both DE ranks and PTM tables.
4) Enzyme threshold is based on OVERLAP with the rank (not total substrates).
5) Multi-symbol fields are split (AAK1;BMP2K etc.).
6) Manifest records whether PTM was empty and why.

This is a single-file, paste-over replacement. No external changes required.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests




LOG = logging.getLogger("omnipath_layers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)


# ----------------------------- utils -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_tsv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", engine="python")

def _write_tsv(df: pd.DataFrame, p: Path) -> None:
    _ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(p)

def _write_json(obj: dict, p: Path) -> None:
    _ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(p)

def _guess_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols:
            return cols[key]
    for cand in candidates:
        key = str(cand).strip().lower()
        for lc, orig in cols.items():
            if key and (key in lc):
                return orig
    return None

def _norm_sym(x: object) -> str:
    return str(x).strip().upper()

def _split_symbols(x: object) -> List[str]:
    """
    Split multi-symbol fields into tokens: ';' ',' '|' '/' '\\' whitespace.
    """
    s = _norm_sym(x)
    if not s or s in {"NA", "NAN", "NONE", "NULL"}:
        return []
    for delim in [";", ",", "|", "/", "\\"]:
        s = s.replace(delim, " ")
    toks = [t for t in s.split() if t and t not in {"NA", "NAN", "NONE", "NULL"}]
    return toks


# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class OmniPathLayersConfig:
    out_root: Path
    disease: str
    ipaa_disease_dir: Path
    evidence_bundle_dir: Optional[Path] = None
    allow_http: bool = True
    http_timeout_s: int = 60

    # enzyme scoring
    min_substrates: int = 5
    n_perm: int = 200
    random_seed: int = 17


def _cache_dir(out_root: Path) -> Path:
    return out_root / "engines" / "_cache" / "omnipath"


# ----------------------------- rank computation -----------------------------

def compute_rank_from_gene_table(df: pd.DataFrame) -> pd.Series:
    """
    rank_score = sign(logFC) * -log10(q or p)
    Falls back to sign(logFC) if no p/q.

    IMPORTANT: gene IDs are normalized UPPER/strip.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    gcol = _guess_col(df, ["gene", "symbol", "gene_id", "id", "hgnc", "gene_name"])
    lcol = _guess_col(df, ["logfc", "log2fc", "log2foldchange", "lfc"])
    qcol = _guess_col(df, ["q", "qval", "q-value", "fdr", "padj", "adj_p"])
    pcol = _guess_col(df, ["p", "pval", "p-value", "pvalue"])

    if not gcol or not lcol:
        raise ValueError(f"Cannot compute rank: need gene + logFC. Columns={list(df.columns)}")

    cols_keep: List[str] = [gcol, lcol]
    if qcol:
        cols_keep.append(qcol)
    elif pcol:
        cols_keep.append(pcol)

    work = df[cols_keep].copy()
    work[gcol] = work[gcol].astype(str).str.strip().str.upper()
    work[lcol] = pd.to_numeric(work[lcol], errors="coerce").fillna(0.0)

    usep = qcol if (qcol and qcol in work.columns) else (pcol if (pcol and pcol in work.columns) else None)

    if usep:
        vals = pd.to_numeric(work[usep], errors="coerce").clip(lower=1e-300)
        mag = -vals.apply(lambda x: math.log10(x) if (x and x > 0) else -300.0)
        score = mag * work[lcol].apply(lambda x: 1.0 if x >= 0 else -1.0)
    else:
        LOG.warning("[rank] no p/q column found; using sign(logFC) only")
        score = work[lcol].apply(lambda x: 1.0 if x >= 0 else -1.0)

    score.index = work[gcol].values
    score = score[~pd.Index(score.index).duplicated(keep="first")]
    return score


# ----------------------------- PTM fetch/load -----------------------------

def _fetch_ptm_via_http(timeout_s: int) -> pd.DataFrame:
    """
    HARD ENFORCEMENT:
    Use OmniPath PTMs endpoint with genesymbols, because enzsub often returns UniProt IDs and causes 0 overlap.
    """
    if requests is None:
        LOG.warning("[ptm] requests not installed; cannot fetch PTMs")
        return pd.DataFrame()

    urls = [
        "https://omnipathdb.org/ptms?format=tsv&genesymbols=yes",
        "https://omnipathdb.org/ptms?format=csv&genesymbols=yes",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=int(timeout_s))
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt:
                continue
            df = pd.read_csv(StringIO(txt), sep="\t") if "format=tsv" in url else pd.read_csv(StringIO(txt))
            if df is not None and not df.empty:
                LOG.info("[ptm] fetched via HTTP: %s", url)
                return df
        except Exception as e:
            LOG.info("[ptm] HTTP fetch failed (%s): %s", url, e)

    return pd.DataFrame()


def load_omnipath_ptm(cache_dir: Path, refresh: bool, allow_http: bool, timeout_s: int) -> pd.DataFrame:
    """
    Loads PTM table from cache or (if refresh) re-downloads via PTMs endpoint.
    Cache path is fixed: engines/_cache/omnipath/omnipath_ptm.csv
    """
    _ensure_dir(cache_dir)
    cache_fp = cache_dir / "omnipath_ptm.csv"

    if cache_fp.exists() and not refresh:
        try:
            df = pd.read_csv(cache_fp)
            if df is not None and not df.empty:
                LOG.info("[ptm] using cached: %s", cache_fp)
                return df
        except Exception:
            LOG.warning("[ptm] cached file unreadable: %s", cache_fp)

    if not allow_http:
        return pd.DataFrame()

    df = _fetch_ptm_via_http(timeout_s=timeout_s)
    if df is None or df.empty:
        LOG.warning("[ptm] PTM unavailable; returning empty table")
        return pd.DataFrame()

    df.to_csv(cache_fp, index=False)
    LOG.info("[ptm] cached: %s", cache_fp)
    return df


def _standardize_ptm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer genesymbol columns over enzyme/substrate.
    Your cache has: enzyme, substrate, enzyme_genesymbol, substrate_genesymbol, residue_type, residue_offset, modification
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["enzyme", "substrate", "residue"])

    cols = {str(c).strip().lower(): c for c in df.columns}

    enz = cols.get("enzyme_genesymbol") or cols.get("enzyme_genesymbols")
    sub = cols.get("substrate_genesymbol") or cols.get("substrate_genesymbols")

    # If genesymbol columns are missing, we refuse to score, because scoring UniProt vs HGNC will be empty anyway.
    if enz is None or sub is None:
        LOG.warning("[ptm] missing *_genesymbol columns -> refusing to score. cols=%s", list(df.columns))
        return pd.DataFrame(columns=["enzyme", "substrate", "residue"])

    out = pd.DataFrame()
    out["enzyme"] = df[enz].astype(str).str.strip().str.upper()
    out["substrate"] = df[sub].astype(str).str.strip().str.upper()

    # Residue optional
    residue = cols.get("residue") or cols.get("residue_type") or cols.get("residue_offset")
    out["residue"] = df[residue].astype(str) if (residue and residue in df.columns) else ""

    out = out[(out["enzyme"] != "") & (out["substrate"] != "")]
    out = out[(out["enzyme"].str.lower() != "nan") & (out["substrate"].str.lower() != "nan")]
    return out


def build_enzyme_sets(ptm_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    enzyme -> substrates (gene symbols)
    Splits multi-symbol fields (rare, but safe).
    """
    pdf = _standardize_ptm_columns(ptm_df)
    if pdf.empty:
        return {}

    out: Dict[str, Set[str]] = {}
    for _, r in pdf.iterrows():
        enzymes = _split_symbols(r["enzyme"])
        subs = _split_symbols(r["substrate"])
        if not enzymes or not subs:
            continue
        for e in enzymes:
            out.setdefault(e, set()).update(subs)

    return out


# ----------------------------- KSEA-like scoring -----------------------------

def _mean_on_set(series: pd.Series, items: Set[str]) -> float:
    s = series.reindex(list(items)).dropna()
    return float(s.mean()) if not s.empty else 0.0

def _permute_null(series: pd.Series, k: int, n_perm: int, rng: random.Random) -> Tuple[float, float]:
    vals = series.dropna().values
    if len(vals) < 2 or k <= 1:
        return 0.0, 1.0
    means: List[float] = []
    for _ in range(int(n_perm)):
        idx = rng.sample(range(len(vals)), k)
        means.append(float(vals[idx].mean()))
    mu = float(sum(means) / max(1, len(means)))
    var = float(sum((x - mu) ** 2 for x in means) / max(1, len(means)))
    sd = math.sqrt(var) if var > 1e-12 else 1.0
    return mu, sd

def score_enzyme_activity(rank: pd.Series, enz_sets: Dict[str, Set[str]], min_substrates: int, n_perm: int, seed: int) -> pd.DataFrame:
    """
    IMPORTANT:
    min_substrates is enforced on OVERLAP with rank, not total set size.
    """
    rng = random.Random(int(seed))
    rank = rank.dropna()
    rows: List[Dict[str, object]] = []

    if rank.empty or not enz_sets:
        return pd.DataFrame(columns=["enzyme", "substrates_total", "substrates_overlap", "raw_mean", "mu_null", "sd_null", "NES"])

    for enz, subs in enz_sets.items():
        subs = set(subs)
        overlap = rank.reindex(list(subs)).dropna()
        overlap_n = int(len(overlap))
        if overlap_n < int(min_substrates):
            continue

        raw = float(overlap.mean())
        mu, sd = _permute_null(rank, overlap_n, n_perm=n_perm, rng=rng)
        nes = (raw - mu) / sd if sd > 1e-8 else 0.0

        rows.append({
            "enzyme": enz,
            "substrates_total": int(len(subs)),
            "substrates_overlap": int(overlap_n),
            "raw_mean": float(raw),
            "mu_null": float(mu),
            "sd_null": float(sd),
            "NES": float(nes),
        })

    if not rows:
        return pd.DataFrame(columns=["enzyme", "substrates_total", "substrates_overlap", "raw_mean", "mu_null", "sd_null", "NES"])

    return pd.DataFrame(rows).sort_values("NES", ascending=False).reset_index(drop=True)


# ----------------------------- intercell fetch/load -----------------------------

def _fetch_intercell_via_http(timeout_s: int) -> pd.DataFrame:
    if requests is None:
        LOG.warning("[intercell] requests not installed; cannot fetch intercell")
        return pd.DataFrame()
    url = "https://omnipathdb.org/intercell?format=tsv"
    try:
        r = requests.get(url, timeout=int(timeout_s))
        r.raise_for_status()
        txt = (r.text or "").strip()
        if not txt:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(txt), sep="\t")
        return df if (df is not None and not df.empty) else pd.DataFrame()
    except Exception as e:
        LOG.warning("[intercell] HTTP fetch failed: %s", e)
        return pd.DataFrame()

def load_omnipath_intercell(cache_dir: Path, refresh: bool, allow_http: bool, timeout_s: int) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache_fp = cache_dir / "omnipath_intercell.csv"

    if cache_fp.exists() and not refresh:
        try:
            df = pd.read_csv(cache_fp)
            if df is not None and not df.empty:
                LOG.info("[intercell] using cached: %s", cache_fp)
                return df
        except Exception:
            LOG.warning("[intercell] cached file unreadable: %s", cache_fp)

    if not allow_http:
        return pd.DataFrame()

    df = _fetch_intercell_via_http(timeout_s=timeout_s)
    if df is None or df.empty:
        LOG.warning("[intercell] intercell unavailable; returning empty table")
        return pd.DataFrame()

    df.to_csv(cache_fp, index=False)
    LOG.info("[intercell] cached: %s", cache_fp)
    return df


def _standardize_intercell(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    gene_col = (
        cols.get("genesymbol")
        or cols.get("gene_symbol")
        or cols.get("target_genesymbol")
        or cols.get("source_genesymbol")
        or cols.get("uniprot")
    )
    if not gene_col:
        raise ValueError("Cannot find gene symbol column in intercell table.")

    def safe(col: Optional[str]) -> pd.Series:
        if col and col in df.columns:
            return df[col].astype(str)
        return pd.Series([""] * len(df), index=df.index)

    out = pd.DataFrame()
    out["Gene"] = df[gene_col].astype(str).str.strip().str.upper()
    out["category"] = safe(cols.get("category"))
    out["parent"] = safe(cols.get("parent"))
    out["generic_cat"] = safe(cols.get("generic_category") or cols.get("generic_cat"))
    out["secreted"] = safe(cols.get("secreted"))
    out["pm_transmem"] = safe(cols.get("plasma_membrane_transmembrane"))
    out["pm_peripheral"] = safe(cols.get("plasma_membrane_peripheral"))
    return out


def build_intercell_roles(rank: pd.Series, intercell_df: pd.DataFrame) -> pd.DataFrame:
    if intercell_df.empty or rank.empty:
        return pd.DataFrame()

    ic = _standardize_intercell(intercell_df)
    genes = set(rank.index.astype(str))
    ic = ic[ic["Gene"].isin(genes)].copy()
    if ic.empty:
        return pd.DataFrame()

    def agg(series: pd.Series) -> str:
        vals = sorted({v for v in series.astype(str).tolist() if v and v != "nan"})
        return ";".join(vals)

    grouped = ic.groupby("Gene").agg({
        "category": agg,
        "parent": agg,
        "generic_cat": agg,
        "secreted": agg,
        "pm_transmem": agg,
        "pm_peripheral": agg,
    }).reset_index()

    grouped["rank_score"] = grouped["Gene"].map(rank.to_dict())
    return grouped


# ----------------------------- main engine entry -----------------------------

def _read_genes_evidence_or_fallback(cfg: OmniPathLayersConfig) -> pd.DataFrame:
    # preferred: Engine 0 output
    if cfg.evidence_bundle_dir:
        p = cfg.evidence_bundle_dir / "genes_evidence.tsv"
        if p.exists():
            return _read_tsv(p)

    # fallback: IPAA output
    p2 = cfg.ipaa_disease_dir / "de_gene_stats.tsv"
    if p2.exists():
        return _read_tsv(p2)

    raise FileNotFoundError("No genes table found (genes_evidence.tsv or de_gene_stats.tsv).")


def run_omnipath_layers(cfg: OmniPathLayersConfig, *, refresh_cache: bool = False) -> Dict[str, Path]:
    out_root = cfg.out_root.expanduser().resolve()
    ipaa_dir = cfg.ipaa_disease_dir.expanduser().resolve()

    engine_dir = out_root / "engines" / "causal_pathway_features" / cfg.disease
    _ensure_dir(engine_dir)

    cache = _cache_dir(out_root)
    _ensure_dir(cache)

    out_ptm = engine_dir / "omnipath_ptm_kinase_activity.tsv"
    out_ic = engine_dir / "omnipath_intercell_roles.tsv"
    manifest = engine_dir / "ENGINE_MANIFEST.omnipath_layers.json"

    genes_df = _read_genes_evidence_or_fallback(cfg)
    rank = compute_rank_from_gene_table(genes_df)

    outputs: Dict[str, Path] = {}
    diagnostics: Dict[str, object] = {
        "rank_n": int(len(rank)),
        "ptm_cache_path": str(cache / "omnipath_ptm.csv"),
        "intercell_cache_path": str(cache / "omnipath_intercell.csv"),
    }

    # ---- PTM scoring ----
    ptm_df = load_omnipath_ptm(
        cache_dir=cache,
        refresh=bool(refresh_cache),
        allow_http=bool(cfg.allow_http),
        timeout_s=int(cfg.http_timeout_s),
    )
    diagnostics["ptm_rows"] = int(len(ptm_df))
    diagnostics["ptm_cols"] = list(ptm_df.columns) if not ptm_df.empty else []

    enz_sets = build_enzyme_sets(ptm_df)
    diagnostics["enzymes_n"] = int(len(enz_sets))

    act = score_enzyme_activity(
        rank=rank,
        enz_sets=enz_sets,
        min_substrates=int(cfg.min_substrates),
        n_perm=int(cfg.n_perm),
        seed=int(cfg.random_seed),
    )
    _write_tsv(act, out_ptm)
    outputs["omnipath_ptm_kinase_activity"] = out_ptm
    diagnostics["ptm_activity_rows"] = int(len(act))

    if act.empty:
        LOG.warning("[ptm] activity is empty. Most likely: PTM table lacks *_genesymbol columns or overlap threshold too strict.")

    # ---- Intercell roles ----
    inter_df = load_omnipath_intercell(
        cache_dir=cache,
        refresh=bool(refresh_cache),
        allow_http=bool(cfg.allow_http),
        timeout_s=int(cfg.http_timeout_s),
    )
    diagnostics["intercell_rows"] = int(len(inter_df))
    diagnostics["intercell_cols"] = list(inter_df.columns) if not inter_df.empty else []

    ic_roles = build_intercell_roles(rank, inter_df)
    if not ic_roles.empty:
        _write_tsv(ic_roles, out_ic)
        outputs["omnipath_intercell_roles"] = out_ic
        diagnostics["intercell_roles_rows"] = int(len(ic_roles))
    else:
        diagnostics["intercell_roles_rows"] = 0
        LOG.warning("[intercell] roles empty (no matching genes)")

    # Manifest
    man = {
        "engine": "omnipath_layers",
        "version": "1.2-full-dropin",
        "disease": cfg.disease,
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "ipaa_disease_dir": str(ipaa_dir),
            "evidence_bundle_dir": str(cfg.evidence_bundle_dir) if cfg.evidence_bundle_dir else None,
        },
        "cache_dir": str(cache),
        "params": {
            "allow_http": cfg.allow_http,
            "http_timeout_s": cfg.http_timeout_s,
            "min_substrates": cfg.min_substrates,
            "n_perm": cfg.n_perm,
            "random_seed": cfg.random_seed,
        },
        "diagnostics": diagnostics,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "notes": [
            "PTM is fetched from OmniPath PTMs endpoint with genesymbols to ensure overlap with DE gene symbols.",
            "PTM scoring enforces min_substrates on overlap with rank genes.",
        ],
    }
    _write_json(man, manifest)
    outputs["manifest"] = manifest

    return outputs
