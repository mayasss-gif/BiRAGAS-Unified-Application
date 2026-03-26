#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import deque
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import networkx as nx

from omnipath.requests import Intercell, Enzsub

from .mdp_engine.activity import ipaa_activity
from .mdp_engine.exceptions import DataError, ValidationError
from .mdp_engine.logging_utils import configure_logging, get_logger

log = get_logger("run_engine2_3_mechanistic")



# -----------------------------
# Minimal default cell programs
# -----------------------------
DEFAULT_CELL_PROGRAMS: Dict[str, Set[str]] = {
    "Immune": {"PTPRC", "LST1", "TYROBP", "SPI1", "FCER1G", "LYZ", "HLA-DRA", "HLA-DRB1"},
    "T_cell": {"CD3D", "CD3E", "TRAC", "TRBC1", "IL7R", "LTB", "MALAT1"},
    "B_cell": {"MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA", "CD37", "BANK1"},
    "Myeloid": {"LYZ", "S100A8", "S100A9", "FCN1", "CTSS", "LST1", "TYROBP"},
    "NK": {"NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "KLRD1", "FCGR3A"},
    "Fibroblast": {"COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "TAGLN"},
    "Endothelial": {"PECAM1", "VWF", "KDR", "RAMP2", "EMCN", "ESAM"},
    "Epithelial": {"EPCAM", "KRT8", "KRT18", "KRT19", "MSLN"},
    "CellCycle": {"MKI67", "TOP2A", "HMGB2", "CENPF", "BUB1", "TYMS"},
    "Interferon": {"ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1"},
}

_UNIPROT_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])$")


# =========================
# Shared IO helpers
# =========================
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_tsv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=index)
    tmp.replace(path)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"Missing file: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, engine="python")
    if df.empty:
        raise DataError(f"Empty file: {path}")
    return df


def _read_table_indexed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"Missing file: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, engine="python", index_col=0)
    if df.empty:
        raise DataError(f"Empty file: {path}")
    return df


def _cohort_dir(out_root: Path, disease: str) -> Path:
    d1 = out_root / disease
    d2 = out_root / "cohorts" / disease
    if d1.exists():
        return d1
    if d2.exists():
        return d2
    raise DataError(f"Cannot find cohort folder for '{disease}' in {out_root} (tried {d1} and {d2}).")


def _find_expression_used(cohort: Path) -> Optional[Path]:
    for cand in ("expression_used.tsv", "expression_used.csv", "expression_used.txt"):
        p = cohort / cand
        if p.exists():
            return p
    return None


def _read_expression_gene_set(expr_path: Path) -> Set[str]:
    """
    Cheap gene-set extraction: read header only (nrows=1) to get columns.
    expression_used is indexed; genes should be columns (after orientation fix elsewhere).
    """
    try:
        sep = "\t" if expr_path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(expr_path, sep=sep, engine="python", index_col=0, nrows=1)
        genes = {str(c).strip().upper() for c in df.columns.astype(str)}
        genes.discard("")
        return genes
    except Exception:
        try:
            full = _read_table_indexed(expr_path)
            genes = {str(c).strip().upper() for c in full.columns.astype(str)}
            genes.discard("")
            return genes
        except Exception:
            return set()


def _dedup_gene_columns(expr: pd.DataFrame) -> pd.DataFrame:
    if not expr.columns.duplicated().any():
        return expr
    log.warning("Duplicate gene columns detected; collapsing duplicates by mean.")
    t = expr.T
    t = t.groupby(t.index).mean()
    out = t.T
    out.columns = out.columns.astype(str)
    return out


def _ensure_samples_by_genes(expr: pd.DataFrame) -> pd.DataFrame:
    markers = set()
    for s in DEFAULT_CELL_PROGRAMS.values():
        markers |= set(g.upper() for g in s)

    idx_hits = sum(1 for x in expr.index.astype(str).str.upper() if x in markers)
    col_hits = sum(1 for x in expr.columns.astype(str).str.upper() if x in markers)

    if idx_hits > col_hits:
        log.info("Expression looks like genes x samples; transposing to samples x genes.")
        expr = expr.T

    return expr


def _read_expression(expr_path: Path) -> pd.DataFrame:
    df = _read_table_indexed(expr_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = _ensure_samples_by_genes(df)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    df = _dedup_gene_columns(df)
    return df


def _find_engine1_feature_matrix(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "feature_matrix.tsv"
    return p if p.exists() else None


def _find_engine1_tf_activity(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "tf_activity.tsv"
    return p if p.exists() else None


def _find_engine1_footprints(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "pathway_footprint_activity.tsv"
    return p if p.exists() else None


def _find_engine2_confounding(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "confounding" / disease / "confounding_report.tsv"
    return p if p.exists() else None


# -----------------------------
# Engine3 TF-activity fallback (ONLY for index-only tf_activity.tsv: 0 TF columns)
# -----------------------------
def _find_regulators_evidence(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "evidence_bundle" / disease / "regulators_evidence.tsv"
    return p if p.exists() else None


def _build_tf_activity_fallback(
    out_root: Path,
    disease: str,
    cohort: Path,
    *,
    max_tfs: int,
) -> Optional[pd.DataFrame]:
    """
    Build a minimal TF-activity matrix from expression_used:
    - TF list comes from evidence_bundle/regulators_evidence.tsv
    - TF activity = z-scored expression of TF genes across samples
    Returns samples x TFs with columns prefixed as TF:<symbol>.
    """
    expr_path = _find_expression_used(cohort)
    reg_path = _find_regulators_evidence(out_root, disease)
    if expr_path is None or reg_path is None:
        return None

    expr = _read_expression(expr_path)  # samples x genes, genes uppercased

    reg = pd.read_csv(reg_path, sep="\t", engine="python")
    if reg is None or reg.empty:
        return None

    cols_lc = {str(c).strip().lower(): str(c) for c in reg.columns}
    tf_col = cols_lc.get("regulator") or cols_lc.get("tf") or cols_lc.get("gene") or cols_lc.get("name")
    if tf_col is None:
        tf_col = reg.columns[0]

    tfs = (
        reg[tf_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NA": np.nan, "NAN": np.nan, "NONE": np.nan, "NULL": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    if not tfs:
        return None

    # Keep only TF genes present in expression
    tfs = [t for t in tfs if t in expr.columns]
    if not tfs:
        return None

    # Cap to avoid huge matrices
    tfs = tfs[: int(max_tfs)]

    X = expr[tfs].copy()

    # z-score per TF across samples (avoid sd=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0.0, 1.0)
    Z = (X - mu) / sd

    Z.columns = [f"TF:{c}" for c in Z.columns.astype(str)]
    return Z


def _discover_diseases(out_root: Path) -> List[str]:
    found: Set[str] = set()

    def scan(base: Path) -> None:
        if not base.exists():
            return
        for d in base.iterdir():
            if not d.is_dir():
                continue
            sentinels = ["de_gene_stats.tsv", "pathway_activity.tsv", "pathway_stats.tsv", "labels_used.tsv"]
            if any((d / s).exists() for s in sentinels):
                found.add(d.name)

    scan(out_root)
    scan(out_root / "cohorts")
    return sorted(found)


def _parse_diseases_arg(values: Optional[List[str]]) -> List[str]:
    """
    Accepts repeated --disease and supports comma-separated values:
      --disease RA --disease SLE
      --disease RA,SLE
    """
    if not values:
        return []
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    # de-dup, preserve order
    seen: Set[str] = set()
    uniq: List[str] = []
    for d in out:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


# ==========================================================
# OmniPath Layer (PTM + Intercell + Build PKN edges for E3)
# ==========================================================
@dataclass(frozen=True)
class OmniPathLayerResult:
    disease: str
    status: str
    out_dir: Path
    ptm_activity: Path
    intercell_roles: Path
    pkn_edges_global: Path
    manifest: Path
    skipped_flag: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_de_table_for_disease(cohort: Path) -> Optional[Path]:
    candidates = [
        cohort / "de_gene_stats.tsv",
        cohort / "de_gene_stats.csv",
        cohort / "degs_from_counts.csv",
        cohort / "DEGs.tsv",
        cohort / "DEGs.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_de_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, engine="python")
    if df.empty:
        raise DataError(f"Empty DE table: {path}")
    return df


def _standardize_de_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in out.columns}

    gene = cols.get("gene") or cols.get("genes") or cols.get("symbol") or cols.get("genesymbol") or cols.get("hgnc")
    lfc = cols.get("log2foldchange") or cols.get("log2fc") or cols.get("logfc") or cols.get("lfc")
    padj = cols.get("padj") or cols.get("adjpval") or cols.get("adjpvalue") or cols.get("fdr") or cols.get("qvalue")
    pval = cols.get("pvalue") or cols.get("pval") or cols.get("p")

    if gene is None:
        raise DataError("DE table missing gene column (Gene/SYMBOL/GeneSymbol).")
    if lfc is None:
        raise DataError("DE table missing log2FC column (log2FoldChange/log2FC/logFC).")

    ren = {gene: "Gene", lfc: "log2FoldChange"}
    if padj is not None:
        ren[padj] = "padj"
    if pval is not None:
        ren[pval] = "pvalue"

    out = out.rename(columns=ren)

    out["Gene"] = out["Gene"].astype(str).str.strip().str.upper()
    out["log2FoldChange"] = pd.to_numeric(out["log2FoldChange"], errors="coerce").fillna(0.0)
    if "padj" in out.columns:
        out["padj"] = pd.to_numeric(out["padj"], errors="coerce")
    if "pvalue" in out.columns:
        out["pvalue"] = pd.to_numeric(out["pvalue"], errors="coerce")

    return out


def _compute_rank_from_de(de: pd.DataFrame) -> pd.Series:
    df = de.copy()
    use_p = "padj" if "padj" in df.columns else ("pvalue" if "pvalue" in df.columns else None)

    if use_p is None:
        score = df["log2FoldChange"].fillna(0.0).to_numpy(dtype=float)
    else:
        p = df[use_p].copy()
        p = p.fillna(1.0).clip(lower=1e-300)
        score = (-np.log10(p.to_numpy(dtype=float))) * np.sign(df["log2FoldChange"].to_numpy(dtype=float))

    s = pd.Series(score, index=df["Gene"].astype(str).values)
    s = s.groupby(s.index).apply(lambda x: x.loc[x.abs().idxmax()] if len(x) > 1 else float(x.iloc[0]))
    s = s.sort_values(ascending=False)
    return s


def _to_float01(series: pd.Series) -> np.ndarray:
    if series is None:
        return np.zeros(0, dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    s = series.astype(str).str.strip().str.lower()
    true_set = {"true", "t", "yes", "y", "1"}
    false_set = {"false", "f", "no", "n", "0", "", "nan", "none"}
    out = np.zeros(len(s), dtype=float)
    out[s.isin(true_set)] = 1.0
    out[s.isin(false_set)] = 0.0

    mask_other = ~(s.isin(true_set) | s.isin(false_set))
    if mask_other.any():
        out[mask_other.to_numpy()] = pd.to_numeric(s[mask_other], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return out


# ------------- OmniPath PTM (kinase/enzyme activity) ----------------
def _ptm_has_genesymbol_cols(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    return any(
        c in cols
        for c in (
            "enzyme_genesymbol",
            "substrate_genesymbol",
            "enzyme_genesymbols",
            "substrate_genesymbols",
            "enzymes_genesymbol",
            "substrates_genesymbol",
        )
    )


def _looks_like_uniprot(x: object) -> bool:
    s = str(x).strip().upper()
    return bool(_UNIPROT_RE.match(s))


def _fetch_ptm_via_client() -> Optional[pd.DataFrame]:
    """
    Prefer OmniPath python client if available.
    CRITICAL: request genesymbols=1/True so we get enzyme_genesymbol/substrate_genesymbol columns.
    """
    try:
        

        for val in (True, 1, "1", "yes", "YES"):
            try:
                df = Enzsub.get(genesymbols=val)
                if df is not None and not df.empty:
                    if _ptm_has_genesymbol_cols(df):
                        log.info("[omnipath:ptm] fetched via python client (genesymbols=%s)", val)
                        return df
                    df2 = df
                else:
                    df2 = None
            except Exception:
                df2 = None

        try:
            df = Enzsub.get()
            if df is not None and not df.empty:
                log.info("[omnipath:ptm] fetched via python client (no genesymbols)")
                return df
        except Exception:
            pass

        return None
    except Exception as e:
        log.info("[omnipath:ptm] client fetch failed: %s", e)
        return None


def _fetch_ptm_via_http(timeout: int = 120) -> Optional[pd.DataFrame]:
    """
    HTTP fallback.

    OmniPath webservice expects genesymbols=1 (not 'yes') to include *_genesymbol columns.
    """
    if requests is None:
        return None

    urls = [
        "https://omnipathdb.org/enz_sub?format=tsv&genesymbols=1",
        "https://omnipathdb.org/ptms?format=tsv&genesymbols=1",
        "https://omnipathdb.org/queries/enz_sub?format=tsv&genesymbols=1",
        "https://omnipathdb.org/queries/ptms?format=tsv&genesymbols=1",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=int(timeout))
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt:
                continue
            if "Unknown argument" in txt or "Something is not entirely good" in txt:
                log.info("[omnipath:ptm] endpoint returned error text (skipping): %s", url)
                continue
            df = pd.read_csv(StringIO(r.text), sep="\t")
            if df is not None and not df.empty:
                log.info("[omnipath:ptm] fetched via HTTP: %s", url)
                return df
        except Exception as e:
            log.debug("[omnipath:ptm] HTTP fetch failed %s: %s", url, e)
    return None


def load_omnipath_ptm(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """
    Loads OmniPath PTM (enzyme-substrate) table and caches it as omnipath_ptm.csv.

    Auto-heals bad caches:
    - If cache exists but contains only UniProt IDs (no *_genesymbol columns), we refetch
      even if refresh=False.
    """
    _ensure_dir(cache_dir)
    cache_fp = cache_dir / "omnipath_ptm.csv"

    if cache_fp.exists() and not refresh:
        try:
            df = pd.read_csv(cache_fp)
            if df is not None and not df.empty:
                if _ptm_has_genesymbol_cols(df):
                    return df

                cols = {str(c).strip().lower(): str(c) for c in df.columns}
                enz = cols.get("enzyme")
                sub = cols.get("substrate")
                if enz and sub:
                    sample = pd.concat(
                        [df[enz].head(200).astype(str), df[sub].head(200).astype(str)], ignore_index=True
                    )
                    uni_frac = float(np.mean([_looks_like_uniprot(x) for x in sample.tolist()])) if len(sample) else 0.0
                    if uni_frac >= 0.50:
                        log.warning(
                            "[omnipath:ptm] cached PTM looks UniProt-only (no genesymbol cols). "
                            "Auto-refetching with genesymbols=1. Cache=%s",
                            cache_fp,
                        )
                    else:
                        return df
        except Exception:
            pass

    df = _fetch_ptm_via_client()
    if df is None or df.empty or (not _ptm_has_genesymbol_cols(df) and df is not None and not df.empty):
        df_http = _fetch_ptm_via_http()
        if df_http is not None and not df_http.empty:
            df = df_http

    if df is None or df.empty:
        return pd.DataFrame()

    df.to_csv(cache_fp, index=False)
    return df


def _split_symbols(x: object) -> List[str]:
    s = str(x).strip().upper()
    if not s or s in {"NA", "NAN", "NONE", "NULL"}:
        return []
    for delim in [";", ",", "|", "/", "\\"]:
        s = s.replace(delim, " ")
    toks = [t for t in s.split() if t and t not in {"NA", "NAN", "NONE", "NULL"}]
    return toks


def _standardize_ptm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer enzyme_genesymbol/substrate_genesymbol if available.
    If only enzyme/substrate exist and they look like UniProt IDs, return empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["enzyme", "substrate", "residue"])

    cols = {str(c).strip().lower(): str(c) for c in df.columns}

    enz_gs = cols.get("enzyme_genesymbol") or cols.get("enzyme_genesymbols")
    sub_gs = cols.get("substrate_genesymbol") or cols.get("substrate_genesymbols")

    enz_raw = cols.get("enzyme")
    sub_raw = cols.get("substrate")

    residue = cols.get("residue")

    if enz_gs and sub_gs:
        enz_series = df[enz_gs]
        sub_series = df[sub_gs]
    else:
        if enz_raw is None or sub_raw is None:
            return pd.DataFrame(columns=["enzyme", "substrate", "residue"])
        enz_series = df[enz_raw]
        sub_series = df[sub_raw]

        sample = pd.concat(
            [enz_series.head(200).astype(str), sub_series.head(200).astype(str)],
            ignore_index=True,
        )
        uni_frac = float(np.mean([_looks_like_uniprot(x) for x in sample.tolist()])) if len(sample) else 0.0
        if uni_frac >= 0.50:
            log.warning(
                "[omnipath:ptm] PTM table appears UniProt-only (enzyme/substrate are UniProt IDs). "
                "You MUST fetch with genesymbols=1."
            )
            return pd.DataFrame(columns=["enzyme", "substrate", "residue"])

    out = pd.DataFrame()
    out["enzyme"] = enz_series.astype(str).str.strip().str.upper()
    out["substrate"] = sub_series.astype(str).str.strip().str.upper()
    out["residue"] = df[residue].astype(str) if (residue and residue in df.columns) else ""

    out = out[
        (out["enzyme"] != "")
        & (out["substrate"] != "")
        & (out["enzyme"].str.lower() != "nan")
        & (out["substrate"].str.lower() != "nan")
    ]
    return out


def build_enzyme_sets(ptm_df: pd.DataFrame) -> Dict[str, Set[str]]:
    pdf = _standardize_ptm_columns(ptm_df)
    if pdf.empty:
        return {}

    subs: Dict[str, Set[str]] = {}
    for _, r in pdf.iterrows():
        enzymes = _split_symbols(r["enzyme"])
        substrates = _split_symbols(r["substrate"])
        if not enzymes or not substrates:
            continue
        for enz in enzymes:
            subs.setdefault(enz, set()).update(substrates)
    return subs


def score_enzyme_activity(
    rank: pd.Series,
    enz_sets: Dict[str, Set[str]],
    *,
    min_substrates: int = 5,
    n_perm: int = 200,
    seed: int = 17,
) -> pd.DataFrame:
    rng = random.Random(int(seed))
    values = rank.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame(columns=["enzyme", "substrates", "raw_mean", "mu_null", "sd_null", "NES", "hit_substrates"])

    rows: List[Dict[str, object]] = []

    for enz, subs in enz_sets.items():
        subs = set(subs)

        hit = rank.reindex(list(subs)).dropna().to_numpy(dtype=float)
        if len(hit) < int(min_substrates):
            continue

        raw = float(np.mean(hit))
        k = len(hit)

        null_means: List[float] = []
        for _ in range(int(n_perm)):
            idx = rng.sample(range(len(values)), k)
            null_means.append(float(np.mean(values[idx])))

        mu = float(np.mean(null_means)) if null_means else 0.0
        sd = float(np.std(null_means)) if null_means else 1.0
        if sd <= 1e-8:
            sd = 1.0
        nes = (raw - mu) / sd

        rows.append(
            {
                "enzyme": enz,
                "substrates": int(len(subs)),
                "hit_substrates": int(k),
                "raw_mean": raw,
                "mu_null": mu,
                "sd_null": sd,
                "NES": float(nes),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["enzyme", "substrates", "raw_mean", "mu_null", "sd_null", "NES", "hit_substrates"])

    return pd.DataFrame(rows).sort_values("NES", ascending=False).reset_index(drop=True)


# ------------- OmniPath Intercell roles (robust) ----------------
def _fetch_intercell_via_client() -> Optional[pd.DataFrame]:
    try:
        df = Intercell.get()
        if df is not None and not df.empty:
            log.info("[omnipath:intercell] fetched via python client")
            return df
    except Exception as e:
        log.info("[omnipath:intercell] client fetch failed: %s", e)
    return None


def _fetch_intercell_http(timeout: int = 120) -> Optional[pd.DataFrame]:
    if requests is None:
        return None

    urls = [
        "https://omnipathdb.org/intercell?format=tsv",
        "https://omnipathdb.org/queries/intercell?format=tsv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()

            txt = (r.text or "").strip()
            if not txt:
                continue
            if "Unknown argument" in txt or "Something is not entirely good" in txt:
                log.info("[omnipath:intercell] endpoint returned error text (skipping): %s", url)
                continue

            df = pd.read_csv(StringIO(r.text), sep="\t")
            if df is not None and not df.empty:
                log.info("[omnipath:intercell] fetched via HTTP: %s", url)
                return df
        except Exception as e:
            log.debug("[omnipath:intercell] HTTP fetch failed %s: %s", url, e)
    return None


def load_omnipath_intercell(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache_csv = cache_dir / "omnipath_intercell.csv"

    def _has_gene_col(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        cols = {str(c).strip().lower() for c in df.columns}
        return any(
            c in cols
            for c in (
                "genesymbol",
                "gene_symbol",
                "genesymbols",
                "target_genesymbol",
                "source_genesymbol",
                "uniprot",
                "uniprot_id",
                "uniprotkb",
                "uniprotkb_id",
            )
        )

    if cache_csv.exists() and not refresh:
        try:
            df = pd.read_csv(cache_csv)
            if _has_gene_col(df):
                return df
            log.warning("[omnipath:intercell] cached file lacks gene column; ignoring cache: %s", cache_csv)
        except Exception:
            pass

    df = _fetch_intercell_via_client()
    if df is None or df.empty:
        df = _fetch_intercell_http()

    if df is None or df.empty or not _has_gene_col(df):
        return pd.DataFrame()

    df.to_csv(cache_csv, index=False)
    return df


def _standardize_intercell(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}

    gene_col = (
        cols.get("genesymbol")
        or cols.get("gene_symbol")
        or cols.get("genesymbols")
        or cols.get("target_genesymbol")
        or cols.get("source_genesymbol")
        or cols.get("uniprot")
        or cols.get("uniprot_id")
        or cols.get("uniprotkb")
        or cols.get("uniprotkb_id")
    )
    if gene_col is None:
        raise DataError("[intercell] cannot find genesymbol/uniprot column")

    category_col = cols.get("category")
    parent_col = cols.get("parent")
    generic_col = cols.get("generic_category") or cols.get("generic_cat")

    def _safe(col: Optional[str]) -> pd.Series:
        if col and col in df.columns:
            return df[col].astype(str)
        return pd.Series([""] * len(df), index=df.index)

    out = pd.DataFrame()
    out["Gene"] = df[gene_col].astype(str).str.strip().str.upper()
    out["category"] = _safe(category_col)
    out["parent"] = _safe(parent_col)
    out["generic_cat"] = _safe(generic_col)
    return out


def _agg_unique(series: pd.Series) -> str:
    vals = sorted({v for v in series.astype(str).tolist() if v and v != "nan"})
    return ";".join(vals)


# ------------- Build global PKN edges for Engine3 ----------------
def _fetch_interactions_http(timeout: int = 180) -> Optional[pd.DataFrame]:
    if requests is None:
        return None
    url = "https://omnipathdb.org/interactions?format=tsv&genesymbols=1"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep="\t")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.info("[omnipath:interactions] HTTP fetch failed: %s", e)
    return None


def load_omnipath_interactions(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache_fp = cache_dir / "omnipath_interactions.tsv"
    if cache_fp.exists() and not refresh:
        try:
            df = pd.read_csv(cache_fp, sep="\t")
            if not df.empty:
                return df
        except Exception:
            pass

    df = _fetch_interactions_http()
    if df is None or df.empty:
        return pd.DataFrame()

    df.to_csv(cache_fp, sep="\t", index=False)
    return df


def _normalize_pkn_edges_from_any(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source", "target", "sign"])

    cols = {str(c).strip().lower(): str(c) for c in df.columns}

    if "source" in cols and "target" in cols and ("source_genesymbol" not in cols and "target_genesymbol" not in cols):
        src = cols["source"]
        dst = cols["target"]
        out = pd.DataFrame()
        out["source"] = df[src].astype(str).str.strip().str.upper()
        out["target"] = df[dst].astype(str).str.strip().str.upper()

        sign_col = cols.get("sign")
        if sign_col:
            s = pd.to_numeric(df[sign_col], errors="coerce").fillna(0.0)
            out["sign"] = s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype(int)
        else:
            out["sign"] = 0
        out = out[(out["source"] != "") & (out["target"] != "")]
        out = out.drop_duplicates()
        return out[["source", "target", "sign"]]

    src = cols.get("source_genesymbol") or cols.get("sourcegenesymbol") or cols.get("source")
    dst = cols.get("target_genesymbol") or cols.get("targetgenesymbol") or cols.get("target")
    if src is None or dst is None:
        raise DataError("Could not normalize PKN edges: missing source/target gene symbol columns.")

    out = pd.DataFrame()
    out["source"] = df[src].astype(str).str.strip().str.upper()
    out["target"] = df[dst].astype(str).str.strip().str.upper()

    directed_col = cols.get("is_directed") or cols.get("isdirected")
    if directed_col and directed_col in df.columns:
        try:
            is_dir = _to_float01(df[directed_col])
            out = out.loc[is_dir.astype(int) == 1].copy()
            df = df.loc[out.index].copy()
        except Exception:
            pass

    dir_col = cols.get("consensus_direction") or cols.get("direction") or cols.get("consensusdirection")
    stim = cols.get("consensus_stimulation") or cols.get("stimulation") or cols.get("is_stimulation")
    inhib = cols.get("consensus_inhibition") or cols.get("inhibition") or cols.get("is_inhibition")
    effect = cols.get("effect") or cols.get("interaction") or cols.get("consensus_effect")

    sign = np.zeros(len(out), dtype=int)

    if dir_col and dir_col in df.columns:
        d = pd.to_numeric(df[dir_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sign[d > 0] = 1
        sign[d < 0] = -1
    elif stim and stim in df.columns and inhib and inhib in df.columns:
        st = _to_float01(df[stim])
        ih = _to_float01(df[inhib])
        sign[(st > 0) & (ih <= 0)] = 1
        sign[(ih > 0) & (st <= 0)] = -1
    elif effect and effect in df.columns:
        txt = df[effect].astype(str).str.lower()
        sign[txt.str.contains("activ")] = 1
        sign[txt.str.contains("inhib")] = -1

    out["sign"] = sign.astype(int)

    out = out[(out["source"] != "") & (out["target"] != "")]
    out = out.drop_duplicates(subset=["source", "target", "sign"])
    return out[["source", "target", "sign"]]


def build_global_pkn_edges(
    out_root: Path,
    omnipath_cache_dir: Path,
    *,
    refresh: bool = False,
    signor_edges_path: Optional[Path] = None,
    strict: bool = False,
) -> Tuple[bool, Optional[Path], str]:
    pkn_cache = out_root / "engines" / "pkn_cache"
    _ensure_dir(pkn_cache)
    pkn_out = pkn_cache / "pkn_edges.tsv"

    if pkn_out.exists() and not refresh:
        return True, pkn_out, "using existing pkn_edges.tsv"

    df_int = load_omnipath_interactions(omnipath_cache_dir, refresh=refresh)
    if df_int.empty:
        msg = "OmniPath interactions not available (network/cache). Cannot build PKN."
        if strict:
            raise DataError(msg)
        return False, None, msg

    try:
        edges = _normalize_pkn_edges_from_any(df_int)
    except Exception as e:
        msg = f"Failed normalizing OmniPath interactions into edges: {type(e).__name__}: {e}"
        if strict:
            raise
        return False, None, msg

    if signor_edges_path is not None:
        if signor_edges_path.exists():
            try:
                sdf = pd.read_csv(
                    signor_edges_path,
                    sep="\t" if signor_edges_path.suffix.lower() in {".tsv", ".txt"} else ",",
                )
                sedges = _normalize_pkn_edges_from_any(sdf)
                edges = pd.concat([edges, sedges], ignore_index=True)
                edges = edges.drop_duplicates(subset=["source", "target", "sign"])
            except Exception as e:
                log.warning("SIGNOR merge failed (ignored): %s", e)
        else:
            log.warning("SIGNOR edges path does not exist (ignored): %s", signor_edges_path)

    edges.to_csv(pkn_out, sep="\t", index=False)

    n_edges = int(len(edges))
    n_signed = int((edges["sign"].astype(int) != 0).sum()) if n_edges > 0 and "sign" in edges.columns else 0
    frac = (float(n_signed) / float(n_edges)) if n_edges else 0.0
    log.info("[pkn] edges=%d signed=%d (%.3f)", n_edges, n_signed, frac)
    if frac < 0.01:
        log.warning("[pkn] Very low signed-edge fraction. Engine3 may need unsigned fallback.")

    return True, pkn_out, "built/updated pkn_edges.tsv"


def run_omnipath_layer_for_disease(
    out_root: Path,
    disease: str,
    *,
    refresh_cache: bool = False,
    build_pkn: bool = True,
    refresh_pkn: bool = False,
    signor_edges_path: Optional[Path] = None,
    ptm_min_substrates: int = 5,
    ptm_n_perm: int = 200,
    strict: bool = False,
) -> OmniPathLayerResult:
    out_root = out_root.resolve()
    cohort = _cohort_dir(out_root, disease)

    out_dir = out_root / "engines" / "omnipath_layer" / disease
    _ensure_dir(out_dir)

    omnipath_cache_dir = out_root / "engines" / "omnipath_cache"
    _ensure_dir(omnipath_cache_dir)

    res = OmniPathLayerResult(
        disease=disease,
        status="ok",
        out_dir=out_dir,
        ptm_activity=out_dir / f"{disease}_PTM_kinase_activity.tsv",
        intercell_roles=out_dir / f"{disease}_Intercell_roles.tsv",
        pkn_edges_global=out_root / "engines" / "pkn_cache" / "pkn_edges.tsv",
        manifest=out_dir / "ENGINE_MANIFEST.json",
        skipped_flag=out_dir / "SKIPPED.txt",
    )

    de_path = _find_de_table_for_disease(cohort)
    if de_path is None:
        msg = f"OmniPath layer skipped: no DE table found in {cohort} (expected de_gene_stats.tsv or DEGs.*)"
        if strict:
            raise DataError(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "omnipath_layer", "status": "skipped", "reason": msg}, indent=2))
        return OmniPathLayerResult(**{**res.__dict__, "status": "skipped"})

    try:
        de_raw = _read_de_table(de_path)
        de = _standardize_de_columns(de_raw)
        rank = _compute_rank_from_de(de)
    except Exception as e:
        msg = f"OmniPath layer skipped: DE parsing/ranking failed ({type(e).__name__}): {e}"
        if strict:
            raise
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "omnipath_layer", "status": "skipped", "reason": msg}, indent=2))
        return OmniPathLayerResult(**{**res.__dict__, "status": "skipped"})

    ptm_status = "ok"
    ptm_debug: Dict[str, object] = {}
    try:
        ptm_df = load_omnipath_ptm(omnipath_cache_dir, refresh=refresh_cache)
        ptm_debug["ptm_rows"] = int(len(ptm_df)) if ptm_df is not None else 0
        ptm_debug["ptm_cols"] = list(ptm_df.columns) if ptm_df is not None else []

        enz_sets = build_enzyme_sets(ptm_df)
        ptm_debug["enzymes_total"] = int(len(enz_sets))

        if enz_sets:
            subs_all = set().union(*enz_sets.values()) if enz_sets else set()
            hits = rank.index.intersection(pd.Index(list(subs_all)))
            ptm_debug["substrates_total_unique"] = int(len(subs_all))
            ptm_debug["substrates_overlap_with_rank"] = int(len(hits))

        if enz_sets:
            act = score_enzyme_activity(
                rank,
                enz_sets,
                min_substrates=int(ptm_min_substrates),
                n_perm=int(ptm_n_perm),
                seed=17,
            )
            _atomic_write_tsv(act, res.ptm_activity, index=False)
            if act.empty:
                ptm_status = "skipped_no_overlap"
        else:
            ptm_status = "skipped_no_ptm_or_uniprot_only"
            _atomic_write_tsv(pd.DataFrame(), res.ptm_activity, index=False)

        _atomic_write_text(out_dir / "PTM_DEBUG.json", json.dumps(ptm_debug, indent=2))
    except Exception as e:
        ptm_status = f"failed:{type(e).__name__}"
        log.warning("PTM activity failed for %s (continuing): %s", disease, e)
        _atomic_write_tsv(pd.DataFrame(), res.ptm_activity, index=False)
        _atomic_write_text(out_dir / "PTM_DEBUG.json", json.dumps({**ptm_debug, "error": str(e)}, indent=2))

    ic_status = "ok"
    try:
        ic_df = load_omnipath_intercell(omnipath_cache_dir, refresh=refresh_cache)
        if ic_df.empty:
            ic_status = "skipped_no_intercell"
            _atomic_write_tsv(pd.DataFrame(), res.intercell_roles, index=False)
        else:
            ic_std = _standardize_intercell(ic_df)
            gene_universe = set(rank.index.astype(str))
            ic_std = ic_std[ic_std["Gene"].isin(gene_universe)].copy()
            if ic_std.empty:
                ic_status = "skipped_no_overlap"
                _atomic_write_tsv(pd.DataFrame(), res.intercell_roles, index=False)
            else:
                grouped = ic_std.groupby("Gene").agg(
                    {
                        "category": _agg_unique,
                        "parent": _agg_unique,
                        "generic_cat": _agg_unique,
                    }
                ).reset_index()
                grouped["rank_score"] = grouped["Gene"].map(rank.to_dict())
                grouped.to_csv(res.intercell_roles, sep="\t", index=False)
    except Exception as e:
        ic_status = f"failed:{type(e).__name__}"
        log.warning("Intercell roles failed for %s (continuing): %s", disease, e)
        _atomic_write_tsv(pd.DataFrame(), res.intercell_roles, index=False)

    pkn_status = "not_requested"
    pkn_msg = ""
    if build_pkn:
        ok, pkn_path, msg = build_global_pkn_edges(
            out_root=out_root,
            omnipath_cache_dir=omnipath_cache_dir,
            refresh=bool(refresh_pkn),
            signor_edges_path=signor_edges_path,
            strict=bool(strict),
        )
        pkn_status = "ok" if ok else "skipped"
        pkn_msg = msg
        if pkn_path is not None:
            res = OmniPathLayerResult(**{**res.__dict__, "pkn_edges_global": pkn_path})

    manifest = {
        "engine": "omnipath_layer",
        "version": "1.2.0",
        "status": "ok",
        "inputs": {
            "out_root": str(out_root),
            "disease": disease,
            "cohort_dir": str(cohort),
            "de_table": str(de_path),
        },
        "params": {
            "refresh_cache": bool(refresh_cache),
            "build_pkn": bool(build_pkn),
            "refresh_pkn": bool(refresh_pkn),
            "ptm_min_substrates": int(ptm_min_substrates),
            "ptm_n_perm": int(ptm_n_perm),
        },
        "substatus": {
            "ptm": ptm_status,
            "intercell": ic_status,
            "pkn": pkn_status,
            "pkn_msg": pkn_msg,
        },
        "outputs": {
            "ptm_activity": str(res.ptm_activity),
            "intercell_roles": str(res.intercell_roles),
            "pkn_edges_global": str(res.pkn_edges_global) if res.pkn_edges_global else None,
            "ptm_debug": str(out_dir / "PTM_DEBUG.json"),
        },
        "notes": [
            "PTM and intercell are evidence layers.",
            "PTM FIX: PTM fetching uses genesymbols=1; UniProt-only caches are auto-healed.",
        ],
    }
    _atomic_write_text(res.manifest, json.dumps(manifest, indent=2))

    if res.skipped_flag.exists():
        try:
            res.skipped_flag.unlink()
        except Exception:
            pass

    return res


# =========================
# Engine 2: Confounding
# ==========================
@dataclass(frozen=True)
class Engine2Result:
    disease: str
    status: str
    out_dir: Path
    cell_type_scores: Path
    confounding_report: Path
    manifest: Path
    skipped_flag: Path


def run_engine2_confounding(
    out_root: Path,
    disease: str,
    *,
    corr_method: str = "spearman",
    corr_flag_threshold: float = 0.40,
    min_markers: int = 5,
    strict: bool = False,
) -> Engine2Result:
    out_root = out_root.resolve()
    cohort = _cohort_dir(out_root, disease)

    out_dir = out_root / "engines" / "confounding" / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    res = Engine2Result(
        disease=disease,
        status="ok",
        out_dir=out_dir,
        cell_type_scores=out_dir / "cell_type_scores.tsv",
        confounding_report=out_dir / "confounding_report.tsv",
        manifest=out_dir / "ENGINE_MANIFEST.json",
        skipped_flag=out_dir / "SKIPPED.txt",
    )

    expr_path = _find_expression_used(cohort)
    if expr_path is None:
        msg = f"Engine2 skipped: No expression_used.* found in {cohort}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_tsv(pd.DataFrame(), res.cell_type_scores, index=True)
        _atomic_write_tsv(
            pd.DataFrame(
                columns=[
                    "feature",
                    "feature_type",
                    "max_abs_corr",
                    "top_cell_program",
                    "top_corr",
                    "penalty",
                    "flag_high",
                ]
            ),
            res.confounding_report,
            index=False,
        )
        _atomic_write_text(
            res.manifest,
            json.dumps(
                {
                    "engine": "confounding",
                    "version": "1.0.0",
                    "status": "skipped",
                    "reason": msg,
                    "inputs": {"cohort_dir": str(cohort)},
                    "outputs": {"skipped_flag": str(res.skipped_flag)},
                },
                indent=2,
            ),
        )
        return Engine2Result(**{**res.__dict__, "status": "skipped"})

    expr = _read_expression(expr_path)
    programs = {k: {g.upper() for g in v} for k, v in DEFAULT_CELL_PROGRAMS.items()}

    try:
        scores = ipaa_activity(
            expression=expr,
            pathways=programs,
            method="mean",
            standardize_pathways=True,
            min_size=int(min_markers),
        )
    except Exception as e:
        msg = f"Engine2 skipped: cell program scoring failed ({type(e).__name__}): {e}"
        if strict:
            raise
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_tsv(pd.DataFrame(), res.cell_type_scores, index=True)
        _atomic_write_tsv(
            pd.DataFrame(
                columns=[
                    "feature",
                    "feature_type",
                    "max_abs_corr",
                    "top_cell_program",
                    "top_corr",
                    "penalty",
                    "flag_high",
                ]
            ),
            res.confounding_report,
            index=False,
        )
        _atomic_write_text(
            res.manifest,
            json.dumps(
                {
                    "engine": "confounding",
                    "version": "1.0.0",
                    "status": "skipped",
                    "reason": msg,
                    "inputs": {"expression_used": str(expr_path)},
                    "outputs": {"skipped_flag": str(res.skipped_flag)},
                },
                indent=2,
            ),
        )
        return Engine2Result(**{**res.__dict__, "status": "skipped"})

    scores.columns = [f"CELL:{c}" for c in scores.columns.astype(str)]
    _atomic_write_tsv(scores, res.cell_type_scores, index=True)

    feature_matrix_path = _find_engine1_feature_matrix(out_root, disease)
    if feature_matrix_path is None:
        rep = pd.DataFrame(
            columns=["feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"]
        )
        _atomic_write_tsv(rep, res.confounding_report, index=False)
    else:
        feat = pd.read_csv(feature_matrix_path, sep="\t", index_col=0)
        feat.index = feat.index.astype(str).str.strip()
        feat = feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        common = scores.index.intersection(feat.index)
        if len(common) < 5:
            rep = pd.DataFrame(
                columns=["feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"]
            )
            _atomic_write_tsv(rep, res.confounding_report, index=False)
        else:
            S = scores.loc[common]
            F = feat.loc[common]
            corr = pd.concat([S, F], axis=1).corr(method=corr_method).loc[S.columns, F.columns]

            rows: List[Dict[str, object]] = []
            for f in corr.columns:
                cvec = corr[f].astype(float)
                if cvec.empty or cvec.isna().all():
                    continue
                top_cell = str(cvec.abs().idxmax())
                top_corr = float(cvec.loc[top_cell])
                max_abs = float(abs(top_corr))
                ftype = "TF" if str(f).startswith("TF:") else ("PW" if str(f).startswith("PW:") else "OTHER")
                rows.append(
                    {
                        "feature": str(f),
                        "feature_type": ftype,
                        "max_abs_corr": max_abs,
                        "top_cell_program": top_cell,
                        "top_corr": top_corr,
                        "penalty": max_abs,
                        "flag_high": int(max_abs >= float(corr_flag_threshold)),
                    }
                )

            rep = pd.DataFrame(rows).sort_values(["flag_high", "max_abs_corr"], ascending=[False, False]).reset_index(
                drop=True
            )
            _atomic_write_tsv(rep, res.confounding_report, index=False)

    _atomic_write_text(
        res.manifest,
        json.dumps(
            {
                "engine": "confounding",
                "version": "1.0.0",
                "status": "ok",
                "inputs": {
                    "out_root": str(out_root),
                    "disease": disease,
                    "cohort_dir": str(cohort),
                    "expression_used": str(expr_path),
                    "feature_matrix": str(feature_matrix_path) if feature_matrix_path else None,
                },
                "params": {
                    "corr_method": corr_method,
                    "corr_flag_threshold": float(corr_flag_threshold),
                    "min_markers": int(min_markers),
                    "strict": bool(strict),
                },
                "outputs": {
                    "cell_type_scores": str(res.cell_type_scores),
                    "confounding_report": str(res.confounding_report),
                },
            },
            indent=2,
        ),
    )

    if res.skipped_flag.exists():
        try:
            res.skipped_flag.unlink()
        except Exception:
            pass

    return res


# =========================
# Engine 3: Contextualization
# ==========================
@dataclass(frozen=True)
class Engine3Result:
    disease: str
    status: str
    out_dir: Path
    pkn_edges: Path
    causal_edges: Path
    causal_nodes: Path
    drivers_ranked: Path
    mechanism_cards_json: Path
    mechanism_cards_tsv: Path
    graphml: Path
    manifest: Path
    skipped_flag: Path


def _find_pkn_edges(out_root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override if override.exists() else None
    candidates = [
        out_root / "engines" / "pkn_cache" / "pkn_edges.tsv",
        out_root / "data" / "omnipath_cache" / "edges.tsv",
        out_root / "pkn_edges.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _normalize_pkn_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    src = low.get("source") or low.get("src") or low.get("from") or low.get("a")
    dst = low.get("target") or low.get("dst") or low.get("to") or low.get("b")
    if src is None or dst is None:
        raise DataError("PKN edges must include source/target (or src/dst).")

    df = df.rename(columns={src: "source", dst: "target"})
    df["source"] = df["source"].astype(str).str.strip().str.upper()
    df["target"] = df["target"].astype(str).str.strip().str.upper()

    sign_col = low.get("sign")
    inter_col = low.get("interaction") or low.get("effect")

    if sign_col is not None:
        s = pd.to_numeric(df[sign_col], errors="coerce").fillna(0.0)
        df["sign"] = s
    elif inter_col is not None:
        txt = df[inter_col].astype(str).str.lower()
        df["sign"] = 0.0
        df.loc[txt.str.contains("activ"), "sign"] = 1.0
        df.loc[txt.str.contains("inhib"), "sign"] = -1.0
    else:
        df["sign"] = 0.0

    df["sign"] = df["sign"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df = df[(df["source"] != "") & (df["target"] != "")]
    df = df.drop_duplicates(subset=["source", "target", "sign"])
    return df[["source", "target", "sign"]]


def _build_incoming(edges: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    inc: Dict[str, List[Tuple[str, int]]] = {}
    for _, r in edges.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        sg = int(r["sign"])
        if sg == 0:
            continue
        inc.setdefault(t, []).append((s, sg))
    return inc


def _build_incoming_unsigned(edges: pd.DataFrame) -> Dict[str, List[str]]:
    inc: Dict[str, List[str]] = {}
    for _, r in edges.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        inc.setdefault(t, []).append(s)
    return inc


def _bfs_upstream(incoming: Dict[str, List[Tuple[str, int]]], tf: str, max_steps: int) -> Dict[Tuple[str, int], int]:
    best: Dict[Tuple[str, int], int] = {(tf, 1): 0}
    q = deque([(tf, 1, 0)])

    while q:
        node, sign_to_tf, depth = q.popleft()
        if depth >= max_steps:
            continue
        for src, esign in incoming.get(node, []):
            new_sign = int(sign_to_tf * esign)
            st = (src, new_sign)
            nd = depth + 1
            if st not in best or nd < best[st]:
                best[st] = nd
                q.append((src, new_sign, nd))
    return best


def _bfs_upstream_unsigned(incoming: Dict[str, List[str]], tf: str, max_steps: int) -> Dict[str, int]:
    best: Dict[str, int] = {tf: 0}
    q = deque([(tf, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= max_steps:
            continue
        for src in incoming.get(node, []):
            nd = depth + 1
            if src not in best or nd < best[src]:
                best[src] = nd
                q.append((src, nd))
    return best


def run_engine3_contextualization(
    out_root: Path,
    disease: str,
    *,
    pkn_edges_override: Optional[Path] = None,
    max_steps: int = 3,
    top_tfs: int = 30,
    confound_penalty_threshold: float = 0.40,
    strict: bool = False,
) -> Engine3Result:
    out_root = out_root.resolve()
    cohort = _cohort_dir(out_root, disease)

    out_dir = out_root / "engines" / "causal_pathway_context" / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    res = Engine3Result(
        disease=disease,
        status="ok",
        out_dir=out_dir,
        pkn_edges=out_dir / "pkn_edges.tsv",
        causal_edges=out_dir / "causal_subnetwork_edges.tsv",
        causal_nodes=out_dir / "causal_subnetwork_nodes.tsv",
        drivers_ranked=out_dir / "drivers_ranked.tsv",
        mechanism_cards_json=out_dir / "mechanism_cards.json",
        mechanism_cards_tsv=out_dir / "mechanism_cards.tsv",
        graphml=out_dir / "causal_network.graphml",
        manifest=out_dir / "ENGINE_MANIFEST.json",
        skipped_flag=out_dir / "SKIPPED.txt",
    )

    tf_path = _find_engine1_tf_activity(out_root, disease)
    if tf_path is None:
        msg = f"Engine3 skipped: missing Engine1 tf_activity.tsv for {disease}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "causal_pathway_context", "status": "skipped", "reason": msg}, indent=2))
        return Engine3Result(**{**res.__dict__, "status": "skipped"})

    pkn_path = _find_pkn_edges(out_root, pkn_edges_override)
    if pkn_path is None:
        msg = "Engine3 skipped: PKN edges not found. (Tip: OmniPath layer can build it at OUT_ROOT/engines/pkn_cache/pkn_edges.tsv)"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "causal_pathway_context", "status": "skipped", "reason": msg}, indent=2))
        return Engine3Result(**{**res.__dict__, "status": "skipped"})

    footprints_path = _find_engine1_footprints(out_root, disease)
    footprints_available = footprints_path is not None

    tf_df = pd.read_csv(tf_path, sep="\t", index_col=0)

    # FIX: handle "index-only" tf_activity.tsv (0 TF columns) with a minimal fallback
    if tf_df.shape[1] == 0:
        fb = _build_tf_activity_fallback(
            out_root=out_root,
            disease=disease,
            cohort=cohort,
            max_tfs=int(max(top_tfs * 10, 300)),  # generous cap, still safe
        )
        if fb is not None and fb.shape[1] > 0 and fb.shape[0] > 0:
            log.warning(
                "Engine3: tf_activity.tsv had 0 TF columns; using fallback from expression_used (writing back to %s).",
                tf_path,
            )
            _atomic_write_tsv(fb, tf_path, index=True)
            tf_df = fb

    if tf_df.empty:
        msg = f"Engine3 skipped: TF activity table empty: {tf_path}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "causal_pathway_context", "status": "skipped", "reason": msg}, indent=2))
        return Engine3Result(**{**res.__dict__, "status": "skipped"})

    tf_mean = tf_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0)
    tf_mean.index = [str(t).replace("TF:", "", 1).strip().upper() for t in tf_mean.index.astype(str)]
    tf_mean = tf_mean.sort_values(key=lambda s: s.abs(), ascending=False)
    tf_keep = list(tf_mean.index[: int(top_tfs)])

    conf_path = _find_engine2_confounding(out_root, disease)
    conf_pen: Dict[str, float] = {}
    if conf_path is not None:
        cdf = pd.read_csv(conf_path, sep="\t")
        if "feature" in cdf.columns and "penalty" in cdf.columns:
            for _, r in cdf.iterrows():
                feat = str(r["feature"])
                if feat.startswith("TF:"):
                    tfname = feat.replace("TF:", "", 1).strip().upper()
                    try:
                        conf_pen[tfname] = float(r.get("penalty", 0.0))
                    except Exception:
                        conf_pen[tfname] = 0.0

    raw_edges = pd.read_csv(pkn_path, sep="\t" if pkn_path.suffix.lower() in {".tsv", ".txt"} else ",")
    edges = _normalize_pkn_edges(raw_edges)

    expr_path = _find_expression_used(cohort)
    expressed_genes = _read_expression_gene_set(expr_path) if expr_path else set()
    if expressed_genes:
        keep_nodes = set(expressed_genes) | set(tf_keep)
        edges = edges[edges["source"].isin(keep_nodes) & edges["target"].isin(keep_nodes)].copy()

    edges.to_csv(res.pkn_edges, sep="\t", index=False)

    n_edges = int(len(edges))
    n_signed = int((edges["sign"].astype(int) != 0).sum()) if n_edges else 0
    signed_frac = (float(n_signed) / float(n_edges)) if n_edges else 0.0

    edges_signed = edges[edges["sign"].astype(int) != 0].copy()
    incoming_signed = _build_incoming(edges_signed)
    incoming_unsigned = _build_incoming_unsigned(edges)

    def _confidence(tf_act: float, route_support: bool, footprints_ok: bool, penalty: float) -> Tuple[str, float]:
        score = 0.0
        if abs(tf_act) >= 0.75:
            score += 1.0
        if route_support:
            score += 1.0
        if footprints_ok:
            score += 1.0
        if penalty >= float(confound_penalty_threshold):
            score -= 1.0

        if score >= 2.5:
            return "High", score
        if score >= 1.5:
            return "Medium", score
        return "Low", score

    def _run_signed() -> Tuple[Set[Tuple[str, str, int]], Dict[str, float], Dict[str, List[int]], List[Dict[str, object]]]:
        sub_edges_set: Set[Tuple[str, str, int]] = set()
        driver_score: Dict[str, float] = {}
        node_sign_votes: Dict[str, List[int]] = {}
        cards: List[Dict[str, object]] = []

        for tf in tf_keep:
            tf_act = float(tf_mean.loc[tf])
            tf_dir = 1 if tf_act > 0 else (-1 if tf_act < 0 else 0)

            best = _bfs_upstream(incoming_signed, tf, max_steps=int(max_steps))
            route_support = bool(len(best) > 1)

            nodes_in = {n for (n, _sg) in best.keys()}
            for node in list(nodes_in):
                for src, esign in incoming_signed.get(node, []):
                    if src in nodes_in and esign != 0:
                        sub_edges_set.add((src, node, int(esign)))

            for (node, sign_to_tf), depth in best.items():
                if node == tf or depth <= 0:
                    continue
                implied = int(tf_dir * sign_to_tf) if tf_dir != 0 else 0
                if implied != 0:
                    node_sign_votes.setdefault(node, []).append(implied)
                driver_score[node] = driver_score.get(node, 0.0) + (abs(tf_act) / float(depth))

            penalty = float(conf_pen.get(tf, 0.0))
            conf_label, conf_score = _confidence(tf_act, route_support, footprints_available, penalty)

            evidence_present = ["TF activity (Engine1)"]
            evidence_missing = []
            if route_support:
                evidence_present.append("Signed route support (PKN)")
            else:
                evidence_missing.append("Signed route support (PKN)")
            if footprints_available:
                evidence_present.append("Pathway footprints (Engine1)")
            else:
                evidence_missing.append("Pathway footprints (Engine1)")
            if conf_path is not None:
                evidence_present.append("Confounding penalty (Engine2)")
            else:
                evidence_missing.append("Confounding penalty (Engine2)")

            cards.append(
                {
                    "type": "TF_route",
                    "tf": tf,
                    "tf_mean_activity": tf_act,
                    "tf_direction": "UP" if tf_dir > 0 else ("DOWN" if tf_dir < 0 else "ZERO"),
                    "route_support": bool(route_support),
                    "footprints_available": bool(footprints_available),
                    "confounding_penalty": penalty,
                    "confidence": conf_label,
                    "confidence_score": float(conf_score),
                    "unsigned_fallback": False,
                    "evidence_present": evidence_present,
                    "evidence_missing": evidence_missing,
                    "disclaimer": (
                        "This is mechanistic plausibility support (directed/signed consistency with prior biology), "
                        "not statistical causality proof."
                    ),
                }
            )

        return sub_edges_set, driver_score, node_sign_votes, cards

    def _run_unsigned() -> Tuple[Set[Tuple[str, str, int]], Dict[str, float], Dict[str, List[int]], List[Dict[str, object]]]:
        sub_edges_set: Set[Tuple[str, str, int]] = set()
        driver_score: Dict[str, float] = {}
        node_sign_votes: Dict[str, List[int]] = {}
        cards: List[Dict[str, object]] = []

        for tf in tf_keep:
            tf_act = float(tf_mean.loc[tf])

            best_u = _bfs_upstream_unsigned(incoming_unsigned, tf, max_steps=int(max_steps))
            route_support = bool(len(best_u) > 1)

            nodes_in = set(best_u.keys())
            for node in list(nodes_in):
                for src in incoming_unsigned.get(node, []):
                    if src in nodes_in:
                        sub_edges_set.add((src, node, 0))

            for node, depth in best_u.items():
                if node == tf or depth <= 0:
                    continue
                driver_score[node] = driver_score.get(node, 0.0) + (abs(tf_act) / float(depth))

            penalty = float(conf_pen.get(tf, 0.0))

            cards.append(
                {
                    "type": "TF_route",
                    "tf": tf,
                    "tf_mean_activity": tf_act,
                    "tf_direction": "UNKNOWN",
                    "route_support": bool(route_support),
                    "footprints_available": bool(footprints_available),
                    "confounding_penalty": penalty,
                    "confidence": "Low",
                    "confidence_score": 0.0,
                    "unsigned_fallback": True,
                    "evidence_present": ["TF activity (Engine1)"],
                    "evidence_missing": ["Signed route support (PKN)"],
                    "disclaimer": (
                        "Unsigned fallback used: connectivity-only (sign=0). Treat as hypothesis generation, "
                        "not mechanistic direction proof."
                    ),
                }
            )

        return sub_edges_set, driver_score, node_sign_votes, cards

    sub_edges_set, driver_score, node_sign_votes, cards = _run_signed()

    if not sub_edges_set:
        log.warning("Engine3: signed-only traversal produced no edges; retrying with unsigned-edge fallback.")
        sub_edges_set, driver_score, node_sign_votes, cards = _run_unsigned()

    if not sub_edges_set:
        msg = "Engine3 skipped: No causal subnetwork edges formed (check PKN coverage / gene symbol mapping)."
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(res.skipped_flag, msg + "\n")
        _atomic_write_text(res.manifest, json.dumps({"engine": "causal_pathway_context", "status": "skipped", "reason": msg}, indent=2))
        return Engine3Result(**{**res.__dict__, "status": "skipped"})

    sub_edges = pd.DataFrame(list(sub_edges_set), columns=["source", "target", "sign"])
    sub_edges.to_csv(res.causal_edges, sep="\t", index=False)

    nodes = sorted(set(sub_edges["source"]) | set(sub_edges["target"]))
    degree = (
        pd.concat([sub_edges["source"].value_counts(), sub_edges["target"].value_counts()], axis=1)
        .fillna(0)
        .sum(axis=1)
        .to_dict()
    )

    node_rows: List[Dict[str, object]] = []
    for n in nodes:
        votes = node_sign_votes.get(n, [])
        implied_sign = 0
        if votes:
            implied_sign = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
        node_rows.append(
            {
                "node": n,
                "driver_score": float(driver_score.get(n, 0.0)),
                "degree": float(degree.get(n, 0.0)),
                "implied_sign": int(implied_sign),
            }
        )

    nodes_df = pd.DataFrame(node_rows).sort_values(["driver_score", "degree"], ascending=[False, False])
    nodes_df.to_csv(res.causal_nodes, sep="\t", index=False)

    drivers = nodes_df[nodes_df["driver_score"] > 0].copy().reset_index(drop=True)
    drivers.to_csv(res.drivers_ranked, sep="\t", index=False)

    res.mechanism_cards_json.write_text(json.dumps({"disease": disease, "cards": cards}, indent=2), encoding="utf-8")
    pd.DataFrame(cards).to_csv(res.mechanism_cards_tsv, sep="\t", index=False)

    graphml_status = "skipped_no_networkx"
    if nx is not None:
        try:
            G = nx.DiGraph()
            for _, r in sub_edges.iterrows():
                G.add_edge(str(r["source"]), str(r["target"]), sign=int(r["sign"]))
            for _, r in nodes_df.iterrows():
                n = str(r["node"])
                if n in G.nodes:
                    G.nodes[n]["driver_score"] = float(r["driver_score"])
                    G.nodes[n]["degree"] = float(r["degree"])
                    G.nodes[n]["implied_sign"] = int(r["implied_sign"])
            nx.write_graphml(G, res.graphml)
            graphml_status = "ok"
        except Exception as e:
            graphml_status = f"failed:{type(e).__name__}"
            log.warning("GraphML write failed (ignored): %s", e)

    _atomic_write_text(
        res.manifest,
        json.dumps(
            {
                "engine": "causal_pathway_context",
                "version": "1.1.0",
                "status": "ok",
                "inputs": {
                    "out_root": str(out_root),
                    "disease": disease,
                    "tf_activity": str(tf_path),
                    "pathway_footprints": str(footprints_path) if footprints_path else None,
                    "pkn_edges": str(pkn_path),
                    "confounding_report": str(conf_path) if conf_path else None,
                    "expression_used": str(expr_path) if expr_path else None,
                },
                "params": {
                    "max_steps": int(max_steps),
                    "top_tfs": int(top_tfs),
                    "confound_penalty_threshold": float(confound_penalty_threshold),
                    "strict": bool(strict),
                },
                "pkn_stats": {
                    "edges_total": int(n_edges),
                    "edges_signed": int(n_signed),
                    "signed_fraction": float(signed_frac),
                    "filtered_to_expressed": bool(bool(expressed_genes)),
                },
                "outputs": {
                    "pkn_edges_filtered": str(res.pkn_edges),
                    "causal_edges": str(res.causal_edges),
                    "causal_nodes": str(res.causal_nodes),
                    "drivers_ranked": str(res.drivers_ranked),
                    "mechanism_cards_json": str(res.mechanism_cards_json),
                    "mechanism_cards_tsv": str(res.mechanism_cards_tsv),
                    "causal_network_graphml": str(res.graphml) if res.graphml.exists() else None,
                    "graphml_status": graphml_status,
                },
                "disclaimer": (
                    "Outputs represent mechanistic plausibility support (directed/signed consistency with prior biology), "
                    "not statistical causality proof."
                ),
            },
            indent=2,
        ),
    )

    if res.skipped_flag.exists():
        try:
            res.skipped_flag.unlink()
        except Exception:
            pass

    return res


# =========================
# One CLI to run OmniPath + Engine2 + Engine3
# =========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run OmniPath layer + Engine2 (confounding) + Engine3 (context) on IPAA/MDP OUT_ROOT."
    )
    ap.add_argument("--out-root", required=True, help="OUT_ROOT produced by IPAA/main pipeline.")

    # NOTE: repeatable AND comma-separated is supported.
    ap.add_argument(
        "--disease",
        action="append",
        default=None,
        help="Disease name (repeatable, and comma-separated allowed). If omitted, use --all.",
    )
    ap.add_argument("--disease-file", default=None, help="Optional text file with one disease per line.")
    ap.add_argument("--all", action="store_true", help="Run on all discovered diseases under OUT_ROOT.")

    ap.add_argument("--strict", action="store_true", help="If set, missing inputs raise errors instead of SKIPPED.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ap.add_argument("--no-omnipath-layer", action="store_true", help="Disable OmniPath layer completely.")
    ap.add_argument("--refresh-omnipath-cache", action="store_true", help="Re-download OmniPath PTM/intercell/interactions caches.")

    # IMPORTANT: build global PKN once (recommended).
    ap.add_argument("--build-pkn", action="store_true", help="Build/refresh global PKN edges from OmniPath interactions.")
    ap.add_argument("--refresh-pkn", action="store_true", help="Force rebuild pkn_edges.tsv (requires --build-pkn).")
    ap.add_argument("--signor-edges", default=None, help="Optional local SIGNOR edges file to merge (tsv/csv).")

    ap.add_argument("--ptm-min-substrates", type=int, default=5)
    ap.add_argument("--ptm-n-perm", type=int, default=200)

    ap.add_argument("--corr-method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--corr-flag-threshold", type=float, default=0.40)
    ap.add_argument("--min-markers", type=int, default=5)

    ap.add_argument("--pkn-edges", default=None, help="Optional override PKN edges path. Otherwise uses OUT_ROOT/engines/pkn_cache/pkn_edges.tsv.")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--top-tfs", type=int, default=30)
    ap.add_argument("--confound-penalty-threshold", type=float, default=0.40)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    # ---- choose diseases ----
    diseases: List[str] = []

    # from file
    if args.disease_file:
        fp = Path(args.disease_file).expanduser().resolve()
        if not fp.exists():
            raise ValidationError(f"--disease-file not found: {fp}")
        lines = [ln.strip() for ln in fp.read_text(encoding="utf-8").splitlines()]
        diseases.extend([ln for ln in lines if ln and not ln.startswith("#")])

    # from CLI (repeatable + comma-separated)
    diseases.extend(_parse_diseases_arg(args.disease))

    if args.all:
        diseases = _discover_diseases(out_root)
        if not diseases:
            raise DataError(f"No diseases discovered under {out_root}")
    else:
        if not diseases:
            raise ValidationError("Provide --disease DiseaseName (repeatable) OR use --all (or --disease-file).")

    # de-dup diseases again (in case file + cli overlap)
    seen: Set[str] = set()
    diseases = [d for d in diseases if not (d in seen or seen.add(d))]

    pkn_override = Path(args.pkn_edges).expanduser().resolve() if args.pkn_edges else None
    signor_path = Path(args.signor_edges).expanduser().resolve() if args.signor_edges else None

    # ---- Build global PKN ONCE if requested and not overridden ----
    if args.build_pkn and pkn_override is None:
        omnipath_cache_dir = out_root / "engines" / "omnipath_cache"
        _ensure_dir(omnipath_cache_dir)
        try:
            ok, pkn_path, msg = build_global_pkn_edges(
                out_root=out_root,
                omnipath_cache_dir=omnipath_cache_dir,
                refresh=bool(args.refresh_pkn),
                signor_edges_path=signor_path,
                strict=bool(args.strict),
            )
            log.info("[pkn] build_once status=%s path=%s msg=%s", ok, str(pkn_path) if pkn_path else None, msg)
        except Exception as e:
            log.exception("Global PKN build failed: %s", e)
            if args.strict:
                raise

    summary: List[Dict[str, str]] = []

    for d in diseases:
        log.info("=== Running OmniPath + Engines for disease: %s ===", d)

        omnipath_status = "disabled"
        try:
            if not args.no_omnipath_layer:
                # IMPORTANT: per-disease OmniPath layer does NOT rebuild PKN; we do that once above.
                o = run_omnipath_layer_for_disease(
                    out_root=out_root,
                    disease=d,
                    refresh_cache=bool(args.refresh_omnipath_cache),
                    build_pkn=False,
                    refresh_pkn=False,
                    signor_edges_path=signor_path,
                    ptm_min_substrates=int(args.ptm_min_substrates),
                    ptm_n_perm=int(args.ptm_n_perm),
                    strict=bool(args.strict),
                )
                omnipath_status = o.status
        except Exception as e:
            log.exception("OmniPath layer failed for %s: %s", d, e)
            if args.strict:
                raise
            omnipath_status = "failed"

        try:
            e2 = run_engine2_confounding(
                out_root=out_root,
                disease=d,
                corr_method=args.corr_method,
                corr_flag_threshold=float(args.corr_flag_threshold),
                min_markers=int(args.min_markers),
                strict=bool(args.strict),
            )
            log.info("Engine2 status=%s outputs=%s", e2.status, e2.out_dir)
        except Exception as e:
            log.exception("Engine2 failed for %s: %s", d, e)
            if args.strict:
                raise
            e2 = Engine2Result(
                disease=d,
                status="failed",
                out_dir=out_root / "engines" / "confounding" / d,
                cell_type_scores=out_root / "engines" / "confounding" / d / "cell_type_scores.tsv",
                confounding_report=out_root / "engines" / "confounding" / d / "confounding_report.tsv",
                manifest=out_root / "engines" / "confounding" / d / "ENGINE_MANIFEST.json",
                skipped_flag=out_root / "engines" / "confounding" / d / "SKIPPED.txt",
            )

        try:
            e3 = run_engine3_contextualization(
                out_root=out_root,
                disease=d,
                pkn_edges_override=pkn_override,
                max_steps=int(args.max_steps),
                top_tfs=int(args.top_tfs),
                confound_penalty_threshold=float(args.confound_penalty_threshold),
                strict=bool(args.strict),
            )
            log.info("Engine3 status=%s outputs=%s", e3.status, e3.out_dir)
        except Exception as e:
            log.exception("Engine3 failed for %s: %s", d, e)
            if args.strict:
                raise
            e3 = Engine3Result(
                disease=d,
                status="failed",
                out_dir=out_root / "engines" / "causal_pathway_context" / d,
                pkn_edges=out_root / "engines" / "causal_pathway_context" / d / "pkn_edges.tsv",
                causal_edges=out_root / "engines" / "causal_pathway_context" / d / "causal_subnetwork_edges.tsv",
                causal_nodes=out_root / "engines" / "causal_pathway_context" / d / "causal_subnetwork_nodes.tsv",
                drivers_ranked=out_root / "engines" / "causal_pathway_context" / d / "drivers_ranked.tsv",
                mechanism_cards_json=out_root / "engines" / "causal_pathway_context" / d / "mechanism_cards.json",
                mechanism_cards_tsv=out_root / "engines" / "causal_pathway_context" / d / "mechanism_cards.tsv",
                graphml=out_root / "engines" / "causal_pathway_context" / d / "causal_network.graphml",
                manifest=out_root / "engines" / "causal_pathway_context" / d / "ENGINE_MANIFEST.json",
                skipped_flag=out_root / "engines" / "causal_pathway_context" / d / "SKIPPED.txt",
            )

        summary.append({"disease": d, "omnipath_layer": omnipath_status, "engine2": e2.status, "engine3": e3.status})

    summary_path = out_root / "engines" / "ENGINE2_3_OMNIPATH_RUN_SUMMARY.tsv"
    _atomic_write_tsv(pd.DataFrame(summary), summary_path, index=False)
    log.info("Wrote run summary: %s", summary_path)


if __name__ == "__main__":
    main()
