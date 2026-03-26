#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IPAA/m6_processing.py

Drop-in processing module for the IPAA multi-cohort pipeline.

NO-SURPRISES policy:
- Keeps existing outputs/filenames and existing logic paths.
- Only adds robustness and additive functionality.
- Fixes the “MSigDB c2.cp GMT not found locally” failure by:
    1) using local c2.cp*.gmt if present (same as before), else
    2) attempting online fetch via gseapy.Msigdb (additive fallback),
    3) if that fails, raises the same actionable error.

Key features:
- Multi-cohort per-spec execution helpers
- IPAA pathway activity scoring (rank^2 + 2.5% trim + within-sample z)
- Robust stats (Welch if possible; delta-mean fallback if low n)
- Optional supporting layers (decoupler ULM/VIPER) with disk caching:
  * CollecTRI (TF)
  * PROGENy (signaling)
  * OmniPath Intercell categories
  * OmniPath SignedPTMs (optional kinase proxy)
- OmniPath caching pattern:
  download once -> save to IPAA/data/omnipath_cache -> reuse on subsequent runs
  only re-download with --refresh-omnipath

Compatibility note:
- Primary output layout: <out_root>/<cohort_name>/
- Legacy mirror/redirect: <out_root>/cohorts/<cohort_name>/ (additive only)
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
from scipy.stats import fisher_exact, ttest_ind, spearmanr, chi2_contingency
import omnipath as op  # type: ignore
import decoupler as dc  # type: ignore
import lxml  # noqa: F401
import mygene 

from gseapy import Msigdb  # type: ignore
import gseapy as gp  # type: ignore

LOG = logging.getLogger("IPAA_M6")


# =============================================================================
# Small utilities
# =============================================================================
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require(pkg: str, import_name: Optional[str] = None) -> None:
    try:
        __import__(import_name or pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install:\n  pip install -U {pkg}\n\nOriginal error: {e}"
        ) from e


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _sep_from_suffix(path: Path) -> str:
    suf = path.suffix.lower()
    return "\t" if suf in {".tsv", ".txt"} else ","


def _is_windows_drive_path(s: str) -> bool:
    # D:\temp\file.csv or D:/temp/file.csv
    return bool(re.match(r"^[A-Za-z]:[\\/]", s.strip()))


def _to_wsl_mnt_path(win_path: str) -> Path:
    # D:\temp\a.csv -> /mnt/d/temp/a.csv
    s = win_path.strip().replace("\\", "/")
    drive = s[0].lower()
    rest = s[2:]
    rest = rest.lstrip("/")
    return Path("/mnt") / drive / rest


def normalize_path(p: Union[Path, str]) -> Path:
    """
    Robust path normalization:
    - Expands ~ and env vars.
    - Under WSL/Linux: if user passes Windows drive path (D:\\...),
      map to /mnt/d/... ONLY if that mapped path exists (or its parent exists).
      Otherwise, keep the original (to avoid surprise relocation).
    """
    s = str(p)
    s = os.path.expandvars(os.path.expanduser(s))

    if os.name != "nt" and _is_windows_drive_path(s):
        candidate = _to_wsl_mnt_path(s)
        if candidate.exists() or candidate.parent.exists():
            return candidate
        return Path(s)

    return Path(s)


def _read_table_auto(path: Path) -> pd.DataFrame:
    path = normalize_path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    sep = _sep_from_suffix(path)
    try:
        return pd.read_csv(path, sep=sep)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin1")
    except Exception as e:
        raise RuntimeError(f"Failed to read table: {path} ({e})") from e


def _read_matrix(path: Path) -> pd.DataFrame:
    path = normalize_path(path)
    try:
        df = pd.read_csv(path, sep=_sep_from_suffix(path), index_col=0)
    except Exception as e:
        raise RuntimeError(f"Failed to read expression matrix: {path} ({e})") from e

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"Expression matrix is empty after parsing: {path}")
    return df


def setup_logging(outdir: Path, verbose: bool) -> None:
    outdir = normalize_path(outdir)
    _safe_mkdir(outdir)

    level = logging.DEBUG if verbose else logging.INFO
    LOG.setLevel(level)
    LOG.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    LOG.addHandler(ch)

    try:

        fh = RotatingFileHandler(outdir / "run.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    except Exception:
        fh = logging.FileHandler(outdir / "run.log", encoding="utf-8")

    fh.setLevel(level)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


# =============================================================================
# OmniPath caching (drop-in pattern)
# =============================================================================
def _default_cache_dir() -> Path:
    return Path(__file__).parent / "data" / "omnipath_cache"


def _lock_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".lock")


def _acquire_lock(lock_file: Path, timeout_s: int = 300) -> None:
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_file}")
            time.sleep(0.2 + random.random() * 0.3)


def _release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def _can_parquet() -> bool:
    import pyarrow  # noqa: F401
    return True



def _write_df_cache(df: pd.DataFrame, base: Path) -> Path:
    _safe_mkdir(base.parent)
    if _can_parquet():
        out = base.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    out = base.with_suffix(".tsv")
    df.to_csv(out, sep="\t", index=False)
    return out


def _read_df_cache(base: Path) -> Optional[pd.DataFrame]:
    p_parq = base.with_suffix(".parquet")
    p_tsv = base.with_suffix(".tsv")
    if p_parq.exists():
        try:
            return pd.read_parquet(p_parq)
        except Exception:
            return None
    if p_tsv.exists():
        try:
            return pd.read_csv(p_tsv, sep="\t")
        except Exception:
            return None
    return None


def get_omnipath_intercell(refresh: bool = False, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    _require("omnipath")
    cache_dir = cache_dir or _default_cache_dir()
    base = cache_dir / "omnipath_intercell"

    if not refresh:
        cached = _read_df_cache(base)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

    lock = _lock_path(base.with_suffix(".download"))
    _acquire_lock(lock)
    try:
        if not refresh:
            cached = _read_df_cache(base)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached

        
        df = op.requests.Intercell.get()
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("OmniPath Intercell.get() returned empty.")
        _write_df_cache(df, base)
        return df
    finally:
        _release_lock(lock)


def get_omnipath_signedptms(refresh: bool = False, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    _require("omnipath")
    cache_dir = cache_dir or _default_cache_dir()
    base = cache_dir / "omnipath_signedptms"

    if not refresh:
        cached = _read_df_cache(base)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

    lock = _lock_path(base.with_suffix(".download"))
    _acquire_lock(lock)
    try:
        if not refresh:
            cached = _read_df_cache(base)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached

        df = op.requests.SignedPTMs.get()
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("OmniPath SignedPTMs.get() returned empty.")
        _write_df_cache(df, base)
        return df
    finally:
        _release_lock(lock)


def get_collectri_net(refresh: bool = False, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    _require("decoupler")
    cache_dir = cache_dir or _default_cache_dir()
    base = cache_dir / "decoupler_collectri_human"

    if not refresh:
        cached = _read_df_cache(base)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

    lock = _lock_path(base.with_suffix(".download"))
    _acquire_lock(lock)
    try:
        if not refresh:
            cached = _read_df_cache(base)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached

        
        net = dc.op.collectri(organism="human", license="academic", verbose=False)
        if not isinstance(net, pd.DataFrame) or net.empty:
            raise RuntimeError("decoupler.op.collectri returned empty.")
        _write_df_cache(net, base)
        return net
    finally:
        _release_lock(lock)


def get_progeny_net(refresh: bool = False, cache_dir: Optional[Path] = None, top: int = 100) -> pd.DataFrame:
    _require("decoupler")
    cache_dir = cache_dir or _default_cache_dir()
    base = cache_dir / f"decoupler_progeny_human_top{int(top)}"

    if not refresh:
        cached = _read_df_cache(base)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached

    lock = _lock_path(base.with_suffix(".download"))
    _acquire_lock(lock)
    try:
        if not refresh:
            cached = _read_df_cache(base)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached


        net = dc.op.progeny(organism="human", top=int(top), verbose=False)
        if not isinstance(net, pd.DataFrame) or net.empty:
            raise RuntimeError("decoupler.op.progeny returned empty.")
        _write_df_cache(net, base)
        return net
    finally:
        _release_lock(lock)


# =============================================================================
# Expression loading + preprocessing
# =============================================================================
def load_expression_matrix(expr_path: Path) -> pd.DataFrame:
    """
    Returns X: samples x genes.
    Auto-orient:
      - If rows >> cols (genes x samples typical), transpose.
    """
    expr_path = normalize_path(expr_path)
    df = _read_matrix(expr_path)

    if df.shape[0] > df.shape[1] * 3:
        X = df.T
        LOG.info("Auto-orient: treated input as genes x samples; transposed to samples x genes.")
    else:
        X = df
        LOG.info("Auto-orient: treated input as samples x genes (no transpose).")

    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)

    if X.shape[0] < 2:
        raise ValueError(f"Too few samples (n={X.shape[0]}). Need >=2. File: {expr_path}")
    return X


def standardize_gene_symbols_cols(X: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in X.columns.astype(str):
        c2 = c.strip()
        if c2.startswith("ENSG") and "." in c2:
            c2 = c2.split(".")[0]
        cols.append(c2.upper())
    X2 = X.copy()
    X2.columns = pd.Index(cols)
    if X2.columns.duplicated().any():
        X2 = X2.groupby(X2.columns, axis=1).mean()
    return X2


def _looks_like_ensembl_cols(columns: Iterable[str], min_ratio: float = 0.1, min_hits: int = 100) -> bool:
    cols = [str(c).strip() for c in columns]
    if not cols:
        return False
    ens = sum(1 for c in cols if c.upper().startswith("ENSG"))
    return ens >= int(min_hits) or (ens / max(1, len(cols))) >= float(min_ratio)


def _find_ensembl_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns.astype(str))
    candidates = []
    candidates += [c for c in cols if _pick_col_by_regex([c], [r"ensembl", r"^gene[_\s]*id$"])]
    for c in cols:
        if c not in candidates:
            candidates.append(c)
    for c in candidates:
        s = df[c].astype(str).str.strip()
        ratio = s.str.upper().str.startswith("ENSG").mean()
        if ratio >= 0.5:
            return c
    return None


def _find_symbol_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns.astype(str))
    preferred = []
    for pat in [r"^gene\s*name$", r"gene\s*symbol", r"^symbol$", r"hgnc"]:
        c = _pick_col_by_regex(cols, [pat])
        if c is not None:
            preferred.append(c)
    if not preferred:
        c = _pick_col_by_regex(cols, [r"^gene$"])
        if c is not None:
            preferred.append(c)
    for c in preferred:
        s = df[c].astype(str).str.strip()
        ratio = s.str.upper().str.startswith("ENSG").mean()
        if ratio < 0.2:
            return c
    return None


def _read_ensembl_symbol_cache(path: Path) -> Dict[str, str]:
    try:
        p = normalize_path(path)
        if not p.exists():
            return {}
        df = _read_table_auto(p)
        if df.empty:
            return {}
        if not {"ensembl_id", "gene_symbol"}.issubset(df.columns):
            return {}
        sub = df[["ensembl_id", "gene_symbol"]].dropna().copy()
        sub["ensembl_id"] = sub["ensembl_id"].astype(str).str.strip()
        sub["ensembl_id"] = sub["ensembl_id"].str.replace(r"\..*$", "", regex=True)
        sub["gene_symbol"] = sub["gene_symbol"].astype(str).str.strip().str.upper()
        sub = sub[sub["ensembl_id"].str.upper().str.startswith("ENSG")]
        sub = sub[sub["gene_symbol"] != ""]
        if sub.empty:
            return {}
        return dict(zip(sub["ensembl_id"].values, sub["gene_symbol"].values))
    except Exception:
        return {}


def _write_ensembl_symbol_cache(path: Path, mapping: Dict[str, str]) -> None:
    try:
        if not mapping:
            return
        p = normalize_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"ensembl_id": list(mapping.keys()), "gene_symbol": list(mapping.values())}
        )
        df.to_csv(p, sep="\t", index=False)
    except Exception as e:
        LOG.warning("Failed to write Ensembl->HGNC cache: %s", e)


def _build_ensembl_to_symbol_map_mygene(ensembl_ids: Iterable[str]) -> Dict[str, str]:

    ids = []
    for x in ensembl_ids:
        s = str(x).strip()
        if s.upper().startswith("ENSG"):
            s = s.split(".")[0]
            ids.append(s)
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}

    try:
        mg = mygene.MyGeneInfo()
        res = mg.querymany(
            ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=False,
        )
    except Exception as e:
        LOG.warning("mygene query failed: %s", e)
        return {}

    out: Dict[str, str] = {}
    for r in res:
        if not isinstance(r, dict):
            continue
        if r.get("notfound"):
            continue
        q = r.get("query")
        sym = r.get("symbol")
        if isinstance(q, str) and isinstance(sym, str) and sym.strip():
            out[q.strip()] = sym.strip().upper()
    return out


def maybe_convert_ensembl_to_hgnc_cols(
    X: pd.DataFrame,
    data_dir: Path,
    hpa_file: str,
    gtex_file: str,
    fantom_file: str,
) -> pd.DataFrame:
    cols = list(X.columns.astype(str))
    ens_count = sum(1 for c in cols if c.strip().upper().startswith("ENSG"))
    LOG.info("Gene ID check: %d/%d columns start with ENSG.", ens_count, len(cols))
    if not _looks_like_ensembl_cols(cols):
        return X
    cache_path = data_dir / "ensembl_to_hgnc.tsv"
    ens2sym = _read_ensembl_symbol_cache(cache_path)
    if ens2sym:
        LOG.info("Loaded Ensembl->HGNC cache: %s (%d entries).", cache_path, len(ens2sym))
    if not ens2sym:
        ens2sym = _build_ensembl_to_symbol_map_mygene(X.columns)
        if not ens2sym:
            LOG.warning("Ensembl IDs detected, but no Ensembl->HGNC map could be built from baseline files or mygene.")
            return X
        _write_ensembl_symbol_cache(cache_path, ens2sym)
    else:
        missing = [c for c in X.columns.astype(str) if c.strip().upper().startswith("ENSG") and c.split(".")[0] not in ens2sym]
        if missing:
            extra = _build_ensembl_to_symbol_map_mygene(missing)
            for k, v in extra.items():
                ens2sym.setdefault(k, v)
            if extra:
                _write_ensembl_symbol_cache(cache_path, ens2sym)
    new_cols: List[str] = []
    mapped = 0
    for c in X.columns.astype(str):
        c2 = c.strip()
        if c2.startswith("ENSG") and "." in c2:
            c2 = c2.split(".")[0]
        sym = ens2sym.get(c2)
        if sym:
            new_cols.append(sym.upper())
            mapped += 1
        else:
            new_cols.append(c2)
    LOG.info("Ensembl->HGNC mapping: mapped %d/%d gene columns (%.1f%%).", mapped, len(new_cols), 100.0 * mapped / max(1, len(new_cols)))
    if mapped == 0:
        LOG.warning("Ensembl->HGNC mapping found zero matches; check baseline files or provide HGNC symbols in input.")
        return X
    X2 = X.copy()
    X2.columns = pd.Index(new_cols)
    return X2


def counts_to_logcpm(X_counts: pd.DataFrame) -> pd.DataFrame:
    X = X_counts.copy()
    X = X.fillna(0.0)
    X[X < 0] = 0.0
    lib = X.sum(axis=1).replace(0.0, np.nan)
    cpm = X.div(lib, axis=0) * 1e6
    logcpm = np.log2(cpm + 1.0)
    logcpm = logcpm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return logcpm


# =============================================================================
# Metadata loading (UPDATED: additive robustness, no surprises)
# =============================================================================
def _auto_detect_meta_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [str(c) for c in df.columns]

    def pick(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            rx = re.compile(pat, flags=re.IGNORECASE)
            for c in cols:
                if rx.search(c):
                    return c
        return None

    sample_col = pick([r"^sample$", r"sample[_\s]*id", r"^id$", r"^sid$", r"^run$", r"^name$"])
    group_col = pick([r"^group$", r"condition", r"phenotype", r"status", r"class", r"label", r"case[_\s-]*control"])

    if sample_col is None or group_col is None:
        raise ValueError(
            "Could not auto-detect metadata columns.\n"
            f"Available columns: {cols}\n"
            "Fix: set CohortSpec.meta_sample_col and CohortSpec.meta_group_col to the correct column names."
        )

    return sample_col, group_col


def _canonical_sample_id(x: object) -> str:
    s = str(x).strip().strip('"').strip("'")
    s2 = s.replace("\\", "/")
    if "/" in s2:
        s2 = s2.split("/")[-1]

    suffixes = [
        ".fastq.gz", ".fq.gz", ".fastq", ".fq",
        ".bam", ".sam", ".cram",
    ]
    lowered = s2.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lowered.endswith(suf):
                s2 = s2[: -len(suf)]
                lowered = s2.lower()
                changed = True
                break

    return s2.strip().lower()


def _normalize_group_label(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().strip('"').strip("'").lower()

    if s in {"0", "0.0", "false", "f", "no"}:
        return "control"
    if s in {"1", "1.0", "true", "t", "yes"}:
        return "case"

    if s in {"case", "control"}:
        return s

    control_hits = ["control", "ctrl", "healthy", "normal", "baseline", "untreated", "vehicle", "wt", "wildtype"]
    case_hits = ["case", "patient", "disease", "tumor", "treated", "mutant", "ko", "knockout", "affected"]

    if any(tok in s for tok in control_hits):
        return "control"
    if any(tok in s for tok in case_hits):
        return "case"

    return s


def _normalize_groups_series(raw: pd.Series, group_map: Optional[Dict[str, str]] = None) -> pd.Series:
    s = raw.astype(str)

    if group_map:
        gm = {str(k).strip().lower(): str(v).strip().lower() for k, v in group_map.items()}
        mapped: List[str] = []
        for x in s.tolist():
            k = str(x).strip().strip('"').strip("'").lower()
            mapped.append(gm.get(k, x))
        s = pd.Series(mapped, index=raw.index, name=raw.name)

    norm = s.map(_normalize_group_label).astype(str)
    norm = norm.replace({"": np.nan}).dropna()

    uniq = sorted(set(norm.unique()))
    if set(uniq).issubset({"case", "control"}):
        return norm.rename("group")

    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]

        def looks_control(x: str) -> bool:
            return (
                x in {"control", "ctrl", "healthy", "normal", "baseline", "untreated", "vehicle", "wt", "wildtype"}
                or "control" in x
                or "healthy" in x
                or "normal" in x
            )

        def looks_case(x: str) -> bool:
            return (
                x in {"case", "patient", "disease", "tumor", "treated", "mutant", "ko", "knockout", "affected"}
                or "patient" in x
                or "disease" in x
                or "tumor" in x
                or "treated" in x
            )

        if looks_control(a) and not looks_control(b):
            norm = norm.replace({a: "control", b: "case"})
        elif looks_control(b) and not looks_control(a):
            norm = norm.replace({b: "control", a: "case"})
        elif looks_case(a) and not looks_case(b):
            norm = norm.replace({a: "case", b: "control"})
        elif looks_case(b) and not looks_case(a):
            norm = norm.replace({b: "case", a: "control"})
        else:
            raise ValueError(
                "Metadata group labels are two-class but ambiguous.\n"
                f"Found: {uniq}\n"
                "Fix: provide CohortSpec.group_map, e.g. {'healthy':'control','sle':'case'} "
                "or set your group column to contain case/control."
            )

        uniq2 = sorted(set(norm.unique()))
        if not set(uniq2).issubset({"case", "control"}):
            raise ValueError(f"Failed to normalize metadata groups. Final labels: {uniq2}")
        return norm.rename("group")

    raise ValueError(
        "Metadata group column must resolve to exactly 2 groups: case/control.\n"
        f"Found labels after normalization: {uniq}\n"
        "Fix: clean your metadata, or provide CohortSpec.group_map."
    )


def _align_labels_to_samples(y_meta: pd.Series, sample_index: pd.Index) -> pd.Series:
    canon_to_label: Dict[str, str] = {}
    dup = 0
    for sid, lab in y_meta.items():
        key = _canonical_sample_id(sid)
        if key in canon_to_label:
            dup += 1
            continue
        canon_to_label[key] = str(lab)

    if dup:
        LOG.warning("Metadata had %d duplicate sample IDs after canonicalization; kept first occurrence.", dup)

    labels: List[Optional[str]] = []
    missing: List[str] = []
    for s in sample_index.astype(str):
        key = _canonical_sample_id(s)
        if key in canon_to_label:
            labels.append(canon_to_label[key])
        else:
            labels.append(None)
            missing.append(str(s))

    y = pd.Series(labels, index=sample_index.astype(str), name="group")

    if y.isna().any():
        ex = missing[:10]
        raise ValueError(
            "Metadata is missing labels for some expression samples (after robust matching).\n"
            f"Missing count: {len(missing)} / {len(sample_index)}\n"
            f"Examples (expression sample IDs): {ex}\n"
            "Fix: ensure meta sample IDs match expression sample names (or adjust meta_sample_col)."
        )

    return y


def load_metadata(
    meta_path: Path,
    sample_col: str,
    group_col: str,
    group_map: Optional[Dict[str, str]] = None,
) -> pd.Series:
    meta_path = normalize_path(meta_path)
    df = _read_table_auto(meta_path)

    if sample_col not in df.columns or group_col not in df.columns:
        s_col, g_col = _auto_detect_meta_cols(df)
        LOG.warning(
            "Metadata columns not found as requested (sample_col=%s, group_col=%s). "
            "Auto-detected (sample_col=%s, group_col=%s).",
            sample_col, group_col, s_col, g_col,
        )
        sample_col, group_col = s_col, g_col

    y_raw = df.set_index(sample_col)[group_col]
    y_norm = _normalize_groups_series(y_raw, group_map=group_map)
    y_norm.index = y_raw.index.astype(str)
    return y_norm


def infer_groups_from_sample_names(
    samples: Iterable[str],
    control_regex: str,
    case_regex: Optional[str] = None,
) -> pd.Series:
    """
    Rule:
      - If matches control_regex => control
      - Else => case
    Require >=1 case and >=1 control.
    """
    ctrl_pat = re.compile(control_regex, flags=re.IGNORECASE)
    case_pat = re.compile(case_regex, flags=re.IGNORECASE) if case_regex else None

    labels: Dict[str, str] = {}
    for s in samples:
        s2 = str(s)
        if ctrl_pat.search(s2):
            labels[s2] = "control"
        else:
            labels[s2] = "case"
            _ = case_pat.search(s2) if case_pat else None

    y = pd.Series(labels, name="group")
    n_case = int((y == "case").sum())
    n_ctrl = int((y == "control").sum())
    if n_case < 1 or n_ctrl < 1:
        raise ValueError(
            f"Need >=1 case and >=1 control. Got case={n_case}, control={n_ctrl}.\n"
            f"Fix: adjust control_regex or provide metadata."
        )
    return y


# =============================================================================
# Stats (ROBUST)
# =============================================================================
def welch_t_stats(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust Welch stats:
    - If BOTH groups have >=2 samples: Welch t + p
    - Else (>=1 each): fallback to delta-mean "t_like", p=1
    """
    _require("scipy")
    

    common = X.index.intersection(y.index)
    if common.empty:
        raise ValueError("No overlapping samples between expression matrix and labels.")

    X2 = X.loc[common]
    y2 = y.loc[common]

    case = X2.loc[y2 == "case"]
    ctrl = X2.loc[y2 == "control"]

    n_case = int(case.shape[0])
    n_ctrl = int(ctrl.shape[0])

    delta = np.nanmean(case.values, axis=0) - np.nanmean(ctrl.values, axis=0)

    if n_case < 2 or n_ctrl < 2:
        t_like = np.asarray(delta, dtype=float)
        p = np.ones_like(t_like, dtype=float)
        return t_like, p

    t, p = ttest_ind(case.values, ctrl.values, axis=0, equal_var=False, nan_policy="omit")
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), np.finfo(float).tiny, 1.0)

    if not np.isfinite(t).any():
        t = np.asarray(delta, dtype=float)
        p = np.ones_like(t, dtype=float)

    return t, p


def differential_gene_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    t, p = welch_t_stats(X, y)
    case = X.loc[y == "case"]
    ctrl = X.loc[y == "control"]
    logfc = np.nanmean(case.values, axis=0) - np.nanmean(ctrl.values, axis=0)
    fdr = _bh_fdr(p)

    out = pd.DataFrame({"gene": X.columns, "logFC": logfc, "t": t, "p": p, "FDR": fdr})
    out = out.sort_values("p", ascending=True)
    return out


def build_prerank(stats: pd.DataFrame, score_col: str = "t") -> pd.DataFrame:
    """
    Robust prerank:
    - Prefer score_col
    - If too few finite values / too few unique scores, fallback to logFC if possible
    NOTE: We do NOT add jitter/tie-break noise (avoid surprise changes).
    """
    def _make(col: str) -> pd.DataFrame:
        df = stats[["gene", col]].copy()
        df = df.rename(columns={col: "score"})
        df["gene"] = df["gene"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["gene", "score"])
        if df.empty:
            return df
        df["_abs"] = df["score"].abs()
        df = df.sort_values("_abs", ascending=False).drop_duplicates("gene", keep="first")
        df = df.drop(columns=["_abs"]).sort_values("score", ascending=False)
        return df[["gene", "score"]]

    rnk = _make(score_col)
    if rnk.shape[0] < 50 or rnk["score"].nunique(dropna=True) < 2:
        if score_col != "logFC" and "logFC" in stats.columns:
            rnk2 = _make("logFC")
            if rnk2.shape[0] >= 50 and rnk2["score"].nunique(dropna=True) >= 2:
                return rnk2
    return rnk


# =============================================================================
# MSigDB C2 CP (GSEApy) - FIXED: local-first, online fallback (additive)
# =============================================================================
def _parse_gmt_file(gmt_path: Path) -> Dict[str, List[str]]:
    gmt_path = normalize_path(gmt_path)
    out: Dict[str, List[str]] = {}
    with gmt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = [str(g).strip().upper() for g in parts[2:] if str(g).strip()]
            if name and genes:
                out[name] = genes
    return out


def _find_local_c2cp_gmt(dbver: Optional[str]) -> Optional[Path]:
    """
    Search for c2.cp*.gmt near this module and under IPAA/data.
    If dbver is provided, prefer a filename containing dbver.
    """
    base_dirs = [
        Path(__file__).parent,
        Path(__file__).parent / "data",
        Path(__file__).parent / "data" / "msigdb",
        Path(__file__).parent / "data" / "gmt",
    ]
    cand: List[Path] = []
    for d in base_dirs:
        if d.exists():
            cand.extend(sorted(d.glob("c2.cp*.gmt")))

    if not cand:
        return None

    if dbver:
        dbver_s = str(dbver)
        exact = [p for p in cand if dbver_s in p.name]
        if exact:
            return sorted(exact)[-1]

    return sorted(cand)[-1]


def fetch_msigdb_c2_cp_gmt(dbver: Optional[str]) -> Dict[str, List[str]]:
    """
    Local-first (same as your older behavior), online fallback (additive).
    """
    local = _find_local_c2cp_gmt(dbver)
    if local and local.exists():
        gmt = _parse_gmt_file(local)
        if gmt:
            LOG.info("Using local MSigDB c2.cp GMT: %s", local)
            return gmt

    # additive fallback: online via gseapy
    try:
        _require("gseapy")
        
        

        msig = Msigdb()

        if not dbver:
            versions = msig.list_dbver()
            if isinstance(versions, pd.DataFrame) and "Name" in versions.columns:
                cands = versions["Name"].astype(str).tolist()
            elif isinstance(versions, (list, tuple)):
                cands = [str(v) for v in versions]
            else:
                cands = [str(v) for v in list(versions)]
            hs = [v for v in cands if v.endswith(".Hs")]
            if not hs:
                raise RuntimeError("Could not find human MSigDB versions.")
            dbver = sorted(hs)[-1]
            LOG.warning("No msigdb_dbver provided: auto-selected latest: %s (NOT paper-frozen).", dbver)

        gmt = msig.get_gmt(category="c2.cp", dbver=dbver)
        if not isinstance(gmt, dict) or not gmt:
            raise RuntimeError("MSigDB get_gmt(c2.cp) returned empty.")

        gmt2: Dict[str, List[str]] = {}
        for k, genes in gmt.items():
            genes2 = [str(g).strip().upper() for g in genes if isinstance(g, str) and str(g).strip()]
            if genes2:
                gmt2[str(k)] = genes2

        if not gmt2:
            raise RuntimeError("MSigDB returned no valid gene sets after cleaning.")

        LOG.info("Fetched MSigDB c2.cp via gseapy (dbver=%s).", dbver)
        return gmt2

    except Exception as e:
        # preserve the original actionable message (no surprises)
        raise FileNotFoundError(
            "Could not find MSigDB c2.cp GMT locally and online fetch failed.\n"
            "Place a c2.cp*.gmt under IPAA/data (or next to m6_processing.py).\n"
            f"Online fetch error: {e}"
        ) from e


def write_gmt(gmt: Dict[str, List[str]], out_path: Path) -> None:
    _safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for gs, genes in gmt.items():
            genes = [g for g in genes if isinstance(g, str) and g.strip()]
            f.write(gs + "\tNA\t" + "\t".join(genes) + "\n")


def reduce_overlap_gmt(
    gmt: Dict[str, List[str]],
    jaccard_thresh: float = 0.5,
    min_genes: int = 10,
) -> Dict[str, List[str]]:
    items = [(k, set(v)) for k, v in gmt.items() if len(set(v)) >= min_genes]
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    kept: List[Tuple[str, set]] = []
    for name, genes in items:
        ok = True
        for _, g2 in kept:
            inter = len(genes & g2)
            if inter == 0:
                continue
            union = len(genes | g2)
            jac = inter / union if union else 0.0
            if jac > jaccard_thresh:
                ok = False
                break
        if ok:
            kept.append((name, genes))

    out = {k: sorted(list(v)) for k, v in kept}
    LOG.info("Reduced-overlap GMT: %d -> %d (Jaccard>%s pruned)", len(gmt), len(out), jaccard_thresh)
    return out


def run_gsea_prerank(
    rnk: pd.DataFrame,
    gmt: Dict[str, List[str]],
    outdir: Path,
    permutation_num: int,
    seed: int,
    threads: int,
    min_size: int = 10,
    max_size: int = 2000,
) -> pd.DataFrame:
    _require("gseapy")
    

    _safe_mkdir(outdir)

    if rnk is None or rnk.empty:
        LOG.warning("GSEA skipped: ranking is empty.")
        (outdir / "GSEA_SKIPPED.txt").write_text("Ranking empty.\n", encoding="utf-8")
        return pd.DataFrame()

    rnk2 = rnk.copy()
    rnk2["score"] = pd.to_numeric(rnk2["score"], errors="coerce")
    rnk2 = rnk2.dropna(subset=["gene", "score"])

    if rnk2.shape[0] < 50:
        LOG.warning("GSEA skipped: too few ranked genes (%d).", rnk2.shape[0])
        (outdir / "GSEA_SKIPPED.txt").write_text(f"Too few ranked genes: {rnk2.shape[0]}\n", encoding="utf-8")
        return pd.DataFrame()

    if rnk2["score"].nunique(dropna=True) < 2:
        LOG.warning("GSEA skipped: ranking scores have <2 unique values.")
        (outdir / "GSEA_SKIPPED.txt").write_text("Ranking scores <2 unique values.\n", encoding="utf-8")
        return pd.DataFrame()

    gmt_path = outdir / "msigdb_c2cp.gmt"
    write_gmt(gmt, gmt_path)

    try:
        pre_res = gp.prerank(
            rnk=rnk2,
            gene_sets=str(gmt_path),
            outdir=str(outdir),
            min_size=min_size,
            max_size=max_size,
            permutation_num=int(permutation_num),
            seed=int(seed),
            threads=int(threads),
            no_plot=True,
            verbose=False,
        )
        res = pre_res.res2d.copy()
        res.to_csv(outdir / "gsea_prerank_results.tsv", sep="\t", index=True)
        return res
    except AssertionError as e:
        LOG.warning("GSEA skipped: gseapy assertion failed (%s).", e)
        (outdir / "GSEA_SKIPPED.txt").write_text(f"AssertionError: {e}\n", encoding="utf-8")
        return pd.DataFrame()
    except Exception as e:
        LOG.warning("GSEA skipped: prerank failed (%s).", e)
        (outdir / "GSEA_SKIPPED.txt").write_text(f"Exception: {e}\n", encoding="utf-8")
        return pd.DataFrame()


# =============================================================================
# IPAA pathway activity (core)
# =============================================================================
def _trim_mean(vals: np.ndarray, trim_frac: float) -> float:
    if vals.size == 0:
        return np.nan
    v = np.sort(vals)
    k = int(np.floor(trim_frac * v.size))
    if 2 * k >= v.size:
        return float(np.mean(v))
    return float(np.mean(v[k: v.size - k]))


def compute_ipaa_activity(
    X: pd.DataFrame,
    gmt: Dict[str, List[str]],
    trim_frac: float = 0.025,
    min_genes: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(X.columns)
    gene_index = {g: i for i, g in enumerate(genes)}
    n_genes_total = len(genes)

    p2idx: Dict[str, np.ndarray] = {}
    n_used: Dict[str, int] = {}
    for p, glist in gmt.items():
        idx = [gene_index[g] for g in glist if g in gene_index]
        if len(idx) >= min_genes:
            p2idx[p] = np.array(idx, dtype=int)
            n_used[p] = len(idx)

    pathways = list(p2idx.keys())
    if not pathways:
        raise RuntimeError("No pathways left after intersecting with expression genes (C2 CP).")

    A = np.zeros((X.shape[0], len(pathways)), dtype=float)
    Xv = X.values

    for si in range(X.shape[0]):
        expr = Xv[si, :]
        order = np.argsort(expr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, n_genes_total + 1, dtype=float)
        rank2 = ranks ** 2

        for pj, p in enumerate(pathways):
            vals = rank2[p2idx[p]]
            A[si, pj] = _trim_mean(vals, trim_frac=trim_frac)

        mu = np.nanmean(A[si, :])
        sd = np.nanstd(A[si, :], ddof=0)
        if sd == 0 or not np.isfinite(sd):
            sd = 1.0
        A[si, :] = (A[si, :] - mu) / sd

    activity = pd.DataFrame(A, index=X.index, columns=pathways)
    meta = pd.DataFrame({"pathway": pathways, "n_genes_used": [n_used[p] for p in pathways]})
    return activity, meta


def differential_pathway_stats(activity: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    t, p = welch_t_stats(activity, y)
    case = activity.loc[y == "case"]
    ctrl = activity.loc[y == "control"]
    delta = np.nanmean(case.values, axis=0) - np.nanmean(ctrl.values, axis=0)
    fdr = _bh_fdr(p)

    dir_float = np.sign(t)
    dir_int = np.where(np.isfinite(dir_float), dir_float, 0).astype(int)

    out = pd.DataFrame(
        {
            "pathway": activity.columns.astype(str),
            "delta_activity": delta,
            "t": t,
            "p": p,
            "FDR": fdr,
            "direction": dir_int,
        }
    ).sort_values("p", ascending=True)
    return out


# =============================================================================
# Baseline consensus (optional)
# =============================================================================
def _pick_col_by_regex(columns: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in columns:
            if rx.search(c):
                return c
    return None


# --- NEW (additive, no-surprises): pick the correct numeric value column for baseline files
def _pick_best_value_col(columns: List[str]) -> Optional[str]:
    """
    Baseline inputs often provide multiple numeric columns.
    Prefer stable, normalized measures:
      - GTEx: nTPM > TPM > pTPM
      - FANTOM/HPA: Normalized tags per million > Tags per million > Scaled tags per million
    """
    cols = list(columns)
    priority_patterns = [
        r"^ntpm$",
        r"\bntpm\b",
        r"normalized\s*tags\s*per\s*million",
        r"normalized\s*tpm",
        r"^tpm$",
        r"\btpm\b",
        r"tags\s*per\s*million",
        r"^ptpm$",
        r"\bptpm\b",
        r"scaled\s*tags\s*per\s*million",
        r"scaled\s*tpm",
    ]
    for pat in priority_patterns:
        c = _pick_col_by_regex(cols, [pat])
        if c is not None:
            return c
    return None


# --- NEW (additive, no-surprises): prefer Gene name (symbol) over Gene (ENSG)
def _pick_gene_symbol_col(columns: List[str]) -> Optional[str]:
    cols = list(columns)
    c = _pick_col_by_regex(cols, [r"^gene\s*name$", r"gene\s*symbol", r"^symbol$"])
    if c is not None:
        return c
    c = _pick_col_by_regex(cols, [r"^gene$"])
    return c


def _standardize_long_any(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust standardizer for baseline files.

    Goal: produce long table with columns:
      - gene_symbol (HGNC-like, uppercase)
      - tissue
      - value (float)

    Critical fix: prevent measurement columns (e.g., "Normalized tags per million")
    from being melted into the "tissue" axis.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["gene_symbol", "tissue", "value"])

    cols = list(df.columns.astype(str))

    gene_col = _pick_gene_symbol_col(cols) or cols[0]
    tissue_col = _pick_col_by_regex(cols, [r"^tissue$", r"\btissue\b", r"anatomy", r"region", r"organ"])
    value_col = _pick_best_value_col(cols)
    if value_col is None:
        value_col = _pick_col_by_regex(cols, [r"^value$", r"expression", r"zscore", r"^z$"])

    # Preferred path: explicit (gene, tissue, value) present (your HPA/GTEx/FANTOM format).
    if tissue_col is not None and value_col is not None and gene_col in cols:
        out = df[[gene_col, tissue_col, value_col]].copy()
        out.columns = ["gene_symbol", "tissue", "value"]

        out["gene_symbol"] = out["gene_symbol"].astype(str).str.strip().str.upper()
        out["tissue"] = out["tissue"].astype(str).str.strip()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

        out = out.dropna(subset=["gene_symbol", "tissue", "value"])
        out = out[(out["gene_symbol"] != "") & (out["tissue"] != "")]
        return out

    # Last resort: wide -> melt (kept for compatibility; avoids surprises)
    id_vars = [gene_col]
    value_vars = [c for c in cols if c != gene_col]
    out = df[id_vars + value_vars].copy()
    out = out.melt(id_vars=[gene_col], var_name="tissue", value_name="value")
    out = out.rename(columns={gene_col: "gene_symbol"})

    out["gene_symbol"] = out["gene_symbol"].astype(str).str.strip().str.upper()
    out["tissue"] = out["tissue"].astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["gene_symbol", "tissue", "value"])
    out = out[(out["gene_symbol"] != "") & (out["tissue"] != "")]
    return out


def _z_by_gene(tissue_gene: pd.DataFrame) -> pd.DataFrame:
    M = tissue_gene.copy()
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=0).replace(0.0, np.nan)
    Z = (M - mu) / sd
    return Z.replace([np.inf, -np.inf], np.nan)


def build_consensus_baseline(data_dir: Path, hpa_file: str, gtex_file: str, fantom_file: str) -> pd.DataFrame:
    data_dir = normalize_path(data_dir)
    paths = [data_dir / hpa_file, data_dir / gtex_file, data_dir / fantom_file]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Baseline file missing: {p}")

    hpa = _standardize_long_any(_read_table_auto(paths[0]))
    gtx = _standardize_long_any(_read_table_auto(paths[1]))
    fan = _standardize_long_any(_read_table_auto(paths[2]))

    def to_matrix(longdf: pd.DataFrame) -> pd.DataFrame:
        return longdf.pivot_table(index="tissue", columns="gene_symbol", values="value", aggfunc="mean")

    H = to_matrix(hpa)
    G = to_matrix(gtx)
    F = to_matrix(fan)

    all_tissues = sorted(set(H.index) | set(G.index) | set(F.index))
    all_genes = sorted(set(H.columns) | set(G.columns) | set(F.columns))
    H = H.reindex(index=all_tissues, columns=all_genes)
    G = G.reindex(index=all_tissues, columns=all_genes)
    F = F.reindex(index=all_tissues, columns=all_genes)

    ZH = _z_by_gene(H)
    ZG = _z_by_gene(G)
    ZF = _z_by_gene(F)

    stack = np.stack([ZH.values, ZG.values, ZF.values], axis=0)
    count = np.isfinite(stack).sum(axis=0)
    summ = np.nansum(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = summ / np.where(count == 0, np.nan, count)

    return pd.DataFrame(Z, index=all_tissues, columns=all_genes)


def consensus_pathway_expectations(Z_cons: pd.DataFrame, gmt: Dict[str, List[str]], min_genes: int = 5) -> pd.DataFrame:
    gene_set = set(Z_cons.columns.astype(str))
    rows = []
    overlap_total = 0

    for p, genes in gmt.items():
        gg = [g for g in genes if g in gene_set]
        overlap_total += len(gg)
        if len(gg) < min_genes:
            continue
        exp = Z_cons[gg].mean(axis=1)
        for tissue, v in exp.items():
            rows.append((tissue, p, float(v), len(gg)))

    out = pd.DataFrame(rows, columns=["tissue", "pathway", "expectation", "n_genes"])

    # Additive diagnostics only (no behavioral change besides logging)
    if out.empty:
        LOG.warning(
            "Baseline expectations table is EMPTY. Likely gene-id mismatch between baseline Z_cons columns and GMT genes.\n"
            "Debug: baseline_genes=%d, gmt_sets=%d, total_overlap_hits=%d, min_genes=%d.\n"
            "Fix: baseline gene columns should be HGNC symbols (e.g., from 'Gene name').",
            len(gene_set), len(gmt), overlap_total, int(min_genes),
        )

    return out


def infer_tissue_from_ipaa_mean(
    cohort_name: str,
    ipaa_activity: pd.DataFrame,
    tissue_expect_long: pd.DataFrame,
    top_k: int = 3,
    min_shared_pathways: int = 50,
) -> List[Tuple[str, float]]:
    _require("scipy")
    

    mean_vec = ipaa_activity.mean(axis=0)
    pivot = tissue_expect_long.pivot_table(index="tissue", columns="pathway", values="expectation", aggfunc="mean")
    shared = pivot.columns.intersection(mean_vec.index)
    if len(shared) < min_shared_pathways:
        LOG.warning("[%s] Tissue inference skipped: only %d shared pathways (<%d).",
                    cohort_name, len(shared), min_shared_pathways)
        return []

    scores = []
    for tissue in pivot.index:
        a = pivot.loc[tissue, shared].values
        b = mean_vec.loc[shared].values
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < max(30, min_shared_pathways // 2):
            continue
        r, _ = spearmanr(a[m], b[m])
        scores.append((str(tissue), float(r)))

    scores.sort(key=lambda x: (np.nan_to_num(x[1], nan=-999.0)), reverse=True)
    return scores[:top_k]


def _attach_baseline_to_pathway_stats(
    pathway_stats: pd.DataFrame,
    selected_tissue: str,
    tissue_expect_long: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = pathway_stats.copy()
    out["selected_tissue"] = selected_tissue
    if tissue_expect_long is None or tissue_expect_long.empty:
        out["baseline_expectation"] = np.nan
        return out

    sub = tissue_expect_long[tissue_expect_long["tissue"].astype(str) == str(selected_tissue)]
    if sub.empty:
        out["baseline_expectation"] = np.nan
        return out

    m = sub.set_index("pathway")["expectation"]
    out["baseline_expectation"] = out["pathway"].map(m).astype(float)
    return out


# =============================================================================
# Supporting layers (optional) - decoupler wrappers
# =============================================================================
def _net_standardize(net: pd.DataFrame) -> pd.DataFrame:
    net = net.copy()
    cols = {c.lower(): c for c in net.columns}

    if "source" in cols and "target" in cols:
        if "weight" not in cols:
            if "mor" in cols:
                net["weight"] = pd.to_numeric(net[cols["mor"]], errors="coerce").fillna(1.0)
            else:
                net["weight"] = 1.0
            cols = {c.lower(): c for c in net.columns}
        net = net.rename(columns={cols["source"]: "source", cols["target"]: "target", cols["weight"]: "weight"})
        return net[["source", "target", "weight"]]

    for s_col in ["tf", "regulator", "kinase", "pathway", "enzyme"]:
        if s_col in cols and "target" in cols:
            if "weight" not in cols:
                if "mor" in cols:
                    net["weight"] = pd.to_numeric(net[cols["mor"]], errors="coerce").fillna(1.0)
                else:
                    net["weight"] = 1.0
            net = net.rename(columns={cols[s_col]: "source", cols["target"]: "target"})
            cols = {c.lower(): c for c in net.columns}
            if "weight" in cols:
                net = net.rename(columns={cols["weight"]: "weight"})
            return net[["source", "target", "weight"]]

    raise ValueError(f"Network columns not recognized: {net.columns.tolist()}")


def _dc_ulm(mat: pd.DataFrame, net: pd.DataFrame, min_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    net = _net_standardize(net)

    if hasattr(dc, "run_ulm"):
        est, pvals = dc.run_ulm(
            mat=mat, net=net, source="source", target="target", weight="weight",
            min_n=min_n, verbose=False, use_raw=True
        )
        return est, pvals

    sources = sorted(net["source"].astype(str).unique())
    src_to_targets = {}
    for s in sources:
        sub = net[net["source"] == s]
        sub = sub[sub["target"].isin(mat.columns)]
        if sub["target"].nunique() >= min_n:
            src_to_targets[s] = (sub["target"].values, sub["weight"].values.astype(float))
    if not src_to_targets:
        raise RuntimeError("ULM fallback: no regulators with enough targets after intersecting expression genes.")

    Xv = mat.values
    mu = Xv.mean(axis=1, keepdims=True)
    sd = Xv.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (Xv - mu) / sd
    Xz_df = pd.DataFrame(Xz, index=mat.index, columns=mat.columns)

    est_out = pd.DataFrame(index=mat.index, columns=list(src_to_targets.keys()), dtype=float)
    for s, (tg, w) in src_to_targets.items():
        w = np.asarray(w, dtype=float)
        w = w / (np.sum(np.abs(w)) + 1e-12)
        est_out[s] = (Xz_df[tg].values * w.reshape(1, -1)).sum(axis=1)

    pvs_out = pd.DataFrame(np.nan, index=est_out.index, columns=est_out.columns)
    return est_out, pvs_out


def _dc_viper(mat: pd.DataFrame, net: pd.DataFrame, min_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    net = _net_standardize(net)

    if hasattr(dc, "run_viper"):
        est, pvals = dc.run_viper(
            mat=mat, net=net, source="source", target="target", weight="weight",
            min_n=min_n, verbose=False, use_raw=True
        )
        return est, pvals

    return _dc_ulm(mat, net, min_n=min_n)


def differential_activity_stats(A: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    t, p = welch_t_stats(A, y)
    case = A.loc[y == "case"]
    ctrl = A.loc[y == "control"]
    delta = np.nanmean(case.values, axis=0) - np.nanmean(ctrl.values, axis=0)
    fdr = _bh_fdr(p)
    out = pd.DataFrame(
        {"regulator": A.columns.astype(str), "delta_activity": delta, "t": t, "p": p, "FDR": fdr}
    ).sort_values("p", ascending=True)
    return out


# =============================================================================
# Cross-cohort compare stage
# =============================================================================
def _spearman_corr_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require("scipy")
    

    cols = list(df.columns)
    R = np.eye(len(cols), dtype=float)
    P = np.zeros((len(cols), len(cols)), dtype=float)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = df.iloc[:, i].values
            b = df.iloc[:, j].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 5:
                r, p = np.nan, 1.0
            else:
                r, p = spearmanr(a[m], b[m])
            R[i, j] = R[j, i] = float(r)
            P[i, j] = P[j, i] = float(p)

    return pd.DataFrame(R, index=cols, columns=cols), pd.DataFrame(P, index=cols, columns=cols)


def _chi2_direction_agreement(a_dir: pd.Series, b_dir: pd.Series) -> Tuple[float, float, float, int]:
    _require("scipy")


    shared = a_dir.index.intersection(b_dir.index)
    if len(shared) < 5:
        return np.nan, 1.0, np.nan, int(len(shared))

    A = a_dir.loc[shared].astype(int)
    B = b_dir.loc[shared].astype(int)

    npp = int(((A == 1) & (B == 1)).sum())
    npn = int(((A == 1) & (B == -1)).sum())
    nnp = int(((A == -1) & (B == 1)).sum())
    nnn = int(((A == -1) & (B == -1)).sum())

    table = np.array([[npp, npn], [nnp, nnn]], dtype=float)
    if table.sum() == 0 or (table == 0).all():
        return np.nan, 1.0, np.nan, int(len(shared))

    chi2, p, _, _ = chi2_contingency(table, correction=False)
    concord = (npp + nnn) / max(1.0, table.sum())
    return float(chi2), float(p), float(concord), int(len(shared))


def _fisher_overlap(sig_a: set, sig_b: set, universe_n: int) -> Tuple[float, float, int, int, int]:
    _require("scipy")
    

    A = set(sig_a)
    B = set(sig_b)
    inter = len(A & B)
    nA = len(A)
    nB = len(B)

    a = inter
    b = nA - inter
    c = nB - inter
    d = universe_n - (nA + nB - inter)
    d = max(0, d)
    table = [[a, b], [c, d]]
    odds, p = fisher_exact(table, alternative="two-sided")
    return float(odds), float(p), int(nA), int(nB), int(inter)


def _plot_heatmap(mat: pd.DataFrame, out_png: Path, title: str) -> None:
    _require("matplotlib")
    

    fig = plt.figure(figsize=(max(6, 0.6 * mat.shape[1]), max(5, 0.6 * mat.shape[0])))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_compare_stage(
    cohort_runs: List[Dict[str, str]],
    spec: "PipelineSpec",
    out_root: Path,
    universe_n: int,
) -> None:
    comp_dir = normalize_path(out_root) / "compare"
    _safe_mkdir(comp_dir)

    t_map: Dict[str, pd.Series] = {}
    fdr_map: Dict[str, pd.Series] = {}
    dir_map: Dict[str, pd.Series] = {}

    for cr in cohort_runs:
        name = cr["name"]
        ps = pd.read_csv(cr["pathway_stats"], sep="\t")
        ps = ps.dropna(subset=["pathway", "t"])
        ps["pathway"] = ps["pathway"].astype(str)
        t_map[name] = ps.set_index("pathway")["t"]
        fdr_map[name] = ps.set_index("pathway")["FDR"]

        d = np.sign(ps.set_index("pathway")["t"])
        d = d.replace(0, 1)
        d = d.where(np.isfinite(d), other=0).astype(int)
        dir_map[name] = d

    all_pathways = sorted(set().union(*[set(s.index) for s in t_map.values()]))
    t_df = pd.DataFrame({k: v.reindex(all_pathways) for k, v in t_map.items()}, index=all_pathways)

    R, P = _spearman_corr_matrix(t_df)
    R.to_csv(comp_dir / "pathway_t_spearman.tsv", sep="\t")
    P.to_csv(comp_dir / "pathway_t_spearman_p.tsv", sep="\t")
    _plot_heatmap(R, comp_dir / "pathway_t_spearman.png", "Spearman corr: pathway t-like stats")

    rows = []
    cohorts = list(t_map.keys())
    for i in range(len(cohorts)):
        for j in range(i + 1, len(cohorts)):
            a = cohorts[i]
            b = cohorts[j]

            a_sig = fdr_map[a][fdr_map[a] <= spec.sig_fdr].index
            b_sig = fdr_map[b][fdr_map[b] <= spec.sig_fdr].index
            a_sig = set(a_sig)
            b_sig = set(b_sig)

            if spec.sig_top_n > 0:
                a_top = (
                    set(t_map[a].loc[list(a_sig)].abs().sort_values(ascending=False).head(spec.sig_top_n).index)
                    if a_sig else set()
                )
                b_top = (
                    set(t_map[b].loc[list(b_sig)].abs().sort_values(ascending=False).head(spec.sig_top_n).index)
                    if b_sig else set()
                )
                a_sig = a_top
                b_sig = b_top

            shared = sorted(a_sig & b_sig)
            a_dir = dir_map[a].reindex(shared).dropna()
            b_dir = dir_map[b].reindex(shared).dropna()
            a_dir = a_dir[a_dir.isin([-1, 1])]
            b_dir = b_dir[b_dir.isin([-1, 1])]
            shared2 = a_dir.index.intersection(b_dir.index)

            chi2, p, concord, _ = _chi2_direction_agreement(a_dir.loc[shared2], b_dir.loc[shared2])
            odds, p_fish, nA, nB, nI = _fisher_overlap(a_sig, b_sig, universe_n=universe_n)

            rows.append(
                {
                    "cohort_a": a,
                    "cohort_b": b,
                    "sig_fdr": spec.sig_fdr,
                    "sig_top_n": spec.sig_top_n,
                    "n_sig_a": nA,
                    "n_sig_b": nB,
                    "n_shared_sig": nI,
                    "direction_chi2": chi2,
                    "direction_p": p,
                    "direction_concordance": concord,
                    "overlap_oddsratio": odds,
                    "overlap_p": p_fish,
                }
            )

    agree = pd.DataFrame(rows)
    if not agree.empty:
        agree["direction_FDR"] = _bh_fdr(agree["direction_p"].values)
        agree["overlap_FDR"] = _bh_fdr(agree["overlap_p"].values)
    agree.to_csv(comp_dir / "directional_agreement_chi2.tsv", sep="\t", index=False)
    agree.to_csv(comp_dir / "overlap_fisher.tsv", sep="\t", index=False)

    concord_mat = pd.DataFrame(np.nan, index=cohorts, columns=cohorts)
    for _, r in agree.iterrows():
        concord_mat.loc[r["cohort_a"], r["cohort_b"]] = r["direction_concordance"]
        concord_mat.loc[r["cohort_b"], r["cohort_a"]] = r["direction_concordance"]
    np.fill_diagonal(concord_mat.values, 1.0)
    concord_mat.to_csv(comp_dir / "directional_concordance_matrix.tsv", sep="\t")
    _plot_heatmap(concord_mat, comp_dir / "directional_agreement_chi2.png", "Directional concordance (shared sig)")

    lines = []
    lines.append("# IPAA multi-cohort report\n")
    lines.append(f"- Generated: {_now_utc()}\n")
    lines.append(f"- Cohorts: {', '.join(cohorts)}\n")
    lines.append(f"- sig_fdr: {spec.sig_fdr}, sig_top_n: {spec.sig_top_n}\n")
    lines.append("\n## Key artifacts\n")
    lines.append("- pathway_t_spearman.tsv / .png\n")
    lines.append("- directional_agreement_chi2.tsv / .png\n")
    lines.append("- overlap_fisher.tsv\n")
    (comp_dir / "compare_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Spec dataclasses
# =============================================================================
@dataclass
class SupportingToggles:
    tf: bool = False
    progeny: bool = False
    intercell: bool = False
    kinase: bool = False


@dataclass
class CohortSpec:
    name: str
    expr: str
    counts: Optional[bool] = None
    meta: Optional[str] = None
    meta_sample_col: str = "sample"
    meta_group_col: str = "group"
    group_map: Optional[Dict[str, str]] = None  # NEW (optional): explicit mapping
    control_regex: str = r"(control|ctrl|normal|healthy|untreated)"
    case_regex: Optional[str] = None
    tissue: Optional[str] = None


@dataclass
class PipelineSpec:
    msigdb_dbver: Optional[str] = None
    gsea_permutations: int = 1000
    threads: int = 4
    reduced_overlap_jaccard: Optional[float] = None
    trim_frac: float = 0.025
    min_pathway_genes: int = 10

    sig_fdr: float = 0.01
    sig_top_n: int = 300

    supporting: SupportingToggles = field(default_factory=SupportingToggles)

    run_baseline: bool = False
    baseline_dir: Optional[str] = None
    hpa_file: str = "HPA.tsv"
    gtex_file: str = "GTEx.tsv"
    fantom_file: str = "FANTOM.tsv"

    auto_select_tissue: bool = False
    tissue_top_k: int = 3

    cohorts: List[CohortSpec] = field(default_factory=list)


def load_spec(path: Path) -> PipelineSpec:
    path = normalize_path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    sup = raw.get("supporting", {}) or {}
    supporting = SupportingToggles(
        tf=bool(sup.get("tf", False)),
        progeny=bool(sup.get("progeny", False)),
        intercell=bool(sup.get("intercell", False)),
        kinase=bool(sup.get("kinase", False)),
    )

    ps = PipelineSpec(
        msigdb_dbver=raw.get("msigdb_dbver"),
        gsea_permutations=int(raw.get("gsea_permutations", 1000)),
        threads=int(raw.get("threads", 4)),
        reduced_overlap_jaccard=raw.get("reduced_overlap_jaccard"),
        trim_frac=float(raw.get("trim_frac", 0.025)),
        min_pathway_genes=int(raw.get("min_pathway_genes", 10)),
        sig_fdr=float(raw.get("sig_fdr", 0.01)),
        sig_top_n=int(raw.get("sig_top_n", 300)),
        supporting=supporting,
        run_baseline=bool(raw.get("run_baseline", False)),
        baseline_dir=raw.get("baseline_dir"),
        hpa_file=raw.get("hpa_file", "HPA.tsv"),
        gtex_file=raw.get("gtex_file", "GTEx.tsv"),
        fantom_file=raw.get("fantom_file", "FANTOM.tsv"),
        auto_select_tissue=bool(raw.get("auto_select_tissue", False)),
        tissue_top_k=int(raw.get("tissue_top_k", 3)),
        cohorts=[],
    )

    cohorts_raw = raw.get("cohorts", [])
    if not cohorts_raw:
        raise ValueError("Spec must contain a non-empty 'cohorts' list.")
    ps.cohorts = [CohortSpec(**c) for c in cohorts_raw]
    return ps


# =============================================================================
# Output layout helpers (FIXES your downstream expectations)
# =============================================================================
def _cohort_dirs(out_root: Path, cohort_name: str) -> Tuple[Path, Path]:
    out_root = normalize_path(out_root)
    primary = out_root / cohort_name
    legacy = out_root / "cohorts" / cohort_name
    return primary, legacy


def _maybe_write_legacy_redirect(legacy_dir: Path, primary_dir: Path) -> None:
    try:
        if legacy_dir.parent.exists() or legacy_dir.exists():
            _safe_mkdir(legacy_dir)
            (legacy_dir / "REDIRECT.txt").write_text(
                f"This cohort output is written to:\n{primary_dir}\n",
                encoding="utf-8",
            )
    except Exception:
        return


def _mirror_key_outputs_to_legacy(primary_dir: Path, legacy_dir: Path) -> None:
    try:
        if not legacy_dir.exists():
            return
        keys = [
            "pathway_stats.tsv",
            "pathway_stats_with_baseline.tsv",
            "de_gene_stats.tsv",
            "prerank.rnk.tsv",
            "COHORT_MANIFEST.json",
            "TISSUE_SELECTION.json",
        ]
        for k in keys:
            src = primary_dir / k
            dst = legacy_dir / k
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
    except Exception:
        return


# =============================================================================
# Per-cohort runner
# =============================================================================
def run_one_cohort(
    cohort: CohortSpec,
    spec: PipelineSpec,
    out_root: Path,
    gmt: Dict[str, List[str]],
    counts_default: bool,
    tissue_expect: Optional[pd.DataFrame],
    seed: int = 0,
    refresh_omnipath: bool = False,
) -> Dict[str, str]:
    out_root = normalize_path(out_root)
    cohort_dir, legacy_dir = _cohort_dirs(out_root, cohort.name)
    _safe_mkdir(cohort_dir)
    _maybe_write_legacy_redirect(legacy_dir, cohort_dir)

    expr_path = normalize_path(cohort.expr)
    if not expr_path.exists():
        raise FileNotFoundError(f"[{cohort.name}] Expression file not found: {expr_path}")

    LOG.info("=== Cohort: %s ===", cohort.name)
    X_raw = load_expression_matrix(expr_path)
    LOG.info("[%s] Loaded X: samples=%d, genes=%d", cohort.name, X_raw.shape[0], X_raw.shape[1])

    # labels (UPDATED: metadata alignment is now robust; no behavioral change when already matching)
    if cohort.meta:
        meta_path = normalize_path(cohort.meta)
        y_meta = load_metadata(
            meta_path,
            cohort.meta_sample_col,
            cohort.meta_group_col,
            group_map=cohort.group_map,
        )
        y = _align_labels_to_samples(y_meta, X_raw.index)

        n_case = int((y == "case").sum())
        n_ctrl = int((y == "control").sum())
        if n_case < 1 or n_ctrl < 1:
            raise ValueError(
                f"[{cohort.name}] Need >=1 case and >=1 control after metadata alignment. "
                f"Got case={n_case}, control={n_ctrl}."
            )
    else:
        y = infer_groups_from_sample_names(
            X_raw.index,
            control_regex=cohort.control_regex,
            case_regex=cohort.case_regex,
        )

    counts_mode = counts_default if cohort.counts is None else bool(cohort.counts)
    X = counts_to_logcpm(X_raw) if counts_mode else X_raw.copy()
    base_dir = normalize_path(spec.baseline_dir).expanduser() if spec.baseline_dir else (Path(__file__).parent / "data")
    X = maybe_convert_ensembl_to_hgnc_cols(X, base_dir, spec.hpa_file, spec.gtex_file, spec.fantom_file)
    X = standardize_gene_symbols_cols(X)

    # save used inputs (same outputs)
    X.to_csv(cohort_dir / "expression_used.tsv", sep="\t", index=True)
    y.to_frame().to_csv(cohort_dir / "labels_used.tsv", sep="\t", index=True)
    write_gmt(gmt, cohort_dir / "msigdb_c2cp.gmt")

    # IPAA activity
    activity, activity_meta = compute_ipaa_activity(X, gmt, trim_frac=spec.trim_frac, min_genes=spec.min_pathway_genes)
    activity.to_csv(cohort_dir / "pathway_activity.tsv", sep="\t", index=True)
    activity_meta.to_csv(cohort_dir / "pathway_activity_meta.tsv", sep="\t", index=False)

    # AUTO tissue selection (optional)
    tissue_selected = (cohort.tissue or "").strip() if cohort.tissue else ""
    tissue_ranked: List[Tuple[str, float]] = []
    tissue_method = "user" if tissue_selected else "none"
    if not tissue_selected and spec.auto_select_tissue and tissue_expect is not None and not tissue_expect.empty:
        tissue_ranked = infer_tissue_from_ipaa_mean(
            cohort.name, activity, tissue_expect, top_k=max(1, spec.tissue_top_k)
        )
        if tissue_ranked:
            tissue_selected = tissue_ranked[0][0]
            tissue_method = "auto_ipaa_spearman"
            LOG.info("[%s] AUTO tissue selected: %s (r=%.4f)", cohort.name, tissue_selected, tissue_ranked[0][1])
        else:
            tissue_method = "auto_failed"

    # pathway stats
    pstats = differential_pathway_stats(activity, y)
    pstats.to_csv(cohort_dir / "pathway_stats.tsv", sep="\t", index=False)

    # baseline attach
    pstats2 = _attach_baseline_to_pathway_stats(pstats, tissue_selected, tissue_expect)
    pstats2.to_csv(cohort_dir / "pathway_stats_with_baseline.tsv", sep="\t", index=False)

    # gene stats + GSEA
    gene_stats = differential_gene_stats(X, y)
    gene_stats.to_csv(cohort_dir / "de_gene_stats.tsv", sep="\t", index=False)

    rnk = build_prerank(gene_stats, score_col="t")
    rnk.to_csv(cohort_dir / "prerank.rnk.tsv", sep="\t", index=False)

    gsea_dir = cohort_dir / "gsea_c2cp"
    gsea_res = run_gsea_prerank(
        rnk=rnk,
        gmt=gmt,
        outdir=gsea_dir,
        permutation_num=spec.gsea_permutations,
        seed=seed,
        threads=spec.threads,
    )
    if gsea_res is not None and not gsea_res.empty:
        gsea_res.reset_index().to_csv(cohort_dir / "gsea_c2cp_top.tsv", sep="\t", index=False)
    else:
        (cohort_dir / "gsea_c2cp_top.tsv").write_text("", encoding="utf-8")

    if spec.reduced_overlap_jaccard is not None:
        gmt_red = reduce_overlap_gmt(gmt, jaccard_thresh=float(spec.reduced_overlap_jaccard))
        gsea_dir2 = cohort_dir / f"gsea_c2cp_reduced_j{spec.reduced_overlap_jaccard}"
        gsea_res2 = run_gsea_prerank(
            rnk=rnk,
            gmt=gmt_red,
            outdir=gsea_dir2,
            permutation_num=spec.gsea_permutations,
            seed=seed,
            threads=spec.threads,
        )
        if gsea_res2 is not None and not gsea_res2.empty:
            gsea_res2.reset_index().to_csv(cohort_dir / "gsea_c2cp_reduced_top.tsv", sep="\t", index=False)
        else:
            (cohort_dir / "gsea_c2cp_reduced_top.tsv").write_text("", encoding="utf-8")

    # Optional supporting layers (non-fatal by design)
    if spec.supporting.tf:
        try:
            _require("decoupler")
            net = get_collectri_net(refresh=refresh_omnipath)
            net = _net_standardize(net)

            X2 = X.copy()
            X2.columns = X2.columns.astype(str).str.upper()
            net["source"] = net["source"].astype(str).str.upper()
            net["target"] = net["target"].astype(str).str.upper()
            net = net[net["target"].isin(set(X2.columns))].copy()

            if not net.empty:
                tf_dir = cohort_dir / "tf_activity"
                _safe_mkdir(tf_dir)

                ulm_est, ulm_p = _dc_ulm(X2, net, min_n=5)
                vip_est, vip_p = _dc_viper(X2, net, min_n=5)

                ulm_est.to_csv(tf_dir / "tf_ulm.tsv", sep="\t")
                vip_est.to_csv(tf_dir / "tf_viper.tsv", sep="\t")
                ulm_p.to_csv(tf_dir / "tf_ulm_pvals.tsv", sep="\t")
                vip_p.to_csv(tf_dir / "tf_viper_pvals.tsv", sep="\t")

                differential_activity_stats(ulm_est, y).to_csv(tf_dir / "tf_ulm_diff.tsv", sep="\t", index=False)
                differential_activity_stats(vip_est, y).to_csv(tf_dir / "tf_viper_diff.tsv", sep="\t", index=False)
        except Exception as e:
            LOG.warning("[%s] TF activity failed (non-fatal): %s", cohort.name, e)

    if spec.supporting.progeny:
        try:
            _require("decoupler")
            net = get_progeny_net(refresh=refresh_omnipath, top=100)
            if "pathway" in net.columns:
                net = net.rename(columns={"pathway": "source"})
            net = _net_standardize(net)

            X2 = X.copy()
            X2.columns = X2.columns.astype(str).str.upper()
            net["source"] = net["source"].astype(str)
            net["target"] = net["target"].astype(str).str.upper()
            net = net[net["target"].isin(set(X2.columns))].copy()

            if not net.empty:
                prog_dir = cohort_dir / "signaling_progeny"
                _safe_mkdir(prog_dir)
                est, pvs = _dc_ulm(X2, net, min_n=5)
                est.to_csv(prog_dir / "progeny_ulm.tsv", sep="\t")
                pvs.to_csv(prog_dir / "progeny_ulm_pvals.tsv", sep="\t")
                differential_activity_stats(est, y).to_csv(prog_dir / "progeny_ulm_diff.tsv", sep="\t", index=False)
        except Exception as e:
            LOG.warning("[%s] PROGENy failed (non-fatal): %s", cohort.name, e)

    if spec.supporting.intercell:
        try:
            df = get_omnipath_intercell(refresh=refresh_omnipath)
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = list(df.columns.astype(str))
                gene_col = _pick_col_by_regex(cols, [r"gene.*symbol"]) or _pick_col_by_regex(cols, [r"symbol"])
                cat_col = _pick_col_by_regex(cols, [r"category"]) or _pick_col_by_regex(cols, [r"parent"])
                if gene_col and cat_col:
                    net = df[[cat_col, gene_col]].copy()
                    net.columns = ["source", "target"]
                    net["weight"] = 1.0
                    net["source"] = net["source"].astype(str)
                    net["target"] = net["target"].astype(str).str.upper()

                    X2 = X.copy()
                    X2.columns = X2.columns.astype(str).str.upper()
                    net = net[net["target"].isin(set(X2.columns))].dropna().drop_duplicates()

                    counts = net.groupby("source")["target"].nunique()
                    keep = set(counts[counts >= 5].index)
                    net = net[net["source"].isin(keep)]
                    if not net.empty:
                        inter_dir = cohort_dir / "intercell_categories"
                        _safe_mkdir(inter_dir)
                        est, pvs = _dc_ulm(X2, net, min_n=5)
                        est.to_csv(inter_dir / "intercell_ulm.tsv", sep="\t")
                        pvs.to_csv(inter_dir / "intercell_ulm_pvals.tsv", sep="\t")
                        net.to_csv(inter_dir / "intercell_net.tsv", sep="\t", index=False)
                        differential_activity_stats(est, y).to_csv(inter_dir / "intercell_ulm_diff.tsv", sep="\t", index=False)
        except Exception as e:
            LOG.warning("[%s] Intercell failed (non-fatal): %s", cohort.name, e)

    if spec.supporting.kinase:
        try:
            df = get_omnipath_signedptms(refresh=refresh_omnipath)
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = list(df.columns.astype(str))
                src = _pick_col_by_regex(cols, [r"enzyme.*genesymbol", r"enzyme", r"kinase", r"source"])
                tgt = _pick_col_by_regex(cols, [r"substrate.*genesymbol", r"substrate", r"target"])
                wcol = _pick_col_by_regex(cols, [r"mor", r"effect", r"sign", r"weight"])

                if src and tgt:
                    net = df[[src, tgt]].copy()
                    net.columns = ["source", "target"]
                    if wcol and wcol in df.columns:
                        ww = pd.to_numeric(df[wcol], errors="coerce")
                        ww = ww.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                        net["weight"] = ww.values
                    else:
                        net["weight"] = 1.0

                    net["source"] = net["source"].astype(str).str.upper()
                    net["target"] = net["target"].astype(str).str.upper()
                    net = net.dropna().drop_duplicates()

                    X2 = X.copy()
                    X2.columns = X2.columns.astype(str).str.upper()
                    net = net[net["target"].isin(set(X2.columns))].copy()

                    counts = net.groupby("source")["target"].nunique()
                    keep = set(counts[counts >= 5].index)
                    net = net[net["source"].isin(keep)]
                    if not net.empty:
                        kin_dir = cohort_dir / "kinase_activity"
                        _safe_mkdir(kin_dir)
                        est, pvs = _dc_viper(X2, net, min_n=5)
                        est.to_csv(kin_dir / "kinase_viper.tsv", sep="\t")
                        pvs.to_csv(kin_dir / "kinase_viper_pvals.tsv", sep="\t")
                        net.to_csv(kin_dir / "kinase_net.tsv", sep="\t", index=False)
                        differential_activity_stats(est, y).to_csv(kin_dir / "kinase_viper_diff.tsv", sep="\t", index=False)
        except Exception as e:
            LOG.warning("[%s] Kinase proxy failed (non-fatal): %s", cohort.name, e)

    # Tissue selection manifest (same)
    tissue_block = {
        "selected_tissue": tissue_selected,
        "method": tissue_method,
        "top_candidates": [{"tissue": t, "spearman_r": r} for t, r in tissue_ranked],
    }
    (cohort_dir / "TISSUE_SELECTION.json").write_text(json.dumps(tissue_block, indent=2), encoding="utf-8")

    cm = {
        "timestamp": _now_utc(),
        "cohort": asdict(cohort),
        "counts_mode_used": counts_mode,
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "msigdb_dbver": spec.msigdb_dbver,
        "trim_frac": spec.trim_frac,
        "min_pathway_genes": spec.min_pathway_genes,
        "tissue": tissue_block,
        "refresh_omnipath": bool(refresh_omnipath),
        "paths": {
            "primary_cohort_dir": str(cohort_dir),
            "legacy_cohort_dir": str(legacy_dir),
        },
    }
    (cohort_dir / "COHORT_MANIFEST.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")

    _mirror_key_outputs_to_legacy(cohort_dir, legacy_dir)

    return {
        "name": cohort.name,
        "cohort_dir": str(cohort_dir),
        "pathway_stats": str(cohort_dir / "pathway_stats.tsv"),
        "pathway_activity": str(cohort_dir / "pathway_activity.tsv"),
    }


# =============================================================================
# Parallel cohort budgeting helper
# =============================================================================
def choose_workers(n_cohorts: int, threads_per_cohort: int) -> Tuple[int, int, int]:
    """
    Returns (workers, cpu, threads_per_cohort_used).

    We assume GSEA uses threads_per_cohort; using too many process workers can oversubscribe.
    """
    cpu = os.cpu_count() or 1
    tpc = max(1, int(threads_per_cohort))
    workers = max(1, cpu // tpc)
    workers = min(workers, n_cohorts)
    return int(workers), int(cpu), int(tpc)
