#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .mdp_engine.activity import ipaa_activity
from .mdp_engine.exceptions import DataError, ValidationError
from .mdp_engine.logging_utils import configure_logging, get_logger

log = get_logger("run_causality_all")


# -----------------------------
# Small safe IO helpers
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_tsv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    _ensure_dir(path.parent)
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
    """
    Heuristic orientation check. If index seems more like gene symbols than columns, transpose.
    """
    idx = expr.index.astype(str)
    cols = expr.columns.astype(str)

    idx_geneish = sum(1 for x in idx[:200] if re.fullmatch(r"[A-Z0-9\-]{2,20}", str(x).strip().upper()) is not None)
    col_geneish = sum(1 for x in cols[:200] if re.fullmatch(r"[A-Z0-9\-]{2,20}", str(x).strip().upper()) is not None)

    if idx_geneish > col_geneish:
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
    if df.empty:
        raise DataError(f"Expression is empty after cleaning: {expr_path}")
    return df


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


# -----------------------------
# Dynamic import of Engine2/3 script
# -----------------------------
@dataclass
class Engine23Module:
    run_omnipath_layer_for_disease: Any
    run_engine2_confounding: Any
    run_engine3_contextualization: Any


def _import_engine2_3_module(script_path: Path) -> Engine23Module:
    if not script_path.exists():
        raise ValidationError(f"Cannot find Engine2/3 script: {script_path}")

    spec = importlib.util.spec_from_file_location("engine2_3_mechanistic", str(script_path))
    if spec is None or spec.loader is None:
        raise ValidationError(f"Could not import module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["engine2_3_mechanistic"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    required = ["run_omnipath_layer_for_disease", "run_engine2_confounding", "run_engine3_contextualization"]
    for name in required:
        if not hasattr(module, name):
            raise ValidationError(f"Imported {script_path} but missing required symbol: {name}")

    return Engine23Module(
        run_omnipath_layer_for_disease=getattr(module, "run_omnipath_layer_for_disease"),
        run_engine2_confounding=getattr(module, "run_engine2_confounding"),
        run_engine3_contextualization=getattr(module, "run_engine3_contextualization"),
    )


# ==========================================================
# Engine1: Causal Features Builder (minimal)
# ==========================================================
def _find_gmt(cohort: Path) -> Optional[Path]:
    candidates = [
        cohort / "msigdb_c2cp.gmt",
        cohort / "gsea_c2cp" / "msigdb_c2cp.gmt",
        cohort / "gsea_c2cp" / "gene_sets.gmt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_gmt(gmt_path: Path, max_sets: int = 2000) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with gmt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= int(max_sets):
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = {str(g).strip().upper() for g in parts[2:] if str(g).strip()}
            if len(genes) >= 3:
                out[name] = genes
    return out


def _parse_tf_names_from_enrichr_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            return []

    term_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"term", "tf", "name"}:
            term_col = c
            break
    if term_col is None or df.empty:
        return []

    tfs: List[str] = []
    for raw in df[term_col].astype(str).tolist():
        s = raw.strip()
        if not s:
            continue
        sym = re.split(r"[_\s\(\)]", s, maxsplit=1)[0]
        sym = sym.strip().upper()
        if re.fullmatch(r"[A-Z0-9\-]{2,20}", sym) is None:
            continue
        tfs.append(sym)

    seen: Set[str] = set()
    uniq: List[str] = []
    for x in tfs:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _normalize_engine1_method(m: str) -> str:
    """
    Minimal engine guarantees only 'mean'. Others are accepted but normalized so config never lies.
    """
    m0 = str(m).strip().lower()
    if m0 != "mean":
        log.warning("Engine1 minimal: method '%s' not supported here; using 'mean'.", m0)
    return "mean"


def _build_tf_activity_proxy_from_expr(expr: pd.DataFrame, cohort: Path, max_tfs: int = 300) -> pd.DataFrame:
    candidates: List[str] = []
    candidates += _parse_tf_names_from_enrichr_csv(cohort / "tf_enrich_up.csv")
    candidates += _parse_tf_names_from_enrichr_csv(cohort / "tf_enrich_down.csv")
    if not candidates:
        return pd.DataFrame(index=expr.index)

    genes = set(expr.columns.astype(str))
    keep = [t for t in candidates if t in genes][: int(max_tfs)]
    if not keep:
        return pd.DataFrame(index=expr.index)

    X = expr[keep].copy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0.0, 1.0)
    Z = (X - mu) / sd
    Z.columns = [f"TF:{c}" for c in Z.columns.astype(str)]
    return Z


def _build_pathway_footprints(
    expr: pd.DataFrame,
    gmt: Dict[str, Set[str]],
    method: str,
    min_size: int,
    max_pathways: int,
) -> pd.DataFrame:
    if not gmt:
        return pd.DataFrame(index=expr.index)

    items = [(k, v) for (k, v) in gmt.items() if len(v) >= int(min_size)]
    if not items:
        return pd.DataFrame(index=expr.index)

    if len(items) > int(max_pathways):
        items = items[: int(max_pathways)]

    gmt2 = {k: v for k, v in items}
    pw = ipaa_activity(
        expression=expr,
        pathways=gmt2,
        method=str(method),
        standardize_pathways=True,
        min_size=int(min_size),
    )
    pw.columns = [f"PW:{c}" for c in pw.columns.astype(str)]
    return pw


def run_engine1_causal_features_minimal(
    out_root: Path,
    disease: str,
    *,
    method: str = "mean",
    min_size: int = 10,
    max_pathways: int = 1500,
    max_tfs: int = 300,
    force: bool = True,
    strict: bool = False,
) -> Tuple[str, Path]:
    """
    Produces:
      OUT_ROOT/engines/causal_pathway_features/<disease>/
        - tf_activity.tsv
        - pathway_footprint_activity.tsv
        - feature_matrix.tsv
        - feature_provenance.json
        - ENGINE_MANIFEST.json

    Minimal mode requires expression_used.* to exist for per-sample outputs.
    """
    out_root = out_root.resolve()
    cohort = _cohort_dir(out_root, disease)

    out_dir = out_root / "engines" / "causal_pathway_features" / disease
    _ensure_dir(out_dir)

    tf_path = out_dir / "tf_activity.tsv"
    pw_path = out_dir / "pathway_footprint_activity.tsv"
    fm_path = out_dir / "feature_matrix.tsv"
    prov_path = out_dir / "feature_provenance.json"
    manifest_path = out_dir / "ENGINE_MANIFEST.json"
    skipped_path = out_dir / "SKIPPED.txt"

    if (tf_path.exists() and pw_path.exists() and fm_path.exists()) and not force:
        return "ok_existing", out_dir

    method_used = _normalize_engine1_method(method)

    expr_path = _find_expression_used(cohort)
    if expr_path is None:
        msg = f"Engine1 skipped: missing expression_used.* in {cohort}"
        if strict:
            raise DataError(msg)
        _atomic_write_text(skipped_path, msg + "\n")
        _atomic_write_text(
            manifest_path,
            json.dumps({"engine": "causal_pathway_features", "status": "skipped", "reason": msg}, indent=2),
        )
        _atomic_write_tsv(pd.DataFrame(), tf_path, index=True)
        _atomic_write_tsv(pd.DataFrame(), pw_path, index=True)
        _atomic_write_tsv(pd.DataFrame(), fm_path, index=True)
        _atomic_write_text(prov_path, json.dumps({"mode": "skipped"}, indent=2))
        return "skipped", out_dir

    gmt_path = _find_gmt(cohort)
    gmt = _load_gmt(gmt_path, max_sets=5000) if gmt_path else {}

    expr = _read_expression(expr_path)
    tf_df = _build_tf_activity_proxy_from_expr(expr, cohort, max_tfs=int(max_tfs))
    _atomic_write_tsv(tf_df, tf_path, index=True)

    try:
        pw_df = _build_pathway_footprints(
            expr=expr,
            gmt=gmt,
            method=str(method_used),
            min_size=int(min_size),
            max_pathways=int(max_pathways),
        )
    except Exception as e:
        if strict:
            raise
        log.warning("Engine1: pathway footprints failed (%s): %s", type(e).__name__, e)
        pw_df = pd.DataFrame(index=expr.index)

    _atomic_write_tsv(pw_df, pw_path, index=True)

    fm = pd.concat([tf_df, pw_df], axis=1)
    fm = fm.loc[:, ~fm.columns.duplicated()].copy()
    fm = fm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    _atomic_write_tsv(fm, fm_path, index=True)

    provenance = {
        "engine": "engine1_minimal",
        "mode": "per_sample",
        "expression_used": str(expr_path),
        "gmt_used": str(gmt_path) if gmt_path else None,
        "tf_method": "zscore(TF_gene_expression) from Enrichr TF list (tf_enrich_up/down.csv)",
        "pw_method": f"ipaa_activity(method={method_used}) on GMT gene sets",
        "requested_pw_method": str(method),
        "params": {"min_size": int(min_size), "max_pathways": int(max_pathways), "max_tfs": int(max_tfs)},
        "notes": [
            "Minimal Engine1: TF activity is a proxy (TF gene z-score), not regulon-based activity.",
        ],
    }
    _atomic_write_text(prov_path, json.dumps(provenance, indent=2))

    manifest = {
        "engine": "causal_pathway_features",
        "version": "1.1.0-minimal",
        "status": "ok",
        "mode": "per_sample",
        "inputs": {"cohort_dir": str(cohort), "expression_used": str(expr_path), "gmt": str(gmt_path) if gmt_path else None},
        "outputs": {"tf_activity": str(tf_path), "pathway_footprints": str(pw_path), "feature_matrix": str(fm_path), "provenance": str(prov_path)},
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    if skipped_path.exists():
        try:
            skipped_path.unlink()
        except Exception:
            pass

    return "ok", out_dir


# ==========================================================
# Engine0: Evidence Bundle Builder (minimal join-point)
# ==========================================================
def _pick_de_gene_stats(cohort: Path) -> Optional[Path]:
    for p in (cohort / "de_gene_stats.tsv", cohort / "de_gene_stats.csv"):
        if p.exists():
            return p
    return None


def _pick_pathway_stats(cohort: Path) -> Optional[Path]:
    candidates = [
        cohort / "pathway_stats_with_baseline_filtered_classified.tsv",
        cohort / "pathway_stats_with_baseline_filtered.tsv",
        cohort / "pathway_stats_with_baseline.tsv",
        cohort / "pathway_stats.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_engine0_evidence_bundle_minimal(
    out_root: Path,
    disease: str,
    *,
    force: bool = True,
    strict: bool = False,
) -> Tuple[str, Path]:
    """
    Minimal evidence bundle join-point. Writes:
      OUT_ROOT/engines/evidence_bundle/<disease>/
        - genes_evidence.tsv
        - pathways_evidence.tsv
        - regulators_evidence.tsv
        - mechanism_summary.json
        - ENGINE_MANIFEST.json
    """
    out_root = out_root.resolve()
    cohort = _cohort_dir(out_root, disease)

    out_dir = out_root / "engines" / "evidence_bundle" / disease
    _ensure_dir(out_dir)

    genes_out = out_dir / "genes_evidence.tsv"
    pws_out = out_dir / "pathways_evidence.tsv"
    regs_out = out_dir / "regulators_evidence.tsv"
    summary_out = out_dir / "mechanism_summary.json"
    manifest_out = out_dir / "ENGINE_MANIFEST.json"
    skipped_out = out_dir / "SKIPPED.txt"

    if (genes_out.exists() and pws_out.exists() and regs_out.exists() and summary_out.exists()) and not force:
        return "ok_existing", out_dir

    de_path = _pick_de_gene_stats(cohort)
    pstats_path = _pick_pathway_stats(cohort)
    expr_path = _find_expression_used(cohort)

    if de_path is None and pstats_path is None:
        msg = f"Engine0 skipped: missing de_gene_stats and pathway_stats for {disease}"
        if strict:
            raise DataError(msg)
        _atomic_write_text(skipped_out, msg + "\n")
        _atomic_write_text(manifest_out, json.dumps({"engine": "evidence_bundle", "status": "skipped", "reason": msg}, indent=2))
        _atomic_write_tsv(pd.DataFrame(), genes_out, index=False)
        _atomic_write_tsv(pd.DataFrame(), pws_out, index=False)
        _atomic_write_tsv(pd.DataFrame(), regs_out, index=False)
        _atomic_write_text(summary_out, json.dumps({"disease": disease, "status": "skipped", "reason": msg}, indent=2))
        return "skipped", out_dir

    expressed_genes: Set[str] = set()
    if expr_path is not None:
        try:
            sep = "\t" if expr_path.suffix.lower() in {".tsv", ".txt"} else ","
            tmp = pd.read_csv(expr_path, sep=sep, engine="python", index_col=0, nrows=1)
            expressed_genes = {str(c).strip().upper() for c in tmp.columns.astype(str)}
        except Exception:
            expressed_genes = set()

    # genes evidence
    genes_df = pd.DataFrame(columns=["gene_id", "symbol", "logFC", "stat", "p", "q", "direction", "expressed_flag"])
    if de_path is not None:
        try:
            de = _read_table(de_path)
            low = {str(c).lower().replace(" ", "").replace("_", ""): c for c in de.columns}
            gene_col = low.get("gene") or low.get("genes") or low.get("symbol") or low.get("genesymbol")
            lfc_col = low.get("log2foldchange") or low.get("log2fc") or low.get("logfc") or low.get("lfc")
            stat_col = low.get("stat") or low.get("t") or low.get("tlike") or low.get("t_like")
            pval_col = low.get("pvalue") or low.get("pval") or low.get("p")
            q_col = low.get("padj") or low.get("fdr") or low.get("qvalue") or low.get("adjpvalue") or low.get("adjpval")

            if gene_col and lfc_col:
                sym = de[gene_col].astype(str).str.strip().str.upper()
                logfc = pd.to_numeric(de[lfc_col], errors="coerce").fillna(0.0)

                stat = pd.to_numeric(de[stat_col], errors="coerce") if (stat_col and stat_col in de.columns) else pd.Series([np.nan] * len(de))
                p = pd.to_numeric(de[pval_col], errors="coerce") if (pval_col and pval_col in de.columns) else pd.Series([np.nan] * len(de))
                q = pd.to_numeric(de[q_col], errors="coerce") if (q_col and q_col in de.columns) else pd.Series([np.nan] * len(de))

                direction = np.where(logfc > 0, "UP", np.where(logfc < 0, "DOWN", "ZERO"))
                expressed_flag = sym.isin(pd.Index(list(expressed_genes))).astype(int) if expressed_genes else pd.Series([np.nan] * len(de))

                genes_df = pd.DataFrame(
                    {
                        "gene_id": sym,
                        "symbol": sym,
                        "logFC": logfc,
                        "stat": stat,
                        "p": p,
                        "q": q,
                        "direction": direction,
                        "expressed_flag": expressed_flag,
                    }
                )
                genes_df["_abs"] = genes_df["logFC"].abs()
                genes_df = genes_df.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(500).reset_index(drop=True)
        except Exception as e:
            if strict:
                raise
            log.warning("Engine0: genes_evidence build failed (%s): %s", type(e).__name__, e)

    _atomic_write_tsv(genes_df, genes_out, index=False)

    # pathways evidence (minimal placeholder; you can enrich later)
    pws_df = pd.DataFrame(columns=["pathway_id", "name", "direction", "Main_Class", "Sub_Class"])
    if pstats_path is not None:
        try:
            ps = _read_table(pstats_path)
            name_col = None
            for c in ps.columns:
                if str(c).strip().lower() in {"pathway", "term", "name"}:
                    name_col = c
                    break
            if name_col is None:
                name_col = ps.columns[0]

            tmp = ps.copy()
            tmp["name"] = tmp[name_col].astype(str)

            if "direction" in tmp.columns:
                direction = tmp["direction"].astype(str)
            elif "t_like" in tmp.columns:
                tl = pd.to_numeric(tmp["t_like"], errors="coerce").fillna(0.0)
                direction = np.where(tl > 0, "UP", np.where(tl < 0, "DOWN", "ZERO"))
            else:
                direction = "NA"

            out_rows = []
            for _, r in tmp.iterrows():
                nm = str(r["name"])
                out_rows.append(
                    {
                        "pathway_id": nm,
                        "name": nm,
                        "direction": str(direction.iloc[int(r.name)]) if hasattr(direction, "iloc") else str(direction),
                        "Main_Class": str(r.get("Main_Class", "")) if "Main_Class" in tmp.columns else "",
                        "Sub_Class": str(r.get("Sub_Class", "")) if "Sub_Class" in tmp.columns else "",
                    }
                )
            pws_df = pd.DataFrame(out_rows).head(500)
        except Exception as e:
            if strict:
                raise
            log.warning("Engine0: pathways_evidence build failed (%s): %s", type(e).__name__, e)

    _atomic_write_tsv(pws_df, pws_out, index=False)

    # regulators evidence (minimal: TF mean + PTM NES if present)
    regs_rows: List[Dict[str, Any]] = []
    tf_path = out_root / "engines" / "causal_pathway_features" / disease / "tf_activity.tsv"
    if tf_path.exists():
        try:
            tf = pd.read_csv(tf_path, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            m = tf.mean(axis=0).sort_values(key=lambda s: s.abs(), ascending=False).head(300)
            for k, v in m.items():
                rid = str(k).replace("TF:", "", 1).strip().upper()
                direction = "UP" if float(v) > 0 else ("DOWN" if float(v) < 0 else "ZERO")
                regs_rows.append(
                    {
                        "regulator_id": rid,
                        "type": "TF",
                        "activity": float(v),
                        "p": np.nan,
                        "direction": direction,
                        "method": "Engine1_TF_proxy",
                        "regulon_source": "Enrichr_TF_list (proxy only)",
                    }
                )
        except Exception as e:
            if strict:
                raise
            log.warning("Engine0: tf_activity read failed (%s): %s", type(e).__name__, e)

    ptm_path = out_root / "engines" / "omnipath_layer" / disease / f"{disease}_PTM_kinase_activity.tsv"
    if ptm_path.exists():
        try:
            ptm = pd.read_csv(ptm_path, sep="\t")
            if not ptm.empty and "enzyme" in ptm.columns and ("NES" in ptm.columns):
                ptm = ptm.sort_values("NES", ascending=False)
                for _, r in ptm.head(300).iterrows():
                    enz = str(r.get("enzyme", "")).strip().upper()
                    val = float(r.get("NES", 0.0))
                    direction = "UP" if val > 0 else ("DOWN" if val < 0 else "ZERO")
                    regs_rows.append(
                        {
                            "regulator_id": enz,
                            "type": "Kinase",
                            "activity": val,
                            "p": np.nan,
                            "direction": direction,
                            "method": "OmniPath_PTM_rank_enrichment",
                            "regulon_source": "OmniPath_PTMs",
                        }
                    )
        except Exception as e:
            if strict:
                raise
            log.warning("Engine0: PTM activity read failed (%s): %s", type(e).__name__, e)

    regs_df = pd.DataFrame(
        regs_rows,
        columns=["regulator_id", "type", "activity", "p", "direction", "method", "regulon_source"],
    )
    _atomic_write_tsv(regs_df, regs_out, index=False)

    summary = {
        "disease": disease,
        "status": "ok",
        "inputs": {
            "cohort_dir": str(cohort),
            "de_gene_stats": str(de_path) if de_path else None,
            "pathway_stats": str(pstats_path) if pstats_path else None,
            "expression_used": str(expr_path) if expr_path else None,
        },
        "counts": {
            "genes_rows": int(len(genes_df)),
            "pathways_rows": int(len(pws_df)),
            "regulators_rows": int(len(regs_df)),
        },
        "disclaimer": "Engine0 is a minimal evidence bundle join-point; it is best-effort and additive.",
    }
    _atomic_write_text(summary_out, json.dumps(summary, indent=2))

    manifest = {
        "engine": "evidence_bundle",
        "version": "1.1.0-minimal",
        "status": "ok",
        "outputs": {
            "genes_evidence": str(genes_out),
            "pathways_evidence": str(pws_out),
            "regulators_evidence": str(regs_out),
            "mechanism_summary": str(summary_out),
        },
    }
    _atomic_write_text(manifest_out, json.dumps(manifest, indent=2))

    if skipped_out.exists():
        try:
            skipped_out.unlink()
        except Exception:
            pass

    return "ok", out_dir


# ==========================================================
# CLI
# ==========================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a full causality workflow for one or many diseases under an IPAA OUT_ROOT:\n"
            "  Engine1 (features) + OmniPath layer (+PKN) + Engine2 (confounding) + Engine3 (context) + Engine0 (evidence bundle)\n"
            "\n"
            "IMPORTANT DEFAULTS:\n"
            "  --refresh-omnipath-cache is ON by default\n"
            "  --build-pkn is ON by default\n"
            "  --refresh-pkn is ON by default\n"
            "  --force-engine1 is ON by default\n"
            "  --force-engine0 is ON by default\n"
            "Use the corresponding --no-* flags to disable."
        )
    )

    ap.add_argument("--out-root", required=True, help="OUT_ROOT produced by IPAA.")
    ap.add_argument("--disease", action="append", default=None, help="Disease name (repeatable). If omitted, use --all.")
    ap.add_argument("--all", action="store_true", help="Run on all discovered diseases under OUT_ROOT.")
    ap.add_argument("--strict", action="store_true", help="If set, missing inputs raise errors instead of SKIPPED.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Contract-safe import path for engine2/3 module
    ap.add_argument(
        "--engine23-script",
        default=None,
        help="Path to run_engine2_3_mechanistic.py. Default: same folder as this script.",
    )

    # Engine toggles (OFF by default; user opts out explicitly)
    ap.add_argument("--no-engine0", action="store_true", help="Disable Engine0 evidence bundle.")
    ap.add_argument("--no-engine1", action="store_true", help="Disable Engine1 causal features builder.")
    ap.add_argument("--no-engine2", action="store_true", help="Disable Engine2 confounding.")
    ap.add_argument("--no-engine3", action="store_true", help="Disable Engine3 contextualization.")
    ap.add_argument("--no-omnipath-layer", action="store_true", help="Disable OmniPath layer (used to build caches/PKN).")

    # Force rebuild (DEFAULT ON)
    ap.add_argument(
        "--force-engine1",
        dest="force_engine1",
        action="store_true",
        default=True,
        help="Force rebuild Engine1 outputs (default: ON). Use --no-force-engine1 to disable.",
    )
    ap.add_argument(
        "--no-force-engine1",
        dest="force_engine1",
        action="store_false",
        help="Do not force rebuild Engine1 outputs.",
    )

    ap.add_argument(
        "--force-engine0",
        dest="force_engine0",
        action="store_true",
        default=True,
        help="Force rebuild Engine0 outputs (default: ON). Use --no-force-engine0 to disable.",
    )
    ap.add_argument(
        "--no-force-engine0",
        dest="force_engine0",
        action="store_false",
        help="Do not force rebuild Engine0 outputs.",
    )

    # Engine1 params
    ap.add_argument("--engine1-method", default="mean", choices=["mean", "zscore", "pca1", "ssgsea"])
    ap.add_argument("--engine1-min-size", type=int, default=10)
    ap.add_argument("--engine1-max-pathways", type=int, default=1500)
    ap.add_argument("--engine1-max-tfs", type=int, default=300)

    # OmniPath/PKN controls (DEFAULT ON)
    ap.add_argument(
        "--refresh-omnipath-cache",
        dest="refresh_omnipath_cache",
        action="store_true",
        default=True,
        help="Re-download OmniPath PTM/intercell/interactions caches (default: ON). Use --no-refresh-omnipath-cache to disable.",
    )
    ap.add_argument(
        "--no-refresh-omnipath-cache",
        dest="refresh_omnipath_cache",
        action="store_false",
        help="Disable OmniPath cache refresh.",
    )

    ap.add_argument(
        "--build-pkn",
        dest="build_pkn",
        action="store_true",
        default=True,
        help="Build/ensure global PKN edges from OmniPath interactions (default: ON). Use --no-build-pkn to disable.",
    )
    ap.add_argument(
        "--no-build-pkn",
        dest="build_pkn",
        action="store_false",
        help="Disable PKN building.",
    )

    ap.add_argument(
        "--refresh-pkn",
        dest="refresh_pkn",
        action="store_true",
        default=True,
        help="Force rebuild pkn_edges.tsv (default: ON). Use --no-refresh-pkn to disable.",
    )
    ap.add_argument(
        "--no-refresh-pkn",
        dest="refresh_pkn",
        action="store_false",
        help="Disable PKN refresh.",
    )

    ap.add_argument("--signor-edges", default=None)
    ap.add_argument("--ptm-min-substrates", type=int, default=5)
    ap.add_argument("--ptm-n-perm", type=int, default=200)

    # Engine2 params
    ap.add_argument("--corr-method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--corr-flag-threshold", type=float, default=0.40)
    ap.add_argument("--min-markers", type=int, default=5)

    # Engine3 params
    ap.add_argument("--pkn-edges", default=None, help="Optional override PKN edges path.")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--top-tfs", type=int, default=30)
    ap.add_argument("--confound-penalty-threshold", type=float, default=0.40)

    return ap.parse_args()


def _config_to_args(cfg: Any) -> argparse.Namespace:
    """Build args-like object from CausalityPhaseConfig or dict."""
    diseases = getattr(cfg, "diseases", None) if hasattr(cfg, "diseases") else (cfg.get("diseases") if isinstance(cfg, dict) else None)
    run_all = getattr(cfg, "run_all", cfg.get("run_all", True)) if hasattr(cfg, "run_all") else cfg.get("run_all", True)
    use_all = run_all or not (diseases and len(diseases) > 0)
    return argparse.Namespace(
        out_root=getattr(cfg, "out_root", cfg.get("out_root", "")) if hasattr(cfg, "out_root") else cfg.get("out_root", ""),
        disease=diseases if (diseases and len(diseases) > 0 and not run_all) else None,
        all=use_all,
        strict=getattr(cfg, "strict", cfg.get("strict", False)),
        log_level=getattr(cfg, "log_level", cfg.get("log_level", "INFO")),
        engine23_script=getattr(cfg, "engine23_script", cfg.get("engine23_script")),
        no_engine0=getattr(cfg, "no_engine0", cfg.get("no_engine0", False)),
        no_engine1=getattr(cfg, "no_engine1", cfg.get("no_engine1", False)),
        no_engine2=getattr(cfg, "no_engine2", cfg.get("no_engine2", False)),
        no_engine3=getattr(cfg, "no_engine3", cfg.get("no_engine3", False)),
        no_omnipath_layer=getattr(cfg, "no_omnipath_layer", cfg.get("no_omnipath_layer", False)),
        force_engine1=not getattr(cfg, "no_force_engine1", cfg.get("no_force_engine1", False)),
        force_engine0=not getattr(cfg, "no_force_engine0", cfg.get("no_force_engine0", False)),
        refresh_omnipath_cache=not getattr(cfg, "no_refresh_omnipath_cache", cfg.get("no_refresh_omnipath_cache", False)),
        build_pkn=not getattr(cfg, "no_build_pkn", cfg.get("no_build_pkn", False)),
        refresh_pkn=not getattr(cfg, "no_refresh_pkn", cfg.get("no_refresh_pkn", False)),
        signor_edges=getattr(cfg, "signor_edges", cfg.get("signor_edges")),
        ptm_min_substrates=getattr(cfg, "ptm_min_substrates", cfg.get("ptm_min_substrates", 5)),
        ptm_n_perm=getattr(cfg, "ptm_n_perm", cfg.get("ptm_n_perm", 200)),
        engine1_method=getattr(cfg, "engine1_method", cfg.get("engine1_method", "mean")),
        engine1_min_size=getattr(cfg, "engine1_min_size", cfg.get("engine1_min_size", 10)),
        engine1_max_pathways=getattr(cfg, "engine1_max_pathways", cfg.get("engine1_max_pathways", 1500)),
        engine1_max_tfs=getattr(cfg, "engine1_max_tfs", cfg.get("engine1_max_tfs", 300)),
        corr_method=getattr(cfg, "corr_method", cfg.get("corr_method", "spearman")),
        corr_flag_threshold=getattr(cfg, "corr_flag_threshold", cfg.get("corr_flag_threshold", 0.40)),
        min_markers=getattr(cfg, "min_markers", cfg.get("min_markers", 5)),
        pkn_edges=getattr(cfg, "pkn_edges", cfg.get("pkn_edges")),
        max_steps=getattr(cfg, "max_steps", cfg.get("max_steps", 3)),
        top_tfs=getattr(cfg, "top_tfs", cfg.get("top_tfs", 30)),
        confound_penalty_threshold=getattr(cfg, "confound_penalty_threshold", cfg.get("confound_penalty_threshold", 0.40)),
    )


def run_causality_from_config(config: Any) -> None:
    """
    Run causality workflow from config (CausalityPhaseConfig or dict).
    No argparse, no sys.argv. Used by causality_service.
    """
    args = _config_to_args(config)
    _main_impl(args)


def main() -> None:
    args = parse_args()
    _main_impl(args)


def _main_impl(args: argparse.Namespace) -> None:
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    if args.all:
        diseases = _discover_diseases(out_root)
        if not diseases:
            raise DataError(f"No diseases discovered under {out_root}")
    else:
        diseases = args.disease or []
        if not diseases:
            raise ValidationError("Provide --disease DiseaseName (repeatable) OR use --all")

    # Import Engine2/3 mechanistic module
    this_dir = Path(__file__).resolve().parent
    engine23_path = (
        Path(args.engine23_script).expanduser().resolve()
        if args.engine23_script
        else (this_dir / "run_engine2_3_mechanistic.py")
    )
    engine23 = _import_engine2_3_module(engine23_path)

    pkn_override = Path(args.pkn_edges).expanduser().resolve() if args.pkn_edges else None
    signor_path = Path(args.signor_edges).expanduser().resolve() if args.signor_edges else None

    # Small sanity: refresh-pkn without build-pkn does nothing
    if bool(args.refresh_pkn) and (not bool(args.build_pkn)):
        log.warning("--refresh-pkn requested but --no-build-pkn is set; disabling refresh_pkn.")
        args.refresh_pkn = False

    summary_rows: List[Dict[str, str]] = []

    for d in diseases:
        log.info("=== Causality ALL workflow for disease: %s ===", d)

        e1_status = "disabled"
        if not args.no_engine1:
            try:
                e1_status, e1_dir = run_engine1_causal_features_minimal(
                    out_root=out_root,
                    disease=d,
                    method=str(args.engine1_method),
                    min_size=int(args.engine1_min_size),
                    max_pathways=int(args.engine1_max_pathways),
                    max_tfs=int(args.engine1_max_tfs),
                    force=bool(args.force_engine1),
                    strict=bool(args.strict),
                )
                log.info("Engine1 status=%s outputs=%s", e1_status, e1_dir)
            except Exception as e:
                log.exception("Engine1 failed for %s: %s", d, e)
                if args.strict:
                    raise
                e1_status = "failed"

        omnipath_status = "disabled"
        if not args.no_omnipath_layer:
            try:
                o = engine23.run_omnipath_layer_for_disease(
                    out_root=out_root,
                    disease=d,
                    refresh_cache=bool(args.refresh_omnipath_cache),
                    build_pkn=bool(args.build_pkn),
                    refresh_pkn=bool(args.refresh_pkn),
                    signor_edges_path=signor_path,
                    ptm_min_substrates=int(args.ptm_min_substrates),
                    ptm_n_perm=int(args.ptm_n_perm),
                    strict=bool(args.strict),
                )
                omnipath_status = str(getattr(o, "status", "ok"))
                log.info("OmniPath layer status=%s outputs=%s", omnipath_status, getattr(o, "out_dir", ""))
            except Exception as e:
                log.exception("OmniPath layer failed for %s: %s", d, e)
                if args.strict:
                    raise
                omnipath_status = "failed"

        e2_status = "disabled"
        if not args.no_engine2:
            try:
                e2 = engine23.run_engine2_confounding(
                    out_root=out_root,
                    disease=d,
                    corr_method=str(args.corr_method),
                    corr_flag_threshold=float(args.corr_flag_threshold),
                    min_markers=int(args.min_markers),
                    strict=bool(args.strict),
                )
                e2_status = str(getattr(e2, "status", "ok"))
                log.info("Engine2 status=%s outputs=%s", e2_status, getattr(e2, "out_dir", ""))
            except Exception as e:
                log.exception("Engine2 failed for %s: %s", d, e)
                if args.strict:
                    raise
                e2_status = "failed"

        e3_status = "disabled"
        if not args.no_engine3:
            try:
                e3 = engine23.run_engine3_contextualization(
                    out_root=out_root,
                    disease=d,
                    pkn_edges_override=pkn_override,
                    max_steps=int(args.max_steps),
                    top_tfs=int(args.top_tfs),
                    confound_penalty_threshold=float(args.confound_penalty_threshold),
                    strict=bool(args.strict),
                )
                e3_status = str(getattr(e3, "status", "ok"))
                log.info("Engine3 status=%s outputs=%s", e3_status, getattr(e3, "out_dir", ""))
            except Exception as e:
                log.exception("Engine3 failed for %s: %s", d, e)
                if args.strict:
                    raise
                e3_status = "failed"

        e0_status = "disabled"
        if not args.no_engine0:
            try:
                e0_status, e0_dir = run_engine0_evidence_bundle_minimal(
                    out_root=out_root,
                    disease=d,
                    force=bool(args.force_engine0),
                    strict=bool(args.strict),
                )
                log.info("Engine0 status=%s outputs=%s", e0_status, e0_dir)
            except Exception as e:
                log.exception("Engine0 failed for %s: %s", d, e)
                if args.strict:
                    raise
                e0_status = "failed"

        summary_rows.append(
            {
                "disease": d,
                "engine1_features": e1_status,
                "omnipath_layer": omnipath_status,
                "engine2_confounding": e2_status,
                "engine3_context": e3_status,
                "engine0_evidence": e0_status,
            }
        )

    summary_path = out_root / "engines" / "CAUSALITY_ALL_RUN_SUMMARY.tsv"
    _atomic_write_tsv(pd.DataFrame(summary_rows), summary_path, index=False)
    log.info("Wrote workflow summary: %s", summary_path)


if __name__ == "__main__":
    main()
