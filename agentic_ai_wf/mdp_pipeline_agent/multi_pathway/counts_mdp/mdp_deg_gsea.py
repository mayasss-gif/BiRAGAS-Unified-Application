# mdp_deg_gsea.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from .mdp_logging import warn, trace
from .mdp_config import CONFIG

def _resolve_deg_columns(
    df: pd.DataFrame,
    gene_hint: str | None,
    lfc_hint: str | None,
    q_hint: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    gene = None
    for k in [
        gene_hint,
        "gene", "genes", "symbol", "hgnc_symbol",
        "gene_symbol", "ensembl", "ensembl_id",
    ]:
        if k and isinstance(k, str) and k.lower() in cols:
            gene = cols[k.lower()]
            break
    if gene is None:
        best, best_score = None, -1
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
                s = df[c].astype(str)
                n_unique = s.nunique(dropna=True)
                tok = s.head(500).str.replace(r"[^A-Za-z]", "", regex=True)
                frac_uc = (tok.str.match(r"^[A-Z]+$")).mean()
                is_bad = c.lower() in {"sample", "sampleid", "id", "patient", "subject", "barcode"}
                score = (n_unique > 100) * 2 + (frac_uc > 0.6) * 1 - (is_bad * 3)
                if score > best_score:
                    best, best_score = c, score
        gene = best if best is not None else list(df.columns)[0]
    lfc = None
    for k in [lfc_hint, "log2fc", "log2foldchange", "logfc", "lfc", "effect", "stat", "t", "score"]:
        if k and isinstance(k, str) and k.lower() in cols:
            lfc = cols[k.lower()]
            break
    if lfc is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        lfc = (
            pd.Series(
                {c: pd.to_numeric(df[c], errors="coerce").dropna().std() for c in num_cols}
            ).idxmax()
            if num_cols
            else (list(df.columns)[1] if len(df.columns) > 1 else gene)
        )
    q = None
    for k in [
        q_hint,
        "fdr", "padj", "adj_p", "adj.p", "qval",
        "qvalue", "fdr_bh", "bh_fdr", "adj_p_value",
    ]:
        if k and isinstance(k, str) and k.lower() in cols:
            q = cols[k.lower()]
            break
    return gene, lfc, q

def build_rank_from_degs(
    df: pd.DataFrame,
    gene_col_hint: str,
    lfc_col_hint: str,
    q_col_hint: Optional[str],
    q_max: Optional[float],
) -> pd.Series:
    d = df.copy()
    try:
        for c in d.columns:
            if d[c].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
                d[c] = d[c].astype(str)
        gcol, lfc_col, qcol = _resolve_deg_columns(d, gene_col_hint, lfc_col_hint, q_col_hint or None)
        if gcol not in d.columns or lfc_col not in d.columns:
            raise ValueError(f"DEGs must have {gcol} and {lfc_col}")
        d[gcol] = d[gcol].astype(str).str.upper().str.strip()
        d[lfc_col] = pd.to_numeric(d[lfc_col], errors="coerce")
        d = d.dropna(subset=[gcol, lfc_col])
        if q_max is not None and qcol and qcol in d.columns:
            d[qcol] = pd.to_numeric(d[qcol], errors="coerce")
            d = d[d[qcol] <= q_max].copy()
        d = d[[gcol, lfc_col]].drop_duplicates(subset=[gcol])
        d = d[~d[gcol].str.startswith("UNNAMED")]
        rnk = d.set_index(gcol)[lfc_col]
        return rnk
    except Exception as e:
        warn(f"build_rank_from_degs failed: {trace(e)}")
        return pd.Series(dtype=float)

def split_up_down_from_degs(
    df: pd.DataFrame,
    gene_col_hint: str,
    lfc_col_hint: str,
    q_col_hint: Optional[str],
    q_max: Optional[float],
) -> Tuple[List[str], List[str], str, str]:
    try:
        d = df.copy()
        for c in d.columns:
            if d[c].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
                d[c] = d[c].astype(str)
        gcol, lfc_col, qcol = _resolve_deg_columns(d, gene_col_hint, lfc_col_hint, q_col_hint or None)
        d[gcol] = d[gcol].astype(str).str.upper().str.strip()
        d[lfc_col] = pd.to_numeric(d[lfc_col], errors="coerce")
        if q_max is not None and qcol and qcol in d.columns:
            d[qcol] = pd.to_numeric(d[qcol], errors="coerce")
            d = d[d[qcol] <= q_max].copy()
        d = d.dropna(subset=[gcol, lfc_col]).drop_duplicates(subset=[gcol])
        up = d.loc[d[lfc_col] > 0, gcol].tolist()
        down = d.loc[d[lfc_col] < 0, gcol].tolist()
        return up, down, gcol, lfc_col
    except Exception as e:
        warn(f"split_up_down_from_degs failed: {trace(e)}")
        return [], [], gene_col_hint or "Gene", lfc_col_hint or "log2FC"

def write_delta_vs_consensus(
    gsea_path: Path,
    expectations_tsv: Path,
    out_path: Path,
) -> Optional[Path]:
    try:
        base = pd.read_csv(expectations_tsv, sep="\t")
        gsea = pd.read_csv(gsea_path, sep="\t")
    except Exception:
        return None
    try:
        if "term" not in gsea.columns and "Term" in gsea.columns:
            gsea = gsea.rename(columns={"Term": "term"})
        metric = "NES" if "NES" in gsea.columns else ("ES" if "ES" in gsea.columns else None)
        if metric is None:
            return None
        joined = gsea.merge(base, left_on="term", right_on="pathway_id", how="inner")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if joined.empty:
            pd.DataFrame(
                columns=[
                    "term", "pathway_id", metric,
                    "zmean", "expected_weight",
                    f"{metric}_minus_zmean", f"{metric}_minus_expected",
                ]
            ).to_csv(out_path, sep="\t", index=False)
            return out_path
        val = pd.to_numeric(joined[metric], errors="coerce")
        joined[f"{metric}_minus_zmean"] = val - pd.to_numeric(joined.get("zmean"), errors="coerce")
        joined[f"{metric}_minus_expected"] = val - pd.to_numeric(joined.get("expected_weight"), errors="coerce")
        keep = [
            "term", "pathway_id", metric,
            "zmean", "expected_weight",
            f"{metric}_minus_zmean", f"{metric}_minus_expected",
        ]
        joined[keep].to_csv(out_path, sep="\t", index=False)
        return out_path
    except Exception as e:
        warn(f"write_delta_vs_consensus failed: {trace(e)}")
        return None
