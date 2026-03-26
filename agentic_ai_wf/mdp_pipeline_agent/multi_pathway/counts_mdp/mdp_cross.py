# mdp_cross.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .mdp_logging import info, warn, trace
from .mdp_config import CONFIG

def compare_two(
    gsea_path_A: str,
    gsea_path_B: str,
    tagA: str,
    tagB: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not Path(gsea_path_A).exists() or not Path(gsea_path_B).exists():
            warn("[compare] Missing GSEA summaries; skipping.")
            return
        A = pd.read_csv(gsea_path_A, sep="\t")
        B = pd.read_csv(gsea_path_B, sep="\t")
        if "term" not in A.columns and "Term" in A.columns:
            A = A.rename(columns={"Term": "term"})
        if "term" not in B.columns and "Term" in B.columns:
            B = B.rename(columns={"Term": "term"})
        if "term" not in A.columns or "term" not in B.columns:
            warn("[compare] Missing 'term' column; skipping.")
            return
        fa = "FDR" if "FDR" in A.columns else ("qval" if "qval" in A.columns else None)
        fb = "FDR" if "FDR" in B.columns else ("qval" if "qval" in B.columns else None)
        na = "NES" if "NES" in A.columns else ("ES" if "ES" in A.columns else None)
        nb = "NES" if "NES" in B.columns else ("ES" if "ES" in B.columns else None)
        A2 = A["term"].to_frame()
        if fa:
            A2[fa] = A[fa]
        if na:
            A2[na] = A[na]
        B2 = B["term"].to_frame()
        if fb:
            B2[fb] = B[fb]
        if nb:
            B2[nb] = B[nb]
        M = A2.merge(B2, on="term", suffixes=(f"_{tagA}", f"_{tagB}"))
        if M.empty:
            warn("[compare] No overlapping terms.")
            return
        M.to_csv(out_dir / f"{tagA}_vs_{tagB}_overlap.tsv", sep="\t", index=False)
        if na and nb and (f"{na}_{tagA}" in M.columns) and (f"{nb}_{tagB}" in M.columns):
            try:
                x = pd.to_numeric(M[f"{na}_{tagA}"], errors="coerce")
                y = pd.to_numeric(M[f"{nb}_{tagB}"], errors="coerce")
                plt.figure(figsize=(6, 6))
                plt.scatter(x, y, s=12)
                plt.axhline(0, lw=1, ls="--")
                plt.axvline(0, lw=1, ls="--")
                plt.xlabel(f"{tagA} {na}")
                plt.ylabel(f"{tagB} {nb}")
                plt.title(f"NES scatter: {tagA} vs {tagB}")
                for _, row in M.head(20).iterrows():
                    if pd.notna(row[f"{na}_{tagA}"]) and pd.notna(row[f"{nb}_{tagB}"]):
                        plt.annotate(
                            row["term"][:20],
                            (row[f"{na}_{tagA}"], row[f"{nb}_{tagB}"]),
                            fontsize=6,
                        )
                plt.tight_layout()
                plt.savefig(out_dir / f"{tagA}_vs_{tagB}_NES_scatter.png", dpi=CONFIG["FIG_DPI"])
                plt.close()
            except Exception as e:
                warn(f"[compare] plotting failed: {trace(e)}")
    except Exception as e:
        warn(f"compare_two failed: {trace(e)}")

def _normalize_term_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    if CONFIG.get("NORMALIZE_TERMS_DROP_PREFIX", True):
        return s.split(":", 1)[-1].strip()
    return s.strip()

def multi_cohort_union_and_shared(
    results: List[dict],
    out_dir: Path,
    require_significant: bool = False,
    fdr_threshold: float = CONFIG.get("FDR_Q", 0.05),
    fdr_col_preference: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        long_rows: List[pd.DataFrame] = []
        for r in results:
            tag = r.get("name", "cohort")
            p = Path(r.get("gsea_summary", ""))
            if not p.exists():
                warn(f"[multi] Missing GSEA/ORA table for {tag}: {p}")
                continue
            df = pd.read_csv(p, sep="\t")
            if "term" not in df.columns and "Term" in df.columns:
                df = df.rename(columns={"Term": "term"})
            if "term" not in df.columns:
                warn(f"[multi] No 'term' column in {p}; skipping.")
                continue
            df["term"] = df["term"].astype(str).map(_normalize_term_name)
            eff_col = "NES" if "NES" in df.columns else ("ES" if "ES" in df.columns else None)
            candidates = [
                "FDR", "qval", "padj", "Adjusted_P-value", "adj_p",
                "p.adjust", "p.adj", "q-value", "qvalue", "FDR_q", "pvalue_adj",
            ]
            if fdr_col_preference and fdr_col_preference in df.columns:
                candidates = [fdr_col_preference] + candidates
            fdr_col = next((c for c in candidates if c in df.columns), None)
            if eff_col:
                df[eff_col] = pd.to_numeric(df[eff_col], errors="coerce")
            if fdr_col:
                df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
            keep = ["term"] + ([eff_col] if eff_col else []) + ([fdr_col] if fdr_col else [])
            sub = df[keep].drop_duplicates(subset=["term"]).copy()
            sub["cohort"] = tag
            sub["__eff_col__"] = eff_col or ""
            sub["__fdr_col__"] = fdr_col or ""
            if require_significant and fdr_col:
                sub = sub[sub[fdr_col] <= float(fdr_threshold)].copy()
            long_rows.append(sub)
        if not long_rows:
            warn("[multi] No cohort summaries to combine.")
            return
        L = pd.concat(long_rows, ignore_index=True)
        cohorts = sorted(L["cohort"].unique().tolist())
        wide_eff = None
        for tag in cohorts:
            sub = L[L["cohort"] == tag].copy()
            ec = sub["__eff_col__"].iloc[0]
            if ec:
                w = sub[["term", ec]].rename(columns={ec: f"NES_or_ES__{tag}"}).drop_duplicates("term")
            else:
                w = sub[["term"]].assign(**{f"NES_or_ES__{tag}": np.nan}).drop_duplicates("term")
            wide_eff = w if wide_eff is None else wide_eff.merge(w, on="term", how="outer")
        wide_fdr = None
        for tag in cohorts:
            sub = L[L["cohort"] == tag].copy()
            fc = sub["__fdr_col__"].iloc[0]
            if fc:
                w = sub[["term", fc]].rename(columns={fc: f"FDR__{tag}"}).drop_duplicates("term")
            else:
                w = sub[["term"]].assign(**{f"FDR__{tag}": np.nan}).drop_duplicates("term")
            wide_fdr = w if wide_fdr is None else wide_fdr.merge(w, on="term", how="outer")
        W = wide_eff.merge(wide_fdr, on="term", how="outer")
        for tag in cohorts:
            has_any = W[[c for c in W.columns if c in (f"NES_or_ES__{tag}", f"FDR__{tag}")]].notna().any(axis=1)
            W[f"present__{tag}"] = has_any.astype(int)
        pres_cols = [c for c in W.columns if c.startswith("present__")]
        W["present_in"] = W[pres_cols].sum(axis=1)
        W["n_cohorts"] = len(cohorts)
        eff_cols = [c for c in W.columns if c.startswith("NES_or_ES__")]
        if eff_cols:
            W["mean_effect"] = W[eff_cols].mean(axis=1, skipna=True)
        W["shared_across_all"] = (W["present_in"] == W["n_cohorts"]).astype(bool)
        W["category"] = np.where(W["shared_across_all"], "shared_all", "nonshared")
        order_cols = [
            "term", "category", "shared_across_all", "present_in",
            "n_cohorts", "mean_effect",
        ]
        order_cols += sorted([c for c in W.columns if c.startswith("NES_or_ES__")])
        order_cols += sorted([c for c in W.columns if c.startswith("FDR__")])
        order_cols += sorted([c for c in W.columns if c.startswith("present__")])
        order_cols = [c for c in order_cols if c in W.columns]
        W = W[order_cols]
        combined_path = out_dir / "ALL_cohorts_combined_pathways.tsv"
        W.sort_values(
            ["category", "present_in", "mean_effect"],
            ascending=[True, False, False],
        ).to_csv(combined_path, sep="\t", index=False)
        info(f"[multi] wrote: {combined_path}")
    except Exception as e:
        warn(f"multi_cohort_union_and_shared failed: {trace(e)}")
