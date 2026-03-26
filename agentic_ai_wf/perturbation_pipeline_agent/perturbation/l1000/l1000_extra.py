# src/l1000_extra.py
"""
Extra L1000 analyses that reuse an existing run:
- N1: comparison of causal readouts (CIS, CIS_causal, ATE, etc.)
- N2: refined causal analysis (CIS + control-aware ATE)
- N3: best results exports (overall / per-gene / strict / unique)
- N4: single-gene readouts + pathway-level CIS + optional models

This script:
- DOES NOT rerun your main pipeline
- ONLY reads existing outputs under the latest runs/L1000_Run_* directory
"""

from __future__ import annotations
import os, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Basic paths: assume this file lives in src/, project root above it
# -------------------------------------------------------------------


# Core L1000 input files (names from your ls InputFiles)
from .constants import CORE_SIG_PATH, CORE_PERT_PATH, CORE_CELL_PATH, CORE_GENE_PATH, CORE_GCTX_PATH


# -------------------------------------------------------------------
# Small logging helper
# -------------------------------------------------------------------
from datetime import datetime

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | INFO | L1000-EXTRA | {msg}")

# -------------------------------------------------------------------
# Simple utils used by several blocks
# -------------------------------------------------------------------
def ensure_exists(*paths: Path) -> None:
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file: {p}")

def normalize_unit(u):
    if pd.isna(u): return u
    s = str(u)
    return (s.replace("µ", "u")
             .replace("μ", "u")
             .strip())

def _to_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)

def _norm01(s: pd.Series) -> pd.Series:
    s = _to_float(s)
    mn, mx = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn + 1e-12)

# -------------------------------------------------------------------
# Hill model + fit (same logic as in your main pipeline)
# -------------------------------------------------------------------
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

def hill(dose, ec50, h, emax, b):
    return b + (emax - b) * (dose**h / (ec50**h + dose**h))

def fit_hill(dose_arr, resp_arr):
    x, y = np.asarray(dose_arr, float), np.asarray(resp_arr, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if np.unique(x).size < 4 or np.isfinite(y).sum() < 4:
        return None
    p0 = [np.nanmedian(x), 1.0, np.nanmax(y), np.nanmin(y)]
    bounds = ([1e-12, 0.1, -np.inf, -np.inf], [np.inf, 10.0, np.inf, np.inf])
    try:
        popt, _ = curve_fit(hill, x, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = hill(x, *popt); r2 = r2_score(y, pred)
        ec50, hcoef, emax, base = popt
        si = abs(emax - base) / (ec50 + 1e-8)
        return dict(EC50=ec50, Hill=hcoef, R2=r2, Emax=emax, Baseline=base, SensitivityIndex=si)
    except Exception:
        return None


# -------------------------------------------------------------------
#  N1: Comparison of causal experiments (CIS / ATE / etc.)
# -------------------------------------------------------------------
def run_N1_comparison(RUN_DIR: Path, FIG_DIR: Path):
    from scipy.stats import spearmanr
    CMP_DIR = RUN_DIR / "comparison"
    CMP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    PREFIX = "N1_"

    def _prefixed(path: Path) -> Path:
        return path.with_name(f"{PREFIX}{path.name}")

    raw_plus_base = RUN_DIR / "causal_link_table_plus_ATE.csv"
    raw_core      = RUN_DIR / "causal_link_table_with_relevance.csv"
    ate_tbl       = RUN_DIR / "control_ate_table.csv"

    # choose which table to start from
    if (RUN_DIR / "N2_causal_link_table_plus_ATE.csv").exists():
        # If N2 already ran once, prefer that as canonical
        df = pd.read_csv(RUN_DIR / "N2_causal_link_table_plus_ATE.csv")
        log("N1: starting from N2_causal_link_table_plus_ATE.csv")
    else:
        if raw_plus_base.exists():
            log("N1: loading causal_link_table_plus_ATE.csv")
            df = pd.read_csv(raw_plus_base)
        elif raw_core.exists():
            log("N1: loading causal_link_table_with_relevance.csv")
            df = pd.read_csv(raw_core)
            if ate_tbl.exists():
                log("N1: merging ATE from control_ate_table.csv")
                ate = pd.read_csv(ate_tbl)
                key_cols = [c for c in ["Gene","Drug","DoseUnit"] if c in df.columns and c in ate.columns]
                df = df.merge(ate, on=key_cols, how="left")
            df.to_csv(raw_plus_base, index=False)
            log(f"N1: wrote merged {raw_plus_base.name} ({len(df)})")
        else:
            log("N1: no causal outputs found, skipping N1.")
            return

    # sanitize numeric
    numeric_candidates = [
        "CIS","CIS_causal","Sensitivity Index","SI_clamped","R²","ATE","ATE_p",
        "EC50 (µM)","Emax","Baseline"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = _to_float(df[c])

    if {"Emax","Baseline"} <= set(df.columns):
        df["amp_abs"] = (_to_float(df["Emax"]) - _to_float(df["Baseline"])).abs()
    if "Sensitivity Index" in df.columns and "SI_clamped" not in df.columns:
        df["SI_clamped"] = _to_float(df["Sensitivity Index"])
    if "ATE" in df.columns:
        df["abs_ATE"] = _to_float(df["ATE"]).abs()
    if "ATE_p" in df.columns:
        df["p_weight"] = (1.0 - np.minimum(1.0, _to_float(df["ATE_p"]) / 0.05)).clip(lower=0, upper=1)

    # recompute CIS_causal if missing but ATE present
    if "CIS_causal" not in df.columns and {"ATE","Gene"} <= set(df.columns):
        log("N1: recomputing CIS_causal from ATE.")
        def robust_norm(vals, genes):
            out = pd.Series(index=vals.index, dtype=float)
            for g, idx in genes.groupby(genes).groups.items():
                v = _to_float(vals.loc[idx])
                if v.notna().sum() == 0:
                    out.loc[idx] = 0.0; continue
                lo, hi = np.nanpercentile(v, [5, 95])
                if not np.isfinite(hi - lo) or (hi - lo) < 1e-9:
                    rnk = v.rank(method="average", na_option="keep")
                    out.loc[idx] = (rnk - rnk.min())/(rnk.max()-rnk.min()+1e-12)
                else:
                    out.loc[idx] = np.clip((v - lo)/(hi - lo + 1e-12), 0, 1)
            return out
        abs_ate = df["ATE"].abs()
        p = _to_float(df["ATE_p"]) if "ATE_p" in df.columns else pd.Series(np.nan, index=df.index)
        p_weight = (1.0 - np.minimum(1.0, p/0.05)).fillna(0.0)
        df["CIS_causal"] = 0.7 * robust_norm(abs_ate, df["Gene"]) + 0.3 * p_weight

    # recompute CIS if missing but required columns present
    if "CIS" not in df.columns and {"Emax","Baseline","R²","Sensitivity Index","Gene"} <= set(df.columns):
        log("N1: recomputing CIS from Emax, Baseline, R², and Sensitivity Index.")
        def _norm_by_gene(df_local, col):
            def _mm(x):
                x = pd.to_numeric(x, errors="coerce")
                rng = x.max() - x.min()
                if not np.isfinite(rng) or rng == 0:
                    return pd.Series(np.zeros(len(x)), index=x.index)
                return (x - x.min()) / (rng + 1e-12)
            return df_local.groupby("Gene")[col].transform(_mm)
        
        emax = _to_float(df["Emax"])
        base = _to_float(df["Baseline"])
        r2v = _to_float(df["R²"])
        sival = _to_float(df["Sensitivity Index"])
        
        df["amp_abs"] = (emax - base).abs()
        df["_R2"] = r2v
        df["_SI"] = sival
        df["n_amp"] = _norm_by_gene(df, "amp_abs")
        df["n_r2"] = _norm_by_gene(df, "_R2")
        df["n_si"] = _norm_by_gene(df, "_SI")
        df["CIS"] = 0.4 * df["n_amp"] + 0.2 * df["n_r2"] + 0.4 * df["n_si"]
        # Clean up temporary columns
        df = df.drop(columns=["amp_abs", "n_amp", "n_r2", "n_si", "_R2", "_SI"], errors="ignore")

    # Save canonical comparison input
    df.to_csv(_prefixed(RUN_DIR / "causal_link_table_plus_ATE.csv"), index=False)
    # Also save non-prefixed version for N4 and other downstream steps
    df.to_csv(RUN_DIR / "causal_link_table_plus_ATE.csv", index=False)

    key_cols = [c for c in ["Gene","Drug","DoseUnit"] if c in df.columns]
    present_cols = [c for c in ["CIS","CIS_causal","abs_ATE","SI_clamped","R²","amp_abs"] if c in df.columns]
    cmp = df.dropna(subset=present_cols, how="all").copy()
    cmp = cmp[key_cols + present_cols].copy()
    cmp.to_csv(_prefixed(CMP_DIR / "comparison_raw_matrix.csv"), index=False)

    # ranks and correlations
    rank_cols = {
        f"rank_{m}": _to_float(cmp[m]).rank(method="average", ascending=False)
        for m in present_cols
    }
    rank_df = cmp.join(pd.DataFrame(rank_cols))
    rank_df.to_csv(_prefixed(CMP_DIR / "comparison_ranks.csv"), index=False)

    metrics = present_cols
    rho_mat = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    p_mat   = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for mi in metrics:
        for mj in metrics:
            xi = _to_float(rank_df[f"rank_{mi}"])
            xj = _to_float(rank_df[f"rank_{mj}"])
            ok = xi.notna() & xj.notna()
            rho, pval = spearmanr(xi[ok], xj[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
            rho_mat.loc[mi, mj] = rho; p_mat.loc[mi, mj] = pval
    rho_mat.to_csv(_prefixed(CMP_DIR / "rank_spearman_rho.csv"))
    p_mat.to_csv(_prefixed(CMP_DIR / "rank_spearman_p.csv"))

    # per-gene concordance
    per_gene = []
    for g, sub in rank_df.groupby("Gene"):
        row = {"Gene": g, "n_pairs": len(sub)}
        for (a, b) in [("CIS","CIS_causal"),("CIS","abs_ATE"),("CIS_causal","abs_ATE")]:
            if f"rank_{a}" in sub and f"rank_{b}" in sub and a in present_cols and b in present_cols:
                x, y = sub[f"rank_{a}"], sub[f"rank_{b}"]
                ok = x.notna() & y.notna()
                rho, pv = spearmanr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
                row[f"rho_{a}_{b}"], row[f"p_{a}_{b}"] = rho, pv
        per_gene.append(row)
    per_gene_cmp = pd.DataFrame(per_gene)
    per_gene_cmp.to_csv(_prefixed(CMP_DIR / "per_gene_concordance.csv"), index=False)

    # top-K overlaps
    def topk_ids(df_in, metric, k=50):
        d = df_in.dropna(subset=[metric]).sort_values(metric, ascending=False)
        return set(map(tuple, d[key_cols].head(k).values.tolist()))

    K = 50
    overlaps = {}
    candidates = [m for m in ["CIS","CIS_causal","abs_ATE","SI_clamped"] if m in cmp.columns]
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if j <= i: continue
            A, B = topk_ids(cmp, a, K), topk_ids(cmp, b, K)
            inter, union = len(A & B), len(A | B)
            overlaps[f"{a}__vs__{b}"] = {
                "K": K,
                "overlap": inter,
                "jaccard": (inter/union if union else np.nan)
            }
    with open(_prefixed(CMP_DIR / "topK_overlaps.json"), "w") as f:
        json.dump(overlaps, f, indent=2)

    # simple rho heatmap if seaborn available
    try:
        import seaborn as sns
        if rho_mat.size:
            H = rho_mat.astype(float)
            H.index = H.index.map(str); H.columns = H.columns.map(str)
            H = H.loc[sorted(set(H.index) & set(H.columns))]
            plt.figure(figsize=(7,6))
            sns.heatmap(H, vmin=-1, vmax=1, cmap="coolwarm",
                        annot=True, fmt=".2f",
                        cbar_kws={"label": "Spearman ρ"},
                        square=True, linewidths=0.5, linecolor="white")
            plt.title("Rank Correlation between Causal Readouts")
            plt.tight_layout()
            plt.savefig(_prefixed(CMP_DIR / "rank_correlation_heatmap.png"), dpi=150)
            plt.close()
    except Exception:
        pass

    # ATE volcano (if present)
    if {"ATE","ATE_p"} <= set(df.columns):
        x, p = _to_float(df["ATE"]), _to_float(df["ATE_p"])
        ok = x.notna() & p.notna() & (p > 0)
        if ok.sum() >= 5:
            plt.figure(figsize=(7,5))
            plt.scatter(x[ok], -np.log10(p[ok]), s=15, alpha=0.6)
            plt.axhline(-np.log10(0.05), color="k", linestyle="--")
            plt.xlabel("ATE"); plt.ylabel("-log10(p)"); plt.title("ATE Volcano")
            plt.tight_layout()
            plt.savefig(_prefixed(CMP_DIR / "volcano_ATE.png"), dpi=150)
            plt.close()

    # guardrail + global CIS variant
    table = df.copy()
    if "amp_abs" not in table and {"Emax","Baseline"} <= set(table.columns):
        table["amp_abs"] = (_to_float(table["Emax"]) - _to_float(table["Baseline"])).abs()
    if "SI_clamped" not in table and "Sensitivity Index" in table.columns:
        table["SI_clamped"] = _to_float(table["Sensitivity Index"])

    flip_flag = False
    if "CIS" in table and "amp_abs" in table:
        CIS_raw = _to_float(table["CIS"])
        rho_guard, _ = spearmanr(CIS_raw, table["amp_abs"], nan_policy="omit")
        CIS_norm = _norm01(CIS_raw)
        table["CIS_flipped"] = 1 - CIS_norm if np.isfinite(rho_guard) and rho_guard < 0 else CIS_norm
        flip_flag = np.isfinite(rho_guard) and rho_guard < 0

    if {"amp_abs","R²","SI_clamped"} <= set(table.columns):
        def _z01(s):
            s = _to_float(s)
            mu, sd = np.nanmean(s), np.nanstd(s)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.zeros(len(s)), index=s.index)
            z = (s - mu) / (sd + 1e-12)
            return 0.5 * (1 + pd.Series(z).apply(lambda t: math.erf(t / math.sqrt(2))))
        table["CIS_global"] = (
            0.5 * _z01(table["amp_abs"]) +
            0.2 * _z01(table["R²"]) +
            0.3 * _z01(table["SI_clamped"])
        )

    table.to_csv(_prefixed(RUN_DIR / "causal_comparison_metrics.csv"), index=False)

    try:
        import seaborn as sns
        cols = ["CIS_flipped","CIS_global","CIS_causal","abs_ATE","SI_clamped","R²","amp_abs"]
        cols = [c for c in cols if c in table.columns]
        if len(cols) >= 2:
            M = pd.DataFrame(index=cols, columns=cols, dtype=float)
            for a in cols:
                for b in cols:
                    ra, rb = _to_float(table[a]), _to_float(table[b])
                    ok = ra.notna() & rb.notna()
                    M.loc[a, b] = spearmanr(ra[ok], rb[ok])[0] if ok.sum() >= 3 else np.nan
            H = M.astype(float)
            H.index = H.index.map(str); H.columns = H.columns.map(str)
            plt.figure(figsize=(7,6))
            sns.heatmap(H, vmin=-1, vmax=1, cmap="coolwarm",
                        annot=True, fmt=".2f",
                        cbar_kws={"label": "Spearman ρ"},
                        square=True, linewidths=0.5, linecolor="white")
            plt.title(f"Causal Readouts Correlation{' (flipped)' if flip_flag else ''}")
            plt.tight_layout()
            plt.savefig(_prefixed(FIG_DIR / "causal_readouts_correlation.png"), dpi=150)
            plt.close()
    except Exception:
        pass

    with open(_prefixed(CMP_DIR / "_README.txt"), "w") as f:
        f.write(
            "Comparison outputs (prefixed with N1_):\n"
            "- N1_comparison_raw_matrix.csv\n"
            "- N1_comparison_ranks.csv\n"
            "- N1_rank_spearman_rho.csv / N1_rank_spearman_p.csv\n"
            "- N1_per_gene_concordance.csv\n"
            "- N1_topK_overlaps.json\n"
            "- N1_rank_correlation_heatmap.png / N1_volcano_ATE.png\n"
            "- N1_causal_comparison_metrics.csv\n"
            "- N1_causal_readouts_correlation.png\n"
            "- N1_causal_link_table_plus_ATE.csv (canonical merged table)\n"
        )

    log("N1: comparison complete.")

# -------------------------------------------------------------------
#  N2 / N3 / N4: we piggy-back on existing causal_link_table_plus_ATE
#  For sanity / simplicity we don't re-run big v2 Hill+ATE refits here;
#  we assume your main pipeline already made causal_link_table_plus_ATE.csv.
#  We just do best-results and single-gene/pathway summaries.
# -------------------------------------------------------------------

def run_N3_best_results(RUN_DIR: Path):
    PREFIX = "N3_"
    def _prefixed(path: Path) -> Path:
        return path.with_name(f"{PREFIX}{path.name}")

    TOP_K_OVERALL   = 200
    TOP_K_PER_GENE  = 3
    MIN_R2          = 0.80
    ATE_P_THRESH    = 0.05
    REQUIRE_QC_PASS = False
    PASS_ON_ERROR   = True

    try:
        in_path = RUN_DIR / "causal_link_table_plus_ATE.csv"
        if not in_path.exists():
            msg = f"Missing: {in_path}. Run your main drug pipeline first."
            if PASS_ON_ERROR:
                log(f"N3: SKIPPED (no causal_link_table_plus_ATE.csv) – {msg}")
                return
            else:
                raise FileNotFoundError(msg)

        df = pd.read_csv(in_path)

        num_cols = ["CIS","CIS_causal","ATE","ATE_p","R²","EC50 (µM)","Emax",
                    "Baseline","Sensitivity Index","SI_clamped","amp_abs"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        # p-weight
        if "ATE_p" in df.columns:
            p_weight = (1.0 - np.minimum(1.0, df["ATE_p"]/ATE_P_THRESH)).clip(lower=0, upper=1)
        else:
            p_weight = pd.Series(0.0, index=df.index)

        cis_causal_n = _norm01(df["CIS_causal"]) if "CIS_causal" in df.columns else 0.0
        cis_n        = _norm01(df["CIS"])        if "CIS"        in df.columns else 0.0
        best_score   = 0.6*cis_causal_n + 0.3*cis_n + 0.1*p_weight
        df["BEST_SCORE"] = best_score

        # QC_pass if missing (simple)
        if "QC_pass" not in df.columns:
            df["QC_pass"] = df.get("R²", 0.0).astype(float) >= MIN_R2

        rank_src = df[df["QC_pass"]==True].copy() if REQUIRE_QC_PASS else df.copy()

        def _sort_key(dfin):
            cols = [("BEST_SCORE", False)]
            if "CIS_causal" in dfin.columns: cols.append(("CIS_causal", False))
            if "CIS" in dfin.columns:        cols.append(("CIS", False))
            if isinstance(p_weight, pd.Series):
                dfin = dfin.assign(_pw=p_weight)
                cols.append(("_pw", False))
            by  = [c for c,_ in cols]
            asc = [a for _,a in cols]
            return dfin.sort_values(by=by, ascending=asc, kind="mergesort")

        overall_sorted = _sort_key(rank_src)
        best_overall = overall_sorted.head(TOP_K_OVERALL).copy()
        best_overall_path = _prefixed(RUN_DIR / "best_overall_topK.csv")
        best_overall.to_csv(best_overall_path, index=False)

        per_gene = (_sort_key(rank_src)
                    .groupby("Gene", as_index=False)
                    .head(TOP_K_PER_GENE))
        best_per_gene_path = _prefixed(RUN_DIR / "best_per_gene_topK.csv")
        per_gene.to_csv(best_per_gene_path, index=False)

        strict = rank_src.copy()
        if "ATE_p" in strict.columns:
            strict = strict[(pd.to_numeric(strict["ATE_p"], errors="coerce") <= ATE_P_THRESH)]
        strict = strict[(strict["QC_pass"]==True)]
        strict = _sort_key(strict)
        strict_path = _prefixed(RUN_DIR / "best_significant_QC.csv")
        strict.to_csv(strict_path, index=False)

        best_unique = (_sort_key(rank_src)
                       .drop_duplicates(subset=["Gene"], keep="first")
                       .reset_index(drop=True))
        best_unique_path = _prefixed(RUN_DIR / "best_unique_gene.csv")
        best_unique.to_csv(best_unique_path, index=False)

        manifest = {
            "inputs": str(in_path),
            "params": {
                "TOP_K_OVERALL": TOP_K_OVERALL,
                "TOP_K_PER_GENE": TOP_K_PER_GENE,
                "MIN_R2": MIN_R2,
                "ATE_P_THRESH": ATE_P_THRESH,
                "REQUIRE_QC_PASS": REQUIRE_QC_PASS
            },
            "outputs": {
                "N3_best_overall_topK.csv": str(best_overall_path),
                "N3_best_per_gene_topK.csv": str(best_per_gene_path),
                "N3_best_significant_QC.csv": str(strict_path),
                "N3_best_unique_gene.csv": str(best_unique_path)
            }
        }
        with open(_prefixed(RUN_DIR / "best_results_manifest.json"),"w") as f:
            json.dump(manifest, f, indent=2)

        log("N3: best-results CSVs written.")
    except Exception as e:
        if PASS_ON_ERROR:
            log(f"N3: FAILED but continuing – {repr(e)}")
        else:
            raise

# -------------------------------------------------------------------
# N4: simpler single-gene readouts & pathway-level CIS
# (we’ll only use existing causal_link_table_plus_ATE.csv and pathway map)
# -------------------------------------------------------------------
def run_N4_single_gene_and_pathways(RUN_DIR: Path, FIG_DIR: Path):
    PREFIX = "N4_"
    def _prefixed(path: Path) -> Path:
        return path.with_name(f"{PREFIX}{path.name}")

    in_path = RUN_DIR / "causal_link_table_plus_ATE.csv"
    if not in_path.exists():
        log("N4: skipping (no causal_link_table_plus_ATE.csv).")
        return
    df = pd.read_csv(in_path)

    # --- single-gene readouts (just tidy subset) ---
    cols_single = [
        "Gene","Drug","DoseUnit",
        "EC50 (µM)","R²","Hill Slope",
        "Baseline","Emax",
        "Sensitivity Index","CIS","CIS_causal",
        "Therapeutic_Relevance","Gene_Pathway","Gene_Main_Class",
        "Gene_Sub_Class","Gene_Regulation","Gene_DEG_Trend"
    ]
    cols_single = [c for c in cols_single if c in df.columns]
    sg = df[cols_single].copy()
    sg_path = _prefixed(RUN_DIR / "single_gene_readouts.csv")
    sg.to_csv(sg_path, index=False)
    log(f"N4: single-gene readouts → {sg_path.name} (rows={len(sg)})")

    # --- simple pathway-level CIS: mean CIS per (Pathway, Drug, DoseUnit) ---
    if "Gene_Pathway" not in df.columns or "CIS" not in df.columns:
        log("N4: pathway-level CIS skipped (need Gene_Pathway and CIS columns).")
        return
    pw = df.dropna(subset=["Gene_Pathway","CIS"]).copy()
    pw_agg = (pw.groupby(["Gene_Pathway","Drug","DoseUnit"])
                .agg(
                    n_genes=("Gene", "nunique"),
                    CIS_mean=("CIS","mean"),
                    CIS_median=("CIS","median"),
                    CIS_max=("CIS","max")
                ).reset_index()
             )
    pw_agg = pw_agg.rename(columns={"Gene_Pathway":"Pathway"})
    pw_path = _prefixed(RUN_DIR / "pathway_level_CIS.csv")
    pw_agg.to_csv(pw_path, index=False)
    log(f"N4: pathway-level CIS → {pw_path.name} (rows={len(pw_agg)})")

# -------------------------------------------------------------------
# MAIN entrypoint
# -------------------------------------------------------------------
def run_l1000_extra(output_dir: Path):
    """
    Main function you call from run_l1000_extra.py.
    It will:
      - auto-detect the latest run under runs/L1000_Run_*
      - run N1 (comparison), N3 (best results), N4 (single-gene / pathway)
    """
    RUN_DIR = output_dir 
    FIG_DIR = output_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log("=== EXTRA ANALYSES: N1 (comparison) ===")
    run_N1_comparison(RUN_DIR, FIG_DIR)

    log("=== EXTRA ANALYSES: N3 (best results) ===")
    run_N3_best_results(RUN_DIR)

    log("=== EXTRA ANALYSES: N4 (single-gene & pathway CIS) ===")
    run_N4_single_gene_and_pathways(RUN_DIR, FIG_DIR)

    log("✅ All extra analyses complete.")

# if __name__ == "__main__":
#     run_l1000_extra()

