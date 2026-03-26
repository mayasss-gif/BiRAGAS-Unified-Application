# src/depmap_codependency_uif.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, kendalltau
from statsmodels.stats.multitest import multipletests
import plotly.express as px


def _ensure_dirs(outdir: Path) -> None:
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "html").mkdir(parents=True, exist_ok=True)


def _build_effect_matrix(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Build ModelID × Gene matrix of ChronosGeneEffect from tidy table.
    """
    required = {"ModelID", "Gene", "ChronosGeneEffect"}
    if not required <= set(tidy.columns):
        missing = required - set(tidy.columns)
        raise ValueError(f"tidy is missing required columns: {missing}")

    mat = tidy.pivot_table(
        index="ModelID",
        columns="Gene",
        values="ChronosGeneEffect",
        aggfunc="median",
    )
    mat = mat.reset_index()
    return mat


def _default_config(config: Optional[Dict]) -> Dict:
    """
    Fill in sane defaults for the co-dependency configuration.
    """
    if config is None:
        config = {}
    cfg = dict(config)  # shallow copy

    # Minimum number of overlapping models per gene pair
    cfg.setdefault("MIN_OVERLAP", 3)

    # Maximum number of genes to include in correlation (for speed)
    cfg.setdefault("MAX_CORR_GENES", 200)

    # Correlation methods
    # If CORR_METHODS is not provided, use a single CORR_METHOD (default spearman).
    if "CORR_METHODS" not in cfg:
        cfg.setdefault("CORR_METHOD", "spearman")

    return cfg


def run_codependency_correlation(
    tidy: pd.DataFrame,
    outdir: Path,
    config: Optional[Dict] = None,
    logger=None,
) -> None:
    """
    Multi-method dependency / co-dependency correlation module.

    Parameters
    ----------
    tidy : pd.DataFrame
        Long-format dependency table with columns including:
        ['Gene','ModelID','ChronosGeneEffect','DependencyProbability', ...]
        ⚠ This table already encodes WHICH MODELS are used.
          If you want "all models", make sure the upstream step
          (build_tidy_dependencies / Selected_Models.csv) contains all models.
    outdir : Path
        Output folder (typically BASE_OUTPUT_DIR / 'DepMap_Dependencies').
    config : dict or None
        Optional config with keys like:
          - 'MIN_OVERLAP'    (default 3)
          - 'MAX_CORR_GENES' (default 200)
          - 'MIN_PROB'       (optional DepProb mask)
          - 'CORR_METHODS'   (e.g. ['spearman','pearson','cosine'])
          - 'CORR_METHOD'    (single method if CORR_METHODS is not given)
    logger : logging.Logger or None
        Optional logger; if None, prints to stdout.
    """
    log = logger.info if logger is not None else print

    _ensure_dirs(outdir)

    # --------------------------
    # 0. Build 'effect' matrix
    # --------------------------
    effect = _build_effect_matrix(tidy)
    if "ModelID" not in effect.columns:
        raise ValueError("The 'effect' matrix must contain a 'ModelID' column.")

    # --------------------------
    # 1. Settings
    # --------------------------
    cfg = _default_config(config)

    MIN_OVERLAP_BASE = int(cfg.get("MIN_OVERLAP", 3))
    MAX_CORR_GENES = int(cfg.get("MAX_CORR_GENES", 200))

    # Methods to run
    methods_cfg = cfg.get("CORR_METHODS", None)
    if methods_cfg is None:
        single = cfg.get("CORR_METHOD", "spearman")
        methods = [single.lower()]
    else:
        methods = [m.lower() for m in methods_cfg]

    supported = {"spearman", "pearson", "kendall", "cosine"}
    methods = [m for m in methods if m in supported]
    # If no valid methods provided → use ALL supported methods if not methods:
    methods = list(supported)

    method_pretty = {
        "spearman": "Spearman",
        "pearson": "Pearson",
        "kendall": "KendallTau",
        "cosine": "Cosine",
    }

    log(f"📐 Co-dependency correlation methods: {methods}")

    # --------------------------------
    # 2. Prepare Model × Gene matrix
    # --------------------------------
    df_corr = effect.set_index("ModelID").dropna(axis=1, how="all").copy()
    # strip " (1234)" from gene names if present
    df_corr.columns = df_corr.columns.str.replace(r" \(\d+\)$", "", regex=True)

    n_models, n_genes_initial = df_corr.shape
    log(f"\n📦 Co-dependency: starting with {n_models} models and {n_genes_initial} genes.")

    # Optional DepProb mask
    if cfg.get("MIN_PROB", None) is not None and "DependencyProbability" in tidy.columns:
        thr = float(cfg["MIN_PROB"])
        dep_mask = (
            tidy.groupby(["ModelID", "Gene"])["DependencyProbability"]
            .mean()
            .unstack()
        )
        dep_mask = dep_mask.reindex(index=df_corr.index, columns=df_corr.columns)
        before = df_corr.count().sum()
        df_corr = df_corr.where(dep_mask.fillna(0) >= thr)
        after = df_corr.count().sum()
        log(f"ℹ️ Applied MIN_PROB ≥ {thr}: non-NaN cells {before} → {after}")

    # Gene cap
    if df_corr.shape[1] > MAX_CORR_GENES:
        log(f"ℹ️ Limiting to first {MAX_CORR_GENES} genes for speed.")
        df_corr = df_corr.iloc[:, :MAX_CORR_GENES]

    genes = df_corr.columns.tolist()
    n = len(genes)

    # Small-n logic
    n_models = df_corr.shape[0]
    MIN_OVERLAP = MIN_OVERLAP_BASE

    if n_models < 3:
        log("\n⚠ WARNING: Very few models for co-dependency analysis.")
        log(f"   n_models = {n_models}. Treat correlations as descriptive only.")
        MIN_OVERLAP = max(2, MIN_OVERLAP_BASE)
    elif n_models < 5:
        log("\n⚠ NOTE: Small sample size for dependency analysis.")
        log(f"   n_models = {n_models} (3–4). Correlations will be unstable.")

    log(f"\nMIN_OVERLAP for this run: {MIN_OVERLAP}")
    log(f"Genes used in correlation: {n}")

    all_top_summaries: List[pd.DataFrame] = []

    # --------------------------------
    # 3. Core correlation loop
    # --------------------------------
    for method in methods:
        corr_name = method_pretty.get(method, method.capitalize())
        log("\n============================")
        log(f"Running co-dependency with {corr_name}")
        log("============================")

        rho_mat = pd.DataFrame(np.eye(n), index=genes, columns=genes, dtype=float)
        p_mat = pd.DataFrame(np.zeros((n, n)), index=genes, columns=genes, dtype=float)

        # 3a. Compute pairwise correlations / similarities
        for i, g1 in enumerate(genes):
            v1 = df_corr[g1]
            for j in range(i, n):
                g2 = genes[j]
                v2 = df_corr[g2]

                mask = v1.notna() & v2.notna()
                k = int(mask.sum())
                if k < MIN_OVERLAP:
                    r = np.nan
                    p = np.nan
                else:
                    a = v1[mask].values
                    b = v2[mask].values

                    if method == "pearson":
                        r, p = pearsonr(a, b)
                    elif method == "spearman":
                        r, p = spearmanr(a, b)
                    elif method == "kendall":
                        r, p = kendalltau(a, b)
                    elif method == "cosine":
                        denom = (np.linalg.norm(a) * np.linalg.norm(b))
                        r = float(np.dot(a, b) / denom) if denom > 0 else np.nan
                        p = np.nan
                    else:
                        r = np.nan
                        p = np.nan

                    r = float(r) if r is not None else np.nan
                    p = float(p) if p is not None else np.nan

                rho_mat.iat[i, j] = rho_mat.iat[j, i] = r
                p_mat.iat[i, j] = p_mat.iat[j, i] = p

        # 3b. FDR BH
        pairs, pvals = [], []
        for i in range(n):
            for j in range(i + 1, n):
                pv = p_mat.iat[i, j]
                if np.isfinite(pv) and not np.isnan(pv):
                    pairs.append((genes[i], genes[j]))
                    pvals.append(pv)

        if len(pvals) and method in {"spearman", "pearson", "kendall"}:
            _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            q_map = {(g1, g2): q for (g1, g2), q in zip(pairs, qvals)}
        else:
            q_map = {}

        def lookup_q(g1: str, g2: str) -> float:
            if g1 == g2:
                return 0.0
            return q_map.get((g1, g2), q_map.get((g2, g1), np.nan))

        # --------------------------------
        # 4. Save matrices & heatmaps
        # --------------------------------
        rho_path = outdir / f"{corr_name}_Correlation_Matrix.csv"
        rho_mat.to_csv(rho_path)
        log(f"💾 Saved rho matrix: {rho_path}")

        plt.figure(figsize=(0.8 * len(rho_mat), 0.8 * len(rho_mat)))
        sns.heatmap(
            rho_mat,
            cmap="coolwarm",
            center=0 if method != "cosine" else None,
            annot=(len(rho_mat) <= 20),
            fmt=".2f",
            square=True,
            cbar_kws={
                "label": f"{corr_name} correlation" if method != "cosine" else corr_name
            },
        )
        plt.title(f"Co-dependency correlation (Chronos) — {corr_name}")
        plt.tight_layout()
        plt.savefig(outdir / f"figs/codependency_heatmap_{method.lower()}.png", dpi=300)
        plt.close()

        fig_corr = px.imshow(
            rho_mat,
            x=rho_mat.columns,
            y=rho_mat.index,
            color_continuous_scale="RdBu_r" if method != "cosine" else "Viridis",
            zmin=-1 if method != "cosine" else None,
            zmax=1 if method != "cosine" else None,
            title=f"Interactive co-dependency heatmap ({corr_name})",
            labels=dict(color=corr_name),
            width=800,
            height=800,
        )
        fig_corr.write_html(
            outdir / f"html/codependency_heatmap_interactive_{method.lower()}.html"
        )

        # --------------------------------
        # 5. Long table + top 20 pairs
        # --------------------------------
        corr_long = (
            rho_mat.reset_index()
            .melt(id_vars="index", var_name="Gene2", value_name="Correlation")
            .rename(columns={"index": "Gene1"})
        )
        p_long = (
            p_mat.reset_index()
            .melt(id_vars="index", var_name="Gene2", value_name="p_value")
            .rename(columns={"index": "Gene1"})
        )
        long_df = (
            corr_long.merge(p_long, on=["Gene1", "Gene2"], how="inner")
            .query("Gene1 != Gene2")
            .copy()
        )

        if method in {"spearman", "pearson", "kendall"}:
            long_df["q_value_fdr_bh"] = long_df.apply(
                lambda r: lookup_q(r["Gene1"], r["Gene2"]), axis=1
            )
        else:
            long_df["q_value_fdr_bh"] = np.nan

        long_df["abs_corr"] = long_df["Correlation"].abs()

        # Sorting for long_df depends on method
        if method in {"spearman", "pearson", "kendall"}:
            # Use q-value + abs correlation
            long_df = long_df.sort_values(
                ["q_value_fdr_bh", "abs_corr"],
                ascending=[True, False],
            )
        else:
            # COSINE: no p-values, so sort only by absolute correlation
            long_df = long_df.sort_values("abs_corr", ascending=False)

        long_path = outdir / f"{corr_name}_Correlation_Long.csv"
        long_df.to_csv(long_path, index=False)
        log(f"💾 Saved long (r, p, q): {long_path}")

        # Top 20 pairs by absolute correlation
        top_pairs = long_df.sort_values("abs_corr", ascending=False).head(20).copy()
        top_pairs["Method"] = corr_name
        top_pairs_path = outdir / f"Top_CoDependent_GenePairs_{corr_name}.csv"
        top_pairs.to_csv(top_pairs_path, index=False)
        log(f"💾 Saved top 20 pairs: {top_pairs_path}")

        all_top_summaries.append(top_pairs)

    # --------------------------------
    # 6. Combined summary over all methods
    # --------------------------------
    if all_top_summaries:
        combined = pd.concat(all_top_summaries, ignore_index=True)
        combined_path = outdir / "Top_CoDependent_GenePairs_ALL_Methods.csv"
        combined.to_csv(combined_path, index=False)
        log(f"\n💾 Saved combined top pairs for all methods: {combined_path}")

    log(f"\n✅ Co-dependency correlation finished for methods: {methods}")
    if n_models < 5:
        log("   Reminder: with <5 models, treat correlations as exploratory only.")
