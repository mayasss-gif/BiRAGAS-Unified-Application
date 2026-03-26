from __future__ import annotations
# src/scoring_modules.py
import logging
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score



# Optional Plotly for interactive HTML
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

HGNC_PATH = Path(__file__).parent / "data" / "HGNC_GenesSymbols.csv"

if not HGNC_PATH.exists():
    raise FileNotFoundError(f"HGNC_GenesSymbols.csv not found at {HGNC_PATH}")

DRUGABILITY_PATH = Path(__file__).parent / "data" / "Drugability_Toxicity.csv"
if not DRUGABILITY_PATH.exists():
    raise FileNotFoundError(f"Drugability_Toxicity.csv not found at {DRUGABILITY_PATH}")


# =====================================================================
# GLOBALS – weights & HGNC mapping
# =====================================================================

# Module weights (must sum to 1.0)
W_EFFECT = 0.25       # Effect Strength (L1000)
W_ESS = 0.25          # Disease Essentiality (DepMap)
W_SAFETY = 0.20       # Safety Profile (tox alerts)
W_DRUG = 0.20         # Druggability (chemistry)
W_CONN = 0.10         # Connectivity (network)

# Cached HGNC mapping (dict: symbol_or_synonym -> Normalized_Gene)
_HGNC_MAP: dict | None = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rank_to_score(rank_series: pd.Series) -> pd.Series:
    """
    Convert a rank (1 = best, N = worst) into a score 0–100.

    IMPORTANT: This does NOT touch raw metrics – it's just turning rank
    into a percentage-like score.

    best (rank 1)  →  100
    worst (rank N) →  0
    """
    r = pd.to_numeric(rank_series, errors="coerce")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(
            [100.0 if not pd.isna(v) else np.nan for v in r],
            index=r.index,
        )
    score = (n - r) / (n - 1) * 100.0
    return score


def _load_hgnc_map(logger: logging.Logger) -> dict:
    """
    Load HGNC_GenesSymbols.csv and build a map:
        key   = UPPERCASE symbol or synonym
        value = Normalized_Gene (UPPERCASE)

    File is expected in the integration root:
        paths.INTEGRATION_ROOT / "HGNC_GenesSymbols.csv"
    """
    global _HGNC_MAP
    if _HGNC_MAP is not None:
        return _HGNC_MAP

    hgnc_path = HGNC_PATH
    if not hgnc_path.exists():
        logger.warning(
            "HGNC_GenesSymbols.csv not found at %s – HGNC_Gene will be empty.",
            hgnc_path,
        )
        _HGNC_MAP = {}
        return _HGNC_MAP

    df = pd.read_csv(hgnc_path)
    df.columns = [c.strip() for c in df.columns]

    if "Normalized_Gene" not in df.columns:
        logger.warning(
            "HGNC_GenesSymbols.csv missing 'Normalized_Gene' column – skipping HGNC mapping."
        )
        _HGNC_MAP = {}
        return _HGNC_MAP

    df["Normalized_Gene"] = df["Normalized_Gene"].astype(str).str.upper().str.strip()
    if "HGNC_Synonyms" in df.columns:
        df["HGNC_Synonyms"] = df["HGNC_Synonyms"].fillna("").astype(str)
    else:
        df["HGNC_Synonyms"] = ""

    gene_map: dict[str, str] = {}

    for _, row in df.iterrows():
        canon = row["Normalized_Gene"]
        # Map the canonical symbol to itself
        gene_map[canon] = canon

        syns = row["HGNC_Synonyms"]
        if syns:
            for s in syns.split(";"):
                sym = s.strip()
                if not sym:
                    continue
                sym_up = sym.upper()
                gene_map[sym_up] = canon

    _HGNC_MAP = gene_map
    logger.info("Loaded HGNC mapping for %d symbols/synonyms.", len(gene_map))
    return _HGNC_MAP


def _normalise_gene_series(genes: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    Build an HGNC-based key **without changing the original symbols**.

    1. Convert to uppercase / strip.
    2. If symbol exists in Normalized_Gene, keep it.
    3. If symbol exists as a synonym in HGNC_Synonyms, map to Normalized_Gene.
    4. If not recognised, keep original uppercase and log a warning (once).
    """
    gene_map = _load_hgnc_map(logger)
    g_up = genes.astype(str).str.upper().str.strip()

    if not gene_map:
        # No mapping available – still return uppercase key
        return g_up

    mapped = g_up.map(gene_map)
    result = mapped.fillna(g_up)

    # Warn on unmapped symbols (not too spammy)
    unmapped = sorted(set(g_up[~g_up.isin(gene_map.keys())].tolist()))
    if unmapped:
        preview = ", ".join(unmapped[:20])
        logger.warning(
            "Some gene symbols were not found in HGNC mapping (showing up to 20): %s",
            preview,
        )

    return result


# ---------------------------------------------------------------------
# L1000 loader used across modules
# ---------------------------------------------------------------------
def load_l1000_pairs(logger: logging.Logger, l1000_path: Path) -> pd.DataFrame:
    """
    Load causal_link_table_with_relevance.csv from l1000_path and:

      * Filter to Therapeutic_Relevance == 'Reversal' (if column exists).
      * Filter to R² >= 0.8 (if column exists).
      * Ensure numeric L1000 metrics.
      * Keep all remaining rows (gene–drug–doseunit fits).

    We do NOT normalise raw metrics. We **preserve the original Gene symbol**
    and add an extra 'HGNC_Gene' column used only for cross-file linkage.
    """
    # DEBUG: Check if path is None
    logger.info(f"[DEBUG] load_l1000_pairs called with l1000_path={l1000_path} (type: {type(l1000_path)})")
    if l1000_path is None:
        logger.error("[DEBUG] l1000_path is None! Cannot proceed with L1000 data loading.")
        raise ValueError("l1000_path cannot be None")
    
    if not (l1000_path / "causal_link_table_with_relevance.csv").exists():
        raise FileNotFoundError(f"Missing L1000 file: {l1000_path / 'causal_link_table_with_relevance.csv'}")

    df = pd.read_csv(l1000_path / "causal_link_table_with_relevance.csv")
    df.columns = [c.strip() for c in df.columns]

    # Filter to reversal only
    if "Therapeutic_Relevance" in df.columns:
        before = len(df)
        df = df[df["Therapeutic_Relevance"].astype(str).str.lower() == "reversal"].copy()
        logger.info(f"L1000: Reversal filter: {before} → {len(df)} rows")

    # R² filter
    if "R²" in df.columns:
        df["R²"] = pd.to_numeric(df["R²"], errors="coerce")
        before = len(df)
        df = df[df["R²"] >= 0.80].copy()
        logger.info(f"L1000: R² ≥ 0.8 filter: {before} → {len(df)} rows")

    # Numeric L1000 metrics
    for col in ["Sensitivity Index", "EC50 (µM)", "R²", "Hill Slope", "Emax", "Baseline"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve your gene symbol; add HGNC_Gene key
    df["Gene"] = df["Gene"].astype(str)
    df["HGNC_Gene"] = _normalise_gene_series(df["Gene"], logger)

    df["Drug"] = df["Drug"].astype(str)

    logger.info(f"L1000: {len(df)} rows after QC")
    return df


# =====================================================================
# 1) EFFECT STRENGTH – per-gene summary from L1000
# =====================================================================
def compute_effect_strength(output_dir: Path, logger: logging.Logger, l1000_path: Path | None) -> pd.DataFrame:
    """
    Construct per-gene perturbation effect summary:

    Input: causal_link_table_with_relevance.csv (via load_l1000_pairs).

    For each Gene (your original symbol):
        n_drug_pairs           – number of Gene–Drug rows
        max_SensitivityIndex   – maximum SI across all drugs
        mean_SensitivityIndex  – mean SI
        best_EC50              – minimum EC50
        best_R2                – maximum R²
        best_Drug              – drug achieving max SI

        effect_rank            – rank by max_SensitivityIndex (1 = best)
        effect_score_0_100     – rank converted to 0–100 (% style)

    Raw metrics are untouched; scores use only ranks.

    Adds:
        HGNC_Gene              – HGNC-based key for cross-module linkage
    """
    if l1000_path is None:
        logger.warning("L1000 path is None, returning empty effect strength DataFrame")
        empty_df = pd.DataFrame(columns=["Gene", "HGNC_Gene", "n_drug_pairs", "max_SensitivityIndex", 
                                         "mean_SensitivityIndex", "best_EC50", "best_R2", "best_Drug",
                                         "effect_rank", "effect_score_0_100"])
        out_path = output_dir / "EffectStrength_by_gene.csv"
        empty_df.to_csv(out_path, index=False)
        logger.info("Saved empty effect strength table → %s", out_path)
        return empty_df
    
    df = load_l1000_pairs(logger, l1000_path)

    # Aggregate per original gene symbol
    g = df.groupby("Gene", as_index=False).agg(
        n_drug_pairs=("Drug", "count"),
        max_SensitivityIndex=("Sensitivity Index", "max"),
        mean_SensitivityIndex=("Sensitivity Index", "mean"),
        best_EC50=("EC50 (µM)", "min"),
        best_R2=("R²", "max"),
    )

    # Best drug per gene (by SI)
    idx_best = df.groupby("Gene")["Sensitivity Index"].idxmax()
    best_rows = df.loc[idx_best, ["Gene", "Drug"]].rename(columns={"Drug": "best_Drug"})
    g = g.merge(best_rows, on="Gene", how="left")

    # HGNC key for linkage (keeps your Gene symbol as display)
    g["HGNC_Gene"] = _normalise_gene_series(g["Gene"], logger)

    # Rank and score (only from SI)
    r = g["max_SensitivityIndex"].rank(method="min", ascending=False)
    g["effect_rank"] = r.astype("Int64")
    g["effect_score_0_100"] = _rank_to_score(g["effect_rank"])

    # Save
    out_path = output_dir / "EffectStrength_by_gene.csv"
    g.to_csv(out_path, index=False)
    logger.info("Saved effect strength table → %s", out_path)

    # ---- PLOTS ----

    # Top 30 genes by max SI
    top = g.sort_values("max_SensitivityIndex", ascending=False).head(30)
    plt.figure(figsize=(10, 6))
    plt.barh(top["Gene"], top["max_SensitivityIndex"])
    plt.gca().invert_yaxis()
    plt.xlabel("Max Sensitivity Index (raw)")
    plt.title("Top 30 genes by L1000 effect strength (max SI)")
    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / "effect_strength_top30.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Saved figure → %s", fig_path)

    # All genes: EC50 vs max SI
    plt.figure(figsize=(8, 6))
    plt.scatter(
        g["best_EC50"],
        g["max_SensitivityIndex"],
        s=20,
        alpha=0.7,
    )
    plt.xscale("log")
    plt.xlabel("Best EC50 (µM, raw; lower = more potent)")
    plt.ylabel("Max Sensitivity Index (raw)")
    plt.title("L1000 effect strength – all genes")
    plt.tight_layout()
    fig_scatter = plots_dir / "effect_strength_all_genes_scatter.png"
    plt.savefig(fig_scatter, dpi=150)
    plt.close()
    logger.info("Saved figure → %s", fig_scatter)

    if PLOTLY_AVAILABLE:
        fig_html = plots_dir / "effect_strength_all_genes_scatter.html"
        fig = px.scatter(
            g,
            x="best_EC50",
            y="max_SensitivityIndex",
            hover_name="Gene",
            log_x=True,
            title="L1000 effect strength – all genes (raw metrics)",
        )
        fig.update_xaxes(title="Best EC50 (µM, raw)")
        fig.update_yaxes(title="Max Sensitivity Index (raw)")
        fig.write_html(fig_html)
        logger.info("Saved interactive HTML → %s", fig_html)

    return g


# =====================================================================
# 2) ESSENTIALITY – DepMap median gene effect (HGNC linkage only)
# =====================================================================
def compute_essentiality(output_dir: Path, logger: logging.Logger, depmap_path: Path) -> pd.DataFrame:
    """
    Use GeneEssentiality_ByMedian.csv (raw).

    Preserves original 'Gene' symbol and adds 'HGNC_Gene' for linkage.

    Keeps:
        Gene (original symbol)
        HGNC_Gene
        n_models, median_effect, q10, q90, etc.
        BiologicalTag

    Adds:
        rank_gene_essentiality (more negative = more essential = better)
        essentiality_score_0_100 (from rank)
        QC metrics: F1, AUC (logged)
    """
    # DEBUG: Check if path is None
    logger.info(f"[DEBUG] compute_essentiality called with depmap_path={depmap_path} (type: {type(depmap_path)})")
    if depmap_path is None:
        logger.error("[DEBUG] depmap_path is None! Cannot proceed with DepMap essentiality loading.")
        raise ValueError("depmap_path cannot be None")
    
    if not (depmap_path / "DepMap_Dependencies" / "GeneEssentiality_ByMedian.csv").exists():
        raise FileNotFoundError(f"Missing essentiality file: {depmap_path / 'DepMap_Dependencies' / 'GeneEssentiality_ByMedian.csv'}")

    ess = pd.read_csv(depmap_path / "DepMap_Dependencies" / "GeneEssentiality_ByMedian.csv")
    ess.columns = [c.strip() for c in ess.columns]

    if "Gene" not in ess.columns or "median_effect" not in ess.columns:
        raise ValueError("GeneEssentiality_ByMedian.csv must contain 'Gene' and 'median_effect'.")

    # Preserve original symbol, add HGNC key
    ess["Gene"] = ess["Gene"].astype(str)
    ess["HGNC_Gene"] = _normalise_gene_series(ess["Gene"], logger)

    # Convert numeric fields
    num_to_numeric = [
        "median_effect",
        "n_models",
        "q10",
        "q90",
        "n_prob50",
        "n_strong_lt_1",
    ]
    for col in num_to_numeric:
        if col in ess.columns:
            ess[col] = pd.to_numeric(ess[col], errors="coerce")

    ess["median_effect"] = pd.to_numeric(ess["median_effect"], errors="coerce")

    # QC label: BiologicalTag-based essential / non-essential
    tag_col = "BiologicalTag" if "BiologicalTag" in ess.columns else None
    if tag_col:
        tags = ess[tag_col].astype(str)
        is_essential = tags.str.contains("essential", case=False)
        y_true = is_essential.astype(int)
        scores = -ess["median_effect"]  # more negative effect = larger "essentiality score"

        try:
            f1 = f1_score(y_true, scores > np.median(scores))
            auc = roc_auc_score(y_true, scores)
            logger.info("Essentiality QC: F1=%.3f, AUC=%.3f", f1, auc)
        except Exception:
            logger.warning("Could not compute F1/AUC for essentiality (degenerate labels).")

    # Rank and score (more negative effect = better)
    r = ess["median_effect"].rank(method="min", ascending=True)
    ess["rank_gene_essentiality"] = r.astype("Int64")
    ess["essentiality_score_0_100"] = _rank_to_score(ess["rank_gene_essentiality"])

    out_path = output_dir / "Essentiality_by_gene.csv"
    ess.to_csv(out_path, index=False)
    logger.info("Saved essentiality table → %s", out_path)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plots – distribution by BiologicalTag if available
    if tag_col:
        # Histogram
        plt.figure(figsize=(10, 6))
        for tag, sub in ess.groupby(tag_col):
            vals = sub["median_effect"].dropna()
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=40, alpha=0.4, label=tag)
        plt.xlabel("Median gene effect (raw DepMap)")
        plt.ylabel("Count")
        plt.title("Distribution of median essentiality by biological tag")
        plt.legend()
        plt.tight_layout()
        fig_path = plots_dir / "essentiality_distribution_by_tag.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Saved figure → %s", fig_path)

        # Jittered scatter
        plt.figure(figsize=(8, 6))
        tag_ids = {t: i for i, t in enumerate(sorted(ess[tag_col].unique()))}
        x = ess[tag_col].map(tag_ids).astype(float)
        x_jit = x + np.random.uniform(-0.15, 0.15, size=len(x))
        plt.scatter(x_jit, ess["median_effect"], alpha=0.5, s=10)
        plt.xticks(list(tag_ids.values()), list(tag_ids.keys()), rotation=45, ha="right")
        plt.ylabel("Median gene effect (raw)")
        plt.title("Essentiality per tag (DepMap)")
        plt.tight_layout()
        fig_path2 = plots_dir / "essentiality_score_vs_tag.png"
        plt.savefig(fig_path2, dpi=150)
        plt.close()
        logger.info("Saved figure → %s", fig_path2)

        if PLOTLY_AVAILABLE:
            fig_html = plots_dir / "essentiality_violin_by_tag.html"
            fig = px.violin(
                ess,
                x=tag_col,
                y="median_effect",
                box=True,
                points="all",
                title="Essentiality (median effect) by biological tag (raw)",
            )
            fig.update_yaxes(title="Median gene effect (raw)")
            fig.write_html(fig_html)
            logger.info("Saved interactive HTML → %s", fig_html)

    return ess


# =====================================================================
# 3) CONNECTIVITY – PPI / CGC (HGNC linkage only)
# =====================================================================
def compute_connectivity(output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Use Connectivity_Score.csv (raw).

    Expected columns:
        Gene, CGC, PPI_Degree

    Preserves original 'Gene' symbol and adds 'HGNC_Gene' for linkage.

    Adds:
        rank_connectivity (higher degree = better)
        connectivity_score_0_100
    """
    
    if not (output_dir / "Connectivity_Score.csv").exists():
        raise FileNotFoundError(f"Missing file: {output_dir / 'Connectivity_Score.csv'}")

    df = pd.read_csv(output_dir / "Connectivity_Score.csv")
    df.columns = [c.strip() for c in df.columns]

    if "Gene" not in df.columns:
        raise ValueError("Connectivity_Score.csv needs a 'Gene' column.")

    df["Gene"] = df["Gene"].astype(str)
    df["HGNC_Gene"] = _normalise_gene_series(df["Gene"], logger)

    if "PPI_Degree" in df.columns:
        df["PPI_Degree"] = pd.to_numeric(df["PPI_Degree"], errors="coerce")
        r = df["PPI_Degree"].rank(method="min", ascending=False)
        df["rank_connectivity"] = r.astype("Int64")
        df["connectivity_score_0_100"] = _rank_to_score(df["rank_connectivity"])
    else:
        df["PPI_Degree"] = np.nan
        df["rank_connectivity"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["connectivity_score_0_100"] = np.nan

    if "CGC" in df.columns:
        df["CGC"] = pd.to_numeric(df["CGC"], errors="coerce").fillna(0).astype(int)
    else:
        df["CGC"] = 0

    out_path = output_dir / "Connectivity_by_gene.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved connectivity table → %s", out_path)

    # Plot: top 30 by PPI_Degree
    if "PPI_Degree" in df.columns:
        top = df.sort_values("PPI_Degree", ascending=False).head(30)
        plt.figure(figsize=(10, 6))
        plt.barh(top["Gene"], top["PPI_Degree"])
        plt.gca().invert_yaxis()
        plt.xlabel("PPI Degree (raw)")
        plt.title("Top 30 genes by connectivity")
        plt.tight_layout()
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plots_dir / "connectivity_top30.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Saved figure → %s", fig_path)

        if PLOTLY_AVAILABLE:
            fig_html = plots_dir / "connectivity_top30.html"
            fig = px.bar(
                top.sort_values("PPI_Degree", ascending=True),
                x="PPI_Degree",
                y="Gene",
                orientation="h",
                title="Top 30 genes by connectivity (raw PPI degree)",
            )
            fig.update_xaxes(title="PPI Degree")
            fig.write_html(fig_html)
            logger.info("Saved interactive HTML → %s", fig_html)

    return df


# =====================================================================
# 4) DRUGGABILITY & SAFETY – per drug
# =====================================================================
def compute_druggability_and_safety(output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Use Drugability_Toxicity.csv (raw).

    Keeps raw descriptors and flags.

    Adds:
        rank_druggability (Overall Drugability Score descending)
        druggability_score_0_100
        safety_score_0_100 (penalty for PAINS/Brenk/reactive toxicophores)

    Outputs:
        - Outputs/Drugability_by_drug.csv
        - Plots/druggability_qed_vs_overall.(png/html)
        - Plots/safety_alert_summary.(png/html)
        - Plots/top_drugs_druggability_safety.(png/html)   <-- NEW
    """
    if not DRUGABILITY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DRUGABILITY_PATH}")

    df = pd.read_csv(DRUGABILITY_PATH)
    df.columns = [c.strip() for c in df.columns]

    if "pert_iname" not in df.columns:
        raise ValueError("Drugability_Toxicity.csv must contain 'pert_iname'.")

    df["pert_iname"] = df["pert_iname"].astype(str)

    # ---- numeric columns ----
    num_cols = [
        "Quantitative Estimate of Drug-likeness (QED)",
        "Molecular Weight (MW)",
        "Lipophilicity / Partition Coefficient (LogP)",
        "Hydrogen Bond Donors (HBD)",
        "Hydrogen Bond Acceptors (HBA)",
        "Lipinski Rule-of-5 Violations",
        "Topological Polar Surface Area (TPSA)",
        "Rotatable Bonds Count",
        "Aromatic Ring Count",
        "Total Ring Count",
        "Fraction of sp3 Carbon Atoms (FractionCSP3)",
        "Overall Drugability Score",
        "Num_Reactive_Toxicophores",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- alert flags ----
    for col in ["Has_PAINS_Alert", "Has_Brenk_Alert"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df[col] = False

    if "Num_Reactive_Toxicophores" not in df.columns:
        df["Num_Reactive_Toxicophores"] = 0.0

    # ---- Druggability rank & score (0–100) ----
    if "Overall Drugability Score" in df.columns:
        r = df["Overall Drugability Score"].rank(method="min", ascending=False)
        df["rank_druggability"] = r.astype("Int64")
        df["druggability_score_0_100"] = _rank_to_score(df["rank_druggability"])
    else:
        df["rank_druggability"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["druggability_score_0_100"] = np.nan
        logger.warning("Drugability_Toxicity.csv missing 'Overall Drugability Score'.")

    # ---- Safety score: 100 - penalties ----
    pains_penalty = np.where(df["Has_PAINS_Alert"], 30.0, 0.0)
    brenk_penalty = np.where(df["Has_Brenk_Alert"], 20.0, 0.0)
    reactive_penalty = np.clip(df["Num_Reactive_Toxicophores"].fillna(0.0), 0, 5) * 10.0

    safety_raw = 100.0 - (pains_penalty + brenk_penalty + reactive_penalty)
    df["safety_score_0_100"] = np.clip(safety_raw, 0.0, 100.0)

    # ---- write main output table ----
    out_path = output_dir / "Drugability_by_drug.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved druggability table → %s", out_path)

    # ---- PLOTS ----

    # (1) Druggability scatter: QED vs Overall score
    if (
        "Quantitative Estimate of Drug-likeness (QED)" in df.columns
        and "Overall Drugability Score" in df.columns
    ):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df["Quantitative Estimate of Drug-likeness (QED)"],
            df["Overall Drugability Score"],
            s=15,
            alpha=0.6,
        )
        plt.xlabel("QED (raw)")
        plt.ylabel("Overall Drugability Score (raw)")
        plt.title("Druggability landscape – all drugs")
        plt.tight_layout()
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plots_dir / "druggability_qed_vs_overall.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Saved figure → %s", fig_path)

        if PLOTLY_AVAILABLE:
            fig_html = plots_dir / "druggability_qed_vs_overall.html"
            fig = px.scatter(
                df,
                x="Quantitative Estimate of Drug-likeness (QED)",
                y="Overall Drugability Score",
                hover_name="pert_iname",
                title="Druggability landscape – all drugs (raw values)",
            )
            fig.update_xaxes(title="QED (raw)")
            fig.update_yaxes(title="Overall Drugability Score (raw)")
            fig.write_html(fig_html)
            logger.info("Saved interactive HTML → %s", fig_html)

    # (2) Safety summary: alert-free vs flagged
    alert_flag = (
        df["Has_PAINS_Alert"]
        | df["Has_Brenk_Alert"]
        | (df["Num_Reactive_Toxicophores"].fillna(0.0) > 0)
    )
    df["AnyAlert"] = np.where(alert_flag, "Has structural alerts", "No alerts")

    counts = df["AnyAlert"].value_counts().reset_index()
    counts.columns = ["AnyAlert", "n"]

    plt.figure(figsize=(6, 4))
    plt.bar(counts["AnyAlert"], counts["n"])
    plt.ylabel("# of compounds")
    plt.title("Safety profile – structural alert summary")
    plt.tight_layout()
    fig_path = plots_dir / "safety_alert_summary.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Saved figure → %s", fig_path)

    if PLOTLY_AVAILABLE:
        fig_html = plots_dir / "safety_alert_summary.html"
        fig = px.bar(
            counts,
            x="AnyAlert",
            y="n",
            title="Safety profile – structural alert summary",
        )
        fig.update_yaxes(title="# of compounds")
        fig.write_html(fig_html)
        logger.info("Saved interactive HTML → %s", fig_html)

    # (3) NEW: Druggability + safety view per drug (heatmap + grouped bar)
    # Top 50 by Overall Drugability Score
    if "Overall Drugability Score" in df.columns:
        top_drugs = df.sort_values(
            "Overall Drugability Score", ascending=False
        ).head(50)

        # Grouped bar (PNG)
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(top_drugs))
        width = 0.4

        plt.bar(
            x_pos - width / 2,
            top_drugs["druggability_score_0_100"],
            width=width,
            label="Druggability score (0–100)",
        )
        plt.bar(
            x_pos + width / 2,
            top_drugs["safety_score_0_100"],
            width=width,
            label="Safety score (0–100)",
        )
        plt.xticks(x_pos, top_drugs["pert_iname"], rotation=90)
        plt.ylabel("Score (0–100)")
        plt.title("Top drugs – druggability & safety scores")
        plt.legend()
        plt.tight_layout()
        fig_path = plots_dir / "top_drugs_druggability_safety.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Saved figure → %s", fig_path)

        # Heatmap (HTML)
        if PLOTLY_AVAILABLE:
            heat_df = top_drugs.set_index("pert_iname")[
                ["druggability_score_0_100", "safety_score_0_100"]
            ]
            fig_html = plots_dir / "top_drugs_druggability_safety_heatmap.html"
            fig = px.imshow(
                heat_df,
                labels=dict(x="Metric", y="Drug", color="Score (0–100)"),
                title="Top drugs – druggability & safety (heatmap)",
                aspect="auto",
            )
            fig.write_html(fig_html)
            logger.info("Saved interactive HTML → %s", fig_html)

    return df


# =====================================================================
# 5) FINAL PRIORITIZATION – gene-level + gene–drug pairs
# =====================================================================
def build_final_prioritization(
    output_dir: Path,
    logger: logging.Logger,
    l1000_path: Path | None,
    effect_df: pd.DataFrame,
    ess_df: pd.DataFrame,
    conn_df: pd.DataFrame,
    drug_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integrate all modules.

    1) Gene-level table (Final_Gene_Priorities.csv)
        Gene  (your L1000 gene symbol – unchanged)
        HGNC_Gene (internal key for matching)
        L1000 effect metrics + effect_score_0_100
        Essentiality metrics + essentiality_score_0_100
        Connectivity metrics + connectivity_score_0_100
        Best reversing drug + its druggability & safety scores

        Module weights:
            Effect Strength      25%
            Disease Essentiality 25%
            Safety Profile       20%
            Druggability         20%
            Connectivity         10%

        IntegratedScore_0_100 = sum(weight_m * score_m)

    2) Gene–drug pair-level table (Final_GeneDrug_Pairs.csv)
        All L1000 gene–drug pairs after QC, with:
            - raw L1000 metrics
            - gene essentiality metrics
            - drug druggability & safety
            - PairScore_0_100 = same weighted formula at pair level

    Also generates:
        - final_integrated_scores_all_genes.html
        - final_gene_module_contributions_stacked.(png/html)
        - top_genes_druggability_safety.(html)   <-- NEW
    """
    # ----------------- Gene-level integration -----------------
    # Start from effect_df if available, otherwise from ess_df
    if effect_df.empty:
        logger.warning("Effect strength DataFrame is empty, starting integration from essentiality data")
        gene = ess_df[["Gene", "HGNC_Gene"]].copy() if "Gene" in ess_df.columns else pd.DataFrame(columns=["Gene", "HGNC_Gene"])
        if gene.empty and "HGNC_Gene" in ess_df.columns:
            gene = pd.DataFrame({"HGNC_Gene": ess_df["HGNC_Gene"].unique()})
            gene["Gene"] = gene["HGNC_Gene"]
    else:
        gene = effect_df.copy()

    if "HGNC_Gene" not in gene.columns:
        gene["HGNC_Gene"] = _normalise_gene_series(gene["Gene"], logger)

    # Essentiality – merge by HGNC_Gene (keep your Gene symbol from L1000)
    ess_sub = ess_df[
        [
            c
            for c in [
                "HGNC_Gene",
                "median_effect",
                "BiologicalTag",
                "rank_gene_essentiality",
                "essentiality_score_0_100",
            ]
            if c in ess_df.columns
        ]
    ].copy()
    gene = gene.merge(ess_sub, on="HGNC_Gene", how="left")

    # Connectivity – merge by HGNC_Gene
    if "HGNC_Gene" in conn_df.columns:
        conn_sub = conn_df[
            [
                c
                for c in [
                    "HGNC_Gene",
                    "PPI_Degree",
                    "CGC",
                    "rank_connectivity",
                    "connectivity_score_0_100",
                ]
                if c in conn_df.columns
            ]
        ].copy()
        gene = gene.merge(conn_sub, on="HGNC_Gene", how="left")
    else:
        gene["PPI_Degree"] = np.nan
        gene["CGC"] = np.nan
        gene["rank_connectivity"] = pd.Series([pd.NA] * len(gene), dtype="Int64")
        gene["connectivity_score_0_100"] = np.nan

    # Druggability & safety for best drug per gene
    if not effect_df.empty and "best_Drug" in effect_df.columns and not drug_df.empty:
        dsub = drug_df.copy()
        dsub = dsub.rename(columns={"pert_iname": "best_Drug"})
        keep_cols = [
            "best_Drug",
            "Overall Drugability Score",
            "Quantitative Estimate of Drug-likeness (QED)",
            "Has_PAINS_Alert",
            "Has_Brenk_Alert",
            "Num_Reactive_Toxicophores",
            "rank_druggability",
            "druggability_score_0_100",
            "safety_score_0_100",
        ]
        keep_cols = [c for c in keep_cols if c in dsub.columns]
        dsub = dsub[keep_cols].drop_duplicates("best_Drug")
        gene = gene.merge(dsub, on="best_Drug", how="left")
    else:
        gene["druggability_score_0_100"] = np.nan
        gene["safety_score_0_100"] = np.nan
        if "best_Drug" not in gene.columns:
            gene["best_Drug"] = ""

    # ------------------------------------------------------------------
    # Weighted gene-level integrated score
    # Missing modules are treated as 0 contribution.
    # ------------------------------------------------------------------
    gene["effect_score_0_100"] = pd.to_numeric(gene.get("effect_score_0_100"), errors="coerce")
    gene["essentiality_score_0_100"] = pd.to_numeric(gene.get("essentiality_score_0_100"), errors="coerce")
    gene["connectivity_score_0_100"] = pd.to_numeric(gene.get("connectivity_score_0_100"), errors="coerce")
    gene["druggability_score_0_100"] = pd.to_numeric(gene.get("druggability_score_0_100"), errors="coerce")
    gene["safety_score_0_100"] = pd.to_numeric(gene.get("safety_score_0_100"), errors="coerce")

    gene["effect_contrib"] = W_EFFECT * gene["effect_score_0_100"].fillna(0.0)
    gene["essentiality_contrib"] = W_ESS * gene["essentiality_score_0_100"].fillna(0.0)
    gene["connectivity_contrib"] = W_CONN * gene["connectivity_score_0_100"].fillna(0.0)
    gene["druggability_contrib"] = W_DRUG * gene["druggability_score_0_100"].fillna(0.0)
    gene["safety_contrib"] = W_SAFETY * gene["safety_score_0_100"].fillna(0.0)

    gene["IntegratedScore_0_100"] = (
        gene["effect_contrib"]
        + gene["essentiality_contrib"]
        + gene["connectivity_contrib"]
        + gene["druggability_contrib"]
        + gene["safety_contrib"]
    )

    gene = gene.sort_values("IntegratedScore_0_100", ascending=False)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_gene = output_dir / "Final_Gene_Priorities.csv"
    gene.to_csv(out_gene, index=False)
    logger.info("Saved final gene priority table → %s", out_gene)

    # ------------------------------------------------------------------
    # PLOT 1: Top 30 genes by integrated score (PNG)
    # ------------------------------------------------------------------
    top_g = gene.head(30).copy()
    plt.figure(figsize=(10, 6))
    plt.barh(top_g["Gene"], top_g["IntegratedScore_0_100"])
    plt.gca().invert_yaxis()
    plt.xlabel("IntegratedScore_0_100 (higher = better)")
    plt.title("Top 30 genes – integrated therapeutic priority (weighted)")
    plt.tight_layout()
    fig_path = output_dir / "plots" / "final_prioritized_genes_top30.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Saved figure → %s", fig_path)

    # ------------------------------------------------------------------
    # PLOT 2: Interactive HTML – integrated scores (all genes)
    # ------------------------------------------------------------------
    if PLOTLY_AVAILABLE:
        gene_for_html = gene.copy()

        fig_html_all = plots_dir / "final_integrated_scores_all_genes.html"
        fig = px.bar(
            gene_for_html.sort_values("IntegratedScore_0_100", ascending=True),
            x="IntegratedScore_0_100",
            y="Gene",
            orientation="h",
            hover_data={
                "IntegratedScore_0_100": ":.2f",
                "effect_score_0_100": ":.2f",
                "essentiality_score_0_100": ":.2f",
                "connectivity_score_0_100": ":.2f",
                "druggability_score_0_100": ":.2f",
                "safety_score_0_100": ":.2f",
                "effect_contrib": ":.2f",
                "essentiality_contrib": ":.2f",
                "connectivity_contrib": ":.2f",
                "druggability_contrib": ":.2f",
                "safety_contrib": ":.2f",
                "best_Drug": True,
            },
            title="Integrated gene scores (0–100, weighted) – hover for module scores",
        )
        fig.update_xaxes(title="Integrated score (0–100, higher = better)")
        fig.update_yaxes(title="Gene")
        fig.write_html(fig_html_all)
        logger.info("Saved interactive HTML → %s", fig_html_all)

    # ------------------------------------------------------------------
    # PLOT 3: Stacked bar – module-wise contributions (weighted)
    # ------------------------------------------------------------------
    contrib_cols = [
        "effect_contrib",
        "essentiality_contrib",
        "safety_contrib",
        "druggability_contrib",
        "connectivity_contrib",
    ]
    stack_df = top_g.set_index("Gene")[contrib_cols].fillna(0.0)

    # PNG stacked bar (module contributions per gene, sum = IntegratedScore)
    plt.figure(figsize=(10, 7))
    stack_df.plot(
        kind="barh",
        stacked=True,
        ax=plt.gca(),
    )
    plt.xlabel("Weighted contribution to integrated score (0–100 total)")
    plt.ylabel("Gene")
    plt.title("Module-wise contributions for top genes (stacked, weighted)")
    plt.tight_layout()
    stacked_png = plots_dir / "final_gene_module_contributions_stacked.png"
    plt.savefig(stacked_png, dpi=150)
    plt.close()
    logger.info("Saved stacked contribution figure → %s", stacked_png)

    if PLOTLY_AVAILABLE:
        melted = (
            stack_df.reset_index()
            .melt(id_vars="Gene", var_name="Module", value_name="Contribution")
        )
        stacked_html = plots_dir / "final_gene_module_contributions_stacked.html"
        fig_stack = px.bar(
            melted,
            x="Contribution",
            y="Gene",
            color="Module",
            orientation="h",
            title="Module-wise contributions for top genes (stacked, weighted)",
            hover_data={
                "Contribution": ":.2f",
                "Module": True,
                "Gene": True,
            },
        )
        fig_stack.update_xaxes(title="Contribution to integrated score (points out of 100)")
        fig_stack.update_yaxes(title="Gene")
        fig_stack.write_html(stacked_html)
        logger.info("Saved interactive HTML → %s", stacked_html)

    # ------------------------------------------------------------------
    # NEW PLOT 4: For top genes – show best drug's druggability & safety
    # ------------------------------------------------------------------
    if PLOTLY_AVAILABLE:
        top_drug_view = top_g[
            ["Gene", "best_Drug", "druggability_score_0_100", "safety_score_0_100"]
        ].copy()
        fig_html = plots_dir / "top_genes_druggability_safety.html"
        fig = px.bar(
            top_drug_view,
            x="druggability_score_0_100",
            y="Gene",
            orientation="h",
            hover_data={
                "best_Drug": True,
                "safety_score_0_100": ":.2f",
            },
            title="Top genes – best drug druggability (hover for safety)",
        )
        fig.update_xaxes(title="Druggability score (0–100)")
        fig.update_yaxes(title="Gene")
        fig.write_html(fig_html)
        logger.info("Saved interactive HTML → %s", fig_html)

    # ----------------- Pair-level integration -----------------
    if l1000_path is None:
        logger.warning("L1000 path is None, skipping gene-drug pair generation")
        empty_pairs = pd.DataFrame(columns=["Gene", "Drug", "HGNC_Gene", "PairScore_0_100"])
        out_pairs = output_dir / "Final_GeneDrug_Pairs.csv"
        empty_pairs.to_csv(out_pairs, index=False)
        logger.info("Saved empty gene-drug pairs table → %s", out_pairs)
        return gene, empty_pairs
    
    l1000_pairs = load_l1000_pairs(logger, l1000_path)

    if "HGNC_Gene" not in l1000_pairs.columns:
        l1000_pairs["HGNC_Gene"] = _normalise_gene_series(l1000_pairs["Gene"], logger)

    # Map gene-level scores to each pair via HGNC_Gene
    g_sub = gene[
        [
            "HGNC_Gene",
            "effect_score_0_100",
            "essentiality_score_0_100",
            "connectivity_score_0_100",
            "druggability_score_0_100",
            "safety_score_0_100",
            "IntegratedScore_0_100",
        ]
    ].copy()
    pair = l1000_pairs.merge(g_sub, on="HGNC_Gene", how="left")

    # Map drug-level scores to each pair
    if not drug_df.empty:
        dsub2 = drug_df.rename(columns={"pert_iname": "Drug"})
        keep = [
            "Drug",
            "Overall Drugability Score",
            "druggability_score_0_100",
            "safety_score_0_100",
            "Has_PAINS_Alert",
            "Has_Brenk_Alert",
            "Num_Reactive_Toxicophores",
        ]
        keep = [c for c in keep if c in dsub2.columns]
        dsub2 = dsub2[keep]
        pair = pair.merge(dsub2, on="Drug", how="left")

    # Weighted pair-level score
    for col in [
        "effect_score_0_100",
        "essentiality_score_0_100",
        "connectivity_score_0_100",
        "druggability_score_0_100",
        "safety_score_0_100",
    ]:
        pair[col] = pd.to_numeric(pair.get(col), errors="coerce")

    pair["PairScore_0_100"] = (
        W_EFFECT * pair["effect_score_0_100"].fillna(0.0)
        + W_ESS * pair["essentiality_score_0_100"].fillna(0.0)
        + W_CONN * pair["connectivity_score_0_100"].fillna(0.0)
        + W_DRUG * pair["druggability_score_0_100"].fillna(0.0)
        + W_SAFETY * pair["safety_score_0_100"].fillna(0.0)
    )

    pair = pair.sort_values("PairScore_0_100", ascending=False)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_pair = output_dir / "Final_GeneDrug_Pairs.csv"
    pair.to_csv(out_pair, index=False)
    logger.info("Saved final Gene–Drug pair table → %s", out_pair)

    # Plot: top 30 pairs
    top_p = pair.head(30).copy()
    plt.figure(figsize=(10, 7))
    labels = top_p["Gene"] + " | " + top_p["Drug"]
    plt.barh(labels, top_p["PairScore_0_100"])
    plt.gca().invert_yaxis()
    plt.xlabel("PairScore_0_100 (higher = more attractive pair)")
    plt.title("Top 30 Gene–Drug pairs – integrated scores (weighted)")
    plt.tight_layout()
    fig_path = plots_dir / "final_gene_drug_pairs_top30.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Saved figure → %s", fig_path)

    if PLOTLY_AVAILABLE:
        fig_html_pairs = plots_dir / "final_gene_drug_pairs_top30.html"
        fig_pairs = px.bar(
            top_p.sort_values("PairScore_0_100", ascending=True),
            x="PairScore_0_100",
            y=labels,
            orientation="h",
            title="Top 30 Gene–Drug pairs – integrated scores (weighted)",
        )
        fig_pairs.update_xaxes(title="PairScore_0_100")
        fig_pairs.write_html(fig_html_pairs)
        logger.info("Saved interactive HTML → %s", fig_html_pairs)

    # ----------------- Write formulae / meta -----------------
    formula_path = output_dir / "Integration_Formulae.txt"
    with open(formula_path, "w") as f:
        f.write(
            "INTEGRATION FORMULAE (all raw metrics preserved)\n"
            "------------------------------------------------\n\n"
            "MODULE WEIGHTS:\n"
            f"  Effect Strength      = {W_EFFECT*100:.0f}%\n"
            f"  Disease Essentiality = {W_ESS*100:.0f}%\n"
            f"  Safety Profile       = {W_SAFETY*100:.0f}%\n"
            f"  Druggability         = {W_DRUG*100:.0f}%\n"
            f"  Connectivity         = {W_CONN*100:.0f}%\n\n"
            "NOTE: Original Gene symbols are preserved. 'HGNC_Gene' is used only\n"
            "for matching the same biology across L1000, DepMap, and connectivity.\n\n"
            "1) L1000 effect per gene:\n"
            "   - max_SensitivityIndex = max SI across all reversal drug pairs (R² ≥ 0.8).\n"
            "   - effect_rank = rank(max_SensitivityIndex, descending; 1 = strongest).\n"
            "   - effect_score_0_100 = (N - effect_rank) / (N - 1) * 100.\n\n"
            "2) Essentiality (DepMap):\n"
            "   - rank_gene_essentiality = rank(median_effect, ascending; more negative = better).\n"
            "   - essentiality_score_0_100 = (N - rank_gene_essentiality) / (N - 1) * 100.\n\n"
            "3) Connectivity:\n"
            "   - rank_connectivity = rank(PPI_Degree, descending).\n"
            "   - connectivity_score_0_100 = (N - rank_connectivity) / (N - 1) * 100.\n\n"
            "4) Druggability & Safety:\n"
            "   - rank_druggability = rank(Overall Drugability Score, descending).\n"
            "   - druggability_score_0_100 = (N - rank_druggability) / (N - 1) * 100.\n"
            "   - safety_score_0_100 = 100 - [30*PAINS + 20*Brenk + 10*min(Num_Reactive_Toxicophores,5)],\n"
            "     clipped to [0,100].\n\n"
            "5) Gene-level IntegratedScore_0_100 (weighted):\n"
            "   - effect_contrib        = 0.25 * effect_score_0_100\n"
            "   - essentiality_contrib  = 0.25 * essentiality_score_0_100\n"
            "   - safety_contrib        = 0.20 * safety_score_0_100\n"
            "   - druggability_contrib  = 0.20 * druggability_score_0_100\n"
            "   - connectivity_contrib  = 0.10 * connectivity_score_0_100\n"
            "   - IntegratedScore_0_100 = sum of the above contributions.\n\n"
            "6) Gene–Drug PairScore_0_100 (weighted):\n"
            "   - Same weights applied to the module scores mapped to that pair via HGNC_Gene.\n"
        )
    logger.info("Saved integration formulae → %s", formula_path)

    return gene, pair
