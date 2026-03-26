#!/usr/bin/env python
"""
Visualization of top drugs for L1000 pipeline.

Reads:
  - Latest run directory: runs/L1000_Run_*/
  - N3_best_significant_QC.csv  (top hits table)
  - drug_similarity_<metric>.csv   (from run_drugsimilarity_more.py)

Writes (into that run's figures/):
  - top_hits_barplot.png
  - top_hits_ec50_vs_si.png
  - top_hits_ATE_volcano.png
  - top_drugs_dendrogram_<metric>.png
  - top_drugs_heatmap_<metric>.png   (if seaborn installed)
"""
from __future__ import annotations

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# seaborn is optional
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_best_table(run_dir: Path) -> pd.DataFrame:
    """
    Load N3_best_significant_QC.csv (preferred) or fall back to
    best_significant_QC.csv if needed.
    """
    pref = run_dir / "N3_best_significant_QC.csv"
    fallback = run_dir / "best_significant_QC.csv"

    if pref.is_file():
        path = pref
    elif fallback.is_file():
        path = fallback
    else:
        raise FileNotFoundError(
            f"Could not find N3_best_significant_QC.csv or best_significant_QC.csv in {run_dir}"
        )

    print(f"📥 Loading top hits table from: {path}")
    df = pd.read_csv(path)

    # numeric sanitisation
    num_cols = [
        "EC50 (µM)",
        "Sensitivity Index",
        "SI_clamped",
        "R²",
        "ATE",
        "ATE_p",
        "BEST_SCORE",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _select_one_best_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full N3_best_significant_QC table, choose **one best row per gene**.

    Priority:
      1. BEST_SCORE  (descending)
      2. CIS_causal  (descending, if present)
      3. CIS         (descending, if present)
    """
    if "BEST_SCORE" not in df.columns:
        # simple fallback: use CIS or Sensitivity Index
        sort_cols = []
        if "CIS_causal" in df.columns:
            sort_cols.append(("CIS_causal", False))
        if "CIS" in df.columns:
            sort_cols.append(("CIS", False))
        if "Sensitivity Index" in df.columns:
            sort_cols.append(("Sensitivity Index", False))
        if not sort_cols:
            sort_cols = [("Gene", True)]
    else:
        sort_cols = [("BEST_SCORE", False)]
        if "CIS_causal" in df.columns:
            sort_cols.append(("CIS_causal", False))
        if "CIS" in df.columns:
            sort_cols.append(("CIS", False))

    by = [c for c, _ in sort_cols]
    asc = [a for _, a in sort_cols]

    ranked = df.sort_values(by=by, ascending=asc, kind="mergesort")
    best = ranked.drop_duplicates(subset=["Gene"], keep="first").reset_index(drop=True)

    print(f"✅ Selected {len(best)} genes (one best drug per gene).")
    return best


# ---------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------
def _relevance_color_map():
    # Nice, consistent colours
    return {
        "Reversal": "#1b9e77",      # teal / green
        "Aggravating": "#d95f02",   # orange
        "Ambiguous": "#7570b3",     # purple
        "NA": "#999999",
        "UNKNOWN": "#999999",
    }


def _map_relevance(series: pd.Series) -> list[str]:
    cmap = _relevance_color_map()
    out = []
    for v in series.astype(str):
        key = v.strip()
        if not key:
            key = "NA"
        out.append(cmap.get(key, "#999999"))
    return out


# ---------------------------------------------------------------------
# Plot 1: barplot of one best drug per gene
# ---------------------------------------------------------------------
def plot_top_hits_barplot(best: pd.DataFrame, fig_dir: Path):
    """
    Horizontal barplot:
      y-axis: Gene
      bar length: BEST_SCORE
      colour: Therapeutic_Relevance
      label: drug name
    """
    if "BEST_SCORE" not in best.columns:
        print("⚠️ BEST_SCORE not in table; skipping barplot.")
        return

    best = best.copy()
    best = best.sort_values("BEST_SCORE", ascending=True)
    genes = best["Gene"].astype(str).tolist()
    drugs = best["Drug"].astype(str).tolist()
    scores = best["BEST_SCORE"].values
    colors = _map_relevance(best.get("Therapeutic_Relevance", "NA"))

    fig_h = max(4.0, 0.4 * len(genes) + 1.5)
    plt.figure(figsize=(10, fig_h))
    ypos = np.arange(len(genes))

    plt.barh(ypos, scores, color=colors, alpha=0.9, edgecolor="none")

    # text labels (drug names) to the right of bars
    for y, s, d in zip(ypos, scores, drugs):
        plt.text(
            s + 0.01,
            y,
            d,
            va="center",
            fontsize=8,
        )

    plt.yticks(ypos, genes, fontsize=9)
    plt.xlabel("BEST_SCORE", fontsize=11)
    plt.title("Top hits: one best drug per gene", fontsize=13)

    # legend for Therapeutic_Relevance
    cmap = _relevance_color_map()
    handles = []
    labels = []
    for lab, col in cmap.items():
        if lab in best.get("Therapeutic_Relevance", "").unique().tolist():
            handles.append(plt.Line2D([0], [0], marker="s", color=col, linestyle=""))
            labels.append(lab)
    if handles:
        plt.legend(handles, labels, title="Therapeutic relevance", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out = fig_dir / "top_hits_barplot.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"💾 Saved barplot → {out}")


# ---------------------------------------------------------------------
# Plot 2: EC50 vs Sensitivity Index scatter
# ---------------------------------------------------------------------
def plot_ec50_vs_sensitivity(best: pd.DataFrame, fig_dir: Path):
    """
    Scatter of log10(EC50) vs Sensitivity Index (or SI_clamped).
    """
    if "EC50 (µM)" not in best.columns:
        print("⚠️ EC50 (µM) not in table; skipping EC50 vs Sensitivity plot.")
        return

    y_col = "SI_clamped" if "SI_clamped" in best.columns else "Sensitivity Index"
    if y_col not in best.columns:
        print("⚠️ No Sensitivity Index / SI_clamped column; skipping EC50 vs sensitivity plot.")
        return

    df = best.copy()
    df["EC50"] = pd.to_numeric(df["EC50 (µM)"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # require positive EC50 to take log10
    df = df[(df["EC50"] > 0) & df[y_col].notna()]
    if df.empty:
        print("⚠️ All EC50 values non-positive/NaN; skipping EC50 vs sensitivity plot.")
        return

    df["log10_EC50"] = np.log10(df["EC50"])
    colors = _map_relevance(df.get("Therapeutic_Relevance", "NA"))

    # marker by direction
    dir_values = df.get("Direction", "NA").astype(str)
    marker_map = {
        "activation_like": "^",
        "repression_like": "v",
    }

    plt.figure(figsize=(7, 5))
    for direction in sorted(dir_values.unique()):
        sub = df[dir_values == direction]
        if sub.empty:
            continue
        m = marker_map.get(direction, "o")
        plt.scatter(
            sub["log10_EC50"],
            sub[y_col],
            s=55,
            marker=m,
            c=_map_relevance(sub.get("Therapeutic_Relevance", "NA")),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
            label=direction,
        )

    # annotate a few top points (best scores)
    if "BEST_SCORE" in df.columns:
        top_annot = df.sort_values("BEST_SCORE", ascending=False).head(min(10, len(df)))
        for _, r in top_annot.iterrows():
            gene = str(r.get("Gene"))
            drug = str(r.get("Drug"))
            label = f"{gene}:{drug}"
            plt.text(
                r["log10_EC50"],
                r[y_col],
                label,
                fontsize=7,
                ha="left",
                va="bottom",
            )

    plt.xlabel("log10(EC50 [µM])")
    plt.ylabel(y_col)
    plt.title("Top hits: potency vs sensitivity")

    # legend
    plt.legend(title="Direction", loc="best", fontsize=8)

    plt.tight_layout()
    out = fig_dir / "top_hits_ec50_vs_si.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"💾 Saved EC50 vs sensitivity scatter → {out}")


# ---------------------------------------------------------------------
# Plot 3: ATE volcano
# ---------------------------------------------------------------------
def plot_ate_volcano(best: pd.DataFrame, fig_dir: Path):
    """
    Volcano plot: ATE vs -log10(ATE_p) for one best drug per gene.
    """
    if "ATE" not in best.columns or "ATE_p" not in best.columns:
        print("⚠️ ATE / ATE_p columns missing; skipping volcano plot.")
        return

    df = best.copy()
    df["ATE"] = pd.to_numeric(df["ATE"], errors="coerce")
    df["ATE_p"] = pd.to_numeric(df["ATE_p"], errors="coerce")

    df = df[(df["ATE"].notna()) & (df["ATE_p"] > 0)]
    if df.empty:
        print("⚠️ No finite ATE / positive p-values; skipping volcano plot.")
        return

    df["neglog10_p"] = -np.log10(df["ATE_p"])
    colors = _map_relevance(df.get("Therapeutic_Relevance", "NA"))

    plt.figure(figsize=(7, 5))
    plt.scatter(df["ATE"], df["neglog10_p"], s=55, c=colors, alpha=0.8, edgecolors="black", linewidths=0.3)

    # significance line at p=0.05
    sig_y = -math.log10(0.05)
    plt.axhline(sig_y, color="grey", linestyle="--", linewidth=1)
    plt.text(df["ATE"].min(), sig_y + 0.05, "p = 0.05", fontsize=8, va="bottom", color="grey")

    # annotate top hits by significance and magnitude
    df_sorted = df.sort_values(["neglog10_p", "ATE"], ascending=[False, False])
    for _, r in df_sorted.head(min(12, len(df_sorted))).iterrows():
        label = f"{r['Gene']}:{r['Drug']}"
        plt.text(
            r["ATE"],
            r["neglog10_p"],
            label,
            fontsize=7,
            ha="left",
            va="bottom",
        )

    plt.xlabel("ATE (treatment – control)")
    plt.ylabel("-log10(ATE_p)")
    plt.title("Top hits: ATE volcano")

    # custom legend for Therapeutic_Relevance
    cmap = _relevance_color_map()
    handles = []
    labels = []
    for lab, col in cmap.items():
        if lab in best.get("Therapeutic_Relevance", "").unique().tolist():
            handles.append(plt.Line2D([0], [0], marker="o", color=col, linestyle="", markersize=6))
            labels.append(lab)
    if handles:
        plt.legend(handles, labels, title="Therapeutic relevance", loc="best", fontsize=8)

    plt.tight_layout()
    out = fig_dir / "top_hits_ATE_volcano.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"💾 Saved ATE volcano → {out}")


# ---------------------------------------------------------------------
# Plot 4: hierarchical clustering dendrogram + heatmap for top drugs
# ---------------------------------------------------------------------
def _load_similarity_for_top_drugs(run_dir: Path, metric: str, top_drugs: list[str]) -> pd.DataFrame | None:
    """
    Load similarity matrix computed by run_drugsimilarity_more.py, then
    restrict to the subset of 'top_drugs'.
    """
    sim_path = run_dir / f"drug_similarity_{metric}.csv"
    if not sim_path.is_file():
        print(f"⚠️ Similarity file not found: {sim_path} – skipping dendrogram/heatmap.")
        return None

    sim = pd.read_csv(sim_path, index_col=0)
    # keep only the drugs we care about, and only those that exist in the matrix
    keep = [d for d in top_drugs if d in sim.index]
    if len(keep) < 2:
        print("⚠️ Fewer than 2 top drugs found in similarity matrix; skipping dendrogram/heatmap.")
        return None

    sim_sub = sim.loc[keep, keep]
    print(f"📊 Similarity matrix for top drugs: {sim_sub.shape[0]}×{sim_sub.shape[1]} (metric={metric})")
    return sim_sub


def plot_top_drug_dendrogram_and_heatmap(best: pd.DataFrame, run_dir: Path, fig_dir: Path, metric: str = "cosine"):
    """
    Hierarchical clustering dendrogram + optional heatmap for top drugs only.
    """
    top_drugs = sorted(best["Drug"].astype(str).unique().tolist())
    if len(top_drugs) < 2:
        print("⚠️ Need at least 2 top drugs for dendrogram; skipping.")
        return

    sim_sub = _load_similarity_for_top_drugs(run_dir, metric, top_drugs)
    if sim_sub is None:
        return

    # convert similarity (1==identical) to distance
    # clamp similarities for numerical safety
    S = sim_sub.values.astype(float)
    S = np.clip(S, -1.0, 1.0)
    D = 1.0 - S

    # condensed distance vector
    dist_vec = squareform(D, checks=False)

    Z = linkage(dist_vec, method="average")

    # --- dendrogram ---
    plt.figure(figsize=(max(6, 0.3 * len(top_drugs)), 4.5))
    dendrogram(Z, labels=sim_sub.index.tolist(), leaf_rotation=90)
    plt.title(f"Top drugs hierarchical clustering ({metric})")
    plt.ylabel("Distance (1 - similarity)")
    plt.tight_layout()
    dend_path = fig_dir / f"top_drugs_dendrogram_{metric}.png"
    plt.savefig(dend_path, dpi=200)
    plt.close()
    print(f"💾 Saved dendrogram → {dend_path}")

    # --- heatmap (optional, nicer like the example pharmaco heatmap) ---
    if _HAS_SNS:
        plt.figure(figsize=(max(6, 0.5 * len(top_drugs)), max(4, 0.35 * len(top_drugs))))
        sns.heatmap(
            sim_sub,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": f"{metric} similarity"},
        )
        plt.title(f"Top drugs similarity heatmap ({metric})")
        plt.tight_layout()
        heat_path = fig_dir / f"top_drugs_heatmap_{metric}.png"
        plt.savefig(heat_path, dpi=200)
        plt.close()
        print(f"💾 Saved heatmap → {heat_path}")
    else:
        print("ℹ️ seaborn not available; skipping top-drug heatmap.")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def run_visualizations(output_dir: Path, metric: str = "cosine"):
    """
    Main entry point for CLI.

    Parameters
    ----------
    metric : str
        Similarity metric name to use when reading drug_similarity_<metric>.csv
        (e.g. 'cosine', 'pearson', 'spearman', 'kendall').
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Using output directory: {output_dir}")
    print(f"🔍 Similarity metric for dendrogram: {metric}")

    best_table = _load_best_table(output_dir)
    best_per_gene = _select_one_best_per_gene(best_table)

    # Plots
    plot_top_hits_barplot(best_per_gene, fig_dir)
    plot_ec50_vs_sensitivity(best_per_gene, fig_dir)
    plot_ate_volcano(best_per_gene, fig_dir)
    plot_top_drug_dendrogram_and_heatmap(best_per_gene, output_dir, fig_dir, metric=metric)

    print("✅ All visualizations complete.")


# if __name__ == "__main__":
#     # Optional metric argument from CLI, e.g.
#     #   python visualization.py cosine
#     #   python visualization.py spearman
#     m = sys.argv[1] if len(sys.argv) > 1 else "cosine"
#     run_visualizations(metric=m)
