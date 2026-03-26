#!/usr/bin/env python3
"""
mdp_category_compare.py

Drop-in, client-ready "right off the bat" comparison of enrichment CATEGORY LANDSCAPES
across many diseases/cohorts produced by mdp_pipeline_3.

Core idea:
- Use *_classified outputs when present (Main_Class/Sub_Class).
- Prefer a "source label" column automatically: Ontology_Source vs library/Library
  (whichever has fewer UNKNOWN/NaN).
- Split into ontology TYPES (do not mix):
    GO Biological Process, GO Molecular Function, GO Cellular Component,
    Pathways (KEGG/Reactome/WikiPathways), Hallmarks, Other.
- Keep only significant terms (q/FDR <= 0.10).
- Cap to top 300 terms per disease per TYPE per dataset (to avoid dominance).
- Produce immediately interpretable, publication-ready plots + summary tables:
    1) Heatmap: disease × Main_Class (weighted by -log10(q))
    2) Heatmap: disease × Main_Class (counts)
    3) Similarity heatmap: disease × disease (cosine similarity of Main_Class profiles)
    4) "Theme dashboard": per TYPE, top Main_Class globally + top diseases
    5) Optional direction split for GSEA (NES>0 vs NES<0) if NES exists

Modes supported by auto-detection:
- COUNTS/DEGS: OUT_ROOT/<Disease>/...
- GL:         OUT_ROOT/GL_enrich/<Disease>/...
- GC:         OUT_ROOT/GC_enrich/<Disease>/...

Usage:
  python mdp_category_compare.py --root /mnt/d/Counts_Out_Single_Samples --out /mnt/d/Counts_Out_Single_Samples/CATEGORY_COMPARE

Or GL:
  python mdp_category_compare.py --root /mnt/d/GL_Out --out /mnt/d/GL_Out/CATEGORY_COMPARE

Outputs:
  <out>/
    tables/
      per_term_long.tsv
      per_disease_type_mainclass_summary.tsv
      disease_mainclass_weighted_matrix.tsv
      disease_mainclass_count_matrix.tsv
      disease_similarity_cosine.tsv
    plots/
      heatmap_weighted.png
      heatmap_counts.png
      similarity_cosine.png
      dashboard_top_mainclass_by_type.png
      (optional) heatmap_weighted_GSEA_UP.png / _DOWN.png

Notes:
- No seaborn; matplotlib only.
- SciPy is optional (for clustering). If not installed, uses original order.

"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# -----------------------------
# Config / constants
# -----------------------------

SIG_THRESH_DEFAULT = 0.10
CAP_TERMS_DEFAULT = 300

BAD_STRINGS = {"", "NA", "N/A", "NULL", "NONE", "UNKNOWN", "UNK", "?"}

# Type classifiers (based on library/ontology label strings)
GO_BP_PREFIX = "GO_Biological_Process"
GO_MF_PREFIX = "GO_Molecular_Function"
GO_CC_PREFIX = "GO_Cellular_Component"
KEGG_PREFIX = "KEGG"
REACTOME_PREFIX = "Reactome"
WIKIPW_PREFIX = "WikiPathways"
HALLMARK_PREFIX = "Hallmark"

# Common column candidates across modes
COL_MAIN = "Main_Class"
COL_SUB = "Sub_Class"
COL_ONTO = "Ontology_Source"

# Enrichr-like (counts/degs ORA)
COL_LIB_LOWER = "library"
COL_TERM_LOWER = "term"
COL_QVAL_LOWER = "qval"
COL_PVAL_LOWER = "pval"

# GSEA-like (GL/GC and counts GSEA)
COL_TERM_GSEA = "term"
COL_FDR_GSEA = "FDR q-val"
COL_FDR_GSEA_ALT = "FDR q-val"  # keep same; some tools vary, we'll search case-insensitively
COL_NOM_P = "NOM p-val"
COL_NES = "NES"


@dataclass
class ModeLayout:
    mode: str  # counts|degs|gl|gc
    base_dir: Path  # where disease folders live (OUT_ROOT for counts/degs; OUT_ROOT/GL_enrich for gl; OUT_ROOT/GC_enrich for gc)


# -----------------------------
# Small utilities
# -----------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_bad_label(x: object) -> bool:
    s = normalize_str(x).upper()
    return (s in BAD_STRINGS) or (s == "")


def pick_best_source_label_col(df: pd.DataFrame) -> Optional[str]:
    """
    Choose best between Ontology_Source and (library/Library) for classifying TYPE.
    Picks the one with lower bad-rate (NaN/UNKNOWN/empty).
    """
    candidates = []
    if COL_ONTO in df.columns:
        candidates.append(COL_ONTO)

    # library variants
    for c in ["Library", "library", "Gene_set", "Gene_set"]:
        if c in df.columns and c not in candidates:
            candidates.append(c)

    if not candidates:
        return None

    best = None
    best_bad = float("inf")
    n = len(df)
    if n == 0:
        return candidates[0]

    for c in candidates:
        bad = df[c].apply(is_bad_label).sum()
        bad_rate = bad / max(1, n)
        if bad_rate < best_bad:
            best_bad = bad_rate
            best = c
    return best


def col_lookup_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find column matching name case-insensitively (exact match after lower())."""
    target = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target:
            return c
    return None


def safe_neglog10(q: float) -> float:
    try:
        if q is None or pd.isna(q):
            return 0.0
        qv = float(q)
        if qv <= 0:
            return 50.0
        return -math.log10(qv)
    except Exception:
        return 0.0


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity for rows of X.
    Returns NxN.
    """
    # Normalize rows
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    Xn = X / denom
    return Xn @ Xn.T


def try_cluster_order(matrix: np.ndarray) -> Optional[np.ndarray]:
    """
    Optional hierarchical clustering order (SciPy). Returns reorder indices or None.
    """

    if matrix.shape[0] < 3:
        return None

    # Use correlation distance on rows (profile similarity)
    d = pdist(matrix, metric="correlation")
    Z = linkage(d, method="average")
    order = leaves_list(Z)
    return order


# -----------------------------
# Ontology TYPE classification
# -----------------------------

def classify_type(source_label: str) -> str:
    """
    Map a Library/Ontology_Source string into a TYPE bucket.
    """
    s = normalize_str(source_label)
    if not s:
        return "Other"

    if s.startswith(GO_BP_PREFIX):
        return "GO_Biological_Process"
    if s.startswith(GO_MF_PREFIX):
        return "GO_Molecular_Function"
    if s.startswith(GO_CC_PREFIX):
        return "GO_Cellular_Component"

    # Pathway databases
    if s.startswith(KEGG_PREFIX) or s.startswith(REACTOME_PREFIX) or s.startswith(WIKIPW_PREFIX):
        return "Pathways"

    if s.startswith(HALLMARK_PREFIX):
        return "Hallmarks"

    # Common other pathway libs may still be pathways; leave as Other unless you extend rules.
    return "Other"


# -----------------------------
# Discover mode + diseases
# -----------------------------

def detect_layout(root: Path) -> ModeLayout:
    """
    Determine where disease folders live.
    Priority:
      - GL_enrich/
      - GC_enrich/
      - else treat root itself as counts/degs-style
    """
    gl = root / "GL_enrich"
    if gl.exists() and gl.is_dir():
        return ModeLayout(mode="gl", base_dir=gl)

    gc = root / "GC_enrich"
    if gc.exists() and gc.is_dir():
        return ModeLayout(mode="gc", base_dir=gc)

    # counts/degs: diseases under root
    return ModeLayout(mode="counts_or_degs", base_dir=root)


def list_disease_dirs(base_dir: Path) -> List[Path]:
    """
    Disease folders = directories that contain at least one expected enrichment file.
    Exclude technical/global folders.
    """
    skip = {"baseline_consensus", "comparison", "results", "OmniPath_cache", "jsons_all_folder", "CATEGORY_COMPARE"}
    out: List[Path] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name in skip:
            continue

        # Heuristic: has gsea_prerank or core enrich or overlap JSON
        if (p / "gsea_prerank.tsv").exists() or (p / "gsea_prerank_classified.tsv").exists():
            out.append(p)
            continue
        if (p / "core_enrich_up.csv").exists() or (p / "core_enrich_up_classified.csv").exists():
            out.append(p)
            continue
        if (p / "overlap" / "pathway_entity_overlap.json").exists():
            out.append(p)
            continue

    return out


# -----------------------------
# Load + standardize per-disease classified terms
# -----------------------------

def load_classified_terms_for_disease(
    disease_dir: Path,
    sig_thresh: float,
    cap_terms: int,
) -> pd.DataFrame:
    """
    Returns LONG table of significant terms for a disease across:
      - GSEA (preferred: gsea_prerank_classified.tsv, else gsea_prerank.tsv)
      - Core ORA up/down (preferred *_classified.csv, else *.csv)
    Columns returned (standard):
      Disease, Dataset (GSEA|CORE_UP|CORE_DOWN), Term, Main_Class, Sub_Class,
      SourceLabelCol, SourceLabelVal, Type, QValue, NES (optional)
    """
    disease = disease_dir.name
    frames: List[pd.DataFrame] = []

    # ---- GSEA
    gsea_path = None
    for name in ["gsea_prerank_classified.tsv", "gsea_prerank.tsv"]:
        p = disease_dir / name
        if p.exists():
            gsea_path = p
            break

    if gsea_path is not None:
        df = read_table(gsea_path)

        # find key cols
        term_c = col_lookup_case_insensitive(df, COL_TERM_GSEA) or "Term"
        fdr_c = col_lookup_case_insensitive(df, COL_FDR_GSEA) or col_lookup_case_insensitive(df, "FDR q-val")
        nom_c = col_lookup_case_insensitive(df, COL_NOM_P)
        nes_c = col_lookup_case_insensitive(df, COL_NES)

        main_c = COL_MAIN if COL_MAIN in df.columns else None
        sub_c = COL_SUB if COL_SUB in df.columns else None

        # If classification missing, still proceed but Main_Class will be "Unclassified"
        if main_c is None:
            df[COL_MAIN] = "Unclassified"
            main_c = COL_MAIN
        if sub_c is None:
            df[COL_SUB] = "Unclassified"
            sub_c = COL_SUB

        # significance col
        qcol = fdr_c or nom_c
        if qcol is None or qcol not in df.columns:
            # no significance column -> skip
            pass
        else:
            df = df.copy()
            df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
            df = df.dropna(subset=[qcol])

            df_sig = df[df[qcol] <= sig_thresh].copy()
            if not df_sig.empty:
                # cap
                df_sig = df_sig.sort_values(by=qcol, ascending=True).head(cap_terms)

                source_label_col = pick_best_source_label_col(df_sig) or COL_ONTO
                if source_label_col not in df_sig.columns:
                    df_sig[source_label_col] = "UNKNOWN"

                out = pd.DataFrame({
                    "Disease": disease,
                    "Dataset": "GSEA",
                    "Term": df_sig[term_c].astype(str),
                    "Main_Class": df_sig[main_c].astype(str),
                    "Sub_Class": df_sig[sub_c].astype(str),
                    "SourceLabelCol": source_label_col,
                    "SourceLabelVal": df_sig[source_label_col].astype(str),
                    "QValue": df_sig[qcol].astype(float),
                    "NES": pd.to_numeric(df_sig[nes_c], errors="coerce") if (nes_c and nes_c in df_sig.columns) else np.nan,
                })
                out["Type"] = out["SourceLabelVal"].apply(classify_type)
                frames.append(out)

    # ---- CORE ORA UP/DOWN (counts/degs only typically, but harmless to try)
    for dataset, fname_list in [
        ("CORE_UP", ["core_enrich_up_classified.csv", "core_enrich_up.csv"]),
        ("CORE_DOWN", ["core_enrich_down_classified.csv", "core_enrich_down.csv"]),
    ]:
        path = None
        for fn in fname_list:
            p = disease_dir / fn
            if p.exists():
                path = p
                break
        if path is None:
            continue

        df = read_table(path)

        # expected columns (counts/degs)
        term_c = col_lookup_case_insensitive(df, COL_TERM_LOWER) or "Term"
        q_c = col_lookup_case_insensitive(df, COL_QVAL_LOWER) or col_lookup_case_insensitive(df, "FDR_q") or col_lookup_case_insensitive(df, "FDR_q")
        p_c = col_lookup_case_insensitive(df, COL_PVAL_LOWER) or col_lookup_case_insensitive(df, "P-value")

        main_c = COL_MAIN if COL_MAIN in df.columns else None
        sub_c = COL_SUB if COL_SUB in df.columns else None

        if main_c is None:
            df[COL_MAIN] = "Unclassified"
            main_c = COL_MAIN
        if sub_c is None:
            df[COL_SUB] = "Unclassified"
            sub_c = COL_SUB

        qcol = q_c or p_c
        if qcol is None or qcol not in df.columns:
            continue

        df = df.copy()
        df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
        df = df.dropna(subset=[qcol])
        df_sig = df[df[qcol] <= sig_thresh].copy()
        if df_sig.empty:
            continue

        df_sig = df_sig.sort_values(by=qcol, ascending=True).head(cap_terms)
        source_label_col = pick_best_source_label_col(df_sig) or "library"
        if source_label_col not in df_sig.columns:
            df_sig[source_label_col] = "UNKNOWN"

        out = pd.DataFrame({
            "Disease": disease,
            "Dataset": dataset,
            "Term": df_sig[term_c].astype(str),
            "Main_Class": df_sig[main_c].astype(str),
            "Sub_Class": df_sig[sub_c].astype(str),
            "SourceLabelCol": source_label_col,
            "SourceLabelVal": df_sig[source_label_col].astype(str),
            "QValue": df_sig[qcol].astype(float),
            "NES": np.nan,
        })
        out["Type"] = out["SourceLabelVal"].apply(classify_type)
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=[
            "Disease", "Dataset", "Term", "Main_Class", "Sub_Class",
            "SourceLabelCol", "SourceLabelVal", "Type", "QValue", "NES"
        ])

    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Summaries and matrices
# -----------------------------

def build_mainclass_matrices(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - weighted matrix: sum(-log10(QValue)) by Disease × Main_Class
      - count matrix: number of terms by Disease × Main_Class
    Uses ALL TYPES & DATASETS together (holistic picture).
    You can later split by Type/Dataset if desired.
    """
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = long_df.copy()
    df["Weight"] = df["QValue"].apply(safe_neglog10)

    weighted = (
        df.groupby(["Disease", "Main_Class"], as_index=False)["Weight"]
        .sum()
        .pivot(index="Disease", columns="Main_Class", values="Weight")
        .fillna(0.0)
    )

    counts = (
        df.groupby(["Disease", "Main_Class"], as_index=False)["Term"]
        .count()
        .rename(columns={"Term": "Count"})
        .pivot(index="Disease", columns="Main_Class", values="Count")
        .fillna(0)
        .astype(int)
    )

    # stable column order: by total weight
    col_order = weighted.sum(axis=0).sort_values(ascending=False).index.tolist()
    weighted = weighted[col_order]
    counts = counts[col_order]
    return weighted, counts


def build_type_mainclass_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize by Disease × Type × Main_Class:
      - n_terms
      - sum_weight = sum(-log10(q))
      - mean_q
      - (if NES exists) mean_NES
    """
    if long_df.empty:
        return pd.DataFrame()

    df = long_df.copy()
    df["Weight"] = df["QValue"].apply(safe_neglog10)

    out = df.groupby(["Disease", "Type", "Main_Class"], as_index=False).agg(
        n_terms=("Term", "count"),
        sum_weight=("Weight", "sum"),
        mean_q=("QValue", "mean"),
        mean_nes=("NES", "mean"),
    )
    out = out.sort_values(["Type", "sum_weight"], ascending=[True, False])
    return out


# -----------------------------
# Plotting helpers (matplotlib only)
# -----------------------------

def plot_heatmap(
    mat: pd.DataFrame,
    title: str,
    out_png: Path,
    cluster: bool = True,
    xlabel: str = "Main_Class",
    ylabel: str = "Disease",
) -> None:
    if mat.empty:
        eprint(f"[plot] Skipping heatmap (empty): {title}")
        return

    data = mat.values.astype(float)
    diseases = mat.index.tolist()
    classes = mat.columns.tolist()

    # optional clustering on rows (diseases)
    row_order = np.arange(len(diseases))
    if cluster:
        order = try_cluster_order(data)
        if order is not None:
            row_order = order

    data = data[row_order, :]
    diseases = [diseases[i] for i in row_order]

    fig, ax = plt.subplots(figsize=(max(10, 0.22 * len(classes)), max(6, 0.25 * len(diseases))), dpi=300)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels(diseases, fontsize=7)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_similarity_heatmap(
    sim: pd.DataFrame,
    title: str,
    out_png: Path,
    cluster: bool = True,
) -> None:
    if sim.empty:
        eprint("[plot] Skipping similarity heatmap (empty).")
        return

    data = sim.values.astype(float)
    labels = sim.index.tolist()

    order = np.arange(len(labels))
    if cluster:
        o = try_cluster_order(data)
        if o is not None:
            order = o

    data = data[order, :][:, order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(labels)), max(7, 0.28 * len(labels))), dpi=300)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_dashboard_top_mainclass_by_type(
    summary: pd.DataFrame,
    out_png: Path,
    top_k: int = 10,
) -> None:
    """
    One compact client slide-style dashboard:
    For each Type, plot top K Main_Class by global sum_weight.
    """
    if summary.empty:
        eprint("[plot] Skipping dashboard (empty).")
        return

    # global aggregation
    g = summary.groupby(["Type", "Main_Class"], as_index=False)["sum_weight"].sum()
    types = [t for t in ["GO_Biological_Process", "GO_Molecular_Function", "GO_Cellular_Component", "Pathways", "Hallmarks", "Other"]
             if t in set(g["Type"])]

    nrows = len(types)
    fig_h = max(6, 2.1 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, fig_h), dpi=300)

    if nrows == 1:
        axes = [axes]

    for ax, t in zip(axes, types):
        sub = g[g["Type"] == t].sort_values("sum_weight", ascending=False).head(top_k)
        ax.barh(sub["Main_Class"].astype(str)[::-1], sub["sum_weight"][::-1])
        ax.set_title(f"{t}: top {top_k} Main_Class (global sum(-log10(q)))")
        ax.set_xlabel("sum(-log10(q))")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Client-ready holistic category comparison across mdp_pipeline_3 disease outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", required=True, help="Run root (COUNTS/DEGS) or OUT_ROOT (GL/GC).")
    ap.add_argument("--out", required=True, help="Output folder for plots/tables.")
    ap.add_argument("--sig", type=float, default=SIG_THRESH_DEFAULT, help="Significance threshold (q/FDR <= sig).")
    ap.add_argument("--cap", type=int, default=CAP_TERMS_DEFAULT, help="Cap per disease per Type per dataset.")
    ap.add_argument("--no-cluster", action="store_true", help="Disable hierarchical clustering (if SciPy available).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    ensure_dir(out)
    tables_dir = ensure_dir(out / "tables")
    plots_dir = ensure_dir(out / "plots")

    layout = detect_layout(root)
    print(f"[info] Detected layout mode={layout.mode} base_dir={layout.base_dir}")

    disease_dirs = list_disease_dirs(layout.base_dir)
    if not disease_dirs:
        raise SystemExit(f"No disease folders detected under: {layout.base_dir}")

    print(f"[info] Found {len(disease_dirs)} disease folders.")

    all_long: List[pd.DataFrame] = []
    for ddir in disease_dirs:
        df_long = load_classified_terms_for_disease(
            disease_dir=ddir,
            sig_thresh=args.sig,
            cap_terms=args.cap,
        )
        if df_long.empty:
            eprint(f"[warn] No significant classified terms for disease: {ddir.name}")
        all_long.append(df_long)

    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    long_path = tables_dir / "per_term_long.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)
    print(f"[write] {long_path}")

    summary = build_type_mainclass_summary(long_df)
    sum_path = tables_dir / "per_disease_type_mainclass_summary.tsv"
    summary.to_csv(sum_path, sep="\t", index=False)
    print(f"[write] {sum_path}")

    weighted, counts = build_mainclass_matrices(long_df)

    w_path = tables_dir / "disease_mainclass_weighted_matrix.tsv"
    c_path = tables_dir / "disease_mainclass_count_matrix.tsv"
    weighted.to_csv(w_path, sep="\t")
    counts.to_csv(c_path, sep="\t")
    print(f"[write] {w_path}")
    print(f"[write] {c_path}")

    # Similarity (cosine) on weighted profiles
    if not weighted.empty:
        sim = cosine_similarity_matrix(weighted.values.astype(float))
        sim_df = pd.DataFrame(sim, index=weighted.index, columns=weighted.index)
    else:
        sim_df = pd.DataFrame()

    sim_path = tables_dir / "disease_similarity_cosine.tsv"
    sim_df.to_csv(sim_path, sep="\t")
    print(f"[write] {sim_path}")

    cluster = not args.no_cluster

    # Plots: holistic
    plot_heatmap(
        mat=weighted,
        title=f"Disease × Main_Class (sum(-log10(q)))  [sig≤{args.sig}, cap≤{args.cap}]",
        out_png=plots_dir / "heatmap_weighted.png",
        cluster=cluster,
    )
    plot_heatmap(
        mat=counts,
        title=f"Disease × Main_Class (count of significant terms)  [sig≤{args.sig}, cap≤{args.cap}]",
        out_png=plots_dir / "heatmap_counts.png",
        cluster=cluster,
    )
    plot_similarity_heatmap(
        sim=sim_df,
        title="Disease × Disease similarity (cosine of Main_Class weighted profiles)",
        out_png=plots_dir / "similarity_cosine.png",
        cluster=cluster,
    )
    plot_dashboard_top_mainclass_by_type(
        summary=summary,
        out_png=plots_dir / "dashboard_top_mainclass_by_type.png",
        top_k=12,
    )

    # Optional: direction split for GSEA (if NES present)
    if not long_df.empty and "NES" in long_df.columns and long_df["NES"].notna().any():
        gsea = long_df[long_df["Dataset"] == "GSEA"].copy()
        if not gsea.empty:
            gsea["Direction"] = np.where(gsea["NES"] >= 0, "UP", "DOWN")
            for direction in ["UP", "DOWN"]:
                sub = gsea[gsea["Direction"] == direction].copy()
                w_sub, _ = build_mainclass_matrices(sub)
                plot_heatmap(
                    mat=w_sub,
                    title=f"GSEA {direction}: Disease × Main_Class (sum(-log10(FDR)))",
                    out_png=plots_dir / f"heatmap_weighted_GSEA_{direction}.png",
                    cluster=cluster,
                )

    print(f"[done] Outputs written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
