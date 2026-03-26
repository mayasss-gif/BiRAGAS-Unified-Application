#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, json

from .common_io import ensure_dir, read_table, first_existing, col_ci
from .common_filters import (
    pick_significance_col,
    apply_sig_and_cap,
    pick_best_source_label_col,
    ensure_class_cols,
    classify_type,
    safe_neglog10,
    TYPE_ORDER,
)
from .common_scoring import cosine_similarity_matrix, try_cluster_order
from .common_plotting import plot_heatmap, plot_barh


def _detect_base(root: Path) -> Path:
    gl = root / "GL_enrich"
    if gl.exists() and gl.is_dir():
        return gl
    gc = root / "GC_enrich"
    if gc.exists() and gc.is_dir():
        return gc
    return root


def _list_diseases(base: Path) -> List[Path]:
    skip = {"baseline_consensus", "comparison", "results", "OmniPath_cache", "jsons_all_folder", "CATEGORY_COMPARE"}
    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name in skip:
            continue
        if (p / "gsea_prerank_classified.tsv").exists() or (p / "gsea_prerank.tsv").exists():
            out.append(p)
            continue
        if (p / "core_enrich_up_classified.csv").exists() or (p / "core_enrich_up.csv").exists():
            out.append(p)
            continue
        if (p / "overlap" / "pathway_entity_overlap.json").exists():
            out.append(p)
            continue
    return out


def _load_one_disease(disease_dir: Path, sig: float, cap: int) -> pd.DataFrame:
    disease = disease_dir.name
    frames: List[pd.DataFrame] = []

    # GSEA
    gsea = first_existing(disease_dir / "gsea_prerank_classified.tsv", disease_dir / "gsea_prerank.tsv")
    if gsea is not None:
        df = read_table(gsea)
        df, main_c, sub_c = ensure_class_cols(df)
        sig_c = pick_significance_col(df)
        term_c = col_ci(df, "term") or col_ci(df, "pathway") or col_ci(df, "NAME") or "Term"
        if sig_c and term_c in df.columns:
            df_sig = apply_sig_and_cap(df, sig_c, sig, cap)
            if not df_sig.empty:
                src_col = pick_best_source_label_col(df_sig) or "Ontology_Source"
                if src_col not in df_sig.columns:
                    df_sig[src_col] = "UNKNOWN"
                nes_c = col_ci(df_sig, "NES")
                out = pd.DataFrame({
                    "Disease": disease,
                    "Dataset": "GSEA",
                    "Term": df_sig[term_c].astype(str),
                    "Main_Class": df_sig[main_c].astype(str),
                    "Sub_Class": df_sig[sub_c].astype(str),
                    "SourceLabel": df_sig[src_col].astype(str),
                    "QValue": pd.to_numeric(df_sig[sig_c], errors="coerce"),
                    "NES": pd.to_numeric(df_sig[nes_c], errors="coerce") if nes_c else np.nan,
                })
                out["Type"] = out["SourceLabel"].apply(classify_type)
                frames.append(out)

    # ORA core up/down (counts/degs)
    for label, fn in [
        ("CORE_UP", first_existing(disease_dir / "core_enrich_up_classified.csv", disease_dir / "core_enrich_up.csv")),
        ("CORE_DOWN", first_existing(disease_dir / "core_enrich_down_classified.csv", disease_dir / "core_enrich_down.csv")),
    ]:
        if fn is None:
            continue
        df = read_table(fn)
        df, main_c, sub_c = ensure_class_cols(df)
        sig_c = pick_significance_col(df)
        term_c = col_ci(df, "term") or col_ci(df, "pathway") or "term" if "term" in df.columns else "Term"
        if sig_c and term_c in df.columns:
            df_sig = apply_sig_and_cap(df, sig_c, sig, cap)
            if not df_sig.empty:
                src_col = pick_best_source_label_col(df_sig) or "library"
                if src_col not in df_sig.columns:
                    df_sig[src_col] = "UNKNOWN"
                out = pd.DataFrame({
                    "Disease": disease,
                    "Dataset": label,
                    "Term": df_sig[term_c].astype(str),
                    "Main_Class": df_sig[main_c].astype(str),
                    "Sub_Class": df_sig[sub_c].astype(str),
                    "SourceLabel": df_sig[src_col].astype(str),
                    "QValue": pd.to_numeric(df_sig[sig_c], errors="coerce"),
                    "NES": np.nan,
                })
                out["Type"] = out["SourceLabel"].apply(classify_type)
                frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["Disease", "Dataset", "Term", "Main_Class", "Sub_Class", "SourceLabel", "Type", "QValue", "NES"])
    return pd.concat(frames, ignore_index=True)


def run_category_landscape_compare(root: str, out: str, sig: float = 0.1, cap: int = 300, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    base = _detect_base(root_p)
    diseases = _list_diseases(base)
    if not diseases:
        raise SystemExit(f"No disease folders detected under: {base}")

    long_df = pd.concat([_load_one_disease(d, sig=sig, cap=cap) for d in diseases], ignore_index=True)
    long_path = tables / "per_term_long.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)

    if long_df.empty:
        return {"out": str(out_p), "per_term_long": str(long_path)}

    # Weight and matrices
    df = long_df.copy()
    df["Weight"] = df["QValue"].apply(safe_neglog10)

    weighted = df.groupby(["Disease", "Main_Class"], as_index=False)["Weight"].sum().pivot(index="Disease", columns="Main_Class", values="Weight").fillna(0.0)
    counts = df.groupby(["Disease", "Main_Class"], as_index=False)["Term"].count().rename(columns={"Term": "Count"}).pivot(index="Disease", columns="Main_Class", values="Count").fillna(0).astype(int)

    # stable col order by global weight
    col_order = weighted.sum(axis=0).sort_values(ascending=False).index.tolist()
    weighted = weighted[col_order]
    counts = counts[col_order]

    weighted_path = tables / "disease_mainclass_weighted.tsv"
    counts_path = tables / "disease_mainclass_counts.tsv"
    weighted.to_csv(weighted_path, sep="\t")
    counts.to_csv(counts_path, sep="\t")

    # Similarity
    sim = cosine_similarity_matrix(weighted.values.astype(float))
    sim_df = pd.DataFrame(sim, index=weighted.index, columns=weighted.index)
    sim_path = tables / "disease_similarity_cosine.tsv"
    sim_df.to_csv(sim_path, sep="\t")

    order = None
    if cluster:
        order = try_cluster_order(weighted.values.astype(float))

    # plots
    plot_heatmap(weighted, plots / "heatmap_weighted.png",
                 f"Disease × Main_Class (sum(-log10(q)))  [sig≤{sig}, cap≤{cap}]", cluster_order=order)
    plot_heatmap(counts, plots / "heatmap_counts.png",
                 f"Disease × Main_Class (count significant)  [sig≤{sig}, cap≤{cap}]", cluster_order=order)

    plot_heatmap(sim_df, plots / "similarity_cosine.png",
                 "Disease × Disease similarity (cosine of Main_Class weighted profiles)", cluster_order=order, vmin=-1.0, vmax=1.0)

    # per-type top themes
    g = df.groupby(["Type", "Main_Class"], as_index=False)["Weight"].sum()
    dash_png = plots / "dashboard_top_mainclass_by_type.png"
    # Build one figure with stacked panels (simple and readable)
    
    types = [t for t in TYPE_ORDER if t in set(g["Type"])]
    n = max(1, len(types))
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, max(5, 2.1 * n)), dpi=300)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, types):
        sub = g[g["Type"] == t].sort_values("Weight", ascending=False).head(12)
        ax.barh(sub["Main_Class"].astype(str)[::-1], sub["Weight"][::-1])
        ax.set_title(f"{t}: top Main_Class (global sum(-log10(q)))")
        ax.set_xlabel("sum(-log10(q))")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(dash_png)
    plt.close(fig)

    return {
        "out": str(out_p),
        "per_term_long": str(long_path),
        "weighted_matrix": str(weighted_path),
        "count_matrix": str(counts_path),
        "similarity_table": str(sim_path),
        "plot_weighted": str(plots / "heatmap_weighted.png"),
        "plot_counts": str(plots / "heatmap_counts.png"),
        "plot_similarity": str(plots / "similarity_cosine.png"),
        "plot_dashboard": str(dash_png),
    }


def main() -> int:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sig", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    res = run_category_landscape_compare(args.root, args.out, sig=args.sig, cap=args.cap, cluster=(not args.no_cluster))

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
