#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 Automatic Volcano Plot Generator

Automatically selects top N perturbations based on responder_rate
and generates volcano plots for each.

Inputs:
  --deg_table processed_stage4/stage4_deg_top_markers.tsv
  --ranked_table processed_stage3/stage3_perturbation_ranked.tsv
  --out_dir processed_stage4

Outputs:
  processed_stage4/stage4_volcano_<PERT>.html
  processed_stage4/stage4_volcano_<PERT>.png
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def select_top_perturbations(ranked_df: pd.DataFrame, top_n: int = 5):
    """
    Select top N perturbations dynamically based on responder_rate.
    """
    ranked_df = ranked_df.copy()

    if "responder_rate" not in ranked_df.columns:
        raise RuntimeError("responder_rate column missing in ranked table.")

    ranked_df["responder_rate"] = pd.to_numeric(
        ranked_df["responder_rate"], errors="coerce"
    )

    ranked_df = ranked_df.sort_values(
        ["responder_rate", "n_cells"],
        ascending=[False, False]
    )

    return ranked_df["perturbation_id"].head(top_n).tolist()


def build_volcano(df, perturbation, padj_thr, lfc_thr):

    df = df.copy()
    df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")
    df["pval_adj"] = pd.to_numeric(df["pval_adj"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["logfoldchange", "pval_adj"]
    )

    df["neg_log10_padj"] = -np.log10(df["pval_adj"] + 1e-300)

    df["significance"] = "NS"
    mask = (df["pval_adj"] < padj_thr) & (df["logfoldchange"].abs() > lfc_thr)
    df.loc[mask, "significance"] = "Significant"

    fig = px.scatter(
        df,
        x="logfoldchange",
        y="neg_log10_padj",
        color="significance",
        hover_data=["gene", "pval_adj", "logfoldchange"],
        title=f"Volcano Plot: {perturbation}",
        color_discrete_map={
            "Significant": "#d62728",
            "NS": "#9ca3af"
        }
    )

    fig.add_hline(y=-np.log10(padj_thr), line_dash="dash")
    fig.add_vline(x=lfc_thr, line_dash="dash")
    fig.add_vline(x=-lfc_thr, line_dash="dash")

    fig.update_layout(template="plotly_white", height=650)

    return fig


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--deg_table", required=True)
    ap.add_argument("--ranked_table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--padj_threshold", type=float, default=0.05)
    ap.add_argument("--lfc_threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deg_df = pd.read_csv(args.deg_table, sep="\t")
    ranked_df = pd.read_csv(args.ranked_table, sep="\t")

    top_perts = select_top_perturbations(ranked_df, top_n=args.top_n)

    print(f"[INFO] Auto-selected perturbations: {top_perts}")

    for pert in top_perts:

        sub = deg_df[deg_df["perturbation_id"] == pert].copy()
        if sub.empty:
            print(f"[WARN] No DEG rows for {pert}")
            continue

        fig = build_volcano(
            sub,
            pert,
            args.padj_threshold,
            args.lfc_threshold
        )

        safe_pert = safe_filename(pert)

        html_path = out_dir / f"stage4_volcano_{safe_pert}.html"
        png_path = out_dir / f"stage4_volcano_{safe_pert}.png"

        pio.write_html(fig, html_path, auto_open=False)
        fig.write_image(png_path, width=1100, height=850, scale=2)

        print(f"[OK] Generated volcano for {pert}")

    print("[DONE] All volcano plots generated.")


if __name__ == "__main__":
    main()

