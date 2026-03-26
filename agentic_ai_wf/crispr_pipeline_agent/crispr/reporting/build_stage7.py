#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 7 — Prediction and Ranking.

Dynamically discovers ranking and cell-prediction tables.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match,
)


def make_confidence_hist(cell_df, out_png: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(cell_df["pred_confidence"], bins=40)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Cells")
    plt.title("Confidence Distribution")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def make_top_conf_barplot(rank_df, out_png: Path):
    top = rank_df.head(15)
    plt.figure(figsize=(6, 4))
    plt.barh(top["pred_perturbation_id"], top["mean_conf"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Confidence")
    plt.title("Top 15 Perturbations")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def build_stage7_block(rank_df, rank_interp, cell_df, cell_interp,
                       hist_b64, bar_b64):
    rank_table = df_to_scrollable_table(
        rank_df.head(50), "stage7_rank",
        numeric_cols=["n_cells", "mean_conf"],
        max_height_px=260,
    )

    cell_table = df_to_scrollable_table(
        cell_df.head(50), "stage7_cells",
        numeric_cols=["pred_confidence"],
        max_height_px=260,
    )

    hist_html = f"""
    <div class="fig-card">
        <img class="figure-img" src="data:image/png;base64,{hist_b64}">
        <div class="caption">Figure 7A. Distribution of per-cell prediction confidence.</div>
    </div>
    """ if hist_b64 else ""

    bar_html = f"""
    <div class="fig-card">
        <img class="figure-img" src="data:image/png;base64,{bar_b64}">
        <div class="caption">Figure 7B. Top perturbations ranked by mean prediction confidence.</div>
    </div>
    """ if bar_b64 else ""

    return f"""
<!-- STAGE:STAGE7 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#7c3aed,#a78bfa);">
      Stage 7: Prediction and Ranking
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 7 applies the best-performing model to generate per-cell perturbation predictions
    and associated confidence scores. Predictions are aggregated to produce a perturbation-level
    ranking by mean confidence, enabling prioritization of high-certainty perturbations.
    </p>

    {hist_html}

    <div class="panel-grid" style="grid-template-columns: 1fr 1fr 1fr;">

      <div class="panel-card">
        <div class="panel-title">Perturbation Ranking (Top 50)</div>
        {rank_table}
        <div class="caption">Table 7A. Perturbation-level mean confidence.</div>
        <div class="interpretation">{rank_interp}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Per-Cell Predictions (Top 50)</div>
        {cell_table}
        <div class="caption">Table 7B. Example per-cell predictions.</div>
        <div class="interpretation">{cell_interp}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Top Confidence Overview</div>
        {bar_html}
      </div>

    </div>

  </div>
</div>
<!-- END:STAGE7 -->
"""


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 7 block from pipeline outputs in *sample_dir*."""
    rank_path = first_match(sample_dir, "stage7_perturbation_ranked_by_confidence.tsv")
    cell_path = first_match(sample_dir, "stage7_cell_predictions.tsv")

    if rank_path is None and cell_path is None:
        return html

    rank_df = pd.read_csv(rank_path, sep="\t") if rank_path else pd.DataFrame()
    cell_df = pd.read_csv(cell_path, sep="\t") if cell_path else pd.DataFrame()

    rank_interp = llm_interpret_table(
        client, "Stage 7 Perturbation Ranking", rank_df.head(30),
        "Explain confidence distribution and top perturbations."
    )
    cell_interp = llm_interpret_table(
        client, "Stage 7 Per Cell Predictions", cell_df.head(30),
        "Comment on prediction confidence variation."
    )

    hist_png = sample_dir / "stage7_confidence_hist.png"
    bar_png = sample_dir / "stage7_top_conf_barplot.png"

    try:
        if not cell_df.empty and "pred_confidence" in cell_df.columns:
            make_confidence_hist(cell_df, hist_png)
    except Exception:
        pass

    try:
        if not rank_df.empty and "pred_perturbation_id" in rank_df.columns:
            make_top_conf_barplot(rank_df, bar_png)
    except Exception:
        pass

    hist_b64 = encode_image_to_base64(hist_png)
    bar_b64 = encode_image_to_base64(bar_png)

    block = build_stage7_block(
        rank_df, rank_interp, cell_df, cell_interp,
        hist_b64, bar_b64,
    )
    return insert_or_replace_block(html, "STAGE7", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 7")
