#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 3 — Cell Type Annotation & Post-Mixscape Consolidation.

Integration method directories are discovered dynamically (no hardcoded timestamps).
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match, safe_read_tsv,
    discover_integration_method_dirs,
)


def load_marker_table(method_fig_path: Path) -> pd.DataFrame:
    marker_path = method_fig_path.parent / "tables" / "markers_leiden.csv"
    if not marker_path.exists():
        marker_path = method_fig_path / "tables" / "markers_leiden.csv"
    if not marker_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(marker_path)
    if "logfoldchanges" in df.columns:
        df = df.sort_values(
            by=["logfoldchanges", "pvals_adj"],
            ascending=[False, True],
        )
    return df


def build_stage3_block(ranked_df, ranked_interp, resp_hist_b64, method_blocks_html):
    ranked_table_html = df_to_scrollable_table(
        ranked_df, table_id="stage3_ranked",
        max_rows=50,
        numeric_cols=["n_cells", "responder_rate"],
        percent_cols=["responder_rate"],
        max_height_px=260,
    )

    hist_html = ""
    if resp_hist_b64:
        hist_html = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{resp_hist_b64}">
          <div class="caption">
            Figure 3A. Global distribution of responder rates across perturbations.
          </div>
        </div>
        """

    return f"""
<!-- STAGE:STAGE3 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#7c2d12,#ea580c);">
      Stage 3: Cell Type Annotation & Post-Mixscape Consolidation
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 3 integrates corrected perturbation labels with automated 
    <b>cell type annotation</b> (CellTypist + scANVI majority voting) 
    and performs post-Mixscape consolidation. This stage evaluates 
    perturbation-specific responder enrichment, clustering robustness, 
    and transcriptional identity stabilization across integration strategies.
    </p>

    <div class="subsection-title">
        Perturbation Responder Ranking
    </div>

    <div class="two-col">
      <div>
        <div class="panel-card">
          <div class="panel-title">Top Perturbations by Responder Rate (Top 50)</div>
          {ranked_table_html}
          <div class="caption">
            Table 3A. Ranked perturbations sorted by responder rate.
          </div>
          <div class="interpretation">{ranked_interp}</div>
        </div>
      </div>
      <div>
        {hist_html}
      </div>
    </div>

    <div class="subsection-title">
        Integration Strategy Comparison
    </div>

    {method_blocks_html}

  </div>
</div>
<!-- END:STAGE3 -->
"""


def build_method_block(method_name: str, method_path: Path, client):
    umap_b64 = encode_image_to_base64(method_path / "umap_final_celltype.png")
    paga_b64 = encode_image_to_base64(method_path / "paga_graph.png")

    marker_df = load_marker_table(method_path)

    marker_table_html = ""
    marker_interp = ""

    if not marker_df.empty:
        marker_table_html = df_to_scrollable_table(
            marker_df,
            table_id=f"{method_name}_markers",
            max_rows=100,
            numeric_cols=["scores", "logfoldchanges"],
            max_height_px=300,
        )

        marker_interp = llm_interpret_table(
            client,
            f"{method_name} Marker Gene Table",
            marker_df.head(50),
            "Interpret marker ranking patterns and transcriptional separation across clusters."
        )

    figs_html = ""

    if umap_b64:
        figs_html += f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{umap_b64}">
          <div class="caption">
            UMAP visualization colored by final cell type annotation.
          </div>
        </div>
        """

    if paga_b64:
        figs_html += f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{paga_b64}">
          <div class="caption">
            PAGA graph illustrating cluster connectivity structure.
          </div>
        </div>
        """

    return f"""
    <div class="method-card">
      <div class="method-title">{method_name} Integration</div>

      <div class="fig-grid">
        {figs_html}
      </div>

      <div style="margin-top:12px;">
        <div class="panel-title">Top Marker Genes (Top 100)</div>
        {marker_table_html}
        <div class="caption">
            Table 3B. Top marker genes ranked by log fold change.
        </div>
        <div class="interpretation">{marker_interp}</div>
      </div>

    </div>
    """


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 3 block from pipeline outputs in *sample_dir*."""
    stage3_dir = sample_dir / "processed_stage3"
    ranked_path = first_match(stage3_dir, "stage3_perturbation_ranked.tsv")
    if ranked_path is None:
        ranked_path = first_match(sample_dir, "**/stage3_perturbation_ranked.tsv")
    if ranked_path is None:
        return html

    ranked_df = pd.read_csv(ranked_path, sep="\t")

    ranked_interp = llm_interpret_table(
        client, "Stage 3 Ranked Perturbations", ranked_df.head(50),
        "Explain responder enrichment strength and implications for perturbation robustness."
    )

    hist_path = first_match(stage3_dir, "stage3_responder_rate_hist.png")
    resp_hist_b64 = encode_image_to_base64(hist_path) if hist_path else None

    method_dirs = discover_integration_method_dirs(stage3_dir)
    method_blocks = ""
    for method_name, method_path in method_dirs.items():
        method_blocks += build_method_block(method_name, method_path, client)

    stage3_block = build_stage3_block(ranked_df, ranked_interp, resp_hist_b64, method_blocks)
    return insert_or_replace_block(html, "STAGE3", stage3_block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 3")
