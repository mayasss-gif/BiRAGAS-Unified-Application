#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 10-11 — Latent Representation and UI Exploration.

Dynamically discovers cluster, enrichment, and UI card tables from processed_stage11/.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match,
)


def build_stage_block(cluster_df, cluster_interp,
                      enrich_df, enrich_interp,
                      cards_df, cards_interp,
                      fig_cluster_b64, fig_label_b64):
    cluster_table = df_to_scrollable_table(
        cluster_df, table_id="stage11_cluster_summary",
        max_rows=50, max_height_px=260,
    )
    enrich_table = df_to_scrollable_table(
        enrich_df, table_id="stage11_enrichment",
        max_rows=50, max_height_px=260,
    )
    cards_table = df_to_scrollable_table(
        cards_df, table_id="stage11_cards",
        max_rows=50, max_height_px=260,
    )

    fig1 = f"""
    <div class="fig-card">
      <img class="figure-img" src="data:image/png;base64,{fig_cluster_b64}">
      <div class="caption">Figure 10A. UMAP visualization of latent clusters.</div>
    </div>
    """ if fig_cluster_b64 else ""

    fig2 = f"""
    <div class="fig-card">
      <img class="figure-img" src="data:image/png;base64,{fig_label_b64}">
      <div class="caption">Figure 10B. UMAP colored by predicted perturbation label.</div>
    </div>
    """ if fig_label_b64 else ""

    return f"""
<!-- STAGE:STAGE10_11 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#7c3aed,#9333ea);">
      Stage 10\u201311: Latent Representation and UI Exploration
  </div>

  <div class="stage-content">

    <p class="stage-description">
    This stage constructs a latent manifold from high-confidence cells and identifies 
    transcriptionally coherent clusters using Leiden community detection. 
    Cluster-level summaries and perturbation enrichment metrics are used to 
    generate user-facing exploration cards for interactive interpretation.
    </p>

    <div class="two-col">

      <div class="panel-card">
        <div class="panel-title">Latent Cluster Summary (Top 50)</div>
        {cluster_table}
        <div class="caption">
            Table 10A. Cluster sizes and dominant predicted perturbation labels.
        </div>
        <div class="interpretation">{cluster_interp}</div>
      </div>

      <div>
        {fig1}
      </div>

    </div>

    <div class="two-col" style="margin-top:35px;">

      <div class="panel-card">
        <div class="panel-title">Perturbation \u2192 Cluster Enrichment (Top 50)</div>
        {enrich_table}
        <div class="caption">
            Table 10B. Perturbations enriched in specific latent clusters.
        </div>
        <div class="interpretation">{enrich_interp}</div>
      </div>

      <div>
        {fig2}
      </div>

    </div>

    <div style="margin-top:35px;">
      <div class="panel-card">
        <div class="panel-title">UI Cluster Cards (Top 50)</div>
        {cards_table}
        <div class="caption">
            Table 10C. User-facing cluster summaries with top gene markers.
        </div>
        <div class="interpretation">{cards_interp}</div>
      </div>
    </div>

  </div>
</div>
<!-- END:STAGE10_11 -->
"""


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 10-11 block from pipeline outputs in *sample_dir*."""
    s11 = sample_dir / "processed_stage11"

    cluster_path = first_match(s11, "**/stage11_cluster_summary.tsv")
    if cluster_path is None:
        cluster_path = first_match(sample_dir, "**/stage11_cluster_summary.tsv")

    enrich_path = first_match(s11, "**/stage11_perturbation_cluster_enrichment.tsv")
    if enrich_path is None:
        enrich_path = first_match(sample_dir, "**/stage11_perturbation_cluster_enrichment.tsv")

    cards_path = first_match(s11, "**/stage11_ui_cards.tsv")
    if cards_path is None:
        cards_path = first_match(sample_dir, "**/stage11_ui_cards.tsv")

    if cluster_path is None and enrich_path is None and cards_path is None:
        return html

    cluster_df = pd.read_csv(cluster_path, sep="\t") if cluster_path else pd.DataFrame()
    enrich_df = pd.read_csv(enrich_path, sep="\t") if enrich_path else pd.DataFrame()
    cards_df = pd.read_csv(cards_path, sep="\t") if cards_path else pd.DataFrame()

    cluster_interp = llm_interpret_table(
        client, "Stage 10 Cluster Summary", cluster_df.head(50),
        "Interpret cluster size heterogeneity and dominant perturbation identity."
    )
    enrich_interp = llm_interpret_table(
        client, "Stage 10 Perturbation Cluster Enrichment", enrich_df.head(50),
        "Explain perturbation-specific cluster localization patterns."
    )
    cards_interp = llm_interpret_table(
        client, "Stage 10 UI Cards", cards_df.head(50),
        "Summarize how cluster summaries improve interpretability of perturbation programs."
    )

    fig_cluster = first_match(s11, "**/stage11_umap_clusters.png")
    if fig_cluster is None:
        fig_cluster = first_match(sample_dir, "**/stage11_umap_clusters.png")

    fig_label = first_match(s11, "**/stage11_umap_predlabel.png")
    if fig_label is None:
        fig_label = first_match(sample_dir, "**/stage11_umap_predlabel.png")

    fig_cluster_b64 = encode_image_to_base64(fig_cluster) if fig_cluster else None
    fig_label_b64 = encode_image_to_base64(fig_label) if fig_label else None

    block = build_stage_block(
        cluster_df, cluster_interp,
        enrich_df, enrich_interp,
        cards_df, cards_interp,
        fig_cluster_b64, fig_label_b64,
    )
    return insert_or_replace_block(html, "STAGE10_11", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 10-11")
