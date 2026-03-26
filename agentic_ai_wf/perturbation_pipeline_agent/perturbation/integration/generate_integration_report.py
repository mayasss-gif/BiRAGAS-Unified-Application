from __future__ import annotations
import argparse
import os
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from string import Template

import pandas as pd




def rel(from_path: Path, to_path: Path) -> str:
    """Relative path from output HTML directory to an asset (plots)."""
    return os.path.relpath(to_path, start=from_path)


def file_to_data_uri(path: Path, mime_override: str | None = None) -> str:
    """Embed a file as a data URI (portable single-file HTML)."""
    if not path or not Path(path).is_file():
        return ""
    if mime_override:
        mime = mime_override
    else:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def img_to_data_uri(img_path: Path) -> str:
    """Embed an image as a data URI."""
    # force image/png fallback
    return file_to_data_uri(img_path, mime_override=None)


def html_to_data_uri(html_path: Path) -> str:
    """Embed an HTML file as a clickable data URI link."""
    # Using text/html ensures browsers open it as HTML in a new tab.
    return file_to_data_uri(html_path, mime_override="text/html")


# -----------------------------
# Tables
# -----------------------------
def build_tables(outputs_dir: Path):
    # --- L1000 effect strength per gene ---
    eff_path = outputs_dir / "EffectStrength_by_gene.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path)
    else:
        eff = pd.DataFrame(columns=["Gene", "HGNC_Gene", "n_drug_pairs", "max_SensitivityIndex",
                                    "mean_SensitivityIndex", "best_EC50", "best_R2", "best_Drug",
                                    "effect_score_0_100"])
    if "effect_score_0_100" in eff.columns:
        eff = eff.sort_values("effect_score_0_100", ascending=False)

    effect_cols = [
        "Gene",
        "HGNC_Gene",
        "n_drug_pairs",
        "max_SensitivityIndex",
        "mean_SensitivityIndex",
        "best_EC50",
        "best_R2",
        "best_Drug",
        "effect_score_0_100",
    ]
    eff = eff[[c for c in effect_cols if c in eff.columns]]
    effect_table_html = eff.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_effect_strength",
    )

    # --- Essentiality ---
    ess = pd.read_csv(outputs_dir / "Essentiality_by_gene.csv")
    if "essentiality_score_0_100" in ess.columns:
        ess = ess.sort_values("essentiality_score_0_100", ascending=False)

    ess_cols = [
        "Gene",
        "n_models",
        "median_effect",
        "q10",
        "q90",
        "n_prob50",
        "n_strong_lt_1",
        "BiologicalTag",
        "IsEssential_byMedianRule",
        "HGNC_Gene",
        "rank_gene_essentiality",
        "essentiality_score_0_100",
    ]
    ess = ess[[c for c in ess_cols if c in ess.columns]]
    essentiality_table_html = ess.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_essentiality",
    )

    # --- Connectivity ---
    conn = pd.read_csv(outputs_dir / "Connectivity_by_gene.csv")
    if "connectivity_score_0_100" in conn.columns:
        conn = conn.sort_values("connectivity_score_0_100", ascending=False)

    conn_cols = [
        "Gene",
        "HGNC_Gene",
        "CGC",
        "PPI_Degree",
        "rank_connectivity",
        "connectivity_score_0_100",
    ]
    conn = conn[[c for c in conn_cols if c in conn.columns]]
    connectivity_table_html = conn.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_connectivity",
    )

    # --- Druggability (no *_comment columns) ---
    drug = pd.read_csv(outputs_dir / "Drugability_by_drug.csv")
    sort_col = (
        "druggability_score_0_100"
        if "druggability_score_0_100" in drug.columns
        else "Overall Drugability Score"
    )
    if sort_col in drug.columns:
        drug = drug.sort_values(sort_col, ascending=False)

    drop_cols = [c for c in drug.columns if c.endswith("_comment")]
    drug = drug.drop(columns=drop_cols, errors="ignore")

    druggability_table_html = drug.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_druggability",
    )

    # --- Final gene priorities ---
    final_gene = pd.read_csv(outputs_dir / "Final_Gene_Priorities.csv")
    if "IntegratedScore_0_100" in final_gene.columns:
        final_gene = final_gene.sort_values("IntegratedScore_0_100", ascending=False)

    final_gene_table_html = final_gene.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_final_genes",
    )

    # --- Final gene–drug pairs ---
    pairs_path = outputs_dir / "Final_GeneDrug_Pairs.csv"
    if pairs_path.exists():
        final_pairs = pd.read_csv(pairs_path)
    else:
        final_pairs = pd.DataFrame(columns=["Gene", "Drug", "HGNC_Gene", "PairScore_0_100"])

    # Remove requested columns
    final_pairs = final_pairs.drop(
        columns=["druggability_score_0_100", "safety_score_0_100"],
        errors="ignore",
    )

    # Sorting preference
    pair_sort = None
    for candidate in ["PairScore_0_100", "IntegratedScore_0_100", "effect_score_0_100"]:
        if candidate in final_pairs.columns:
            pair_sort = candidate
            break
    if pair_sort:
        final_pairs = final_pairs.sort_values(pair_sort, ascending=False)

    final_pairs_table_html = final_pairs.to_html(
        index=False,
        classes="dataframe data-table",
        border=0,
        table_id="table_final_pairs",
    )

    summary = {
        "n_effect_genes": eff.shape[0],
        "n_essentiality_genes": ess.shape[0],
        "n_connectivity_genes": conn.shape[0],
        "n_drugs_druggability": drug.shape[0],  # computed but NOT shown in overview
        "n_final_genes": final_gene.shape[0],
        "n_final_pairs": final_pairs.shape[0],
        "top_genes": (
            final_gene["Gene"].head(3).tolist()
            if "Gene" in final_gene.columns
            else []
        ),
    }

    return (
        effect_table_html,
        essentiality_table_html,
        connectivity_table_html,
        druggability_table_html,
        final_gene_table_html,
        final_pairs_table_html,
        summary,
    )


# -----------------------------
# HTML
# -----------------------------
def build_html(
    out_path: Path,
    logo_left_path: Path,
    logo_right_path: Path,
    plots_dir: Path,
    tables,
    summary,
):
    (
        effect_table_html,
        essentiality_table_html,
        connectivity_table_html,
        druggability_table_html,
        final_gene_table_html,
        final_pairs_table_html,
    ) = tables

    n_effect_genes = summary["n_effect_genes"]
    n_essentiality_genes = summary["n_essentiality_genes"]
    n_connectivity_genes = summary["n_connectivity_genes"]
    n_final_genes = summary["n_final_genes"]
    n_final_pairs = summary["n_final_pairs"]
    top_genes_list = summary["top_genes"]
    top_genes_str = ", ".join(top_genes_list) if top_genes_list else "-"

    logo_left_uri = img_to_data_uri(logo_left_path)
    logo_right_uri = img_to_data_uri(logo_right_path)

    def plot_img_uri(name: str) -> str:
        return img_to_data_uri(plots_dir / name)

    def plot_html_uri(name: str) -> str:
        return html_to_data_uri(plots_dir / name)

    tpl = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ayass Bioscience · Therapeutic Target Integration</title>
  <style>
    :root {
      --bg-main: #f5f7ff;
      --bg-card: #ffffff;
      --bg-soft: #f1f4ff;
      --border-subtle: #dde2f2;
      --text-main: #0f172a;
      --text-soft: #64748b;
      --shadow-soft: 0 14px 30px rgba(148, 163, 184, 0.25);
      --radius-xl: 26px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "Segoe UI", sans-serif;
      background: var(--bg-main);
      color: var(--text-main);
      -webkit-font-smoothing: antialiased;
    }

    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 22px 20px 60px;
    }

    /* Header / hero */
    header.hero {
      border-radius: 34px;
      padding: 26px 26px;
      background: linear-gradient(135deg, #06a8b5 0%, #0aa7a7 35%, #0aa36b 100%);
      color: #ffffff;
      box-shadow: 0 24px 55px rgba(2, 132, 199, 0.25);
      position: relative;
      overflow: hidden;
      margin-bottom: 18px;
    }

    header.hero:before {
      content: "";
      position: absolute;
      top: -120px;
      right: -120px;
      width: 360px;
      height: 360px;
      border-radius: 999px;
      background: rgba(255,255,255,0.10);
    }

    header.hero:after {
      content: "";
      position: absolute;
      bottom: -140px;
      left: -140px;
      width: 420px;
      height: 420px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
    }

    .hero-row {
      position: relative;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      z-index: 1;
    }

    .hero-left {
      display: flex;
      align-items: flex-start;
      gap: 16px;
      min-width: 0;
    }

    /* REMOVE white background boxes from logo containers */
    .hero-logo {
      width: 64px;
      height: 64px;
      border-radius: 18px;
      background: transparent;     /* changed */
      box-shadow: none;           /* changed */
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      flex: 0 0 auto;
    }

    .hero-logo img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: transparent;
    }

    .hero-brand {
      font-size: 13px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      opacity: 0.95;
      margin-top: 2px;
      font-weight: 650;
    }

    .hero-title {
      margin: 8px 0 6px;
      font-size: 34px;
      line-height: 1.12;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .hero-subtitle {
      margin: 0;
      font-size: 14px;
      opacity: 0.95;
      max-width: 780px;
      line-height: 1.5;
    }

    .hero-right {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 10px;
      min-width: 260px;
      flex: 0 0 auto;
    }

    .pill-row-top {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(255,255,255,0.20);
      font-size: 12px;
      font-weight: 650;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .pill small {
      font-size: 12px;
      letter-spacing: 0.06em;
      opacity: 0.95;
      text-transform: uppercase;
      font-weight: 700;
    }

    /* Right logo container without white background */
    .hero-right-logo {
      width: 150px;
      height: 46px;
      border-radius: 14px;
      background: transparent;     /* changed */
      border: none;               /* changed */
      box-shadow: none;           /* changed */
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .hero-right-logo img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: transparent;
    }

    .section { margin-top: 22px; margin-bottom: 30px; }

    .section-title {
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 12px;
    }

    .section-title h3 { margin: 0; font-size: 18px; }
    .section-title span { font-size: 12px; color: var(--text-soft); }

    .card {
      background: var(--bg-card);
      border-radius: var(--radius-xl);
      padding: 18px 20px;
      box-shadow: var(--shadow-soft);
      border: 1px solid var(--border-subtle);
    }

    /* Overview metrics: 3 cards (Unique drugs removed) */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 10px;
    }

    .metric-card {
      border-radius: 16px;
      padding: 10px 12px;
      background: linear-gradient(135deg, #ffffff, #f2f5ff);
      border: 1px solid var(--border-subtle);
    }

    .metric-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-soft);
      margin-bottom: 4px;
    }

    .metric-value {
      font-size: 20px;
      font-weight: 800;
      color: #0f2a55;
    }

    .metric-hint {
      font-size: 11px;
      color: var(--text-soft);
      margin-top: 2px;
    }

    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    .pill-mini {
      font-size: 11px;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f3f4ff;
      border: 1px solid var(--border-subtle);
      color: var(--text-soft);
    }

    .weights-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }

    .weight-card {
      border-radius: 20px;
      padding: 12px 14px;
      box-shadow: 0 10px 20px rgba(148, 163, 184, 0.35);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 90px;
      color: #1f2933;
    }

    .weight-card h4 {
      margin: 0 0 4px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .weight-card span { font-size: 11px; color: var(--text-soft); }
    .weight-card p { margin: 8px 0 0; font-size: 11px; color: var(--text-soft); }

    .weight-effect { background: linear-gradient(135deg, #ffe4ef, #ffd6b8); }
    .weight-essentiality { background: linear-gradient(135deg, #ffe9c8, #ffd0b8); }
    .weight-safety { background: linear-gradient(135deg, #e6e7ff, #dde6ff); }
    .weight-druggability { background: linear-gradient(135deg, #d7f5ec, #d6f1ff); }
    .weight-connectivity { background: linear-gradient(135deg, #ffe9d2, #ffe0e0); }

    .two-column {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
      gap: 18px;
    }

    .table-card { display: flex; flex-direction: column; gap: 10px; }

    .table-header-line {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }

    .table-header-line h4 { margin: 0; font-size: 14px; }
    .table-caption { font-size: 11px; color: var(--text-soft); }

    .search-input {
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      font-size: 11px;
      min-width: 160px;
      background: #ffffff;
    }

    .table-wrapper {
      background: var(--bg-soft);
      border-radius: 14px;
      padding: 8px;
      border: 1px solid var(--border-subtle);
    }

    .table-scroll {
      max-height: 360px;
      overflow-y: auto;
      overflow-x: auto;
      border-radius: 12px;
      background: #ffffff;
    }

    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }

    table.data-table thead tr {
      background: #f3f4ff;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    table.data-table th,
    table.data-table td {
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7f0;
      white-space: nowrap;
    }

    table.data-table th {
      text-align: left;
      font-weight: 700;
      font-size: 11px;
      color: #4b4f68;
    }

    table.data-table tbody tr:nth-child(even) { background: #fafbff; }
    table.data-table tbody tr:hover { background: #eef2ff; }

    .figure-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }

    .figure-card img {
      width: 100%;
      border-radius: 16px;
      box-shadow: var(--shadow-soft);
      border: 1px solid var(--border-subtle);
      display: block;
      background: #ffffff;
    }

    .figure-card small {
      display: block;
      margin-top: 6px;
      font-size: 11px;
      color: var(--text-soft);
    }

    .figure-links { margin-top: 4px; font-size: 11px; }
    .figure-links a { color: #0f2a55; text-decoration: none; }
    .figure-links a:hover { text-decoration: underline; }

    .clinical-box {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: #f3faf7;
      border: 1px solid #bee3f8;
      font-size: 11px;
    }

    .clinical-box h4 { margin: 0 0 6px; font-size: 12px; }
    .clinical-box ul { margin: 0; padding-left: 18px; }
    .clinical-box li { margin-bottom: 4px; }

    .reference-list {
      font-size: 11px;
      color: var(--text-soft);
      padding-left: 16px;
    }

    .reference-list li { margin-bottom: 4px; }

    .footer-note {
      margin-top: 12px;
      font-size: 12px;
      color: var(--text-soft);
    }

    @media (max-width: 960px) {
      .metric-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); }
      .weights-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two-column { grid-template-columns: minmax(0, 1fr); }
      .hero-title { font-size: 28px; }
      .hero-right { align-items: flex-start; min-width: 0; }
      .pill-row-top { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="page">

    <header class="hero">
      <div class="hero-row">
        <div class="hero-left">
          <div class="hero-logo">$logo_left_html</div>
          <div style="min-width:0;">
            <div class="hero-brand">AYASS BIOSCIENCE · THERAPEUTIC TARGET INTEGRATION</div>
            <div class="hero-title">Therapeutic Target Assessment</div>
            <p class="hero-subtitle">
              DepMap CRISPR essentiality · L1000 transcriptional reversal · Connectivity, druggability &amp; safety integration
            </p>
          </div>
        </div>

        <div class="hero-right">
          <div class="pill-row-top">
            <div class="pill"><small>Generated</small>: $run_timestamp</div>
          </div>
          $logo_right_block
        </div>
      </div>
    </header>

    <section class="section">
      <div class="section-title">
        <h3>Integrated overview</h3>
        <span>Multi-module integration across effect strength, essentiality, safety, druggability and connectivity.</span>
      </div>
      <div class="card">
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Prioritized genes</div>
            <div class="metric-value">$n_final_genes</div>
            <div class="metric-hint">Final integrated targets with non-zero scores.</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Gene–drug pairs</div>
            <div class="metric-value">$n_final_pairs</div>
            <div class="metric-hint">High-quality reversal links carried into integration.</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Top genes</div>
            <div class="metric-value" style="font-size:13px;">$top_genes_str</div>
            <div class="metric-hint">Highest integrated scores in this disease context.</div>
          </div>
        </div>
        <div class="pill-row">
          <span class="pill-mini">$n_effect_genes genes with L1000 reversal metrics</span>
          <span class="pill-mini">$n_essentiality_genes genes with DepMap essentiality</span>
          <span class="pill-mini">$n_connectivity_genes genes with connectivity scores</span>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>Module weighting for integrated target scores</h3>
        <span>Multi-criteria ranking with fixed module weights</span>
      </div>
      <div class="card">
        <div class="weights-grid">
          <div class="weight-card weight-effect">
            <div>
              <h4>Effect strength (25%)</h4>
              <span>Sensitivity of reversal signatures</span>
            </div>
            <p>Maximal L1000 sensitivity index across high-R² reversal interactions per gene.</p>
          </div>
          <div class="weight-card weight-essentiality">
            <div>
              <h4>Disease essentiality (25%)</h4>
              <span>CRISPR survival dependency</span>
            </div>
            <p>Median Chronos scores across DepMap models, prioritising strong depletion in disease-relevant tumour context.</p>
          </div>
          <div class="weight-card weight-safety">
            <div>
              <h4>Safety profile (20%)</h4>
              <span>Structural risk &amp; alert burden</span>
            </div>
            <p>Penalises PAINS, Brenk and reactive toxicophores to down-rank unsafe chemical series.</p>
          </div>
          <div class="weight-card weight-druggability">
            <div>
              <h4>Druggability (20%)</h4>
              <span>Chemical tractability &amp; ADME</span>
            </div>
            <p>Combines QED, physicochemical ranges and oral drug-likeness of aligned compounds.</p>
          </div>
          <div class="weight-card weight-connectivity">
            <div>
              <h4>Connectivity (10%)</h4>
              <span>Network and cancer-gene context</span>
            </div>
            <p>PPI degree and CGC membership, favouring well-connected or known oncogenic regulators.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- moved up -->
    <section class="section">
      <div class="section-title">
        <h3>Integrated gene-level prioritisation</h3>
        <span>Weighted combination of five modules</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 3. Final gene priorities and module contributions</h4>
              <div class="table-caption">
                Per-gene scores across effect strength, essentiality, connectivity, druggability and safety with final integrated score.
              </div>
            </div>
            <input id="search_final_genes" class="search-input" placeholder="Search genes&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $final_gene_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="figure-grid">

            <div class="figure-card">
              <img src="$final_prioritized_genes_top30" alt="Top prioritized genes" />
              <small>Figure G1. Top genes by integrated score in this disease context.</small>
            </div>

            <div class="figure-card">
              <img src="$final_gene_module_contributions_stacked" alt="Module contribution stacked bars" />
              <small>Figure G2. Stacked contributions of each module to integrated scores per gene.</small>
              <div class="figure-links">
                Interactive:
                <a href="$final_gene_module_contributions_stacked_html" target="_blank" rel="noopener">module contributions (HTML)</a>
                · <a href="$final_integrated_scores_all_genes_html" target="_blank" rel="noopener">all-genes integrated scores (HTML)</a>
              </div>
            </div>

          </div>
          <div class="clinical-box">
            <h4>Clinical interpretation – integrated scores</h4>
            <ul>
              <li>Integrated scores prioritise genes where reversal strength, tumour essentiality and chemical feasibility align.</li>
              <li>Genes with high effect but poor safety/druggability are downgraded, guiding focus toward tractable biology.</li>
              <li>Connectivity adds modest weight, favouring regulators embedded in relevant oncogenic networks.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>L1000 effect strength per gene</h3>
        <span>Causal reversal links and sensitivity indices</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 1. Effect strength metrics by gene</h4>
              <div class="table-caption">
                Ranked by <em>effect_score_0_100</em>, derived from the maximum L1000 sensitivity index across high-quality reversal drug pairs.
              </div>
            </div>
            <input id="search_effect_strength" class="search-input" placeholder="Search genes&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $effect_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="figure-grid">
            <div class="figure-card">
              <img src="$effect_strength_top30" alt="Top genes ranked by L1000 effect strength" />
              <small>Figure E1. Top genes ranked by L1000 reversal effect strength (maximum sensitivity index).</small>
            </div>

            <div class="figure-card">
              <img src="$effect_strength_scatter" alt="All genes: L1000 effect strength vs rank" />
              <small>Figure E2. Distribution of effect strength scores across all genes.</small>
              <div class="figure-links">
                Interactive: <a href="$effect_strength_scatter_html" target="_blank" rel="noopener">scatter plot (HTML)</a>
              </div>
            </div>
          </div>

          <div class="clinical-box">
            <h4>Clinical interpretation – effect strength</h4>
            <ul>
              <li>Genes with high effect scores show strong, coherent reversal of disease-direction L1000 signatures.</li>
              <li>These genes represent transcriptional leverage points where drug perturbations robustly oppose oncogenic programs.</li>
              <li>Genes with many reversal pairs and high sensitivity indices are favoured for experimental validation of rescue phenotypes.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>Disease essentiality (DepMap)</h3>
        <span>CRISPR loss-of-function dependence</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 2. Essentiality metrics by gene</h4>
              <div class="table-caption">
                Ranked by <em>essentiality_score_0_100</em> from median Chronos effects across disease-matched models.
              </div>
            </div>
            <input id="search_essentiality" class="search-input" placeholder="Search genes&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $essentiality_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="figure-grid">

            <div class="figure-card">
              <img src="$essentiality_dist" alt="Distribution of essentiality tags" />
              <small>Figure ES1. Distribution of genes by essentiality tag across DepMap models.</small>
            </div>

            <div class="figure-card">
              <img src="$essentiality_score_vs_tag" alt="Essentiality score vs biological tag" />
              <small>Figure ES2. Essentiality scores mapped onto biological tags (core essential vs weak/contextual).</small>
            </div>

            <div class="figure-card">
              <small>Figure ES3. Interactive violin plots of Chronos effects by tag.</small>
              <div class="figure-links">
                Interactive: <a href="$essentiality_violin_html" target="_blank" rel="noopener">essentiality violin plots (HTML)</a>
              </div>
            </div>

          </div>

          <div class="clinical-box">
            <h4>Clinical interpretation – essentiality</h4>
            <ul>
              <li>Genes tagged as “core” or strongly essential with negative median Chronos effects are attractive cytotoxic targets.</li>
              <li>Weak or contextual essentiality genes may support synthetic lethality or biomarker-defined strategies.</li>
              <li>Genes with low essentiality scores are deprioritised to reduce risk of on-target toxicity in normal tissues.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>Connectivity and cancer-gene context</h3>
        <span>PPI degree and cancer gene census membership</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 4. Connectivity metrics by gene</h4>
              <div class="table-caption">
                Ranked by <em>connectivity_score_0_100</em> derived from PPI degree and CGC involvement.
              </div>
            </div>
            <input id="search_connectivity" class="search-input" placeholder="Search genes&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $connectivity_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="figure-grid">
            <div class="figure-card">
              <img src="$connectivity_top30" alt="Top connectivity genes" />
              <small>Figure C1. Top genes ranked by connectivity score (PPI context and CGC status).</small>
              <div class="figure-links">
                Interactive: <a href="$connectivity_top30_html" target="_blank" rel="noopener">connectivity view (HTML)</a>
              </div>
            </div>
          </div>

          <div class="clinical-box">
            <h4>Clinical interpretation – connectivity</h4>
            <ul>
              <li>Highly connected genes may act as pathway hubs, where modulation affects multiple oncogenic circuits.</li>
              <li>CGC membership flags known cancer drivers whose targeting has established biological rationale.</li>
              <li>Extremely hub-like genes may also carry safety risks; integration balances connectivity against other modules.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>Druggability and safety landscape</h3>
        <span>Chemical tractability, physicochemical space and structural alerts</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 5. Druggability metrics by compound</h4>
              <div class="table-caption">
                Ranked by <em>druggability_score_0_100</em> using overall drug-likeness, QED and structural features.
              </div>
            </div>
            <input id="search_druggability" class="search-input" placeholder="Search drugs&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $druggability_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="figure-grid">

            <div class="figure-card">
              <img src="$druggability_qed_vs_overall" alt="QED vs overall druggability" />
              <small>Figure D1. Relationship between QED and overall druggability scores.</small>
              <div class="figure-links">
                Interactive: <a href="$druggability_qed_vs_overall_html" target="_blank" rel="noopener">QED vs druggability (HTML)</a>
              </div>
            </div>

            <div class="figure-card">
              <img src="$safety_alert_summary" alt="Safety alert summary" />
              <small>Figure S1. Summary of PAINS, Brenk and reactive toxicophore alerts across compounds.</small>
              <div class="figure-links">
                Interactive: <a href="$safety_alert_summary_html" target="_blank" rel="noopener">safety alert view (HTML)</a>
              </div>
            </div>

            <div class="figure-card">
              <img src="$top_drugs_druggability_safety" alt="Top drugs by druggability and safety" />
              <small>Figure D2. Top compounds combining high druggability and favourable safety scores.</small>
              <div class="figure-links">
                Interactive: <a href="$top_drugs_druggability_safety_heatmap" target="_blank" rel="noopener">top-drug heatmap (HTML)</a>
              </div>
            </div>

            <div class="figure-card">
              <small>Figure D3. Interactive mapping of top genes to druggability-safety profiles.</small>
              <div class="figure-links">
                Interactive: <a href="$top_genes_druggability_safety_html" target="_blank" rel="noopener">top-gene druggability view (HTML)</a>
              </div>
            </div>

          </div>

          <div class="clinical-box">
            <h4>Clinical interpretation – druggability and safety</h4>
            <ul>
              <li>High druggability scores highlight scaffolds with oral-drug-like size, polarity and lipophilicity.</li>
              <li>Safety scores down-weight PAINS, Brenk and reactive motifs; such alerts signal potential ADMET liabilities.</li>
              <li>Compounds combining high druggability and high safety are preferred for further optimisation or repurposing.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>Gene–drug pair view</h3>
        <span>Reversal-optimised pairs with integrated context</span>
      </div>
      <div class="two-column">
        <div class="card table-card">
          <div class="table-header-line">
            <div>
              <h4>Table 6. Final gene–drug pairs</h4>
              <div class="table-caption">
                Sorted by <em>PairScore_0_100</em> where available, combining gene-level integration with druggability and safety.
              </div>
            </div>
            <input id="search_final_pairs" class="search-input" placeholder="Search gene or drug&hellip;" />
          </div>
          <div class="table-wrapper">
            <div class="table-scroll">
              $final_pairs_table_html
            </div>
          </div>
        </div>
        <div class="card">
          <div class="clinical-box" style="margin-top:0;">
            <h4>Clinical interpretation – gene–drug pairs</h4>
            <ul>
              <li>Pairs represent reversal-focused, tractable combinations: a prioritised gene plus a compound with suitable profile.</li>
              <li>Pairs with high PairScore, strong L1000 reversal and acceptable safety are candidates for experimental validation.</li>
              <li>This view is designed for hypothesis generation and pre-clinical design rather than direct treatment recommendation.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">
        <h3>References</h3>
        <span>Key resources underpinning this integration</span>
      </div>
      <div class="card">
        <ol class="reference-list">
          <li>Subramanian A, Narayan R, Corsello SM, <em>et al.</em> A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. <em>Cell</em>. 2017;171(6):1437–1452.e17.</li>
          <li>Integrated target assessment methodology inspired by multi-criteria oncology pipelines.</li>
        </ol>
        <div class="footer-note"><strong>Please review the downloadable files for more details.</strong></div>
      </div>
    </section>
  </div>

  <script>
    function setupTableSearch(inputId, tableId) {
      const input = document.getElementById(inputId);
      const table = document.getElementById(tableId);
      if (!input || !table) return;
      const tbody = table.querySelector('tbody');
      if (!tbody) return;

      input.addEventListener('input', function () {
        const query = this.value.toLowerCase();
        for (const row of tbody.rows) {
          let visible = false;
          for (const cell of row.cells) {
            if (cell.textContent.toLowerCase().includes(query)) {
              visible = true;
              break;
            }
          }
          row.style.display = visible ? '' : 'none';
        }
      });
    }

    document.addEventListener('DOMContentLoaded', function () {
      setupTableSearch('search_effect_strength', 'table_effect_strength');
      setupTableSearch('search_essentiality', 'table_essentiality');
      setupTableSearch('search_connectivity', 'table_connectivity');
      setupTableSearch('search_druggability', 'table_druggability');
      setupTableSearch('search_final_genes', 'table_final_genes');
      setupTableSearch('search_final_pairs', 'table_final_pairs');
    });
  </script>
</body>
</html>
""")

    logo_left_html = f'<img src="{logo_left_uri}" alt="Ayass Bioscience logo" />' if logo_left_uri else ""
    logo_right_block = (
        f'<div class="hero-right-logo"><img src="{logo_right_uri}" alt="Ayass Bioscience" /></div>'
        if logo_right_uri
        else ""
    )

    # Generate timestamp for report generation
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = tpl.substitute(
        logo_left_html=logo_left_html,
        logo_right_block=logo_right_block,
        run_timestamp=run_timestamp,

        n_final_genes=n_final_genes,
        n_final_pairs=n_final_pairs,
        top_genes_str=top_genes_str,

        n_effect_genes=n_effect_genes,
        n_essentiality_genes=n_essentiality_genes,
        n_connectivity_genes=n_connectivity_genes,

        effect_table_html=effect_table_html,
        essentiality_table_html=essentiality_table_html,
        connectivity_table_html=connectivity_table_html,
        druggability_table_html=druggability_table_html,
        final_gene_table_html=final_gene_table_html,
        final_pairs_table_html=final_pairs_table_html,

        # Embedded PNGs
        effect_strength_top30=plot_img_uri("effect_strength_top30.png"),
        effect_strength_scatter=plot_img_uri("effect_strength_all_genes_scatter.png"),
        essentiality_dist=plot_img_uri("essentiality_distribution_by_tag.png"),
        essentiality_score_vs_tag=plot_img_uri("essentiality_score_vs_tag.png"),
        connectivity_top30=plot_img_uri("connectivity_top30.png"),
        druggability_qed_vs_overall=plot_img_uri("druggability_qed_vs_overall.png"),
        safety_alert_summary=plot_img_uri("safety_alert_summary.png"),
        top_drugs_druggability_safety=plot_img_uri("top_drugs_druggability_safety.png"),
        final_prioritized_genes_top30=plot_img_uri("final_prioritized_genes_top30.png"),
        final_gene_module_contributions_stacked=plot_img_uri("final_gene_module_contributions_stacked.png"),

        # Embedded interactive HTML links
        effect_strength_scatter_html=plot_html_uri("effect_strength_all_genes_scatter.html"),
        essentiality_violin_html=plot_html_uri("essentiality_violin_by_tag.html"),
        connectivity_top30_html=plot_html_uri("connectivity_top30.html"),
        druggability_qed_vs_overall_html=plot_html_uri("druggability_qed_vs_overall.html"),
        safety_alert_summary_html=plot_html_uri("safety_alert_summary.html"),
        top_drugs_druggability_safety_heatmap=plot_html_uri("top_drugs_druggability_safety_heatmap.html"),
        top_genes_druggability_safety_html=plot_html_uri("top_genes_druggability_safety.html"),
        final_gene_module_contributions_stacked_html=plot_html_uri("final_gene_module_contributions_stacked.html"),
        final_integrated_scores_all_genes_html=plot_html_uri("final_integrated_scores_all_genes.html"),
    )

    out_path.write_text(html, encoding="utf-8")



def generate_integration_report(output_dir: Path, logo_left_path: Path, logo_right_path: Path):

    print(f"[INFO] Using integration run: {output_dir}")
    

    (
        effect_table_html,
        essentiality_table_html,
        connectivity_table_html,
        druggability_table_html,
        final_gene_table_html,
        final_pairs_table_html,
        summary,
    ) = build_tables(output_dir)

    build_html(
        out_path=output_dir / "integration_report.html",
        logo_left_path=logo_left_path,
        logo_right_path=logo_right_path,
        plots_dir=output_dir / "plots",
        tables=(
            effect_table_html,
            essentiality_table_html,
            connectivity_table_html,
            druggability_table_html,
            final_gene_table_html,
            final_pairs_table_html,
        ),
        summary=summary,
    )

    print(f"[INFO] Wrote integration report to: {output_dir / 'integration_report.html'}")


# if __name__ == "__main__":
    
#     generate_integration_report(
#         output_dir=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\Integration_Output"),
#         logo_left_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_left.png"),
#         logo_right_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_right.png"),
#     )
