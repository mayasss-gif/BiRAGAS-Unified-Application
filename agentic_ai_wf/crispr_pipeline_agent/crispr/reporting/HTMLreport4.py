from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from openai import OpenAI


# ==========================================================
# Config
# ==========================================================

LEFT_LOGO = "ABS_left.png"
RIGHT_LOGO = "ABS_Right.png"

API_KEY_FILE = "OpenApiKey.txt"
OUTPUT_HTML = "CRISPRModel_Simulator_Report.html"

DATASET_SUMMARY_PATH = "LLM_support/step-1_datasetsummary.txt"

# Stage 1
STAGE1_TABLE_PATH = "tables/sample_control_perturbation_summary.tsv"

# Stage 2
STAGE2_TABLE_CONDITION = "tables/stage2_condition_class_summary.tsv"
STAGE2_TABLE_COUNTS = "tables/stage2_mixscape_class_counts.tsv"
STAGE2_TABLE_SUMMARY = "tables/stage2_mixscape_summary.tsv"
# Representative figures (embedded as base64)
STAGE2_FIG_PERTURBSCORE = "figures/mixscape_perturbscore_ARHGAP22_pDS458.png"
STAGE2_FIG_VIOLIN = "figures/mixscape_violin_ARHGAP22_pDS458.png"
# NOTE: You asked to remove the UMAP figure from Stage 2, so we do NOT embed umap_mixscape_class.png.

# Stage 3
PROCESSED_STAGE3_DIR = "processed_stage3"
STAGE3_PERT_RANKED = "processed_stage3/stage3_perturbation_ranked.tsv"
STAGE3_RESP_HIST = "processed_stage3/stage3_responder_rate_hist.png"


# ==========================================================
# Utilities
# ==========================================================

def read_api_key(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def encode_image_to_base64(image_path: str) -> str | None:
    p = Path(image_path)
    if not p.exists():
        return None
    data = p.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def insert_or_replace_block(html_content: str, block_id: str, new_block: str) -> str:
    start_tag = f"<!-- STAGE:{block_id} -->"
    end_tag = f"<!-- END:{block_id} -->"

    if start_tag in html_content and end_tag in html_content:
        before = html_content.split(start_tag)[0]
        after = html_content.split(end_tag)[1]
        return before + new_block + after

    # fallback: append before </body>
    return html_content.replace("</body>", new_block + "\n</body>")


def safe_read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def df_to_html_table(
    df: pd.DataFrame,
    table_id: str,
    max_rows: int | None = None,
    drop_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    percent_cols: list[str] | None = None,
    small: bool = True
) -> str:
    out = df.copy()

    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows)

    # numeric formatting
    numeric_cols = numeric_cols or []
    percent_cols = percent_cols or []

    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in percent_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Build header
    ths = "".join([f"<th>{str(c)}</th>" for c in out.columns])

    # Build rows
    rows_html = ""
    for _, r in out.iterrows():
        tds = []
        for c in out.columns:
            val = r[c]
            cls = ""
            cell = ""

            if c in percent_cols and pd.notna(val):
                cell = f"{float(val)*100:.2f}%"
                cls = "num"
            elif c in numeric_cols and pd.notna(val):
                # ints vs floats
                if float(val).is_integer():
                    cell = f"{int(val):,}"
                else:
                    cell = f"{float(val):.6f}".rstrip("0").rstrip(".")
                cls = "num"
            else:
                cell = "" if pd.isna(val) else str(val)

            tds.append(f"<td class='{cls}'>{cell}</td>")
        rows_html += "<tr>" + "".join(tds) + "</tr>\n"

    size_class = "table-small" if small else ""
    return f"""
<div class="table-scroll {size_class}" id="{table_id}">
  <table class="styled-table">
    <thead><tr>{ths}</tr></thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
""".strip()


def llm_interpret_table(client: OpenAI, title: str, df: pd.DataFrame, guidance: str) -> str:
    # Keep prompt deterministic and scientific; CRISPR-focused.
    prompt = f"""
You are writing a scientific HTML report for a CRISPR Perturb-seq (transcriptomic perturbation) simulator pipeline.

Task:
Write 3–4 concise scientific sentences interpreting the table titled: "{title}".
- Focus strictly on CRISPR perturbation / guide assignment / responder vs non-responder behavior / cell composition.
- Avoid generic statements.
- Do not mention "LLM", "AI", or "language model".
- Use precise quantitative references from the table where helpful.

Additional guidance:
{guidance}

Table:
{df.to_string(index=False)}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


def find_latest_method_figures(method: str) -> Path | None:
    """
    method: 'bbknn' | 'harmony' | 'scvi' | 'none'
    Finds the latest folder matching '{method}-*' in processed_stage3,
    then returns <folder>/<method>/figures
    """
    root = Path(PROCESSED_STAGE3_DIR)
    if not root.exists():
        return None

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith(f"{method}-")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not candidates:
        return None

    latest = candidates[0]
    fig_dir = latest / method / "figures"
    if fig_dir.exists():
        return fig_dir
    return None


def read_markers_csv(markers_path: Path) -> pd.DataFrame:
    df = pd.read_csv(markers_path)

    # standardize column names (some pipelines differ)
    # expected: group, names, scores, logfoldchanges, pvals, pvals_adj, pct_nz_group, pct_nz_reference
    # ensure numeric sorting works
    for c in ["logfoldchanges", "pvals_adj", "pvals", "scores"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by LFC desc then adjusted p asc (as requested)
    if "logfoldchanges" in df.columns and "pvals_adj" in df.columns:
        df = df.sort_values(["logfoldchanges", "pvals_adj"], ascending=[False, True])
    elif "pvals_adj" in df.columns:
        df = df.sort_values(["pvals_adj"], ascending=[True])

    return df


# ==========================================================
# Header + Dataset + Stage0
# ==========================================================

def build_header_block() -> str:
    left_logo = encode_image_to_base64(LEFT_LOGO)
    right_logo = encode_image_to_base64(RIGHT_LOGO)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    left_img = f'<img src="data:image/png;base64,{left_logo}" style="height:70px;">' if left_logo else ""
    right_img = f'<img src="data:image/png;base64,{right_logo}" style="height:70px;">' if right_logo else ""

    return f"""
<!-- STAGE:HEADER -->
<div style="background: linear-gradient(90deg,#0f766e,#14b8a6);
            padding:40px 60px;
            color:white;
            display:flex;
            align-items:center;
            justify-content:space-between;">

    {left_img}

    <div style="text-align:center;">
        <div style="font-size:28px;font-weight:600;">
            CRISPR Perturb-seq Simulator Report
        </div>
        <div style="font-size:16px;opacity:0.9;margin-top:6px;">
            Transcriptomic CRISPR Perturbation Modeling Framework
        </div>
        <div style="font-size:13px;margin-top:8px;opacity:0.8;">
            Generated: {today}
        </div>
    </div>

    {right_img}

</div>
<!-- END:HEADER -->
""".strip()


def build_dataset_block() -> str:
    summary_text = safe_read_text(DATASET_SUMMARY_PATH)
    return f"""
<!-- STAGE:DATASET -->
<div class="stage-container">
    <div class="stage-header" style="background: linear-gradient(90deg,#1e3a8a,#3b82f6);">
        Dataset Overview
    </div>
    <div class="stage-content">
        <div class="dataset-summary">
            {summary_text}
        </div>
    </div>
</div>
<!-- END:DATASET -->
""".strip()


def build_stage0_block() -> str:
    return """
<!-- STAGE:STAGE0 -->
<div class="stage-container">
    <div class="stage-header" style="background: linear-gradient(90deg,#7c3aed,#a78bfa);">
        Stage 0: Data Ingestion and Preprocessing
    </div>
    <div class="stage-content">
        <p class="stage-description">
        Raw gene–barcode matrices were imported into structured <b>AnnData</b> objects and linked to guide identity metadata.
        This step consolidates per-cell guide signals and coverage metrics, enabling consistent downstream
        <i>transcriptomic CRISPR perturbation</i> modeling across all subsequent stages.
        </p>
    </div>
</div>
<!-- END:STAGE0 -->
""".strip()


# ==========================================================
# Stage 1 (Your preferred formatting)
# ==========================================================

def format_top_guides(raw_string: str) -> str:
    try:
        guide_dict = json.loads(raw_string.replace("'", '"'))
    except Exception:
        guide_dict = eval(raw_string)

    html = "<ol class='guide-list'>"
    for g, c in guide_dict.items():
        html += f"<li><b>{g}</b>: {c}</li>"
    html += "</ol>"
    return html


def build_stage1_block(df: pd.DataFrame, interpretation: str) -> str:
    row = df.iloc[0]
    total = int(row["total_cells"])
    multiplets = int(row["multiplets"])
    multiplet_rate = (multiplets / total) * 100
    confidence = float(row["mean_perturbation_confidence"])
    guides_html = format_top_guides(row["top10_guides"])

    return f"""
<!-- STAGE:STAGE1 -->
<div class="stage-container">

    <div class="stage-header">
        Stage 1: Guide Assignment and Perturbation Classification
    </div>

    <div class="stage-content">

        <p class="stage-description">
        This stage normalizes guide RNA identities, assigns cells into <b>control</b>, <b>perturbed</b>, or <b>unknown</b> classes,
        and flags likely multiplets. A composite <i>perturbation confidence</i> score summarizes guide evidence quality prior to
        responder modeling and downstream inference.
        </p>

        <div class="table-wrapper">
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>GSM ID</th>
                        <th>Total</th>
                        <th>Control</th>
                        <th>Perturbed</th>
                        <th>Unknown</th>
                        <th>Multiplets</th>
                        <th>Confidence</th>
                        <th>Top Guides</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{row["gsm_id"]}</td>
                        <td class="num">{total:,}</td>
                        <td class="num">{int(row["control_cells"]):,}</td>
                        <td class="num">{int(row["perturbed_cells"]):,}</td>
                        <td class="num">{int(row["unknown_cells"]):,}</td>
                        <td class="num multiplet">
                            {multiplets:,} ({multiplet_rate:.1f}%)
                        </td>
                        <td class="num confidence">{confidence:.4f}</td>
                        <td>{guides_html}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="caption">
            Table 1. Perturbation assignment statistics and guide-level distribution.
        </div>

        <div class="interpretation">
            {interpretation}
        </div>

    </div>
</div>
<!-- END:STAGE1 -->
""".strip()


# ==========================================================
# Stage 2
# ==========================================================

def build_stage2_block(
    t_condition: pd.DataFrame, interp_condition: str,
    t_counts_top50: pd.DataFrame, interp_counts: str,
    t_summary_top50: pd.DataFrame, interp_summary: str,
    fig_perturbscore_b64: str | None,
    fig_violin_b64: str | None
) -> str:

    # Tables (scrollable)
    tableA = df_to_html_table(
        t_condition,
        table_id="tbl_stage2A",
        max_rows=50,
        numeric_cols=["n_cells"],
        small=True
    )
    tableB = df_to_html_table(
        t_counts_top50,
        table_id="tbl_stage2B",
        max_rows=50,
        numeric_cols=["n_cells"],
        small=True
    )
    # Remove mean_mixscape_score from Table 2C (requested)
    tableC = df_to_html_table(
        t_summary_top50,
        table_id="tbl_stage2C",
        max_rows=50,
        drop_cols=["mean_mixscape_score"],
        numeric_cols=["n_cells", "responder_rate"],
        percent_cols=["responder_rate"],
        small=True
    )

    fig1 = ""
    if fig_perturbscore_b64:
        fig1 = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{fig_perturbscore_b64}" alt="Mixscape perturbation score density">
          <div class="caption">Figure 2A. Mixscape perturbation score distribution for a representative perturbation.</div>
        </div>
        """.strip()

    fig2 = ""
    if fig_violin_b64:
        fig2 = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{fig_violin_b64}" alt="Mixscape violin plot">
          <div class="caption">Figure 2B. Mixscape class separation (KO vs NP vs control) for a representative perturbation.</div>
        </div>
        """.strip()

    return f"""
<!-- STAGE:STAGE2 -->
<div class="stage-container">

    <div class="stage-header" style="background: linear-gradient(90deg,#0b7285,#22b8cf);">
        Stage 2: Mixscape Responder Deconvolution
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Stage 2 applies <b>Pertpy Mixscape</b> to separate <b>responders (KO-like)</b> from <b>non-responders (NP)</b> within each
        perturbation group. This provides a responder-aware view of CRISPR perturbation effects and supports downstream
        perturbation ranking and consolidation.
        </p>

        <div class="panel-grid">

            <div class="panel-card">
                <div class="panel-title">Condition Class Distribution</div>
                {tableA}
                <div class="caption">Table 2A. Global condition class composition.</div>
                <div class="interpretation">{interp_condition}</div>
            </div>

            <div class="panel-card">
                <div class="panel-title">KO / NP Counts (Top 50)</div>
                {tableB}
                <div class="caption">Table 2B. Responder and non-responder cell counts per perturbation (Top 50 rows).</div>
                <div class="interpretation">{interp_counts}</div>
            </div>

            <div class="panel-card">
                <div class="panel-title">Responder Summary (Top 50)</div>
                {tableC}
                <div class="caption">Table 2C. Responder rate per perturbation (Top 50 rows).</div>
                <div class="interpretation">{interp_summary}</div>
            </div>

        </div>

        <div class="subsection-title">Representative Mixscape Visualizations</div>

        <div class="fig-grid">
            {fig1}
            {fig2}
        </div>

    </div>
</div>
<!-- END:STAGE2 -->
""".strip()


# ==========================================================
# Stage 3
# ==========================================================

def build_stage3_block(
    pert_ranked_top50: pd.DataFrame,
    interp_pert_ranked: str,
    responder_hist_b64: str | None,
    method_blocks_html: str
) -> str:

    table_ranked = df_to_html_table(
        pert_ranked_top50,
        table_id="tbl_stage3_ranked",
        max_rows=50,
        drop_cols=["mean_mixscape_score"],   # requested
        numeric_cols=["n_cells", "responder_rate"],
        percent_cols=["responder_rate"],
        small=True
    )

    fig_hist = ""
    if responder_hist_b64:
        fig_hist = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{responder_hist_b64}" alt="Responder rate histogram">
          <div class="caption">Figure 3A. Distribution of responder rates after post-Mixscape consolidation.</div>
        </div>
        """.strip()

    return f"""
<!-- STAGE:STAGE3 -->
<div class="stage-container">

    <div class="stage-header" style="background: linear-gradient(90deg,#c2410c,#fb923c);">
        Stage 3: Cell Type Annotation and Post-Mixscape Consolidation
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Stage 3 adds biological context by assigning <b>cell types</b> using CellTypist and scANVI with majority voting,
        then consolidates perturbation labels post-Mixscape to stabilize responder calls. This stage produces a ranked
        view of perturbations based on responder behavior and prepares inputs for downstream modeling and inference.
        </p>

        <div class="subsection-title">Post-Mixscape Consolidation</div>

        <div class="two-col">
          <div class="panel-card">
              <div class="panel-title">Perturbation Ranking (Top 50)</div>
              {table_ranked}
              <div class="caption">Table 3A. Top perturbations ranked by responder rate (Top 50 rows).</div>
              <div class="interpretation">{interp_pert_ranked}</div>
          </div>

          <div class="panel-card">
              <div class="panel-title">Responder Rate Histogram</div>
              {fig_hist if fig_hist else "<div class='muted'>Figure not found.</div>"}
          </div>
        </div>

        <div class="subsection-title">Integration Methods and Cell Type Annotation Outputs</div>

        {method_blocks_html}

    </div>
</div>
<!-- END:STAGE3 -->
""".strip()


def build_method_block(
    method_name: str,
    figures_dir: Path | None,
    client: OpenAI
) -> str:

    pretty = {
        "bbknn": "BBKNN",
        "harmony": "Harmony",
        "scvi": "scVI",
        "none": "None (Baseline)"
    }.get(method_name, method_name)

    if not figures_dir or not figures_dir.exists():
        return f"""
        <div class="method-card">
          <div class="method-title">{pretty}</div>
          <div class="muted">No figures/tables detected for this method.</div>
        </div>
        """.strip()

    umap_path = figures_dir / "umap_final_celltype.png"
    paga_path = figures_dir / "paga_graph.png"
    markers_path = figures_dir / "tables" / "markers_leiden.csv"

    umap_b64 = encode_image_to_base64(str(umap_path))
    paga_b64 = encode_image_to_base64(str(paga_path))

    # Marker table
    interp = "<i>Marker table not found.</i>"
    table_html = "<div class='muted'>Marker table not found.</div>"

    if markers_path.exists():
        mdf = read_markers_csv(markers_path)

        # Keep only top 100 (requested)
        mdf_top = mdf.head(100).copy()

        # Table
        table_html = df_to_html_table(
            mdf_top,
            table_id=f"tbl_markers_{method_name}",
            max_rows=100,
            numeric_cols=[c for c in ["scores", "logfoldchanges", "pvals", "pvals_adj", "pct_nz_group", "pct_nz_reference"] if c in mdf_top.columns],
            small=True
        )

        interp = llm_interpret_table(
            client=client,
            title=f"Stage 3 Marker Genes ({pretty})",
            df=mdf_top,
            guidance="Interpret marker patterns as they relate to inferred cell states under CRISPR perturbation context; focus on what marker enrichment suggests about population structure."
        )

    # Figures
    umap_html = (f"""
      <div class="fig-card">
        <img class="figure-img" src="data:image/png;base64,{umap_b64}" alt="UMAP final celltype">
        <div class="caption">Figure 3.{pretty}A. UMAP projection colored by final cell type labels.</div>
      </div>
    """.strip()) if umap_b64 else "<div class='muted'>UMAP figure not found.</div>"

    paga_html = (f"""
      <div class="fig-card">
        <img class="figure-img" src="data:image/png;base64,{paga_b64}" alt="PAGA graph">
        <div class="caption">Figure 3.{pretty}B. PAGA graph summarizing inferred topology between cell populations.</div>
      </div>
    """.strip()) if paga_b64 else "<div class='muted'>PAGA figure not found.</div>"

    return f"""
<div class="method-card">
  <div class="method-title">{pretty}</div>

  <div class="fig-grid method-figs">
    {umap_html}
    {paga_html}
  </div>

  <div class="panel-card" style="margin-top:14px;">
    <div class="panel-title">Marker Genes (Top 100; sorted by logfoldchanges and adjusted p-value)</div>
    {table_html}
    <div class="caption">Table 3.{pretty}C. Top marker genes (Top 100 rows).</div>
    <div class="interpretation">{interp}</div>
  </div>
</div>
""".strip()


# ==========================================================
# Base Report HTML + CSS
# ==========================================================

def create_base_report() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CRISPR Perturb-seq Simulator Report</title>

<style>
  body {
      font-family: "Segoe UI", Arial, sans-serif;
      background-color: #f3f4f6;
      margin: 0;
  }

  .stage-container {
      margin: 48px auto;
      width: 88%;
      border-radius: 14px;
      background: white;
      box-shadow: 0 10px 25px rgba(0,0,0,0.05);
      overflow: hidden;
  }

  .stage-header {
      background: linear-gradient(90deg, #0f766e, #14b8a6);
      color: white;
      padding: 18px 30px;
      font-size: 22px;
      font-weight: 600;
  }

  .stage-content {
      padding: 34px 46px;
  }

  .stage-description {
      font-size: 15px;
      line-height: 1.7;
      margin-bottom: 18px;
      color: #374151;
  }

  .dataset-summary {
      font-size: 15px;
      line-height: 1.8;
      color: #374151;
  }

  .subsection-title{
      margin-top: 18px;
      margin-bottom: 10px;
      font-size: 16px;
      font-weight: 700;
      color: #111827;
  }

  .panel-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;            /* tighter (fix gaps) */
      margin-top: 12px;
  }

  .two-col {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 14px;
      margin-top: 10px;
  }

  .panel-card {
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px 14px;
      background: #ffffff;
  }

  .panel-title {
      font-weight: 700;
      margin-bottom: 10px;
      color: #111827;
      font-size: 14px;
  }

  .table-wrapper {
      overflow-x: auto;
  }

  .table-scroll {
      max-height: 280px;
      overflow: auto;
      border-radius: 10px;
      border: 1px solid #eef2f7;
  }

  .table-small table {
      font-size: 12px;      /* reduce table text size (requested) */
  }

  .styled-table {
      width: 100%;
      border-collapse: collapse;
  }

  .styled-table th {
      text-align: left;
      padding: 10px 10px;
      background-color: #e6f4f1;
      position: sticky;
      top: 0;
      z-index: 2;
      font-size: 12px;
  }

  .styled-table td {
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      vertical-align: top;
      font-size: 12px;
      color: #111827;
  }

  .num {
      text-align: right;
      font-variant-numeric: tabular-nums;
  }

  .multiplet {
      font-style: italic;
      background-color: #f8f1e5;
  }

  .confidence {
      font-weight: bold;
  }

  .caption {
      font-size: 12.5px;
      margin-top: 8px;
      font-style: italic;
      color: #6b7280;
  }

  .interpretation {
      font-size: 13.5px;
      margin-top: 10px;
      font-style: italic;
      color: #374151;
      line-height: 1.6;
  }

  .guide-list {
      margin: 0;
      padding-left: 18px;
  }

  .fig-grid{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 12px;
  }

  .fig-card{
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px;
      background: #fff;
  }

  .figure-img{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid #f3f4f6;
      background: #fff;
  }

  .method-card{
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px;
      background: #ffffff;
      margin-top: 14px;
  }

  .method-title{
      font-weight: 800;
      font-size: 15px;
      color: #0f172a;
      margin-bottom: 10px;
  }

  .muted{
      color:#6b7280;
      font-size: 13px;
  }

  /* Responsive */
  @media (max-width: 1200px){
      .panel-grid { grid-template-columns: 1fr; }
      .two-col { grid-template-columns: 1fr; }
      .fig-grid { grid-template-columns: 1fr; }
  }
</style>
</head>

<body>

<!-- STAGE:HEADER -->
<!-- END:HEADER -->

<!-- STAGE:DATASET -->
<!-- END:DATASET -->

<!-- STAGE:STAGE0 -->
<!-- END:STAGE0 -->

<!-- STAGE:STAGE1 -->
<!-- END:STAGE1 -->

<!-- STAGE:STAGE2 -->
<!-- END:STAGE2 -->

<!-- STAGE:STAGE3 -->
<!-- END:STAGE3 -->

</body>
</html>
""".strip()


# ==========================================================
# Main
# ==========================================================

def main():
    client = OpenAI(api_key=read_api_key(API_KEY_FILE))
    report_path = Path(OUTPUT_HTML)

    # Create base report once
    if not report_path.exists():
        report_path.write_text(create_base_report(), encoding="utf-8")

    html = report_path.read_text(encoding="utf-8")

    # Header + dataset + stage0
    html = insert_or_replace_block(html, "HEADER", build_header_block())
    html = insert_or_replace_block(html, "DATASET", build_dataset_block())
    html = insert_or_replace_block(html, "STAGE0", build_stage0_block())

    # --------------------------
    # Stage 1 (LLM interpretation)
    # --------------------------
    df1 = pd.read_csv(STAGE1_TABLE_PATH, sep="\t")
    interp1 = llm_interpret_table(
        client=client,
        title="Stage 1 Perturbation Assignment Summary",
        df=df1,
        guidance="Highlight perturbation coverage, multiplet burden, confidence interpretation, and guide dominance if present."
    )
    html = insert_or_replace_block(html, "STAGE1", build_stage1_block(df1, interp1))

    # --------------------------
    # Stage 2 (3 tables + 2 figs; 1 interpretation per table)
    # --------------------------
    t2a = pd.read_csv(STAGE2_TABLE_CONDITION, sep="\t")
    t2b = pd.read_csv(STAGE2_TABLE_COUNTS, sep="\t").head(50).copy()
    t2c = pd.read_csv(STAGE2_TABLE_SUMMARY, sep="\t").head(50).copy()

    # Remove mean_mixscape_score from table 2C display (requested)
    if "mean_mixscape_score" in t2c.columns:
        t2c = t2c.drop(columns=["mean_mixscape_score"], errors="ignore")

    interp2a = llm_interpret_table(
        client=client,
        title="Stage 2 Condition Class Distribution",
        df=t2a,
        guidance="Explain how cell composition (control/perturbed/multipert/unknown) impacts Mixscape responder inference."
    )
    interp2b = llm_interpret_table(
        client=client,
        title="Stage 2 KO/NP Counts per Perturbation (Top 50)",
        df=t2b,
        guidance="Interpret KO vs NP counts as evidence of responder separation strength across perturbations."
    )
    interp2c = llm_interpret_table(
        client=client,
        title="Stage 2 Responder Summary (Top 50)",
        df=t2c,
        guidance="Focus on responder_rate patterns: which perturbations show strong responder behavior and what that implies."
    )

    fig2a = encode_image_to_base64(STAGE2_FIG_PERTURBSCORE)
    fig2b = encode_image_to_base64(STAGE2_FIG_VIOLIN)

    html = insert_or_replace_block(
        html,
        "STAGE2",
        build_stage2_block(
            t_condition=t2a, interp_condition=interp2a,
            t_counts_top50=t2b, interp_counts=interp2b,
            t_summary_top50=t2c, interp_summary=interp2c,
            fig_perturbscore_b64=fig2a,
            fig_violin_b64=fig2b
        )
    )

    # --------------------------
    # Stage 3
    # --------------------------
    t3_rank = pd.read_csv(STAGE3_PERT_RANKED, sep="\t").copy()

    # Remove mean_mixscape_score (requested)
    if "mean_mixscape_score" in t3_rank.columns:
        t3_rank = t3_rank.drop(columns=["mean_mixscape_score"], errors="ignore")

    # Ensure sorting by responder_rate desc
    if "responder_rate" in t3_rank.columns:
        t3_rank["responder_rate"] = pd.to_numeric(t3_rank["responder_rate"], errors="coerce")
        t3_rank = t3_rank.sort_values(["responder_rate"], ascending=[False])

    t3_top50 = t3_rank.head(50).copy()

    interp3_rank = llm_interpret_table(
        client=client,
        title="Stage 3 Perturbation Ranking (Top 50)",
        df=t3_top50,
        guidance="Interpret which perturbations are most KO-like post-consolidation and what stable high responder rates imply."
    )

    fig3_hist = encode_image_to_base64(STAGE3_RESP_HIST)

    # Integration methods
    method_html_parts = []
    for method in ["bbknn", "harmony", "scvi", "none"]:
        fig_dir = find_latest_method_figures(method)
        method_html_parts.append(build_method_block(method, fig_dir, client))

    methods_html = "\n".join(method_html_parts)

    html = insert_or_replace_block(
        html,
        "STAGE3",
        build_stage3_block(
            pert_ranked_top50=t3_top50,
            interp_pert_ranked=interp3_rank,
            responder_hist_b64=fig3_hist,
            method_blocks_html=methods_html
        )
    )

    # Write final
    report_path.write_text(html, encoding="utf-8")
    print("✅ Report updated: Header + Dataset + Stage0 + Stage1 + Stage2 + Stage3 rendered successfully.")


if __name__ == "__main__":
    main()
