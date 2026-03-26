#!/usr/bin/env python3

import base64
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI


LEFT_LOGO = "ABS_left.png"
RIGHT_LOGO = "ABS_Right.png"
DATASET_SUMMARY_PATH = "LLM_support/step-1_datasetsummary.txt"

STAGE1_TABLE_PATH = "tables/sample_control_perturbation_summary.tsv"

STAGE2_COND_PATH = "tables/stage2_condition_class_summary.tsv"
STAGE2_CLASS_COUNTS_PATH = "tables/stage2_mixscape_class_counts.tsv"
STAGE2_SUMMARY_PATH = "tables/stage2_mixscape_summary.tsv"

API_KEY_FILE = "OpenApiKey.txt"
OUTPUT_HTML = "CRISPRModel_Simulator_Report.html"


# ==========================================================
# Utilities
# ==========================================================

def read_api_key(path):
    return Path(path).read_text().strip()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def insert_or_replace_block(html_content, block_id, new_block):
    start_tag = f"<!-- STAGE:{block_id} -->"
    end_tag = f"<!-- END:{block_id} -->"

    if start_tag in html_content:
        before = html_content.split(start_tag)[0]
        after = html_content.split(end_tag)[1]
        return before + new_block + after
    else:
        return html_content.replace("</body>", new_block + "\n</body>")


# ==========================================================
# Header
# ==========================================================

def build_header_block():
    left_logo = encode_image_to_base64(LEFT_LOGO)
    right_logo = encode_image_to_base64(RIGHT_LOGO)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""
<!-- STAGE:HEADER -->
<div style="background: linear-gradient(90deg,#0f766e,#14b8a6);
            padding:40px 60px;
            color:white;
            display:flex;
            align-items:center;
            justify-content:space-between;">

    <img src="data:image/png;base64,{left_logo}" style="height:70px;">

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

    <img src="data:image/png;base64,{right_logo}" style="height:70px;">
</div>
<!-- END:HEADER -->
"""


# ==========================================================
# Dataset
# ==========================================================

def build_dataset_block():
    summary_text = Path(DATASET_SUMMARY_PATH).read_text().strip()

    return f"""
<!-- STAGE:DATASET -->
<div class="stage-container">
    <div class="stage-header stage-blue">
        Dataset Overview
    </div>
    <div class="stage-content">
        <div class="dataset-summary">
            {summary_text}
        </div>
    </div>
</div>
<!-- END:DATASET -->
"""


# ==========================================================
# Stage 0
# ==========================================================

def build_stage0_block():
    return """
<!-- STAGE:STAGE0 -->
<div class="stage-container">
    <div class="stage-header stage-purple">
        Stage 0: Data Ingestion and Preprocessing
    </div>
    <div class="stage-content">
        <p class="stage-description">
        Raw gene-barcode matrices were imported into structured AnnData objects.
        Cell-level metadata, guide identities, and coverage metrics were integrated.
        Quality control filtering, normalization, dimensionality reduction,
        and neighborhood graph construction were performed to prepare
        transcriptomic profiles for downstream CRISPR perturbation modeling.
        </p>
    </div>
</div>
<!-- END:STAGE0 -->
"""


# ==========================================================
# Stage 1
# ==========================================================

def format_top_guides(raw_string):
    try:
        guide_dict = json.loads(raw_string.replace("'", '"'))
    except:
        guide_dict = eval(raw_string)

    html = "<ol style='margin:0; padding-left:18px;'>"
    for g, c in guide_dict.items():
        html += f"<li><b>{g}</b>: {c}</li>"
    html += "</ol>"
    return html


def build_stage1_block(df, interpretation):
    row = df.iloc[0]
    guides_html = format_top_guides(row["top10_guides"])

    return f"""
<!-- STAGE:STAGE1 -->
<div class="stage-container">

    <div class="stage-header">
        Stage 1: Guide Assignment and Perturbation Classification
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Guide RNA identity normalization, control classification, multiplet detection,
        and perturbation confidence scoring were performed to establish
        perturbation assignment integrity prior to Mixscape analysis.
        </p>

        <div class="table-wrapper">
            {df.to_html(index=False, classes="styled-table", border=0)}
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
"""


# ==========================================================
# Stage 2
# ==========================================================

def dataframe_to_scrollable_html(df, max_rows=50):
    df = df.head(max_rows)
    return f"""
    <div class="scroll-table">
        {df.to_html(index=False, classes="inner-table", border=0)}
    </div>
    """


def build_stage2_block(cond_df, class_counts_df, summary_df, interpretation):

    cond_html = dataframe_to_scrollable_html(cond_df)
    class_counts_html = dataframe_to_scrollable_html(class_counts_df, 50)
    summary_html = dataframe_to_scrollable_html(summary_df, 50)

    return f"""
<!-- STAGE:STAGE2 -->
<div class="stage-container">

    <div class="stage-header stage-blue-dark">
        Stage 2: Mixscape Responder Deconvolution
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Pertpy Mixscape was applied to distinguish transcriptional responders (KO)
        from non-responders (NP) within each CRISPR perturbation group.
        This analysis quantifies functional gene disruption efficiency
        and refines perturbation-level interpretation.
        </p>

        <div class="panel-grid">

            <div class="panel">
                <div class="panel-title">Condition Class Distribution</div>
                {cond_html}
                <div class="caption">Table 2A. Global condition class composition.</div>
            </div>

            <div class="panel">
                <div class="panel-title">KO / NP Counts (Top 50)</div>
                {class_counts_html}
                <div class="caption">Table 2B. Responder and non-responder cell counts per perturbation.</div>
            </div>

            <div class="panel">
                <div class="panel-title">Responder Summary (Top 50)</div>
                {summary_html}
                <div class="caption">Table 2C. Responder rate and mean Mixscape score per perturbation.</div>
            </div>

        </div>

        <div class="figure-section">
            <div class="figure-title">Representative Mixscape Visualizations</div>

            <div class="figure-grid">
                <div class="fig">
                    <img src="figures/mixscape_perturbscore_ARHGAP22_pDS458.png">
                </div>
                <div class="fig">
                    <img src="figures/mixscape_violin_ARHGAP22_pDS458.png">
                </div>
                <div class="fig">
                    <img src="figures/umap_mixscape_class.png">
                </div>
            </div>
        </div>

        <div class="interpretation">
            {interpretation}
        </div>

    </div>
</div>
<!-- END:STAGE2 -->
"""


# ==========================================================
# Base Template
# ==========================================================

def create_base_report():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CRISPR Perturb-seq Simulator Report</title>

<style>

body { font-family: "Segoe UI", Arial; background:#f3f4f6; margin:0; }

.stage-container { margin:60px auto; width:88%; border-radius:14px;
background:white; box-shadow:0 10px 25px rgba(0,0,0,0.05); overflow:hidden; }

.stage-header { padding:18px 30px; font-size:22px; font-weight:600; color:white;
background: linear-gradient(90deg,#0f766e,#14b8a6); }

.stage-blue { background: linear-gradient(90deg,#1e3a8a,#3b82f6); }
.stage-purple { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.stage-blue-dark { background: linear-gradient(90deg,#1e3a8a,#2563eb); }

.stage-content { padding:40px 50px; }

.dataset-summary, .stage-description { font-size:15px; line-height:1.8; color:#374151; }

.styled-table { width:100%; border-collapse:collapse; font-size:13px; }
.styled-table th { background:#e6f4f1; padding:10px; text-align:left; }
.styled-table td { padding:8px; border-bottom:1px solid #e5e7eb; }

.caption { font-size:13px; margin-top:10px; font-style:italic; color:#6b7280; }
.interpretation { font-size:14px; margin-top:25px; font-style:italic; color:#374151; }

.panel-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:30px; margin-top:30px; }
.panel { background:#f9fafb; padding:20px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.04); }
.panel-title { font-weight:600; margin-bottom:15px; }

.scroll-table { max-height:280px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:6px; }
.inner-table { width:100%; font-size:12px; border-collapse:collapse; }
.inner-table th { background:#eef2ff; padding:8px; text-align:left; }
.inner-table td { padding:6px; border-bottom:1px solid #e5e7eb; }

.figure-section { margin-top:50px; }
.figure-title { font-weight:600; margin-bottom:20px; }
.figure-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:25px; }
.fig img { width:100%; border-radius:8px; }

</style>
</head>

<body>

<!-- STAGE:HEADER --><!-- END:HEADER -->
<!-- STAGE:DATASET --><!-- END:DATASET -->
<!-- STAGE:STAGE0 --><!-- END:STAGE0 -->
<!-- STAGE:STAGE1 --><!-- END:STAGE1 -->
<!-- STAGE:STAGE2 --><!-- END:STAGE2 -->

</body>
</html>
"""


# ==========================================================
# Main
# ==========================================================

def main():

    client = OpenAI(api_key=read_api_key(API_KEY_FILE))
    report_path = Path(OUTPUT_HTML)

    if not report_path.exists():
        report_path.write_text(create_base_report(), encoding="utf-8")

    html_content = report_path.read_text(encoding="utf-8")

    html_content = insert_or_replace_block(html_content, "HEADER", build_header_block())
    html_content = insert_or_replace_block(html_content, "DATASET", build_dataset_block())
    html_content = insert_or_replace_block(html_content, "STAGE0", build_stage0_block())

    # Stage 1
    df1 = pd.read_csv(STAGE1_TABLE_PATH, sep="\t")

    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Interpret CRISPR perturbation summary in 4 scientific sentences.\n{df1.to_string(index=False)}"}],
        temperature=0.2
    )
    stage1_block = build_stage1_block(df1, response1.choices[0].message.content.strip())
    html_content = insert_or_replace_block(html_content, "STAGE1", stage1_block)

    # Stage 2
    cond_df = pd.read_csv(STAGE2_COND_PATH, sep="\t")
    class_counts_df = pd.read_csv(STAGE2_CLASS_COUNTS_PATH, sep="\t")
    summary_df = pd.read_csv(STAGE2_SUMMARY_PATH, sep="\t")

    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Interpret Mixscape responder analysis in 5 scientific sentences.\n{summary_df.head(15).to_string(index=False)}"}],
        temperature=0.2
    )

    stage2_block = build_stage2_block(cond_df, class_counts_df, summary_df,
                                      response2.choices[0].message.content.strip())

    html_content = insert_or_replace_block(html_content, "STAGE2", stage2_block)

    report_path.write_text(html_content, encoding="utf-8")

    print("✅ Full CRISPR Report (Stage 0–2) successfully generated.")


if __name__ == "__main__":
    main()

