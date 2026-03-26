#!/usr/bin/env python3

import base64
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI


# ===============================
# PATHS
# ===============================

LEFT_LOGO = "ABS_left.png"
RIGHT_LOGO = "ABS_Right.png"

DATASET_SUMMARY_PATH = "LLM_support/step-1_datasetsummary.txt"
STAGE1_TABLE_PATH = "tables/sample_control_perturbation_summary.tsv"

STAGE2_CONDITION_PATH = "tables/stage2_condition_class_summary.tsv"
STAGE2_CLASS_COUNTS_PATH = "tables/stage2_mixscape_class_counts.tsv"
STAGE2_SUMMARY_PATH = "tables/stage2_mixscape_summary.tsv"

FIGURE_DENSITY = "figures/mixscape_perturbscore_ARHGAP22_pDS458.png"
FIGURE_VIOLIN = "figures/mixscape_violin_ARHGAP22_pDS458.png"

API_KEY_FILE = "OpenApiKey.txt"
OUTPUT_HTML = "CRISPRModel_Simulator_Report.html"


# ==========================================================
# Utilities
# ==========================================================

def read_api_key(path):
    return Path(path).read_text().strip()


def encode_image_to_base64(image_path):
    if not Path(image_path).exists():
        return ""
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
            padding:35px 60px;
            color:white;
            display:flex;
            align-items:center;
            justify-content:space-between;">

    <img src="data:image/png;base64,{left_logo}" style="height:65px;">

    <div style="text-align:center;">
        <div style="font-size:26px;font-weight:600;">
            CRISPR Perturb-seq Simulator Report
        </div>
        <div style="font-size:15px;opacity:0.9;margin-top:5px;">
            Transcriptomic CRISPR Perturbation Modeling Framework
        </div>
        <div style="font-size:12px;margin-top:6px;opacity:0.8;">
            Generated: {today}
        </div>
    </div>

    <img src="data:image/png;base64,{right_logo}" style="height:65px;">
</div>
<!-- END:HEADER -->
"""


# ==========================================================
# Dataset Overview
# ==========================================================

def build_dataset_block():

    summary_text = Path(DATASET_SUMMARY_PATH).read_text().strip()

    return f"""
<!-- STAGE:DATASET -->
<div class="stage-container">

    <div class="stage-header dataset-header">
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
# Stage 2 Block (UPDATED)
# ==========================================================

def build_stage2_block(client):

    cond_df = pd.read_csv(STAGE2_CONDITION_PATH, sep="\t")
    class_df = pd.read_csv(STAGE2_CLASS_COUNTS_PATH, sep="\t").head(50)
    summary_df = pd.read_csv(STAGE2_SUMMARY_PATH, sep="\t").head(50)

    # 🔴 REMOVE mean_mixscape_score column
    if "mean_mixscape_score" in summary_df.columns:
        summary_df = summary_df.drop(columns=["mean_mixscape_score"])

    # Interpretation via LLM
    prompt = f"""
Interpret CRISPR Mixscape responder analysis in 5 concise scientific sentences.

Condition distribution:
{cond_df.to_string(index=False)}

Responder summary:
{summary_df.head(10).to_string(index=False)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    interpretation = response.choices[0].message.content.strip()

    def df_to_html(df, scroll=False):
        table = df.to_html(index=False, classes="styled-table", border=0)
        if scroll:
            return f"<div class='scroll-box'>{table}</div>"
        return table

    density_img = encode_image_to_base64(FIGURE_DENSITY)
    violin_img = encode_image_to_base64(FIGURE_VIOLIN)

    return f"""
<!-- STAGE:STAGE2 -->
<div class="stage-container">

    <div class="stage-header stage2-header">
        Stage 2: Mixscape Responder Deconvolution
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Pertpy Mixscape was applied to deconvolve responder (<b>KO</b>)
        and non-responder (<b>NP</b>) cells within each perturbation.
        This stage quantifies functional perturbation efficacy and
        transcriptional response heterogeneity.
        </p>

        <div class="panel-grid">

            <div class="panel">
                <div class="panel-title">Condition Class Distribution</div>
                {df_to_html(cond_df)}
                <div class="caption">Table 2A. Global condition class composition.</div>
            </div>

            <div class="panel">
                <div class="panel-title">KO / NP Counts (Top 50)</div>
                {df_to_html(class_df, scroll=True)}
                <div class="caption">Table 2B. Responder and non-responder cell counts per perturbation.</div>
            </div>

            <div class="panel">
                <div class="panel-title">Responder Summary (Top 50)</div>
                {df_to_html(summary_df, scroll=True)}
                <div class="caption">Table 2C. Responder rate per perturbation.</div>
            </div>

        </div>

        <div style="margin-top:25px;">
            <div class="panel-title">Representative Mixscape Visualizations</div>
            <div class="figure-row">

                <img src="data:image/png;base64,{density_img}" class="figure-img">
                <img src="data:image/png;base64,{violin_img}" class="figure-img">

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
# Base Report (SPACING FIXED)
# ==========================================================

def create_base_report():

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
    margin: 35px auto;
    width: 90%;
    border-radius: 12px;
    background: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    overflow: hidden;
}

.stage-header {
    padding: 16px 28px;
    font-size: 20px;
    font-weight: 600;
    color: white;
}

.dataset-header {
    background: linear-gradient(90deg,#1e3a8a,#3b82f6);
}

.stage2-header {
    background: linear-gradient(90deg,#92400e,#f59e0b);
}

.stage-content {
    padding: 28px 35px;
}

.stage-description {
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 20px;
    color: #374151;
}

.dataset-summary {
    font-size: 14px;
    line-height: 1.7;
}

.panel-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.panel {
    background: #f9fafb;
    padding: 15px;
    border-radius: 10px;
}

.panel-title {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 10px;
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.styled-table th {
    text-align: left;
    padding: 6px;
    background-color: #e5e7eb;
}

.styled-table td {
    padding: 5px;
    border-bottom: 1px solid #e5e7eb;
}

.scroll-box {
    max-height: 250px;
    overflow-y: auto;
}

.figure-row {
    display: flex;
    gap: 20px;
    margin-top: 10px;
}

.figure-img {
    width: 48%;
    border-radius: 8px;
}

.caption {
    font-size: 11px;
    margin-top: 6px;
    font-style: italic;
    color: #6b7280;
}

.interpretation {
    font-size: 13px;
    margin-top: 20px;
    font-style: italic;
}

</style>
</head>

<body>

<!-- STAGE:HEADER --><!-- END:HEADER -->
<!-- STAGE:DATASET --><!-- END:DATASET -->
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

    report_path.write_text(create_base_report(), encoding="utf-8")
    html_content = report_path.read_text()

    html_content = insert_or_replace_block(html_content, "HEADER", build_header_block())
    html_content = insert_or_replace_block(html_content, "DATASET", build_dataset_block())
    html_content = insert_or_replace_block(html_content, "STAGE2", build_stage2_block(client))

    report_path.write_text(html_content, encoding="utf-8")

    print("✅ Full report rebuilt cleanly with corrected Stage 2.")


if __name__ == "__main__":
    main()

