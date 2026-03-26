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
# Header Block
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
# Dataset Overview Block
# ==========================================================

def build_dataset_block():

    summary_text = Path(DATASET_SUMMARY_PATH).read_text().strip()

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
"""


# ==========================================================
# Stage 0 Block
# ==========================================================

def build_stage0_block():

    return f"""
<!-- STAGE:STAGE0 -->
<div class="stage-container">

    <div class="stage-header" style="background: linear-gradient(90deg,#7c3aed,#a78bfa);">
        Stage 0: Data Ingestion and Preprocessing
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Raw gene-barcode matrices were imported into structured AnnData objects.
        Cell-level metadata, guide identities, and coverage metrics were integrated.
        Quality control filtering and normalization were performed to ensure robust
        downstream CRISPR perturbation modeling.
        </p>

    </div>

</div>
<!-- END:STAGE0 -->
"""


# ==========================================================
# Stage 1 Formatting (UNCHANGED FROM YOUR VERSION)
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
        This stage performs guide RNA identity normalization, control classification,
        multiplet detection, and composite perturbation confidence scoring.
        The summary below reflects perturbation coverage and assignment quality
        prior to downstream transcriptomic modeling.
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
"""


# ==========================================================
# Base Report
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
    margin: 60px auto;
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
    padding: 40px 50px;
}

.stage-description {
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 30px;
    color: #374151;
}

.dataset-summary {
    font-size: 15px;
    line-height: 1.8;
    color: #374151;
}

.table-wrapper {
    overflow-x: auto;
}

.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.styled-table th {
    text-align: left;
    padding: 12px;
    background-color: #e6f4f1;
}

.styled-table td {
    padding: 10px;
    border-bottom: 1px solid #e5e7eb;
    vertical-align: top;
}

.num {
    text-align: right;
}

.multiplet {
    font-style: italic;
    background-color: #f8f1e5;
}

.confidence {
    font-weight: bold;
}

.caption {
    font-size: 13px;
    margin-top: 10px;
    font-style: italic;
    color: #6b7280;
}

.interpretation {
    font-size: 14px;
    margin-top: 25px;
    font-style: italic;
    color: #374151;
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

    # Insert header
    html_content = insert_or_replace_block(html_content, "HEADER", build_header_block())

    # Insert dataset
    html_content = insert_or_replace_block(html_content, "DATASET", build_dataset_block())

    # Insert stage0
    html_content = insert_or_replace_block(html_content, "STAGE0", build_stage0_block())

    # Stage1
    df = pd.read_csv(STAGE1_TABLE_PATH, sep="\t")

    prompt = f"""
Interpret CRISPR perturbation summary in 4 concise scientific sentences.

{df.to_string(index=False)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    interpretation = response.choices[0].message.content.strip()

    stage1_block = build_stage1_block(df, interpretation)

    html_content = insert_or_replace_block(html_content, "STAGE1", stage1_block)

    report_path.write_text(html_content, encoding="utf-8")

    print("✅ Header, Dataset, Stage 0 and Stage 1 successfully rendered.")


if __name__ == "__main__":
    main()

