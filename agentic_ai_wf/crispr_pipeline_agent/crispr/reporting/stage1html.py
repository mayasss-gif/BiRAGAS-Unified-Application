#!/usr/bin/env python3

import base64
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


def read_api_key(path):
    return Path(path).read_text().strip()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def generate_stage1_interpretation(df, client):

    prompt = f"""
Interpret this CRISPR perturbation assignment summary.
Focus on perturbation coverage, multiplet burden, guide diversity,
and confidence score.

Write 4–5 scientific sentences.
Be concise and analytical.

{df.to_string(index=False)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You write technical CRISPR perturbation interpretations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def dataframe_to_html_table(df):

    df = df.copy()

    # Wrap long text column
    if "top10_guides" in df.columns:
        df["top10_guides"] = df["top10_guides"].apply(
            lambda x: f'<div class="long-text">{x}</div>'
        )

    return df.to_html(index=False, classes="styled-table", escape=False)


def insert_or_replace_block(html_content, block_id, new_block):

    start_tag = f"<!-- STAGE:{block_id} -->"
    end_tag = f"<!-- END:{block_id} -->"

    if start_tag in html_content:
        before = html_content.split(start_tag)[0]
        after = html_content.split(end_tag)[1]
        return before + new_block + after
    else:
        return html_content.replace("</body>", new_block + "\n</body>")


def create_base_report():

    left_logo_b64 = encode_image_to_base64(LEFT_LOGO)
    right_logo_b64 = encode_image_to_base64(RIGHT_LOGO)
    dataset_summary = Path(DATASET_SUMMARY_PATH).read_text().strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CRISPR Perturb-seq Simulator Report</title>

<style>
body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 0;
    background-color: #f4f6f9;
    color: #1a1a1a;
}}

.header {{
    background: linear-gradient(135deg, #0f766e, #14b8a6);
    color: white;
    padding: 60px 40px;
    position: relative;
}}

.header img {{ height: 70px; }}

.header-left {{ position: absolute; top: 35px; left: 40px; }}
.header-right {{ position: absolute; top: 35px; right: 40px; }}

.title {{
    text-align: center;
    font-size: 30px;
    font-weight: 600;
    margin-top: 40px;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    margin-top: 15px;
    max-width: 950px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}}

.badges {{ text-align: center; margin-top: 25px; }}

.badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 8px 18px;
    border-radius: 20px;
    margin: 6px;
    font-size: 14px;
}}

.section {{
    background: white;
    margin: 50px auto;
    padding: 50px;
    width: 85%;
    border-radius: 12px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.05);
}}

.section h2 {{
    color: #0f766e;
}}

.table-container {{
    overflow-x: auto;
    margin-top: 20px;
}}

.styled-table {{
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}}

.styled-table th {{
    background-color: #e0f2f1;
    padding: 12px;
    text-align: left;
}}

.styled-table td {{
    padding: 10px;
    border-bottom: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
}}

.long-text {{
    max-height: 120px;
    overflow-y: auto;
    font-size: 13px;
}}

.caption {{
    font-size: 14px;
    margin-top: 10px;
    color: #555;
}}

.interpretation {{
    font-style: italic;
    margin-top: 20px;
}}

.footer {{
    text-align: center;
    padding: 35px;
    font-size: 13px;
    color: #555;
}}
</style>
</head>

<body>

<!-- STAGE:HEADER -->
<div class="header">
    <div class="header-left">
        <img src="data:image/png;base64,{left_logo_b64}" />
    </div>
    <div class="header-right">
        <img src="data:image/png;base64,{right_logo_b64}" />
    </div>

    <div class="title">
        CRISPR Perturb-seq Simulator Report:<br>
        Guide Assignment, Response Classification, and Downstream Inference
    </div>

    <div class="subtitle">
        End-to-end computational modeling of transcriptomic CRISPR perturbations integrating guide assignment,
        perturbation response deconvolution, differential transcriptional profiling,
        supervised predictive modeling, and network-level causal inference.
    </div>

    <div class="badges">
        <span class="badge">Transcriptomic CRISPR Perturbation Profiling</span>
        <span class="badge">CRISPRModel Simulator</span>
        <span class="badge">Generated: {timestamp}</span>
    </div>
</div>
<!-- END:HEADER -->

<!-- STAGE:DATASET -->
<div class="section">
<h2>Dataset Summary</h2>
<p>{dataset_summary}</p>
</div>
<!-- END:DATASET -->

<div class="footer">
CRISPRModel Simulator • Transcriptomic Perturbation Modeling Framework
</div>

</body>
</html>
"""


def main():

    client = OpenAI(api_key=read_api_key(API_KEY_FILE))
    report_path = Path(OUTPUT_HTML)

    if not report_path.exists():
        report_path.write_text(create_base_report(), encoding="utf-8")

    html_content = report_path.read_text(encoding="utf-8")

    df = pd.read_csv(STAGE1_TABLE_PATH, sep="\t")
    table_html = dataframe_to_html_table(df)
    interpretation = generate_stage1_interpretation(df, client)

    stage1_block = f"""
<!-- STAGE:STAGE1 -->
<div class="section">
<h2>Stage 1: Perturbation Assignment Summary</h2>

<div class="table-container">
{table_html}
</div>

<div class="caption">
Table 1. Summary of perturbation assignment and guide distribution.
</div>

<div class="interpretation">
{interpretation}
</div>
</div>
<!-- END:STAGE1 -->
"""

    updated_html = insert_or_replace_block(html_content, "STAGE1", stage1_block)
    report_path.write_text(updated_html, encoding="utf-8")

    print("✅ Stage 1 updated with proper layout control.")


if __name__ == "__main__":
    main()

