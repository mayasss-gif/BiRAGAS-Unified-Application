#!/usr/bin/env python3

import base64
from pathlib import Path
from datetime import datetime


# =====================================================
# Configuration
# =====================================================

LEFT_LOGO = "ABS_left.png"
RIGHT_LOGO = "ABS_Right.png"
DATASET_SUMMARY_PATH = "LLM_support/step-1_datasetsummary.txt"
OUTPUT_HTML = "Stage0:CRISPRModel_Simulator_Report.html"


# =====================================================
# Utility Functions
# =====================================================

def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Logo not found: {image_path}")

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def read_dataset_summary(path):
    """Read dataset summary text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset summary not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# =====================================================
# Report Builder
# =====================================================

def build_report():

    # Encode logos
    left_logo_b64 = encode_image_to_base64(LEFT_LOGO)
    right_logo_b64 = encode_image_to_base64(RIGHT_LOGO)

    # Read dataset summary
    dataset_summary = read_dataset_summary(DATASET_SUMMARY_PATH)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = f"""
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

.header img {{
    height: 70px;
}}

.header-left {{
    position: absolute;
    top: 35px;
    left: 40px;
}}

.header-right {{
    position: absolute;
    top: 35px;
    right: 40px;
}}

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

.badges {{
    text-align: center;
    margin-top: 25px;
}}

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
    margin-top: 0;
    color: #0f766e;
    font-weight: 600;
    margin-bottom: 20px;
}}

.section p {{
    line-height: 1.8;
    font-size: 15px;
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

<!-- ===================================================== -->
<!-- Header -->
<!-- ===================================================== -->

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
        <span class="badge">Platform: Transcriptomic CRISPR Perturbation Profiling</span>
        <span class="badge">Framework: CRISPRModel Simulator</span>
        <span class="badge">Generated: {timestamp}</span>
    </div>

</div>


<!-- ===================================================== -->
<!-- Pipeline Overview -->
<!-- ===================================================== -->

<div class="section">
    <h2>Pipeline Overview</h2>
    <p>
    The CRISPRModel Simulator provides a modular computational framework for the structured analysis 
    of transcriptomic CRISPR perturbation data. The system integrates guide RNA assignment,
    perturbation response characterization, transcriptional signature modeling, predictive classification,
    and causal dependency inference within a unified analytical environment.
    </p>

    <p>
    Through progressive analytical refinement, raw perturbation measurements are transformed into
    interpretable molecular signatures, ranked perturbation profiles, quantitative effect estimates,
    and network-level dependency structures. This design enables robust downstream functional
    interpretation, perturbation prioritization, and systems-level biological inference.
    </p>
</div>


<!-- ===================================================== -->
<!-- Dataset Summary -->
<!-- ===================================================== -->

<div class="section">
    <h2>Dataset Summary</h2>
    <p>
    {dataset_summary}
    </p>
</div>


<!-- ===================================================== -->
<!-- Footer -->
<!-- ===================================================== -->

<div class="footer">
    CRISPRModel Simulator • Transcriptomic Perturbation Modeling Framework
</div>

</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ CRISPRModel Simulator report generated: {OUTPUT_HTML}\n")


if __name__ == "__main__":
    build_report()

