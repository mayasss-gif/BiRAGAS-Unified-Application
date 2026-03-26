#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 12 — Global Interpretation and Clinical Translation.

Uses LLM to synthesize a scientific conclusion and clinical interpretation
from the key outputs of all prior stages.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    safe_read_tsv, llm_generate, first_match,
)


def build_stage12_block(conclusion_text: str, clinical_text: str) -> str:
    if not conclusion_text:
        conclusion_text = (
            "<em>Scientific conclusion not available. "
            "Configure OPENAI_API_KEY in .env to enable LLM-driven interpretation.</em>"
        )
    if not clinical_text:
        clinical_text = (
            "<em>Clinical interpretation not available. "
            "Configure OPENAI_API_KEY in .env to enable LLM-driven interpretation.</em>"
        )

    return f"""
<!-- STAGE:STAGE12 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#1e3a8a,#2563eb);">
      Stage 12: Global Interpretation and Clinical Translation
  </div>

  <div class="stage-content">

    <div class="panel-card">
        <div class="panel-title">Integrated Scientific Conclusion</div>
        <div class="interpretation">{conclusion_text}</div>
    </div>

    <div style="margin-top:40px;"></div>

    <div class="panel-card">
        <div class="panel-title">Clinical & Translational Interpretation</div>
        <div class="interpretation">{clinical_text}</div>
    </div>

  </div>
</div>
<!-- END:STAGE12 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 12 block from pipeline outputs in *sample_dir*."""
    s2 = safe_read_tsv(first_match(sample_dir, "**/stage2_mixscape_summary.tsv"))
    s4 = safe_read_tsv(first_match(sample_dir, "**/stage4_deg_top_markers.tsv"))
    s6 = safe_read_tsv(first_match(sample_dir, "stage6_metrics_test.tsv"))
    s9 = safe_read_tsv(first_match(sample_dir, "**/stage9_diagnostics.tsv"))
    s11 = safe_read_tsv(first_match(sample_dir, "**/stage11_cluster_summary.tsv"))

    top_perts = []
    if s2 is not None and "responder_rate" in s2.columns:
        top_perts = (
            s2.sort_values("responder_rate", ascending=False)
            .head(5)["perturbation_id"].tolist()
        )

    gene_context = ""
    if s4 is not None and top_perts:
        for p in top_perts:
            genes = (
                s4[s4["perturbation_id"] == p]
                .sort_values("logfoldchange", ascending=False)
                .head(5)["gene"].tolist()
            )
            gene_context += f"\n{p}: {', '.join(genes)}"

    context = f"""
Top responder perturbations:
{top_perts}

Top DE genes for strongest perturbations:
{gene_context}

Model performance:
{s6.to_string(index=False) if s6 is not None else "NA"}

IV diagnostics:
{s9.to_string(index=False) if s9 is not None else "NA"}

Latent clusters:
{s11.head(10).to_string(index=False) if s11 is not None else "NA"}
"""

    conclusion_text = llm_generate(
        client,
        f"""You are a senior computational systems biologist.

Below are summarized outputs from a CRISPR Perturb-seq modeling framework:

{context}

Write a rigorous scientific conclusion covering:
- Perturbation strength
- Responder modeling quality
- Predictive separability
- Confidence reliability
- Latent structure coherence
- Causal inference strength
- Strengths and limitations

Formal scientific tone. No marketing language.""",
        model="gpt-4o",
        temperature=0.3,
    )

    clinical_text = llm_generate(
        client,
        f"""You are a translational cancer biologist.

Using the perturbations and genes below:

Top perturbations:
{top_perts}

Top DE genes:
{gene_context}

Write one clinically oriented paragraph discussing:
- What biological pathways may be perturbed
- Whether these genes are linked to ER stress, proteostasis, mitochondrial function, UPR, or cell cycle regulation
- Potential therapeutic vulnerabilities
- How this perturbation landscape could inform precision medicine

Be cautious and scientifically grounded. Avoid overstated therapeutic claims.""",
        model="gpt-4o",
        temperature=0.4,
    )

    block = build_stage12_block(conclusion_text, clinical_text)
    return insert_or_replace_block(html, "STAGE12", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 12")
