#!/usr/bin/env python3
"""Example: run CRISPR screening pipeline and generate HTML report."""

from agentic_ai_wf.crispr_pipeline_agent.screening_crispr import run_screening

result = run_screening(
    input_dir="agentic_ai_wf/crispr_pipeline_agent/input_data/screening_data",
    output_dir="agentic_ai_wf/shared/crispr_data/screening/output_data",
    modes=[3],              # RRA + MLE  (change to [1], [6], [1,3,6], etc.)
    generate_report=True,   # auto-generate HTML report after pipeline finishes
)

print(result.message)

if result.report_path:
    print(f"\nReport: {result.report_path}")

if not result.success:
    print("\nPipeline did not complete successfully.")
    for mr in result.mode_results:
        if not mr.success:
            print(f"  Failed: Mode {mr.mode} — exit code {mr.return_code}")
            print(f"  Command: {mr.command}")
