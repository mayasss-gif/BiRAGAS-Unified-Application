#!/usr/bin/env python3
"""
main.py — Entry point for the targeted CRISPR-seq pipeline.

Usage:
  uv run main.py
"""

from agentic_ai_wf.crispr_pipeline_agent.targeted import run_targeted_pipeline


def main():

    # Gene-based reference (Mode A)
    # run_targeted_pipeline(
    #     input_dir="PRJNA1240319",
    #     output_dir="output",

    #     # Reference resolution — supply ONE of: target_gene, region, or reference_seq
    #     target_gene="RAB11A",
    #     protospacer="GGTGGATCCTATTCTAAACG",

    # )

    #########################################################
    
    # Region-based reference (Mode B)
    # run_targeted_pipeline(
    #     input_dir="PRJNA1240319",
    #     output_dir="output",
    #     region="chr15:65869459-65891991",
    #     protospacer="GGTGGATCCTATTCTAAACG",
    # )

    #########################################################

    # Direct sequence (Mode C)
    # run_targeted_pipeline(
    #     input_dir="PRJNA1240319",
    #     output_dir="output",
    #     reference_seq="ACGTACGT...",
    #     protospacer="GGTGGATCCTATTCTAAACG",
    # )

    #########################################################

    # SRA workflow — Fetch metadata + download FASTQs

    run_targeted_pipeline(
        input_dir="agentic_ai_wf/crispr_pipeline_agent/input_data",
        output_dir="agentic_ai_wf/shared/crispr_data/targeted/output_data",
        project_id="PRJNA1240319",
        protospacer="GGTGGATCCTATTCTAAACG",
        target_gene="RAB11A",
        extract_metadata=True,
        download_fastq=True,
    )


if __name__ == "__main__":
    main()
