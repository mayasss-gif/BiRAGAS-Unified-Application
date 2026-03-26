from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

from .fastq_tools import (
    convert_sample_transcripts_to_genes,
    detect_and_generate_metadata_from_sra,
    load_fastqc_summary,
    run_cutadapt,
    run_fastqc,
    run_multiqc,
    run_salmon,
    run_trimmomatic_pe,
    trimming_decision,
)


@function_tool
def make_counts_matrix(
    quant_sf_path: str, output_dir: str, sample_name: str | None = None
) -> str:
    """
    Extract gene-level counts from Salmon quant.sf file and save as counts matrix.
    """
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(quant_sf_path, sep="\t")
    if "Name" not in df or "NumReads" not in df:
        raise ValueError(f"quant.sf at {quant_sf_path} does not have expected columns.")

    if sample_name is None:
        sample_name = Path(output_dir).name.replace("matrices", "").strip("_")

    counts = df[["Name", "NumReads"]].copy()
    counts = counts.rename(columns={"NumReads": sample_name})
    counts.set_index("Name", inplace=True)

    counts_csv_path = (
        Path(output_dir) / f"counts_{sample_name}.csv"
        if sample_name
        else Path(output_dir) / "counts.csv"
    )
    counts.to_csv(counts_csv_path)
    return str(counts_csv_path)


@function_tool
def combine_matrices(
    results_dir: str, dataset_label: str | None = None, disease_name: str | None = None, kind: str = "transcript"
) -> dict:
    """
    Combine per-sample count matrices into a single matrix for downstream analysis.
    """
    from pathlib import Path

    base = Path(results_dir) / dataset_label if dataset_label else Path(results_dir)
    if not base.exists():
        return {"error": f"Directory not found: {base}"}

    count_dfs = []
    for sample_dir in base.iterdir():
        if not sample_dir.is_dir() or sample_dir.name == "combined":
            continue
        mat_dir = sample_dir / "matrices"
        if not mat_dir.exists():
            continue
        pattern = "counts_gene_*.csv" if kind == "gene" else "counts_*.csv"
        for counts_path in mat_dir.glob(pattern):
            df = pd.read_csv(counts_path, index_col=0)
            count_dfs.append(df)

    out_dir = base / "combined"
    out_dir.mkdir(exist_ok=True)

    if not count_dfs:
        return {"error": f"No {kind} matrices were found to combine."}

    combined_counts = pd.concat(count_dfs, axis=1, join="outer").fillna(0)
    try:
        combined_counts = combined_counts.astype(int)
    except Exception:
        pass

    counts_name = f"{disease_name}_{kind}_counts.csv" if disease_name else f"{kind}_counts.csv"
    counts_path = out_dir / counts_name
    combined_counts.to_csv(counts_path)
    return {"counts_matrix": str(counts_path)}


def create_fastq_pipeline_agent(model: str = "gpt-4o") -> Agent:
    """Build the FASTQ processing agent with all available tools."""
    instructions = """
    "You are an autonomous agent for RNA-seq FASTQ preprocessing.\n"

    "Your task is to **fully process every sample** listed in the `jobs` input array. "
    "For each job/sample, strictly follow this workflow end-to-end without asking the user anything.\n"

    "==================== Per-Sample Workflow ====================\n"
    "1. Run FastQC on each input FASTQ file (single-end: R1 only, paired-end: R1 & R2).\n"
    "2. Load and print the FastQC `summary.txt` for each file using `load_fastqc_summary`.\n"
    "3. Analyze the FastQC summaries. If any file shows 'FAIL' or 'WARN' in any of:\n"
    "     - 'Per base sequence quality'\n"
    "     - 'Adapter content'\n"
    "     - 'Overrepresented sequences'\n"
    "   then the sample must be trimmed.\n"
    "4. Decide which trimming method to apply using `trimming_decision`:\n"
    "     - Use Cutadapt if adapter content is flagged.\n"
    "     - Use Trimmomatic if sequence quality or overrepresentation is flagged.\n"
    "     - If no important issues found, skip trimming.\n"
    "   You must decide autonomously and clearly print the decision and reasoning.\n"
    "5. Run the selected trimming tool. Print terminal logs for what was done.\n"
    "6. Re-run FastQC on trimmed output (if trimming used). Print new `summary.txt` and state if quality improved.\n"
    "7. Run Salmon quantification on the final FASTQ files (trimmed if available, else original).\n"

    "8. Use `make_counts_matrix` to extract **transcript-level counts** from Salmon `quant.sf`.\n"
    "   Save per-sample transcript matrix as:\n"
    "     results/{disease_name}_fastq/{sample_name}/matrices/counts_{sample_name}.csv\n"

    "==================== After All Samples ====================\n"
    "9. Immediately after transcript counts are saved, convert that single-sample transcript matrix\n"
    "   into HGNC gene-level counts:\n"
    "     - Call `convert_sample_transcripts_to_genes`\n"
    "       transcript_counts_path = the file from step 8\n"
    "       output_dir = results/{disease_name}_fastq/{sample_name}/matrices/\n"
    "       sample_name = {sample_name}\n"
    "     - This must produce:\n"
    "         results/{disease_name}_fastq/{sample_name}/matrices/counts_gene_{sample_name}.csv\n"

    "10. After all samples are processed, if combine_after = True:\n"
    "   - Call `combine_matrices(kind=\"gene\")`.\n"
    "   - Combine ONLY counts_gene_*.csv files.\n"
    "   - Save combined gene matrix to:\n"
    "         results/{disease_name}_fastq/combined/{disease_name}_gene_counts.csv\n"

    "11. Metadata generation (mandatory if metadata file exists in FASTQ input folder):\n"
    "    - Call `detect_and_generate_metadata_from_sra`\n"
    "       counts_matrix_path = combined gene matrix from step 10\n"
    "       fastq_path = jobs[0].fastq_1\n"
    "    - Save output to:\n"
    "         results/{disease_name}_fastq/combined/metadata_from_sra.csv\n"

    "12. Run `run_multiqc` on the entire results directory and save a MultiQC report to:\n"
    "     results/{disease_name}_fastq/multiqc_report/multiqc_report.html\n"

    "==================== Additional Execution Rules ====================\n"
    "- Print real-time progress logs for each step.\n"
    "- Stream stdout/stderr for external tools (FastQC, trimmers, Salmon, MultiQC) where possible.\n"
    "- If an output already exists for a step, skip re-running it and say so.\n"
    "- Never stop early: iterate through **every** sample in `jobs`.\n"
    "- If trimming does not improve QC, warn but continue.\n"
    "- If any single sample fails, log the error and proceed with remaining samples.\n"
    "- At the end, print a clean summary listing:\n"
    "     (a) per-sample transcript counts paths,\n"
    "     (b) combined transcript matrix path,\n"
    "     (c) combined HGNC gene matrix path,\n"
    "     (d) metadata_from_sra path if generated,\n"
    "     (e) MultiQC report path.\n"
    """

    return Agent(
        name="FASTQ Preprocessing Agent",
        model=model,
        tools=[
            run_fastqc,
            trimming_decision,
            run_cutadapt,
            run_trimmomatic_pe,
            run_salmon,
            make_counts_matrix,
            combine_matrices,
            convert_sample_transcripts_to_genes,
            load_fastqc_summary,
            run_multiqc,
            detect_and_generate_metadata_from_sra,
        ],
        instructions=instructions,
    )


async def run_core_pipeline(
    agent_input: Dict[str, Any],
    model: str = "gpt-4o",
    max_turns: int = 120,
) -> str:
    """Execute the agent-driven FASTQ pipeline asynchronously."""
    agent = create_fastq_pipeline_agent(model=model)
    return await Runner.run(
        starting_agent=agent,
        input=json.dumps(agent_input),
        max_turns=max_turns,
    )

