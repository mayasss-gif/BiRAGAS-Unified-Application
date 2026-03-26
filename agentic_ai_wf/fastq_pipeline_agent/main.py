from pathlib import Path

from fastq import run_pipeline


if __name__ == "__main__":
    # Example invocation; adjust parameters as needed for your environment.
    output_dir = run_pipeline(
        input_path=Path("agentic_ai_wf/shared/cohort_data/fastq_data/SRP631214"),
        results_root=Path("agentic_ai_wf/shared/cohort_data/fastq_data/SRP631214/results"),
        disease_name="Breast Cancer",
        combine_after=True,
    )

    print(f"Pipeline completed! Results saved to: {output_dir}")
