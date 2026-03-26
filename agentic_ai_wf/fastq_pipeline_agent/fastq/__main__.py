from __future__ import annotations

import argparse
from pathlib import Path

from .main_module import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FASTQ pipeline.")
    parser.add_argument("input_path", type=Path, help="Path to a folder containing FASTQ files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help='Root directory for outputs (default: "results").',
    )
    parser.add_argument(
        "--disease-name",
        type=str,
        required=True,
        help="Label used to name the output folder (<disease_name>_fastq).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use for the agent (default: gpt-4o).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=120,
        help="Maximum turns allowed for the agent (default: 120).",
    )
    parser.add_argument(
        "--no-combine",
        action="store_false",
        dest="combine_after",
        help="Disable combining per-sample matrices at the end.",
    )

    args = parser.parse_args()

    output_dir = run_pipeline(
        input_path=args.input_path,
        results_root=args.results_root,
        disease_name=args.disease_name,
        combine_after=args.combine_after,
        model=args.model,
        max_turns=args.max_turns,
    )
    print(f"Pipeline completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

