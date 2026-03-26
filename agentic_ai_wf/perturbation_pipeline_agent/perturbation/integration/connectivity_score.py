#!/usr/bin/env python

from pathlib import Path
from logging import Logger
import pandas as pd


def build_connectivity_score(
    deg_path: Path,
    output_dir: Path,
    logger: Logger,
) -> Path:
    """
    Builds the connectivity score for the given DEG file.

    Args:
        deg_path: Path, (e.g. "lupus_DEGs_prioritized.csv")
        output_dir: Path, (e.g. "Integration_Output")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building connectivity score for {deg_path}")
    df = pd.read_csv(deg_path)

    required_cols = ["Gene", "CGC", "PPI_Degree"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DEG file '{deg_path}' is missing required columns: {missing}. "
            "Expected at least: Gene, CGC, PPI_Degree."
        )

    connectivity = df[required_cols].copy()

    # Optional: sort by something (e.g., Composite_Score or Rank if present)
    # if "Rank" in df.columns:
    #     connectivity = connectivity.set_index("Gene").loc[
    #         df.sort_values("Rank")["Gene"]
    #     ].reset_index()

    connectivity.to_csv(output_dir / "Connectivity_Score.csv", index=False)
    logger.info(f"Saved connectivity score table → {output_dir / "Connectivity_Score.csv"}")
    return output_dir / "Connectivity_Score.csv"


# if __name__ == "__main__":
#     build_connectivity_score(
#         deg_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\lupus_DEGs_prioritized.csv"),
#         output_dir=Path("Integration_Output"),
#         logger=logger,
#     )

