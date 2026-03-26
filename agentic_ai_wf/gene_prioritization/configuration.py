#!/usr/bin/env python3
"""
configuration.py

Pipeline configuration: environment loading, constants, logging, and CLI parsing.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

# Load environment and ensure OPENAI_API_KEY
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "Missing environment variable: OPENAI_API_KEY. "
        "Please set it in your environment or .env file."
    )

# ─── Configuration Constants & Named Thresholds ────────────────────────────────
MAX_FILE_SIZE_MB: int = 500  # Maximum allowed file size
LARGE_FILE_ROW_WARNING: int = 100_000  # Warn if DataFrame rows exceed this

HISTOGRAM_BINS: int = 30
HISTOGRAM_COLOR: str = "skyblue"
HISTOGRAM_EDGE_COLOR: str = "black"

# Fixed threshold cutoffs
FIXED_THRESHOLDS = [
    {"min_abs_log2fc": 4.0, "threshold": 1.0},
    {"min_abs_log2fc": 3.0, "threshold": 0.8},
    {"min_abs_log2fc": 2.0, "threshold": 0.6},
    {"min_abs_log2fc": 1.5, "threshold": 0.4},
    {"min_abs_log2fc": 1.0, "threshold": 0.3},
]
DEFAULT_THRESHOLD: float = 0.5


def configure_logging(output_dir: Union[str, Path], level: str) -> None:
    """
    Initialize logging to stdout and to 'pipeline.log' under output_dir.
    """
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "pipeline.log"

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # file
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root.setLevel(level)
    root.addHandler(ch)
    root.addHandler(fh)


def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments for patient-dir, cohort-dir, output-dir, log-level.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-End DEG Pipeline (Autonomous Agents) with Merging and Annotation"
        )
    )
    parser.add_argument(
        "--patient-dir", type=Path, required=True,
        help="Path to folder containing patient expression files."
    )
    parser.add_argument(
        "--cohort-dir", type=Path, required=True,
        help="Path to folder containing cohort expression files."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory where all outputs will be written."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)."
    )
    return parser.parse_args()