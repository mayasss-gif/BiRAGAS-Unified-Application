from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Dict

from .loader import make_jobs, print_samples, scan_fastq_samples
from .pipeline import run_core_pipeline
from .utils import TeeLogger, setup_logging


def _run_async(coro: Awaitable[Any]) -> Any:
    """Run a coroutine whether or not an event loop already exists."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:  # pragma: no cover - defensive fallback
        if "asyncio.run() cannot be called from a running event loop" in str(exc):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


def run_pipeline(
    input_path: str | Path,
    results_root: str | Path = "results",
    disease_name: str = "disease",
    *,
    combine_after: bool = True,
    model: str = "gpt-4o",
    max_turns: int = 120,
) -> Path:
    """
    Run the FASTQ processing pipeline programmatically.

    Parameters
    ----------
    input_path : str | Path
        Path to a directory containing FASTQ files.
    results_root : str | Path, default "results"
        Root directory where pipeline outputs will be written.
    disease_name : str, default "disease"
        Label used to name the results folder (<disease_name>_fastq).
    combine_after : bool, default True
        Whether to combine per-sample matrices at the end.
    model : str, default "gpt-4o"
        Model name for the autonomous agent runner.
    max_turns : int, default 120
        Maximum turns allowed for the agent.

    Returns
    -------
    Path
        Path to the pipeline output directory.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    results_root = Path(results_root)
    disease_label = disease_name.replace(" ", "_")
    results_dir = results_root / f"{disease_label}_fastq"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging and tee stdout/stderr
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = results_dir / "logs" / f"run_{timestamp}.log"

    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    tee_logger = TeeLogger(log_file)
    sys.stdout = sys.stderr = tee_logger
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting FASTQ pipeline")
    logger.info("Input: %s", input_path)
    logger.info("Results root: %s", results_root)
    logger.info("Disease label: %s", disease_label)

    try:
        samples = scan_fastq_samples(input_path)
        if not samples:
            raise ValueError(f"No FASTQ files found in {input_path}")
        print_samples(samples)

        jobs = make_jobs(samples, results_dir)
        agent_input: Dict[str, Any] = {
            "jobs": jobs,
            "disease_name": disease_label,
            "combine_after": combine_after,
        }

        logger.info("Launching FASTQ pipeline agent...")
        result = _run_async(run_core_pipeline(agent_input, model=model, max_turns=max_turns))
        logger.info("Pipeline run complete.")
        if result:
            print(result)
    finally:
        sys.stdout = previous_stdout
        sys.stderr = previous_stderr

    return results_dir

