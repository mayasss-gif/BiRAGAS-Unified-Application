#!/usr/bin/env python3
"""
downloading.py

Download FASTQ files from ENA for all runs listed in a ``runs.csv``.

Can be used as:
  - A callable function:  download_fastqs("my_project/")
  - A CLI script:         python downloading.py my_project/
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

GREEN = "\033[92m"
RESET = "\033[0m"
TICK = f"{GREEN}\u2714{RESET}"


def _ena_fastq_urls(run_id: str, layout: str) -> list[str]:
    """Build ENA FTP URLs for a given SRA run."""
    prefix = run_id[:6]
    bucket = run_id[-3:]
    base = f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/{bucket}/{run_id}"

    if layout == "PAIRED":
        return [f"{base}/{run_id}_1.fastq.gz", f"{base}/{run_id}_2.fastq.gz"]
    return [f"{base}/{run_id}.fastq.gz"]


def download_fastqs(project_dir: str) -> str:
    """Download FASTQ files for all runs in ``<project_dir>/runs.csv``.

    Parameters
    ----------
    project_dir : str
        Directory containing ``runs.csv`` (as produced by
        :func:`extractor.extract_metadata`).  A ``fastq/``
        sub-directory is created here for the downloaded files.

    Returns
    -------
    str
        Absolute path to the ``fastq/`` directory.
    """
    pdir = Path(project_dir)
    runs_csv = pdir / "runs.csv"
    fastq_dir = pdir / "fastq"

    if not runs_csv.is_file():
        raise FileNotFoundError(f"runs.csv not found in {pdir}")

    fastq_dir.mkdir(parents=True, exist_ok=True)

    with open(runs_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    paired = 0
    single = 0
    downloaded = 0

    print(f"\n[DOWNLOADER] Starting download of {total} samples\n")

    for idx, row in enumerate(rows, start=1):
        run_id = row["Run"]
        layout = row.get("LibraryLayout", "PAIRED").upper()

        if layout == "PAIRED":
            paired += 1
        else:
            single += 1

        urls = _ena_fastq_urls(run_id, layout)
        print(f"  [{idx}/{total}] {run_id} ", end="", flush=True)

        for url in urls:
            dest = fastq_dir / Path(url).name
            if dest.exists() and dest.stat().st_size > 0:
                print("(cached) ", end="", flush=True)
                downloaded += 1
                continue
            subprocess.run(
                ["wget", "-c", "-q", "-P", str(fastq_dir), url],
                check=True,
            )
            downloaded += 1

        print(TICK)

    print(f"\n[DOWNLOADER] Complete — {total} samples, "
          f"{paired} paired / {single} single, "
          f"{downloaded} FASTQ files")

    return str(fastq_dir.resolve())


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(
        description="Download FASTQ files from ENA using runs.csv",
    )
    ap.add_argument(
        "project_dir",
        help="Directory containing runs.csv",
    )
    args = ap.parse_args()

    try:
        path = download_fastqs(args.project_dir)
        print(f"[DOWNLOADER] FASTQs saved to: {path}")
    except Exception as exc:
        print(f"[DOWNLOADER ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
