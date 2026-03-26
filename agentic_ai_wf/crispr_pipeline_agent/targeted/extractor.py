#!/usr/bin/env python3
"""
extractor.py

Fetch SRA run metadata for a BioProject and write ``runs.csv`` to a
specified directory.

Can be used as:
  - A callable function:  extract_metadata("PRJNA1240319", "my_project/")
  - A CLI script:         python extractor.py PRJNA1240319 --output my_project/
"""

import argparse
import csv
import sys
from pathlib import Path

import requests

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _fetch_sra_runinfo(bioproject_id: str) -> str:
    """Query NCBI E-Utilities for the SRA RunInfo CSV."""
    search_url = f"{NCBI_EUTILS}/esearch.fcgi"
    resp = requests.get(
        search_url,
        params={"db": "sra", "term": bioproject_id, "retmode": "json"},
        timeout=60,
    )
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]

    if not ids:
        raise RuntimeError(f"No SRA records found for {bioproject_id}")

    fetch_url = f"{NCBI_EUTILS}/efetch.fcgi"
    resp = requests.get(
        fetch_url,
        params={
            "db": "sra",
            "id": ",".join(ids),
            "rettype": "runinfo",
            "retmode": "text",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.text


def extract_metadata(bioproject_id: str, output_dir: str) -> str:
    """Fetch SRA metadata and write ``runs.csv`` to *output_dir*.

    Parameters
    ----------
    bioproject_id : str
        NCBI BioProject accession (e.g. ``"PRJNA1240319"``).
    output_dir : str
        Directory where ``runs.csv`` will be written.  Created if it
        does not exist.

    Returns
    -------
    str
        Absolute path to the generated ``runs.csv``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACTOR] Fetching SRA metadata for {bioproject_id} ...")
    runinfo_text = _fetch_sra_runinfo(bioproject_id)

    runs_csv = out / "runs.csv"
    runs_csv.write_text(runinfo_text, encoding="utf-8")
    print(f"[EXTRACTOR] Raw run metadata saved: {runs_csv}")

    reader = csv.DictReader(runinfo_text.splitlines())
    rows = list(reader)
    print(f"[EXTRACTOR] Found {len(rows)} SRA runs")

    return str(runs_csv.resolve())


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(
        description="Fetch SRA run metadata for a BioProject",
    )
    ap.add_argument("bioproject_id", help="NCBI BioProject accession (e.g. PRJNA1240319)")
    ap.add_argument(
        "--output", "-o", default=".",
        help="Directory to write runs.csv into (default: current dir)",
    )
    args = ap.parse_args()

    try:
        path = extract_metadata(args.bioproject_id, args.output)
        print(f"[EXTRACTOR] Done: {path}")
    except Exception as exc:
        print(f"[EXTRACTOR ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
