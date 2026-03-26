#!/usr/bin/env python3
"""
Stage 0a — Dataset Validator for CRISPR Perturb-seq Pipeline

Validates input data structure and integrity BEFORE running Stages 0-12.
Can be called programmatically from run_pipeline() or standalone via CLI.

Required dataset format:
    input_data/DATASET_NAME/
        GSMXXXXXX_10XYYY_matrix.mtx(.gz or .txt)
        GSMXXXXXX_10XYYY_genes.tsv OR features.tsv(.gz)
        GSMXXXXXX_10XYYY_barcodes.tsv(.gz)
        GSMXXXXXX_10XYYY_cell_identities.csv(.gz)

    cell_identities.csv must contain:
        - A barcode column ("barcode" or "cell BC")
        - A guide column ("guide" or "perturb")
        - Control guides: "*" or patterns NTC, NON, CONTROL, SCRAMBLE, NEG
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import mmread

logger = logging.getLogger(__name__)

CONTROL_PATTERNS = ["NTC", "NON", "CONTROL", "SCRAMBLE", "NEG"]


# ------------------------------------------------------------------ #
#  File discovery
# ------------------------------------------------------------------ #

def find_sample_ids(gse_dir: Path) -> List[str]:
    """Discover GSM sample IDs from filenames in a GSE directory."""
    samples = set()
    for p in gse_dir.glob("GSM*"):
        parts = p.name.split("_")
        if len(parts) >= 2:
            samples.add(parts[0] + "_" + parts[1])
    return sorted(samples)


def locate_files(gse_dir: Path, sample_id: str) -> Dict[str, Optional[Path]]:
    """Locate the required input files for a given sample ID."""
    def first(patterns):
        for pat in patterns:
            hits = list(gse_dir.glob(f"{sample_id}*{pat}*"))
            if hits:
                return hits[0]
        return None

    return {
        "matrix": first(["matrix.mtx", "matrix.mtx.txt"]),
        "genes": first(["genes.tsv", "features.tsv"]),
        "barcodes": first(["barcodes.tsv"]),
        "cell_id": first(["cell_identities.csv"]),
    }


# ------------------------------------------------------------------ #
#  Column detection helpers
# ------------------------------------------------------------------ #

def detect_barcode_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "barcode" in c.lower() or "cell bc" in c.lower():
            return c
    return None


def detect_guide_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "guide" in c.lower() or "perturb" in c.lower():
            return c
    return None


def detect_controls(series: pd.Series) -> int:
    s = series.astype(str).str.upper()
    regex_controls = s.str.contains("|".join(CONTROL_PATTERNS), regex=True)
    star_controls = s == "*"
    return int((regex_controls | star_controls).sum())


# ------------------------------------------------------------------ #
#  Per-sample validation
# ------------------------------------------------------------------ #

def validate_sample(gse_dir: Path, sample_id: str) -> Dict:
    """
    Validate a single sample's files and return a result dict.

    Returns a dict with keys:
        sample_id, n_cells, n_genes, n_guides_detected, n_control_like,
        matrix_shape, errors (list), warnings (list)
    """
    result = {
        "sample_id": sample_id,
        "n_cells": None,
        "n_genes": None,
        "n_guides_detected": None,
        "n_control_like": None,
        "matrix_shape": None,
        "errors": [],
        "warnings": [],
    }

    files = locate_files(gse_dir, sample_id)

    for key in ["matrix", "genes", "barcodes"]:
        if files[key] is None:
            result["errors"].append(f"Missing required file: {key}")

    if result["errors"]:
        return result

    try:
        shape = mmread(files["matrix"]).shape
        result["matrix_shape"] = tuple(int(x) for x in shape)
    except Exception as e:
        result["errors"].append(f"Failed to read matrix: {e}")
        return result

    try:
        genes = pd.read_csv(files["genes"], sep="\t", header=None)
        barcodes = pd.read_csv(files["barcodes"], sep="\t", header=None)

        n_genes = int(genes.shape[0])
        n_cells = int(barcodes.shape[0])

        result["n_genes"] = n_genes
        result["n_cells"] = n_cells

        shape = result["matrix_shape"]

        if shape not in [(n_genes, n_cells), (n_cells, n_genes)]:
            result["errors"].append(
                f"Matrix shape {shape} inconsistent with genes={n_genes}, cells={n_cells}"
            )

    except Exception as e:
        result["errors"].append(f"Gene/barcode validation failed: {e}")
        return result

    if files["cell_id"] is None:
        result["warnings"].append("No cell_identities file found (Stage 1 may fail)")
        return result

    try:
        meta = pd.read_csv(files["cell_id"])
        bc_col = detect_barcode_column(meta)
        guide_col = detect_guide_column(meta)

        if bc_col is None:
            result["errors"].append("No barcode column detected in cell_identities")

        if guide_col is None:
            result["errors"].append("No guide/perturbation column detected")

        if bc_col and guide_col:
            result["n_guides_detected"] = int(meta[guide_col].notna().sum())
            result["n_control_like"] = detect_controls(meta[guide_col])

            vc = meta[guide_col].value_counts()
            small_perts = vc[vc < 5]

            if len(small_perts) > 0:
                result["warnings"].append(
                    f"{len(small_perts)} perturbations have <5 cells"
                )

            if result["n_control_like"] == 0:
                result["warnings"].append(
                    "No control-like guides detected (Stage 2 risk)"
                )

    except Exception as e:
        result["errors"].append(f"Failed reading cell_identities: {e}")

    return result


# ------------------------------------------------------------------ #
#  Public API — validate entire GSE directory
# ------------------------------------------------------------------ #

def validate_dataset(
    gse_dir: Path,
    samples: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[bool, List[Dict]]:
    """
    Validate all (or selected) samples in a GSE directory.

    Parameters
    ----------
    gse_dir : Path
        Path to the GSE input directory.
    samples : list of str, optional
        Specific sample IDs to validate. If None, all detected samples
        are validated.
    out_dir : Path, optional
        If provided, writes ``validation_summary.csv`` and
        ``validation_report.txt`` to this directory.

    Returns
    -------
    (is_valid, results) : tuple
        is_valid : bool
            True if no errors were found across all samples.
        results : list of dict
            Per-sample validation results.
    """
    gse_dir = Path(gse_dir)
    if not gse_dir.exists():
        raise FileNotFoundError(f"GSE directory not found: {gse_dir}")

    all_sample_ids = find_sample_ids(gse_dir)

    if samples is not None:
        sample_ids = [s for s in samples if s in all_sample_ids]
        missing = [s for s in samples if s not in all_sample_ids]
        if missing:
            logger.warning("Samples not found in %s: %s", gse_dir, missing)
    else:
        sample_ids = all_sample_ids

    if not sample_ids:
        raise ValueError(f"No GSM-style samples detected in {gse_dir}")

    results = []
    for sid in sample_ids:
        logger.info("Validating %s", sid)
        results.append(validate_sample(gse_dir, sid))

    total_errors = sum(len(r["errors"]) for r in results)
    is_valid = total_errors == 0

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(results, out_dir)

    return is_valid, results


def format_validation_report(results: List[Dict]) -> str:
    """Format validation results into a human-readable string."""
    lines = []
    for r in results:
        status = "PASS" if not r["errors"] else "FAIL"
        lines.append(f"  [{status}] {r['sample_id']}")
        lines.append(f"         cells={r['n_cells']}  genes={r['n_genes']}  "
                      f"guides={r['n_guides_detected']}  controls={r['n_control_like']}  "
                      f"matrix={r['matrix_shape']}")
        for e in r["errors"]:
            lines.append(f"         ERROR: {e}")
        for w in r["warnings"]:
            lines.append(f"         WARN:  {w}")
    return "\n".join(lines)


def _write_outputs(results: List[Dict], out_dir: Path):
    """Write CSV summary and text report to disk."""
    df = pd.DataFrame(results)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: int(x) if isinstance(x, np.integer) else x
        )
    df.to_csv(out_dir / "validation_summary.csv", index=False)

    with (out_dir / "validation_report.txt").open("w") as f:
        f.write("CRISPR Pipeline — Dataset Validation Report\n")
        f.write("=" * 50 + "\n\n")
        for row in results:
            f.write(f"\n=== Sample: {row['sample_id']} ===\n")
            for k, v in row.items():
                f.write(f"  {k}: {v}\n")


# ------------------------------------------------------------------ #
#  CLI entry point
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Validate CRISPR Perturb-seq input dataset structure"
    )
    ap.add_argument("--gse_dir", required=True, help="Path to GSE input directory")
    ap.add_argument("--out_dir", default="validation_output", help="Output directory")
    args = ap.parse_args()

    gse_dir = Path(args.gse_dir)
    out_dir = Path(args.out_dir)

    print("\n" + "=" * 60)
    print("Stage 0a: Dataset Validation")
    print("=" * 60)

    is_valid, results = validate_dataset(gse_dir, out_dir=out_dir)

    print(format_validation_report(results))

    if is_valid:
        print(f"\n[PASS] Dataset is pipeline-ready.")
    else:
        print(f"\n[FAIL] Dataset has errors. See {out_dir}/validation_summary.csv")
        raise SystemExit(1)

    print(f"[DONE] Validation outputs in {out_dir}/\n")


if __name__ == "__main__":
    main()
