"""
geo_metadata_extractor_v3.py

Robust GEO Series Matrix Metadata Extractor for Agent Pipelines

Features:
- Uses GEOparse to parse Series Matrix metadata
- Automatically selects the most informative condition column
- Explicitly handles .txt and .gz files (not sequencing FASTQ data)
- Detects and blocks invalid input types like `.fastq.gz`
- Produces metadata with sample names for traceability
- Returns JSON with paths and structure compatible with DEG agent
- Fully documented for use in self-coding RNA-seq pipelines
"""

import os
import json
import pandas as pd
import GEOparse
from typing import Optional, List, Tuple
from agents import function_tool
import shutil  

@function_tool
def extract_metadata(
    counts_file: str,
    series_matrix_file: str,
    outdir: str = "prep",
    priority_keyword: Optional[str] = "genotype"
) -> str:
    """
    Extract structured metadata from a GEO Series Matrix file and align it with a count matrix.

    This tool is designed for RNA-seq workflows and will reject sequencing files (.fastq.gz).
    It will parse the GEO metadata, identify the most likely condition column, and output a
    `metadata.csv` file that aligns with the sample IDs from the expression matrix.

    Parameters
    ----------
    counts_file : str
        Path to raw counts matrix file (CSV or TSV with genes × samples).
    series_matrix_file : str
        Path to GEO Series Matrix file (.txt or .gz). Not a FASTQ file.
    outdir : str, optional
        Output directory to store `metadata.csv`. Defaults to "prep".
    priority_keyword : str, optional
        Keyword to prioritize condition extraction column (e.g., "genotype", "disease").

    Returns
    -------
    str
        JSON string with:
        - status: "success" or "error"
        - metadata_file: full path to metadata.csv
        - counts_file: full path to counts file (unchanged)
        - sample_ids: list of sample names
        - condition_column: name of metadata column used
        - log: step-by-step trace for agent interpretability
    """
    os.makedirs(outdir, exist_ok=True)
    log = []

    # Step 1: Validate counts file
    try:
        counts_df = pd.read_csv(counts_file, sep=None, engine="python", index_col=0)
        sample_ids = list(counts_df.columns)
        log.append(f"✅ Loaded count matrix with {len(sample_ids)} samples and {counts_df.shape[0]} genes.")
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"❌ Failed to read counts file: {e}",
            "log": log
        })

    # Step 2: Check for FASTQ or sequencing files (invalid here)
    if series_matrix_file.endswith((".fastq.gz", ".fq.gz", ".fastq", ".fq")):
        return json.dumps({
            "status": "error",
            "message": f"🚫 Invalid input: {series_matrix_file} appears to be a sequencing file (.fastq.gz), not GEO metadata.",
            "log": log
        })

    # Step 3: Parse Series Matrix file using GEOparse
    try:
        gse = GEOparse.get_GEO(filepath=series_matrix_file)
        pheno = gse.phenotype_data.copy()
        log.append(f"✅ Parsed GEO matrix with {pheno.shape[0]} samples and {pheno.shape[1]} metadata fields.")
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"❌ Failed to parse GEO Series Matrix: {e}",
            "log": log
        })

    # Step 4: Match sample IDs if possible
    if set(sample_ids).issubset(set(pheno.index)):
        pheno = pheno.loc[sample_ids]
        log.append("✅ Matched sample IDs between count matrix and metadata.")
    else:
        pheno = pheno.iloc[:len(sample_ids)]
        log.append("⚠️ Sample ID mismatch: falling back to positional alignment.")

    # Step 5: Extract best condition column
    condition_col, condition_values = find_best_metadata_column(pheno, sample_ids, priority_keyword, log)
    if condition_col is None:
        return json.dumps({
            "status": "error",
            "message": "❌ No suitable metadata column found for condition labels.",
            "log": log
        })

    # Step 6: Generate dynamic output filenames based on counts file
    base_name = os.path.splitext(os.path.basename(counts_file))[0].split(".")[0]
    metadata_path = os.path.join(outdir, f"{base_name}_metadata.csv")
    counts_out_path = os.path.join(outdir, f"{base_name}_counts.csv")

    # Save metadata
    metadata_df = pd.DataFrame({
        "sample": sample_ids,
        "condition": condition_values
    })
    metadata_df.to_csv(metadata_path, index=False)
    log.append(f"✅ Saved aligned metadata to {metadata_path}")

    # Save a copy of the counts matrix under the same naming convention
    shutil.copy2(counts_file, counts_out_path)
    log.append(f"✅ Copied original counts file to {counts_out_path}")

    return json.dumps({
        "status": "success",
        "metadata_file": metadata_path,
        "counts_file": counts_out_path,
        "sample_ids": sample_ids,
        "condition_column": condition_col,
        "log": log
    })


def find_best_metadata_column(
    pheno_df: pd.DataFrame,
    sample_ids: List[str],
    keyword: Optional[str],
    log: List[str]
) -> Tuple[Optional[str], Optional[List[str]]]:
    sample_count = len(sample_ids)
    scored_columns = []

    for col in pheno_df.columns:
        values = pheno_df[col].fillna("").astype(str).tolist()
        if abs(len(values) - sample_count) > 1:
            continue

        key_score = 2 if keyword and keyword.lower() in col.lower() else 0
        val_score = sum(keyword.lower() in v.lower() for v in values) if keyword else 0
        uniq_score = len(set(values))
        total = key_score + val_score + uniq_score

        scored_columns.append((total, col, values))
        log.append(f"🧪 Scored column '{col}': total={total} (unique={uniq_score}, keyword_matches={val_score})")

    if not scored_columns:
        return None, None

    scored_columns.sort(reverse=True)
    _, best_col, raw_values = scored_columns[0]
    parsed_values = parse_characteristics(raw_values)
    return best_col, parsed_values


def parse_characteristics(values: List[str]) -> List[str]:
    parsed = []
    for v in values:
        if ":" in v:
            parsed.append(v.split(":", 1)[-1].strip())
        elif "=" in v:
            parsed.append(v.split("=", 1)[-1].strip())
        else:
            parsed.append(v.strip())
    return parsed