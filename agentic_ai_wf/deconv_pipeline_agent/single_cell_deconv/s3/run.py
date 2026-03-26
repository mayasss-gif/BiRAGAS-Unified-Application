# src/bisque_deconv/s3/run.py
from __future__ import annotations
import os
from pathlib import Path
from .api import s3_enhanced_analysis

def _pick_props_file(out_s2: str) -> str:
    """
    Prefer the real R output first, but fall back to legacy/dummy names
    to keep backward compatibility.
    """
    candidates = [
        "Bisque_proportions.csv",        # <-- written by the R step
        "bisque_bulk_proportions.tsv",   # legacy name
        "bisque_deconv_results.tsv",     # very old/dummy S2
    ]
    base = Path(out_s2)
    for name in candidates:
        p = base / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"[S3] None of the expected proportions files found in {out_s2}:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )

def run_s3(
    *,
    out_s2: str,
    metadata_candidate: str | None,
    sample_col: str | None = None,
    condition_col: str | None = None,
    control_label: str | None = None,
    patient_label: str | None = None,
    out_s3: str,
):
    """
    Stage 3: Analyze Bisque deconvolution outputs + optional metadata.
    
    All metadata-related parameters use intelligent auto-detection if None.
    Compatible with all metadata files validated by deg_file_validator_tools.py.
    """
    os.makedirs(out_s3, exist_ok=True)
    print(f"[S3] Starting post-processing. Input (S2): {out_s2}")

    props_path = _pick_props_file(out_s2)
    print(f"[S3] Using proportions file: {props_path}")

    result = s3_enhanced_analysis(
        input_tsv=props_path,  # loader auto-detects CSV/TSV
        metadata_path=metadata_candidate,
        sample_col=sample_col,
        condition_col=condition_col,
        control_label=control_label,
        patient_label=patient_label,
        out_dir=out_s3,
        plots=True,
    )
    print(f"[S3] Analysis outputs in: {out_s3}")
    return result
