#!/usr/bin/env python3
"""
S3 (analysis) — Public wrapper
s3_enhanced_analysis
"""
from __future__ import annotations

from __future__ import annotations

from typing import Any, Dict, Optional

from .analyzer import DeconvolutionAnalyzer

__all__ = ["s3_enhanced_analysis"]


def s3_enhanced_analysis(
    input_tsv: str,
    metadata_path: Optional[str] = None,
    sample_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    control_label: Optional[str] = None,
    patient_label: Optional[str] = None,
    out_dir: str = "deconvolution_analysis",
    plots: bool = True,
) -> Dict[str, Any]:
    """
    High-level convenience API for S3 analysis with intelligent auto-detection.

    Parameters
    ----------
    input_tsv : str
        Path to `bisque_bulk_proportions.tsv` (or CSV/TSV with cell types in the first column).
    metadata_path : str | None
        Optional table with cohort labels. Auto-detects columns if not specified.
    sample_col : str | None
        Column in metadata that holds sample IDs. Auto-detected if None.
        Will match against proportions samples and use keywords like 'sample', 'sample_id', etc.
    condition_col : str | None
        Column in metadata with cohort labels. Auto-detected if None.
        Will find categorical columns with 2-10 unique values and keywords like 'condition', 'group', etc.
    control_label : str | None
        Label in `condition_col` that denotes controls. Auto-detected if None.
        Will search for labels containing 'control', 'ctrl', 'normal', 'healthy', etc.
    patient_label : str | None
        Label in `condition_col` that denotes patients/cases. Auto-detected if None.
        Will search for labels containing 'disease', 'patient', 'treatment', etc.
    out_dir : str
        Output directory root for plots/, data/, reports/.
    plots : bool
        If True, attempts to render figures via `s3.plots` (optional module).

    Returns
    -------
    Dict[str, Any]
        Result dictionary from `DeconvolutionAnalyzer.analyze(...)`.
        
    Notes
    -----
    All metadata-related parameters (sample_col, condition_col, control_label, patient_label)
    use intelligent auto-detection by default. This makes the analyzer compatible with
    all metadata files validated by deg_file_validator_tools.py.
    """
    analyzer = DeconvolutionAnalyzer(output_dir=out_dir, plots_enabled=plots)
    return analyzer.analyze(
        proportions_file=input_tsv,
        metadata_file=metadata_path,
        sample_col=sample_col,
        condition_col=condition_col,
        control_label=control_label,
        patient_label=patient_label,
    )
