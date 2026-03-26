"""
CRISPR Perturb-seq Reporting Package

Generates comprehensive HTML reports from pipeline stage outputs.

Usage:
    from crispr.reporting import generate_report
    generate_report(sample_dir=Path("crispr_output/GSE90546/GSM2406675_10X001"))
"""

from .report import generate_report

__all__ = ["generate_report"]
