"""
DataDiscoveryAgent — Autonomous CRISPR File Discovery
=======================================================
Automatically discovers and catalogs all CRISPR data files across
Screening, Simulator, and Targeted directories without manual paths.

Discovers:
    - Brunello library (brunello_library.tsv)
    - MAGeCK RRA/MLE gene summaries (mode1-5)
    - BAGEL2 essentiality outputs
    - FluteMLE enrichment/pathway data
    - Essential/nonessential gene lists
    - Drug screen results
    - Perturb-seq h5ad files + RF model
    - Targeted amplicon results (cigar/indel)
    - Design matrices and samplesheets
"""

import logging
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger("biragas.crispr.discovery")


@dataclass
class DiscoveredFile:
    """A discovered CRISPR data file."""
    path: str = ""
    filename: str = ""
    file_type: str = ""        # brunello, rra_summary, mle_summary, drug_screen, h5ad, etc.
    category: str = ""         # screening, simulator, targeted, input
    mode: str = ""             # mode1-5 for screening
    size_mb: float = 0.0
    readable: bool = True


@dataclass
class DiscoveryReport:
    """Complete discovery report for a CRISPR data directory."""
    root_dir: str = ""
    total_files: int = 0
    files: List[DiscoveredFile] = field(default_factory=list)

    # Key files (paths)
    brunello_library: str = ""
    rra_gene_summaries: List[str] = field(default_factory=list)
    mle_gene_summaries: List[str] = field(default_factory=list)
    drug_screen_summaries: List[str] = field(default_factory=list)
    essential_genes: str = ""
    nonessential_genes: str = ""
    design_matrices: List[str] = field(default_factory=list)
    count_tables: List[str] = field(default_factory=list)
    h5ad_files: List[str] = field(default_factory=list)
    rf_models: List[str] = field(default_factory=list)
    cigar_files: List[str] = field(default_factory=list)
    html_reports: List[str] = field(default_factory=list)
    flute_enrichment: List[str] = field(default_factory=list)

    # Status
    has_screening: bool = False
    has_simulator: bool = False
    has_targeted: bool = False
    modes_available: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataDiscoveryAgent:
    """
    Autonomous agent that discovers all CRISPR data files in a directory tree.

    Scans recursively for known file patterns and catalogs everything
    for downstream agents to consume.

    Usage:
        agent = DataDiscoveryAgent()
        report = agent.discover("/path/to/CRISPR/")
        print(report.brunello_library)  # Auto-found path
        print(report.modes_available)   # ['mode1', 'mode3', 'mode5']
    """

    # File pattern matching
    PATTERNS = {
        'brunello': lambda f: 'brunello' in f.lower() and f.endswith('.tsv'),
        'rra_summary': lambda f: 'gene_summary' in f.lower() and 'rra' in str(Path(f).parent).lower(),
        'mle_summary': lambda f: 'gene_summary' in f.lower() and ('mle' in str(Path(f).parent).lower() or 'design' in f.lower()),
        'drug_screen': lambda f: 'gene_summary' in f.lower() and ('drug' in f.lower() or 'mode5' in str(Path(f)).lower()),
        'essential_genes': lambda f: f == 'essential_genes.txt',
        'nonessential_genes': lambda f: f == 'nonessential_genes.txt',
        'count_table': lambda f: 'count_table' in f.lower() or 'count.txt' in f.lower(),
        'design_matrix': lambda f: 'design_matrix' in f.lower(),
        'h5ad': lambda f: f.endswith('.h5ad'),
        'rf_model': lambda f: f.endswith('.joblib'),
        'cigar': lambda f: f.endswith('.csv') and 'cigar' in str(Path(f).parent).lower(),
        'indel': lambda f: 'indel' in f.lower() and f.endswith('.csv'),
        'edits': lambda f: 'edits' in f.lower() and f.endswith('.csv'),
        'html_report': lambda f: f.endswith('.html') and ('report' in f.lower() or 'master' in f.lower()),
        'flute_enrichment': lambda f: f.endswith('.txt') and 'enrich' in f.lower(),
        'samplesheet': lambda f: 'samplesheet' in f.lower() and f.endswith('.csv'),
        'sgrna_summary': lambda f: 'sgrna_summary' in f.lower(),
        'pathway_view': lambda f: 'pathwayview' in f.lower().replace('_', ''),
    }

    def __init__(self):
        self._discovered: Dict[str, List[str]] = {}

    def discover(self, root_dir: str, max_depth: int = 6) -> DiscoveryReport:
        """
        Recursively discover all CRISPR data files.

        Args:
            root_dir: Root directory to scan
            max_depth: Maximum directory depth to search

        Returns:
            DiscoveryReport with all found files cataloged
        """
        report = DiscoveryReport(root_dir=root_dir)

        if not os.path.isdir(root_dir):
            report.errors.append(f"Directory not found: {root_dir}")
            return report

        logger.info(f"Discovering CRISPR files in: {root_dir}")

        # Walk directory tree
        for dirpath, dirnames, filenames in os.walk(root_dir):
            depth = dirpath.replace(root_dir, '').count(os.sep)
            if depth > max_depth:
                continue

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                except OSError:
                    size_mb = 0

                # Classify file
                file_type = self._classify_file(filename, filepath)
                if file_type:
                    category = self._get_category(filepath)
                    mode = self._get_mode(filepath)

                    df = DiscoveredFile(
                        path=filepath,
                        filename=filename,
                        file_type=file_type,
                        category=category,
                        mode=mode,
                        size_mb=round(size_mb, 2),
                    )
                    report.files.append(df)
                    report.total_files += 1

                    # Map to specific fields
                    self._map_to_report(report, df)

        # Detect available modules
        report.has_screening = any(f.category == 'screening' for f in report.files)
        report.has_simulator = any(f.category == 'simulator' for f in report.files)
        report.has_targeted = any(f.category == 'targeted' for f in report.files)
        report.modes_available = sorted(set(f.mode for f in report.files if f.mode))

        logger.info(
            f"Discovery complete: {report.total_files} files | "
            f"Screening: {report.has_screening} | Simulator: {report.has_simulator} | "
            f"Targeted: {report.has_targeted} | Modes: {report.modes_available}"
        )

        return report

    def _classify_file(self, filename: str, filepath: str) -> str:
        """Classify a file by matching against known patterns."""
        for file_type, pattern_fn in self.PATTERNS.items():
            try:
                if pattern_fn(filename) or pattern_fn(filepath):
                    return file_type
            except Exception:
                continue

        # Additional classification by extension
        if filename.endswith('.h5ad'):
            return 'h5ad'
        if filename.endswith('.joblib'):
            return 'rf_model'
        if filename.endswith('.mtx') or filename.endswith('.mtx.txt'):
            return 'matrix'

        return ""

    def _get_category(self, filepath: str) -> str:
        """Determine category from directory path."""
        fp = filepath.lower()
        if 'screen' in fp:
            return 'screening'
        elif 'simulat' in fp or 'perturb' in fp:
            return 'simulator'
        elif 'target' in fp or 'amplicon' in fp:
            return 'targeted'
        elif 'input' in fp:
            return 'input'
        return 'other'

    def _get_mode(self, filepath: str) -> str:
        """Extract mode number from path."""
        fp = filepath.lower()
        for m in ['mode1', 'mode2', 'mode3', 'mode4', 'mode5', 'mode6']:
            if m in fp:
                return m
        return ""

    def _map_to_report(self, report: DiscoveryReport, df: DiscoveredFile):
        """Map discovered file to the appropriate report field."""
        if df.file_type == 'brunello':
            report.brunello_library = df.path
        elif df.file_type == 'rra_summary':
            report.rra_gene_summaries.append(df.path)
        elif df.file_type == 'mle_summary':
            report.mle_gene_summaries.append(df.path)
        elif df.file_type == 'drug_screen':
            report.drug_screen_summaries.append(df.path)
        elif df.file_type == 'essential_genes':
            report.essential_genes = df.path
        elif df.file_type == 'nonessential_genes':
            report.nonessential_genes = df.path
        elif df.file_type == 'count_table':
            report.count_tables.append(df.path)
        elif df.file_type == 'design_matrix':
            report.design_matrices.append(df.path)
        elif df.file_type == 'h5ad':
            report.h5ad_files.append(df.path)
        elif df.file_type == 'rf_model':
            report.rf_models.append(df.path)
        elif df.file_type in ('cigar', 'indel', 'edits'):
            report.cigar_files.append(df.path)
        elif df.file_type == 'html_report':
            report.html_reports.append(df.path)
        elif df.file_type == 'flute_enrichment':
            report.flute_enrichment.append(df.path)

    def export_report(self, report: DiscoveryReport, filepath: str):
        """Export discovery report to JSON."""
        data = {
            'root_dir': report.root_dir,
            'total_files': report.total_files,
            'has_screening': report.has_screening,
            'has_simulator': report.has_simulator,
            'has_targeted': report.has_targeted,
            'modes_available': report.modes_available,
            'brunello_library': report.brunello_library,
            'rra_gene_summaries': report.rra_gene_summaries,
            'mle_gene_summaries': report.mle_gene_summaries,
            'drug_screen_summaries': report.drug_screen_summaries,
            'essential_genes': report.essential_genes,
            'h5ad_files': report.h5ad_files,
            'rf_models': report.rf_models,
            'cigar_files': len(report.cigar_files),
            'html_reports': report.html_reports,
            'flute_enrichment': len(report.flute_enrichment),
            'errors': report.errors,
            'files': [{'path': f.path, 'type': f.file_type, 'category': f.category, 'mode': f.mode, 'size_mb': f.size_mb} for f in report.files[:100]],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Discovery report exported: {filepath}")
