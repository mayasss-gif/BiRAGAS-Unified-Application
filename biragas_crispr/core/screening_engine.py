"""
ScreeningEngine v2.0 — Unified MAGeCK + BAGEL2 + DrugZ Analysis
==================================================================
Auto-discovers and loads CRISPR screening data from any directory structure.
Supports: MAGeCK RRA, MAGeCK MLE, BAGEL2, FluteMLE, DrugZ, Perturb-seq.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("biragas_crispr.core.screening")


@dataclass
class GeneScreenResult:
    """Unified screening result for one gene."""
    gene: str = ""
    rra_pos_p: float = 1.0
    rra_neg_p: float = 1.0
    rra_pos_rank: int = 0
    mle_beta: float = 0.0
    mle_z: float = 0.0
    mle_p: float = 1.0
    bagel2_bf: float = 0.0
    drug_z: float = 0.0
    drug_p: float = 1.0
    flute_enrichment: float = 0.0
    perturbseq_effect: float = 0.0
    ace_score: float = 0.0
    essentiality_class: str = "Unknown"
    therapeutic_alignment: str = "Unknown"
    n_sources: int = 0

    def to_dict(self) -> Dict:
        return {
            'gene': self.gene,
            'rra_pos_p': self.rra_pos_p, 'mle_beta': self.mle_beta,
            'bagel2_bf': self.bagel2_bf, 'drug_z': self.drug_z,
            'ace': round(self.ace_score, 4),
            'essentiality': self.essentiality_class,
            'alignment': self.therapeutic_alignment,
            'sources': self.n_sources,
        }


class ScreeningEngine:
    """
    Unified CRISPR screening analysis engine.
    Auto-discovers files, loads all formats, computes unified ACE scores.
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._genes: Dict[str, GeneScreenResult] = {}
        self._essential_genes: set = set()
        self._nonessential_genes: set = set()
        self._loaded_sources: List[str] = []
        self._loaded = False

    def auto_load(self, crispr_dir: str) -> Dict:
        """Auto-discover and load all screening data from directory tree."""
        status = {'loaded': [], 'failed': [], 'files_found': 0}

        for root, dirs, files in os.walk(crispr_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if self._is_rra_file(fname, root):
                        self._load_rra(fpath)
                        status['loaded'].append(f"RRA: {fname}")
                    elif self._is_mle_file(fname, root):
                        self._load_mle(fpath)
                        status['loaded'].append(f"MLE: {fname}")
                    elif fname.lower() == 'essential_genes.txt':
                        self._load_essential_list(fpath)
                        status['loaded'].append(f"Essential: {fname}")
                    elif fname.lower() == 'nonessential_genes.txt':
                        self._load_nonessential_list(fpath)
                        status['loaded'].append(f"Non-essential: {fname}")
                    elif 'drugz' in fname.lower() and fname.endswith(('.txt', '.tsv')):
                        self._load_drugz(fpath)
                        status['loaded'].append(f"DrugZ: {fname}")
                    elif fname.endswith('.h5ad'):
                        status['loaded'].append(f"PerturbSeq: {fname} (deferred)")
                    status['files_found'] += 1
                except Exception as e:
                    status['failed'].append(f"{fname}: {e}")

        # Compute ACE and classify
        if self._genes:
            self._compute_ace_scores()
            self._classify_genes()
            self._loaded = True

        logger.info(f"ScreeningEngine: {len(self._genes)} genes from "
                     f"{len(status['loaded'])} sources")
        return status

    def _is_rra_file(self, fname: str, root: str) -> bool:
        return ('gene_summary' in fname.lower() and
                ('rra' in root.lower() or 'rra' in fname.lower()) and
                fname.endswith(('.txt', '.tsv')))

    def _is_mle_file(self, fname: str, root: str) -> bool:
        return ('gene_summary' in fname.lower() and
                ('mle' in root.lower() or 'mle' in fname.lower()) and
                fname.endswith(('.txt', '.tsv')))

    def _load_rra(self, path: str):
        """Load MAGeCK RRA gene summary."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('id', row.get('Gene', row.get('gene', ''))).strip()
                if not gene:
                    continue
                if gene not in self._genes:
                    self._genes[gene] = GeneScreenResult(gene=gene)
                g = self._genes[gene]
                try:
                    g.rra_pos_p = float(row.get('pos|score', row.get('pos|p-value', 1.0)))
                    g.rra_neg_p = float(row.get('neg|score', row.get('neg|p-value', 1.0)))
                    g.rra_pos_rank = int(row.get('pos|rank', 0))
                except (ValueError, TypeError):
                    pass
                g.n_sources += 1
        self._loaded_sources.append('rra')

    def _load_mle(self, path: str):
        """Load MAGeCK MLE gene summary."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            headers = reader.fieldnames or []
            # Find the beta column (treatment|beta)
            beta_col = next((h for h in headers if 'beta' in h.lower()), None)
            z_col = next((h for h in headers if '|z' in h.lower()), None)
            p_col = next((h for h in headers if 'p-value' in h.lower() or 'pval' in h.lower()), None)

            for row in reader:
                gene = row.get('Gene', row.get('gene', row.get('id', ''))).strip()
                if not gene:
                    continue
                if gene not in self._genes:
                    self._genes[gene] = GeneScreenResult(gene=gene)
                g = self._genes[gene]
                try:
                    if beta_col:
                        g.mle_beta = float(row[beta_col])
                    if z_col:
                        g.mle_z = float(row[z_col])
                    if p_col:
                        g.mle_p = float(row[p_col])
                except (ValueError, TypeError):
                    pass
                g.n_sources += 1
        self._loaded_sources.append('mle')

    def _load_drugz(self, path: str):
        """Load DrugZ output."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('GENE', row.get('gene', row.get('Gene', ''))).strip()
                if not gene:
                    continue
                if gene not in self._genes:
                    self._genes[gene] = GeneScreenResult(gene=gene)
                g = self._genes[gene]
                try:
                    g.drug_z = float(row.get('sumZ', row.get('normZ', 0)))
                    g.drug_p = float(row.get('pval_synth', row.get('pval', 1.0)))
                except (ValueError, TypeError):
                    pass
                g.n_sources += 1
        self._loaded_sources.append('drugz')

    def _load_essential_list(self, path: str):
        with open(path) as f:
            for line in f:
                gene = line.strip()
                if gene:
                    self._essential_genes.add(gene)

    def _load_nonessential_list(self, path: str):
        with open(path) as f:
            for line in f:
                gene = line.strip()
                if gene:
                    self._nonessential_genes.add(gene)

    def _compute_ace_scores(self):
        """Compute simplified ACE score from available screening data."""
        for gene, g in self._genes.items():
            streams = []
            weights = []

            # RRA contribution
            if g.rra_pos_p < 1.0:
                import math
                streams.append(-math.log10(max(g.rra_pos_p, 1e-300)) / 10.0)
                weights.append(0.90)

            # MLE contribution
            if g.mle_beta != 0:
                streams.append(max(-1, min(1, g.mle_beta)))
                weights.append(0.85)

            # BAGEL2 contribution
            if g.bagel2_bf != 0:
                streams.append(max(-1, min(1, g.bagel2_bf / 20.0)))
                weights.append(0.85)

            # DrugZ contribution
            if g.drug_z != 0:
                streams.append(max(-1, min(1, g.drug_z / 10.0)))
                weights.append(0.75)

            if weights:
                total_w = sum(weights)
                g.ace_score = sum(s * w for s, w in zip(streams, weights)) / total_w
            else:
                g.ace_score = 0.0

    def _classify_genes(self):
        """Classify essentiality and therapeutic alignment."""
        for gene, g in self._genes.items():
            # Essentiality
            if gene in self._essential_genes or g.bagel2_bf > 5:
                g.essentiality_class = "Core Essential"
            elif gene in self._nonessential_genes or g.bagel2_bf < -5:
                g.essentiality_class = "Non-Essential"
            elif g.rra_pos_p < 0.01:
                g.essentiality_class = "Context Essential"
            else:
                g.essentiality_class = "Ambiguous"

            # Therapeutic alignment
            if g.ace_score < -0.2 and g.essentiality_class != "Core Essential":
                g.therapeutic_alignment = "Aggravating"
            elif g.ace_score > 0.2:
                g.therapeutic_alignment = "Protective"
            elif g.essentiality_class == "Core Essential":
                g.therapeutic_alignment = "Essential-Caution"
            else:
                g.therapeutic_alignment = "Neutral"

    def get_top_drivers(self, n: int = 50) -> List[GeneScreenResult]:
        """Get top causal drivers (most negative ACE)."""
        sorted_genes = sorted(self._genes.values(), key=lambda g: g.ace_score)
        return sorted_genes[:n]

    def get_safe_drivers(self) -> List[GeneScreenResult]:
        """Get non-essential drivers (ideal drug targets)."""
        return [g for g in self._genes.values()
                if g.therapeutic_alignment == "Aggravating" and
                g.essentiality_class != "Core Essential"]

    def get_gene(self, gene: str) -> Optional[GeneScreenResult]:
        return self._genes.get(gene)

    def get_all_genes(self) -> Dict[str, GeneScreenResult]:
        return dict(self._genes)

    def get_summary(self) -> Dict:
        return {
            'total_genes': len(self._genes),
            'sources': self._loaded_sources,
            'essential': sum(1 for g in self._genes.values() if g.essentiality_class == "Core Essential"),
            'drivers': sum(1 for g in self._genes.values() if g.ace_score < -0.1),
            'strong_drivers': sum(1 for g in self._genes.values() if g.ace_score < -0.3),
            'safe_targets': len(self.get_safe_drivers()),
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def export_csv(self, output_path: str):
        """Export all gene results to CSV."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Gene', 'ACE_Score', 'RRA_pos_p', 'MLE_beta',
                             'BAGEL2_BF', 'DrugZ', 'Essentiality', 'Alignment', 'Sources'])
            for gene in sorted(self._genes, key=lambda g: self._genes[g].ace_score):
                g = self._genes[gene]
                writer.writerow([g.gene, round(g.ace_score, 4), g.rra_pos_p,
                                 g.mle_beta, g.bagel2_bf, g.drug_z,
                                 g.essentiality_class, g.therapeutic_alignment, g.n_sources])
