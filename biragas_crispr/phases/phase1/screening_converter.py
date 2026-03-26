"""
ScreeningToPhase1 — Converts MAGeCK Outputs to Phase 1 Input Files
=====================================================================
Fixes Gaps 1-5: Transforms MAGeCK RRA/MLE/BAGEL2 screening outputs
into the exact 3 CSV files expected by DAGBuilder._load_perturbation().

Input: MAGeCK gene_summary.txt files (RRA + MLE)
Output: CausalDrivers_Ranked.csv, GeneEssentiality_ByMedian.csv,
        causal_link_table_with_relevance.csv

Column mapping:
    RRA neg|lfc → ACE score (calibrated to DepMap scale)
    RRA neg|score → driver significance
    MLE beta → alternative ACE score
    neg|fdr < 0.05 + neg|lfc < -0.5 → "Aggravating" alignment
    Essential gene list membership → essentiality classification
"""

import csv
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("biragas.crispr_phase1.converter")

# Known drug-gene associations (curated subset for common targets)
DRUG_DATABASE = {
    "ABL1": ("imatinib", "Approved"), "ABL2": ("dasatinib", "Approved"),
    "AURKA": ("alisertib", "Phase III"), "BRAF": ("dabrafenib", "Approved"),
    "CDK1": ("dinaciclib", "Phase II"), "CDK4": ("palbociclib", "Approved"),
    "EGFR": ("osimertinib", "Approved"), "ERBB2": ("trastuzumab", "Approved"),
    "JAK1": ("tofacitinib", "Approved"), "JAK2": ("ruxolitinib", "Approved"),
    "KRAS": ("sotorasib", "Approved"), "MTOR": ("everolimus", "Approved"),
    "PIK3CA": ("alpelisib", "Approved"), "VEGFA": ("bevacizumab", "Approved"),
    "TNF": ("adalimumab", "Approved"), "IL6": ("tocilizumab", "Approved"),
    "BTK": ("ibrutinib", "Approved"), "BCL2": ("venetoclax", "Approved"),
}


@dataclass
class ConverterConfig:
    """Configuration for screening → Phase 1 conversion."""
    # ACE calibration
    ace_method: str = "median"         # median, mle_only, rra_only, weighted
    ace_mle_weight: float = 0.6        # Weight for MLE beta in weighted method
    ace_rra_weight: float = 0.4        # Weight for RRA LFC in weighted method
    ace_scale_factor: float = 1.0      # Multiply ACE by this (for DepMap calibration)

    # Driver thresholds
    driver_fdr: float = 0.05           # FDR threshold for driver classification
    driver_lfc: float = -0.3           # LFC threshold for Aggravating alignment
    strong_driver_lfc: float = -0.5    # LFC for Validated Driver verdict
    reversal_lfc: float = 0.3          # Positive LFC threshold for Reversal

    # Essentiality
    essential_bf_threshold: float = 5.0  # BAGEL2 BF for essential
    use_gene_lists: bool = True        # Use essential_genes.txt as fallback


class ScreeningToPhase1:
    """
    Converts MAGeCK screening outputs to BiRAGAS Phase 1 input files.

    Bridges the gap between CRISPR screening pipeline outputs
    (gene_summary.txt with RRA/MLE scores) and the BiRAGAS DAGBuilder's
    expected perturbation_data/ format.

    Usage:
        converter = ScreeningToPhase1()

        # Load MAGeCK results
        converter.load_rra("mode3/.../treatment_control.gene_summary.txt")
        converter.load_mle("mode3/.../treatment_vs_control.gene_summary.txt")

        # Optional: load essentiality data
        converter.load_essential_genes("inputs/essential_genes.txt")
        converter.load_nonessential_genes("inputs/nonessential_genes.txt")

        # Export BiRAGAS-compatible files
        converter.export_phase1_files("./perturbation_data/")
    """

    def __init__(self, config: Optional[ConverterConfig] = None):
        self.config = config or ConverterConfig()
        self.genes: Dict[str, Dict] = {}
        self.essential_set: set = set()
        self.nonessential_set: set = set()

    def load_rra(self, filepath: str):
        """Load MAGeCK RRA gene summary."""
        if not os.path.exists(filepath):
            logger.warning(f"RRA file not found: {filepath}")
            return

        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('id', '').strip()
                if not gene:
                    continue

                g = self.genes.setdefault(gene, {'gene': gene})
                try:
                    g['rra_neg_score'] = float(row.get('neg|score', 1.0))
                    g['rra_neg_pval'] = float(row.get('neg|p-value', 1.0))
                    g['rra_neg_fdr'] = float(row.get('neg|fdr', 1.0))
                    g['rra_neg_rank'] = int(row.get('neg|rank', 0))
                    g['rra_neg_lfc'] = float(row.get('neg|lfc', 0))
                    g['rra_neg_goodsgrna'] = row.get('neg|goodsgrna', '')
                    g['rra_pos_score'] = float(row.get('pos|score', 1.0))
                    g['rra_pos_lfc'] = float(row.get('pos|lfc', 0))
                    g['n_guides'] = int(row.get('num', 0))
                except (ValueError, TypeError):
                    pass

        logger.info(f"Loaded RRA: {len(self.genes)} genes from {os.path.basename(filepath)}")

    def load_mle(self, filepath: str):
        """Load MAGeCK MLE gene summary."""
        if not os.path.exists(filepath):
            logger.warning(f"MLE file not found: {filepath}")
            return

        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('Gene', '').strip()
                if not gene:
                    continue

                g = self.genes.setdefault(gene, {'gene': gene})

                # Find beta/z/p columns dynamically
                for col in row:
                    cl = col.lower()
                    if 'beta' in cl and 'se' not in cl and 'mle_beta' not in g:
                        try:
                            g['mle_beta'] = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if cl.endswith('|z') or cl.endswith('_z'):
                        try:
                            g['mle_z'] = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if 'wald-p' in cl or 'wald_p' in cl:
                        try:
                            g['mle_pval'] = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if 'wald-fdr' in cl or 'wald_fdr' in cl:
                        try:
                            g['mle_fdr'] = float(row[col])
                        except (ValueError, TypeError):
                            pass

        n_with_mle = sum(1 for g in self.genes.values() if 'mle_beta' in g)
        logger.info(f"Loaded MLE: {n_with_mle} genes with beta scores from {os.path.basename(filepath)}")

    def load_essential_genes(self, filepath: str):
        """Load essential gene reference list."""
        if not os.path.exists(filepath):
            return
        with open(filepath) as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('#'):
                    self.essential_set.add(gene)
        logger.info(f"Loaded {len(self.essential_set)} essential genes")

    def load_nonessential_genes(self, filepath: str):
        """Load nonessential gene reference list."""
        if not os.path.exists(filepath):
            return
        with open(filepath) as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('#'):
                    self.nonessential_set.add(gene)
        logger.info(f"Loaded {len(self.nonessential_set)} nonessential genes")

    def load_bagel2(self, filepath: str):
        """Load BAGEL2 Bayes Factor results (if available)."""
        if not os.path.exists(filepath):
            return
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('Gene', row.get('gene', '')).strip()
                if gene in self.genes:
                    try:
                        self.genes[gene]['bayes_factor'] = float(row.get('BF', 0))
                    except (ValueError, TypeError):
                        pass

    def compute_ace_scores(self):
        """
        Compute ACE scores from MAGeCK data.

        ACE calibration: Maps MAGeCK LFC/beta to the ACE scale
        where more negative = stronger driver (matching DepMap CERES convention).

        Methods:
        - median: median of available scores (MLE beta, RRA LFC)
        - mle_only: use only MLE beta
        - rra_only: use only RRA neg|lfc
        - weighted: 0.6 × MLE + 0.4 × RRA
        """
        cfg = self.config

        for g in self.genes.values():
            mle = g.get('mle_beta', None)
            rra = g.get('rra_neg_lfc', None)
            scores = []

            if mle is not None and mle != 0:
                scores.append(mle)
            if rra is not None and rra != 0:
                scores.append(rra)

            if cfg.ace_method == "median" and scores:
                g['ace'] = float(np.median(scores)) * cfg.ace_scale_factor
            elif cfg.ace_method == "mle_only" and mle is not None:
                g['ace'] = mle * cfg.ace_scale_factor
            elif cfg.ace_method == "rra_only" and rra is not None:
                g['ace'] = rra * cfg.ace_scale_factor
            elif cfg.ace_method == "weighted" and scores:
                w_mle = mle * cfg.ace_mle_weight if mle else 0
                w_rra = rra * cfg.ace_rra_weight if rra else 0
                denom = (cfg.ace_mle_weight if mle else 0) + (cfg.ace_rra_weight if rra else 0)
                g['ace'] = (w_mle + w_rra) / max(denom, 0.01) * cfg.ace_scale_factor
            else:
                g['ace'] = 0.0

            # Therapeutic alignment
            fdr = g.get('rra_neg_fdr', g.get('mle_fdr', 1.0))
            ace = g['ace']

            if ace <= cfg.driver_lfc and fdr < cfg.driver_fdr:
                g['alignment'] = 'Aggravating'
                g['verdict'] = 'Validated Driver' if ace <= cfg.strong_driver_lfc else 'Secondary'
            elif ace >= cfg.reversal_lfc:
                g['alignment'] = 'Reversal'
                g['verdict'] = 'Secondary'
            else:
                g['alignment'] = 'Unknown'
                g['verdict'] = 'Unknown'

            # Essentiality
            bf = g.get('bayes_factor', 0)
            gene_name = g['gene']

            if bf >= cfg.essential_bf_threshold:
                g['essentiality'] = 'Core Essential'
            elif cfg.use_gene_lists and gene_name in self.essential_set:
                g['essentiality'] = 'Core Essential'
            elif cfg.use_gene_lists and gene_name in self.nonessential_set:
                g['essentiality'] = 'Non-Essential'
            elif ace <= -0.5:
                g['essentiality'] = 'Tumor-Selective Dependency'
            else:
                g['essentiality'] = 'Non-Essential'

        n_drivers = sum(1 for g in self.genes.values() if g.get('alignment') == 'Aggravating')
        n_essential = sum(1 for g in self.genes.values() if g.get('essentiality') == 'Core Essential')
        logger.info(f"ACE computed: {n_drivers} drivers, {n_essential} essential, {len(self.genes)} total")

    def export_phase1_files(self, output_dir: str):
        """
        Export the 3 BiRAGAS-compatible files for Phase 1 DAGBuilder.

        Creates:
        1. CausalDrivers_Ranked.csv — gene, ACE, TherapeuticAlignment, Verdict, BestEssentialityTag
        2. GeneEssentiality_ByMedian.csv — Gene, IsEssential_byMedianRule
        3. causal_link_table_with_relevance.csv — Gene, Drug, Therapeutic_Relevance
        """
        os.makedirs(output_dir, exist_ok=True)

        if not any('ace' in g for g in self.genes.values()):
            self.compute_ace_scores()

        # Sort by ACE (most negative first = strongest drivers)
        ranked = sorted(self.genes.values(), key=lambda g: g.get('ace', 0))

        # File 1: CausalDrivers_Ranked.csv
        with open(os.path.join(output_dir, "CausalDrivers_Ranked.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["gene", "ACE", "rank", "TherapeuticAlignment", "Verdict", "BestEssentialityTag"])
            for i, g in enumerate(ranked, 1):
                w.writerow([
                    g['gene'],
                    round(g.get('ace', 0), 4),
                    i,
                    g.get('alignment', 'Unknown'),
                    g.get('verdict', 'Unknown'),
                    g.get('essentiality', 'Unknown'),
                ])

        # File 2: GeneEssentiality_ByMedian.csv
        with open(os.path.join(output_dir, "GeneEssentiality_ByMedian.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Gene", "IsEssential_byMedianRule"])
            for g in self.genes.values():
                is_ess = g.get('essentiality') == 'Core Essential'
                w.writerow([g['gene'], is_ess])

        # File 3: causal_link_table_with_relevance.csv
        with open(os.path.join(output_dir, "causal_link_table_with_relevance.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Gene", "Drug", "Therapeutic_Relevance"])
            for g in self.genes.values():
                gene_name = g['gene']
                if gene_name in DRUG_DATABASE:
                    drug, stage = DRUG_DATABASE[gene_name]
                    w.writerow([gene_name, drug, stage])
                else:
                    w.writerow([gene_name, "None", "Unknown"])

        logger.info(f"Exported Phase 1 files to {output_dir}: {len(self.genes)} genes")

    def get_summary(self) -> Dict:
        """Summary statistics."""
        return {
            "total_genes": len(self.genes),
            "with_rra": sum(1 for g in self.genes.values() if 'rra_neg_score' in g),
            "with_mle": sum(1 for g in self.genes.values() if 'mle_beta' in g),
            "aggravating": sum(1 for g in self.genes.values() if g.get('alignment') == 'Aggravating'),
            "reversal": sum(1 for g in self.genes.values() if g.get('alignment') == 'Reversal'),
            "validated_drivers": sum(1 for g in self.genes.values() if g.get('verdict') == 'Validated Driver'),
            "core_essential": sum(1 for g in self.genes.values() if g.get('essentiality') == 'Core Essential'),
            "non_essential": sum(1 for g in self.genes.values() if g.get('essentiality') == 'Non-Essential'),
            "with_drugs": sum(1 for g in self.genes.values() if g['gene'] in DRUG_DATABASE),
            "essential_training": len(self.essential_set),
            "nonessential_training": len(self.nonessential_set),
        }
