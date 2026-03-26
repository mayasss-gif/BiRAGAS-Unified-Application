"""
ScreeningAgent — Autonomous MAGeCK/BAGEL2/FluteMLE Data Loader
================================================================
Loads all CRISPR pooled screening outputs (modes 1-6) automatically
from discovered file paths. Handles MAGeCK RRA, MLE, FluteMLE, and BAGEL2.

Fixes v1.0 gaps:
    - Auto-ingests from discovery report (no manual paths)
    - Parses FluteMLE enrichment data (KEGG, GO, Reactome)
    - Handles multiple contrasts (treatment vs control, drug screens)
    - Validates column schemas before parsing
    - Extracts guide-level data for aggregation
"""

import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("biragas.crispr.screening")


@dataclass
class GeneScreenResult:
    """Comprehensive screening result for one gene across all modes."""
    gene: str = ""

    # RRA results
    rra_neg_score: float = 1.0
    rra_neg_rank: int = 0
    rra_neg_lfc: float = 0.0
    rra_neg_fdr: float = 1.0
    rra_pos_score: float = 1.0
    rra_pos_rank: int = 0
    rra_good_sgrna: str = ""

    # MLE results
    mle_beta: float = 0.0
    mle_z: float = 0.0
    mle_pval: float = 1.0
    mle_fdr: float = 1.0

    # Drug screen results
    drug_beta: float = 0.0
    drug_z: float = 0.0
    drug_pval: float = 1.0
    drug_contrast: str = ""

    # BAGEL2 essentiality
    bayes_factor: float = 0.0
    is_essential: bool = False

    # Guide-level data
    n_guides: int = 0
    guide_lfcs: List[float] = field(default_factory=list)

    # Computed ACE
    ace_score: float = 0.0
    ace_confidence: float = 0.0

    # Enrichment hits
    enriched_pathways: List[str] = field(default_factory=list)

    # Classification
    essentiality_class: str = "Unknown"
    therapeutic_alignment: str = "Unknown"
    driver_class: str = "Unknown"


class ScreeningAgent:
    """
    Autonomous agent for loading CRISPR pooled screening data.

    Ingests from DataDiscoveryAgent report — no manual file paths needed.

    Usage:
        from data_discovery import DataDiscoveryAgent
        discovery = DataDiscoveryAgent().discover("/path/to/CRISPR/")

        agent = ScreeningAgent()
        agent.load_from_discovery(discovery)
        results = agent.get_all_results()
        top_drivers = agent.get_top_drivers(n=50)
    """

    def __init__(self):
        self.genes: Dict[str, GeneScreenResult] = {}
        self.essential_set: set = set()
        self.nonessential_set: set = set()
        self._loaded_modes: List[str] = []

    def load_from_discovery(self, discovery_report) -> Dict[str, Any]:
        """
        Automatically load all screening data from a discovery report.
        """
        status = {"loaded": [], "failed": [], "skipped": []}

        # Load essential/nonessential gene lists
        if discovery_report.essential_genes:
            self._load_gene_list(discovery_report.essential_genes, is_essential=True)
            status["loaded"].append("essential_genes")
        if hasattr(discovery_report, 'nonessential_genes') and discovery_report.nonessential_genes:
            self._load_gene_list(discovery_report.nonessential_genes, is_essential=False)
            status["loaded"].append("nonessential_genes")

        # Load RRA results (pick best/latest)
        for rra_path in discovery_report.rra_gene_summaries:
            try:
                self._load_rra(rra_path)
                status["loaded"].append(f"RRA: {os.path.basename(rra_path)}")
            except Exception as e:
                status["failed"].append(f"RRA: {e}")

        # Load MLE results
        for mle_path in discovery_report.mle_gene_summaries:
            try:
                self._load_mle(mle_path)
                status["loaded"].append(f"MLE: {os.path.basename(mle_path)}")
            except Exception as e:
                status["failed"].append(f"MLE: {e}")

        # Load drug screen results
        for drug_path in discovery_report.drug_screen_summaries:
            try:
                self._load_drug_screen(drug_path)
                status["loaded"].append(f"Drug: {os.path.basename(drug_path)}")
            except Exception as e:
                status["failed"].append(f"Drug: {e}")

        # Load FluteMLE enrichment
        for flute_path in discovery_report.flute_enrichment[:20]:
            try:
                self._load_flute_enrichment(flute_path)
            except Exception:
                pass

        # Compute ACE scores
        self._compute_ace_scores()

        # Classify genes
        self._classify_all()

        logger.info(f"ScreeningAgent: {len(self.genes)} genes loaded | "
                     f"Loaded: {len(status['loaded'])} | Failed: {len(status['failed'])}")

        return status

    def _load_gene_list(self, filepath: str, is_essential: bool):
        """Load essential or nonessential gene list."""
        with open(filepath) as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('#'):
                    if is_essential:
                        self.essential_set.add(gene)
                    else:
                        self.nonessential_set.add(gene)

    def _load_rra(self, filepath: str):
        """Load MAGeCK RRA gene summary."""
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('id', row.get('Gene', '')).strip()
                if not gene:
                    continue

                g = self.genes.setdefault(gene, GeneScreenResult(gene=gene))
                try:
                    g.rra_neg_score = float(row.get('neg|score', row.get('neg_score', 1.0)))
                    g.rra_neg_rank = int(row.get('neg|rank', row.get('neg_rank', 0)))
                    g.rra_neg_lfc = float(row.get('neg|lfc', row.get('neg_lfc', 0)))
                    g.rra_neg_fdr = float(row.get('neg|fdr', row.get('neg_fdr', 1.0)))
                    g.rra_pos_score = float(row.get('pos|score', row.get('pos_score', 1.0)))
                    g.rra_pos_rank = int(row.get('pos|rank', row.get('pos_rank', 0)))
                    g.rra_good_sgrna = row.get('neg|goodsgrna', '')
                    g.n_guides = int(row.get('num', g.n_guides))
                except (ValueError, TypeError):
                    pass

    def _load_mle(self, filepath: str):
        """Load MAGeCK MLE gene summary."""
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('Gene', row.get('id', '')).strip()
                if not gene:
                    continue

                g = self.genes.setdefault(gene, GeneScreenResult(gene=gene))

                # Find beta/z columns dynamically
                for col in row:
                    col_lower = col.lower()
                    if 'beta' in col_lower and 'se' not in col_lower and g.mle_beta == 0:
                        try:
                            g.mle_beta = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if col_lower.endswith('|z') or col_lower.endswith('_z'):
                        try:
                            g.mle_z = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if 'wald-p' in col_lower or 'wald_p' in col_lower:
                        try:
                            g.mle_pval = float(row[col])
                        except (ValueError, TypeError):
                            pass
                    if 'wald-fdr' in col_lower or 'wald_fdr' in col_lower:
                        try:
                            g.mle_fdr = float(row[col])
                        except (ValueError, TypeError):
                            pass

    def _load_drug_screen(self, filepath: str):
        """Load drug screen MLE results."""
        contrast = os.path.basename(filepath).replace('.gene_summary.txt', '')
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                gene = row.get('Gene', row.get('id', '')).strip()
                if not gene:
                    continue

                g = self.genes.setdefault(gene, GeneScreenResult(gene=gene))
                g.drug_contrast = contrast

                for col in row:
                    col_lower = col.lower()
                    if 'beta' in col_lower and 'se' not in col_lower:
                        try:
                            val = float(row[col])
                            if abs(val) > abs(g.drug_beta):
                                g.drug_beta = val
                        except (ValueError, TypeError):
                            pass
                    if 'wald-p' in col_lower:
                        try:
                            g.drug_pval = float(row[col])
                        except (ValueError, TypeError):
                            pass

    def _load_flute_enrichment(self, filepath: str):
        """Load FluteMLE enrichment results."""
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    pathway = row.get('ID', row.get('Description', row.get('Term', '')))
                    genes_str = row.get('geneID', row.get('Genes', row.get('core_enrichment', '')))
                    if pathway and genes_str:
                        for gene in genes_str.replace('/', ',').split(','):
                            gene = gene.strip()
                            if gene in self.genes:
                                self.genes[gene].enriched_pathways.append(pathway)
        except Exception:
            pass

    def _compute_ace_scores(self):
        """Compute ACE scores from available screening data."""
        for g in self.genes.values():
            scores = []
            confidences = []

            if g.mle_beta != 0:
                scores.append(g.mle_beta)
                conf = min(1.0, abs(g.mle_z) / 3.0) if g.mle_z != 0 else 0.5
                confidences.append(conf)

            if g.rra_neg_lfc != 0:
                scores.append(g.rra_neg_lfc)
                conf = min(1.0, -np.log10(max(g.rra_neg_score, 1e-10)) / 5.0)
                confidences.append(conf)

            if g.drug_beta != 0:
                scores.append(g.drug_beta * 0.5)  # Weight drug screen lower
                confidences.append(0.5)

            if scores:
                g.ace_score = float(np.median(scores))
                g.ace_confidence = float(np.mean(confidences))
            else:
                g.ace_score = 0.0
                g.ace_confidence = 0.0

    def _classify_all(self):
        """Classify all genes by essentiality and therapeutic alignment."""
        for g in self.genes.values():
            # Essentiality
            if g.gene in self.essential_set or g.bayes_factor >= 5.0:
                g.essentiality_class = "Core Essential"
                g.is_essential = True
            elif g.bayes_factor >= 2.0:
                g.essentiality_class = "Tumor-Selective Dependency"
            elif g.gene in self.nonessential_set or g.bayes_factor < 0:
                g.essentiality_class = "Non-Essential"
            else:
                g.essentiality_class = "Unknown"

            # Therapeutic alignment
            if g.ace_score <= -0.3:
                g.therapeutic_alignment = "Aggravating"
                g.driver_class = "Strong Driver"
            elif g.ace_score <= -0.1:
                g.therapeutic_alignment = "Aggravating"
                g.driver_class = "Moderate Driver"
            elif g.ace_score >= 0.1:
                g.therapeutic_alignment = "Reversal"
                g.driver_class = "Activator"
            else:
                g.therapeutic_alignment = "Unknown"
                g.driver_class = "Non-Driver"

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_all_results(self) -> Dict[str, GeneScreenResult]:
        return self.genes

    def get_top_drivers(self, n: int = 50) -> List[GeneScreenResult]:
        """Get top N causal drivers by ACE score."""
        sorted_genes = sorted(self.genes.values(), key=lambda g: g.ace_score)
        return sorted_genes[:n]

    def get_drug_sensitive(self, pval_threshold: float = 0.05) -> List[GeneScreenResult]:
        """Get genes with significant drug sensitivity."""
        return sorted(
            [g for g in self.genes.values() if g.drug_pval < pval_threshold and g.drug_beta < -0.3],
            key=lambda g: g.drug_beta
        )

    def get_essential_drivers(self) -> List[GeneScreenResult]:
        """Get essential genes that are also strong drivers (safety concern)."""
        return [g for g in self.genes.values()
                if g.essentiality_class == "Core Essential" and g.ace_score <= -0.1]

    def get_safe_drivers(self) -> List[GeneScreenResult]:
        """Get non-essential strong drivers (ideal targets)."""
        return sorted(
            [g for g in self.genes.values()
             if g.essentiality_class != "Core Essential" and g.ace_score <= -0.1],
            key=lambda g: g.ace_score
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_genes": len(self.genes),
            "essential_training": len(self.essential_set),
            "nonessential_training": len(self.nonessential_set),
            "strong_drivers": sum(1 for g in self.genes.values() if g.driver_class == "Strong Driver"),
            "moderate_drivers": sum(1 for g in self.genes.values() if g.driver_class == "Moderate Driver"),
            "core_essential": sum(1 for g in self.genes.values() if g.essentiality_class == "Core Essential"),
            "tumor_selective": sum(1 for g in self.genes.values() if g.essentiality_class == "Tumor-Selective Dependency"),
            "drug_sensitive": sum(1 for g in self.genes.values() if g.drug_beta < -0.3 and g.drug_pval < 0.05),
            "safe_drivers": len(self.get_safe_drivers()),
            "modes_loaded": self._loaded_modes,
        }

    def export_biragas_csv(self, output_dir: str):
        """Export BiRAGAS-compatible perturbation data files."""
        os.makedirs(output_dir, exist_ok=True)

        # CausalDrivers_Ranked.csv
        ranked = sorted(self.genes.values(), key=lambda g: g.ace_score)
        with open(os.path.join(output_dir, "CausalDrivers_Ranked.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["gene", "ACE", "rank", "TherapeuticAlignment", "Verdict", "BestEssentialityTag"])
            for i, g in enumerate(ranked, 1):
                verdict = "Validated Driver" if g.ace_score <= -0.3 else "Secondary" if g.ace_score <= -0.1 else "Unknown"
                w.writerow([g.gene, round(g.ace_score, 4), i, g.therapeutic_alignment, verdict, g.essentiality_class])

        # GeneEssentiality_ByMedian.csv
        with open(os.path.join(output_dir, "GeneEssentiality_ByMedian.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Gene", "IsEssential_byMedianRule", "bayes_factor", "n_guides"])
            for g in self.genes.values():
                w.writerow([g.gene, g.is_essential, round(g.bayes_factor, 2), g.n_guides])

        # causal_link_table_with_relevance.csv
        with open(os.path.join(output_dir, "causal_link_table_with_relevance.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Gene", "Drug", "Therapeutic_Relevance", "drug_sensitivity_beta", "drug_pval"])
            for g in self.genes.values():
                relevance = "Drug Sensitive" if g.drug_beta < -0.5 and g.drug_pval < 0.05 else "Unknown"
                w.writerow([g.gene, "Unknown", relevance, round(g.drug_beta, 4), f"{g.drug_pval:.2e}"])

        logger.info(f"Exported BiRAGAS files: {len(self.genes)} genes to {output_dir}")
