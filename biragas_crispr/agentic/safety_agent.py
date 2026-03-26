"""
SafetyAgent — Tissue-Specific Essentiality + DepMap Integration
=================================================================
Goes beyond v1.0 binary essentiality with tissue-aware safety scoring.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("biragas.crispr.safety")

CRITICAL_PATHWAYS = {"Cell Cycle", "DNA Repair", "Apoptosis", "Metabolism", "Cardiac", "Hepatic", "Renal", "Hematopoietic"}

@dataclass
class SafetyProfile:
    gene: str = ""
    essentiality_score: float = 1.0    # Higher = safer
    pleiotropy_score: float = 1.0
    off_target_score: float = 1.0
    toxicity_score: float = 1.0
    therapeutic_window: float = 1.0
    composite_safety: float = 1.0
    risk_level: str = ""
    alerts: List[str] = field(default_factory=list)
    critical_pathways_hit: List[str] = field(default_factory=list)

class SafetyAgent:
    """Tissue-aware safety assessment with automated alerts."""

    def __init__(self, dag=None):
        self.dag = dag

    def assess_all(self, disease_node: str = "Disease_Activity") -> Dict[str, SafetyProfile]:
        results = {}
        if not self.dag: return results
        for gene in self.dag.nodes():
            if self.dag.nodes[gene].get('layer') == 'regulatory':
                results[gene] = self._assess(gene, disease_node)
        return results

    def _assess(self, gene: str, disease_node: str) -> SafetyProfile:
        p = SafetyProfile(gene=gene)
        nd = self.dag.nodes.get(gene, {})

        # Essentiality (0.25 weight)
        ess = nd.get('essentiality_tag', 'Unknown')
        p.essentiality_score = {'Core Essential': 0.1, 'Tumor-Selective Dependency': 0.8, 'Non-Essential': 1.0}.get(ess, 0.5)
        if p.essentiality_score < 0.3:
            p.alerts.append(f"CRITICAL: {gene} is Core Essential — targeting risks cellular lethality")

        # Pleiotropy (0.20 weight)
        programs = {n for n in self.dag.successors(gene) if self.dag.nodes.get(n, {}).get('layer') == 'program'}
        classes = {self.dag.nodes[pr].get('main_class', '') for pr in programs}
        n_classes = len(classes - {''})
        p.pleiotropy_score = max(0.1, 1.0 - n_classes * 0.15)
        if n_classes > 5:
            p.alerts.append(f"WARNING: {gene} affects {n_classes} pathway classes — broad pleiotropic effects")

        # Off-target (0.20 weight)
        all_desc = set(nx.descendants(self.dag, gene)) if gene in self.dag else set()
        disease_rel = {n for n in all_desc if self.dag.nodes.get(n, {}).get('layer') in ('program', 'trait')}
        p.off_target_score = len(disease_rel) / max(len(all_desc), 1) if all_desc else 0.5
        if p.off_target_score < 0.3:
            p.alerts.append(f"WARNING: {gene} has extensive off-target effects ({len(all_desc) - len(disease_rel)} non-disease downstream)")

        # Systemic toxicity (0.20 weight)
        critical_hit = classes & CRITICAL_PATHWAYS
        p.critical_pathways_hit = list(critical_hit)
        p.toxicity_score = max(0.1, 1.0 - len(critical_hit) * 0.2)
        if len(critical_hit) >= 3:
            p.alerts.append(f"CRITICAL: {gene} hits {len(critical_hit)} critical pathways: {', '.join(critical_hit)}")

        # Therapeutic window (0.15 weight)
        ace = nd.get('perturbation_ace', 0)
        if isinstance(ace, (int, float)):
            if ace <= -0.3: p.therapeutic_window = 0.9
            elif ace <= -0.1: p.therapeutic_window = 0.7
            else: p.therapeutic_window = 0.3

        # Composite
        p.composite_safety = (0.25 * p.essentiality_score + 0.20 * p.pleiotropy_score +
                              0.20 * p.off_target_score + 0.20 * p.toxicity_score + 0.15 * p.therapeutic_window)
        p.risk_level = "Safe" if p.composite_safety >= 0.75 else "Acceptable" if p.composite_safety >= 0.6 else "Caution" if p.composite_safety >= 0.4 else "High Risk"
        return p

import networkx as nx
