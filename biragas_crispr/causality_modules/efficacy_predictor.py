"""
Phase 5: PHARMA INTERVENTION — Module 3
EfficacyPredictor (INTENT I_03 Module 3)
==========================================
Predicts drug efficacy using causal DAG simulations.

Efficacy Dimensions:
  1. Causal Effect Magnitude (from CounterfactualSimulator)
  2. Pathway Coverage (fraction of disease-driving pathways affected)
  3. Effect Specificity (on-target vs off-target ratio)
  4. Dose-Response Relationship (ACE gradient)
  5. Patient Coverage (frequency of target across patient DAGs)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EfficacyConfig:
    w_causal_effect: float = 0.30
    w_pathway_coverage: float = 0.25
    w_specificity: float = 0.20
    w_dose_response: float = 0.15
    w_patient_coverage: float = 0.10
    efficacy_threshold: float = 0.50


class EfficacyPredictor:
    """Predicts therapeutic efficacy of targeting a gene."""

    def __init__(self, config: Optional[EfficacyConfig] = None):
        self.config = config or EfficacyConfig()

    def predict_all(self, dag: nx.DiGraph,
                    counterfactual_results: Optional[Dict] = None,
                    disease_node: str = "Disease_Activity") -> Dict[str, Dict]:
        """Predict efficacy for all regulatory genes."""
        counterfactual_results = counterfactual_results or {}
        results = {}

        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']

        for gene, data in genes:
            cf = counterfactual_results.get(gene, {})
            result = self.predict_gene(gene, data, dag, cf, disease_node)
            results[gene] = result
            dag.nodes[gene]['efficacy_score'] = result['score']

        return results

    def predict_gene(self, gene: str, data: Dict, dag: nx.DiGraph,
                     counterfactual: Dict,
                     disease_node: str = "Disease_Activity") -> Dict:
        """Predict efficacy for a single gene target."""
        cfg = self.config

        causal = self._score_causal_effect(counterfactual, data)
        coverage = self._score_pathway_coverage(gene, dag, disease_node)
        specificity = self._score_specificity(gene, dag, disease_node)
        dose_resp = self._score_dose_response(data)
        patient = self._score_patient_coverage(data)

        composite = (cfg.w_causal_effect * causal +
                     cfg.w_pathway_coverage * coverage +
                     cfg.w_specificity * specificity +
                     cfg.w_dose_response * dose_resp +
                     cfg.w_patient_coverage * patient)

        return {
            'gene': gene,
            'score': round(composite, 4),
            'efficacy_class': 'High' if composite >= 0.7 else
                              'Medium' if composite >= cfg.efficacy_threshold else 'Low',
            'dimensions': {
                'causal_effect': round(causal, 4),
                'pathway_coverage': round(coverage, 4),
                'specificity': round(specificity, 4),
                'dose_response': round(dose_resp, 4),
                'patient_coverage': round(patient, 4),
            },
        }

    def _score_causal_effect(self, counterfactual: Dict, data: Dict) -> float:
        """Score based on counterfactual simulation effect magnitude."""
        if counterfactual:
            change = abs(counterfactual.get('relative_change', 0))
            return min(1.0, change * 2.0)

        ace = abs(data.get('perturbation_ace', 0))
        return min(1.0, ace * 2.0)

    def _score_pathway_coverage(self, gene: str, dag: nx.DiGraph,
                                 disease_node: str) -> float:
        """Fraction of disease-driving pathways affected by this gene."""
        all_disease_programs = set()
        if disease_node in dag:
            for pred in dag.predecessors(disease_node):
                if dag.nodes[pred].get('layer') == 'program':
                    all_disease_programs.add(pred)

        if not all_disease_programs:
            return 0.5

        gene_programs = {s for s in dag.successors(gene)
                         if dag.nodes[s].get('layer') == 'program'}
        affected = gene_programs & all_disease_programs

        return len(affected) / len(all_disease_programs)

    def _score_specificity(self, gene: str, dag: nx.DiGraph,
                            disease_node: str) -> float:
        """On-target vs off-target ratio."""
        all_downstream = set(nx.descendants(dag, gene)) if gene in dag else set()
        if not all_downstream:
            return 0.5

        disease_relevant = set()
        if disease_node in dag:
            disease_ancestors = nx.ancestors(dag, disease_node)
            disease_relevant = all_downstream & disease_ancestors

        if len(all_downstream) == 0:
            return 0.5
        return len(disease_relevant) / len(all_downstream)

    def _score_dose_response(self, data: Dict) -> float:
        """Score dose-response relationship from ACE gradient."""
        ace = data.get('perturbation_ace', 0)
        if ace <= -0.2:
            return 0.9
        elif ace <= -0.1:
            return 0.7
        elif ace <= -0.05:
            return 0.5
        elif ace < 0:
            return 0.3
        return 0.2

    def _score_patient_coverage(self, data: Dict) -> float:
        """Score based on target frequency across patient cohort."""
        freq = data.get('patient_frequency', 0.5)
        return min(1.0, freq)
