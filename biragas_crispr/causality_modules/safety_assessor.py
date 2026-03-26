"""
Phase 5: PHARMA INTERVENTION — Module 4
SafetyAssessor (INTENT I_03 Module 4)
=======================================
Predicts safety profile of targeting a gene using causal DAG analysis.

Safety Dimensions:
  1. Essentiality Risk (core essential gene = high risk)
  2. Pleiotropy Risk (affects many pathway classes = more side effects)
  3. Off-Target Effects (non-disease downstream nodes)
  4. Systemic Toxicity (organ-critical pathway involvement)
  5. Therapeutic Window (dose sensitivity from ACE gradient)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

CRITICAL_PATHWAYS = {
    'Cell Cycle', 'DNA Repair', 'Apoptosis', 'Metabolism',
    'Cardiac', 'Hepatic', 'Renal', 'Hematopoietic',
}


@dataclass
class SafetyConfig:
    w_essentiality: float = 0.25
    w_pleiotropy: float = 0.20
    w_off_target: float = 0.20
    w_systemic_toxicity: float = 0.20
    w_therapeutic_window: float = 0.15
    safe_threshold: float = 0.60
    max_acceptable_pleiotropy: int = 5


class SafetyAssessor:
    """Assesses safety profile of targeting a gene."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()

    def assess_all(self, dag: nx.DiGraph,
                   disease_node: str = "Disease_Activity") -> Dict[str, Dict]:
        """Assess safety for all regulatory genes."""
        results = {}
        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']

        for gene, data in genes:
            result = self.assess_gene(gene, data, dag, disease_node)
            results[gene] = result
            dag.nodes[gene]['safety_score'] = result['score']
            dag.nodes[gene]['safety_class'] = result['safety_class']

        return results

    def assess_gene(self, gene: str, data: Dict, dag: nx.DiGraph,
                    disease_node: str = "Disease_Activity") -> Dict:
        """Assess safety profile for a single gene."""
        cfg = self.config

        essentiality = self._score_essentiality(data)
        pleiotropy = self._score_pleiotropy(gene, data, dag)
        off_target = self._score_off_target(gene, dag, disease_node)
        toxicity = self._score_systemic_toxicity(gene, data, dag)
        window = self._score_therapeutic_window(data)

        composite = (cfg.w_essentiality * essentiality +
                     cfg.w_pleiotropy * pleiotropy +
                     cfg.w_off_target * off_target +
                     cfg.w_systemic_toxicity * toxicity +
                     cfg.w_therapeutic_window * window)

        if composite >= 0.75:
            safety_class = 'Safe'
        elif composite >= cfg.safe_threshold:
            safety_class = 'Acceptable'
        elif composite >= 0.40:
            safety_class = 'Caution'
        else:
            safety_class = 'High_Risk'

        alerts = self._generate_safety_alerts(gene, data, dag, essentiality,
                                               pleiotropy, off_target, toxicity)

        return {
            'gene': gene,
            'score': round(composite, 4),
            'safety_class': safety_class,
            'dimensions': {
                'essentiality': round(essentiality, 4),
                'pleiotropy': round(pleiotropy, 4),
                'off_target': round(off_target, 4),
                'systemic_toxicity': round(toxicity, 4),
                'therapeutic_window': round(window, 4),
            },
            'alerts': alerts,
        }

    def _score_essentiality(self, data: Dict) -> float:
        """Score essentiality risk (higher = safer)."""
        ess = data.get('essentiality_tag', 'Unknown')
        if ess == 'Non-Essential':
            return 1.0
        elif ess == 'Tumor-Selective Dependency':
            return 0.8
        elif ess == 'Context-Dependent':
            return 0.5
        elif ess == 'Core Essential':
            return 0.1
        return 0.5

    def _score_pleiotropy(self, gene: str, data: Dict,
                           dag: nx.DiGraph) -> float:
        """Score pleiotropy risk (higher = safer = fewer off-pathway effects)."""
        reach = data.get('pleiotropic_reach', 0)
        if reach <= 1:
            return 1.0
        elif reach <= 3:
            return 0.7
        elif reach <= self.config.max_acceptable_pleiotropy:
            return 0.4
        return 0.1

    def _score_off_target(self, gene: str, dag: nx.DiGraph,
                           disease_node: str) -> float:
        """Score off-target effects (higher = safer = fewer off-target nodes)."""
        all_downstream = set(nx.descendants(dag, gene)) if gene in dag else set()
        if not all_downstream:
            return 0.8

        disease_relevant = set()
        if disease_node in dag:
            disease_ancestors = nx.ancestors(dag, disease_node)
            disease_relevant = all_downstream & disease_ancestors

        off_target = all_downstream - disease_relevant
        if len(all_downstream) == 0:
            return 0.8

        off_target_ratio = len(off_target) / len(all_downstream)
        return max(0.0, 1.0 - off_target_ratio)

    def _score_systemic_toxicity(self, gene: str, data: Dict,
                                  dag: nx.DiGraph) -> float:
        """Score systemic toxicity risk (higher = safer)."""
        if data.get('systemic_toxicity_risk') == 'High':
            return 0.1
        elif data.get('systemic_toxicity_risk') == 'Medium':
            return 0.5

        downstream_programs = [s for s in dag.successors(gene)
                               if dag.nodes[s].get('layer') == 'program']
        critical_count = sum(1 for p in downstream_programs
                             if dag.nodes[p].get('main_class', '') in CRITICAL_PATHWAYS)

        if critical_count >= 3:
            return 0.2
        elif critical_count >= 1:
            return 0.5
        return 0.9

    def _score_therapeutic_window(self, data: Dict) -> float:
        """Score therapeutic window from ACE gradient."""
        ace = data.get('perturbation_ace', 0)
        if ace <= -0.3:
            return 0.9
        elif ace <= -0.1:
            return 0.7
        elif ace < 0:
            return 0.5
        return 0.3

    def _generate_safety_alerts(self, gene: str, data: Dict,
                                 dag: nx.DiGraph,
                                 ess: float, pleio: float,
                                 off: float, tox: float) -> List[str]:
        """Generate specific safety alert messages."""
        alerts = []
        if ess < 0.3:
            alerts.append(f'{gene} is a core essential gene - high lethality risk')
        if pleio < 0.3:
            alerts.append(f'{gene} has broad pleiotropic effects across {data.get("pleiotropic_reach", 0)} pathway classes')
        if off < 0.3:
            alerts.append(f'{gene} has extensive off-target downstream effects')
        if tox < 0.3:
            alerts.append(f'{gene} modulates critical systemic pathways')
        return alerts
