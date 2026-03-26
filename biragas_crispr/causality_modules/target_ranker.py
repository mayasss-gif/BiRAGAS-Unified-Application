"""
Phase 5: PHARMA INTERVENTION — Module 1
TargetRanker (INTENT I_03 Module 1)
=====================================
Ranks drug targets using integrated causal + pharmacological criteria.

Ranking Dimensions (weights):
  1. Causal Evidence Strength (0.25)  - MR, GWAS, CRISPR, statistical
  2. Network Centrality (0.20)        - causal_importance, tier, apex
  3. Druggability (0.15)              - from DruggabilityScorer
  4. Efficacy Potential (0.15)        - from CounterfactualSimulator
  5. Safety Profile (0.10)            - from SafetyAssessor
  6. Resistance Risk (0.10)           - from ResistanceMechanismIdentifier
  7. Clinical Translatability (0.05)  - existing drugs, clinical trials

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TargetRankerConfig:
    w_causal_evidence: float = 0.25
    w_network_centrality: float = 0.20
    w_druggability: float = 0.15
    w_efficacy: float = 0.15
    w_safety: float = 0.10
    w_resistance: float = 0.10
    w_translatability: float = 0.05
    confidence_gate: float = 0.50
    top_n: int = 20


class TargetRanker:
    """Ranks therapeutic targets using multi-dimensional causal-pharmacological scoring."""

    def __init__(self, config: Optional[TargetRankerConfig] = None):
        self.config = config or TargetRankerConfig()

    def rank_targets(self, dag: nx.DiGraph,
                     druggability_scores: Optional[Dict] = None,
                     efficacy_scores: Optional[Dict] = None,
                     safety_scores: Optional[Dict] = None,
                     resistance_scores: Optional[Dict] = None,
                     disease_node: str = "Disease_Activity") -> Dict:
        """Rank all regulatory genes as drug targets."""
        druggability_scores = druggability_scores or {}
        efficacy_scores = efficacy_scores or {}
        safety_scores = safety_scores or {}
        resistance_scores = resistance_scores or {}

        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']
        if not genes:
            return {'ranked_targets': [], 'summary': {}}

        raw = []
        for gene, data in genes:
            scores = {
                'causal_evidence': self._score_causal_evidence(gene, data, dag, disease_node),
                'network_centrality': self._score_network_centrality(data),
                'druggability': druggability_scores.get(gene, {}).get('score', 0.5),
                'efficacy': efficacy_scores.get(gene, {}).get('score', 0.5),
                'safety': safety_scores.get(gene, {}).get('score', 0.7),
                'resistance': 1.0 - resistance_scores.get(gene, {}).get('resistance_score', 0.3),
                'translatability': self._score_translatability(data),
            }
            raw.append({'gene': gene, 'data': data, **scores})

        dims = ['causal_evidence', 'network_centrality', 'druggability',
                'efficacy', 'safety', 'resistance', 'translatability']
        for dim in dims:
            vals = [r[dim] for r in raw]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin if vmax > vmin else 1.0
            for r in raw:
                r[f'{dim}_norm'] = (r[dim] - vmin) / rng

        cfg = self.config
        weights = [cfg.w_causal_evidence, cfg.w_network_centrality, cfg.w_druggability,
                   cfg.w_efficacy, cfg.w_safety, cfg.w_resistance, cfg.w_translatability]

        ranked = []
        for r in raw:
            composite = sum(w * r[f'{d}_norm'] for w, d in zip(weights, dims))
            ranked.append({
                'gene': r['gene'],
                'composite_score': round(composite, 4),
                'passed_gate': composite >= cfg.confidence_gate,
                'dimensions': {d: round(r[d], 4) for d in dims},
                'network_tier': r['data'].get('network_tier', 'Unknown'),
                'causal_tier': r['data'].get('causal_tier', 'Unknown'),
                'therapeutic_alignment': r['data'].get('therapeutic_alignment', 'Unknown'),
            })

        ranked.sort(key=lambda x: x['composite_score'], reverse=True)
        for i, r in enumerate(ranked):
            r['rank'] = i + 1

        return {
            'ranked_targets': ranked,
            'summary': {
                'total_scored': len(ranked),
                'passed_gate': sum(1 for r in ranked if r['passed_gate']),
                'top_targets': [r['gene'] for r in ranked[:cfg.top_n]],
                'tier_1_targets': [r['gene'] for r in ranked
                                   if r['network_tier'] == 'Tier_1_Master_Regulator'],
            },
        }

    def _score_causal_evidence(self, gene: str, data: Dict,
                                dag: nx.DiGraph, disease_node: str) -> float:
        """Score causal evidence strength."""
        score = 0.0
        if data.get('is_gwas_hit'):
            score += 0.3
        if data.get('causal_tier') == 'Validated Driver':
            score += 0.3

        if dag.has_edge(gene, disease_node):
            edge = dag[gene][disease_node]
            evidence = edge.get('evidence', [])
            if isinstance(evidence, set):
                evidence = list(evidence)
            if any('mendelian_randomization' in str(ev) for ev in evidence):
                score += 0.25
            score += min(0.15, edge.get('confidence_score', 0) * 0.15)

        ace = data.get('perturbation_ace', 0)
        if ace <= -0.1:
            score += min(0.15, abs(ace) * 0.3)

        return min(1.0, score)

    def _score_network_centrality(self, data: Dict) -> float:
        """Score from centrality metrics."""
        ci = data.get('causal_importance', 0)
        apex = data.get('apex_score', 0)
        tier = data.get('network_tier', '')
        tier_scores = {'Tier_1_Master_Regulator': 1.0, 'Tier_2_Secondary_Driver': 0.6,
                       'Tier_3_Downstream_Effector': 0.3}
        return min(1.0, ci * 0.4 + apex * 0.3 + tier_scores.get(tier, 0.3) * 0.3)

    def _score_translatability(self, data: Dict) -> float:
        """Score clinical translatability."""
        score = 0.0
        if data.get('has_approved_drug'):
            score += 0.5
        if data.get('in_clinical_trial'):
            score += 0.3
        if data.get('has_chemical_probe'):
            score += 0.2
        return min(1.0, score) if score > 0 else 0.3
