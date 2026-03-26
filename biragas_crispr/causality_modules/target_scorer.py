"""
Phase 2: NETWORK CAUSAL IMPORTANCE — Module 2
TargetScorer (INTENT I_01 Module 3)
====================================
Multi-dimensional target scoring combining centrality, evidence, and therapeutic potential.

5-Dimension Composite: Causal Evidence (0.30), Network Importance (0.25),
Biological Plausibility (0.20), Therapeutic Potential (0.15), Safety Profile (0.10)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TargetScorerConfig:
    w_causal_evidence: float = 0.30
    w_network_importance: float = 0.25
    w_biological_plausibility: float = 0.20
    w_therapeutic_potential: float = 0.15
    w_safety_profile: float = 0.10
    confidence_gate: float = 0.65
    tier_bonus: Dict[str, float] = field(default_factory=lambda: {
        'Tier_1_Master_Regulator': 1.0, 'Tier_2_Secondary_Driver': 0.6,
        'Tier_3_Downstream_Effector': 0.3,
    })


class TargetScorer:
    """Scores and ranks therapeutic targets from enriched DAG."""

    def __init__(self, config: Optional[TargetScorerConfig] = None):
        self.config = config or TargetScorerConfig()

    def score_targets(self, enriched_dag: nx.DiGraph,
                      target_node: str = "Disease_Activity") -> Dict:
        genes = [(n, d) for n, d in enriched_dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']
        if not genes:
            return {'ranked_targets': [], 'summary': {}}

        raw_scores = []
        for gene, data in genes:
            causal = self._score_causal_evidence(gene, data, enriched_dag, target_node)
            network = self._score_network_importance(data)
            bio = self._score_biological_plausibility(gene, data, enriched_dag)
            therapeutic = self._score_therapeutic_potential(data)
            safety = self._score_safety_profile(data)
            raw_scores.append({
                'gene': gene, 'causal_evidence': causal, 'network_importance': network,
                'biological_plausibility': bio, 'therapeutic_potential': therapeutic,
                'safety_profile': safety, 'data': data,
            })

        dims = ['causal_evidence', 'network_importance', 'biological_plausibility',
                'therapeutic_potential', 'safety_profile']
        for dim in dims:
            vals = [s[dim] for s in raw_scores]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin if vmax > vmin else 1.0
            for s in raw_scores:
                s[f'{dim}_norm'] = (s[dim] - vmin) / rng

        cfg = self.config
        ranked = []
        for s in raw_scores:
            composite = (cfg.w_causal_evidence * s['causal_evidence_norm'] +
                         cfg.w_network_importance * s['network_importance_norm'] +
                         cfg.w_biological_plausibility * s['biological_plausibility_norm'] +
                         cfg.w_therapeutic_potential * s['therapeutic_potential_norm'] +
                         cfg.w_safety_profile * s['safety_profile_norm'])

            has_disease_edge = enriched_dag.has_edge(s['gene'], target_node)
            if has_disease_edge:
                composite *= 1.1

            passed_gate = composite >= cfg.confidence_gate
            ranked.append({
                'gene': s['gene'], 'composite_score': round(composite, 4),
                'passed_confidence_gate': passed_gate,
                'dimensions': {d: round(s[d], 4) for d in dims},
                'network_tier': s['data'].get('network_tier', 'Unknown'),
                'causal_tier': s['data'].get('causal_tier', 'Unknown'),
                'has_disease_edge': has_disease_edge,
            })

        ranked.sort(key=lambda x: x['composite_score'], reverse=True)
        for i, r in enumerate(ranked):
            r['rank'] = i + 1

        return {
            'ranked_targets': ranked,
            'summary': {
                'total_scored': len(ranked),
                'passed_gate': sum(1 for r in ranked if r['passed_confidence_gate']),
                'top_5': [r['gene'] for r in ranked[:5]],
            },
        }

    def _score_causal_evidence(self, gene: str, data: Dict, dag: nx.DiGraph,
                                target_node: str) -> float:
        score = 0.0
        if data.get('is_gwas_hit', False):
            score += 0.35
        if dag.has_edge(gene, target_node):
            edge = dag[gene][target_node]
            ev = edge.get('evidence', [])
            if isinstance(ev, set):
                ev = list(ev)
            if 'mendelian_randomization_causality' in ev or 'mendelian_randomization_validated' in ev:
                score += 0.35
            score += min(0.15, edge.get('confidence_score', 0) * 0.15)

        ace = data.get('perturbation_ace', 0)
        if ace <= -0.1:
            score += min(0.15, abs(ace) * 0.3)

        return min(1.0, score)

    def _score_network_importance(self, data: Dict) -> float:
        ci = data.get('causal_importance', 0)
        tier = data.get('network_tier', '')
        tier_bonus = self.config.tier_bonus.get(tier, 0.3)
        apex = data.get('apex_score', 0)
        return min(1.0, ci * 0.5 + tier_bonus * 0.3 + apex * 0.2)

    def _score_biological_plausibility(self, gene: str, data: Dict,
                                        dag: nx.DiGraph) -> float:
        ev_count = data.get('evidence_count', 0)
        n_programs = sum(1 for s in dag.successors(gene)
                         if dag.nodes[s].get('layer') == 'program')
        pleio = data.get('pleiotropic_reach', 0)
        return min(1.0, ev_count * 0.15 + min(n_programs, 10) * 0.05 + pleio * 0.1)

    def _score_therapeutic_potential(self, data: Dict) -> float:
        score = 0.0
        alignment = data.get('therapeutic_alignment', 'Unknown')
        if alignment in ('Aggravating', 'Reversal'):
            score += 0.5
        strategy = data.get('strategy_type', 'Unknown')
        if strategy != 'Unknown':
            score += 0.3
        if data.get('causal_tier') == 'Validated Driver':
            score += 0.2
        return min(1.0, score)

    def _score_safety_profile(self, data: Dict) -> float:
        score = 1.0
        if data.get('systemic_toxicity_risk') == 'High':
            score -= 0.4
        ess = data.get('essentiality_tag', 'Unknown')
        if ess == 'Core Essential':
            score -= 0.3
        elif ess == 'Tumor-Selective Dependency':
            score -= 0.1
        return max(0.0, score)
