"""
CRISPRTargetScorer — Enhanced 7-Dimension Target Scoring
==========================================================
Extends TargetScorer from 5 to 7 dimensions by adding:
    Dim 6: CRISPR Perturbation Evidence (weight 0.12)
    Dim 7: Knockout Prediction Confidence (weight 0.08)

Original 5 dimensions (re-weighted to sum to 0.80):
    Causal Evidence: 0.22 (was 0.30)
    Network Importance: 0.20 (was 0.25)
    Biological Plausibility: 0.16 (was 0.20)
    Therapeutic Potential: 0.12 (was 0.15)
    Safety Profile: 0.10 (unchanged)

New formula:
    composite = 0.22×CE + 0.20×NI + 0.16×BP + 0.12×TP + 0.10×SP
              + 0.12×CRISPR_evidence + 0.08×knockout_confidence
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase2.scorer")


class CRISPRTargetScorer:
    """
    7-dimension target scoring with CRISPR enhancement.

    Adds two new CRISPR-specific dimensions to the standard 5-dimension
    TargetScorer, providing direct integration of perturbation evidence
    into the target ranking pipeline.

    Usage:
        scorer = CRISPRTargetScorer()
        results = scorer.score_all(dag, knockout_results=ko_dict)
    """

    def __init__(self, confidence_gate: float = 0.65, disease_bonus: float = 1.1):
        self.confidence_gate = confidence_gate
        self.disease_bonus = disease_bonus

        # 7-dimension weights (sum = 1.0)
        self.weights = {
            'causal_evidence': 0.22,
            'network_importance': 0.20,
            'biological_plausibility': 0.16,
            'therapeutic_potential': 0.12,
            'safety_profile': 0.10,
            'crispr_evidence': 0.12,        # NEW
            'knockout_confidence': 0.08,     # NEW
        }

        self.tier_bonus = {
            'Tier_1_Master_Regulator': 1.0,
            'Tier_2_Secondary_Driver': 0.6,
            'Tier_3_Downstream_Effector': 0.3,
        }

    def score_all(self, dag: nx.DiGraph, knockout_results: Optional[Dict] = None,
                  disease_node: str = "Disease_Activity") -> Dict:
        """
        Score all regulatory genes with 7-dimension composite.

        Args:
            dag: Enriched DAG from Phase 2 (with centrality + CRISPR attributes)
            knockout_results: Optional dict of gene → KnockoutResult from CRISPR engine
            disease_node: Disease node name

        Returns:
            Dict with scored_targets, ranked list, statistics
        """
        genes = [n for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory']
        raw_scores = {}

        for gene in genes:
            nd = dag.nodes[gene]
            scores = {}

            # Dim 1: Causal Evidence (0.22)
            scores['causal_evidence'] = self._score_causal_evidence(gene, nd, dag, disease_node)

            # Dim 2: Network Importance (0.20)
            scores['network_importance'] = self._score_network_importance(nd)

            # Dim 3: Biological Plausibility (0.16)
            scores['biological_plausibility'] = self._score_biological_plausibility(gene, nd, dag)

            # Dim 4: Therapeutic Potential (0.12)
            scores['therapeutic_potential'] = self._score_therapeutic_potential(nd)

            # Dim 5: Safety Profile (0.10)
            scores['safety_profile'] = self._score_safety(nd)

            # Dim 6: CRISPR Perturbation Evidence (0.12) — NEW
            scores['crispr_evidence'] = self._score_crispr_evidence(nd)

            # Dim 7: Knockout Prediction Confidence (0.08) — NEW
            scores['knockout_confidence'] = self._score_knockout_confidence(gene, knockout_results)

            raw_scores[gene] = scores

        # Min-max normalize each dimension
        normalized = self._normalize(raw_scores)

        # Compute weighted composite
        results = []
        for gene, scores in normalized.items():
            composite = sum(self.weights[dim] * scores[dim] for dim in self.weights)

            # Disease edge bonus
            has_disease_edge = dag.has_edge(gene, disease_node)
            if has_disease_edge:
                composite *= self.disease_bonus

            passed = composite >= self.confidence_gate
            nd = dag.nodes[gene]

            results.append({
                'gene': gene,
                'composite_score': round(composite, 6),
                'passed_gate': passed,
                'dimensions': {k: round(v, 4) for k, v in scores.items()},
                'tier': nd.get('network_tier', 'Unknown'),
                'ace': round(nd.get('perturbation_ace', 0), 4) if isinstance(nd.get('perturbation_ace'), (int, float)) else 0,
                'essentiality': nd.get('essentiality_tag', 'Unknown'),
                'alignment': nd.get('therapeutic_alignment', 'Unknown'),
                'disease_edge': has_disease_edge,
            })

            # Write to DAG
            nd['target_composite_score_7d'] = round(composite, 6)
            nd['target_passed_gate_7d'] = passed

        results.sort(key=lambda x: -x['composite_score'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return {
            'scored_targets': len(results),
            'passed_gate': sum(1 for r in results if r['passed_gate']),
            'ranked_targets': results,
            'dimensions': list(self.weights.keys()),
            'weights': self.weights,
        }

    def _score_causal_evidence(self, gene, nd, dag, disease_node):
        score = 0.0
        if nd.get('gwas_hit'): score += 0.35
        if dag.has_edge(gene, disease_node):
            ed = dag.edges[gene, disease_node]
            if 'mendelian_randomization' in str(ed.get('evidence_types', '')):
                score += 0.35
            conf = ed.get('confidence', 0)
            if isinstance(conf, (int, float)):
                score += min(0.15, conf * 0.15)
        ace = nd.get('perturbation_ace', 0)
        if isinstance(ace, (int, float)) and ace <= -0.1:
            score += min(0.15, abs(ace) * 0.3)
        return min(1.0, score)

    def _score_network_importance(self, nd):
        ci = nd.get('causal_importance', 0)
        if not isinstance(ci, (int, float)): ci = 0
        tier = nd.get('network_tier', '')
        tier_b = self.tier_bonus.get(tier, 0.3)
        apex = nd.get('apex_score', 0)
        if not isinstance(apex, (int, float)): apex = 0
        return min(1.0, ci * 0.5 + tier_b * 0.3 + apex * 0.2)

    def _score_biological_plausibility(self, gene, nd, dag):
        ev_count = len(str(nd.get('evidence_types', '')).split(','))
        n_prog = sum(1 for n in dag.successors(gene) if dag.nodes.get(n, {}).get('layer') == 'program')
        pleio = nd.get('pleiotropic_reach', 0)
        if not isinstance(pleio, (int, float)): pleio = 0
        return min(1.0, ev_count * 0.15 + min(n_prog, 10) * 0.05 + pleio * 0.1)

    def _score_therapeutic_potential(self, nd):
        score = 0.0
        if nd.get('therapeutic_alignment') in ('Aggravating', 'Reversal'): score += 0.5
        if nd.get('strategy_type') and nd['strategy_type'] != 'Unknown': score += 0.3
        if nd.get('causal_tier') == 'Validated Driver': score += 0.2
        return min(1.0, score)

    def _score_safety(self, nd):
        score = 1.0
        if nd.get('systemic_toxicity_risk') == 'High': score -= 0.4
        ess = nd.get('essentiality_tag', '')
        if ess == 'Core Essential': score -= 0.3
        elif ess == 'Tumor-Selective Dependency': score -= 0.1
        return max(0.0, score)

    def _score_crispr_evidence(self, nd):
        """NEW: CRISPR-specific evidence dimension."""
        score = 0.0
        ace = nd.get('perturbation_ace', 0)
        if isinstance(ace, (int, float)):
            if ace <= -0.3: score += 0.5    # Strong driver
            elif ace <= -0.1: score += 0.3  # Moderate driver

        if nd.get('therapeutic_alignment') == 'Aggravating': score += 0.2
        if nd.get('causal_tier') == 'Validated Driver': score += 0.15

        # Enhanced importance bonus
        eci = nd.get('crispr_enhanced_importance', 0)
        if isinstance(eci, (int, float)) and eci > 0:
            score += min(0.15, eci * 0.1)

        return min(1.0, score)

    def _score_knockout_confidence(self, gene, knockout_results):
        """NEW: Confidence from CRISPR knockout engine predictions."""
        if not knockout_results or gene not in knockout_results:
            return 0.0

        ko = knockout_results[gene]
        if hasattr(ko, 'ensemble'):
            return min(1.0, abs(ko.ensemble) * 2)
        elif hasattr(ko, 'ensemble_score'):
            return min(1.0, abs(ko.ensemble_score) * 2)
        elif isinstance(ko, dict):
            return min(1.0, abs(ko.get('ensemble', ko.get('ensemble_score', 0))) * 2)
        return 0.0

    def _normalize(self, raw_scores: Dict) -> Dict:
        """Min-max normalize each dimension across all genes."""
        if not raw_scores:
            return raw_scores

        dims = list(next(iter(raw_scores.values())).keys())
        mins = {d: float('inf') for d in dims}
        maxs = {d: float('-inf') for d in dims}

        for scores in raw_scores.values():
            for d in dims:
                v = scores[d]
                if v < mins[d]: mins[d] = v
                if v > maxs[d]: maxs[d] = v

        normalized = {}
        for gene, scores in raw_scores.items():
            normalized[gene] = {}
            for d in dims:
                rng = maxs[d] - mins[d]
                normalized[gene][d] = (scores[d] - mins[d]) / rng if rng > 0 else 0.5
        return normalized
