"""
CRISPRDrugTargetRanker — Unified 9-Dimension Drug Target Ranking
===================================================================
Extends Phase 5 TargetRanker from 7 to 9 dimensions by adding:
    Dim 8: CRISPR Perturbation Strength (weight 0.08)
    Dim 9: Knockout Ensemble Confidence (weight 0.06)

Also integrates CRISPRTargetScorer's 7D output as a cross-validation signal.

Original 7 dimensions re-weighted (sum 0.86):
    Causal Evidence: 0.20 (was 0.25) | Network Centrality: 0.16 (was 0.20)
    Druggability: 0.12 (was 0.15) | Efficacy: 0.12 (was 0.15)
    Safety: 0.10 (unchanged) | Resistance: 0.08 (was 0.10) | Translatability: 0.08 (was 0.05)

New 2 dimensions (sum 0.14):
    CRISPR Perturbation: 0.08 | Knockout Confidence: 0.06
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase5.ranker")


class CRISPRDrugTargetRanker:
    """
    9-dimension drug target ranking integrating BiRAGAS Phase 5 + CRISPR engines.

    Merges TargetRanker (7D) with CRISPR evidence (2D) for the most
    comprehensive drug target prioritization available.

    Usage:
        ranker = CRISPRDrugTargetRanker()
        results = ranker.rank(dag, knockout_results=ko_dict,
                              druggability=drug_scores, efficacy=eff_scores,
                              safety=safety_scores, resistance=resist_scores)
    """

    WEIGHTS = {
        'causal_evidence': 0.20,
        'network_centrality': 0.16,
        'druggability': 0.12,
        'efficacy': 0.12,
        'safety': 0.10,
        'resistance': 0.08,
        'translatability': 0.08,
        'crispr_perturbation': 0.08,
        'knockout_confidence': 0.06,
    }

    def __init__(self, confidence_gate: float = 0.50, disease_bonus: float = 1.1):
        self.confidence_gate = confidence_gate
        self.disease_bonus = disease_bonus

    def rank(self, dag: nx.DiGraph,
             knockout_results: Optional[Dict] = None,
             druggability: Optional[Dict] = None,
             efficacy: Optional[Dict] = None,
             safety: Optional[Dict] = None,
             resistance: Optional[Dict] = None,
             disease_node: str = "Disease_Activity") -> Dict:
        """
        Rank all regulatory genes with 9-dimension composite.
        """
        genes = [n for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory']
        raw = {}

        for gene in genes:
            nd = dag.nodes[gene]
            scores = {}

            # Dim 1: Causal Evidence (0.20)
            s = 0.0
            if nd.get('gwas_hit'): s += 0.30
            if nd.get('causal_tier') == 'Validated Driver': s += 0.30
            ace = nd.get('perturbation_ace', 0)
            if isinstance(ace, (int, float)) and ace <= -0.1:
                s += min(0.15, abs(ace) * 0.3)
            if dag.has_edge(gene, disease_node):
                ed = dag.edges[gene, disease_node]
                c = ed.get('confidence', ed.get('confidence_score', 0.5))
                if isinstance(c, (int, float)):
                    s += min(0.15, c * 0.15)
                if 'mendelian_randomization' in str(ed.get('evidence_types', '')):
                    s += 0.25
            scores['causal_evidence'] = min(1.0, s)

            # Dim 2: Network Centrality (0.16)
            ci = nd.get('causal_importance', 0)
            tier = nd.get('network_tier', '')
            tier_b = {'Tier_1_Master_Regulator': 1.0, 'Tier_2_Secondary_Driver': 0.6, 'Tier_3_Downstream_Effector': 0.3}.get(tier, 0.3)
            apex = nd.get('apex_score', 0)
            scores['network_centrality'] = min(1.0, (ci if isinstance(ci, (int, float)) else 0) * 0.5 + tier_b * 0.3 + (apex if isinstance(apex, (int, float)) else 0) * 0.2)

            # Dim 3: Druggability (0.12)
            if druggability and gene in druggability:
                ds = druggability[gene]
                scores['druggability'] = ds.get('score', ds.get('druggability_score', 0.5)) if isinstance(ds, dict) else 0.5
            else:
                scores['druggability'] = nd.get('druggability_score', 0.5)

            # Dim 4: Efficacy (0.12)
            if efficacy and gene in efficacy:
                es = efficacy[gene]
                scores['efficacy'] = es.get('score', es.get('efficacy_score', 0.5)) if isinstance(es, dict) else 0.5
            else:
                scores['efficacy'] = nd.get('efficacy_score', 0.5)

            # Dim 5: Safety (0.10)
            if safety and gene in safety:
                ss = safety[gene]
                scores['safety'] = ss.get('score', ss.get('safety_score', 0.7)) if isinstance(ss, dict) else 0.7
            else:
                scores['safety'] = nd.get('safety_score', 0.7)

            # Dim 6: Resistance (0.08, inverted)
            if resistance and gene in resistance:
                rs = resistance[gene]
                r_score = rs.get('resistance_score', 0.3) if isinstance(rs, dict) else 0.3
            else:
                r_score = nd.get('unified_resistance_score', nd.get('resistance_score', 0.3))
                if not isinstance(r_score, (int, float)):
                    r_score = 0.3
            scores['resistance'] = 1.0 - r_score  # Inverted: low resistance = high score

            # Dim 7: Translatability (0.08)
            s = 0.3  # Default
            if nd.get('has_approved_drug'): s = 0.8
            elif nd.get('in_clinical_trial'): s = 0.6
            elif nd.get('has_chemical_probe'): s = 0.5
            scores['translatability'] = s

            # Dim 8: CRISPR Perturbation Strength (0.08) — NEW
            s = 0.0
            if isinstance(ace, (int, float)):
                if ace <= -0.3: s += 0.5
                elif ace <= -0.1: s += 0.3
            if nd.get('therapeutic_alignment') == 'Aggravating': s += 0.2
            if nd.get('causal_tier') == 'Validated Driver': s += 0.15
            eci = nd.get('crispr_enhanced_importance', 0)
            if isinstance(eci, (int, float)) and eci > 0:
                s += min(0.15, eci * 0.1)
            scores['crispr_perturbation'] = min(1.0, s)

            # Dim 9: Knockout Ensemble Confidence (0.06) — NEW
            ko_score = 0.0
            if knockout_results and gene in knockout_results:
                ko = knockout_results[gene]
                if hasattr(ko, 'ensemble'):
                    ko_score = min(1.0, abs(ko.ensemble) * 2)
                elif hasattr(ko, 'ensemble_score'):
                    ko_score = min(1.0, abs(ko.ensemble_score) * 2)
                elif isinstance(ko, dict):
                    ko_score = min(1.0, abs(ko.get('ensemble', ko.get('ensemble_score', 0))) * 2)
            scores['knockout_confidence'] = ko_score

            raw[gene] = scores

        # Normalize
        normalized = self._normalize(raw)

        # Compute composite
        results = []
        for gene, scores in normalized.items():
            composite = sum(self.WEIGHTS[d] * scores[d] for d in self.WEIGHTS)
            if dag.has_edge(gene, disease_node):
                composite *= self.disease_bonus

            nd = dag.nodes[gene]
            results.append({
                'gene': gene,
                'composite_score': round(composite, 6),
                'passed_gate': composite >= self.confidence_gate,
                'dimensions': {k: round(v, 4) for k, v in scores.items()},
                'tier': nd.get('network_tier', 'Unknown'),
                'ace': round(nd.get('perturbation_ace', 0), 4) if isinstance(nd.get('perturbation_ace'), (int, float)) else 0,
                'essentiality': nd.get('essentiality_tag', 'Unknown'),
                'alignment': nd.get('therapeutic_alignment', 'Unknown'),
            })

            nd['drug_target_score_9d'] = round(composite, 6)

        results.sort(key=lambda x: -x['composite_score'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return {
            'dimensions': 9,
            'weights': self.WEIGHTS,
            'scored': len(results),
            'passed_gate': sum(1 for r in results if r['passed_gate']),
            'ranked_targets': results,
        }

    def _normalize(self, raw: Dict) -> Dict:
        if not raw:
            return raw
        dims = list(next(iter(raw.values())).keys())
        mins = {d: min(raw[g][d] for g in raw) for d in dims}
        maxs = {d: max(raw[g][d] for g in raw) for d in dims}
        norm = {}
        for gene, scores in raw.items():
            norm[gene] = {}
            for d in dims:
                rng = maxs[d] - mins[d]
                norm[gene][d] = (scores[d] - mins[d]) / rng if rng > 0 else 0.5
        return norm
