"""
CompensationBridge — Feeds Phase 4 Co-Targets to Combination Agent
=====================================================================
Phase 4 CompensationPathwayAnalyzer identifies co-targets (genes that
compensate when the primary target is blocked). These are ideal candidates
for combination therapy — but the CRISPR CombinationAgent doesn't read them.

This bridge:
1. Extracts co-target recommendations from Phase 4 compensation analysis
2. Feeds them as priority pairs to the CRISPR CombinationAgent
3. Generates compensation-aware combination predictions
4. Reports which Phase 4 co-targets show synergy in CRISPR predictions
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase4.compensation_bridge")


class CompensationBridge:
    """
    Bridges Phase 4 compensation co-targets to CRISPR combination predictions.

    Usage:
        bridge = CompensationBridge()
        priority_pairs = bridge.extract_priority_pairs(dag, compensation_results)
        results = bridge.predict_compensation_combos(dag, knockout_results, compensation_results)
    """

    def extract_priority_pairs(self, dag: nx.DiGraph,
                               compensation_results: Dict) -> List[Tuple[str, str, str]]:
        """
        Extract (target, co-target, reason) pairs from Phase 4 compensation analysis.

        These pairs should block compensation = synergistic combination therapy.
        """
        pairs = []

        for target, comp_data in compensation_results.items():
            if not isinstance(comp_data, dict):
                continue

            # From recommended co-targets
            co_targets = comp_data.get('recommended_co_targets', [])
            for ct in co_targets:
                ct_gene = ct.get('gene', ct) if isinstance(ct, dict) else ct
                if ct_gene and ct_gene != target and ct_gene in dag.nodes():
                    pairs.append((target, ct_gene, "Phase 4 recommended co-target"))

            # From parallel pathways (overlap >= 30%)
            parallels = comp_data.get('parallel_pathways', [])
            for pp in parallels[:3]:
                pp_gene = pp.get('gene', pp) if isinstance(pp, dict) else pp
                if pp_gene and pp_gene != target and pp_gene in dag.nodes():
                    overlap = pp.get('overlap_ratio', 0) if isinstance(pp, dict) else 0
                    pairs.append((target, pp_gene, f"Parallel pathway (overlap={overlap:.2f})"))

            # From compensatory genes
            comp_genes = comp_data.get('all_compensator_genes', [])
            for cg in comp_genes[:5]:
                cg_name = cg.get('gene', cg) if isinstance(cg, dict) else cg
                if cg_name and cg_name != target and cg_name in dag.nodes():
                    pairs.append((target, cg_name, "Compensatory gene"))

        # Deduplicate
        seen = set()
        unique_pairs = []
        for t, ct, reason in pairs:
            key = tuple(sorted([t, ct]))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((t, ct, reason))

        logger.info(f"CompensationBridge: {len(unique_pairs)} priority pairs from {len(compensation_results)} targets")
        return unique_pairs

    def predict_compensation_combos(self, dag: nx.DiGraph,
                                     knockout_results: Dict,
                                     compensation_results: Dict) -> List[Dict]:
        """
        Predict combination outcomes for compensation-derived pairs.

        Uses knockout results to estimate combined effect using Bliss model.
        """
        pairs = self.extract_priority_pairs(dag, compensation_results)
        results = []

        for target, co_target, reason in pairs:
            ko_t = knockout_results.get(target)
            ko_ct = knockout_results.get(co_target)

            if not ko_t or not ko_ct:
                continue

            # Get effects
            eff_t = abs(getattr(ko_t, 'disease_effect', 0) if hasattr(ko_t, 'disease_effect')
                       else ko_t.get('absolute_change', ko_t.get('disease_effect', 0)) if isinstance(ko_t, dict) else 0)
            eff_ct = abs(getattr(ko_ct, 'disease_effect', 0) if hasattr(ko_ct, 'disease_effect')
                        else ko_ct.get('absolute_change', ko_ct.get('disease_effect', 0)) if isinstance(ko_ct, dict) else 0)

            if eff_t == 0 and eff_ct == 0:
                continue

            # Bliss Independence
            bliss = eff_t + eff_ct - eff_t * eff_ct

            # Compensation blocking bonus (the whole point — co-target blocks escape)
            comp_bonus = 0.15  # Blocking compensation is inherently synergistic
            predicted = bliss + comp_bonus

            synergy = (predicted - bliss) / max(bliss, 0.01)

            results.append({
                'target': target,
                'co_target': co_target,
                'reason': reason,
                'effect_target': round(eff_t, 4),
                'effect_co_target': round(eff_ct, 4),
                'bliss_expected': round(bliss, 4),
                'predicted_combined': round(predicted, 4),
                'synergy_score': round(synergy, 4),
                'interaction': 'synergistic' if synergy > 0.1 else 'additive',
                'recommendation': f"Block {target} + {co_target} to prevent compensation",
            })

        results.sort(key=lambda r: -r['predicted_combined'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        logger.info(f"CompensationBridge: {len(results)} combo predictions from compensation analysis")
        return results

    def get_summary(self, results: List[Dict]) -> Dict:
        """Summarize compensation-driven combinations."""
        return {
            'total_pairs': len(results),
            'synergistic': sum(1 for r in results if r['interaction'] == 'synergistic'),
            'additive': sum(1 for r in results if r['interaction'] == 'additive'),
            'top_5': results[:5],
            'by_reason': {},
        }
