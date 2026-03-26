"""
CRISPRCentralityEnhancer — Enhances Centrality with CRISPR Evidence
======================================================================
The standard CentralityCalculator computes causal_importance as:
    CI = (apex × causal_strength) + betweenness + (ppr × 10)

This module adds a CRISPR-enhanced importance score:
    CI_enhanced = CI + ace_bonus + essentiality_modifier + alignment_bonus

Where:
    ace_bonus = min(0.5, |ACE| × 1.5) if ACE ≤ -0.1
    essentiality_modifier = -0.3 if Core Essential (penalty for safety)
    alignment_bonus = +0.2 if Aggravating with known strategy

This gives CRISPR-validated causal drivers a significant boost in
importance ranking without overriding the network topology evidence.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase2.centrality")


class CRISPRCentralityEnhancer:
    """
    Enhances Phase 2 centrality metrics with CRISPR perturbation evidence.

    Runs AFTER CentralityCalculator.run_pipeline() to add CRISPR-informed
    importance scores alongside the standard causal_importance.

    Usage:
        from modules.centrality_calculator import CentralityCalculator, CentralityConfig
        dag, metrics = CentralityCalculator(CentralityConfig()).run_pipeline(dag)

        enhancer = CRISPRCentralityEnhancer()
        dag, crispr_report = enhancer.enhance(dag)
    """

    def __init__(self, ace_weight: float = 1.5, essentiality_penalty: float = 0.3,
                 alignment_bonus: float = 0.2):
        self.ace_weight = ace_weight
        self.essentiality_penalty = essentiality_penalty
        self.alignment_bonus = alignment_bonus

    def enhance(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Add CRISPR-enhanced importance to the DAG.

        Computes crispr_enhanced_importance for each regulatory gene:
        CI_enhanced = CI + ace_bonus + essentiality_mod + alignment_bonus

        Returns: (enriched_dag, enhancement_report)
        """
        report = {
            'genes_enhanced': 0,
            'ace_boosted': 0,
            'essentiality_penalized': 0,
            'alignment_boosted': 0,
            'original_vs_enhanced': [],
        }

        for gene in dag.nodes():
            nd = dag.nodes[gene]
            if nd.get('layer') != 'regulatory':
                continue

            ci = nd.get('causal_importance', 0)
            ace = nd.get('perturbation_ace', 0)
            ess = nd.get('essentiality_tag', 'Unknown')
            align = nd.get('therapeutic_alignment', 'Unknown')

            # ACE bonus: strong negative ACE = strong causal driver
            ace_bonus = 0.0
            if isinstance(ace, (int, float)) and ace <= -0.1:
                ace_bonus = min(0.5, abs(ace) * self.ace_weight)
                report['ace_boosted'] += 1

            # Essentiality modifier: Core Essential = safety concern
            ess_mod = 0.0
            if ess == 'Core Essential':
                ess_mod = -self.essentiality_penalty
                report['essentiality_penalized'] += 1

            # Alignment bonus: known therapeutic direction
            align_bonus = 0.0
            if align in ('Aggravating', 'Reversal'):
                align_bonus = self.alignment_bonus
                report['alignment_boosted'] += 1

            # Enhanced importance
            ci_enhanced = ci + ace_bonus + ess_mod + align_bonus
            nd['crispr_enhanced_importance'] = round(ci_enhanced, 6)
            nd['ace_bonus'] = round(ace_bonus, 4)
            nd['essentiality_modifier'] = round(ess_mod, 4)
            nd['alignment_bonus'] = round(align_bonus, 4)

            report['genes_enhanced'] += 1
            report['original_vs_enhanced'].append({
                'gene': gene,
                'original_ci': round(ci, 4),
                'enhanced_ci': round(ci_enhanced, 4),
                'delta': round(ci_enhanced - ci, 4),
                'ace': round(ace, 4) if isinstance(ace, (int, float)) else 0,
            })

        # Sort by delta (biggest boost first)
        report['original_vs_enhanced'].sort(key=lambda x: -x['delta'])
        report['original_vs_enhanced'] = report['original_vs_enhanced'][:20]

        # Summary statistics
        all_ci = [dag.nodes[n].get('causal_importance', 0) for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory']
        all_eci = [dag.nodes[n].get('crispr_enhanced_importance', 0) for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory']

        if all_ci:
            report['mean_original_ci'] = round(float(np.mean(all_ci)), 4)
            report['mean_enhanced_ci'] = round(float(np.mean(all_eci)), 4)
            report['max_original_ci'] = round(float(max(all_ci)), 4)
            report['max_enhanced_ci'] = round(float(max(all_eci)), 4)
            report['correlation'] = round(float(np.corrcoef(all_ci, all_eci)[0, 1]), 4) if len(all_ci) > 1 else 1.0

        logger.info(
            f"CRISPR centrality enhancement: {report['genes_enhanced']} genes | "
            f"{report['ace_boosted']} ACE-boosted | {report['essentiality_penalized']} penalized | "
            f"{report['alignment_boosted']} alignment-boosted"
        )

        return dag, report
