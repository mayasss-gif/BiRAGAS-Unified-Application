"""
Phase3KnockoutIntegrator — Feeds Phase 3 QA Results to Knockout Agent
========================================================================
CRITICAL GAP: The knockout agent does NOT read Phase 3 edge attributes:
- hallucination_flags (ignored)
- causal_test_score (ignored)
- direction_confidence (ignored)
- confounding_score (ignored)

This module bridges Phase 3 QA results to the knockout agent by:
1. Computing per-gene QA scores from edge attributes
2. Filtering out genes whose edges are mostly hallucination-flagged
3. Adjusting knockout confidence based on causal test results
4. Providing a QA-filtered gene list for focused knockout prediction
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase3.integrator")


class Phase3KnockoutIntegrator:
    """
    Bridges Phase 3 causal QA results to CRISPR knockout agents.

    The knockout agent ignores Phase 3 annotations. This module
    computes per-gene QA summaries that the knockout agent CAN use.

    Usage:
        integrator = Phase3KnockoutIntegrator()
        dag, report = integrator.integrate(dag)
        clean_genes = integrator.get_qa_filtered_genes(dag, min_qa=0.5)
    """

    def __init__(self, min_causal_score: float = 0.5, max_hallucination_rate: float = 0.5,
                 max_confounding: float = 0.6):
        self.min_causal_score = min_causal_score
        self.max_hallucination_rate = max_hallucination_rate
        self.max_confounding = max_confounding

    def integrate(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Compute per-gene QA scores from Phase 3 edge attributes.
        """
        report = {
            'genes_assessed': 0,
            'genes_clean': 0,
            'genes_flagged': 0,
            'genes_removed': 0,
            'qa_distribution': {},
            'flagged_genes': [],
        }

        for gene in dag.nodes():
            nd = dag.nodes[gene]
            if nd.get('layer') != 'regulatory':
                continue

            report['genes_assessed'] += 1

            # Collect Phase 3 scores from ALL edges connected to this gene
            outgoing = list(dag.edges(gene, data=True))
            incoming = list(dag.in_edges(gene, data=True))
            all_edges = outgoing + [(v, u, d) for u, v, d in incoming]

            if not all_edges:
                nd['qa_score'] = 0.0
                nd['qa_status'] = 'no_edges'
                continue

            # Causal test scores
            causal_scores = []
            for _, _, d in all_edges:
                cs = d.get('causal_test_score')
                if isinstance(cs, (int, float)):
                    causal_scores.append(cs)

            # Hallucination flags
            n_flagged = sum(1 for _, _, d in all_edges if d.get('hallucination_flags'))
            halluc_rate = n_flagged / len(all_edges)

            # Direction confidence
            dir_confs = []
            for _, _, d in all_edges:
                dc = d.get('direction_confidence')
                if isinstance(dc, (int, float)):
                    dir_confs.append(dc)

            # Confounding scores
            conf_scores = []
            for _, _, d in all_edges:
                cs = d.get('confounding_score')
                if isinstance(cs, (int, float)):
                    conf_scores.append(cs)

            # Compute QA score (0-1, higher = better quality)
            qa = 0.0
            components = 0

            if causal_scores:
                qa += float(np.mean(causal_scores)) * 0.35
                components += 1

            qa += (1.0 - halluc_rate) * 0.25  # Low hallucination = good
            components += 1

            if dir_confs:
                qa += float(np.mean(dir_confs)) * 0.20
                components += 1

            if conf_scores:
                qa += (1.0 - float(np.mean(conf_scores))) * 0.20  # Low confounding = good
                components += 1

            # Normalize by available components
            if components > 0:
                qa = qa / (components * 0.25)  # Normalize to 0-1 range approximately

            qa = min(1.0, max(0.0, qa))

            # Classify
            if halluc_rate > self.max_hallucination_rate:
                status = 'high_hallucination'
                report['genes_flagged'] += 1
                report['flagged_genes'].append({
                    'gene': gene, 'qa': round(qa, 4),
                    'halluc_rate': round(halluc_rate, 4),
                    'reason': f"{n_flagged}/{len(all_edges)} edges hallucination-flagged",
                })
            elif conf_scores and float(np.mean(conf_scores)) > self.max_confounding:
                status = 'high_confounding'
                report['genes_flagged'] += 1
                report['flagged_genes'].append({
                    'gene': gene, 'qa': round(qa, 4),
                    'mean_confounding': round(float(np.mean(conf_scores)), 4),
                    'reason': 'High average confounding score',
                })
            elif qa >= self.min_causal_score:
                status = 'clean'
                report['genes_clean'] += 1
            else:
                status = 'low_quality'
                report['genes_flagged'] += 1

            nd['qa_score'] = round(qa, 4)
            nd['qa_status'] = status
            nd['qa_hallucination_rate'] = round(halluc_rate, 4)
            nd['qa_n_edges'] = len(all_edges)

            # QA distribution
            bracket = f"{int(qa * 10) / 10:.1f}-{int(qa * 10 + 1) / 10:.1f}"
            report['qa_distribution'][bracket] = report['qa_distribution'].get(bracket, 0) + 1

        report['flagged_genes'] = report['flagged_genes'][:20]

        logger.info(
            f"Phase3 KO Integrator: {report['genes_assessed']} assessed | "
            f"{report['genes_clean']} clean | {report['genes_flagged']} flagged"
        )

        return dag, report

    def get_qa_filtered_genes(self, dag: nx.DiGraph, min_qa: float = 0.5) -> List[str]:
        """Get genes that pass QA threshold for knockout prediction."""
        genes = []
        for n in dag.nodes():
            nd = dag.nodes[n]
            if nd.get('layer') == 'regulatory' and nd.get('qa_score', 0) >= min_qa:
                genes.append((n, nd.get('qa_score', 0)))
        genes.sort(key=lambda x: -x[1])
        return [g for g, _ in genes]

    def get_knockout_confidence_adjustments(self, dag: nx.DiGraph) -> Dict[str, float]:
        """Get QA-based confidence adjustments for knockout agent."""
        adjustments = {}
        for n in dag.nodes():
            nd = dag.nodes[n]
            if nd.get('layer') == 'regulatory':
                qa = nd.get('qa_score', 0.5)
                # Adjust knockout confidence: QA < 0.5 → reduce, QA > 0.7 → boost
                if qa < 0.5:
                    adjustments[n] = -0.1 * (0.5 - qa) / 0.5  # Up to -0.1
                elif qa > 0.7:
                    adjustments[n] = 0.1 * (qa - 0.7) / 0.3  # Up to +0.1
        return adjustments
