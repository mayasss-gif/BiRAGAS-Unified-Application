"""
CRISPRArbitrationEnhancer — Feeds Knockout Results into LLM Arbitration
==========================================================================
The LLMArbitrator counts CRISPR as 'strong evidence' but doesn't use
knockout prediction results. This module injects knockout scores as
additional evidence for conflict resolution.

When a hallucination-flagged edge connects a gene with high knockout score,
the evidence for retaining it is stronger than the flags suggest.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase7.arbitration")


class CRISPRArbitrationEnhancer:
    """
    Enhances LLM arbitration with CRISPR knockout prediction results.
    """

    def __init__(self, ko_boost_threshold: float = 0.3):
        self.ko_boost_threshold = ko_boost_threshold

    def enhance(self, dag: nx.DiGraph, knockout_results: Dict = None) -> Tuple[nx.DiGraph, Dict]:
        """
        Inject knockout evidence into edges for arbitration enhancement.
        """
        report = {
            'edges_enhanced': 0,
            'strong_evidence_boosted': 0,
            'edges_with_ko_support': 0,
            'arbitration_overrides': 0,
        }

        if not knockout_results:
            return dag, report

        for u, v, d in dag.edges(data=True):
            # Check if either endpoint has knockout results
            u_ko = knockout_results.get(u)
            v_ko = knockout_results.get(v)

            u_score = 0
            v_score = 0
            if u_ko:
                u_score = abs(getattr(u_ko, 'ensemble', 0) if hasattr(u_ko, 'ensemble') else u_ko.get('ensemble', u_ko.get('ensemble_score', 0)) if isinstance(u_ko, dict) else 0)
            if v_ko:
                v_score = abs(getattr(v_ko, 'ensemble', 0) if hasattr(v_ko, 'ensemble') else v_ko.get('ensemble', v_ko.get('ensemble_score', 0)) if isinstance(v_ko, dict) else 0)

            max_score = max(u_score, v_score)

            if max_score >= self.ko_boost_threshold:
                report['edges_with_ko_support'] += 1

                # Count knockout prediction as strong evidence
                evidence_str = str(d.get('evidence_types', d.get('evidence', '')))
                if 'crispr' not in evidence_str.lower():
                    d['evidence_types'] = evidence_str + ',crispr_knockout_predicted'
                    report['strong_evidence_boosted'] += 1

                d['knockout_support_score'] = round(max_score, 4)
                d['knockout_support_gene'] = u if u_score >= v_score else v

                # Override removal flags if strong knockout support
                if d.get('flagged_for_removal') and max_score >= 0.5:
                    d['flagged_for_removal'] = False
                    d['crispr_ko_override'] = True
                    d['arbitration_action'] = 'retain_with_note'
                    d['arbitration_rationale'] = f"Retained: strong CRISPR knockout support (score={max_score:.3f})"
                    report['arbitration_overrides'] += 1

                report['edges_enhanced'] += 1

        logger.info(
            f"CRISPRArbitrationEnhancer: {report['edges_enhanced']} enhanced | "
            f"{report['strong_evidence_boosted']} strong evidence boosted | "
            f"{report['arbitration_overrides']} removal overrides"
        )

        return dag, report
