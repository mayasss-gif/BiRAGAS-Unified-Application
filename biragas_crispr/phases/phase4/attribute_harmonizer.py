"""
AttributeHarmonizer — Fixes confidence_score vs confidence Incompatibility
============================================================================
CRITICAL BUG: Phase 4 modules read edge['confidence_score'] but the CRISPR
agentic engine reads edge['confidence']. If DAGBuilder writes 'confidence_score'
and the knockout agent reads 'confidence', it gets None → crashes or defaults.

This module ensures both attribute names are present on every edge,
enabling Phase 4 modules and CRISPR engines to operate on the same DAG.
"""

import logging
from typing import Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase4.harmonizer")


class AttributeHarmonizer:
    """
    Harmonizes edge attribute names between Phase 4 and CRISPR engines.

    Phase 4 reads: confidence_score, weight
    CRISPR reads: confidence, weight

    After harmonization, every edge has BOTH confidence_score and confidence.
    """

    def harmonize(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Ensure every edge has both 'confidence' and 'confidence_score'.
        """
        report = {
            'edges_checked': 0,
            'confidence_added': 0,
            'confidence_score_added': 0,
            'non_numeric_fixed': 0,
            'missing_weight_fixed': 0,
        }

        for u, v, d in dag.edges(data=True):
            report['edges_checked'] += 1

            # Get whatever confidence exists
            conf = d.get('confidence')
            conf_score = d.get('confidence_score')

            # Fix non-numeric
            if conf is not None and not isinstance(conf, (int, float)):
                try:
                    conf = float(conf)
                except (ValueError, TypeError):
                    conf = 0.5
                report['non_numeric_fixed'] += 1

            if conf_score is not None and not isinstance(conf_score, (int, float)):
                try:
                    conf_score = float(conf_score)
                except (ValueError, TypeError):
                    conf_score = 0.5
                report['non_numeric_fixed'] += 1

            # Harmonize: ensure both exist
            if conf is not None and conf_score is None:
                d['confidence_score'] = conf
                report['confidence_score_added'] += 1
            elif conf_score is not None and conf is None:
                d['confidence'] = conf_score
                report['confidence_added'] += 1
            elif conf is None and conf_score is None:
                d['confidence'] = 0.5
                d['confidence_score'] = 0.5
                report['confidence_added'] += 1
                report['confidence_score_added'] += 1

            # Ensure weight exists
            if 'weight' not in d or not isinstance(d.get('weight'), (int, float)):
                d['weight'] = 0.5
                report['missing_weight_fixed'] += 1

        logger.info(
            f"AttributeHarmonizer: {report['edges_checked']} edges | "
            f"{report['confidence_added']} confidence added | "
            f"{report['confidence_score_added']} confidence_score added | "
            f"{report['non_numeric_fixed']} non-numeric fixed"
        )

        return dag, report
