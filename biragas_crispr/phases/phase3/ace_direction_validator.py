"""
ACEDirectionValidator — Validates Edge Directions Using ACE Asymmetry
=======================================================================
The DirectionalityTester uses ACE as one of 5 evidence sources (weight 0.15).
This module provides deeper ACE-based direction validation:

1. Cross-checks DirectionalityTester results against ACE asymmetry
2. Identifies edges where ACE disagrees with other direction evidence
3. Quantifies CRISPR direction confidence per edge
4. Generates direction conflict report
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase3.direction")


class ACEDirectionValidator:
    """
    Deep ACE-based direction validation for Phase 3 edges.

    Runs AFTER DirectionalityTester to cross-validate directions.

    Usage:
        validator = ACEDirectionValidator()
        dag, report = validator.validate(dag)
    """

    def __init__(self, ace_threshold: float = -0.1, min_delta: float = 0.1):
        self.ace_threshold = ace_threshold
        self.min_delta = min_delta

    def validate(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Cross-validate edge directions against ACE asymmetry.
        """
        report = {
            'edges_validated': 0,
            'ace_confirms': 0,
            'ace_conflicts': 0,
            'ace_neutral': 0,
            'no_ace_data': 0,
            'conflict_edges': [],
            'strong_confirmations': [],
        }

        for u, v, d in dag.edges(data=True):
            u_layer = dag.nodes[u].get('layer', '')
            v_layer = dag.nodes[v].get('layer', '')

            # Only validate regulatory-regulatory and regulatory-program edges
            if u_layer not in ('regulatory',) or v_layer not in ('regulatory', 'program'):
                continue

            report['edges_validated'] += 1

            u_ace = dag.nodes[u].get('perturbation_ace', 0)
            v_ace = dag.nodes[v].get('perturbation_ace', 0)

            u_valid = isinstance(u_ace, (int, float)) and u_ace != 0
            v_valid = isinstance(v_ace, (int, float)) and v_ace != 0

            if not u_valid and not v_valid:
                report['no_ace_data'] += 1
                d['ace_direction_status'] = 'no_data'
                continue

            # Compute ACE-based direction evidence
            u_abs = abs(u_ace) if u_valid else 0
            v_abs = abs(v_ace) if v_valid else 0
            delta = u_abs - v_abs

            if u_valid and v_valid and abs(delta) >= self.min_delta:
                if delta > 0:
                    # u has stronger ACE → u should be upstream (u→v is correct)
                    ace_direction = 'forward'
                else:
                    # v has stronger ACE → v should be upstream (u→v is WRONG)
                    ace_direction = 'reverse'
            elif u_valid and not v_valid:
                ace_direction = 'forward'  # Only u has ACE → u drives
            elif v_valid and not u_valid:
                ace_direction = 'reverse'  # Only v has ACE → v drives
            else:
                ace_direction = 'neutral'

            # Compare with DirectionalityTester result
            dt_score = d.get('direction_score', 0.5)
            dt_direction = 'forward' if isinstance(dt_score, (int, float)) and dt_score >= 0.6 else 'reverse' if isinstance(dt_score, (int, float)) and dt_score < 0.4 else 'uncertain'

            if ace_direction == 'neutral':
                report['ace_neutral'] += 1
                d['ace_direction_status'] = 'neutral'
            elif ace_direction == dt_direction or dt_direction == 'uncertain':
                report['ace_confirms'] += 1
                d['ace_direction_status'] = 'confirmed'
                d['ace_direction_confidence'] = round(abs(delta), 4)

                # Boost direction confidence when ACE agrees
                if isinstance(dt_score, (int, float)):
                    boost = min(0.1, abs(delta) * 0.2)
                    d['direction_confidence'] = min(1.0, d.get('direction_confidence', 0.5) + boost)

                if abs(delta) >= 0.2:
                    report['strong_confirmations'].append({
                        'edge': f"{u} → {v}",
                        'ace_u': round(u_ace, 4) if u_valid else 'N/A',
                        'ace_v': round(v_ace, 4) if v_valid else 'N/A',
                        'delta': round(delta, 4),
                    })
            else:
                report['ace_conflicts'] += 1
                d['ace_direction_status'] = 'CONFLICT'
                d['ace_direction_conflict'] = True
                d['ace_suggests'] = ace_direction

                report['conflict_edges'].append({
                    'edge': f"{u} → {v}",
                    'dt_says': dt_direction,
                    'ace_says': ace_direction,
                    'ace_u': round(u_ace, 4) if u_valid else 'N/A',
                    'ace_v': round(v_ace, 4) if v_valid else 'N/A',
                    'delta': round(delta, 4),
                    'recommendation': 'Review: CRISPR ACE suggests opposite direction',
                })

        report['strong_confirmations'] = report['strong_confirmations'][:15]
        report['conflict_edges'] = report['conflict_edges'][:15]

        if report['edges_validated'] > 0:
            report['confirmation_rate'] = round(report['ace_confirms'] / report['edges_validated'], 4)
            report['conflict_rate'] = round(report['ace_conflicts'] / report['edges_validated'], 4)

        logger.info(
            f"ACE Direction Validator: {report['edges_validated']} validated | "
            f"{report['ace_confirms']} confirmed | {report['ace_conflicts']} conflicts | "
            f"{report['ace_neutral']} neutral"
        )

        return dag, report
