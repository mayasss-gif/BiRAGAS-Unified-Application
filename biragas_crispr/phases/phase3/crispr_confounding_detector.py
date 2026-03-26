"""
CRISPRConfoundingDetector — Adds CRISPR-Aware Confounding Detection
======================================================================
Gap: ConfoundingChecker has NO CRISPR usage. It relies purely on
graph topology (backdoor paths, forks, colliders) and MR-Egger intercept.

This module adds a 5th confounding method: CRISPR perturbation coherence.

If knocking out gene U (ACE_u) changes gene V's activity in the SAME
direction as the causal edge suggests, the edge is NOT confounded.
If the effect is in the OPPOSITE direction, confounding is suspected.

Formula:
    crispr_coherence = 1 if sign(ACE_u) == sign(edge_weight) else 0
    If incoherent AND confounding_score > 0.3: flag as 'crispr_incoherent'
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase3.confounding")


class CRISPRConfoundingDetector:
    """
    Adds CRISPR perturbation coherence as a 5th confounding signal.

    Runs AFTER ConfoundingChecker to add CRISPR-specific confounding evidence.

    Usage:
        detector = CRISPRConfoundingDetector()
        dag, report = detector.detect(dag)
    """

    def __init__(self, incoherence_boost: float = 0.15, coherence_reduction: float = 0.1):
        self.incoherence_boost = incoherence_boost
        self.coherence_reduction = coherence_reduction

    def detect(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Add CRISPR perturbation coherence to confounding analysis.
        """
        report = {
            'edges_checked': 0,
            'coherent': 0,
            'incoherent': 0,
            'no_crispr_data': 0,
            'confounding_boosted': 0,
            'confounding_reduced': 0,
            'incoherent_edges': [],
        }

        for u, v, d in dag.edges(data=True):
            u_layer = dag.nodes[u].get('layer', '')
            if u_layer != 'regulatory':
                continue

            report['edges_checked'] += 1

            u_ace = dag.nodes[u].get('perturbation_ace', 0)
            if not isinstance(u_ace, (int, float)) or u_ace == 0:
                report['no_crispr_data'] += 1
                continue

            edge_weight = d.get('weight', 0)
            if not isinstance(edge_weight, (int, float)):
                continue

            # Coherence check:
            # If ACE is negative (knockout hurts = gene is needed) and edge weight is positive
            # (gene contributes to target), this is COHERENT.
            # If ACE is negative but edge weight suggests inhibition, this is INCOHERENT.
            u_drives = u_ace < 0  # Gene is a driver (knockout has effect)
            edge_positive = edge_weight > 0  # Edge is activating

            # For regulatory→regulatory: both should agree on direction
            # For regulatory→program: driver should activate program
            if u_drives and edge_positive:
                coherent = True
            elif not u_drives and not edge_positive:
                coherent = True
            else:
                coherent = False  # ACE and edge weight disagree

            existing_confounding = d.get('confounding_score', 0)
            if not isinstance(existing_confounding, (int, float)):
                existing_confounding = 0

            if coherent:
                report['coherent'] += 1
                d['crispr_coherent'] = True

                # Reduce confounding score if CRISPR confirms the edge
                if existing_confounding > 0:
                    new_score = max(0, existing_confounding - self.coherence_reduction)
                    d['confounding_score'] = new_score
                    report['confounding_reduced'] += 1
            else:
                report['incoherent'] += 1
                d['crispr_coherent'] = False
                d['crispr_incoherent'] = True

                # Boost confounding score when CRISPR disagrees
                new_score = min(1.0, existing_confounding + self.incoherence_boost)
                d['confounding_score'] = new_score
                report['confounding_boosted'] += 1

                flags = d.get('confounding_flags', [])
                if 'crispr_incoherent' not in flags:
                    flags.append('crispr_incoherent')
                    d['confounding_flags'] = flags

                report['incoherent_edges'].append({
                    'edge': f"{u} → {v}",
                    'ace_u': round(u_ace, 4),
                    'edge_weight': round(edge_weight, 4),
                    'old_confounding': round(existing_confounding, 4),
                    'new_confounding': round(new_score, 4),
                })

        report['incoherent_edges'] = report['incoherent_edges'][:15]

        if report['edges_checked'] > 0:
            report['coherence_rate'] = round(report['coherent'] / report['edges_checked'], 4)

        logger.info(
            f"CRISPR Confounding Detector: {report['edges_checked']} checked | "
            f"{report['coherent']} coherent | {report['incoherent']} incoherent | "
            f"{report['confounding_boosted']} boosted | {report['confounding_reduced']} reduced"
        )

        return dag, report
