"""
CRISPRHallucinationShield — Prevents False Positives Using CRISPR Evidence
============================================================================
The CausalityTester's 5 hallucination categories can flag edges that
have strong CRISPR support. This module provides a CRISPR-aware shield:

1. Edges with CRISPR evidence (ACE ≤ -0.1 on both endpoints) are
   protected from 'spurious_edge' flagging
2. CRISPR-validated edges get a composite boost before hallucination checks
3. Generates a CRISPR-vs-hallucination conflict report
4. Rescues edges that were hallucination-flagged but have CRISPR support

The shield runs AFTER CausalityTester to review and potentially override flags.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase3.shield")


class CRISPRHallucinationShield:
    """
    Reviews hallucination flags and shields CRISPR-supported edges.

    Usage:
        shield = CRISPRHallucinationShield()
        dag, report = shield.shield(dag)
    """

    def __init__(self, rescue_threshold: float = -0.1, min_ace_endpoints: int = 1):
        self.rescue_threshold = rescue_threshold
        self.min_ace_endpoints = min_ace_endpoints

    def shield(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Review hallucination flags and shield CRISPR-supported edges.
        """
        report = {
            'edges_reviewed': 0,
            'flagged_edges': 0,
            'crispr_supported_flagged': 0,
            'rescued': 0,
            'rescue_details': [],
            'hallucination_distribution': {},
            'crispr_protection_rate': 0.0,
        }

        flagged_edges = []
        crispr_edges = []

        for u, v, d in dag.edges(data=True):
            report['edges_reviewed'] += 1
            flags = d.get('hallucination_flags', [])

            if flags:
                report['flagged_edges'] += 1
                for flag in flags:
                    report['hallucination_distribution'][flag] = report['hallucination_distribution'].get(flag, 0) + 1

                # Check CRISPR support on endpoints
                u_ace = dag.nodes[u].get('perturbation_ace', 0)
                v_ace = dag.nodes[v].get('perturbation_ace', 0)
                u_has_ace = isinstance(u_ace, (int, float)) and u_ace <= self.rescue_threshold
                v_has_ace = isinstance(v_ace, (int, float)) and v_ace <= self.rescue_threshold
                n_ace = (1 if u_has_ace else 0) + (1 if v_has_ace else 0)

                if n_ace >= self.min_ace_endpoints:
                    report['crispr_supported_flagged'] += 1

                    # Determine if rescue is warranted
                    can_rescue = True
                    rescue_reason = []

                    for flag in flags:
                        if flag == 'spurious_edge' and n_ace >= 1:
                            rescue_reason.append(f"Spurious edge rescued: {n_ace} endpoint(s) have CRISPR ACE ≤ {self.rescue_threshold}")
                        elif flag == 'confounded_inference' and n_ace >= 2:
                            rescue_reason.append(f"Confounded inference overridden: both endpoints CRISPR-validated")
                        elif flag == 'directionality_error':
                            # Don't rescue directionality errors — CRISPR supports direction, not existence
                            can_rescue = False
                            rescue_reason.append(f"Directionality error NOT rescued (CRISPR validates direction, not this)")
                        elif flag in ('pathway_completeness_gap', 'magnitude_error'):
                            if n_ace >= 2:
                                rescue_reason.append(f"{flag} overridden: both endpoints strongly CRISPR-validated")
                            else:
                                can_rescue = False

                    if can_rescue and rescue_reason:
                        # Remove hallucination flags
                        d['hallucination_flags'] = []
                        d['crispr_rescued'] = True
                        d['crispr_rescue_reason'] = '; '.join(rescue_reason)

                        # Boost composite score
                        old_score = d.get('causal_test_score', 0)
                        if isinstance(old_score, (int, float)):
                            boost = min(0.2, abs(u_ace if u_has_ace else 0) * 0.3)
                            d['causal_test_score'] = min(1.0, old_score + boost)
                            d['causal_test_passed'] = d['causal_test_score'] >= 0.65

                        report['rescued'] += 1
                        report['rescue_details'].append({
                            'edge': f"{u} → {v}",
                            'flags_removed': flags,
                            'reason': '; '.join(rescue_reason),
                            'ace_u': round(u_ace, 4) if isinstance(u_ace, (int, float)) else 0,
                            'ace_v': round(v_ace, 4) if isinstance(v_ace, (int, float)) else 0,
                        })

        # Protection rate
        if report['flagged_edges'] > 0:
            report['crispr_protection_rate'] = round(report['rescued'] / report['flagged_edges'], 4)

        report['rescue_details'] = report['rescue_details'][:20]

        logger.info(
            f"CRISPR Hallucination Shield: {report['edges_reviewed']} reviewed | "
            f"{report['flagged_edges']} flagged | {report['crispr_supported_flagged']} CRISPR-supported | "
            f"{report['rescued']} rescued ({report['crispr_protection_rate']*100:.1f}%)"
        )

        return dag, report
