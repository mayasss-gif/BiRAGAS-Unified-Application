"""
CRISPRMotifValidator — Scores Conserved Motifs by CRISPR Validation
======================================================================
Gap: ConservedMotifsIdentifier scores motifs by conservation fraction only.
A motif where ALL edges are CRISPR-validated is stronger than one where
edges are purely computational — but they get the same conservation score.

Fix: Adds crispr_confidence per motif = fraction of motif edges that are
CRISPR-validated (both endpoints have ACE ≤ -0.1).
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase6.motif_validator")


class CRISPRMotifValidator:
    """
    Adds CRISPR validation scoring to conserved motifs.
    """

    def __init__(self, ace_threshold: float = -0.1):
        self.ace_threshold = ace_threshold

    def validate_motifs(self, dag: nx.DiGraph, motif_results: Dict) -> Dict:
        """
        Add CRISPR validation scores to motif results.

        Args:
            dag: Consensus DAG with CRISPR node attributes
            motif_results: Output from ConservedMotifsIdentifier.identify_motifs()
        """
        report = {
            'motifs_validated': 0,
            'fully_crispr_validated': 0,
            'partially_validated': 0,
            'no_crispr_evidence': 0,
            'validated_motifs': [],
        }

        all_motifs = motif_results.get('top_motifs', motif_results.get('conserved_motifs', []))
        if not all_motifs:
            # Try extracting from per-type results
            for mtype in ['feed_forward_loops', 'cascades', 'hub_spokes', 'diamonds', 'convergent']:
                all_motifs.extend(motif_results.get(mtype, []))

        for motif in all_motifs:
            edges = motif.get('edges', [])
            nodes = motif.get('nodes', [])
            motif_type = motif.get('type', motif.get('motif_type', 'unknown'))
            conservation = motif.get('conservation', motif.get('conservation_score', 0))

            if not edges and nodes:
                # Reconstruct edges from node list
                edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

            n_edges = len(edges)
            if n_edges == 0:
                continue

            # Count CRISPR-validated edges
            n_validated = 0
            edge_details = []

            for edge in edges:
                if isinstance(edge, (tuple, list)) and len(edge) >= 2:
                    u, v = edge[0], edge[1]
                else:
                    continue

                u_ace = dag.nodes.get(u, {}).get('perturbation_ace', 0)
                v_ace = dag.nodes.get(v, {}).get('perturbation_ace', 0)
                u_valid = isinstance(u_ace, (int, float)) and u_ace <= self.ace_threshold
                v_valid = isinstance(v_ace, (int, float)) and v_ace <= self.ace_threshold

                validated = u_valid or v_valid
                if validated:
                    n_validated += 1

                edge_details.append({
                    'edge': f"{u}→{v}",
                    'crispr_validated': validated,
                    'u_ace': round(u_ace, 4) if isinstance(u_ace, (int, float)) else 0,
                    'v_ace': round(v_ace, 4) if isinstance(v_ace, (int, float)) else 0,
                })

            crispr_fraction = n_validated / n_edges if n_edges > 0 else 0

            # Classify
            if crispr_fraction >= 1.0:
                crispr_class = 'fully_validated'
                report['fully_crispr_validated'] += 1
            elif crispr_fraction > 0:
                crispr_class = 'partially_validated'
                report['partially_validated'] += 1
            else:
                crispr_class = 'no_crispr_evidence'
                report['no_crispr_evidence'] += 1

            # Combined score: conservation × (1 + crispr_bonus)
            crispr_bonus = crispr_fraction * 0.5  # Up to 50% bonus for fully validated
            combined_score = conservation * (1 + crispr_bonus)

            validated_motif = {
                'type': motif_type,
                'nodes': nodes[:5] if nodes else [e[0] for e in edges[:3]],
                'n_edges': n_edges,
                'conservation': round(conservation, 4),
                'crispr_fraction': round(crispr_fraction, 4),
                'crispr_class': crispr_class,
                'combined_score': round(combined_score, 4),
                'edge_details': edge_details[:5],
            }
            report['validated_motifs'].append(validated_motif)
            report['motifs_validated'] += 1

        # Sort by combined score
        report['validated_motifs'].sort(key=lambda m: -m['combined_score'])

        logger.info(
            f"CRISPRMotifValidator: {report['motifs_validated']} motifs | "
            f"fully: {report['fully_crispr_validated']} | partial: {report['partially_validated']} | "
            f"none: {report['no_crispr_evidence']}"
        )

        return report
