"""
CRISPRDriverComparator — Annotates Subgroup Drivers with CRISPR Evidence
==========================================================================
Gap: DAGComparator reports unique drivers per subgroup but doesn't flag
which are CRISPR-validated. This module adds CRISPR annotation to
subgroup comparison, enabling clinically actionable subtype differentiation.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase6.comparator")


class CRISPRDriverComparator:
    """
    Enhances subgroup comparison with CRISPR evidence annotations.
    """

    def __init__(self, ace_threshold: float = -0.1):
        self.ace_threshold = ace_threshold

    def compare_with_crispr(self, subgroup_dags: Dict[str, nx.DiGraph]) -> Dict:
        """
        Compare subgroups and annotate drivers with CRISPR status.
        """
        report = {'comparisons': [], 'subgroup_profiles': {}}

        subgroup_ids = list(subgroup_dags.keys())

        # Profile each subgroup
        for sg_id, dag in subgroup_dags.items():
            drivers = []
            for n in dag.nodes():
                nd = dag.nodes[n]
                if nd.get('layer') != 'regulatory':
                    continue
                ace = nd.get('perturbation_ace', 0)
                is_crispr = isinstance(ace, (int, float)) and ace <= self.ace_threshold
                drivers.append({
                    'gene': n,
                    'crispr_validated': is_crispr,
                    'ace': round(ace, 4) if isinstance(ace, (int, float)) else 0,
                    'essentiality': nd.get('essentiality_tag', 'Unknown'),
                    'tier': nd.get('network_tier', 'Unknown'),
                    'importance': round(nd.get('causal_importance', 0), 4) if isinstance(nd.get('causal_importance'), (int, float)) else 0,
                })
            drivers.sort(key=lambda d: -d['importance'])
            report['subgroup_profiles'][sg_id] = {
                'total_drivers': len(drivers),
                'crispr_validated': sum(1 for d in drivers if d['crispr_validated']),
                'top_drivers': drivers[:10],
            }

        # Pairwise comparison
        for i in range(len(subgroup_ids)):
            for j in range(i + 1, len(subgroup_ids)):
                sg_a, sg_b = subgroup_ids[i], subgroup_ids[j]
                genes_a = {n for n in subgroup_dags[sg_a].nodes() if subgroup_dags[sg_a].nodes[n].get('layer') == 'regulatory'}
                genes_b = {n for n in subgroup_dags[sg_b].nodes() if subgroup_dags[sg_b].nodes[n].get('layer') == 'regulatory'}

                shared = genes_a & genes_b
                unique_a = genes_a - genes_b
                unique_b = genes_b - genes_a

                # Annotate unique drivers with CRISPR status
                def annotate(genes, dag):
                    return [{
                        'gene': g,
                        'crispr_validated': isinstance(dag.nodes[g].get('perturbation_ace', 0), (int, float)) and dag.nodes[g].get('perturbation_ace', 0) <= self.ace_threshold,
                        'ace': round(dag.nodes[g].get('perturbation_ace', 0), 4) if isinstance(dag.nodes[g].get('perturbation_ace'), (int, float)) else 0,
                        'essentiality': dag.nodes[g].get('essentiality_tag', 'Unknown'),
                    } for g in sorted(genes)[:10]]

                comparison = {
                    'subgroup_a': sg_a,
                    'subgroup_b': sg_b,
                    'shared_genes': len(shared),
                    'unique_to_a': len(unique_a),
                    'unique_to_b': len(unique_b),
                    'overlap_jaccard': len(shared) / max(len(genes_a | genes_b), 1),
                    'unique_a_crispr': annotate(unique_a, subgroup_dags[sg_a]),
                    'unique_b_crispr': annotate(unique_b, subgroup_dags[sg_b]),
                    'clinical_implication': self._clinical_implication(unique_a, unique_b, subgroup_dags[sg_a], subgroup_dags[sg_b]),
                }
                report['comparisons'].append(comparison)

        logger.info(f"CRISPRDriverComparator: {len(report['comparisons'])} pairwise comparisons")
        return report

    def _clinical_implication(self, unique_a, unique_b, dag_a, dag_b):
        """Generate clinical implication text."""
        crispr_a = sum(1 for g in unique_a if isinstance(dag_a.nodes[g].get('perturbation_ace', 0), (int, float)) and dag_a.nodes[g].get('perturbation_ace', 0) <= self.ace_threshold)
        crispr_b = sum(1 for g in unique_b if isinstance(dag_b.nodes[g].get('perturbation_ace', 0), (int, float)) and dag_b.nodes[g].get('perturbation_ace', 0) <= self.ace_threshold)

        if crispr_a > 0 and crispr_b > 0:
            return f"Both subtypes have CRISPR-validated unique drivers ({crispr_a} vs {crispr_b}) — different drug targets per subtype"
        elif crispr_a > 0:
            return f"Subtype A has {crispr_a} CRISPR-validated unique drivers — actionable targets"
        elif crispr_b > 0:
            return f"Subtype B has {crispr_b} CRISPR-validated unique drivers — actionable targets"
        else:
            return "Neither subtype has CRISPR-validated unique drivers — need experimental validation"
