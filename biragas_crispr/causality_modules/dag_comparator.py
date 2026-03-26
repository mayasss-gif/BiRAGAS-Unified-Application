"""
Phase 6: COMPARATIVE EVOLUTION — Module 2
DAGComparator (INTENT I_04 Module 2)
======================================
Compares causal DAG architectures between patient subgroups.

Comparison Dimensions:
  1. Structural: edge overlap, topology metrics
  2. Causal: shared/unique driver genes, tier distribution
  3. Pathway: differential pathway activation
  4. Effect Size: weight magnitude differences
  5. Confidence: evidence quality comparison

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComparatorConfig:
    significance_threshold: float = 0.20
    min_edge_frequency: float = 0.10


class DAGComparator:
    """Compares causal DAG architectures between patient subgroups."""

    def __init__(self, config: Optional[ComparatorConfig] = None):
        self.config = config or ComparatorConfig()

    def compare_subgroups(self, subgroup_dags: Dict[str, List[nx.DiGraph]],
                          consensus_dag: Optional[nx.DiGraph] = None) -> Dict:
        """Compare DAG architectures across multiple subgroups."""
        subgroup_ids = list(subgroup_dags.keys())
        aggregate_dags = {}
        for sg_id, dags in subgroup_dags.items():
            aggregate_dags[sg_id] = self._aggregate_subgroup(dags)

        pairwise = []
        for i in range(len(subgroup_ids)):
            for j in range(i + 1, len(subgroup_ids)):
                sg_a = subgroup_ids[i]
                sg_b = subgroup_ids[j]
                comparison = self._compare_pair(
                    sg_a, aggregate_dags[sg_a],
                    sg_b, aggregate_dags[sg_b])
                pairwise.append(comparison)

        return {
            'subgroups': subgroup_ids,
            'pairwise_comparisons': pairwise,
            'subgroup_profiles': {sg: self._profile_subgroup(agg)
                                   for sg, agg in aggregate_dags.items()},
        }

    def compare_two(self, dag_a: nx.DiGraph, dag_b: nx.DiGraph,
                     label_a: str = 'A', label_b: str = 'B') -> Dict:
        """Direct comparison of two DAGs."""
        return self._compare_pair(label_a, dag_a, label_b, dag_b)

    def _aggregate_subgroup(self, dags: List[nx.DiGraph]) -> nx.DiGraph:
        """Aggregate multiple patient DAGs into a subgroup consensus."""
        agg = nx.DiGraph()
        edge_counts = {}
        node_counts = {}
        n = len(dags)

        for dag in dags:
            for node, data in dag.nodes(data=True):
                node_counts[node] = node_counts.get(node, 0) + 1
                if node not in agg:
                    agg.add_node(node, **data)

            for u, v, data in dag.edges(data=True):
                key = (u, v)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                if not agg.has_edge(u, v):
                    agg.add_edge(u, v, **data)

        for node in agg.nodes():
            agg.nodes[node]['subgroup_frequency'] = node_counts.get(node, 0) / n

        for u, v in agg.edges():
            agg[u][v]['subgroup_frequency'] = edge_counts.get((u, v), 0) / n

        to_remove = [(u, v) for u, v in agg.edges()
                     if agg[u][v]['subgroup_frequency'] < self.config.min_edge_frequency]
        agg.remove_edges_from(to_remove)

        return agg

    def _compare_pair(self, label_a: str, dag_a: nx.DiGraph,
                       label_b: str, dag_b: nx.DiGraph) -> Dict:
        """Compare two DAGs across all dimensions."""
        structural = self._compare_structural(dag_a, dag_b)
        causal = self._compare_causal_drivers(dag_a, dag_b)
        pathway = self._compare_pathways(dag_a, dag_b)

        similarity = (structural['edge_jaccard'] * 0.4 +
                      causal['driver_overlap'] * 0.3 +
                      pathway['pathway_jaccard'] * 0.3)

        return {
            'subgroup_a': label_a,
            'subgroup_b': label_b,
            'overall_similarity': round(similarity, 4),
            'structural': structural,
            'causal_drivers': causal,
            'pathways': pathway,
        }

    def _compare_structural(self, dag_a: nx.DiGraph,
                             dag_b: nx.DiGraph) -> Dict:
        """Compare structural topology."""
        edges_a = set(dag_a.edges())
        edges_b = set(dag_b.edges())
        shared = edges_a & edges_b
        union = edges_a | edges_b

        jaccard = len(shared) / len(union) if union else 0.0

        return {
            'n_edges_a': len(edges_a),
            'n_edges_b': len(edges_b),
            'shared_edges': len(shared),
            'unique_to_a': len(edges_a - edges_b),
            'unique_to_b': len(edges_b - edges_a),
            'edge_jaccard': round(jaccard, 4),
        }

    def _compare_causal_drivers(self, dag_a: nx.DiGraph,
                                 dag_b: nx.DiGraph) -> Dict:
        """Compare causal driver genes."""
        genes_a = {n for n, d in dag_a.nodes(data=True) if d.get('layer') == 'regulatory'}
        genes_b = {n for n, d in dag_b.nodes(data=True) if d.get('layer') == 'regulatory'}

        shared = genes_a & genes_b
        overlap = len(shared) / len(genes_a | genes_b) if (genes_a | genes_b) else 0.0

        tier_dist_a = self._tier_distribution(dag_a)
        tier_dist_b = self._tier_distribution(dag_b)

        unique_a_drivers = []
        for g in genes_a - genes_b:
            d = dag_a.nodes[g]
            unique_a_drivers.append({
                'gene': g,
                'tier': d.get('network_tier', 'Unknown'),
                'importance': d.get('causal_importance', 0),
            })
        unique_a_drivers.sort(key=lambda x: x['importance'], reverse=True)

        unique_b_drivers = []
        for g in genes_b - genes_a:
            d = dag_b.nodes[g]
            unique_b_drivers.append({
                'gene': g,
                'tier': d.get('network_tier', 'Unknown'),
                'importance': d.get('causal_importance', 0),
            })
        unique_b_drivers.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'shared_genes': len(shared),
            'driver_overlap': round(overlap, 4),
            'tier_distribution_a': tier_dist_a,
            'tier_distribution_b': tier_dist_b,
            'unique_to_a': unique_a_drivers[:5],
            'unique_to_b': unique_b_drivers[:5],
        }

    def _compare_pathways(self, dag_a: nx.DiGraph,
                           dag_b: nx.DiGraph) -> Dict:
        """Compare pathway activation profiles."""
        prog_a = {n for n, d in dag_a.nodes(data=True) if d.get('layer') == 'program'}
        prog_b = {n for n, d in dag_b.nodes(data=True) if d.get('layer') == 'program'}

        shared = prog_a & prog_b
        jaccard = len(shared) / len(prog_a | prog_b) if (prog_a | prog_b) else 0.0

        classes_a = {dag_a.nodes[p].get('main_class', 'Unknown') for p in prog_a}
        classes_b = {dag_b.nodes[p].get('main_class', 'Unknown') for p in prog_b}

        return {
            'shared_pathways': len(shared),
            'pathway_jaccard': round(jaccard, 4),
            'unique_to_a': sorted(prog_a - prog_b),
            'unique_to_b': sorted(prog_b - prog_a),
            'classes_a': sorted(classes_a),
            'classes_b': sorted(classes_b),
        }

    def _tier_distribution(self, dag: nx.DiGraph) -> Dict[str, int]:
        """Count genes per network tier."""
        dist = {}
        for _, d in dag.nodes(data=True):
            if d.get('layer') == 'regulatory':
                tier = d.get('network_tier', 'Unknown')
                dist[tier] = dist.get(tier, 0) + 1
        return dist

    def _profile_subgroup(self, dag: nx.DiGraph) -> Dict:
        """Create a profile summary of a subgroup DAG."""
        return {
            'n_nodes': dag.number_of_nodes(),
            'n_edges': dag.number_of_edges(),
            'n_genes': sum(1 for _, d in dag.nodes(data=True) if d.get('layer') == 'regulatory'),
            'n_programs': sum(1 for _, d in dag.nodes(data=True) if d.get('layer') == 'program'),
            'tier_distribution': self._tier_distribution(dag),
        }
