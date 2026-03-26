"""
Phase 4: IN-SILICO PERTURBATION — Module 2
DAGPropagationEngine (INTENT I_05 Module 2)
=============================================
Propagates intervention effects through the causal DAG using
topological traversal with decay and confidence weighting.

Propagation Model:
  effect(child) = SUM over parents [ effect(parent) * weight * confidence * decay^depth ]

Supports:
  - Forward propagation (intervention -> downstream effects)
  - Backward attribution (disease -> upstream causes)
  - Pathway-specific propagation (restricted to pathway subgraph)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PropagationConfig:
    decay_factor: float = 0.85
    max_depth: int = 15
    min_effect_threshold: float = 0.001
    confidence_floor: float = 0.1
    aggregate_method: str = 'sum'  # 'sum', 'max', 'mean'


class DAGPropagationEngine:
    """Propagates causal effects through the DAG topology."""

    def __init__(self, config: Optional[PropagationConfig] = None):
        self.config = config or PropagationConfig()

    def propagate_forward(self, dag: nx.DiGraph, source: str,
                          initial_effect: float = 1.0) -> Dict:
        """Propagate effect from source node downstream through the DAG."""
        if source not in dag:
            return {'error': f'{source} not in DAG', 'effects': {}}

        effects = {source: initial_effect}
        depth_map = {source: 0}

        try:
            topo = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo = list(nx.bfs_tree(dag, source).nodes())

        start_idx = topo.index(source) if source in topo else 0
        for node in topo[start_idx + 1:]:
            incoming = []
            for pred in dag.predecessors(node):
                if pred in effects:
                    edge = dag[pred][node]
                    w = edge.get('weight', 0.5)
                    c = max(edge.get('confidence_score', 0.5), self.config.confidence_floor)
                    depth = depth_map.get(pred, 0) + 1
                    if depth <= self.config.max_depth:
                        eff = effects[pred] * w * c * (self.config.decay_factor ** depth)
                        incoming.append(eff)
                        depth_map[node] = depth

            if incoming:
                if self.config.aggregate_method == 'max':
                    effects[node] = max(incoming, key=abs)
                elif self.config.aggregate_method == 'mean':
                    effects[node] = float(np.mean(incoming))
                else:
                    effects[node] = sum(incoming)

        effects = {k: round(v, 6) for k, v in effects.items()
                   if abs(v) >= self.config.min_effect_threshold}

        return {
            'source': source,
            'initial_effect': initial_effect,
            'effects': effects,
            'n_affected': len(effects),
            'max_depth': max(depth_map.values()) if depth_map else 0,
            'layer_summary': self._summarize_by_layer(dag, effects),
        }

    def propagate_backward(self, dag: nx.DiGraph, target: str,
                           target_effect: float = 1.0) -> Dict:
        """Backward attribution: trace which upstream nodes contribute to target."""
        if target not in dag:
            return {'error': f'{target} not in DAG', 'attributions': {}}

        rev = dag.reverse(copy=True)
        result = self.propagate_forward(rev, target, target_effect)
        attributions = result.get('effects', {})

        return {
            'target': target,
            'attributions': attributions,
            'top_contributors': sorted(
                [(k, v) for k, v in attributions.items() if k != target],
                key=lambda x: abs(x[1]), reverse=True
            )[:10],
        }

    def propagate_pathway_specific(self, dag: nx.DiGraph, source: str,
                                    pathway: str,
                                    initial_effect: float = 1.0) -> Dict:
        """Propagate through a specific pathway subgraph only."""
        pathway_nodes = {n for n, d in dag.nodes(data=True)
                         if d.get('main_class') == pathway or
                         d.get('pathway') == pathway or n == source}

        for n in list(pathway_nodes):
            pathway_nodes.update(dag.predecessors(n))
            pathway_nodes.update(dag.successors(n))

        subgraph = dag.subgraph(pathway_nodes).copy()
        result = self.propagate_forward(subgraph, source, initial_effect)
        result['pathway'] = pathway
        result['subgraph_size'] = len(subgraph)
        return result

    def compute_total_causal_effect(self, dag: nx.DiGraph, source: str,
                                     target: str) -> Dict:
        """Compute total causal effect of source on target via all paths."""
        if source not in dag or target not in dag:
            return {'total_effect': 0.0, 'n_paths': 0}

        total_effect = 0.0
        paths = []
        try:
            for path in nx.all_simple_paths(dag, source, target, cutoff=self.config.max_depth):
                path_effect = 1.0
                for i in range(len(path) - 1):
                    edge = dag[path[i]][path[i + 1]]
                    w = edge.get('weight', 0.5)
                    c = edge.get('confidence_score', 0.5)
                    path_effect *= w * c * self.config.decay_factor
                total_effect += path_effect
                paths.append({'path': path, 'effect': round(path_effect, 6)})
                if len(paths) >= 50:
                    break
        except nx.NetworkXError:
            pass

        paths.sort(key=lambda x: abs(x['effect']), reverse=True)
        return {
            'source': source,
            'target': target,
            'total_effect': round(total_effect, 6),
            'n_paths': len(paths),
            'top_paths': paths[:10],
        }

    def _summarize_by_layer(self, dag: nx.DiGraph, effects: Dict) -> Dict:
        """Summarize effects grouped by topological layer."""
        summary = {}
        for node, eff in effects.items():
            layer = dag.nodes.get(node, {}).get('layer', 'unknown')
            if layer not in summary:
                summary[layer] = {'count': 0, 'total_effect': 0, 'max_effect': 0}
            summary[layer]['count'] += 1
            summary[layer]['total_effect'] += abs(eff)
            summary[layer]['max_effect'] = max(summary[layer]['max_effect'], abs(eff))
        return summary
