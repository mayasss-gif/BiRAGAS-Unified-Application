"""
Phase 3: CAUSAL CALCULUS — Module 3
ConfoundingChecker (INTENT I_02 Module 3)
==========================================
Detects and quantifies confounding in causal edges.

Methods:
  1. Backdoor path detection (d-separation)
  2. Common parent identification (fork structures)
  3. MR-Egger intercept test (pleiotropy-driven confounding)
  4. Collider bias detection
  5. Adjustment set computation (Pearl's backdoor criterion)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfoundingConfig:
    egger_intercept_threshold: float = 0.05
    min_backdoor_confidence: float = 0.40
    collider_amplification_factor: float = 1.5
    max_adjustment_set_size: int = 10


class ConfoundingChecker:
    """Detects and quantifies confounding bias in the causal DAG."""

    def __init__(self, config: Optional[ConfoundingConfig] = None):
        self.config = config or ConfoundingConfig()

    def check_all_edges(self, dag: nx.DiGraph) -> Dict:
        """Check all edges for confounding. Returns report and annotates DAG."""
        results = []
        confounded = []

        for u, v, data in dag.edges(data=True):
            result = self._check_edge(u, v, data, dag)
            results.append(result)

            dag[u][v]['confounding_score'] = result['confounding_score']
            dag[u][v]['adjustment_set'] = result['adjustment_set']
            dag[u][v]['confounding_flags'] = result['flags']

            if result['is_confounded']:
                confounded.append(result)

        return {
            'results': results,
            'confounded_edges': confounded,
            'summary': {
                'total_checked': len(results),
                'confounded': len(confounded),
                'confounding_rate': round(len(confounded) / max(len(results), 1), 3),
                'most_confounded': sorted(confounded,
                    key=lambda x: x['confounding_score'], reverse=True)[:5],
            },
        }

    def _check_edge(self, u: str, v: str, data: Dict,
                     dag: nx.DiGraph) -> Dict:
        """Check a single edge for confounding."""
        flags = []

        backdoor_score, backdoor_paths = self._check_backdoor_paths(u, v, dag)
        if backdoor_score > 0:
            flags.append('backdoor_path')

        fork_score, common_parents = self._check_common_parents(u, v, dag)
        if fork_score > 0:
            flags.append('common_parent')

        egger_score = self._check_mr_egger(data)
        if egger_score > 0:
            flags.append('pleiotropy_confounding')

        collider_score, colliders = self._check_collider_bias(u, v, dag)
        if collider_score > 0:
            flags.append('collider_bias')

        confounding_score = max(backdoor_score, fork_score, egger_score, collider_score)
        adjustment_set = self._compute_adjustment_set(u, v, dag)

        if adjustment_set:
            adjustment_reduction = min(0.5, len(adjustment_set) * 0.1)
            confounding_score = max(0, confounding_score - adjustment_reduction)

        return {
            'edge': (u, v),
            'confounding_score': round(confounding_score, 4),
            'is_confounded': confounding_score >= self.config.min_backdoor_confidence,
            'flags': flags,
            'backdoor_paths': backdoor_paths,
            'common_parents': common_parents,
            'colliders': colliders,
            'adjustment_set': adjustment_set,
            'dimensions': {
                'backdoor': round(backdoor_score, 4),
                'common_parent': round(fork_score, 4),
                'egger_pleiotropy': round(egger_score, 4),
                'collider_bias': round(collider_score, 4),
            },
        }

    def _check_backdoor_paths(self, u: str, v: str,
                               dag: nx.DiGraph) -> Tuple[float, List]:
        """Detect backdoor paths (non-causal paths from u to v)."""
        undirected = dag.to_undirected()
        dag_copy = dag.copy()
        if dag_copy.has_edge(u, v):
            dag_copy.remove_edge(u, v)

        undirected_copy = dag_copy.to_undirected()
        backdoor_paths = []

        try:
            for path in nx.all_simple_paths(undirected_copy, u, v, cutoff=5):
                if len(path) > 2:
                    backdoor_paths.append(path)
                    if len(backdoor_paths) >= 10:
                        break
        except nx.NetworkXError:
            pass

        if not backdoor_paths:
            return 0.0, []

        score = min(1.0, len(backdoor_paths) * 0.15)
        return score, [list(p) for p in backdoor_paths[:5]]

    def _check_common_parents(self, u: str, v: str,
                               dag: nx.DiGraph) -> Tuple[float, List[str]]:
        """Detect fork structures (common parents of u and v)."""
        parents_u = set(dag.predecessors(u))
        parents_v = set(dag.predecessors(v))
        common = parents_u & parents_v

        if not common:
            return 0.0, []

        score = min(1.0, len(common) * 0.25)
        return score, list(common)

    def _check_mr_egger(self, data: Dict) -> float:
        """Check MR-Egger intercept for pleiotropy-driven confounding."""
        intercept_p = data.get('egger_intercept_pvalue', None)
        if intercept_p is not None and intercept_p < self.config.egger_intercept_threshold:
            return 0.7

        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)
        if any('pleiotropy' in str(ev).lower() for ev in evidence):
            return 0.4
        return 0.0

    def _check_collider_bias(self, u: str, v: str,
                              dag: nx.DiGraph) -> Tuple[float, List[str]]:
        """Detect collider bias (conditioning on a common effect)."""
        children_u = set(dag.successors(u))
        children_v = set(dag.successors(v))
        colliders = children_u & children_v

        if not colliders:
            return 0.0, []

        conditioned = [c for c in colliders if dag.nodes[c].get('conditioned', False)]
        if conditioned:
            return min(1.0, len(conditioned) * 0.3 * self.config.collider_amplification_factor), conditioned
        return min(0.3, len(colliders) * 0.1), list(colliders)

    def _compute_adjustment_set(self, u: str, v: str,
                                 dag: nx.DiGraph) -> List[str]:
        """Compute minimal adjustment set using backdoor criterion."""
        parents_u = set(dag.predecessors(u))
        ancestors_u = nx.ancestors(dag, u) if u in dag else set()

        descendants_u = nx.descendants(dag, u) if u in dag else set()
        adjustment = parents_u - descendants_u - {v}

        if len(adjustment) > self.config.max_adjustment_set_size:
            scored = []
            for node in adjustment:
                n_data = dag.nodes.get(node, {})
                importance = n_data.get('causal_importance', 0)
                scored.append((node, importance))
            scored.sort(key=lambda x: x[1], reverse=True)
            adjustment = {s[0] for s in scored[:self.config.max_adjustment_set_size]}

        return sorted(adjustment)
