"""
Phase 3: CAUSAL CALCULUS — Module 2
DirectionalityTester (INTENT I_02 Module 2)
============================================
Determines the correct causal direction for edges: A -> B or B -> A.

Methods:
  1. Topological Layer Ordering (Source -> Regulatory -> Program -> Trait)
  2. MR Steiger Test (variance-explained directionality)
  3. Temporal Granger Evidence (time-lagged correlation)
  4. Perturbation Asymmetry (CRISPR ACE directionality)
  5. SIGNOR Physical Interaction Directionality

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

LAYER_ORDER = {'source': 0, 'regulatory': 1, 'program': 2, 'trait': 3}


@dataclass
class DirectionalityConfig:
    layer_weight: float = 0.30
    steiger_weight: float = 0.25
    temporal_weight: float = 0.20
    perturbation_weight: float = 0.15
    signor_weight: float = 0.10
    confidence_threshold: float = 0.60
    flip_threshold: float = 0.70


class DirectionalityTester:
    """Tests and corrects edge directionality in the causal DAG."""

    def __init__(self, config: Optional[DirectionalityConfig] = None):
        self.config = config or DirectionalityConfig()

    def test_all_edges(self, dag: nx.DiGraph) -> Dict:
        """Test directionality of all edges. Returns report and modifies DAG."""
        results = []
        flipped = []
        confirmed = []

        edges_to_test = list(dag.edges(data=True))
        for u, v, data in edges_to_test:
            result = self._test_direction(u, v, data, dag)
            results.append(result)

            dag[u][v]['direction_score'] = result['forward_score']
            dag[u][v]['direction_confidence'] = result['confidence']

            if result['recommendation'] == 'flip':
                flipped.append(result)
            else:
                confirmed.append(result)

        flipped_edges = []
        for r in flipped:
            u, v = r['edge']
            if dag.has_edge(u, v) and not dag.has_edge(v, u):
                edge_data = dict(dag[u][v])
                dag.remove_edge(u, v)
                edge_data['direction_flipped'] = True
                edge_data['original_direction'] = (u, v)
                dag.add_edge(v, u, **edge_data)
                flipped_edges.append((u, v, v, u))

        if not nx.is_directed_acyclic_graph(dag):
            logger.warning("DAG has cycles after flipping. Reverting problematic flips.")
            for orig_u, orig_v, new_u, new_v in reversed(flipped_edges):
                if not nx.is_directed_acyclic_graph(dag):
                    edge_data = dict(dag[new_u][new_v])
                    dag.remove_edge(new_u, new_v)
                    edge_data.pop('direction_flipped', None)
                    edge_data.pop('original_direction', None)
                    dag.add_edge(orig_u, orig_v, **edge_data)

        return {
            'results': results,
            'summary': {
                'total_tested': len(results),
                'confirmed': len(confirmed),
                'flipped': len(flipped_edges),
                'is_dag': nx.is_directed_acyclic_graph(dag),
            },
        }

    def _test_direction(self, u: str, v: str, data: Dict,
                        dag: nx.DiGraph) -> Dict:
        """Test whether u -> v is the correct direction."""
        cfg = self.config

        layer_score = self._score_layer_ordering(u, v, dag)
        steiger_score = self._score_steiger(data)
        temporal_score = self._score_temporal(data)
        perturbation_score = self._score_perturbation(u, v, dag)
        signor_score = self._score_signor(data)

        forward_score = (cfg.layer_weight * layer_score +
                         cfg.steiger_weight * steiger_score +
                         cfg.temporal_weight * temporal_score +
                         cfg.perturbation_weight * perturbation_score +
                         cfg.signor_weight * signor_score)

        confidence = abs(forward_score - 0.5) * 2

        if forward_score < (1.0 - cfg.flip_threshold):
            recommendation = 'flip'
        elif forward_score >= cfg.confidence_threshold:
            recommendation = 'confirm'
        else:
            recommendation = 'uncertain'

        return {
            'edge': (u, v),
            'forward_score': round(forward_score, 4),
            'confidence': round(confidence, 4),
            'recommendation': recommendation,
            'dimensions': {
                'layer_ordering': round(layer_score, 4),
                'steiger': round(steiger_score, 4),
                'temporal': round(temporal_score, 4),
                'perturbation': round(perturbation_score, 4),
                'signor': round(signor_score, 4),
            },
        }

    def _score_layer_ordering(self, u: str, v: str, dag: nx.DiGraph) -> float:
        """Score based on topological layer ordering."""
        u_layer = LAYER_ORDER.get(dag.nodes[u].get('layer', ''), -1)
        v_layer = LAYER_ORDER.get(dag.nodes[v].get('layer', ''), -1)

        if u_layer < 0 or v_layer < 0:
            return 0.5
        if v_layer > u_layer:
            return 1.0
        elif v_layer == u_layer:
            return 0.5
        else:
            return 0.0

    def _score_steiger(self, data: Dict) -> float:
        """Score from MR Steiger test result."""
        steiger = data.get('steiger_direction', None)
        if steiger == 'forward':
            return 1.0
        elif steiger == 'reverse':
            return 0.0
        steiger_p = data.get('steiger_pvalue', None)
        if steiger_p is not None and steiger_p < 0.05:
            return 0.8
        return 0.5

    def _score_temporal(self, data: Dict) -> float:
        """Score from temporal/Granger evidence."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        if any('temporal' in str(ev).lower() or 'granger' in str(ev).lower() for ev in evidence):
            lag = data.get('temporal_lag', 0)
            if lag > 0:
                return 0.9
            elif lag < 0:
                return 0.1
            return 0.7
        return 0.5

    def _score_perturbation(self, u: str, v: str, dag: nx.DiGraph) -> float:
        """Score from CRISPR perturbation asymmetry."""
        u_data = dag.nodes.get(u, {})
        v_data = dag.nodes.get(v, {})

        u_ace = abs(u_data.get('perturbation_ace', 0))
        v_ace = abs(v_data.get('perturbation_ace', 0))

        if u_ace > 0 and v_ace > 0:
            if u_ace > v_ace:
                return 0.8
            elif v_ace > u_ace:
                return 0.2
        elif u_ace > 0:
            return 0.7
        elif v_ace > 0:
            return 0.3
        return 0.5

    def _score_signor(self, data: Dict) -> float:
        """Score from SIGNOR physical interaction data."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        if any('signor' in str(ev).lower() for ev in evidence):
            return 0.9
        return 0.5
