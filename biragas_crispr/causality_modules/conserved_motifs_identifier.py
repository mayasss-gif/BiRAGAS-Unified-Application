"""
Phase 6: COMPARATIVE EVOLUTION — Module 3
ConservedMotifsIdentifier (INTENT I_04 Module 3)
==================================================
Identifies conserved causal motifs across patient subgroups.

Motif Types:
  1. Feed-Forward Loop (FFL): A -> B -> C, A -> C
  2. Cascade: A -> B -> C (linear chain)
  3. Hub-Spoke: Central regulator driving multiple programs
  4. Diamond: A -> B, A -> C, B -> D, C -> D
  5. Convergent: Multiple regulators -> single program -> disease

Conservation = present in >= threshold fraction of patient DAGs/subgroups.

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotifsConfig:
    conservation_threshold: float = 0.50
    min_motif_frequency: int = 3
    max_motif_size: int = 5
    top_n_motifs: int = 20


class ConservedMotifsIdentifier:
    """Identifies conserved causal motifs across patient subgroups."""

    def __init__(self, config: Optional[MotifsConfig] = None):
        self.config = config or MotifsConfig()

    def identify_motifs(self, patient_dags: Dict[str, nx.DiGraph],
                        consensus_dag: Optional[nx.DiGraph] = None) -> Dict:
        """Identify conserved motifs across patient DAGs."""
        all_motifs = {
            'feed_forward_loops': [],
            'cascades': [],
            'hub_spokes': [],
            'diamonds': [],
            'convergent': [],
        }

        ref_dag = consensus_dag or self._merge_dags(patient_dags)

        ffl = self._find_feed_forward_loops(ref_dag)
        cascades = self._find_cascades(ref_dag)
        hubs = self._find_hub_spokes(ref_dag)
        diamonds = self._find_diamonds(ref_dag)
        convergent = self._find_convergent(ref_dag)

        all_motifs['feed_forward_loops'] = self._score_conservation(ffl, patient_dags)
        all_motifs['cascades'] = self._score_conservation(cascades, patient_dags)
        all_motifs['hub_spokes'] = self._score_conservation(hubs, patient_dags)
        all_motifs['diamonds'] = self._score_conservation(diamonds, patient_dags)
        all_motifs['convergent'] = self._score_conservation(convergent, patient_dags)

        conserved = {}
        for mtype, motifs in all_motifs.items():
            conserved[mtype] = [m for m in motifs
                                if m['conservation'] >= self.config.conservation_threshold]

        total_conserved = sum(len(v) for v in conserved.values())

        return {
            'all_motifs': all_motifs,
            'conserved_motifs': conserved,
            'summary': {
                'total_motifs_found': sum(len(v) for v in all_motifs.values()),
                'total_conserved': total_conserved,
                'by_type': {k: len(v) for k, v in conserved.items()},
                'most_conserved': self._top_motifs(conserved),
            },
        }

    def _find_feed_forward_loops(self, dag: nx.DiGraph) -> List[Dict]:
        """Find A -> B -> C, A -> C patterns."""
        ffls = []
        for a in dag.nodes():
            for b in dag.successors(a):
                for c in dag.successors(b):
                    if dag.has_edge(a, c) and a != c:
                        ffls.append({
                            'type': 'feed_forward_loop',
                            'nodes': [a, b, c],
                            'edges': [(a, b), (b, c), (a, c)],
                        })
        return ffls

    def _find_cascades(self, dag: nx.DiGraph) -> List[Dict]:
        """Find linear causal chains A -> B -> C through layer hierarchy."""
        cascades = []
        layer_order = {'source': 0, 'regulatory': 1, 'program': 2, 'trait': 3}

        for node in dag.nodes():
            node_layer = dag.nodes[node].get('layer', '')
            if node_layer not in ('source', 'regulatory'):
                continue
            for path in self._dfs_paths(dag, node, max_length=4):
                if len(path) >= 3:
                    layers = [dag.nodes[n].get('layer', '') for n in path]
                    layer_vals = [layer_order.get(l, -1) for l in layers]
                    if all(layer_vals[i] <= layer_vals[i + 1] for i in range(len(layer_vals) - 1)):
                        cascades.append({
                            'type': 'cascade',
                            'nodes': path,
                            'edges': [(path[i], path[i + 1]) for i in range(len(path) - 1)],
                            'layers': layers,
                        })
                        if len(cascades) >= 100:
                            return cascades
        return cascades

    def _find_hub_spokes(self, dag: nx.DiGraph) -> List[Dict]:
        """Find central regulators driving multiple programs."""
        hubs = []
        for node, data in dag.nodes(data=True):
            if data.get('layer') != 'regulatory':
                continue
            programs = [s for s in dag.successors(node)
                        if dag.nodes[s].get('layer') == 'program']
            if len(programs) >= 3:
                hubs.append({
                    'type': 'hub_spoke',
                    'nodes': [node] + programs,
                    'edges': [(node, p) for p in programs],
                    'hub': node,
                    'n_spokes': len(programs),
                })
        return hubs

    def _find_diamonds(self, dag: nx.DiGraph) -> List[Dict]:
        """Find diamond patterns: A -> B, A -> C, B -> D, C -> D."""
        diamonds = []
        for a in dag.nodes():
            children = list(dag.successors(a))
            if len(children) < 2:
                continue
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    b, c = children[i], children[j]
                    b_children = set(dag.successors(b))
                    c_children = set(dag.successors(c))
                    common = b_children & c_children
                    for d in common:
                        diamonds.append({
                            'type': 'diamond',
                            'nodes': [a, b, c, d],
                            'edges': [(a, b), (a, c), (b, d), (c, d)],
                        })
                        if len(diamonds) >= 100:
                            return diamonds
        return diamonds

    def _find_convergent(self, dag: nx.DiGraph) -> List[Dict]:
        """Find multiple regulators converging on a single program."""
        convergent = []
        for node, data in dag.nodes(data=True):
            if data.get('layer') != 'program':
                continue
            regulators = [p for p in dag.predecessors(node)
                          if dag.nodes[p].get('layer') == 'regulatory']
            if len(regulators) >= 2:
                convergent.append({
                    'type': 'convergent',
                    'nodes': regulators + [node],
                    'edges': [(r, node) for r in regulators],
                    'target_program': node,
                    'n_regulators': len(regulators),
                })
        return convergent

    def _score_conservation(self, motifs: List[Dict],
                             patient_dags: Dict[str, nx.DiGraph]) -> List[Dict]:
        """Score how conserved each motif is across patient DAGs."""
        n_patients = len(patient_dags)
        scored = []

        for motif in motifs:
            edges = motif['edges']
            count = 0
            for pid, dag in patient_dags.items():
                if all(dag.has_edge(u, v) for u, v in edges):
                    count += 1
            conservation = count / n_patients if n_patients > 0 else 0.0
            scored.append({**motif, 'conservation': round(conservation, 3),
                           'n_patients': count})

        scored.sort(key=lambda x: x['conservation'], reverse=True)
        return scored[:self.config.top_n_motifs]

    def _dfs_paths(self, dag: nx.DiGraph, start: str,
                    max_length: int = 4) -> List[List[str]]:
        """Yield simple paths from start up to max_length."""
        paths = []
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) >= max_length:
                paths.append(path)
                continue
            extended = False
            for succ in dag.successors(node):
                if succ not in path:
                    stack.append((succ, path + [succ]))
                    extended = True
            if not extended and len(path) >= 3:
                paths.append(path)
        return paths

    def _merge_dags(self, patient_dags: Dict[str, nx.DiGraph]) -> nx.DiGraph:
        """Merge patient DAGs into a union DAG."""
        merged = nx.DiGraph()
        for dag in patient_dags.values():
            for node, data in dag.nodes(data=True):
                if node not in merged:
                    merged.add_node(node, **data)
            for u, v, data in dag.edges(data=True):
                if not merged.has_edge(u, v):
                    merged.add_edge(u, v, **data)
        return merged

    def _top_motifs(self, conserved: Dict[str, List]) -> List[Dict]:
        """Get top conserved motifs across all types."""
        all_motifs = []
        for mtype, motifs in conserved.items():
            all_motifs.extend(motifs)
        all_motifs.sort(key=lambda x: x['conservation'], reverse=True)
        return all_motifs[:10]
