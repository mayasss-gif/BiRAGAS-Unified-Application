"""
Phase 4: IN-SILICO PERTURBATION — Module 4
CompensationPathwayAnalyzer (INTENT I_05 Module 4)
===================================================
Analyzes compensatory pathway activation in response to interventions.

When a causal driver is inhibited, the network may rewire through:
  1. Parallel pathway activation (same function, different genes)
  2. Upstream feedback amplification
  3. Cross-pathway compensation (different functional class)
  4. Downstream buffering (effector redundancy)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompensationConfig:
    parallel_overlap_threshold: float = 0.30
    cross_pathway_depth: int = 3
    min_compensation_score: float = 0.20
    max_compensators: int = 15


class CompensationPathwayAnalyzer:
    """Analyzes compensatory pathways activated when targets are inhibited."""

    def __init__(self, config: Optional[CompensationConfig] = None):
        self.config = config or CompensationConfig()

    def analyze_compensation(self, dag: nx.DiGraph, target_gene: str,
                             disease_node: str = "Disease_Activity") -> Dict:
        """Full compensation analysis for a target gene."""
        if target_gene not in dag:
            return {'error': f'{target_gene} not in DAG'}

        parallel = self._find_parallel_pathways(dag, target_gene)
        feedback = self._find_upstream_feedback(dag, target_gene)
        cross = self._find_cross_pathway_compensation(dag, target_gene)
        downstream = self._find_downstream_buffering(dag, target_gene, disease_node)

        all_compensators = set()
        for p in parallel:
            all_compensators.add(p['gene'])
        for f in feedback:
            all_compensators.add(f['gene'])
        for c in cross:
            all_compensators.add(c['gene'])

        compensation_score = self._compute_compensation_score(
            parallel, feedback, cross, downstream)

        return {
            'target': target_gene,
            'compensation_score': round(compensation_score, 4),
            'compensation_risk': 'High' if compensation_score >= 0.6 else
                                 'Medium' if compensation_score >= 0.3 else 'Low',
            'parallel_pathways': parallel,
            'upstream_feedback': feedback,
            'cross_pathway': cross,
            'downstream_buffering': downstream,
            'all_compensator_genes': sorted(all_compensators),
            'n_compensators': len(all_compensators),
            'recommended_co_targets': self._recommend_co_targets(
                dag, target_gene, all_compensators, disease_node),
        }

    def _find_parallel_pathways(self, dag: nx.DiGraph,
                                 target: str) -> List[Dict]:
        """Find genes that drive the same programs as the target."""
        target_programs = {s for s in dag.successors(target)
                           if dag.nodes[s].get('layer') == 'program'}
        if not target_programs:
            return []

        parallel = []
        for gene, data in dag.nodes(data=True):
            if gene == target or data.get('layer') != 'regulatory':
                continue
            gene_programs = {s for s in dag.successors(gene)
                             if dag.nodes[s].get('layer') == 'program'}
            overlap = target_programs & gene_programs
            if len(overlap) > 0:
                overlap_ratio = len(overlap) / len(target_programs)
                if overlap_ratio >= self.config.parallel_overlap_threshold:
                    parallel.append({
                        'gene': gene,
                        'shared_programs': sorted(overlap),
                        'overlap_ratio': round(overlap_ratio, 3),
                        'network_tier': data.get('network_tier', 'Unknown'),
                        'causal_importance': data.get('causal_importance', 0),
                    })

        parallel.sort(key=lambda x: x['overlap_ratio'], reverse=True)
        return parallel[:self.config.max_compensators]

    def _find_upstream_feedback(self, dag: nx.DiGraph,
                                 target: str) -> List[Dict]:
        """Find upstream regulators that could amplify when target is lost."""
        upstream = list(dag.predecessors(target))
        feedback = []

        for pred in upstream:
            pred_data = dag.nodes.get(pred, {})
            if pred_data.get('layer') != 'regulatory':
                continue
            n_other_targets = sum(1 for s in dag.successors(pred)
                                  if s != target and dag.nodes[s].get('layer') in ('regulatory', 'program'))
            if n_other_targets > 0:
                feedback.append({
                    'gene': pred,
                    'n_alternative_targets': n_other_targets,
                    'causal_importance': pred_data.get('causal_importance', 0),
                })

        feedback.sort(key=lambda x: x['n_alternative_targets'], reverse=True)
        return feedback[:self.config.max_compensators]

    def _find_cross_pathway_compensation(self, dag: nx.DiGraph,
                                          target: str) -> List[Dict]:
        """Find genes in different pathway classes that connect to same disease endpoints."""
        target_data = dag.nodes.get(target, {})
        target_programs = {s for s in dag.successors(target)
                           if dag.nodes[s].get('layer') == 'program'}
        target_classes = {dag.nodes[p].get('main_class', 'Unknown') for p in target_programs}

        cross = []
        for gene, data in dag.nodes(data=True):
            if gene == target or data.get('layer') != 'regulatory':
                continue
            gene_programs = {s for s in dag.successors(gene)
                             if dag.nodes[s].get('layer') == 'program'}
            gene_classes = {dag.nodes[p].get('main_class', 'Unknown') for p in gene_programs}

            different_classes = gene_classes - target_classes - {'Unknown'}
            shared_downstream = set()
            for tp in target_programs:
                for gp in gene_programs:
                    tp_children = set(dag.successors(tp))
                    gp_children = set(dag.successors(gp))
                    shared_downstream |= (tp_children & gp_children)

            if different_classes and shared_downstream:
                cross.append({
                    'gene': gene,
                    'pathway_classes': sorted(different_classes),
                    'shared_downstream': sorted(shared_downstream),
                    'causal_importance': data.get('causal_importance', 0),
                })

        cross.sort(key=lambda x: len(x['shared_downstream']), reverse=True)
        return cross[:self.config.max_compensators]

    def _find_downstream_buffering(self, dag: nx.DiGraph, target: str,
                                    disease_node: str) -> Dict:
        """Assess downstream effector redundancy."""
        target_descendants = nx.descendants(dag, target) if target in dag else set()
        disease_ancestors = nx.ancestors(dag, disease_node) if disease_node in dag else set()

        intermediaries = target_descendants & disease_ancestors
        programs = [n for n in intermediaries if dag.nodes[n].get('layer') == 'program']

        buffered_programs = []
        for prog in programs:
            n_inputs = dag.in_degree(prog)
            if n_inputs > 1:
                buffered_programs.append({
                    'program': prog,
                    'n_inputs': n_inputs,
                    'main_class': dag.nodes[prog].get('main_class', 'Unknown'),
                })

        return {
            'n_intermediary_programs': len(programs),
            'n_buffered': len(buffered_programs),
            'buffered_programs': sorted(buffered_programs,
                                        key=lambda x: x['n_inputs'], reverse=True),
        }

    def _compute_compensation_score(self, parallel, feedback, cross, downstream) -> float:
        """Compute overall compensation risk score."""
        score = 0.0
        if parallel:
            score += min(0.3, len(parallel) * 0.05)
        if feedback:
            score += min(0.2, len(feedback) * 0.05)
        if cross:
            score += min(0.3, len(cross) * 0.05)
        if downstream.get('n_buffered', 0) > 0:
            score += min(0.2, downstream['n_buffered'] * 0.04)
        return min(1.0, score)

    def _recommend_co_targets(self, dag: nx.DiGraph, target: str,
                               compensators: Set[str],
                               disease_node: str) -> List[Dict]:
        """Recommend co-targets that would block compensation."""
        recommendations = []
        for gene in compensators:
            data = dag.nodes.get(gene, {})
            has_disease_path = dag.has_edge(gene, disease_node) or any(
                dag.has_edge(s, disease_node)
                for s in dag.successors(gene) if dag.nodes[s].get('layer') == 'program')
            if has_disease_path:
                recommendations.append({
                    'gene': gene,
                    'network_tier': data.get('network_tier', 'Unknown'),
                    'causal_importance': data.get('causal_importance', 0),
                    'rationale': 'blocks_compensation_pathway',
                })

        recommendations.sort(key=lambda x: x['causal_importance'], reverse=True)
        return recommendations[:5]
