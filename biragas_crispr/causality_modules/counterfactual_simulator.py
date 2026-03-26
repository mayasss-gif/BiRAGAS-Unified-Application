"""
Phase 4: IN-SILICO PERTURBATION — Module 1
CounterfactualSimulator (INTENT I_05 Module 1)
================================================
Simulates counterfactual scenarios using Pearl's do-calculus.

Implements do(X=x): Graph surgery that removes all incoming edges to X,
sets X to the intervention value, and propagates effects downstream.

Key Operations:
  1. do(gene=knockout): Simulate gene knockout (ACE -> 0)
  2. do(gene=overexpress): Simulate gene overexpression (ACE amplified)
  3. do(pathway=inhibit): Simulate pathway blockade
  4. Counterfactual comparison: P(Y|do(X=x)) vs P(Y|do(X=x'))

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualConfig:
    propagation_decay: float = 0.85
    max_propagation_depth: int = 10
    knockout_value: float = 0.0
    overexpression_multiplier: float = 2.0
    significance_threshold: float = 0.05


class CounterfactualSimulator:
    """Simulates counterfactual interventions via do-calculus on the causal DAG."""

    def __init__(self, config: Optional[CounterfactualConfig] = None):
        self.config = config or CounterfactualConfig()

    def simulate_knockout(self, dag: nx.DiGraph, target_gene: str,
                          disease_node: str = "Disease_Activity") -> Dict:
        """Simulate do(gene = knockout) and measure disease impact."""
        return self._simulate_intervention(dag, target_gene, disease_node,
                                           intervention_type='knockout')

    def simulate_overexpression(self, dag: nx.DiGraph, target_gene: str,
                                disease_node: str = "Disease_Activity") -> Dict:
        """Simulate do(gene = overexpress) and measure disease impact."""
        return self._simulate_intervention(dag, target_gene, disease_node,
                                           intervention_type='overexpression')

    def simulate_multi_target(self, dag: nx.DiGraph, targets: List[str],
                              disease_node: str = "Disease_Activity") -> Dict:
        """Simulate simultaneous intervention on multiple targets."""
        mutilated = self._graph_surgery(dag, targets)

        for gene in targets:
            if gene in mutilated:
                mutilated.nodes[gene]['intervention_value'] = self.config.knockout_value
                mutilated.nodes[gene]['intervened'] = True

        effects = self._propagate_effects(mutilated, targets, disease_node)
        baseline = self._get_baseline_activity(dag, disease_node)
        counterfactual = baseline + effects.get(disease_node, 0)

        return {
            'targets': targets,
            'intervention': 'multi_knockout',
            'baseline_disease': round(baseline, 4),
            'counterfactual_disease': round(counterfactual, 4),
            'absolute_change': round(counterfactual - baseline, 4),
            'relative_change': round((counterfactual - baseline) / max(abs(baseline), 1e-9), 4),
            'downstream_effects': {k: round(v, 4) for k, v in effects.items()},
            'n_affected_nodes': len([v for v in effects.values() if abs(v) > 0.01]),
        }

    def compare_counterfactuals(self, dag: nx.DiGraph,
                                gene_a: str, gene_b: str,
                                disease_node: str = "Disease_Activity") -> Dict:
        """Compare do(A=knockout) vs do(B=knockout)."""
        result_a = self.simulate_knockout(dag, gene_a, disease_node)
        result_b = self.simulate_knockout(dag, gene_b, disease_node)

        return {
            'comparison': f'{gene_a} vs {gene_b}',
            'gene_a': result_a,
            'gene_b': result_b,
            'differential_effect': round(
                result_a['absolute_change'] - result_b['absolute_change'], 4),
            'preferred_target': gene_a if abs(result_a['absolute_change']) > abs(result_b['absolute_change']) else gene_b,
        }

    def _simulate_intervention(self, dag: nx.DiGraph, target: str,
                                disease_node: str,
                                intervention_type: str) -> Dict:
        """Core intervention simulation."""
        if target not in dag:
            return {'error': f'{target} not found in DAG', 'target': target}

        mutilated = self._graph_surgery(dag, [target])

        if intervention_type == 'knockout':
            mutilated.nodes[target]['intervention_value'] = self.config.knockout_value
        else:
            original_ace = dag.nodes[target].get('perturbation_ace', 0)
            mutilated.nodes[target]['intervention_value'] = original_ace * self.config.overexpression_multiplier

        mutilated.nodes[target]['intervened'] = True
        effects = self._propagate_effects(mutilated, [target], disease_node)

        baseline = self._get_baseline_activity(dag, disease_node)
        counterfactual = baseline + effects.get(disease_node, 0)

        return {
            'target': target,
            'intervention': intervention_type,
            'baseline_disease': round(baseline, 4),
            'counterfactual_disease': round(counterfactual, 4),
            'absolute_change': round(counterfactual - baseline, 4),
            'relative_change': round((counterfactual - baseline) / max(abs(baseline), 1e-9), 4),
            'downstream_effects': {k: round(v, 4) for k, v in effects.items()},
            'n_affected_nodes': len([v for v in effects.values() if abs(v) > 0.01]),
            'affected_programs': [k for k, v in effects.items()
                                  if abs(v) > 0.01 and mutilated.nodes.get(k, {}).get('layer') == 'program'],
        }

    def _graph_surgery(self, dag: nx.DiGraph, targets: List[str]) -> nx.DiGraph:
        """Pearl's do-calculus: remove all incoming edges to intervention targets."""
        mutilated = dag.copy()
        for target in targets:
            incoming = list(mutilated.in_edges(target))
            mutilated.remove_edges_from(incoming)
        return mutilated

    def _propagate_effects(self, mutilated: nx.DiGraph,
                           sources: List[str],
                           disease_node: str) -> Dict[str, float]:
        """Propagate intervention effects through the DAG."""
        effects = {}
        cfg = self.config

        try:
            topo_order = list(nx.topological_sort(mutilated))
        except nx.NetworkXUnfeasible:
            logger.warning("Graph has cycles, using BFS propagation instead.")
            topo_order = list(nx.bfs_tree(mutilated, sources[0]).nodes()) if sources else []

        for source in sources:
            ace = mutilated.nodes[source].get('perturbation_ace', 0)
            intervention_val = mutilated.nodes[source].get('intervention_value', 0)
            effects[source] = intervention_val - ace

        for node in topo_order:
            if node in sources:
                continue
            total_effect = 0.0
            for pred in mutilated.predecessors(node):
                if pred in effects:
                    edge_data = mutilated[pred][node]
                    weight = edge_data.get('weight', 0.5)
                    conf = edge_data.get('confidence_score', 0.5)
                    total_effect += effects[pred] * weight * conf * cfg.propagation_decay
            if abs(total_effect) > 1e-6:
                effects[node] = total_effect

        return effects

    def _get_baseline_activity(self, dag: nx.DiGraph, disease_node: str) -> float:
        """Get baseline disease activity score."""
        if disease_node not in dag:
            return 1.0
        in_edges = dag.in_edges(disease_node, data=True)
        total = sum(d.get('weight', 0) * d.get('confidence_score', 0.5)
                    for _, _, d in in_edges)
        return total if total > 0 else 1.0
