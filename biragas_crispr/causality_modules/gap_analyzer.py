"""
Phase 7: INSPECTION AND LLM ARBITRATION — Module 2
GapAnalyzer (INTENT I_06 Module 2)
=====================================
Identifies gaps in the causal evidence chain.

Gap Categories:
  1. Missing Evidence Gaps: Edges with insufficient support
  2. Structural Gaps: Missing expected connections in the DAG
  3. Layer Coverage Gaps: Incomplete representation per layer
  4. Pathway Gaps: Disease-connected pathways with no upstream regulators
  5. Validation Gaps: Claims lacking independent validation

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GapAnalyzerConfig:
    min_evidence_per_edge: int = 2
    min_confidence: float = 0.40
    orphan_program_threshold: int = 1
    expected_layers: List[str] = None

    def __post_init__(self):
        if self.expected_layers is None:
            self.expected_layers = ['source', 'regulatory', 'program', 'trait']


class GapAnalyzer:
    """Identifies gaps in causal evidence and DAG structure."""

    def __init__(self, config: Optional[GapAnalyzerConfig] = None):
        self.config = config or GapAnalyzerConfig()

    def analyze_gaps(self, dag: nx.DiGraph,
                     disease_node: str = "Disease_Activity") -> Dict:
        """Full gap analysis of the causal DAG."""
        evidence_gaps = self._find_evidence_gaps(dag)
        structural_gaps = self._find_structural_gaps(dag, disease_node)
        layer_gaps = self._find_layer_gaps(dag)
        pathway_gaps = self._find_pathway_gaps(dag, disease_node)
        validation_gaps = self._find_validation_gaps(dag)

        all_gaps = (evidence_gaps + structural_gaps + layer_gaps +
                    pathway_gaps + validation_gaps)

        priority = self._prioritize_gaps(all_gaps, dag)

        return {
            'gaps': {
                'evidence': evidence_gaps,
                'structural': structural_gaps,
                'layer_coverage': layer_gaps,
                'pathway': pathway_gaps,
                'validation': validation_gaps,
            },
            'prioritized': priority,
            'summary': {
                'total_gaps': len(all_gaps),
                'by_category': {
                    'evidence': len(evidence_gaps),
                    'structural': len(structural_gaps),
                    'layer_coverage': len(layer_gaps),
                    'pathway': len(pathway_gaps),
                    'validation': len(validation_gaps),
                },
                'critical_gaps': len([g for g in all_gaps if g.get('severity') == 'critical']),
            },
        }

    def _find_evidence_gaps(self, dag: nx.DiGraph) -> List[Dict]:
        """Find edges with insufficient evidence."""
        gaps = []
        for u, v, data in dag.edges(data=True):
            evidence = data.get('evidence', [])
            if isinstance(evidence, set):
                evidence = list(evidence)

            n_types = len(set(evidence))
            conf = data.get('confidence_score', 0)

            if n_types < self.config.min_evidence_per_edge:
                gaps.append({
                    'category': 'evidence',
                    'type': 'insufficient_evidence',
                    'edge': (u, v),
                    'n_evidence_types': n_types,
                    'confidence': conf,
                    'severity': 'critical' if n_types == 0 else 'warning',
                    'recommendation': f'Add evidence for {u} -> {v} (currently {n_types} types)',
                })

            if conf < self.config.min_confidence and n_types > 0:
                gaps.append({
                    'category': 'evidence',
                    'type': 'low_confidence',
                    'edge': (u, v),
                    'confidence': conf,
                    'severity': 'warning',
                    'recommendation': f'Validate edge {u} -> {v} (confidence={conf:.2f})',
                })

        return gaps

    def _find_structural_gaps(self, dag: nx.DiGraph,
                               disease_node: str) -> List[Dict]:
        """Find structural issues in the DAG."""
        gaps = []

        isolates = list(nx.isolates(dag))
        for node in isolates:
            gaps.append({
                'category': 'structural',
                'type': 'isolated_node',
                'node': node,
                'severity': 'warning',
                'recommendation': f'Node {node} is disconnected from the DAG',
            })

        regulatory = [n for n, d in dag.nodes(data=True) if d.get('layer') == 'regulatory']
        for gene in regulatory:
            if dag.out_degree(gene) == 0:
                gaps.append({
                    'category': 'structural',
                    'type': 'sink_regulator',
                    'node': gene,
                    'severity': 'warning',
                    'recommendation': f'Gene {gene} has no outgoing causal edges',
                })

        if disease_node in dag and dag.in_degree(disease_node) == 0:
            gaps.append({
                'category': 'structural',
                'type': 'disconnected_disease',
                'node': disease_node,
                'severity': 'critical',
                'recommendation': 'Disease node has no incoming causal edges',
            })

        return gaps

    def _find_layer_gaps(self, dag: nx.DiGraph) -> List[Dict]:
        """Check layer coverage completeness."""
        gaps = []
        layer_counts = {}
        for _, data in dag.nodes(data=True):
            layer = data.get('layer', 'unknown')
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        for expected in self.config.expected_layers:
            if expected not in layer_counts or layer_counts[expected] == 0:
                gaps.append({
                    'category': 'layer_coverage',
                    'type': 'missing_layer',
                    'layer': expected,
                    'severity': 'critical',
                    'recommendation': f'No nodes in expected layer: {expected}',
                })

        cross_layer_edges = 0
        for u, v in dag.edges():
            u_layer = dag.nodes[u].get('layer', '')
            v_layer = dag.nodes[v].get('layer', '')
            if u_layer != v_layer:
                cross_layer_edges += 1

        if cross_layer_edges == 0 and dag.number_of_edges() > 0:
            gaps.append({
                'category': 'layer_coverage',
                'type': 'no_cross_layer_edges',
                'severity': 'critical',
                'recommendation': 'No edges connecting different layers',
            })

        return gaps

    def _find_pathway_gaps(self, dag: nx.DiGraph,
                            disease_node: str) -> List[Dict]:
        """Find pathway-level gaps."""
        gaps = []
        programs = [n for n, d in dag.nodes(data=True) if d.get('layer') == 'program']

        for prog in programs:
            n_regulators = sum(1 for p in dag.predecessors(prog)
                               if dag.nodes[p].get('layer') == 'regulatory')
            if n_regulators < self.config.orphan_program_threshold:
                gaps.append({
                    'category': 'pathway',
                    'type': 'orphan_program',
                    'node': prog,
                    'n_regulators': n_regulators,
                    'severity': 'warning',
                    'recommendation': f'Program {prog} has only {n_regulators} upstream regulators',
                })

            if disease_node in dag and not dag.has_edge(prog, disease_node):
                has_path = False
                try:
                    has_path = nx.has_path(dag, prog, disease_node)
                except nx.NetworkXError:
                    pass
                if not has_path:
                    gaps.append({
                        'category': 'pathway',
                        'type': 'disconnected_program',
                        'node': prog,
                        'severity': 'warning',
                        'recommendation': f'Program {prog} has no path to disease node',
                    })

        return gaps

    def _find_validation_gaps(self, dag: nx.DiGraph) -> List[Dict]:
        """Find claims lacking independent validation."""
        gaps = []
        for u, v, data in dag.edges(data=True):
            evidence = data.get('evidence', [])
            if isinstance(evidence, set):
                evidence = list(evidence)

            categories = set()
            for ev in evidence:
                ev_str = str(ev).lower()
                if 'statistical' in ev_str or 'correlation' in ev_str:
                    categories.add('statistical')
                elif 'mendelian' in ev_str or 'gwas' in ev_str:
                    categories.add('genetic')
                elif 'crispr' in ev_str or 'perturbation' in ev_str:
                    categories.add('experimental')
                elif 'signor' in ev_str or 'database' in ev_str:
                    categories.add('curated')
                else:
                    categories.add('other')

            if len(categories) == 1 and 'statistical' in categories:
                gaps.append({
                    'category': 'validation',
                    'type': 'statistical_only',
                    'edge': (u, v),
                    'severity': 'warning',
                    'recommendation': f'Edge {u} -> {v} supported only by statistical evidence',
                })

        return gaps

    def _prioritize_gaps(self, gaps: List[Dict], dag: nx.DiGraph) -> List[Dict]:
        """Prioritize gaps by severity and impact."""
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        sorted_gaps = sorted(gaps, key=lambda g: severity_order.get(g.get('severity', 'info'), 2))
        return sorted_gaps[:20]
