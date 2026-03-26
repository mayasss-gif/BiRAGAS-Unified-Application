"""
Phase 4: IN-SILICO PERTURBATION — Module 3
ResistanceMechanismIdentifier (INTENT I_05 Module 3)
=====================================================
Identifies potential drug resistance mechanisms from the causal DAG.

Resistance Mechanisms Detected:
  1. Feedback Loops: Compensatory upregulation when target is inhibited
  2. Bypass Pathways: Alternative routes to disease that circumvent target
  3. Efflux/Metabolism: Genes in drug metabolism pathways
  4. Target Mutations: Genes with high variability across patients
  5. Pathway Redundancy: Multiple parallel causal routes to disease

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResistanceConfig:
    bypass_path_cutoff: int = 6
    min_bypass_confidence: float = 0.40
    feedback_detection_depth: int = 4
    redundancy_threshold: int = 3
    max_bypass_paths: int = 20


class ResistanceMechanismIdentifier:
    """Identifies resistance mechanisms that could undermine drug targeting."""

    def __init__(self, config: Optional[ResistanceConfig] = None):
        self.config = config or ResistanceConfig()

    def identify_resistance(self, dag: nx.DiGraph, target_gene: str,
                            disease_node: str = "Disease_Activity") -> Dict:
        """Identify all resistance mechanisms for a target gene."""
        if target_gene not in dag:
            return {'error': f'{target_gene} not in DAG'}

        feedback = self._detect_feedback_loops(dag, target_gene)
        bypass = self._detect_bypass_pathways(dag, target_gene, disease_node)
        redundancy = self._detect_pathway_redundancy(dag, target_gene, disease_node)
        variability = self._assess_target_variability(dag, target_gene)
        compensatory = self._detect_compensatory_genes(dag, target_gene)

        mechanisms = []
        if feedback:
            mechanisms.append({'type': 'feedback_loop', 'details': feedback})
        if bypass:
            mechanisms.append({'type': 'bypass_pathway', 'details': bypass})
        if redundancy['is_redundant']:
            mechanisms.append({'type': 'pathway_redundancy', 'details': redundancy})
        if variability['is_variable']:
            mechanisms.append({'type': 'target_variability', 'details': variability})
        if compensatory:
            mechanisms.append({'type': 'compensatory_upregulation', 'details': compensatory})

        resistance_score = self._compute_resistance_score(mechanisms)

        return {
            'target': target_gene,
            'resistance_score': round(resistance_score, 4),
            'resistance_risk': 'High' if resistance_score >= 0.7 else
                               'Medium' if resistance_score >= 0.4 else 'Low',
            'mechanisms': mechanisms,
            'n_mechanisms': len(mechanisms),
            'summary': {
                'feedback_loops': len(feedback),
                'bypass_pathways': len(bypass),
                'redundant_paths': redundancy.get('n_parallel_paths', 0),
                'compensatory_genes': len(compensatory),
            },
        }

    def identify_resistance_batch(self, dag: nx.DiGraph,
                                   targets: List[str],
                                   disease_node: str = "Disease_Activity") -> Dict:
        """Identify resistance for multiple targets."""
        results = {}
        for gene in targets:
            results[gene] = self.identify_resistance(dag, gene, disease_node)
        ranked = sorted(results.items(), key=lambda x: x[1].get('resistance_score', 0))
        return {
            'per_target': results,
            'least_resistant': [r[0] for r in ranked[:5]],
            'most_resistant': [r[0] for r in reversed(ranked[-5:])],
        }

    def _detect_feedback_loops(self, dag: nx.DiGraph,
                                target: str) -> List[Dict]:
        """Detect feedback loops that could compensate for target inhibition."""
        loops = []
        successors = list(dag.successors(target))

        for succ in successors:
            try:
                back_paths = list(nx.all_simple_paths(
                    dag, succ, target, cutoff=self.config.feedback_detection_depth))
                for path in back_paths[:5]:
                    min_conf = min(dag[path[i]][path[i + 1]].get('confidence_score', 0)
                                   for i in range(len(path) - 1))
                    loops.append({
                        'path': path,
                        'length': len(path),
                        'min_confidence': round(min_conf, 4),
                    })
            except nx.NetworkXError:
                continue

        return loops

    def _detect_bypass_pathways(self, dag: nx.DiGraph, target: str,
                                 disease_node: str) -> List[Dict]:
        """Detect alternative paths to disease that bypass the target."""
        if disease_node not in dag:
            return []

        dag_without = dag.copy()
        dag_without.remove_node(target)

        bypass_sources = [n for n, d in dag.nodes(data=True)
                          if d.get('layer') == 'regulatory' and n != target]

        bypasses = []
        for source in bypass_sources:
            if source not in dag_without:
                continue
            try:
                paths = list(nx.all_simple_paths(
                    dag_without, source, disease_node,
                    cutoff=self.config.bypass_path_cutoff))
                for path in paths[:3]:
                    path_conf = np.mean([
                        dag_without[path[i]][path[i + 1]].get('confidence_score', 0.5)
                        for i in range(len(path) - 1)
                    ])
                    if path_conf >= self.config.min_bypass_confidence:
                        bypasses.append({
                            'source': source,
                            'path': path,
                            'mean_confidence': round(float(path_conf), 4),
                        })
                    if len(bypasses) >= self.config.max_bypass_paths:
                        break
            except nx.NetworkXError:
                continue
            if len(bypasses) >= self.config.max_bypass_paths:
                break

        return bypasses

    def _detect_pathway_redundancy(self, dag: nx.DiGraph, target: str,
                                    disease_node: str) -> Dict:
        """Detect if there are redundant parallel paths to disease."""
        if disease_node not in dag or target not in dag:
            return {'is_redundant': False, 'n_parallel_paths': 0}

        try:
            all_paths = list(nx.all_simple_paths(
                dag, target, disease_node, cutoff=self.config.bypass_path_cutoff))
        except nx.NetworkXError:
            return {'is_redundant': False, 'n_parallel_paths': 0}

        intermediate_nodes = set()
        for path in all_paths:
            for node in path[1:-1]:
                intermediate_nodes.add(node)

        return {
            'is_redundant': len(all_paths) >= self.config.redundancy_threshold,
            'n_parallel_paths': len(all_paths),
            'intermediate_nodes': list(intermediate_nodes),
        }

    def _assess_target_variability(self, dag: nx.DiGraph,
                                    target: str) -> Dict:
        """Assess target gene variability across patients."""
        data = dag.nodes.get(target, {})
        patient_freq = data.get('patient_frequency', 1.0)
        n_evidence = data.get('evidence_count', 0)

        is_variable = patient_freq < 0.5 or n_evidence < 3
        return {
            'is_variable': is_variable,
            'patient_frequency': patient_freq,
            'evidence_count': n_evidence,
        }

    def _detect_compensatory_genes(self, dag: nx.DiGraph,
                                    target: str) -> List[Dict]:
        """Detect genes that could compensate when target is knocked out."""
        target_data = dag.nodes.get(target, {})
        target_programs = set(dag.successors(target)) & {
            n for n, d in dag.nodes(data=True) if d.get('layer') == 'program'}

        compensatory = []
        for gene, data in dag.nodes(data=True):
            if gene == target or data.get('layer') != 'regulatory':
                continue
            gene_programs = set(dag.successors(gene)) & {
                n for n, d in dag.nodes(data=True) if d.get('layer') == 'program'}
            overlap = target_programs & gene_programs
            if len(overlap) >= 1:
                compensatory.append({
                    'gene': gene,
                    'shared_programs': list(overlap),
                    'n_shared': len(overlap),
                    'network_tier': data.get('network_tier', 'Unknown'),
                })

        compensatory.sort(key=lambda x: x['n_shared'], reverse=True)
        return compensatory[:10]

    def _compute_resistance_score(self, mechanisms: List[Dict]) -> float:
        """Compute overall resistance risk score."""
        if not mechanisms:
            return 0.0

        type_weights = {
            'feedback_loop': 0.25,
            'bypass_pathway': 0.30,
            'pathway_redundancy': 0.20,
            'target_variability': 0.10,
            'compensatory_upregulation': 0.15,
        }

        score = 0.0
        for mech in mechanisms:
            w = type_weights.get(mech['type'], 0.1)
            score += w
        return min(1.0, score)
