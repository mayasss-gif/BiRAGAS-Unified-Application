"""
Phase 5: PHARMA INTERVENTION — Module 5
CombinationAnalyzer (INTENT I_03 Module 5)
============================================
Analyzes drug combination strategies using the causal DAG.

Combination Logic:
  1. Synergy Detection: Targets in complementary pathways
  2. Redundancy Avoidance: Skip targets with overlapping effects
  3. Safety Constraints: Combined off-target burden
  4. Coverage Optimization: Maximize disease pathway coverage
  5. Resistance Prevention: Block compensation pathways

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CombinationConfig:
    max_combination_size: int = 3
    min_synergy_score: float = 0.40
    max_off_target_burden: float = 0.60
    pathway_overlap_penalty: float = 0.30
    top_n_combinations: int = 10


class CombinationAnalyzer:
    """Analyzes drug combination strategies from causal DAG topology."""

    def __init__(self, config: Optional[CombinationConfig] = None):
        self.config = config or CombinationConfig()

    def analyze_combinations(self, dag: nx.DiGraph,
                             candidate_targets: List[str],
                             disease_node: str = "Disease_Activity",
                             safety_scores: Optional[Dict] = None,
                             efficacy_scores: Optional[Dict] = None) -> Dict:
        """Analyze all valid combinations of candidate targets."""
        safety_scores = safety_scores or {}
        efficacy_scores = efficacy_scores or {}

        valid_targets = [t for t in candidate_targets if t in dag]
        all_combos = []

        for size in range(2, min(self.config.max_combination_size + 1, len(valid_targets) + 1)):
            for combo in combinations(valid_targets, size):
                result = self._evaluate_combination(
                    dag, list(combo), disease_node, safety_scores, efficacy_scores)
                if result['synergy_score'] >= self.config.min_synergy_score:
                    all_combos.append(result)

        all_combos.sort(key=lambda x: x['combined_score'], reverse=True)
        top = all_combos[:self.config.top_n_combinations]

        return {
            'combinations': top,
            'total_evaluated': len(all_combos),
            'summary': {
                'best_pair': top[0] if top else None,
                'n_synergistic': len(all_combos),
                'coverage_distribution': self._coverage_distribution(top),
            },
        }

    def find_optimal_pair(self, dag: nx.DiGraph,
                          candidate_targets: List[str],
                          disease_node: str = "Disease_Activity") -> Dict:
        """Find the single best target pair."""
        result = self.analyze_combinations(dag, candidate_targets, disease_node)
        combos = result.get('combinations', [])
        pairs = [c for c in combos if len(c['targets']) == 2]
        return pairs[0] if pairs else {'targets': [], 'combined_score': 0}

    def _evaluate_combination(self, dag: nx.DiGraph, targets: List[str],
                               disease_node: str,
                               safety_scores: Dict,
                               efficacy_scores: Dict) -> Dict:
        """Evaluate a specific target combination."""
        synergy = self._compute_synergy(dag, targets, disease_node)
        coverage = self._compute_coverage(dag, targets, disease_node)
        safety = self._compute_combined_safety(targets, safety_scores, dag)
        redundancy = self._compute_redundancy(dag, targets)
        resistance_block = self._compute_resistance_blocking(dag, targets, disease_node)

        combined = (synergy * 0.30 + coverage * 0.25 +
                    safety * 0.20 + (1.0 - redundancy) * 0.10 +
                    resistance_block * 0.15)

        return {
            'targets': targets,
            'combined_score': round(combined, 4),
            'synergy_score': round(synergy, 4),
            'coverage': round(coverage, 4),
            'safety': round(safety, 4),
            'redundancy': round(redundancy, 4),
            'resistance_blocking': round(resistance_block, 4),
        }

    def _compute_synergy(self, dag: nx.DiGraph, targets: List[str],
                          disease_node: str) -> float:
        """Synergy = targets affect different pathway classes."""
        all_classes = []
        for gene in targets:
            programs = [s for s in dag.successors(gene)
                        if dag.nodes[s].get('layer') == 'program']
            classes = {dag.nodes[p].get('main_class', 'Unknown') for p in programs}
            all_classes.append(classes)

        if len(all_classes) < 2:
            return 0.0

        union = set()
        intersection = all_classes[0].copy()
        for cs in all_classes:
            union |= cs
            intersection &= cs

        if not union:
            return 0.0

        unique_ratio = (len(union) - len(intersection)) / len(union)
        return min(1.0, unique_ratio + 0.2)

    def _compute_coverage(self, dag: nx.DiGraph, targets: List[str],
                           disease_node: str) -> float:
        """Fraction of disease-driving pathways covered by combination."""
        disease_programs = set()
        if disease_node in dag:
            for pred in dag.predecessors(disease_node):
                if dag.nodes[pred].get('layer') == 'program':
                    disease_programs.add(pred)

        if not disease_programs:
            return 0.5

        covered = set()
        for gene in targets:
            gene_programs = {s for s in dag.successors(gene)
                             if dag.nodes[s].get('layer') == 'program'}
            covered |= (gene_programs & disease_programs)

        return len(covered) / len(disease_programs)

    def _compute_combined_safety(self, targets: List[str],
                                  safety_scores: Dict,
                                  dag: nx.DiGraph) -> float:
        """Combined safety = minimum individual safety (weakest link)."""
        scores = []
        for gene in targets:
            s = safety_scores.get(gene, {}).get('score', 0.5)
            scores.append(s)
        return min(scores) if scores else 0.5

    def _compute_redundancy(self, dag: nx.DiGraph,
                             targets: List[str]) -> float:
        """Overlap of downstream effects (redundancy is wasteful)."""
        downstream = []
        for gene in targets:
            desc = set(nx.descendants(dag, gene)) if gene in dag else set()
            downstream.append(desc)

        if len(downstream) < 2:
            return 0.0

        union = set()
        for ds in downstream:
            union |= ds

        pairwise_overlaps = []
        for i in range(len(downstream)):
            for j in range(i + 1, len(downstream)):
                if downstream[i] or downstream[j]:
                    overlap = len(downstream[i] & downstream[j])
                    total = len(downstream[i] | downstream[j])
                    pairwise_overlaps.append(overlap / max(total, 1))

        return float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0

    def _compute_resistance_blocking(self, dag: nx.DiGraph,
                                      targets: List[str],
                                      disease_node: str) -> float:
        """Score how well the combination blocks resistance paths."""
        all_programs = set()
        for gene in targets:
            all_programs |= {s for s in dag.successors(gene)
                             if dag.nodes[s].get('layer') == 'program'}

        disease_programs = set()
        if disease_node in dag:
            disease_programs = {p for p in dag.predecessors(disease_node)
                                if dag.nodes[p].get('layer') == 'program'}

        if not disease_programs:
            return 0.5

        blocked = all_programs & disease_programs
        return len(blocked) / len(disease_programs)

    def _coverage_distribution(self, combos: List[Dict]) -> Dict:
        """Summarize coverage across top combinations."""
        if not combos:
            return {}
        coverages = [c['coverage'] for c in combos]
        return {
            'mean': round(float(np.mean(coverages)), 3),
            'max': round(max(coverages), 3),
            'min': round(min(coverages), 3),
        }
