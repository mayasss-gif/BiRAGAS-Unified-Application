"""
Phase 3: CAUSAL CALCULUS — Module 1
CausalityTester (INTENT I_02 Module 1)
=======================================
Tests whether proposed causal relationships hold using multi-modal evidence.

Confidence Gating (>= 0.65):
  Statistical Evidence (0.30) + RAG Knowledge (0.25) + MR Validation (0.25) +
  Multi-Agent Consensus (0.15) + Biological Plausibility (0.05)

5 Hallucination Categories:
  1. Spurious edges (no causal support)
  2. Confounded inference (hidden common cause)
  3. Directionality errors (reversed causal arrow)
  4. Pathway completeness gaps (missing intermediaries)
  5. Magnitude errors (effect size mismatch)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

EVIDENCE_WEIGHTS = {
    'gwas': 0.90, 'mendelian_randomization': 0.95, 'crispr': 0.85,
    'signor': 0.90, 'temporal': 0.65, 'statistical': 0.35,
    'database': 0.80, 'eqtl': 0.85,
}

HALLUCINATION_CATEGORIES = [
    'spurious_edge', 'confounded_inference', 'directionality_error',
    'pathway_completeness_gap', 'magnitude_error',
]


@dataclass
class CausalityTesterConfig:
    confidence_gate: float = 0.65
    w_statistical: float = 0.30
    w_rag_knowledge: float = 0.25
    w_mr_validation: float = 0.25
    w_multi_agent: float = 0.15
    w_plausibility: float = 0.05
    min_evidence_types: int = 2
    spurious_threshold: float = 0.30
    mr_methods: List[str] = field(default_factory=lambda: [
        'ivw', 'egger', 'weighted_median', 'weighted_mode', 'mr_presso'])


class CausalityTester:
    """Tests causal relationships in the DAG using multi-modal evidence."""

    def __init__(self, config: Optional[CausalityTesterConfig] = None):
        self.config = config or CausalityTesterConfig()

    def test_all_edges(self, dag: nx.DiGraph,
                       rag_scores: Optional[Dict] = None) -> Dict:
        """Test all edges in the DAG for causal validity.

        Args:
            dag: Enriched causal DAG from Phase 1-2.
            rag_scores: Optional dict of (u, v) -> float from Bio-RAG lookup.

        Returns:
            Dict with tested_edges, flagged_edges, summary.
        """
        rag_scores = rag_scores or {}
        tested = []
        flagged = []

        for u, v, data in dag.edges(data=True):
            result = self._test_edge(u, v, data, dag, rag_scores)
            tested.append(result)

            dag[u][v]['causal_test_score'] = result['composite_score']
            dag[u][v]['causal_test_passed'] = result['passed_gate']
            dag[u][v]['hallucination_flags'] = result['hallucination_flags']

            if not result['passed_gate']:
                flagged.append(result)

        return {
            'tested_edges': tested,
            'flagged_edges': flagged,
            'summary': {
                'total_tested': len(tested),
                'passed': sum(1 for t in tested if t['passed_gate']),
                'flagged': len(flagged),
                'hallucination_breakdown': self._hallucination_summary(tested),
            },
        }

    def _test_edge(self, u: str, v: str, data: Dict,
                   dag: nx.DiGraph, rag_scores: Dict) -> Dict:
        """Test a single edge for causal validity."""
        cfg = self.config

        stat_score = self._score_statistical(data)
        rag_score = rag_scores.get((u, v), self._estimate_rag_score(u, v, data, dag))
        mr_score = self._score_mr_evidence(data)
        agent_score = self._score_multi_agent(data, dag, u, v)
        plausibility = self._score_plausibility(u, v, data, dag)

        composite = (cfg.w_statistical * stat_score +
                     cfg.w_rag_knowledge * rag_score +
                     cfg.w_mr_validation * mr_score +
                     cfg.w_multi_agent * agent_score +
                     cfg.w_plausibility * plausibility)

        hallucination_flags = self._detect_hallucinations(u, v, data, dag, composite)
        passed = composite >= cfg.confidence_gate and len(hallucination_flags) == 0

        return {
            'edge': (u, v),
            'composite_score': round(composite, 4),
            'passed_gate': passed,
            'dimensions': {
                'statistical': round(stat_score, 4),
                'rag_knowledge': round(rag_score, 4),
                'mr_validation': round(mr_score, 4),
                'multi_agent': round(agent_score, 4),
                'plausibility': round(plausibility, 4),
            },
            'hallucination_flags': hallucination_flags,
            'evidence_types': self._count_evidence_types(data),
        }

    def _score_statistical(self, data: Dict) -> float:
        """Score based on statistical evidence (confidence, weight, p-values)."""
        conf = data.get('confidence_score', 0)
        weight = abs(data.get('weight', 0))
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        score = conf * 0.5 + min(weight, 1.0) * 0.3

        for ev in evidence:
            ev_type = ev.split('_')[0] if isinstance(ev, str) else ''
            w = EVIDENCE_WEIGHTS.get(ev_type, 0.35)
            score += w * 0.05

        return min(1.0, score)

    def _estimate_rag_score(self, u: str, v: str, data: Dict,
                            dag: nx.DiGraph) -> float:
        """Estimate RAG-like score from available biological evidence."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        bio_evidence = {'signor', 'database', 'pathway_enrichment'}
        count = sum(1 for ev in evidence
                    if any(b in str(ev).lower() for b in bio_evidence))

        u_data = dag.nodes.get(u, {})
        v_data = dag.nodes.get(v, {})

        layer_u = u_data.get('layer', '')
        layer_v = v_data.get('layer', '')

        layer_score = 0.0
        if (layer_u, layer_v) in (('source', 'regulatory'), ('regulatory', 'program'),
                                   ('program', 'trait')):
            layer_score = 0.3

        return min(1.0, count * 0.2 + layer_score + data.get('confidence_score', 0) * 0.2)

    def _score_mr_evidence(self, data: Dict) -> float:
        """Score MR validation evidence on the edge."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        mr_terms = {'mendelian_randomization_causality', 'mendelian_randomization_validated',
                    'mr_ivw', 'mr_egger', 'mr_weighted_median'}
        mr_count = sum(1 for ev in evidence if ev in mr_terms)

        if mr_count >= 2:
            return 0.9
        elif mr_count == 1:
            return 0.6

        mr_p = data.get('mr_pvalue', 1.0)
        if mr_p < 0.05:
            return 0.5
        return 0.0

    def _score_multi_agent(self, data: Dict, dag: nx.DiGraph,
                           u: str, v: str) -> float:
        """Score from multi-agent consensus (cross-validation across data sources)."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)
        n_sources = len(set(evidence))

        conf = data.get('confidence_score', 0)
        if n_sources >= 4:
            return min(1.0, 0.6 + conf * 0.4)
        elif n_sources >= 2:
            return min(1.0, 0.3 + conf * 0.3)
        return conf * 0.2

    def _score_plausibility(self, u: str, v: str, data: Dict,
                            dag: nx.DiGraph) -> float:
        """Biological plausibility based on layer ordering and known biology."""
        u_data = dag.nodes.get(u, {})
        v_data = dag.nodes.get(v, {})
        layer_order = {'source': 0, 'regulatory': 1, 'program': 2, 'trait': 3}

        u_layer = layer_order.get(u_data.get('layer', ''), -1)
        v_layer = layer_order.get(v_data.get('layer', ''), -1)

        score = 0.0
        if u_layer >= 0 and v_layer >= 0:
            if v_layer > u_layer:
                score += 0.5
            elif v_layer == u_layer:
                score += 0.2

        if u_data.get('is_gwas_hit') or v_data.get('is_gwas_hit'):
            score += 0.2

        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)
        if any('signor' in str(ev).lower() for ev in evidence):
            score += 0.3

        return min(1.0, score)

    def _detect_hallucinations(self, u: str, v: str, data: Dict,
                               dag: nx.DiGraph, composite: float) -> List[str]:
        """Detect hallucination categories in a causal edge."""
        flags = []
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        if composite < self.config.spurious_threshold and len(evidence) < self.config.min_evidence_types:
            flags.append('spurious_edge')

        u_data = dag.nodes.get(u, {})
        v_data = dag.nodes.get(v, {})
        common_preds = set(dag.predecessors(u)) & set(dag.predecessors(v))
        if len(common_preds) > 0 and composite < self.config.confidence_gate:
            flags.append('confounded_inference')

        layer_order = {'source': 0, 'regulatory': 1, 'program': 2, 'trait': 3}
        u_l = layer_order.get(u_data.get('layer', ''), -1)
        v_l = layer_order.get(v_data.get('layer', ''), -1)
        if u_l > v_l and u_l >= 0 and v_l >= 0:
            flags.append('directionality_error')

        if u_data.get('layer') == 'regulatory' and v_data.get('layer') == 'trait':
            intermediaries = [s for s in dag.successors(u)
                              if dag.nodes[s].get('layer') == 'program' and dag.has_edge(s, v)]
            if len(intermediaries) == 0 and not any('gwas' in str(ev) for ev in evidence):
                flags.append('pathway_completeness_gap')

        weight = abs(data.get('weight', 0))
        if weight > 2.0 and composite < 0.5:
            flags.append('magnitude_error')

        return flags

    def _count_evidence_types(self, data: Dict) -> int:
        """Count distinct evidence type categories."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)
        categories = set()
        for ev in evidence:
            parts = str(ev).split('_')
            categories.add(parts[0] if parts else 'unknown')
        return len(categories)

    def _hallucination_summary(self, tested: List[Dict]) -> Dict[str, int]:
        """Summarize hallucination flags across all tested edges."""
        summary = {cat: 0 for cat in HALLUCINATION_CATEGORIES}
        for t in tested:
            for flag in t.get('hallucination_flags', []):
                if flag in summary:
                    summary[flag] += 1
        return summary
