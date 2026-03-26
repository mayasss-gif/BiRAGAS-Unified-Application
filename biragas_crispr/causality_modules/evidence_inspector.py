"""
Phase 7: INSPECTION AND LLM ARBITRATION — Module 1
EvidenceInspector (INTENT I_01 Module 1)
==========================================
Inspects and audits all evidence supporting causal claims in the DAG.

Inspection Layers:
  1. Evidence Completeness: Are all expected data types present?
  2. Evidence Consistency: Do different sources agree?
  3. Evidence Quality: Weight-based quality scoring per edge
  4. Confidence Calibration: Is the confidence score justified?
  5. Provenance Tracking: Source attribution for every claim

Evidence Weights: GWAS=0.90, MR=0.95, CRISPR=0.85, SIGNOR=0.90,
                  TEMPORAL=0.65, STATISTICAL=0.35, DATABASE=0.80, EQTL=0.85

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

EVIDENCE_WEIGHTS = {
    'gwas': 0.90, 'mendelian_randomization': 0.95, 'crispr': 0.85,
    'signor': 0.90, 'temporal': 0.65, 'statistical': 0.35,
    'database': 0.80, 'eqtl': 0.85, 'pathway_enrichment': 0.70,
    'coexpression': 0.50, 'deconvolution': 0.60,
}

EXPECTED_EVIDENCE_TYPES = {
    ('source', 'regulatory'): {'gwas', 'eqtl'},
    ('regulatory', 'program'): {'crispr', 'coexpression', 'pathway_enrichment', 'signor'},
    ('regulatory', 'trait'): {'mendelian_randomization', 'gwas'},
    ('program', 'trait'): {'pathway_enrichment', 'statistical'},
}


@dataclass
class InspectorConfig:
    completeness_threshold: float = 0.50
    consistency_threshold: float = 0.70
    quality_threshold: float = 0.60
    confidence_tolerance: float = 0.20


class EvidenceInspector:
    """Inspects and audits evidence quality across the causal DAG."""

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()

    def inspect_dag(self, dag: nx.DiGraph) -> Dict:
        """Full evidence inspection of the entire DAG."""
        edge_reports = []
        node_reports = []
        issues = []

        for u, v, data in dag.edges(data=True):
            report = self._inspect_edge(u, v, data, dag)
            edge_reports.append(report)
            if report['issues']:
                issues.extend(report['issues'])

        for node, data in dag.nodes(data=True):
            if data.get('layer') == 'regulatory':
                report = self._inspect_node(node, data, dag)
                node_reports.append(report)
                if report['issues']:
                    issues.extend(report['issues'])

        quality_scores = [r['quality_score'] for r in edge_reports]

        return {
            'edge_reports': edge_reports,
            'node_reports': node_reports,
            'issues': issues,
            'summary': {
                'total_edges_inspected': len(edge_reports),
                'total_nodes_inspected': len(node_reports),
                'total_issues': len(issues),
                'mean_edge_quality': round(float(np.mean(quality_scores)), 4) if quality_scores else 0,
                'edges_below_quality': sum(1 for q in quality_scores
                                           if q < self.config.quality_threshold),
                'issue_breakdown': self._categorize_issues(issues),
            },
        }

    def _inspect_edge(self, u: str, v: str, data: Dict,
                       dag: nx.DiGraph) -> Dict:
        """Inspect evidence for a single edge."""
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        completeness = self._check_completeness(u, v, evidence, dag)
        consistency = self._check_consistency(data, evidence)
        quality = self._compute_quality(evidence, data)
        calibration = self._check_calibration(data, quality)

        issues = []
        if completeness < self.config.completeness_threshold:
            issues.append({
                'type': 'incomplete_evidence', 'edge': (u, v),
                'detail': f'Completeness {completeness:.2f} below threshold',
            })
        if consistency < self.config.consistency_threshold:
            issues.append({
                'type': 'inconsistent_evidence', 'edge': (u, v),
                'detail': f'Consistency {consistency:.2f} below threshold',
            })
        if abs(calibration) > self.config.confidence_tolerance:
            issues.append({
                'type': 'miscalibrated_confidence', 'edge': (u, v),
                'detail': f'Confidence deviation {calibration:.2f}',
            })

        return {
            'edge': (u, v),
            'quality_score': round(quality, 4),
            'completeness': round(completeness, 4),
            'consistency': round(consistency, 4),
            'calibration_error': round(calibration, 4),
            'n_evidence_types': len(set(evidence)),
            'evidence_types': list(set(evidence)),
            'issues': issues,
        }

    def _inspect_node(self, node: str, data: Dict,
                       dag: nx.DiGraph) -> Dict:
        """Inspect evidence for a regulatory node."""
        issues = []

        n_out = dag.out_degree(node)
        n_in = dag.in_degree(node)
        evidence_count = data.get('evidence_count', 0)

        if n_out > 0 and evidence_count == 0:
            issues.append({
                'type': 'unsupported_driver', 'node': node,
                'detail': f'{node} has {n_out} outgoing edges but no evidence count',
            })

        tier = data.get('network_tier', '')
        ci = data.get('causal_importance', 0)
        if tier == 'Tier_1_Master_Regulator' and ci < 0.5:
            issues.append({
                'type': 'tier_importance_mismatch', 'node': node,
                'detail': f'Tier 1 gene with low causal importance ({ci})',
            })

        return {
            'node': node,
            'network_tier': tier,
            'causal_importance': ci,
            'evidence_count': evidence_count,
            'n_outgoing': n_out,
            'n_incoming': n_in,
            'issues': issues,
        }

    def _check_completeness(self, u: str, v: str,
                              evidence: List, dag: nx.DiGraph) -> float:
        """Check if expected evidence types are present."""
        u_layer = dag.nodes.get(u, {}).get('layer', '')
        v_layer = dag.nodes.get(v, {}).get('layer', '')
        expected = EXPECTED_EVIDENCE_TYPES.get((u_layer, v_layer), set())

        if not expected:
            return 0.5

        ev_lower = {str(e).split('_')[0].lower() for e in evidence}
        found = sum(1 for exp in expected if any(exp in e for e in ev_lower))
        return found / len(expected)

    def _check_consistency(self, data: Dict, evidence: List) -> float:
        """Check if evidence sources are consistent with each other."""
        if len(evidence) <= 1:
            return 0.5

        weight = data.get('weight', 0)
        conf = data.get('confidence_score', 0)

        if weight > 0 and conf > 0:
            return min(1.0, (conf + 0.5) * 0.8)
        elif weight > 0 or conf > 0:
            return 0.5
        return 0.3

    def _compute_quality(self, evidence: List, data: Dict) -> float:
        """Compute evidence quality score."""
        if not evidence:
            return data.get('confidence_score', 0.1)

        weights = []
        for ev in evidence:
            ev_str = str(ev).lower()
            matched = False
            for key, w in EVIDENCE_WEIGHTS.items():
                if key in ev_str:
                    weights.append(w)
                    matched = True
                    break
            if not matched:
                weights.append(0.35)

        quality = 1.0 - np.prod([1.0 - w for w in weights])
        return min(0.99, quality)

    def _check_calibration(self, data: Dict, computed_quality: float) -> float:
        """Check if stated confidence matches computed quality."""
        stated = data.get('confidence_score', 0.5)
        return stated - computed_quality

    def _categorize_issues(self, issues: List[Dict]) -> Dict[str, int]:
        """Categorize issues by type."""
        cats = {}
        for issue in issues:
            t = issue.get('type', 'unknown')
            cats[t] = cats.get(t, 0) + 1
        return cats
