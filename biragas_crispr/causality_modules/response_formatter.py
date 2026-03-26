"""
Phase 7: INSPECTION AND LLM ARBITRATION — Module 4
ResponseFormatter (Combined for all modules in each intent)
============================================================
Formats pipeline outputs into structured, human-readable reports.

Output Formats:
  1. Executive Summary (key findings, top targets, confidence)
  2. Detailed Technical Report (all phases, all metrics)
  3. Target Dossier (per-gene comprehensive profile)
  4. Comparison Report (subgroup differences)
  5. JSON Export (machine-readable complete output)

Organization: Ayass Bioscience LLC
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


@dataclass
class FormatterConfig:
    include_executive_summary: bool = True
    include_technical_details: bool = True
    include_target_dossiers: bool = True
    top_n_targets: int = 10
    confidence_labels: Dict[str, float] = None

    def __post_init__(self):
        if self.confidence_labels is None:
            self.confidence_labels = {
                'Very High': 0.85, 'High': 0.65, 'Medium': 0.40, 'Low': 0.0
            }


class ResponseFormatter:
    """Formats all pipeline outputs into structured reports."""

    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()

    def format_full_report(self, dag: nx.DiGraph,
                           phase_results: Dict[str, Any],
                           disease: str = "SLE") -> Dict:
        """Generate complete formatted report from all pipeline phases."""
        report = {
            'meta': {
                'pipeline': 'BiRAGAS Causality Inference Framework',
                'version': '2.0.0',
                'disease': disease,
                'generated_at': datetime.now().isoformat(),
                'organization': 'Ayass Bioscience LLC',
                'dag_stats': {
                    'n_nodes': dag.number_of_nodes(),
                    'n_edges': dag.number_of_edges(),
                    'is_dag': nx.is_directed_acyclic_graph(dag),
                },
            },
        }

        if self.config.include_executive_summary:
            report['executive_summary'] = self._build_executive_summary(dag, phase_results)

        if self.config.include_technical_details:
            report['technical_details'] = self._build_technical_details(phase_results)

        if self.config.include_target_dossiers:
            report['target_dossiers'] = self._build_target_dossiers(dag, phase_results)

        report['quality_metrics'] = self._build_quality_metrics(dag, phase_results)

        return report

    def format_executive_summary(self, dag: nx.DiGraph,
                                  phase_results: Dict) -> str:
        """Generate text executive summary."""
        summary = self._build_executive_summary(dag, phase_results)
        lines = [
            f"BiRAGAS Causal Analysis Report",
            f"{'=' * 40}",
            f"Disease: {summary.get('disease', 'Unknown')}",
            f"DAG: {summary.get('dag_size', 'N/A')}",
            f"",
            f"TOP THERAPEUTIC TARGETS:",
        ]

        for i, target in enumerate(summary.get('top_targets', [])[:5], 1):
            lines.append(f"  {i}. {target['gene']} "
                         f"(Score: {target.get('score', 'N/A')}, "
                         f"Tier: {target.get('tier', 'N/A')})")

        lines.extend([
            f"",
            f"KEY FINDINGS:",
        ])
        for finding in summary.get('key_findings', []):
            lines.append(f"  - {finding}")

        lines.extend([
            f"",
            f"EVIDENCE QUALITY: {summary.get('overall_quality', 'N/A')}",
            f"CONFIDENCE: {summary.get('overall_confidence', 'N/A')}",
        ])

        return "\n".join(lines)

    def format_target_dossier(self, dag: nx.DiGraph, gene: str,
                               phase_results: Dict) -> Dict:
        """Generate a comprehensive dossier for a single target gene."""
        data = dag.nodes.get(gene, {})
        if not data:
            return {'error': f'{gene} not found in DAG'}

        dossier = {
            'gene': gene,
            'classification': {
                'network_tier': data.get('network_tier', 'Unknown'),
                'causal_tier': data.get('causal_tier', 'Unknown'),
                'therapeutic_alignment': data.get('therapeutic_alignment', 'Unknown'),
            },
            'centrality_metrics': {
                'causal_importance': data.get('causal_importance', 0),
                'apex_score': data.get('apex_score', 0),
                'betweenness': data.get('betweenness', 0),
                'disease_proximity': data.get('disease_proximity', 0),
                'pleiotropic_reach': data.get('pleiotropic_reach', 0),
            },
            'causal_evidence': {
                'is_gwas_hit': data.get('is_gwas_hit', False),
                'perturbation_ace': data.get('perturbation_ace', 0),
                'evidence_count': data.get('evidence_count', 0),
            },
            'pharmacology': {
                'druggability_score': data.get('druggability_score', None),
                'druggability_class': data.get('druggability_class', None),
                'efficacy_score': data.get('efficacy_score', None),
                'safety_score': data.get('safety_score', None),
                'safety_class': data.get('safety_class', None),
            },
            'connections': {
                'n_outgoing': dag.out_degree(gene),
                'n_incoming': dag.in_degree(gene),
                'downstream_programs': [s for s in dag.successors(gene)
                                        if dag.nodes[s].get('layer') == 'program'],
                'upstream_regulators': [p for p in dag.predecessors(gene)
                                        if dag.nodes[p].get('layer') in ('source', 'regulatory')],
            },
        }

        return dossier

    def export_json(self, report: Dict, filepath: str) -> str:
        """Export report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=NpEncoder)
        logger.info(f"Report exported to {filepath}")
        return filepath

    def export_dag_json(self, dag: nx.DiGraph, filepath: str) -> str:
        """Export DAG in node_link_data format."""
        data = nx.node_link_data(dag)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)
        logger.info(f"DAG exported to {filepath}")
        return filepath

    def _build_executive_summary(self, dag: nx.DiGraph,
                                  phase_results: Dict) -> Dict:
        """Build executive summary from pipeline results."""
        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']
        genes.sort(key=lambda x: x[1].get('causal_importance', 0), reverse=True)

        top_targets = []
        for gene, data in genes[:self.config.top_n_targets]:
            top_targets.append({
                'gene': gene,
                'score': data.get('causal_importance', 0),
                'tier': data.get('network_tier', 'Unknown'),
                'causal_tier': data.get('causal_tier', 'Unknown'),
                'druggability': data.get('druggability_class', 'Unknown'),
            })

        tier_counts = {}
        for _, d in genes:
            tier = d.get('network_tier', 'Unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        all_conf = [d.get('confidence_score', 0) for _, _, d in dag.edges(data=True)]
        mean_conf = float(np.mean(all_conf)) if all_conf else 0
        overall_confidence = self._confidence_label(mean_conf)

        key_findings = []
        key_findings.append(f"{len(genes)} regulatory genes analyzed, "
                            f"{tier_counts.get('Tier_1_Master_Regulator', 0)} master regulators identified")
        if top_targets:
            key_findings.append(f"Top target: {top_targets[0]['gene']} "
                                f"(importance={top_targets[0]['score']:.4f})")

        inspection = phase_results.get('evidence_inspection', {})
        if inspection:
            n_issues = inspection.get('summary', {}).get('total_issues', 0)
            key_findings.append(f"{n_issues} evidence quality issues identified")

        return {
            'disease': phase_results.get('disease', 'Unknown'),
            'dag_size': f"{dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges",
            'top_targets': top_targets,
            'tier_distribution': tier_counts,
            'overall_confidence': overall_confidence,
            'overall_quality': self._confidence_label(
                float(np.mean([d.get('causal_importance', 0) for _, d in genes])) if genes else 0
            ),
            'key_findings': key_findings,
        }

    def _build_technical_details(self, phase_results: Dict) -> Dict:
        """Build technical details section."""
        return {
            'phase_1_foundation': phase_results.get('dag_building', {}),
            'phase_2_centrality': phase_results.get('centrality', {}),
            'phase_3_calculus': phase_results.get('causal_calculus', {}),
            'phase_4_perturbation': phase_results.get('perturbation', {}),
            'phase_5_pharma': phase_results.get('pharma', {}),
            'phase_6_comparative': phase_results.get('comparative', {}),
            'phase_7_inspection': phase_results.get('inspection', {}),
        }

    def _build_target_dossiers(self, dag: nx.DiGraph,
                                phase_results: Dict) -> List[Dict]:
        """Build dossiers for top targets."""
        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']
        genes.sort(key=lambda x: x[1].get('causal_importance', 0), reverse=True)

        dossiers = []
        for gene, _ in genes[:self.config.top_n_targets]:
            dossiers.append(self.format_target_dossier(dag, gene, phase_results))

        return dossiers

    def _build_quality_metrics(self, dag: nx.DiGraph,
                                phase_results: Dict) -> Dict:
        """Build quality metrics section."""
        all_conf = [d.get('confidence_score', 0) for _, _, d in dag.edges(data=True)]

        return {
            'n_edges_total': dag.number_of_edges(),
            'mean_confidence': round(float(np.mean(all_conf)), 4) if all_conf else 0,
            'high_confidence_edges': sum(1 for c in all_conf if c >= 0.65),
            'low_confidence_edges': sum(1 for c in all_conf if c < 0.40),
            'is_dag': nx.is_directed_acyclic_graph(dag),
            'n_components': nx.number_weakly_connected_components(dag),
        }

    def _confidence_label(self, score: float) -> str:
        """Convert confidence score to label."""
        for label, threshold in sorted(self.config.confidence_labels.items(),
                                       key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return label
        return 'Low'
