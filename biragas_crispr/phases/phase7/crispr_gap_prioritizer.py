"""
CRISPRGapPrioritizer — Prioritizes Gaps for CRISPR Validation
================================================================
GapAnalyzer finds gaps but doesn't recommend CRISPR experiments.
This module identifies which gaps could be filled by CRISPR screening
and prioritizes them for experimental validation.

Priority: statistical_only gaps on (regulatory,program) edges are highest
priority — CRISPR screening would directly fill these gaps.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase7.gaps")


class CRISPRGapPrioritizer:
    """
    Identifies which evidence gaps could be filled by CRISPR experiments.
    """

    def prioritize(self, dag: nx.DiGraph, gap_report: Dict) -> Dict:
        """
        Prioritize gaps for CRISPR experimental validation.
        """
        report = {
            'total_gaps': 0,
            'crispr_fillable': 0,
            'crispr_priorities': [],
            'recommended_experiments': [],
        }

        gaps = gap_report.get('prioritized_gaps', gap_report.get('gaps', []))
        report['total_gaps'] = len(gaps)

        for gap in gaps:
            gap_type = gap.get('type', gap.get('gap_type', ''))
            edge = gap.get('edge', '')
            nodes = gap.get('nodes', gap.get('node', ''))
            severity = gap.get('severity', 'info')

            # Determine if CRISPR can fill this gap
            fillable = False
            experiment = ''
            priority = 0

            if gap_type == 'statistical_only':
                fillable = True
                experiment = f"CRISPR perturbation screen for genes in edge {edge}"
                priority = 3  # Highest

            elif gap_type == 'insufficient_evidence' and severity in ('critical', 'warning'):
                fillable = True
                experiment = f"CRISPR knockout validation for edge {edge}"
                priority = 2

            elif gap_type == 'validation_gap':
                fillable = True
                experiment = f"CRISPR functional validation for {edge or nodes}"
                priority = 2

            elif gap_type == 'low_confidence':
                fillable = True
                experiment = f"Confirmatory CRISPR screen for low-confidence edge {edge}"
                priority = 1

            elif gap_type in ('orphan_program', 'disconnected_program'):
                fillable = True
                experiment = f"CRISPR screen to identify regulators for program {nodes}"
                priority = 1

            if fillable:
                report['crispr_fillable'] += 1
                report['crispr_priorities'].append({
                    'gap_type': gap_type,
                    'edge': edge,
                    'severity': severity,
                    'priority': priority,
                    'experiment': experiment,
                })

        # Sort by priority (highest first), then severity
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        report['crispr_priorities'].sort(key=lambda x: (-x['priority'], severity_order.get(x['severity'], 3)))

        # Generate recommended experiments (deduplicated)
        seen = set()
        for p in report['crispr_priorities']:
            exp = p['experiment']
            if exp not in seen:
                seen.add(exp)
                report['recommended_experiments'].append({
                    'experiment': exp,
                    'priority': p['priority'],
                    'fills_gap': p['gap_type'],
                })

        report['recommended_experiments'] = report['recommended_experiments'][:20]
        report['crispr_priorities'] = report['crispr_priorities'][:20]

        logger.info(
            f"CRISPRGapPrioritizer: {report['total_gaps']} gaps | "
            f"{report['crispr_fillable']} CRISPR-fillable | "
            f"{len(report['recommended_experiments'])} experiments recommended"
        )

        return report
