"""
CRISPRReportGenerator — CRISPR-Specific Sections for Final Clinical Report
=============================================================================
ResponseFormatter includes perturbation_ace in target dossiers but doesn't
generate dedicated CRISPR analysis sections. This module adds:
1. CRISPR Evidence Summary — how many edges have CRISPR support
2. CRISPR Driver Profile — top CRISPR drivers with ACE + essentiality
3. Knockout Prediction Summary — top knockout targets from CRISPR engine
4. CRISPR Validation Recommendations — experiments needed to fill gaps
"""

import logging
from typing import Any, Dict, List

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase7.report")


class CRISPRReportGenerator:
    """
    Generates CRISPR-specific report sections for the final clinical report.
    """

    def generate(self, dag: nx.DiGraph,
                 knockout_results: Dict = None,
                 quality_report: Dict = None,
                 gap_report: Dict = None) -> Dict:
        """
        Generate comprehensive CRISPR report sections.
        """
        report = {}

        # Section 1: CRISPR Evidence Summary
        crispr_genes = []
        for n in dag.nodes():
            nd = dag.nodes[n]
            if nd.get('layer') != 'regulatory':
                continue
            ace = nd.get('perturbation_ace', 0)
            if isinstance(ace, (int, float)) and ace != 0:
                crispr_genes.append({
                    'gene': n,
                    'ace': round(ace, 4),
                    'essentiality': nd.get('essentiality_tag', 'Unknown'),
                    'alignment': nd.get('therapeutic_alignment', 'Unknown'),
                    'tier': nd.get('network_tier', 'Unknown'),
                })

        crispr_genes.sort(key=lambda g: g['ace'])

        report['crispr_evidence_summary'] = {
            'total_crispr_genes': len(crispr_genes),
            'drivers': sum(1 for g in crispr_genes if g['ace'] <= -0.1),
            'strong_drivers': sum(1 for g in crispr_genes if g['ace'] <= -0.3),
            'core_essential': sum(1 for g in crispr_genes if g['essentiality'] == 'Core Essential'),
            'aggravating': sum(1 for g in crispr_genes if g['alignment'] == 'Aggravating'),
        }

        # Section 2: Top CRISPR Drivers
        report['top_crispr_drivers'] = crispr_genes[:20]

        # Section 3: Quality Impact
        if quality_report:
            report['crispr_quality_impact'] = {
                'mean_quality_with_crispr': quality_report.get('mean_quality_with', 0),
                'mean_quality_without_crispr': quality_report.get('mean_quality_without', 0),
                'quality_boost': quality_report.get('quality_boost_mean', 0),
                'edges_saved_by_crispr': quality_report.get('quality_saved', 0),
            }

        # Section 4: Knockout Predictions
        if knockout_results:
            ko_list = []
            for gene, ko in list(knockout_results.items())[:20]:
                if hasattr(ko, 'ensemble'):
                    ko_list.append({'gene': gene, 'score': round(ko.ensemble, 4), 'direction': getattr(ko, 'direction', 'unknown')})
                elif isinstance(ko, dict):
                    ko_list.append({'gene': gene, 'score': round(ko.get('ensemble', ko.get('ensemble_score', 0)), 4), 'direction': ko.get('direction', 'unknown')})
            ko_list.sort(key=lambda k: -k['score'])
            report['knockout_predictions'] = ko_list

        # Section 5: Validation Recommendations
        if gap_report:
            report['validation_recommendations'] = gap_report.get('recommended_experiments', [])[:10]

        # Section 6: Clinical Summary Text
        n_drivers = report['crispr_evidence_summary']['drivers']
        n_strong = report['crispr_evidence_summary']['strong_drivers']
        n_essential = report['crispr_evidence_summary']['core_essential']
        n_total = report['crispr_evidence_summary']['total_crispr_genes']

        report['clinical_narrative'] = (
            f"CRISPR screening identified {n_total} genes with perturbation effects. "
            f"Of these, {n_drivers} are causal drivers (ACE ≤ -0.1) and {n_strong} are strong drivers (ACE ≤ -0.3). "
            f"{n_essential} genes are Core Essential (targeting risks lethality). "
            f"{'The top drivers are suitable for therapeutic targeting.' if n_drivers - n_essential > 0 else 'Caution: most drivers are essential.'} "
            f"CRISPR evidence raises mean edge quality from "
            f"{report.get('crispr_quality_impact', {}).get('mean_quality_without_crispr', 'N/A')} to "
            f"{report.get('crispr_quality_impact', {}).get('mean_quality_with_crispr', 'N/A')} "
            f"(saving {report.get('crispr_quality_impact', {}).get('edges_saved_by_crispr', 0)} edges from quality failure)."
        )

        logger.info(f"CRISPRReportGenerator: {n_total} genes, {n_drivers} drivers, {n_strong} strong")
        return report
