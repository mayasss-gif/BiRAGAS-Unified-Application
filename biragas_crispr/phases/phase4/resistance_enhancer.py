"""
ResistanceEnhancer — Combines Phase 4 Detailed + Agentic Inline Resistance
=============================================================================
Gap: Phase 4 ResistanceMechanismIdentifier has detailed 5-mechanism analysis
with co-target recommendations, but uses BINARY presence weights.
Knockout agent has inline resistance with CONTINUOUS scoring but only 3 factors.

This module merges both approaches:
- Uses Phase 4's detailed 5-mechanism detection (bypass, feedback, redundancy,
  variability, compensation) with its structural analysis
- Uses knockout agent's continuous scoring for fine-grained risk assessment
- Produces unified resistance profile per gene
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase4.resistance")


class ResistanceEnhancer:
    """
    Merges Phase 4 detailed resistance with CRISPR agentic inline resistance.

    Usage:
        enhancer = ResistanceEnhancer()
        dag, report = enhancer.enhance(dag, phase4_resistance, knockout_results)
    """

    def enhance(self, dag: nx.DiGraph,
                phase4_resistance: Dict = None,
                knockout_results: Dict = None) -> Tuple[nx.DiGraph, Dict]:
        """
        Merge resistance information from both sources.
        """
        report = {
            'genes_merged': 0,
            'phase4_only': 0,
            'knockout_only': 0,
            'both_sources': 0,
            'upgraded_risk': 0,
            'downgraded_risk': 0,
        }

        p4 = phase4_resistance or {}
        ko = knockout_results or {}

        all_genes = set(list(p4.keys()) + [g for g in ko.keys()])

        for gene in all_genes:
            if gene not in dag.nodes():
                continue

            nd = dag.nodes[gene]
            p4_data = p4.get(gene, {})
            ko_data = ko.get(gene)

            p4_score = p4_data.get('resistance_score', 0) if isinstance(p4_data, dict) else 0
            p4_risk = p4_data.get('resistance_risk', '') if isinstance(p4_data, dict) else ''

            ko_score = 0
            if ko_data:
                if hasattr(ko_data, 'resistance_score'):
                    ko_score = ko_data.resistance_score
                elif isinstance(ko_data, dict):
                    ko_score = ko_data.get('resistance_score', 0)

            has_p4 = p4_score > 0 or p4_risk
            has_ko = ko_score > 0

            if has_p4 and has_ko:
                # Merge: weighted average with Phase 4 getting more weight (more detailed)
                merged_score = 0.6 * p4_score + 0.4 * ko_score
                report['both_sources'] += 1
            elif has_p4:
                merged_score = p4_score
                report['phase4_only'] += 1
            elif has_ko:
                merged_score = ko_score
                report['knockout_only'] += 1
            else:
                merged_score = 0
                continue

            # Classify risk
            if merged_score >= 0.7:
                risk = 'High'
            elif merged_score >= 0.4:
                risk = 'Medium'
            else:
                risk = 'Low'

            # Check for risk upgrade/downgrade
            old_risk = nd.get('resistance_risk', '')
            if old_risk and old_risk != risk:
                if risk == 'High' and old_risk in ('Medium', 'Low'):
                    report['upgraded_risk'] += 1
                elif risk == 'Low' and old_risk in ('Medium', 'High'):
                    report['downgraded_risk'] += 1

            nd['unified_resistance_score'] = round(merged_score, 4)
            nd['unified_resistance_risk'] = risk
            nd['resistance_sources'] = ('phase4+knockout' if has_p4 and has_ko
                                        else 'phase4' if has_p4 else 'knockout')

            # Copy Phase 4 detailed mechanisms if available
            if isinstance(p4_data, dict):
                nd['resistance_mechanisms'] = p4_data.get('mechanisms', [])
                nd['resistance_n_mechanisms'] = p4_data.get('n_mechanisms', 0)

            report['genes_merged'] += 1

        logger.info(
            f"ResistanceEnhancer: {report['genes_merged']} merged | "
            f"both: {report['both_sources']} | p4: {report['phase4_only']} | ko: {report['knockout_only']}"
        )

        return dag, report
