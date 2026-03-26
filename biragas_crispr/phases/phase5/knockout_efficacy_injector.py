"""
KnockoutEfficacyInjector — Feeds CRISPR KO Predictions to EfficacyPredictor
==============================================================================
Gap: EfficacyPredictor reads counterfactual results from Phase 4 basic
CounterfactualSimulator. The CRISPR agentic engine produces richer
knockout predictions (7-method ensemble, CI, resistance) that should
feed into efficacy scoring.

This module injects CRISPR knockout results as counterfactual data
so EfficacyPredictor uses them for the causal_effect dimension (0.30 weight).
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase5.efficacy")


class KnockoutEfficacyInjector:
    """
    Injects CRISPR knockout predictions into the DAG as counterfactual data
    that EfficacyPredictor can consume.

    Usage:
        injector = KnockoutEfficacyInjector()
        dag, counterfactual_results = injector.inject(dag, knockout_results)
        # Now pass counterfactual_results to EfficacyPredictor.predict_all()
    """

    def inject(self, dag: nx.DiGraph, knockout_results: Dict) -> Tuple[nx.DiGraph, Dict]:
        """
        Convert CRISPR knockout results to Phase 4 counterfactual format.

        Returns:
            (dag with injected attributes, counterfactual_results dict)
        """
        counterfactual = {}
        n_injected = 0

        for gene, ko in knockout_results.items():
            if gene not in dag.nodes():
                continue

            # Extract effect from different result types
            if hasattr(ko, 'disease_effect'):
                effect = ko.disease_effect
                effect_pct = ko.disease_effect_pct if hasattr(ko, 'disease_effect_pct') else 0
                confidence = ko.confidence if hasattr(ko, 'confidence') else 0.5
            elif isinstance(ko, dict):
                effect = ko.get('disease_effect', ko.get('absolute_change', 0))
                effect_pct = ko.get('disease_effect_pct', ko.get('relative_change', 0))
                confidence = ko.get('confidence', 0.5)
            else:
                continue

            # Convert to Phase 4 counterfactual format
            counterfactual[gene] = {
                'target': gene,
                'intervention': 'knockout',
                'baseline_disease': 1.0,
                'counterfactual_disease': 1.0 + effect,
                'absolute_change': effect,
                'relative_change': effect_pct / 100 if isinstance(effect_pct, (int, float)) and abs(effect_pct) > 1 else effect_pct,
                'n_affected_nodes': getattr(ko, 'affected_genes', 0) if hasattr(ko, 'affected_genes') else ko.get('affected_genes', 0) if isinstance(ko, dict) else 0,
                'crispr_source': True,
                'crispr_confidence': confidence,
            }

            # Also inject into DAG node for direct reading
            nd = dag.nodes[gene]
            nd['counterfactual_effect'] = effect
            nd['counterfactual_relative'] = effect_pct
            nd['counterfactual_confidence'] = confidence
            nd['counterfactual_source'] = 'crispr_knockout_engine'

            n_injected += 1

        logger.info(f"KnockoutEfficacyInjector: {n_injected} genes injected as counterfactuals")
        return dag, counterfactual
