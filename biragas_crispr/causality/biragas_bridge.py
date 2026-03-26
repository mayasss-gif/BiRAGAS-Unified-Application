"""
BiRAGAS Bridge — Connects CRISPR Engines to 7-Phase Causality Framework
==========================================================================
Maps CRISPR screening/knockout/combination results into the BiRAGAS DAG
and enriches each phase with CRISPR evidence.

Phase Integration:
    Phase 1: Screening → DAG node attributes (ACE, essentiality)
    Phase 2: ACE scores → centrality boost + tier promotion
    Phase 3: Knockout predictions → hallucination shield validation
    Phase 4: Attribute harmonization (confidence/confidence_score)
    Phase 5: Combination predictions → synergy + efficacy injection
    Phase 6: Subtype-specific knockout mapping
    Phase 7: Quality boost + gap prioritization + report generation
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biragas_crispr.causality.bridge")


class BiRAGASBridge:
    """
    Bidirectional bridge between CRISPR engines and BiRAGAS causality framework.
    """

    def __init__(self):
        self._phase_results = {}
        logger.info("BiRAGAS Bridge initialized")

    def enrich_dag_phase1(self, dag, screening_results: Dict) -> Dict:
        """Phase 1: Inject screening data into DAG nodes."""
        enriched = 0
        for gene, data in screening_results.items():
            if gene in dag:
                nd = dag.nodes[gene]
                if hasattr(data, 'ace_score'):
                    nd['perturbation_ace'] = data.ace_score
                    nd['essentiality_tag'] = data.essentiality_class
                    nd['therapeutic_alignment'] = data.therapeutic_alignment
                    nd['rra_pos_p'] = data.rra_pos_p
                    nd['mle_beta'] = data.mle_beta
                    nd['bagel2_bf'] = data.bagel2_bf
                    enriched += 1
                elif isinstance(data, dict):
                    for k, v in data.items():
                        nd[k] = v
                    enriched += 1

        report = {'phase': 1, 'enriched_nodes': enriched}
        self._phase_results['phase1'] = report
        logger.info(f"Phase 1: Enriched {enriched} DAG nodes with screening data")
        return report

    def enrich_dag_phase2(self, dag, ace_results: Dict) -> Dict:
        """Phase 2: ACE-based centrality boost and tier promotion."""
        promoted = 0
        for gene, result in ace_results.items():
            if gene in dag:
                nd = dag.nodes[gene]
                ace = result.ace_score if hasattr(result, 'ace_score') else result.get('ace_score', 0)

                # Tier promotion based on ACE
                if abs(ace) > 0.5:
                    nd['network_tier'] = 'Tier 1'
                    promoted += 1
                elif abs(ace) > 0.2:
                    nd['network_tier'] = 'Tier 2'
                elif 'network_tier' not in nd:
                    nd['network_tier'] = 'Tier 3'

                # Update edge weights for high-ACE genes
                for succ in dag.successors(gene):
                    w = dag[gene][succ].get('weight', 0.5)
                    ace_boost = min(0.2, abs(ace) * 0.3)
                    dag[gene][succ]['weight'] = min(0.95, w + ace_boost)

        report = {'phase': 2, 'tier_promotions': promoted}
        self._phase_results['phase2'] = report
        logger.info(f"Phase 2: {promoted} tier promotions from ACE scoring")
        return report

    def enrich_dag_phase3(self, dag, knockout_results: Dict) -> Dict:
        """Phase 3: Knockout-validated hallucination shielding."""
        validated = 0
        flagged = 0

        for gene, ko_result in knockout_results.items():
            if gene not in dag:
                continue

            nd = dag.nodes[gene]
            ensemble = ko_result.ensemble_score if hasattr(ko_result, 'ensemble_score') else \
                       ko_result.get('ensemble_score', ko_result.get('ensemble', 0))

            confidence = ko_result.confidence if hasattr(ko_result, 'confidence') else \
                         ko_result.get('confidence', 0)

            # Validate: high knockout effect confirms causal role
            if abs(ensemble) > 0.3 and confidence > 0.5:
                nd['knockout_validated'] = True
                validated += 1
                # Boost edge confidence
                for succ in dag.successors(gene):
                    conf = dag[gene][succ].get('confidence', 0.5)
                    dag[gene][succ]['confidence'] = min(0.95, conf + 0.1)
                    dag[gene][succ]['confidence_score'] = dag[gene][succ]['confidence']
            elif abs(ensemble) < 0.05:
                nd['knockout_validated'] = False
                nd['hallucination_risk'] = True
                flagged += 1

        report = {'phase': 3, 'validated': validated, 'flagged_hallucination': flagged}
        self._phase_results['phase3'] = report
        logger.info(f"Phase 3: {validated} validated, {flagged} flagged as potential hallucinations")
        return report

    def harmonize_phase4(self, dag) -> Dict:
        """Phase 4: Harmonize confidence/confidence_score on all edges."""
        harmonized = 0
        for u, v, data in dag.edges(data=True):
            has_conf = 'confidence' in data
            has_score = 'confidence_score' in data

            if has_conf and not has_score:
                data['confidence_score'] = data['confidence']
                harmonized += 1
            elif has_score and not has_conf:
                data['confidence'] = data['confidence_score']
                harmonized += 1
            elif not has_conf and not has_score:
                data['confidence'] = data.get('weight', 0.5)
                data['confidence_score'] = data['confidence']
                harmonized += 1

        report = {'phase': 4, 'edges_harmonized': harmonized}
        self._phase_results['phase4'] = report
        return report

    def enrich_dag_phase5(self, dag, combination_results: List) -> Dict:
        """Phase 5: Inject combination synergy predictions."""
        synergistic_pairs = 0
        for combo in combination_results:
            gene_a = combo.gene_a if hasattr(combo, 'gene_a') else combo.get('gene_a', '')
            gene_b = combo.gene_b if hasattr(combo, 'gene_b') else combo.get('gene_b', '')
            synergy = combo.synergy_score if hasattr(combo, 'synergy_score') else combo.get('synergy', 0)

            if gene_a in dag and gene_b in dag and synergy > 0.05:
                synergistic_pairs += 1
                # Mark synergistic pair on nodes
                for gene in [gene_a, gene_b]:
                    synergies = dag.nodes[gene].get('synergistic_partners', [])
                    partner = gene_b if gene == gene_a else gene_a
                    if partner not in synergies:
                        synergies.append(partner)
                    dag.nodes[gene]['synergistic_partners'] = synergies

        report = {'phase': 5, 'synergistic_pairs': synergistic_pairs}
        self._phase_results['phase5'] = report
        return report

    def enrich_dag_phase6(self, dag, subtype_mapping: Optional[Dict] = None) -> Dict:
        """Phase 6: Subtype-specific knockout mapping."""
        mapped = 0
        if subtype_mapping:
            for gene, subtypes in subtype_mapping.items():
                if gene in dag:
                    dag.nodes[gene]['subtypes'] = subtypes
                    mapped += 1

        report = {'phase': 6, 'subtype_mapped': mapped}
        self._phase_results['phase6'] = report
        return report

    def generate_phase7_report(self, dag, all_results: Dict) -> Dict:
        """Phase 7: Generate comprehensive clinical report."""
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
                    'knockout_validated': nd.get('knockout_validated', False),
                })

        crispr_genes.sort(key=lambda g: g['ace'])

        n_total = len(crispr_genes)
        n_drivers = sum(1 for g in crispr_genes if g['ace'] <= -0.1)
        n_strong = sum(1 for g in crispr_genes if g['ace'] <= -0.3)
        n_essential = sum(1 for g in crispr_genes if g['essentiality'] == 'Core Essential')
        n_validated = sum(1 for g in crispr_genes if g['knockout_validated'])

        report = {
            'phase': 7,
            'summary': {
                'total_genes': n_total,
                'drivers': n_drivers,
                'strong_drivers': n_strong,
                'core_essential': n_essential,
                'knockout_validated': n_validated,
            },
            'top_targets': crispr_genes[:20],
            'phase_results': self._phase_results,
            'clinical_narrative': (
                f"CRISPR screening identified {n_total} genes with perturbation effects. "
                f"{n_drivers} are causal drivers (ACE <= -0.1), {n_strong} are strong drivers (ACE <= -0.3). "
                f"{n_essential} are Core Essential. {n_validated} genes are knockout-validated. "
                f"{'Top non-essential drivers are recommended for therapeutic targeting.' if n_drivers > n_essential else 'Caution: most drivers are essential.'}"
            ),
        }

        self._phase_results['phase7'] = report
        logger.info(f"Phase 7 report: {n_total} genes, {n_drivers} drivers, {n_validated} validated")
        return report

    def get_all_phase_results(self) -> Dict:
        return dict(self._phase_results)
