"""
CombinationSynergyUpgrader — Upgrades Basic Synergy with 6-Model CRISPR
==========================================================================
Phase 5 CombinationAnalyzer uses 1 synergy model (pathway diversity).
CRISPR CombinationAgent uses 6 models (Bliss, HSA, Loewe, ZIP, Epistasis, Compensation).

This module bridges them:
1. Runs Phase 5 CombinationAnalyzer for topology-based scoring
2. Runs CRISPR CombinationAgent for effect-based synergy
3. Merges both into unified combination ranking
4. Adds compensation-blocking pairs from Phase 4 CompensationBridge
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase5.combination")


class CombinationSynergyUpgrader:
    """
    Merges Phase 5 topology-based combinations with CRISPR effect-based synergy.

    Usage:
        upgrader = CombinationSynergyUpgrader()
        results = upgrader.upgrade(dag, knockout_results, phase5_combos, crispr_combos)
    """

    def upgrade(self, dag: nx.DiGraph,
                knockout_results: Dict,
                phase5_combinations: Optional[Dict] = None,
                crispr_combinations: Optional[List] = None,
                compensation_pairs: Optional[List] = None) -> Dict:
        """
        Merge Phase 5 + CRISPR combination results.
        """
        unified = []
        seen = set()

        # Source 1: CRISPR CombinationAgent results (6 synergy models)
        if crispr_combinations:
            for combo in crispr_combinations:
                genes = combo.genes if hasattr(combo, 'genes') else combo.get('genes', [])
                key = tuple(sorted(genes))
                if key not in seen:
                    seen.add(key)
                    unified.append({
                        'genes': genes,
                        'source': 'crispr_6model',
                        'crispr_composite': getattr(combo, 'composite', 0) if hasattr(combo, 'composite') else combo.get('composite', combo.get('composite_score', 0)),
                        'crispr_synergy': getattr(combo, 'synergy_score', 0) if hasattr(combo, 'synergy_score') else combo.get('synergy_score', combo.get('synergy', 0)),
                        'crispr_interaction': getattr(combo, 'interaction', '') if hasattr(combo, 'interaction') else combo.get('interaction', ''),
                        'pathway_coverage': getattr(combo, 'pathway_coverage', 0) if hasattr(combo, 'pathway_coverage') else combo.get('pathway_coverage', 0),
                        'combined_safety': getattr(combo, 'combined_safety', 0) if hasattr(combo, 'combined_safety') else combo.get('combined_safety', 0),
                        'phase5_composite': 0,
                    })

        # Source 2: Phase 5 CombinationAnalyzer results (topology-based)
        if phase5_combinations:
            combos_list = phase5_combinations.get('combinations', phase5_combinations.get('ranked_combinations', []))
            for combo in combos_list:
                genes = combo.get('genes', combo.get('targets', []))
                key = tuple(sorted(genes))
                if key not in seen:
                    seen.add(key)
                    unified.append({
                        'genes': genes,
                        'source': 'phase5_topology',
                        'crispr_composite': 0,
                        'crispr_synergy': 0,
                        'crispr_interaction': '',
                        'pathway_coverage': combo.get('coverage', combo.get('pathway_coverage', 0)),
                        'combined_safety': combo.get('combined_safety', 0),
                        'phase5_composite': combo.get('composite_score', combo.get('score', 0)),
                    })
                else:
                    # Merge with existing CRISPR entry
                    for u in unified:
                        if tuple(sorted(u['genes'])) == key:
                            u['phase5_composite'] = combo.get('composite_score', combo.get('score', 0))
                            u['source'] = 'both'
                            break

        # Source 3: Compensation-blocking pairs from Phase 4
        if compensation_pairs:
            for pair in compensation_pairs:
                if isinstance(pair, tuple) and len(pair) >= 2:
                    genes = [pair[0], pair[1]]
                elif isinstance(pair, dict):
                    genes = [pair.get('target', ''), pair.get('co_target', '')]
                else:
                    continue

                key = tuple(sorted(genes))
                if key not in seen and all(g in dag.nodes() for g in genes):
                    seen.add(key)
                    unified.append({
                        'genes': genes,
                        'source': 'compensation_blocking',
                        'crispr_composite': 0.15,  # Compensation blocking bonus
                        'crispr_synergy': 0.15,
                        'crispr_interaction': 'synergistic (compensation blocking)',
                        'pathway_coverage': 0,
                        'combined_safety': 0,
                        'phase5_composite': 0,
                    })

        # Compute unified score
        for entry in unified:
            c_score = entry.get('crispr_composite', 0)
            p_score = entry.get('phase5_composite', 0)

            if not isinstance(c_score, (int, float)):
                c_score = 0
            if not isinstance(p_score, (int, float)):
                p_score = 0

            if c_score > 0 and p_score > 0:
                entry['unified_score'] = 0.6 * c_score + 0.4 * p_score + 0.1  # Cross-validation bonus
            elif c_score > 0:
                entry['unified_score'] = c_score
            elif p_score > 0:
                entry['unified_score'] = p_score
            else:
                entry['unified_score'] = 0.05

        unified.sort(key=lambda x: -x['unified_score'])
        for i, u in enumerate(unified):
            u['rank'] = i + 1

        n_both = sum(1 for u in unified if u['source'] == 'both')
        n_crispr = sum(1 for u in unified if u['source'] == 'crispr_6model')
        n_phase5 = sum(1 for u in unified if u['source'] == 'phase5_topology')
        n_comp = sum(1 for u in unified if u['source'] == 'compensation_blocking')

        logger.info(
            f"CombinationSynergyUpgrader: {len(unified)} combos | "
            f"both: {n_both} | crispr: {n_crispr} | phase5: {n_phase5} | compensation: {n_comp}"
        )

        return {
            'total_combinations': len(unified),
            'sources': {'both': n_both, 'crispr_6model': n_crispr, 'phase5_topology': n_phase5, 'compensation_blocking': n_comp},
            'ranked_combinations': unified,
        }
