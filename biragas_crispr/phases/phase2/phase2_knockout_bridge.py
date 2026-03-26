"""
Phase2KnockoutBridge — Feeds Phase 2 Metrics to CRISPR Knockout Agents
=========================================================================
Gap: The knockout agent doesn't directly read causal_importance or network_tier.
This bridge enriches the DAG with Phase 2 summary attributes that the
knockout agent CAN read, and provides a ranked gene list for focused prediction.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase2.bridge")


class Phase2KnockoutBridge:
    """
    Bridges Phase 2 outputs to CRISPR knockout/combination agents.

    1. Enriches DAG with knockout-readable Phase 2 attributes
    2. Provides prioritized gene list for focused knockout prediction
    3. Maps tier classification to knockout strategy recommendations
    """

    def bridge(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Prepare the DAG for CRISPR knockout agents.

        Adds knockout-readable attributes and generates priority lists.
        """
        report = {
            'genes_bridged': 0,
            'tier1_priority': [],
            'tier2_priority': [],
            'safe_drivers': [],
            'essential_warnings': [],
        }

        for gene in dag.nodes():
            nd = dag.nodes[gene]
            if nd.get('layer') != 'regulatory':
                continue

            report['genes_bridged'] += 1

            # Compute knockout priority score combining Phase 2 + CRISPR
            ci = nd.get('causal_importance', 0)
            eci = nd.get('crispr_enhanced_importance', ci)
            ace = nd.get('perturbation_ace', 0)
            tier = nd.get('network_tier', 'Tier_2_Secondary_Driver')
            ess = nd.get('essentiality_tag', 'Unknown')
            composite = nd.get('target_composite_score_7d', nd.get('target_composite_score', 0))

            # Knockout priority: higher = more important to test
            priority = 0.0
            if isinstance(eci, (int, float)):
                priority += eci * 0.3
            if isinstance(ace, (int, float)) and ace <= -0.1:
                priority += abs(ace) * 0.3
            if isinstance(composite, (int, float)):
                priority += composite * 0.4

            nd['knockout_priority'] = round(priority, 4)

            # Strategy recommendation
            if ess == 'Core Essential':
                nd['knockout_strategy'] = 'CAUTION: Core Essential — partial knockdown preferred over full KO'
                report['essential_warnings'].append(gene)
            elif tier == 'Tier_1_Master_Regulator':
                nd['knockout_strategy'] = 'HIGH PRIORITY: Tier 1 Master Regulator — full KO with combination backup'
                report['tier1_priority'].append((gene, round(priority, 4)))
            elif tier == 'Tier_2_Secondary_Driver':
                nd['knockout_strategy'] = 'STANDARD: Tier 2 Secondary Driver — full KO'
                report['tier2_priority'].append((gene, round(priority, 4)))
            else:
                nd['knockout_strategy'] = 'LOW PRIORITY: Tier 3 Effector — consider only in combinations'

            # Track safe high-priority drivers
            if ess != 'Core Essential' and isinstance(ace, (int, float)) and ace <= -0.1:
                report['safe_drivers'].append((gene, round(priority, 4)))

        # Sort priority lists
        report['tier1_priority'].sort(key=lambda x: -x[1])
        report['tier2_priority'].sort(key=lambda x: -x[1])
        report['safe_drivers'].sort(key=lambda x: -x[1])

        # Truncate for report
        report['tier1_priority'] = report['tier1_priority'][:20]
        report['tier2_priority'] = report['tier2_priority'][:20]
        report['safe_drivers'] = report['safe_drivers'][:20]
        report['essential_warnings'] = report['essential_warnings'][:20]

        logger.info(
            f"Phase2 KO Bridge: {report['genes_bridged']} genes | "
            f"Tier 1: {len(report['tier1_priority'])} | Safe drivers: {len(report['safe_drivers'])} | "
            f"Essential warnings: {len(report['essential_warnings'])}"
        )

        return dag, report

    def get_knockout_gene_list(self, dag: nx.DiGraph, top_n: int = 200) -> List[str]:
        """Get prioritized gene list for knockout prediction."""
        genes = []
        for n in dag.nodes():
            nd = dag.nodes[n]
            if nd.get('layer') == 'regulatory':
                priority = nd.get('knockout_priority', 0)
                genes.append((n, priority))
        genes.sort(key=lambda x: -x[1])
        return [g for g, _ in genes[:top_n]]
