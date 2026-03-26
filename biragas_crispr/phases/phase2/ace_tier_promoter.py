"""
ACETierPromoter — Promotes CRISPR-Validated Drivers to Higher Tiers
=====================================================================
The standard CentralityCalculator requires ALL 3 conditions for Tier 1:
    1. apex_score >= 0.60
    2. pleiotropic_reach >= 1
    3. causal_tier == 'Validated Driver'

Problem: Some genes with very strong ACE scores (ACE ≤ -0.3) miss Tier 1
because they have apex < 0.60 or pleiotropy < 1. These are experimentally
validated causal drivers that the network topology alone doesn't recognize.

Solution: This module promotes strong CRISPR drivers that meet relaxed criteria:
    - ACE ≤ -0.3 (strong experimental evidence) → relax apex to 0.40
    - ACE ≤ -0.1 + MR validated → relax pleiotropy to 0 (MR confirms causation)
    - Triple validated (GWAS + MR + CRISPR) → auto-promote to Tier 1
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase2.tier_promoter")


class ACETierPromoter:
    """
    Promotes CRISPR-validated genes to higher tiers based on experimental evidence.

    Standard Tier 1 requires apex ≥ 0.60 + pleiotropy ≥ 1 + Validated Driver.
    This module relaxes criteria when strong CRISPR evidence exists.

    Usage:
        promoter = ACETierPromoter()
        dag, promotion_report = promoter.promote(dag)
    """

    def __init__(self, strong_ace: float = -0.3, moderate_ace: float = -0.1,
                 relaxed_apex: float = 0.40):
        self.strong_ace = strong_ace
        self.moderate_ace = moderate_ace
        self.relaxed_apex = relaxed_apex

    def promote(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Review tier classifications and promote CRISPR-validated drivers.
        """
        report = {
            'promotions': [],
            'total_promoted': 0,
            'tier1_before': 0,
            'tier1_after': 0,
            'promotion_reasons': {},
        }

        # Count before
        for n in dag.nodes():
            if dag.nodes[n].get('network_tier') == 'Tier_1_Master_Regulator':
                report['tier1_before'] += 1

        for gene in dag.nodes():
            nd = dag.nodes[gene]
            if nd.get('layer') != 'regulatory':
                continue

            current_tier = nd.get('network_tier', 'Tier_2_Secondary_Driver')
            if current_tier == 'Tier_1_Master_Regulator':
                continue  # Already Tier 1

            ace = nd.get('perturbation_ace', 0)
            apex = nd.get('apex_score', 0)
            pleiotropy = nd.get('pleiotropic_reach', 0)
            gwas = nd.get('gwas_hit', False)
            mr = nd.get('mr_validated', False) or nd.get('causal_tier') == 'Validated Driver'
            ace_valid = isinstance(ace, (int, float))

            promoted = False
            reason = ""

            # Rule 1: Triple validated (GWAS + MR + CRISPR ACE ≤ -0.1)
            if ace_valid and ace <= self.moderate_ace and gwas and mr:
                promoted = True
                reason = f"Triple validated: GWAS + MR + CRISPR (ACE={ace:.3f})"

            # Rule 2: Strong ACE (≤ -0.3) with relaxed apex (≥ 0.40)
            elif ace_valid and ace <= self.strong_ace and apex >= self.relaxed_apex:
                promoted = True
                reason = f"Strong CRISPR driver: ACE={ace:.3f}, apex={apex:.3f} (relaxed from 0.60)"

            # Rule 3: Strong ACE + MR (skip pleiotropy requirement)
            elif ace_valid and ace <= self.strong_ace and mr:
                promoted = True
                reason = f"CRISPR + MR corroborated: ACE={ace:.3f}, MR validated"

            # Rule 4: Promote Tier 3 → Tier 2 if moderate ACE
            elif current_tier == 'Tier_3_Downstream_Effector' and ace_valid and ace <= self.moderate_ace:
                nd['network_tier'] = 'Tier_2_Secondary_Driver'
                report['promotions'].append({
                    'gene': gene, 'from': 'Tier_3', 'to': 'Tier_2',
                    'reason': f"CRISPR driver promotes from Tier 3: ACE={ace:.3f}",
                })
                report['total_promoted'] += 1
                continue

            if promoted:
                nd['network_tier'] = 'Tier_1_Master_Regulator'
                nd['tier_promotion_reason'] = reason
                report['promotions'].append({
                    'gene': gene, 'from': current_tier, 'to': 'Tier_1',
                    'reason': reason, 'ace': round(ace, 4) if ace_valid else 0,
                    'apex': round(apex, 4),
                })
                report['total_promoted'] += 1
                report['promotion_reasons'][reason.split(':')[0]] = report['promotion_reasons'].get(reason.split(':')[0], 0) + 1

        # Count after
        for n in dag.nodes():
            if dag.nodes[n].get('network_tier') == 'Tier_1_Master_Regulator':
                report['tier1_after'] += 1

        report['new_tier1'] = report['tier1_after'] - report['tier1_before']

        logger.info(
            f"ACE Tier Promoter: {report['total_promoted']} promoted | "
            f"Tier 1: {report['tier1_before']} → {report['tier1_after']} (+{report['new_tier1']})"
        )

        return dag, report
