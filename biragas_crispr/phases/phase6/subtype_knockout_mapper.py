"""
SubtypeKnockoutMapper — Maps Patient Subtypes to Knockout Targets
====================================================================
Gap: Phase 6 identifies subtypes but doesn't connect to knockout agents.
This module maps each patient subtype to its optimal CRISPR knockout targets,
enabling precision medicine: right knockout for right patient.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase6.mapper")


class SubtypeKnockoutMapper:
    """
    Maps molecular subtypes to subtype-specific knockout targets.
    """

    def __init__(self, ace_threshold: float = -0.1):
        self.ace_threshold = ace_threshold

    def map_subtypes(self, subgroup_profiles: Dict,
                     knockout_results: Dict = None) -> Dict:
        """
        For each subtype, identify the best knockout targets.

        Args:
            subgroup_profiles: From CRISPRWeightedStratifier or CohortStratifier
            knockout_results: From CRISPR knockout engine (gene → result)
        """
        report = {'subtype_targets': {}, 'pan_subtype_targets': [], 'precision_recommendations': []}

        all_subtype_drivers = {}

        for sg_id, profile in subgroup_profiles.items():
            if isinstance(profile, dict) and 'profile' in profile:
                prof = profile['profile']
            else:
                prof = profile

            crispr_drivers = prof.get('crispr_drivers', prof.get('dominant_genes', {}))

            # Get knockout scores for this subtype's drivers
            subtype_targets = []
            for gene, freq in crispr_drivers.items():
                target_info = {'gene': gene, 'frequency_in_subtype': freq}

                if knockout_results and gene in knockout_results:
                    ko = knockout_results[gene]
                    if hasattr(ko, 'ensemble'):
                        target_info['knockout_score'] = round(ko.ensemble, 4)
                        target_info['direction'] = ko.direction if hasattr(ko, 'direction') else 'unknown'
                    elif isinstance(ko, dict):
                        target_info['knockout_score'] = round(ko.get('ensemble', ko.get('ensemble_score', 0)), 4)
                        target_info['direction'] = ko.get('direction', 'unknown')

                target_info['priority'] = round(freq * target_info.get('knockout_score', 0.5), 4)
                subtype_targets.append(target_info)

            subtype_targets.sort(key=lambda t: -t['priority'])
            report['subtype_targets'][sg_id] = subtype_targets[:10]
            all_subtype_drivers[sg_id] = set(crispr_drivers.keys())

        # Find pan-subtype targets (present in ALL subtypes)
        if all_subtype_drivers:
            pan = set.intersection(*all_subtype_drivers.values()) if all_subtype_drivers.values() else set()
            report['pan_subtype_targets'] = sorted(pan)[:10]

        # Precision medicine recommendations
        for sg_id, targets in report['subtype_targets'].items():
            if targets:
                top = targets[0]
                unique_to_subtype = top['gene'] not in report['pan_subtype_targets']
                report['precision_recommendations'].append({
                    'subtype': sg_id,
                    'recommended_target': top['gene'],
                    'priority': top['priority'],
                    'subtype_specific': unique_to_subtype,
                    'strategy': f"{'Subtype-specific' if unique_to_subtype else 'Pan-subtype'} knockout of {top['gene']}",
                })

        logger.info(
            f"SubtypeKnockoutMapper: {len(report['subtype_targets'])} subtypes mapped | "
            f"{len(report['pan_subtype_targets'])} pan-subtype targets"
        )

        return report
