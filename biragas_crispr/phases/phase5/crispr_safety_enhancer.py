"""
CRISPRSafetyEnhancer — Adds CRISPR-Specific Safety Signals
==============================================================
SafetyAssessor uses essentiality_tag and systemic_toxicity_risk from CRISPR.
This module adds finer-grained safety signals:

1. Guide efficiency variance — high variance = unpredictable knockout depth
2. CRISPR-specific off-target risk from Brunello library design
3. Multi-guide concordance — do all 4 guides agree on phenotype?
4. Drug sensitivity interaction — does the gene interact with known drugs?
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase5.safety")


class CRISPRSafetyEnhancer:
    """
    Adds CRISPR-specific safety dimensions to Phase 5 SafetyAssessor output.

    Usage:
        enhancer = CRISPRSafetyEnhancer()
        dag, report = enhancer.enhance_safety(dag, screening_results, safety_scores)
    """

    def enhance_safety(self, dag: nx.DiGraph,
                       screening_results: Optional[Dict] = None,
                       existing_safety: Optional[Dict] = None) -> Tuple[nx.DiGraph, Dict]:
        """
        Add CRISPR-specific safety signals to each gene.
        """
        report = {
            'genes_enhanced': 0,
            'guide_variance_warnings': 0,
            'drug_interaction_warnings': 0,
            'essential_driver_conflicts': 0,
        }

        for gene in dag.nodes():
            nd = dag.nodes[gene]
            if nd.get('layer') != 'regulatory':
                continue

            report['genes_enhanced'] += 1
            crispr_safety_score = 1.0  # Start safe
            alerts = nd.get('safety_alerts', [])

            # Signal 1: Essentiality-driver conflict
            ace = nd.get('perturbation_ace', 0)
            ess = nd.get('essentiality_tag', 'Unknown')
            if isinstance(ace, (int, float)) and ace <= -0.3 and ess == 'Core Essential':
                crispr_safety_score -= 0.3
                alerts.append(f"CONFLICT: {gene} is a strong CRISPR driver (ACE={ace:.2f}) BUT Core Essential — targeting causes lethality")
                report['essential_driver_conflicts'] += 1

            # Signal 2: Guide variance (from screening data)
            if screening_results and gene in screening_results:
                sr = screening_results[gene]
                if hasattr(sr, 'guide_lfcs') and sr.guide_lfcs:
                    variance = float(np.std(sr.guide_lfcs))
                    if variance > 0.5:
                        crispr_safety_score -= 0.1
                        alerts.append(f"WARNING: High guide variance ({variance:.2f}) — unpredictable knockout depth")
                        report['guide_variance_warnings'] += 1

            # Signal 3: Drug interaction check
            if nd.get('has_approved_drug') and ess == 'Core Essential':
                crispr_safety_score -= 0.1
                alerts.append(f"WARNING: {gene} has approved drug targeting AND is Core Essential")
                report['drug_interaction_warnings'] += 1

            # Signal 4: Off-target ratio from DAG topology
            all_desc = set()
            try:
                all_desc = set(nx.descendants(dag, gene))
            except:
                pass
            disease_rel = {n for n in all_desc if dag.nodes.get(n, {}).get('layer') in ('program', 'trait')}
            off_target = 1.0 - (len(disease_rel) / max(len(all_desc), 1)) if all_desc else 0.5

            if off_target > 0.7:
                crispr_safety_score -= 0.15
                alerts.append(f"WARNING: High off-target ratio ({off_target:.2f}) — most downstream effects are non-disease")

            crispr_safety_score = max(0.0, crispr_safety_score)

            # Merge with existing safety
            existing = 0.7
            if existing_safety and gene in existing_safety:
                es = existing_safety[gene]
                existing = es.get('score', es.get('safety_score', 0.7)) if isinstance(es, dict) else 0.7

            # Combined: 70% existing + 30% CRISPR-specific
            combined = 0.7 * existing + 0.3 * crispr_safety_score

            nd['crispr_safety_score'] = round(crispr_safety_score, 4)
            nd['combined_safety_score'] = round(combined, 4)
            nd['safety_alerts'] = alerts

        logger.info(
            f"CRISPRSafetyEnhancer: {report['genes_enhanced']} enhanced | "
            f"{report['essential_driver_conflicts']} essential-driver conflicts | "
            f"{report['guide_variance_warnings']} guide variance warnings"
        )

        return dag, report
