"""
MRCRISPRCorroborator — Cross-validates MR and CRISPR Evidence
================================================================
Fixes Gap 6: MR and CRISPR currently operate independently.
This module cross-references their findings to:
1. Boost confidence when both agree on causal direction
2. Flag conflicts when MR and CRISPR disagree
3. Prioritize CRISPR-only drivers for MR validation
4. Generate corroboration report
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase1.corroborator")


class MRCRISPRCorroborator:
    """
    Cross-validates Mendelian Randomization and CRISPR evidence.

    MR validates gene→disease causation using genetic instruments.
    CRISPR validates gene function via direct perturbation.
    When both agree: causal claim is very strong.
    When they conflict: needs investigation.

    Usage:
        corr = MRCRISPRCorroborator()
        dag, report = corr.corroborate(dag)
    """

    def __init__(self, agreement_bonus: float = 0.15, conflict_penalty: float = 0.1):
        self.agreement_bonus = agreement_bonus
        self.conflict_penalty = conflict_penalty

    def corroborate(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Cross-validate MR and CRISPR evidence on the DAG.
        """
        report = {
            'genes_with_both': 0,
            'agreements': 0,
            'conflicts': 0,
            'crispr_only': 0,
            'mr_only': 0,
            'neither': 0,
            'agreement_genes': [],
            'conflict_genes': [],
            'crispr_only_genes': [],
            'mr_only_genes': [],
        }

        for gene in dag.nodes():
            if dag.nodes[gene].get('layer') != 'regulatory':
                continue

            nd = dag.nodes[gene]
            has_crispr = nd.get('perturbation_ace', 0) <= -0.1
            has_mr = nd.get('mr_validated', False) or nd.get('causal_tier') == 'Validated Driver'

            if has_crispr and has_mr:
                # Both agree: STRONG evidence
                report['genes_with_both'] += 1
                report['agreements'] += 1
                report['agreement_genes'].append(gene)

                # Boost confidence on all edges from this gene
                for _, v, d in dag.edges(gene, data=True):
                    old_conf = d.get('confidence', 0.5)
                    if isinstance(old_conf, (int, float)):
                        new_conf = min(0.99, old_conf + self.agreement_bonus)
                        d['confidence'] = new_conf
                        d['mr_crispr_corroborated'] = True

                nd['evidence_strength'] = 'Triple Validated (GWAS+MR+CRISPR)'

            elif has_crispr and not has_mr:
                # CRISPR only: flag for MR validation
                report['crispr_only'] += 1
                report['crispr_only_genes'].append(gene)
                nd['mr_validation_priority'] = 'HIGH'
                nd['evidence_strength'] = 'CRISPR Only — MR validation recommended'

            elif has_mr and not has_crispr:
                # MR only: flag for CRISPR validation
                report['mr_only'] += 1
                report['mr_only_genes'].append(gene)
                nd['crispr_validation_priority'] = 'HIGH'
                nd['evidence_strength'] = 'MR Only — CRISPR validation recommended'

            else:
                report['neither'] += 1

        # Check for directional conflicts
        for gene in report['agreement_genes']:
            nd = dag.nodes[gene]
            ace = nd.get('perturbation_ace', 0)
            mr_beta = nd.get('mr_beta', 0)

            # Conflict: CRISPR says driver (negative ACE) but MR says protective (negative beta)
            if ace < 0 and isinstance(mr_beta, (int, float)) and mr_beta < 0:
                # Both negative = both agree gene drives disease
                pass
            elif ace < 0 and isinstance(mr_beta, (int, float)) and mr_beta > 0:
                # Conflict: CRISPR says driver, MR says protective
                report['conflicts'] += 1
                report['conflict_genes'].append(gene)
                nd['mr_crispr_conflict'] = True
                nd['evidence_strength'] = 'CONFLICT: CRISPR↔MR disagree'

                # Downweight edges from conflicting gene
                for _, v, d in dag.edges(gene, data=True):
                    old_conf = d.get('confidence', 0.5)
                    if isinstance(old_conf, (int, float)):
                        d['confidence'] = max(0.1, old_conf - self.conflict_penalty)
                        d['mr_crispr_conflict'] = True

        # Truncate gene lists for report
        for key in ['agreement_genes', 'conflict_genes', 'crispr_only_genes', 'mr_only_genes']:
            report[key] = report[key][:20]

        logger.info(
            f"MR-CRISPR corroboration: {report['agreements']} agreements, "
            f"{report['conflicts']} conflicts, {report['crispr_only']} CRISPR-only, "
            f"{report['mr_only']} MR-only"
        )

        return dag, report
