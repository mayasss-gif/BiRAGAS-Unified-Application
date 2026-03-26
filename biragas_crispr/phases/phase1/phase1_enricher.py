"""
Phase1CRISPREnricher — Enriches DAG Construction with CRISPR Evidence
========================================================================
Runs AFTER DAGBuilder builds the initial DAG, BEFORE MR validation.
Adds CRISPR-specific enhancements to the Phase 1 pipeline.

Enhancements:
1. Validates perturbation asymmetry constraint was applied correctly
2. Counts CRISPR-oriented edges and reports statistics
3. Cross-references ACE scores with RRA/MLE significance
4. Flags genes with ACE < -0.1 but no GWAS/MR support (CRISPR-only drivers)
5. Generates Phase 1 CRISPR integration report
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase1.enricher")


class Phase1CRISPREnricher:
    """
    Post-DAG-construction CRISPR enrichment for Phase 1.

    Usage:
        enricher = Phase1CRISPREnricher()
        dag, report = enricher.enrich(dag)  # After DAGBuilder.build_consensus_dag()
    """

    def enrich(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
        """
        Enrich the Phase 1 DAG with CRISPR quality metrics.

        Returns: (enriched_dag, enrichment_report)
        """
        report = {}

        # 1. Count CRISPR-annotated nodes
        crispr_nodes = [n for n in dag.nodes() if dag.nodes[n].get('perturbation_ace', 0) != 0]
        report['crispr_annotated_genes'] = len(crispr_nodes)

        # 2. Count perturbation-asymmetry-oriented edges
        pa_edges = [(u, v) for u, v, d in dag.edges(data=True)
                    if 'perturbation_asymmetry' in str(d.get('evidence_types', ''))]
        report['perturbation_asymmetry_edges'] = len(pa_edges)

        # 3. Classify genes by CRISPR evidence status
        crispr_only = []  # ACE driver but no GWAS/MR
        crispr_plus_gwas = []  # ACE driver + GWAS
        crispr_plus_mr = []  # ACE driver + MR
        crispr_triple = []  # ACE + GWAS + MR (strongest)

        for gene in crispr_nodes:
            nd = dag.nodes[gene]
            ace = nd.get('perturbation_ace', 0)
            if ace <= -0.1:
                has_gwas = nd.get('gwas_hit', False)
                has_mr = nd.get('mr_validated', False)
                if has_gwas and has_mr:
                    crispr_triple.append(gene)
                elif has_gwas:
                    crispr_plus_gwas.append(gene)
                elif has_mr:
                    crispr_plus_mr.append(gene)
                else:
                    crispr_only.append(gene)

        report['crispr_only_drivers'] = len(crispr_only)
        report['crispr_gwas_drivers'] = len(crispr_plus_gwas)
        report['crispr_mr_drivers'] = len(crispr_plus_mr)
        report['crispr_triple_drivers'] = len(crispr_triple)
        report['crispr_only_genes'] = crispr_only[:20]
        report['triple_validated_genes'] = crispr_triple[:20]

        # 4. Flag CRISPR-only drivers for MR validation priority
        for gene in crispr_only:
            dag.nodes[gene]['crispr_only_driver'] = True
            dag.nodes[gene]['mr_validation_priority'] = 'HIGH'

        # 5. Compute ACE statistics
        ace_scores = [dag.nodes[n].get('perturbation_ace', 0) for n in crispr_nodes if dag.nodes[n].get('perturbation_ace', 0) != 0]
        if ace_scores:
            report['ace_mean'] = round(float(np.mean(ace_scores)), 4)
            report['ace_median'] = round(float(np.median(ace_scores)), 4)
            report['ace_min'] = round(float(min(ace_scores)), 4)
            report['ace_max'] = round(float(max(ace_scores)), 4)
            report['ace_std'] = round(float(np.std(ace_scores)), 4)
            report['n_drivers'] = sum(1 for a in ace_scores if a <= -0.1)
            report['n_strong_drivers'] = sum(1 for a in ace_scores if a <= -0.3)
        else:
            report['ace_mean'] = report['ace_median'] = report['ace_min'] = report['ace_max'] = 0

        # 6. Validate perturbation asymmetry was correctly applied
        correct_pa = 0
        incorrect_pa = 0
        for u, v, d in dag.edges(data=True):
            ace_u = dag.nodes[u].get('perturbation_ace', 0)
            ace_v = dag.nodes[v].get('perturbation_ace', 0)
            if ace_u <= -0.1 and ace_v > -0.1 and abs(ace_u - ace_v) >= 0.2:
                correct_pa += 1
            elif ace_v <= -0.1 and ace_u > -0.1 and abs(ace_u - ace_v) >= 0.2:
                # This edge should have been flipped
                incorrect_pa += 1
                dag.nodes[u]['pa_needs_review'] = True

        report['correct_pa_edges'] = correct_pa
        report['incorrect_pa_edges'] = incorrect_pa
        report['pa_accuracy'] = round(correct_pa / max(correct_pa + incorrect_pa, 1), 4)

        # 7. Essentiality distribution
        ess_dist = {}
        for n in dag.nodes():
            ess = dag.nodes[n].get('essentiality_tag', 'Unknown')
            ess_dist[ess] = ess_dist.get(ess, 0) + 1
        report['essentiality_distribution'] = ess_dist

        # 8. Therapeutic alignment distribution
        align_dist = {}
        for n in dag.nodes():
            al = dag.nodes[n].get('therapeutic_alignment', 'Unknown')
            align_dist[al] = align_dist.get(al, 0) + 1
        report['alignment_distribution'] = align_dist

        logger.info(
            f"Phase 1 CRISPR enrichment: {len(crispr_nodes)} annotated, "
            f"{len(crispr_triple)} triple-validated, {len(crispr_only)} CRISPR-only, "
            f"PA accuracy: {report['pa_accuracy']}"
        )

        return dag, report
