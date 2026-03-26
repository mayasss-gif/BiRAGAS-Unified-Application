"""
CRISPRQualityBooster — Quantifies CRISPR Impact on Evidence Quality
======================================================================
The Bayesian quality formula 1-Π(1-w_i) means CRISPR (w=0.85) has a
massive impact on edge quality:
    Without CRISPR (statistical only): quality = 0.35
    With CRISPR alone: quality = 0.85
    With CRISPR + GWAS: quality = 0.985

This module computes and reports the quality differential with/without CRISPR.
"""

import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase7.quality")

EVIDENCE_WEIGHTS = {
    'gwas': 0.90, 'mendelian_randomization': 0.95, 'crispr': 0.85,
    'signor': 0.90, 'temporal': 0.65, 'statistical': 0.35,
    'database': 0.80, 'eqtl': 0.85, 'pathway_enrichment': 0.70,
    'coexpression': 0.50, 'deconvolution': 0.60,
}


class CRISPRQualityBooster:
    """
    Quantifies how CRISPR evidence affects edge quality scores.
    """

    def analyze(self, dag: nx.DiGraph) -> Dict:
        """
        Compute quality with/without CRISPR for every edge.
        """
        report = {
            'edges_analyzed': 0,
            'edges_with_crispr': 0,
            'edges_without_crispr': 0,
            'mean_quality_with': 0.0,
            'mean_quality_without': 0.0,
            'quality_boost_mean': 0.0,
            'max_boost': 0.0,
            'quality_saved': 0,  # Edges that pass 0.60 threshold ONLY because of CRISPR
            'distribution': {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0},
            'top_boosted_edges': [],
        }

        qualities_with = []
        qualities_without = []
        boosts = []

        for u, v, d in dag.edges(data=True):
            report['edges_analyzed'] += 1

            # Extract evidence types
            evidence_str = str(d.get('evidence_types', d.get('evidence', '')))
            evidence_types = set()
            for token in evidence_str.replace(',', ' ').replace(';', ' ').replace('_', ' ').split():
                token_lower = token.lower().strip()
                if token_lower in EVIDENCE_WEIGHTS:
                    evidence_types.add(token_lower)

            # Check CRISPR on endpoints
            u_ace = dag.nodes.get(u, {}).get('perturbation_ace', 0)
            v_ace = dag.nodes.get(v, {}).get('perturbation_ace', 0)
            u_has = isinstance(u_ace, (int, float)) and u_ace <= -0.1
            v_has = isinstance(v_ace, (int, float)) and v_ace <= -0.1
            if u_has or v_has:
                evidence_types.add('crispr')

            has_crispr = 'crispr' in evidence_types

            # Quality WITH all evidence
            weights_with = [EVIDENCE_WEIGHTS[e] for e in evidence_types if e in EVIDENCE_WEIGHTS]
            if weights_with:
                q_with = 1.0 - float(np.prod([1.0 - w for w in weights_with]))
            else:
                q_with = float(d.get('confidence', d.get('confidence_score', 0.1)))

            # Quality WITHOUT CRISPR
            non_crispr = evidence_types - {'crispr'}
            weights_without = [EVIDENCE_WEIGHTS[e] for e in non_crispr if e in EVIDENCE_WEIGHTS]
            if weights_without:
                q_without = 1.0 - float(np.prod([1.0 - w for w in weights_without]))
            else:
                q_without = float(d.get('confidence', d.get('confidence_score', 0.1)))

            qualities_with.append(q_with)
            qualities_without.append(q_without)
            boost = q_with - q_without
            boosts.append(boost)

            if has_crispr:
                report['edges_with_crispr'] += 1
            else:
                report['edges_without_crispr'] += 1

            # Track edges saved by CRISPR (pass threshold only with CRISPR)
            if q_with >= 0.60 and q_without < 0.60:
                report['quality_saved'] += 1

            # Quality distribution
            if q_with >= 0.85:
                report['distribution']['very_high'] += 1
            elif q_with >= 0.65:
                report['distribution']['high'] += 1
            elif q_with >= 0.40:
                report['distribution']['medium'] += 1
            else:
                report['distribution']['low'] += 1

            if boost > 0.1:
                report['top_boosted_edges'].append({
                    'edge': f"{u}→{v}",
                    'quality_with': round(q_with, 4),
                    'quality_without': round(q_without, 4),
                    'boost': round(boost, 4),
                    'evidence': list(evidence_types),
                })

            # Write to DAG
            d['quality_with_crispr'] = round(q_with, 4)
            d['quality_without_crispr'] = round(q_without, 4)
            d['crispr_quality_boost'] = round(boost, 4)

        if qualities_with:
            report['mean_quality_with'] = round(float(np.mean(qualities_with)), 4)
            report['mean_quality_without'] = round(float(np.mean(qualities_without)), 4)
            report['quality_boost_mean'] = round(float(np.mean(boosts)), 4)
            report['max_boost'] = round(float(max(boosts)), 4)

        report['top_boosted_edges'].sort(key=lambda x: -x['boost'])
        report['top_boosted_edges'] = report['top_boosted_edges'][:15]

        logger.info(
            f"CRISPRQualityBooster: {report['edges_analyzed']} edges | "
            f"with CRISPR: {report['edges_with_crispr']} | mean boost: {report['quality_boost_mean']:.3f} | "
            f"saved: {report['quality_saved']} edges"
        )

        return report
