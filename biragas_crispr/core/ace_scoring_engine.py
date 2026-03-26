"""
ACEScoringEngine v2.0 — 15-Stream Superior ACE Scoring
=========================================================
Aggregated CRISPR Evidence (ACE) score fusing 15 independent data streams
into a single composite causal evidence score per gene.

Streams:
    1.  MAGeCK RRA p-value
    2.  MAGeCK MLE beta coefficient
    3.  BAGEL2 Bayes Factor
    4.  Perturb-seq effect size
    5.  GWAS hit enrichment
    6.  Mendelian Randomization beta
    7.  eQTL association strength
    8.  SIGNOR database weight
    9.  Network centrality
    10. DAG tier classification
    11. Drug sensitivity (DrugZ)
    12. Conservation score
    13. FluteMLE pathway enrichment
    14. Editing efficiency
    15. Drug Z-score

Formula: ACE = Σ(wᵢ × sᵢ) / Σ(wᵢ) with Bayesian confidence updating
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("biragas_crispr.core.ace")


@dataclass
class ACEResult:
    """ACE score result for a single gene."""
    gene: str = ""
    ace_score: float = 0.0
    confidence: float = 0.0
    n_streams: int = 0
    stream_scores: Dict[str, float] = field(default_factory=dict)
    stream_weights: Dict[str, float] = field(default_factory=dict)
    essentiality_class: str = "Unknown"
    therapeutic_alignment: str = "Unknown"
    direction: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            'gene': self.gene,
            'ace': round(self.ace_score, 4),
            'confidence': round(self.confidence, 3),
            'n_streams': self.n_streams,
            'streams': {k: round(v, 4) for k, v in self.stream_scores.items()},
            'essentiality': self.essentiality_class,
            'alignment': self.therapeutic_alignment,
            'direction': self.direction,
        }


# Evidence stream weights (calibrated on DepMap + Project Score)
STREAM_WEIGHTS = {
    'mageck_rra': 0.90,
    'mageck_mle': 0.85,
    'bagel2_bf': 0.85,
    'perturbseq': 0.80,
    'gwas': 0.90,
    'mr_beta': 0.95,
    'eqtl': 0.85,
    'signor': 0.90,
    'centrality': 0.70,
    'dag_tier': 0.75,
    'drug_sensitivity': 0.80,
    'conservation': 0.65,
    'flute_pathway': 0.70,
    'editing_efficiency': 0.60,
    'drug_zscore': 0.75,
}


class ACEScoringEngine:
    """
    15-stream Superior ACE scoring engine.
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._weights = dict(STREAM_WEIGHTS)
        self._results_cache = {}
        logger.info("ACEScoringEngine v2.0 initialized (15 streams)")

    def score_gene(self, gene: str, dag=None,
                   screening_data: Optional[Dict] = None,
                   genomic_data: Optional[Dict] = None) -> ACEResult:
        """Compute 15-stream ACE score for a single gene."""
        streams = {}
        active_weights = {}

        # Get node data from DAG
        nd = dag.nodes[gene] if dag and gene in dag else {}

        # Stream 1: MAGeCK RRA
        rra_p = self._get_value(nd, screening_data, gene, 'rra_pos_p', 'mageck_rra_p')
        if rra_p is not None and rra_p > 0:
            streams['mageck_rra'] = -math.log10(max(rra_p, 1e-300))
            active_weights['mageck_rra'] = self._weights['mageck_rra']

        # Stream 2: MAGeCK MLE beta
        mle_beta = self._get_value(nd, screening_data, gene, 'mle_beta', 'mageck_mle_beta')
        if mle_beta is not None:
            streams['mageck_mle'] = float(mle_beta)
            active_weights['mageck_mle'] = self._weights['mageck_mle']

        # Stream 3: BAGEL2 Bayes Factor
        bf = self._get_value(nd, screening_data, gene, 'bagel2_bf', 'bayes_factor')
        if bf is not None:
            streams['bagel2_bf'] = float(bf)
            active_weights['bagel2_bf'] = self._weights['bagel2_bf']

        # Stream 4: Perturb-seq effect
        ps_effect = self._get_value(nd, screening_data, gene, 'perturbseq_effect', 'ps_effect')
        if ps_effect is not None:
            streams['perturbseq'] = float(ps_effect)
            active_weights['perturbseq'] = self._weights['perturbseq']

        # Stream 5: GWAS
        gwas_p = self._get_value(nd, genomic_data, gene, 'gwas_p', 'gwas_pvalue')
        if gwas_p is not None and gwas_p > 0:
            streams['gwas'] = -math.log10(max(gwas_p, 1e-300))
            active_weights['gwas'] = self._weights['gwas']

        # Stream 6: MR beta
        mr_beta = self._get_value(nd, genomic_data, gene, 'mr_beta', 'mr_b')
        if mr_beta is not None:
            streams['mr_beta'] = float(mr_beta)
            active_weights['mr_beta'] = self._weights['mr_beta']

        # Stream 7: eQTL
        eqtl = self._get_value(nd, genomic_data, gene, 'eqtl_beta', 'eqtl_effect')
        if eqtl is not None:
            streams['eqtl'] = float(eqtl)
            active_weights['eqtl'] = self._weights['eqtl']

        # Stream 8: SIGNOR
        signor_w = nd.get('signor_weight', nd.get('evidence_signor', None))
        if signor_w is not None:
            streams['signor'] = float(signor_w)
            active_weights['signor'] = self._weights['signor']

        # Stream 9: Network centrality
        centrality = nd.get('betweenness_centrality', nd.get('centrality', None))
        if centrality is not None:
            streams['centrality'] = float(centrality)
            active_weights['centrality'] = self._weights['centrality']

        # Stream 10: DAG tier
        tier = nd.get('network_tier', None)
        tier_map = {'Tier 1': 1.0, 'Tier 2': 0.6, 'Tier 3': 0.3}
        if tier and tier in tier_map:
            streams['dag_tier'] = tier_map[tier]
            active_weights['dag_tier'] = self._weights['dag_tier']

        # Stream 11: Drug sensitivity
        drug_sens = self._get_value(nd, screening_data, gene, 'drug_sensitivity', 'drug_z')
        if drug_sens is not None:
            streams['drug_sensitivity'] = float(drug_sens)
            active_weights['drug_sensitivity'] = self._weights['drug_sensitivity']

        # Stream 12: Conservation
        cons = nd.get('conservation_score', None)
        if cons is not None:
            streams['conservation'] = float(cons)
            active_weights['conservation'] = self._weights['conservation']

        # Stream 13: FluteMLE pathway
        flute = self._get_value(nd, screening_data, gene, 'flute_enrichment', 'pathway_enrichment')
        if flute is not None:
            streams['flute_pathway'] = float(flute)
            active_weights['flute_pathway'] = self._weights['flute_pathway']

        # Stream 14: Editing efficiency
        edit_eff = nd.get('editing_efficiency', None)
        if edit_eff is not None:
            streams['editing_efficiency'] = float(edit_eff)
            active_weights['editing_efficiency'] = self._weights['editing_efficiency']

        # Stream 15: Drug Z-score
        dz = self._get_value(nd, screening_data, gene, 'drugz_score', 'drug_zscore')
        if dz is not None:
            streams['drug_zscore'] = float(dz)
            active_weights['drug_zscore'] = self._weights['drug_zscore']

        # Compute weighted ACE score
        if not active_weights:
            return ACEResult(gene=gene, ace_score=0.0, n_streams=0)

        # Normalize each stream to [-1, 1] range
        normalized = {}
        for k, v in streams.items():
            if k in ('mageck_rra', 'gwas'):
                normalized[k] = min(1.0, v / 10.0)  # -log10(p) scaled
            elif k in ('bagel2_bf',):
                normalized[k] = max(-1.0, min(1.0, v / 20.0))
            else:
                normalized[k] = max(-1.0, min(1.0, v))

        total_weight = sum(active_weights.values())
        ace = sum(normalized[k] * active_weights[k] for k in normalized) / total_weight

        # Bayesian confidence: C = 1 - Π(1 - wᵢ) for active streams
        confidence = 1.0
        for w in active_weights.values():
            confidence *= (1.0 - w * 0.3)
        confidence = 1.0 - confidence

        # Direction
        direction = "suppressive" if ace < -0.1 else ("activating" if ace > 0.1 else "neutral")

        # Essentiality classification
        bf_val = streams.get('bagel2_bf', 0)
        if bf_val > 5:
            ess_class = "Core Essential"
        elif bf_val > 0:
            ess_class = "Context Essential"
        elif bf_val < -5:
            ess_class = "Non-Essential"
        else:
            ess_class = "Ambiguous"

        # Therapeutic alignment
        if ace < -0.2 and ess_class != "Core Essential":
            alignment = "Aggravating"
        elif ace > 0.2 and ess_class != "Core Essential":
            alignment = "Protective"
        elif ess_class == "Core Essential":
            alignment = "Essential-Caution"
        else:
            alignment = "Neutral"

        result = ACEResult(
            gene=gene,
            ace_score=round(ace, 6),
            confidence=round(confidence, 4),
            n_streams=len(streams),
            stream_scores=streams,
            stream_weights=active_weights,
            essentiality_class=ess_class,
            therapeutic_alignment=alignment,
            direction=direction,
        )
        self._results_cache[gene] = result
        return result

    def score_all(self, dag, screening_data: Optional[Dict] = None,
                  genomic_data: Optional[Dict] = None,
                  verbose: bool = True) -> Dict[str, ACEResult]:
        """Score all regulatory genes in the DAG."""
        results = {}
        reg_genes = [n for n in dag.nodes()
                     if dag.nodes[n].get('layer') == 'regulatory']

        for idx, gene in enumerate(reg_genes):
            if verbose and (idx + 1) % 1000 == 0:
                logger.info(f"ACE scoring: {idx+1}/{len(reg_genes)}")
            results[gene] = self.score_gene(gene, dag, screening_data, genomic_data)

        if verbose:
            logger.info(f"Scored {len(results)} genes | "
                        f"Mean ACE: {np.mean([r.ace_score for r in results.values()]):.3f}")
        return results

    def get_top_drivers(self, results: Dict[str, ACEResult],
                        n: int = 50) -> List[ACEResult]:
        """Get top causal drivers sorted by ACE score (most negative first)."""
        sorted_results = sorted(results.values(), key=lambda r: r.ace_score)
        return sorted_results[:n]

    def _get_value(self, nd: Dict, ext_data: Optional[Dict], gene: str,
                   *keys) -> Optional[float]:
        """Try to get a value from node data or external data dict."""
        for key in keys:
            val = nd.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        if ext_data and gene in ext_data:
            gd = ext_data[gene]
            if isinstance(gd, dict):
                for key in keys:
                    val = gd.get(key)
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            pass
        return None
