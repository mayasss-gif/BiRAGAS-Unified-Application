"""
BiRAGAS ACE Superior Scorer
==============================
Enhanced ACE (Accumulated Causal Effect) scoring with 12 evidence streams.

Goes beyond standard MAGeCK ACE by integrating:
1. MAGeCK RRA rank score (statistical depletion)
2. MAGeCK MLE beta score (maximum likelihood effect)
3. BAGEL2 Bayes Factor (essentiality probability)
4. Perturb-seq transcriptomic effect size
5. GWAS genetic association strength
6. Mendelian Randomization causal estimate
7. eQTL expression-variant coupling
8. SIGNOR physical interaction weight
9. Pathway centrality (BiRAGAS Phase 2)
10. Causal DAG position (topological importance)
11. Drug sensitivity (chemogenetic screen)
12. Conservation score (cross-species)

Output: SuperiorACE score (0-1 scale, higher = stronger causal driver)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("biragas.crispr_engine.ace_scorer")


@dataclass
class ACEScorerConfig:
    """Configuration for Superior ACE scoring."""
    # 12-stream weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "mageck_rra": 0.12,           # Statistical depletion rank
        "mageck_mle": 0.12,           # Maximum likelihood beta
        "bagel2_bf": 0.10,            # Essentiality Bayes Factor
        "perturbseq_effect": 0.10,    # Single-cell perturbation
        "gwas_strength": 0.10,        # Genetic association
        "mr_estimate": 0.10,          # Mendelian Randomization
        "eqtl_coupling": 0.06,        # Expression QTL
        "signor_interaction": 0.06,   # Physical interaction
        "pathway_centrality": 0.08,   # Network position
        "dag_importance": 0.08,       # Causal DAG role
        "drug_sensitivity": 0.05,     # Chemogenetic screen
        "conservation": 0.03,         # Cross-species
    })

    # Normalization parameters
    rra_threshold: float = 0.05       # RRA p-value threshold
    mle_threshold: float = -0.2       # MLE beta threshold
    bf_threshold: float = 5.0         # BAGEL2 BF threshold for essential
    ace_strong: float = -0.3          # Strong driver threshold
    min_evidence_streams: int = 3     # Minimum streams for confidence


@dataclass
class SuperiorACEResult:
    """Enhanced ACE score result."""
    gene: str = ""
    superior_ace: float = 0.0          # Final composite (0-1)

    # Per-stream scores (all normalized to 0-1)
    stream_scores: Dict[str, float] = field(default_factory=dict)
    streams_available: int = 0
    streams_missing: List[str] = field(default_factory=list)

    # Classification
    driver_class: str = ""             # "Strong Driver", "Moderate Driver", "Weak", "Non-Driver"
    confidence: float = 0.0
    confidence_label: str = ""         # "Very High", "High", "Medium", "Low"

    # Original metrics
    raw_ace: float = 0.0
    raw_rra: float = 0.0
    raw_mle_beta: float = 0.0
    raw_bf: float = 0.0


class ACESuperiorScorer:
    """
    Computes Superior ACE scores by integrating 12 evidence streams.

    Each gene receives a composite score reflecting convergent evidence
    from CRISPR screening, genetics, network analysis, and functional data.

    Usage:
        scorer = ACESuperiorScorer(dag)
        results = scorer.score_all_genes()
        result = scorer.score_gene("STAT4", gene_data)
    """

    def __init__(self, dag=None, config: Optional[ACEScorerConfig] = None):
        self.dag = dag
        self.config = config or ACEScorerConfig()

    def score_all_genes(self, gene_data: Optional[Dict[str, Dict]] = None) -> Dict[str, SuperiorACEResult]:
        """Score all genes using available evidence streams."""
        results = {}

        if self.dag:
            for gene in self.dag.nodes():
                if self.dag.nodes[gene].get('layer') == 'regulatory':
                    node_data = dict(self.dag.nodes[gene])
                    if gene_data and gene in gene_data:
                        node_data.update(gene_data[gene])
                    results[gene] = self._score_single(gene, node_data)

        return results

    def score_gene(self, gene: str, data: Dict[str, Any]) -> SuperiorACEResult:
        """Score a single gene."""
        return self._score_single(gene, data)

    def _score_single(self, gene: str, data: Dict) -> SuperiorACEResult:
        """Compute Superior ACE for one gene."""
        result = SuperiorACEResult(gene=gene)
        scores = {}
        missing = []

        # Stream 1: MAGeCK RRA
        rra = data.get('rra_score', data.get('neg_score'))
        if rra is not None:
            scores['mageck_rra'] = self._normalize_rra(float(rra))
            result.raw_rra = float(rra)
        else:
            missing.append('mageck_rra')

        # Stream 2: MAGeCK MLE beta
        mle = data.get('mle_beta', data.get('beta'))
        if mle is not None:
            scores['mageck_mle'] = self._normalize_mle(float(mle))
            result.raw_mle_beta = float(mle)
        else:
            missing.append('mageck_mle')

        # Stream 3: BAGEL2 Bayes Factor
        bf = data.get('bayes_factor', data.get('bagel2_bf'))
        if bf is not None:
            scores['bagel2_bf'] = self._normalize_bf(float(bf))
            result.raw_bf = float(bf)
        else:
            missing.append('bagel2_bf')

        # Stream 4: Perturb-seq effect
        ps_effect = data.get('perturbseq_effect', data.get('perturbation_ace'))
        if ps_effect is not None:
            scores['perturbseq_effect'] = min(1.0, abs(float(ps_effect)) * 2)
            result.raw_ace = float(ps_effect)
        else:
            missing.append('perturbseq_effect')

        # Stream 5: GWAS strength
        if data.get('gwas_hit'):
            gwas_pval = data.get('gwas_pval', 5e-8)
            scores['gwas_strength'] = min(1.0, -math.log10(max(float(gwas_pval), 1e-300)) / 30)
        else:
            missing.append('gwas_strength')

        # Stream 6: MR estimate
        if data.get('mr_validated'):
            mr_beta = data.get('mr_beta', 0.3)
            scores['mr_estimate'] = min(1.0, abs(float(mr_beta)) * 2)
        else:
            missing.append('mr_estimate')

        # Stream 7: eQTL coupling
        eqtl = data.get('eqtl_beta')
        if eqtl is not None:
            scores['eqtl_coupling'] = min(1.0, abs(float(eqtl)) * 2)
        else:
            missing.append('eqtl_coupling')

        # Stream 8: SIGNOR interaction
        signor = data.get('signor_weight', data.get('signor_score'))
        if signor is not None:
            scores['signor_interaction'] = min(1.0, float(signor))
        else:
            missing.append('signor_interaction')

        # Stream 9: Pathway centrality (from BiRAGAS Phase 2)
        ci = data.get('causal_importance', 0)
        if ci > 0:
            scores['pathway_centrality'] = min(1.0, float(ci) * 2)
        else:
            missing.append('pathway_centrality')

        # Stream 10: DAG importance
        tier = data.get('causal_tier', '')
        if tier:
            tier_scores = {'Tier_1_Master_Regulator': 1.0, 'Tier_2_Secondary_Driver': 0.6, 'Tier_3_Downstream_Effector': 0.3}
            scores['dag_importance'] = tier_scores.get(tier, 0.3)
        else:
            missing.append('dag_importance')

        # Stream 11: Drug sensitivity
        drug_beta = data.get('drug_sensitivity_beta', data.get('drug_screen_beta'))
        if drug_beta is not None:
            scores['drug_sensitivity'] = min(1.0, abs(float(drug_beta)))
        else:
            missing.append('drug_sensitivity')

        # Stream 12: Conservation
        conservation = data.get('conservation_score', data.get('phastcons'))
        if conservation is not None:
            scores['conservation'] = min(1.0, float(conservation))
        else:
            missing.append('conservation')

        # Compute weighted composite
        total_weight = 0.0
        weighted_sum = 0.0
        for stream, score in scores.items():
            w = self.config.weights.get(stream, 0)
            weighted_sum += w * score
            total_weight += w

        # Normalize by available evidence
        if total_weight > 0:
            result.superior_ace = weighted_sum / total_weight  # Normalize to available
        else:
            result.superior_ace = 0.0

        result.stream_scores = scores
        result.streams_available = len(scores)
        result.streams_missing = missing

        # Classification
        if result.superior_ace >= 0.7:
            result.driver_class = "Strong Driver"
        elif result.superior_ace >= 0.5:
            result.driver_class = "Moderate Driver"
        elif result.superior_ace >= 0.3:
            result.driver_class = "Weak Driver"
        else:
            result.driver_class = "Non-Driver"

        # Confidence based on evidence coverage
        result.confidence = min(1.0, result.streams_available / 12.0 * 1.5)
        if result.confidence >= 0.85:
            result.confidence_label = "Very High"
        elif result.confidence >= 0.65:
            result.confidence_label = "High"
        elif result.confidence >= 0.40:
            result.confidence_label = "Medium"
        else:
            result.confidence_label = "Low"

        return result

    def _normalize_rra(self, rra_score: float) -> float:
        """Normalize RRA score (lower = more depleted = better)."""
        if rra_score <= 0:
            return 1.0
        return min(1.0, -math.log10(max(rra_score, 1e-10)) / 5)

    def _normalize_mle(self, beta: float) -> float:
        """Normalize MLE beta (more negative = stronger depletion)."""
        return min(1.0, max(0, abs(beta) / 1.0))

    def _normalize_bf(self, bf: float) -> float:
        """Normalize BAGEL2 Bayes Factor (higher = more essential)."""
        if bf <= 0:
            return 0.0
        return min(1.0, bf / 20.0)
