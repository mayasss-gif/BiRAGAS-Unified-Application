"""
SuperiorACEAgent v2.0 — 15-Stream Enhanced ACE Scoring
========================================================
Upgrades: 15 streams (was 12), auto-loads from screening + perturbseq agents.
New streams: FluteMLE pathway enrichment, targeted editing efficiency, drug screen Z-score.
"""
import logging, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("biragas.crispr.ace")

@dataclass
class ACEResult:
    gene: str = ""
    superior_ace: float = 0.0
    stream_scores: Dict[str, float] = field(default_factory=dict)
    streams_available: int = 0
    driver_class: str = ""
    confidence: float = 0.0

class SuperiorACEAgent:
    """15-stream ACE scoring integrating all available CRISPR + genomic evidence."""

    WEIGHTS = {
        "mageck_rra": 0.10, "mageck_mle": 0.10, "bagel2_bf": 0.08,
        "perturbseq": 0.08, "gwas": 0.09, "mr": 0.09,
        "eqtl": 0.05, "signor": 0.05, "centrality": 0.07,
        "dag_tier": 0.07, "drug_sensitivity": 0.06, "conservation": 0.03,
        "flute_pathway": 0.05, "editing_efficiency": 0.04, "drug_z": 0.04,
    }

    def __init__(self, dag=None, screening_results=None, perturbseq_results=None):
        self.dag = dag
        self.screening = screening_results or {}
        self.perturbseq = perturbseq_results or {}

    def score_all(self) -> Dict[str, ACEResult]:
        results = {}
        if self.dag:
            for gene in self.dag.nodes():
                if self.dag.nodes[gene].get('layer') == 'regulatory':
                    results[gene] = self._score(gene)
        return results

    def _score(self, gene: str) -> ACEResult:
        r = ACEResult(gene=gene)
        nd = self.dag.nodes[gene] if self.dag and gene in self.dag else {}
        sc = self.screening.get(gene)
        ps = self.perturbseq.get(gene)
        scores = {}

        # Stream 1: RRA
        if sc and hasattr(sc, 'rra_neg_score') and sc.rra_neg_score < 1.0:
            scores['mageck_rra'] = min(1.0, -math.log10(max(sc.rra_neg_score, 1e-10)) / 5)
        # Stream 2: MLE
        if sc and hasattr(sc, 'mle_beta') and sc.mle_beta != 0:
            scores['mageck_mle'] = min(1.0, abs(sc.mle_beta))
        # Stream 3: BAGEL2
        if sc and hasattr(sc, 'bayes_factor') and sc.bayes_factor > 0:
            scores['bagel2_bf'] = min(1.0, sc.bayes_factor / 20)
        # Stream 4: Perturb-seq
        if ps and hasattr(ps, 'mean_effect_size'):
            scores['perturbseq'] = min(1.0, abs(ps.mean_effect_size) * 2)
        elif nd.get('perturbation_ace'):
            scores['perturbseq'] = min(1.0, abs(float(nd['perturbation_ace'])) * 2)
        # Stream 5: GWAS
        if nd.get('gwas_hit'):
            scores['gwas'] = min(1.0, -math.log10(max(float(nd.get('gwas_pval', 5e-8)), 1e-300)) / 30)
        # Stream 6: MR
        if nd.get('mr_validated'):
            scores['mr'] = min(1.0, abs(float(nd.get('mr_beta', 0.3))) * 2)
        # Stream 7: eQTL
        if nd.get('eqtl_beta'):
            scores['eqtl'] = min(1.0, abs(float(nd['eqtl_beta'])) * 2)
        # Stream 8: SIGNOR
        if nd.get('signor_weight'):
            scores['signor'] = min(1.0, float(nd['signor_weight']))
        # Stream 9: Centrality
        ci = nd.get('causal_importance', 0)
        if ci > 0: scores['centrality'] = min(1.0, float(ci) * 2)
        # Stream 10: DAG tier
        tier = nd.get('causal_tier', '')
        if tier: scores['dag_tier'] = {'Tier_1_Master_Regulator': 1.0, 'Tier_2_Secondary_Driver': 0.6, 'Tier_3_Downstream_Effector': 0.3}.get(tier, 0.3)
        # Stream 11: Drug sensitivity
        if sc and hasattr(sc, 'drug_beta') and sc.drug_beta != 0:
            scores['drug_sensitivity'] = min(1.0, abs(sc.drug_beta))
        # Stream 12: Conservation
        if nd.get('conservation_score'):
            scores['conservation'] = min(1.0, float(nd['conservation_score']))
        # Stream 13: FluteMLE pathway (NEW)
        if sc and hasattr(sc, 'enriched_pathways') and sc.enriched_pathways:
            scores['flute_pathway'] = min(1.0, len(sc.enriched_pathways) * 0.1)
        # Stream 14: Editing efficiency (NEW - from targeted)
        if nd.get('editing_efficiency'):
            scores['editing_efficiency'] = min(1.0, float(nd['editing_efficiency']))
        # Stream 15: Drug Z-score (NEW)
        if sc and hasattr(sc, 'drug_z') and sc.drug_z != 0:
            scores['drug_z'] = min(1.0, abs(sc.drug_z) / 3)

        # Composite
        total_w = sum(self.WEIGHTS[k] for k in scores)
        if total_w > 0:
            r.superior_ace = sum(self.WEIGHTS[k] * scores[k] for k in scores) / total_w
        r.stream_scores = scores
        r.streams_available = len(scores)
        r.confidence = min(1.0, len(scores) / 15 * 1.5)
        r.driver_class = "Strong Driver" if r.superior_ace >= 0.7 else "Moderate" if r.superior_ace >= 0.5 else "Weak" if r.superior_ace >= 0.3 else "Non-Driver"
        return r
