"""
BiRAGAS Combination Knockout Predictor
========================================
Predicts outcomes of N×N multi-gene CRISPR knockout combinations.

For 19,091 genes: 19,091 × 19,091 = 364,467,281 possible pairs.
For practical analysis: top 500 × 500 = 250,000 prioritized pairs.

Synergy Models:
    1. Bliss Independence — P(AB) = P(A) + P(B) - P(A)×P(B)
    2. Highest Single Agent (HSA) — max(effect_A, effect_B)
    3. Loewe Additivity — dose equivalence model
    4. Graph-Based Epistasis — non-additive pathway interactions
    5. Compensation Blocking — targets that prevent resistance

Output: Synergy score, interaction type, recommended combinations
"""

import logging
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from itertools import combinations

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_engine.combination")


@dataclass
class CombinationConfig:
    """Configuration for combination prediction."""
    max_combination_size: int = 3          # Up to triples
    top_n_singles: int = 200               # Top singles to combine
    bliss_weight: float = 0.25             # Bliss Independence weight
    hsa_weight: float = 0.20               # HSA weight
    loewe_weight: float = 0.15             # Loewe weight
    epistasis_weight: float = 0.25         # Graph epistasis weight
    compensation_weight: float = 0.15      # Compensation blocking weight
    synergy_threshold: float = 0.1         # Min synergy to report
    min_safety: float = 0.3                # Min combined safety
    top_n_combinations: int = 100          # Top combos to return
    parallel_pathway_bonus: float = 0.2    # Bonus for targeting different pathways


@dataclass
class CombinationResult:
    """Result of a multi-knockout combination prediction."""
    genes: List[str] = field(default_factory=list)
    combination_size: int = 0

    # Individual effects
    individual_effects: Dict[str, float] = field(default_factory=dict)

    # Predicted combined effect
    predicted_effect: float = 0.0
    expected_additive: float = 0.0
    synergy_score: float = 0.0            # >0 = synergistic, <0 = antagonistic, 0 = additive
    interaction_type: str = ""             # "synergistic", "additive", "antagonistic"

    # Model scores
    bliss_score: float = 0.0
    hsa_score: float = 0.0
    loewe_score: float = 0.0
    epistasis_score: float = 0.0
    compensation_blocking: float = 0.0

    # Pathway analysis
    pathway_coverage: float = 0.0          # Fraction of disease pathways hit
    pathway_classes: List[str] = field(default_factory=list)
    unique_pathways: int = 0
    complementary: bool = False            # Targets different pathway classes

    # Safety
    combined_safety: float = 0.0           # Weakest-link safety
    safety_alerts: List[str] = field(default_factory=list)

    # Resistance
    resistance_blocked: float = 0.0        # Fraction of resistance pathways blocked
    compensation_prevented: List[str] = field(default_factory=list)

    # Ranking
    composite_score: float = 0.0
    rank: int = 0


class CombinationPredictor:
    """
    Predicts outcomes of multi-gene CRISPR knockout combinations.

    Uses 5 synergy models with pathway-aware scoring to identify
    optimal 2-3 target combinations for combination therapy.

    Usage:
        predictor = CombinationPredictor(dag, knockout_engine)
        pairs = predictor.predict_all_pairs(top_n=200)
        triples = predictor.predict_triples(top_n=50)
        best = predictor.get_best_combinations(n=20)
    """

    def __init__(self, dag: nx.DiGraph, knockout_results: Dict[str, Any],
                 config: Optional[CombinationConfig] = None):
        self.dag = dag
        self.ko_results = knockout_results  # gene -> KnockoutResult
        self.config = config or CombinationConfig()
        self.rng = np.random.RandomState(42)

        self._disease_node = self._find_disease_node()
        self._disease_programs = self._get_disease_programs()

        logger.info(f"CombinationPredictor: {len(knockout_results)} singles, "
                     f"{len(self._disease_programs)} disease programs")

    def predict_all_pairs(self, top_n_singles: int = 0) -> List[CombinationResult]:
        """
        Predict all pairwise combinations of top knockout targets.

        For top_n singles: C(top_n, 2) pairs evaluated.
        Default top_n=200 → 19,900 pairs.
        """
        n = top_n_singles or self.config.top_n_singles
        top_genes = self._get_top_genes(n)

        start = time.time()
        total_pairs = len(top_genes) * (len(top_genes) - 1) // 2
        logger.info(f"Predicting {total_pairs} pairwise combinations from {len(top_genes)} genes...")

        results = []
        for i, (g1, g2) in enumerate(combinations(top_genes, 2)):
            result = self._predict_pair(g1, g2)
            if result.composite_score > 0:
                results.append(result)

            if (i + 1) % 5000 == 0:
                logger.info(f"  {i+1}/{total_pairs} pairs evaluated")

        results.sort(key=lambda r: -r.composite_score)
        for i, r in enumerate(results):
            r.rank = i + 1

        duration = time.time() - start
        logger.info(f"Pair prediction complete: {len(results)} valid combos in {duration:.1f}s")

        return results[:self.config.top_n_combinations]

    def predict_triples(self, top_n_singles: int = 50) -> List[CombinationResult]:
        """Predict triple combinations from top targets."""
        top_genes = self._get_top_genes(min(top_n_singles, 50))

        results = []
        for g1, g2, g3 in combinations(top_genes, 3):
            result = self._predict_triple(g1, g2, g3)
            if result.composite_score > 0:
                results.append(result)

        results.sort(key=lambda r: -r.composite_score)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:self.config.top_n_combinations]

    def predict_specific(self, genes: List[str]) -> CombinationResult:
        """Predict outcome for a specific gene combination."""
        if len(genes) == 2:
            return self._predict_pair(genes[0], genes[1])
        elif len(genes) == 3:
            return self._predict_triple(genes[0], genes[1], genes[2])
        else:
            raise ValueError(f"Combination size {len(genes)} not supported (max 3)")

    def get_best_combinations(self, n: int = 20) -> List[CombinationResult]:
        """Get best N combinations (pairs + triples combined)."""
        pairs = self.predict_all_pairs()
        triples = self.predict_triples()
        all_combos = pairs + triples
        all_combos.sort(key=lambda r: -r.composite_score)
        for i, r in enumerate(all_combos):
            r.rank = i + 1
        return all_combos[:n]

    # ========================================================================
    # PAIR PREDICTION
    # ========================================================================

    def _predict_pair(self, gene_a: str, gene_b: str) -> CombinationResult:
        """Predict outcome of knocking out gene_a AND gene_b simultaneously."""
        result = CombinationResult(
            genes=[gene_a, gene_b],
            combination_size=2,
        )

        # Get individual effects
        ko_a = self.ko_results.get(gene_a)
        ko_b = self.ko_results.get(gene_b)
        if not ko_a or not ko_b:
            return result

        eff_a = abs(getattr(ko_a, 'disease_effect', 0) or ko_a.get('disease_effect', 0) if isinstance(ko_a, dict) else 0)
        eff_b = abs(getattr(ko_b, 'disease_effect', 0) or ko_b.get('disease_effect', 0) if isinstance(ko_b, dict) else 0)

        if isinstance(ko_a, dict):
            eff_a = abs(ko_a.get('disease_effect', ko_a.get('ensemble_score', 0)))
            eff_b = abs(ko_b.get('disease_effect', ko_b.get('ensemble_score', 0)))
        else:
            eff_a = abs(ko_a.disease_effect) if hasattr(ko_a, 'disease_effect') else abs(ko_a.ensemble_score)
            eff_b = abs(ko_b.disease_effect) if hasattr(ko_b, 'disease_effect') else abs(ko_b.ensemble_score)

        result.individual_effects = {gene_a: eff_a, gene_b: eff_b}

        # Model 1: Bliss Independence
        # P(AB) = P(A) + P(B) - P(A)×P(B)
        bliss_expected = eff_a + eff_b - eff_a * eff_b
        result.bliss_score = bliss_expected

        # Model 2: Highest Single Agent
        hsa = max(eff_a, eff_b)
        result.hsa_score = hsa

        # Model 3: Loewe Additivity
        # Dose equivalence: effect = eff_a + eff_b (simple additive)
        loewe = eff_a + eff_b
        result.loewe_score = min(loewe, 1.0)

        # Model 4: Graph-Based Epistasis
        epistasis = self._compute_epistasis(gene_a, gene_b, eff_a, eff_b)
        result.epistasis_score = epistasis

        # Model 5: Compensation Blocking
        comp_block = self._compute_compensation_blocking(gene_a, gene_b)
        result.compensation_blocking = comp_block

        # Predicted combined effect (weighted ensemble)
        w = self.config
        result.predicted_effect = (
            w.bliss_weight * bliss_expected +
            w.hsa_weight * hsa +
            w.loewe_weight * min(loewe, 1.0) +
            w.epistasis_weight * epistasis +
            w.compensation_weight * comp_block
        )

        # Expected additive (Bliss as reference)
        result.expected_additive = bliss_expected

        # Synergy score: how much better than expected
        if bliss_expected > 0:
            result.synergy_score = (result.predicted_effect - bliss_expected) / bliss_expected
        else:
            result.synergy_score = 0.0

        # Interaction type
        if result.synergy_score > self.config.synergy_threshold:
            result.interaction_type = "synergistic"
        elif result.synergy_score < -self.config.synergy_threshold:
            result.interaction_type = "antagonistic"
        else:
            result.interaction_type = "additive"

        # Pathway coverage
        progs_a = self._get_gene_programs(gene_a)
        progs_b = self._get_gene_programs(gene_b)
        combined_progs = progs_a | progs_b
        result.pathway_coverage = len(combined_progs & self._disease_programs) / max(len(self._disease_programs), 1)
        result.unique_pathways = len(combined_progs)

        # Pathway classes
        classes_a = self._get_pathway_classes(gene_a)
        classes_b = self._get_pathway_classes(gene_b)
        result.pathway_classes = list(classes_a | classes_b)
        result.complementary = len(classes_a & classes_b) == 0 and len(classes_a) > 0 and len(classes_b) > 0

        # Complementary pathway bonus
        pathway_bonus = self.config.parallel_pathway_bonus if result.complementary else 0

        # Safety (weakest link)
        safety_a = self._get_safety(gene_a)
        safety_b = self._get_safety(gene_b)
        result.combined_safety = min(safety_a, safety_b)
        if result.combined_safety < self.config.min_safety:
            result.safety_alerts.append(f"Low combined safety: {result.combined_safety:.2f}")

        # Resistance blocking
        result.resistance_blocked = min(1.0, result.pathway_coverage + comp_block * 0.3)

        # Composite score
        result.composite_score = (
            result.predicted_effect * 0.30 +
            result.synergy_score * 0.20 +
            result.pathway_coverage * 0.15 +
            result.combined_safety * 0.10 +
            result.resistance_blocked * 0.15 +
            pathway_bonus * 0.10
        )

        return result

    def _predict_triple(self, g1: str, g2: str, g3: str) -> CombinationResult:
        """Predict triple combination (simplified — extends pair logic)."""
        # Get pairwise
        pair_ab = self._predict_pair(g1, g2)
        pair_ac = self._predict_pair(g1, g3)
        pair_bc = self._predict_pair(g2, g3)

        result = CombinationResult(
            genes=[g1, g2, g3],
            combination_size=3,
        )

        # Aggregate pairwise synergies
        result.synergy_score = max(pair_ab.synergy_score, pair_ac.synergy_score, pair_bc.synergy_score)
        result.predicted_effect = max(pair_ab.predicted_effect, pair_ac.predicted_effect, pair_bc.predicted_effect) * 1.1

        # Pathway coverage (union of all three)
        progs = self._get_gene_programs(g1) | self._get_gene_programs(g2) | self._get_gene_programs(g3)
        result.pathway_coverage = len(progs & self._disease_programs) / max(len(self._disease_programs), 1)

        classes = self._get_pathway_classes(g1) | self._get_pathway_classes(g2) | self._get_pathway_classes(g3)
        result.pathway_classes = list(classes)
        result.complementary = len(classes) >= 3

        result.combined_safety = min(self._get_safety(g1), self._get_safety(g2), self._get_safety(g3))
        result.resistance_blocked = min(1.0, result.pathway_coverage * 1.2)

        pathway_bonus = self.config.parallel_pathway_bonus if result.complementary else 0

        result.composite_score = (
            result.predicted_effect * 0.25 +
            result.synergy_score * 0.20 +
            result.pathway_coverage * 0.20 +
            result.combined_safety * 0.10 +
            result.resistance_blocked * 0.15 +
            pathway_bonus * 0.10
        )

        if result.synergy_score > 0.1:
            result.interaction_type = "synergistic"
        elif result.synergy_score < -0.1:
            result.interaction_type = "antagonistic"
        else:
            result.interaction_type = "additive"

        return result

    # ========================================================================
    # SYNERGY MODELS
    # ========================================================================

    def _compute_epistasis(self, gene_a: str, gene_b: str, eff_a: float, eff_b: float) -> float:
        """
        Graph-based epistasis detection.

        Epistasis = non-additive interaction between two genes.
        Detected by examining shared downstream targets, convergent pathways,
        and feedback between the two genes' neighborhoods.
        """
        # Check if genes share downstream targets
        desc_a = set(nx.descendants(self.dag, gene_a)) if gene_a in self.dag else set()
        desc_b = set(nx.descendants(self.dag, gene_b)) if gene_b in self.dag else set()
        shared = desc_a & desc_b

        # More shared targets = more potential for non-additive interaction
        jaccard = len(shared) / max(len(desc_a | desc_b), 1)

        # Check for direct interaction (edge between genes)
        direct = 0.2 if self.dag.has_edge(gene_a, gene_b) or self.dag.has_edge(gene_b, gene_a) else 0.0

        # Check for convergent pathways (both feed into same disease program)
        progs_a = self._get_gene_programs(gene_a)
        progs_b = self._get_gene_programs(gene_b)
        convergent = len(progs_a & progs_b) / max(len(progs_a | progs_b), 1)

        # Epistasis score: high shared + convergent = strong epistasis
        epistasis = (eff_a + eff_b) * (1 + jaccard * 0.3 + direct + convergent * 0.2)
        return min(epistasis, 1.0)

    def _compute_compensation_blocking(self, gene_a: str, gene_b: str) -> float:
        """
        Check if gene_b blocks compensation for gene_a knockout (and vice versa).

        If knocking out gene_a activates compensatory gene_b, then
        simultaneously knocking out both blocks the compensation.
        """
        # Check if gene_b is a compensator for gene_a
        progs_a = self._get_gene_programs(gene_a)
        progs_b = self._get_gene_programs(gene_b)

        overlap = len(progs_a & progs_b) / max(len(progs_a), 1)
        reverse_overlap = len(progs_a & progs_b) / max(len(progs_b), 1)

        # High overlap = gene_b compensates for gene_a
        blocking_score = max(overlap, reverse_overlap)
        return min(blocking_score, 1.0)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _get_top_genes(self, n: int) -> List[str]:
        """Get top N genes by knockout effect."""
        scored = []
        for gene, ko in self.ko_results.items():
            if isinstance(ko, dict):
                score = abs(ko.get('ensemble_score', ko.get('disease_effect', 0)))
            else:
                score = abs(getattr(ko, 'ensemble_score', 0))
            scored.append((gene, score))
        scored.sort(key=lambda x: -x[1])
        return [g for g, _ in scored[:n]]

    def _find_disease_node(self) -> str:
        for n in self.dag.nodes():
            if self.dag.nodes[n].get('layer') == 'trait':
                return n
        return 'Disease_Activity'

    def _get_disease_programs(self) -> Set[str]:
        if self._disease_node not in self.dag:
            return set()
        return set(self.dag.predecessors(self._disease_node))

    def _get_gene_programs(self, gene: str) -> Set[str]:
        if gene not in self.dag:
            return set()
        return {n for n in self.dag.successors(gene) if self.dag.nodes.get(n, {}).get('layer') == 'program'}

    def _get_pathway_classes(self, gene: str) -> Set[str]:
        classes = set()
        for prog in self._get_gene_programs(gene):
            cls = self.dag.nodes[prog].get('main_class', '')
            if cls:
                classes.add(cls)
        return classes

    def _get_safety(self, gene: str) -> float:
        if gene not in self.dag.nodes():
            return 0.5
        ess = self.dag.nodes[gene].get('essentiality_tag', 'Unknown')
        if ess == 'Core Essential':
            return 0.1
        elif ess == 'Tumor-Selective Dependency':
            return 0.8
        elif ess == 'Non-Essential':
            return 1.0
        return 0.5

    def export_results_csv(self, results: List[CombinationResult], filepath: str):
        """Export combination results to CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rank', 'genes', 'combination_size', 'predicted_effect',
                'synergy_score', 'interaction_type', 'pathway_coverage',
                'combined_safety', 'resistance_blocked', 'complementary',
                'pathway_classes', 'composite_score',
                'bliss', 'hsa', 'loewe', 'epistasis', 'compensation_blocking',
            ])
            for r in results:
                writer.writerow([
                    r.rank, '+'.join(r.genes), r.combination_size,
                    round(r.predicted_effect, 6), round(r.synergy_score, 4),
                    r.interaction_type, round(r.pathway_coverage, 4),
                    round(r.combined_safety, 4), round(r.resistance_blocked, 4),
                    r.complementary, ';'.join(r.pathway_classes),
                    round(r.composite_score, 6),
                    round(r.bliss_score, 4), round(r.hsa_score, 4),
                    round(r.loewe_score, 4), round(r.epistasis_score, 4),
                    round(r.compensation_blocking, 4),
                ])
        logger.info(f"Exported {len(results)} combinations to {filepath}")
