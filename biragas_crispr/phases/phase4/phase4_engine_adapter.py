"""
Phase4EngineAdapter — Wraps CRISPR Engines as Phase 4 Drop-In Replacements
=============================================================================
Phase 4 basic CounterfactualSimulator: 1 method, O(V+E), no CI, no guide-level.
CRISPR MultiKnockoutAgent: 7 methods, Monte Carlo CI, inline resistance.
CRISPR MegaKnockoutEngine: O(1), 177K scale, vectorized combinations.

This adapter wraps the CRISPR engines to produce Phase 4-compatible outputs
so downstream phases (5, 6, 7) receive results in the expected format.

Auto-selects engine based on gene count:
    < 100 genes → basic CounterfactualSimulator (Phase 4 native)
    100-5000 → MultiKnockoutAgent (7-method ensemble)
    > 5000 → MegaKnockoutEngine (sparse matrix O(1))
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("biragas.crispr_phase4.adapter")

# Ensure CRISPR engine is importable
_CRISPR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CRISPR_MultiKnockout")
if os.path.isdir(_CRISPR_PATH) and _CRISPR_PATH not in sys.path:
    sys.path.insert(0, _CRISPR_PATH)


class Phase4EngineAdapter:
    """
    Wraps CRISPR engines as Phase 4 drop-in replacements.

    Usage:
        adapter = Phase4EngineAdapter()
        results = adapter.run_phase4(dag, engine="auto")

        # results has Phase 4-compatible format:
        # {'counterfactual_results': {gene: {...}}, 'resistance_analysis': {...}, ...}
    """

    def __init__(self):
        self._engine = None
        self._engine_name = ""

    def run_phase4(self, dag: nx.DiGraph, engine: str = "auto",
                   disease_node: str = "Disease_Activity",
                   max_genes: int = 0, crispr_dir: str = "") -> Dict:
        """
        Run Phase 4 using the optimal CRISPR engine.

        Returns Phase 4-compatible output dict.
        """
        n_genes = sum(1 for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory')
        start = time.time()

        # Auto-select engine
        if engine == "auto":
            if n_genes > 5000:
                engine = "mega"
            elif n_genes > 100:
                engine = "v2"
            else:
                engine = "basic"

        logger.info(f"Phase4Adapter: {n_genes} genes → engine={engine}")

        if engine == "basic":
            return self._run_basic(dag, disease_node, max_genes)
        elif engine == "v2":
            return self._run_v2(dag, disease_node, max_genes)
        elif engine == "mega":
            return self._run_mega(dag, disease_node, max_genes)
        else:
            return self._run_basic(dag, disease_node, max_genes)

    def _run_basic(self, dag, disease_node, max_genes) -> Dict:
        """Run Phase 4 with basic CounterfactualSimulator."""
        try:
            from modules.counterfactual_simulator import CounterfactualSimulator, CounterfactualConfig
            from modules.resistance_mechanism_identifier import ResistanceMechanismIdentifier, ResistanceConfig
            from modules.compensation_pathway_analyzer import CompensationPathwayAnalyzer, CompensationConfig

            sim = CounterfactualSimulator(CounterfactualConfig())
            resist = ResistanceMechanismIdentifier(ResistanceConfig())
            comp = CompensationPathwayAnalyzer(CompensationConfig())

            targets = sorted(
                [n for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory'],
                key=lambda g: dag.nodes[g].get('causal_importance', 0), reverse=True
            )[:max_genes or 20]

            cf_results = {}
            for gene in targets:
                try:
                    cf_results[gene] = sim.simulate_knockout(dag, gene, disease_node)
                except Exception:
                    pass

            resistance = {}
            compensation = {}
            for gene in targets[:10]:
                try:
                    resistance[gene] = resist.identify_resistance(dag, gene, disease_node)
                except Exception:
                    pass
                try:
                    compensation[gene] = comp.analyze_compensation(dag, gene, disease_node)
                except Exception:
                    pass

            return {
                'engine': 'basic',
                'counterfactual_results': cf_results,
                'resistance_analysis': resistance,
                'compensation_analysis': compensation,
                'genes_analyzed': len(targets),
            }
        except ImportError as e:
            logger.error(f"Basic Phase 4 modules not available: {e}")
            return {'engine': 'basic', 'error': str(e)}

    def _run_v2(self, dag, disease_node, max_genes) -> Dict:
        """Run Phase 4 with CRISPR MultiKnockoutAgent (7-method)."""
        try:
            from crispr_agentic_engine.knockout_agent import MultiKnockoutAgent

            agent = MultiKnockoutAgent(dag)
            results = agent.predict_all(max_genes=max_genes)

            # Convert to Phase 4-compatible format
            cf_results = {}
            resistance_analysis = {}

            for r in results:
                cf_results[r.gene] = {
                    'target': r.gene,
                    'intervention': 'knockout',
                    'absolute_change': r.disease_effect,
                    'relative_change': r.disease_effect_pct / 100 if r.disease_effect_pct else 0,
                    'n_affected_nodes': r.affected_genes,
                    'affected_programs': r.affected_pathways,
                    'ensemble_score': r.ensemble,
                    'confidence': r.confidence,
                    'ci_low': r.ci_low,
                    'ci_high': r.ci_high,
                    'methods': {
                        'topological': r.topological,
                        'bayesian': r.bayesian,
                        'monte_carlo': r.monte_carlo,
                        'pathway_specific': r.pathway_specific,
                        'feedback_adjusted': r.feedback_adjusted,
                        'ode_dynamic': r.ode_dynamic,
                        'mutual_info': r.mutual_info,
                    },
                }

                resistance_analysis[r.gene] = {
                    'resistance_score': r.resistance_score,
                    'resistance_risk': r.systemic_risk,
                    'bypass_pathways': r.bypass_count,
                    'feedback_loops': r.feedback_count,
                    'compensatory_genes': r.compensators,
                }

            return {
                'engine': 'v2_agentic',
                'counterfactual_results': cf_results,
                'resistance_analysis': resistance_analysis,
                'genes_analyzed': len(results),
                'methods_used': 7,
                'has_confidence_intervals': True,
            }
        except ImportError as e:
            logger.warning(f"v2 not available: {e}, falling back to basic")
            return self._run_basic(dag, disease_node, max_genes)

    def _run_mega(self, dag, disease_node, max_genes) -> Dict:
        """Run Phase 4 with MegaKnockoutEngine (sparse matrix O(1))."""
        try:
            from crispr_agentic_engine.mega_knockout_engine import MegaKnockoutEngine

            engine = MegaKnockoutEngine(dag)
            results = engine.predict_all_knockouts()
            combos = engine.predict_all_combinations(top_n=min(500, max_genes or 500))

            cf_results = {}
            for r in results[:max_genes or len(results)]:
                cf_results[r.gene] = {
                    'target': r.gene,
                    'intervention': 'knockout',
                    'absolute_change': r.disease_effect,
                    'ensemble_score': r.ensemble,
                    'config_type': r.config_type,
                    'n_guides': r.n_guides,
                }

            return {
                'engine': 'mega_sparse',
                'counterfactual_results': cf_results,
                'combination_results': len(combos),
                'genes_analyzed': len(results),
                'scale': engine.get_scale_stats(),
                'knockout_configs': len(results),
            }
        except ImportError as e:
            logger.warning(f"Mega not available: {e}, falling back to v2")
            return self._run_v2(dag, disease_node, max_genes)

    def get_engine_comparison(self) -> Dict:
        """Compare the 3 engines."""
        return {
            'basic': {'methods': 1, 'scalability': 'O(V+E)/gene', 'ci': False, 'guides': False, 'combinations': 'sequential'},
            'v2_agentic': {'methods': 7, 'scalability': 'O(V+E)/gene', 'ci': True, 'guides': False, 'combinations': '6 models'},
            'mega_sparse': {'methods': 'matrix', 'scalability': 'O(1)/gene', 'ci': False, 'guides': True, 'combinations': 'vectorized Bliss'},
        }
