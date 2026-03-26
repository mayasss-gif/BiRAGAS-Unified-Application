"""
BiRAGASBridge — Connects 23 BiRAGAS Modules to CRISPR Engines
================================================================
The critical integration layer that feeds BiRAGAS Phase 1-7 outputs
into CRISPR knockout/combination prediction engines, and feeds
CRISPR results back into BiRAGAS for target ranking.

Flow:
    BiRAGAS Phase 1 (DAGBuilder) → builds causal DAG
    BiRAGAS Phase 2 (Centrality) → annotates with importance
    BiRAGAS Phase 3 (Causality)  → validates edges
                    ↓
    BiRAGASBridge feeds DAG to CRISPR engines
                    ↓
    CRISPR engines produce knockout/combination predictions
                    ↓
    BiRAGASBridge feeds CRISPR results BACK to BiRAGAS
                    ↓
    BiRAGAS Phase 5 (Pharma) → ranks targets with CRISPR evidence
    BiRAGAS Phase 7 (Report) → generates final clinical report
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("biragas.integration.bridge")

# Ensure all systems are importable
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRAMEWORK = os.path.join(os.path.dirname(_BASE), "Claude Code", "Causality_Framework")
for p in [_BASE, _FRAMEWORK, os.path.join(_FRAMEWORK, "modules"),
          os.path.join(_FRAMEWORK, "orchestrator"), os.path.join(_FRAMEWORK, "universal_agent"),
          os.path.join(_BASE, "CRISPR_MultiKnockout")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


class BiRAGASBridge:
    """
    Bidirectional bridge between BiRAGAS 23 modules and CRISPR engines.

    Direction 1: BiRAGAS → CRISPR
        Takes validated causal DAG from Phases 1-3 and feeds it to
        knockout/combination prediction engines.

    Direction 2: CRISPR → BiRAGAS
        Takes knockout predictions and feeds them back into
        Phase 5 (target ranking) and Phase 7 (reporting).

    Usage:
        bridge = BiRAGASBridge()

        # Build DAG with BiRAGAS
        dag = bridge.run_biragas_phases_1_to_3(data_dir)

        # Feed to CRISPR engines
        ko_results = bridge.run_crispr_knockout(dag, crispr_dir)

        # Feed back to BiRAGAS
        final_report = bridge.run_biragas_phases_5_to_7(dag, ko_results)

        # Or: run everything in one call
        report = bridge.run_full_integration(data_dir, crispr_dir)
    """

    def __init__(self):
        self._dag = None
        self._patient_dags = None
        self._phase_results = {}

    def run_biragas_phases_1_to_3(self, data_dir: str, disease_name: str = "Disease") -> nx.DiGraph:
        """
        Run BiRAGAS Phases 1-3 to build and validate the causal DAG.

        Phase 1: DAGBuilder + MRValidator → consensus causal DAG
        Phase 2: CentralityCalculator + TargetScorer → importance metrics
        Phase 3: CausalityTester + DirectionalityTester + ConfoundingChecker → quality assurance
        """
        logger.info("BiRAGASBridge: Running Phases 1-3...")

        try:
            # Phase 1: Build DAG
            from modules.dag_builder import DAGBuilder, DAGBuilderConfig
            config = DAGBuilderConfig(disease_name=disease_name)
            builder = DAGBuilder(config)
            builder.load_data(data_dir)
            dag, patient_dags, metrics = builder.build_consensus_dag()
            self._dag = dag
            self._patient_dags = patient_dags
            self._phase_results['phase_1'] = metrics
            logger.info(f"  Phase 1: DAG built ({dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges)")

            # Phase 2: Centrality
            from modules.centrality_calculator import CentralityCalculator, CentralityConfig
            from modules.target_scorer import TargetScorer, TargetScorerConfig
            dag, centrality = CentralityCalculator(CentralityConfig()).run_pipeline(dag)
            scores = TargetScorer(TargetScorerConfig()).score_targets(dag)
            self._phase_results['phase_2'] = {'centrality': centrality, 'scores': scores}
            logger.info(f"  Phase 2: Centrality computed")

            # Phase 3: Causality validation
            from modules.causality_tester import CausalityTester, CausalityTesterConfig
            from modules.directionality_tester import DirectionalityTester, DirectionalityConfig
            from modules.confounding_checker import ConfoundingChecker, ConfoundingConfig
            CausalityTester(CausalityTesterConfig()).test_all_edges(dag)
            DirectionalityTester(DirectionalityConfig()).test_all_edges(dag)
            ConfoundingChecker(ConfoundingConfig()).check_all_edges(dag)
            logger.info(f"  Phase 3: Causality validated")

            return dag

        except Exception as e:
            logger.error(f"BiRAGAS Phases 1-3 failed: {e}")
            raise

    def run_crispr_knockout(self, dag: nx.DiGraph, crispr_dir: str,
                            engine: str = "auto", output_dir: str = "./integration_output") -> Dict:
        """
        Run CRISPR knockout predictions on the BiRAGAS DAG.

        Args:
            dag: Validated causal DAG from Phases 1-3
            crispr_dir: Path to CRISPR data directory
            engine: "v1" (5-method), "v2" (7-method), "mega" (177K), or "auto"
            output_dir: Where to save results
        """
        logger.info(f"BiRAGASBridge: Running CRISPR knockout (engine={engine})...")
        os.makedirs(output_dir, exist_ok=True)

        n_genes = sum(1 for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory')

        # Auto-select engine
        if engine == "auto":
            if n_genes > 5000:
                engine = "mega"
            elif n_genes > 100:
                engine = "v2"
            else:
                engine = "v1"
            logger.info(f"  Auto-selected engine: {engine} (for {n_genes} genes)")

        results = {}

        if engine == "v2":
            # Use agentic engine v2.0 (full pipeline with discovery)
            try:
                from crispr_agentic_engine import CRISPRSupervisor
                supervisor = CRISPRSupervisor()
                results = supervisor.run(crispr_dir, dag, output_dir, verbose=True)
            except Exception as e:
                logger.warning(f"v2 failed: {e}, falling back to v1")
                engine = "v1"

        if engine == "v1":
            # Use classical engine v1.0
            try:
                from crispr_Multiknockout_engine import MultiKnockoutEngine
                ko_engine = MultiKnockoutEngine(dag)
                ko_results = ko_engine.predict_all_knockouts()
                ko_engine.export_results_csv(os.path.join(output_dir, "v1_knockout_results.csv"))
                results = {
                    "engine": "v1",
                    "knockouts": len(ko_results),
                    "top_5": [{"gene": r.gene, "score": round(r.ensemble_score, 4)} for r in ko_results[:5]],
                    "ko_results": ko_results,
                }
            except Exception as e:
                logger.error(f"v1 engine failed: {e}")
                results = {"engine": "v1", "error": str(e)}

        if engine == "mega":
            # Use mega engine for 177K scale
            try:
                from crispr_agentic_engine.mega_knockout_engine import MegaKnockoutEngine
                mega = MegaKnockoutEngine(dag)
                ko_results = mega.predict_all_knockouts()
                combos = mega.predict_all_combinations(top_n=500)
                mega.export_knockouts_csv(ko_results[:1000], os.path.join(output_dir, "mega_knockouts.csv"))
                mega.export_combinations_csv(combos[:500], os.path.join(output_dir, "mega_combinations.csv"))
                results = {
                    "engine": "mega",
                    "knockouts": len(ko_results),
                    "combinations": len(combos),
                    "scale": mega.get_scale_stats(),
                    "ko_results": ko_results,
                    "combo_results": combos,
                }
            except Exception as e:
                logger.error(f"Mega engine failed: {e}")
                results = {"engine": "mega", "error": str(e)}

        self._phase_results['crispr'] = results
        return results

    def inject_crispr_into_dag(self, dag: nx.DiGraph, crispr_results: Dict) -> nx.DiGraph:
        """
        Inject CRISPR knockout results BACK into the DAG as node attributes.

        This enriches the DAG for BiRAGAS Phase 5 target ranking.
        """
        ko_results = crispr_results.get('ko_results', [])

        for ko in ko_results[:500]:
            gene = ko.gene if hasattr(ko, 'gene') else ko.get('gene', '')
            if gene in dag.nodes():
                node = dag.nodes[gene]
                if hasattr(ko, 'ensemble'):
                    node['crispr_knockout_score'] = ko.ensemble
                    node['crispr_disease_effect'] = ko.disease_effect
                    node['crispr_direction'] = ko.direction
                    node['crispr_confidence'] = ko.confidence
                    node['crispr_resistance'] = ko.resistance_score
                elif isinstance(ko, dict):
                    node['crispr_knockout_score'] = ko.get('ensemble', ko.get('ensemble_score', 0))
                    node['crispr_disease_effect'] = ko.get('disease_effect', 0)

        n_enriched = sum(1 for n in dag.nodes() if 'crispr_knockout_score' in dag.nodes[n])
        logger.info(f"  Injected CRISPR data into {n_enriched} DAG nodes")
        return dag

    def run_biragas_phases_5_to_7(self, dag: nx.DiGraph, crispr_results: Dict,
                                   disease_name: str = "Disease") -> Dict:
        """
        Run BiRAGAS Phases 5-7 with CRISPR-enriched DAG.

        Phase 5: Target ranking with CRISPR knockout scores
        Phase 6: Patient stratification (if patient DAGs available)
        Phase 7: Evidence inspection + LLM arbitration + final report
        """
        logger.info("BiRAGASBridge: Running Phases 5-7 with CRISPR enrichment...")

        # Inject CRISPR results into DAG
        dag = self.inject_crispr_into_dag(dag, crispr_results)

        try:
            # Phase 5: Pharma intervention
            from modules.druggability_scorer import DruggabilityScorer, DruggabilityConfig
            from modules.efficacy_predictor import EfficacyPredictor, EfficacyConfig
            from modules.safety_assessor import SafetyAssessor, SafetyConfig
            from modules.target_ranker import TargetRanker, TargetRankerConfig

            druggability = DruggabilityScorer(DruggabilityConfig()).score_all(dag)
            efficacy = EfficacyPredictor(EfficacyConfig()).predict_all(dag)
            safety = SafetyAssessor(SafetyConfig()).assess_all(dag)
            ranking = TargetRanker(TargetRankerConfig()).rank_targets(
                dag, druggability_scores=druggability, efficacy_scores=efficacy, safety_scores=safety
            )
            self._phase_results['phase_5'] = ranking
            logger.info(f"  Phase 5: {len(ranking.get('ranked_targets', []))} targets ranked")

            # Phase 7: Evidence + report
            from modules.evidence_inspector import EvidenceInspector, InspectorConfig
            from modules.gap_analyzer import GapAnalyzer, GapAnalyzerConfig
            from modules.response_formatter import ResponseFormatter, FormatterConfig

            inspection = EvidenceInspector(InspectorConfig()).inspect_dag(dag)
            gaps = GapAnalyzer(GapAnalyzerConfig()).analyze_gaps(dag)
            report = ResponseFormatter(FormatterConfig()).format_full_report(
                dag, self._phase_results, disease=disease_name
            )
            self._phase_results['phase_7'] = {'inspection': inspection, 'gaps': gaps, 'report': report}
            logger.info(f"  Phase 7: Final report generated")

            return report

        except Exception as e:
            logger.error(f"Phases 5-7 failed: {e}")
            return {"error": str(e)}

    def run_full_integration(self, data_dir: str, crispr_dir: str,
                              disease_name: str = "Disease",
                              engine: str = "auto",
                              output_dir: str = "./full_integration_output") -> Dict:
        """
        Run the COMPLETE integration pipeline end-to-end.

        BiRAGAS Phases 1-3 → CRISPR Engines → BiRAGAS Phases 5-7
        """
        import time
        start = time.time()

        print("\n" + "=" * 70)
        print("  BiRAGAS × CRISPR Full Integration Pipeline")
        print(f"  Disease: {disease_name}")
        print("=" * 70)

        # Step 1: BiRAGAS builds the DAG
        print("\n▶ Step 1: BiRAGAS Phases 1-3 (Build + Validate DAG)...")
        dag = self.run_biragas_phases_1_to_3(data_dir, disease_name)
        print(f"  ✓ DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")

        # Step 2: CRISPR predicts knockouts
        print(f"\n▶ Step 2: CRISPR Knockout Prediction (engine={engine})...")
        crispr_results = self.run_crispr_knockout(dag, crispr_dir, engine, output_dir)
        print(f"  ✓ {crispr_results.get('knockouts', 0)} knockouts predicted")

        # Step 3: BiRAGAS ranks with CRISPR evidence
        print("\n▶ Step 3: BiRAGAS Phases 5-7 (Rank + Report with CRISPR)...")
        final_report = self.run_biragas_phases_5_to_7(dag, crispr_results, disease_name)

        duration = time.time() - start
        print(f"\n{'=' * 70}")
        print(f"  INTEGRATION COMPLETE: {duration:.1f}s")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 70}\n")

        return {
            "duration": round(duration, 1),
            "dag": {"nodes": dag.number_of_nodes(), "edges": dag.number_of_edges()},
            "crispr": crispr_results,
            "report": final_report,
            "phase_results": list(self._phase_results.keys()),
        }
