"""
BiRAGAS Phase Agents
=====================
Specialized agents for each of the 7 phases in the BiRAGAS pipeline.
Each agent encapsulates its phase's modules and handles execution,
error recovery, and result packaging.

Architecture:
    PhaseAgent (base)
    ├── Phase1Agent: Causality Foundation (DAGBuilder, MRValidator)
    ├── Phase2Agent: Network Causal Importance (CentralityCalculator, TargetScorer)
    ├── Phase3Agent: Causal Calculus (CausalityTester, DirectionalityTester, ConfoundingChecker)
    ├── Phase4Agent: In-Silico Perturbation (CounterfactualSimulator, DAGPropagationEngine, ResistanceMechId, CompensationAnalyzer)
    ├── Phase5Agent: Pharma Intervention (TargetRanker, DruggabilityScorer, EfficacyPredictor, SafetyAssessor, CombinationAnalyzer)
    ├── Phase6Agent: Comparative Evolution (CohortStratifier, DAGComparator, ConservedMotifsIdentifier)
    └── Phase7Agent: Inspection & LLM Arbitration (EvidenceInspector, GapAnalyzer, LLMArbitrator, ResponseFormatter)
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("biragas.phase_agents")


class PhaseAgent(ABC):
    """
    Base class for all phase agents.
    Provides common execution, retry, and error handling logic.
    """

    def __init__(self, phase_name: str, phase_number: int, config=None):
        self.phase_name = phase_name
        self.phase_number = phase_number
        self.config = config
        self.execution_log: List[Dict] = []
        self._modules_initialized = False

    @abstractmethod
    def _init_modules(self):
        """Initialize phase-specific modules."""
        pass

    @abstractmethod
    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        """Execute the core phase logic. Must be implemented by each phase."""
        pass

    def execute(self, dag: nx.DiGraph, context: Dict, max_retries: int = 3) -> Dict:
        """
        Execute this phase with retry logic and error handling.
        Returns a result dict with 'success', 'result', 'errors', 'duration'.
        """
        if not self._modules_initialized:
            self._init_modules()
            self._modules_initialized = True

        errors = []
        for attempt in range(1, max_retries + 1):
            start = time.time()
            try:
                logger.info(f"Phase {self.phase_number} [{self.phase_name}]: attempt {attempt}/{max_retries}")
                result = self._execute_core(dag, context)
                duration = time.time() - start

                self.execution_log.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": duration,
                })

                return {
                    "success": True,
                    "phase": self.phase_name,
                    "phase_number": self.phase_number,
                    "result": result,
                    "errors": [],
                    "duration": duration,
                    "attempts": attempt,
                }

            except Exception as e:
                duration = time.time() - start
                error_msg = f"Phase {self.phase_number} attempt {attempt} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append({
                    "attempt": attempt,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "duration": duration,
                })
                self.execution_log.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "duration": duration,
                })

        return {
            "success": False,
            "phase": self.phase_name,
            "phase_number": self.phase_number,
            "result": {},
            "errors": errors,
            "duration": sum(e.get("duration", 0) for e in errors),
            "attempts": max_retries,
        }


class Phase1Agent(PhaseAgent):
    """
    Phase 1: Causality Foundation
    Modules: DAGBuilder, MRValidator
    """

    def __init__(self, config=None):
        super().__init__("Causality Foundation", 1, config)
        self.dag_builder = None
        self.mr_validator = None

    def _init_modules(self):
        import sys, os
        modules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules")
        if modules_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(modules_dir))

        from modules.dag_builder import DAGBuilder, DAGBuilderConfig
        from modules.mr_validator import MRValidator, MRValidatorConfig

        dag_config = DAGBuilderConfig()
        if self.config:
            dag_config.disease_name = getattr(self.config, 'disease_name', 'Disease')
            dag_config.disease_node = getattr(self.config, 'disease_node', 'Disease_Activity')

        self.dag_builder = DAGBuilder(config=dag_config)
        self.mr_validator = MRValidator(config=MRValidatorConfig())
        logger.info("Phase 1 modules initialized: DAGBuilder, MRValidator")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}

        # Step 1: Build consensus DAG
        data_dir = context.get('data_dir', '')
        data_packet = context.get('data_packet')

        if data_packet:
            self.dag_builder.load_data_packet(data_packet)
        elif data_dir:
            self.dag_builder.load_data(data_dir)
        else:
            raise ValueError("Phase 1 requires either 'data_dir' or 'data_packet' in context")

        consensus_dag, patient_dags, build_metrics = self.dag_builder.build_consensus_dag()
        result['build_metrics'] = build_metrics
        result['patient_dags'] = patient_dags

        # Step 2: MR Validation (if eQTL data available)
        eqtl_data = context.get('eqtl_data')
        if eqtl_data is not None:
            disease_node = context.get('disease_node', 'Disease_Activity')
            consensus_dag, mr_results = self.mr_validator.validate_dag_edges(
                consensus_dag, eqtl_data, target_node=disease_node
            )
            result['mr_validation'] = mr_results
        else:
            result['mr_validation'] = {"note": "No eQTL data provided — MR validation skipped"}

        # Store the DAG in context for downstream phases
        context['consensus_dag'] = consensus_dag
        context['patient_dags'] = patient_dags

        # Copy consensus_dag to the mutable dag reference
        dag.update(consensus_dag)

        logger.info(
            f"Phase 1 complete: {consensus_dag.number_of_nodes()} nodes, "
            f"{consensus_dag.number_of_edges()} edges, "
            f"{len(patient_dags)} patient DAGs"
        )
        return result


class Phase2Agent(PhaseAgent):
    """
    Phase 2: Network Causal Importance
    Modules: CentralityCalculator, TargetScorer
    """

    def __init__(self, config=None):
        super().__init__("Network Causal Importance", 2, config)

    def _init_modules(self):
        from modules.centrality_calculator import CentralityCalculator, CentralityConfig
        from modules.target_scorer import TargetScorer, TargetScorerConfig

        self.centrality = CentralityCalculator(config=CentralityConfig())
        self.scorer = TargetScorer(config=TargetScorerConfig())
        logger.info("Phase 2 modules initialized: CentralityCalculator, TargetScorer")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}

        # Step 1: Compute centrality metrics and tier classification
        enriched_dag, centrality_report = self.centrality.run_pipeline(dag)
        result['centrality'] = centrality_report

        # Step 2: Score targets
        disease_node = context.get('disease_node', 'Disease_Activity')
        target_scores = self.scorer.score_targets(enriched_dag, target_node=disease_node)
        result['target_scores'] = target_scores

        tier_counts = {}
        for n in dag.nodes():
            tier = dag.nodes[n].get('causal_tier', 'unclassified')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        result['tier_distribution'] = tier_counts

        logger.info(f"Phase 2 complete: Tiers = {tier_counts}")
        return result


class Phase3Agent(PhaseAgent):
    """
    Phase 3: Causal Calculus
    Modules: CausalityTester, DirectionalityTester, ConfoundingChecker
    """

    def __init__(self, config=None):
        super().__init__("Causal Calculus", 3, config)

    def _init_modules(self):
        from modules.causality_tester import CausalityTester, CausalityTesterConfig
        from modules.directionality_tester import DirectionalityTester, DirectionalityConfig
        from modules.confounding_checker import ConfoundingChecker, ConfoundingConfig

        self.causality_tester = CausalityTester(config=CausalityTesterConfig())
        self.directionality_tester = DirectionalityTester(config=DirectionalityConfig())
        self.confounding_checker = ConfoundingChecker(config=ConfoundingConfig())
        logger.info("Phase 3 modules initialized: CausalityTester, DirectionalityTester, ConfoundingChecker")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}

        # Step 1: Test causality of all edges
        rag_scores = context.get('rag_scores')
        causality_report = self.causality_tester.test_all_edges(dag, rag_scores=rag_scores)
        result['causality_testing'] = causality_report

        # Step 2: Test and potentially fix edge directionality
        directionality_report = self.directionality_tester.test_all_edges(dag)
        result['directionality_testing'] = directionality_report

        # Step 3: Check for confounding
        confounding_report = self.confounding_checker.check_all_edges(dag)
        result['confounding_analysis'] = confounding_report

        # Collect hallucination counts
        hallucinations = sum(
            1 for _, _, d in dag.edges(data=True)
            if d.get('hallucination_flags') and len(d.get('hallucination_flags', [])) > 0
        )
        result['hallucinations_detected'] = hallucinations

        logger.info(f"Phase 3 complete: {hallucinations} hallucination flags")
        return result


class Phase4Agent(PhaseAgent):
    """
    Phase 4: In-Silico Perturbation
    Modules: CounterfactualSimulator, DAGPropagationEngine, ResistanceMechanismIdentifier, CompensationPathwayAnalyzer
    """

    def __init__(self, config=None):
        super().__init__("In-Silico Perturbation", 4, config)

    def _init_modules(self):
        from modules.counterfactual_simulator import CounterfactualSimulator, CounterfactualConfig
        from modules.dag_propagation_engine import DAGPropagationEngine, PropagationConfig
        from modules.resistance_mechanism_identifier import ResistanceMechanismIdentifier, ResistanceConfig
        from modules.compensation_pathway_analyzer import CompensationPathwayAnalyzer, CompensationConfig

        self.counterfactual = CounterfactualSimulator(config=CounterfactualConfig())
        self.propagation = DAGPropagationEngine(config=PropagationConfig())
        self.resistance = ResistanceMechanismIdentifier(config=ResistanceConfig())
        self.compensation = CompensationPathwayAnalyzer(config=CompensationConfig())
        logger.info("Phase 4 modules initialized: CounterfactualSimulator, DAGPropagationEngine, ResistanceMechId, CompensationAnalyzer")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}
        disease_node = context.get('disease_node', 'Disease_Activity')

        # Get top regulatory targets for perturbation
        regulatory_genes = [
            n for n in dag.nodes()
            if dag.nodes[n].get('layer') == 'regulatory'
        ]
        # Sort by causal importance (if available)
        regulatory_genes.sort(
            key=lambda g: dag.nodes[g].get('causal_importance', 0),
            reverse=True
        )
        top_targets = regulatory_genes[:20]  # Top 20 for perturbation

        # Step 1: Counterfactual simulations (knockout each target)
        counterfactual_results = {}
        for gene in top_targets:
            try:
                ko_result = self.counterfactual.simulate_knockout(dag, gene, disease_node=disease_node)
                counterfactual_results[gene] = ko_result
            except Exception as e:
                logger.warning(f"Counterfactual failed for {gene}: {e}")
                counterfactual_results[gene] = {"error": str(e)}

        result['counterfactual_results'] = counterfactual_results
        context['counterfactual_results'] = counterfactual_results

        # Step 2: Resistance mechanism identification
        resistance_results = {}
        for gene in top_targets[:10]:
            try:
                res = self.resistance.identify_resistance(dag, gene, disease_node=disease_node)
                resistance_results[gene] = res
            except Exception as e:
                logger.warning(f"Resistance analysis failed for {gene}: {e}")

        result['resistance_analysis'] = resistance_results
        context['resistance_scores'] = {
            g: r for g, r in resistance_results.items()
            if isinstance(r, dict) and 'resistance_score' in r
        }

        # Step 3: Compensation pathway analysis
        compensation_results = {}
        for gene in top_targets[:10]:
            try:
                comp = self.compensation.analyze_compensation(dag, gene, disease_node=disease_node)
                compensation_results[gene] = comp
            except Exception as e:
                logger.warning(f"Compensation analysis failed for {gene}: {e}")

        result['compensation_analysis'] = compensation_results

        logger.info(f"Phase 4 complete: {len(counterfactual_results)} perturbations simulated")
        return result


class Phase5Agent(PhaseAgent):
    """
    Phase 5: Pharma Intervention
    Modules: TargetRanker, DruggabilityScorer, EfficacyPredictor, SafetyAssessor, CombinationAnalyzer
    """

    def __init__(self, config=None):
        super().__init__("Pharma Intervention", 5, config)

    def _init_modules(self):
        from modules.target_ranker import TargetRanker, TargetRankerConfig
        from modules.druggability_scorer import DruggabilityScorer, DruggabilityConfig
        from modules.efficacy_predictor import EfficacyPredictor, EfficacyConfig
        from modules.safety_assessor import SafetyAssessor, SafetyConfig
        from modules.combination_analyzer import CombinationAnalyzer, CombinationConfig

        self.druggability = DruggabilityScorer(config=DruggabilityConfig())
        self.efficacy = EfficacyPredictor(config=EfficacyConfig())
        self.safety = SafetyAssessor(config=SafetyConfig())
        self.ranker = TargetRanker(config=TargetRankerConfig())
        self.combinations = CombinationAnalyzer(config=CombinationConfig())
        logger.info("Phase 5 modules initialized: DruggabilityScorer, EfficacyPredictor, SafetyAssessor, TargetRanker, CombinationAnalyzer")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}
        disease_node = context.get('disease_node', 'Disease_Activity')

        # Step 1: Druggability scoring
        druggability_scores = self.druggability.score_all(dag)
        result['druggability'] = druggability_scores

        # Step 2: Efficacy prediction
        counterfactual_results = context.get('counterfactual_results', {})
        efficacy_scores = self.efficacy.predict_all(
            dag, counterfactual_results=counterfactual_results,
            disease_node=disease_node
        )
        result['efficacy'] = efficacy_scores

        # Step 3: Safety assessment
        safety_scores = self.safety.assess_all(dag, disease_node=disease_node)
        result['safety'] = safety_scores

        # Step 4: Target ranking (integrates all scores)
        resistance_scores = context.get('resistance_scores', {})
        ranking = self.ranker.rank_targets(
            dag,
            druggability_scores=druggability_scores,
            efficacy_scores=efficacy_scores,
            safety_scores=safety_scores,
            resistance_scores=resistance_scores,
            disease_node=disease_node,
        )
        result['target_ranking'] = ranking
        result['ranked_targets'] = ranking.get('ranked_targets', [])

        # Step 5: Combination analysis on top candidates
        top_candidates = [t['gene'] for t in ranking.get('ranked_targets', [])[:10]]
        if len(top_candidates) >= 2:
            combo_result = self.combinations.analyze_combinations(
                dag, top_candidates,
                disease_node=disease_node,
                safety_scores=safety_scores,
                efficacy_scores=efficacy_scores,
            )
            result['combinations'] = combo_result

        logger.info(f"Phase 5 complete: {len(ranking.get('ranked_targets', []))} targets ranked")
        return result


class Phase6Agent(PhaseAgent):
    """
    Phase 6: Comparative Evolution
    Modules: CohortStratifier, DAGComparator, ConservedMotifsIdentifier
    """

    def __init__(self, config=None):
        super().__init__("Comparative Evolution", 6, config)

    def _init_modules(self):
        from modules.cohort_stratifier import CohortStratifier, StratifierConfig
        from modules.dag_comparator import DAGComparator, ComparatorConfig
        from modules.conserved_motifs_identifier import ConservedMotifsIdentifier, MotifsConfig

        self.stratifier = CohortStratifier(config=StratifierConfig())
        self.comparator = DAGComparator(config=ComparatorConfig())
        self.motifs = ConservedMotifsIdentifier(config=MotifsConfig())
        logger.info("Phase 6 modules initialized: CohortStratifier, DAGComparator, ConservedMotifsIdentifier")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}
        patient_dags = context.get('patient_dags', {})

        if not patient_dags:
            result['note'] = "No patient DAGs available — Phase 6 requires patient-level data"
            logger.warning("Phase 6: No patient DAGs available")
            return result

        # Convert list to dict if needed
        if isinstance(patient_dags, list):
            patient_dags = {f"patient_{i}": d for i, d in enumerate(patient_dags)}

        # Step 1: Cohort stratification
        try:
            stratification = self.stratifier.stratify(patient_dags, consensus_dag=dag)
            result['stratification'] = stratification
        except Exception as e:
            logger.warning(f"Stratification failed: {e}")
            result['stratification'] = {"error": str(e)}

        # Step 2: Conserved motifs
        try:
            motifs_result = self.motifs.identify_motifs(patient_dags, consensus_dag=dag)
            result['conserved_motifs'] = motifs_result
        except Exception as e:
            logger.warning(f"Motif identification failed: {e}")
            result['conserved_motifs'] = {"error": str(e)}

        # Step 3: Subgroup comparison (if stratification succeeded)
        if 'subgroups' in result.get('stratification', {}):
            subgroups = result['stratification']['subgroups']
            subgroup_dags = {}
            for sg_id, sg_data in subgroups.items():
                sg_patient_ids = sg_data.get('patient_ids', [])
                sg_dags = [patient_dags[pid] for pid in sg_patient_ids if pid in patient_dags]
                if sg_dags:
                    subgroup_dags[sg_id] = sg_dags

            if len(subgroup_dags) >= 2:
                try:
                    comparison = self.comparator.compare_subgroups(subgroup_dags, consensus_dag=dag)
                    result['subgroup_comparison'] = comparison
                except Exception as e:
                    logger.warning(f"Subgroup comparison failed: {e}")

        logger.info(f"Phase 6 complete")
        return result


class Phase7Agent(PhaseAgent):
    """
    Phase 7: Inspection & LLM Arbitration
    Modules: EvidenceInspector, GapAnalyzer, LLMArbitrator, ResponseFormatter
    """

    def __init__(self, config=None):
        super().__init__("Inspection & LLM Arbitration", 7, config)

    def _init_modules(self):
        from modules.evidence_inspector import EvidenceInspector, InspectorConfig
        from modules.gap_analyzer import GapAnalyzer, GapAnalyzerConfig
        from modules.llm_arbitrator import LLMArbitrator, ArbitratorConfig
        from modules.response_formatter import ResponseFormatter, FormatterConfig

        self.inspector = EvidenceInspector(config=InspectorConfig())
        self.gap_analyzer = GapAnalyzer(config=GapAnalyzerConfig())

        arb_config = ArbitratorConfig()
        if self.config and hasattr(self.config, 'llm_call'):
            arb_config.llm_call = self.config.llm_call
        if self.config and hasattr(self.config, 'llm_model'):
            arb_config.model = self.config.llm_model

        self.arbitrator = LLMArbitrator(config=arb_config)
        self.formatter = ResponseFormatter(config=FormatterConfig())
        logger.info("Phase 7 modules initialized: EvidenceInspector, GapAnalyzer, LLMArbitrator, ResponseFormatter")

    def _execute_core(self, dag: nx.DiGraph, context: Dict) -> Dict:
        result = {}
        disease = context.get('disease_name', 'Disease')

        # Step 1: Evidence inspection
        inspection_report = self.inspector.inspect_dag(dag)
        result['evidence_inspection'] = inspection_report

        # Step 2: Gap analysis
        disease_node = context.get('disease_node', 'Disease_Activity')
        gap_report = self.gap_analyzer.analyze_gaps(dag, disease_node=disease_node)
        result['gap_analysis'] = gap_report

        # Step 3: LLM Arbitration
        arbitration_report = self.arbitrator.arbitrate_dag(
            dag,
            inspection_report=inspection_report,
            gap_report=gap_report
        )
        result['arbitration'] = arbitration_report

        # Step 4: Format final report
        phase_results = context.get('phase_results', {})
        phase_results['phase_7'] = result
        final_report = self.formatter.format_full_report(dag, phase_results, disease=disease)
        result['final_report'] = final_report

        logger.info(f"Phase 7 complete: final report generated")
        return result
