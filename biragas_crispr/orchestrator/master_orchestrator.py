"""
BiRAGAS Master Orchestrator
=============================
Central coordinator for the 7-phase, 23-module BiRAGAS Causality Framework.
Implements an agentic AI ecosystem with autonomous execution, self-correction,
self-education, and stress test validation.

Architecture:
    MasterOrchestrator
    ├── Phase Agents (7) — autonomous phase execution
    ├── SelfCorrector — DAG integrity and error recovery
    ├── LearningEngine — performance tracking and adaptation
    └── StressTestAgent — differential diagnosis validation

Usage:
    from orchestrator import MasterOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(
        disease_name="SLE",
        data_dir="/path/to/data",
        output_dir="/path/to/output",
    )
    orchestrator = MasterOrchestrator(config)
    report = orchestrator.run_full_pipeline()
"""

import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from .config import OrchestratorConfig
from .phase_agents import (
    Phase1Agent,
    Phase2Agent,
    Phase3Agent,
    Phase4Agent,
    Phase5Agent,
    Phase6Agent,
    Phase7Agent,
)
from .self_corrector import SelfCorrector
from .learning_engine import LearningEngine
from .stress_test_agent import StressTestAgent

logger = logging.getLogger("biragas.orchestrator")


class MasterOrchestrator:
    """
    Central orchestrator for the BiRAGAS Causality Inference Framework.

    Coordinates 7 phase agents, each containing specialized modules, through
    the complete causal inference pipeline from raw multi-modal data to
    validated, arbitrated, FDA-aligned clinical reports.

    Features:
    - Autonomous 7-phase pipeline execution
    - Self-correcting DAG integrity maintenance
    - Self-educating performance optimization
    - 17-scenario stress test validation
    - Comprehensive error handling and retry logic
    - Intermediate result persistence
    - Modular phase enable/disable
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._setup_logging()

        # Ensure modules are importable
        modules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules")
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        for path in [framework_dir, os.path.dirname(framework_dir)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        # Initialize sub-systems
        self.self_corrector = SelfCorrector(config=self.config.self_correction)
        self.learning_engine = LearningEngine(config=self.config.learning)
        self.stress_test_agent = StressTestAgent(config=self.config.stress_test)

        # Initialize phase agents
        self.phases: Dict[str, Any] = {
            "phase_1": Phase1Agent(config=self.config),
            "phase_2": Phase2Agent(config=self.config),
            "phase_3": Phase3Agent(config=self.config),
            "phase_4": Phase4Agent(config=self.config),
            "phase_5": Phase5Agent(config=self.config),
            "phase_6": Phase6Agent(config=self.config),
            "phase_7": Phase7Agent(config=self.config),
        }

        # Pipeline state
        self.dag: nx.DiGraph = nx.DiGraph()
        self.context: Dict[str, Any] = {}
        self.phase_results: Dict[str, Dict] = {}
        self._pipeline_start_time: float = 0

        logger.info(f"MasterOrchestrator initialized: run_id={self.config.run_id}")

    def run_full_pipeline(
        self,
        data_dir: Optional[str] = None,
        data_packet: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute the complete 7-phase BiRAGAS pipeline.

        Args:
            data_dir: Path to input data directory (used by Phase 1)
            data_packet: Pre-loaded data packet (alternative to data_dir)

        Returns:
            Complete pipeline report with all phase results
        """
        self._pipeline_start_time = time.time()
        self.learning_engine.start_run(
            self.config.run_id,
            disease=self.config.disease_name,
        )

        # Set up context
        self.context = {
            'data_dir': data_dir or self.config.data_dir,
            'data_packet': data_packet,
            'disease_name': self.config.disease_name,
            'disease_node': self.config.disease_node,
            'phase_results': self.phase_results,
        }

        # Reset state
        self.dag = nx.DiGraph()
        self.phase_results = {}
        self.context['phase_results'] = self.phase_results

        logger.info("=" * 70)
        logger.info(f"BiRAGAS Pipeline Starting: {self.config.disease_name}")
        logger.info(f"Run ID: {self.config.run_id}")
        logger.info("=" * 70)

        # Execute phases sequentially
        phase_order = [
            "phase_1", "phase_2", "phase_3",
            "phase_4", "phase_5", "phase_6", "phase_7"
        ]

        for phase_key in phase_order:
            phase_config = self.config.phase_configs.get(phase_key)
            if phase_config and not phase_config.enabled:
                logger.info(f"Skipping {phase_key} (disabled)")
                continue

            success = self._execute_phase(phase_key)

            if not success:
                logger.error(f"{phase_key} failed — pipeline halted")
                break

        # Post-pipeline: record metrics and generate summary
        self.learning_engine.record_dag_metrics(self.dag)
        total_corrections = len(self.self_corrector.corrections)
        self.learning_engine.record_corrections(total_corrections)

        total_duration = time.time() - self._pipeline_start_time

        # Compile final report
        report = self._compile_pipeline_report(total_duration)

        # Save outputs
        if self.config.output_dir:
            self._save_outputs(report)

        self.learning_engine.end_run(notes=f"Completed in {total_duration:.1f}s")

        logger.info("=" * 70)
        logger.info(f"Pipeline Complete: {total_duration:.1f}s")
        logger.info(f"DAG: {self.dag.number_of_nodes()} nodes, {self.dag.number_of_edges()} edges")
        logger.info(f"Corrections: {total_corrections}")
        logger.info("=" * 70)

        return report

    def run_stress_test(
        self,
        disease_dags: Optional[Dict[str, nx.DiGraph]] = None,
    ) -> Dict:
        """
        Run the 17-scenario differential diagnosis stress test.

        Args:
            disease_dags: Pre-built DAGs keyed by disease name.
                          If None, synthetic DAGs are used.

        Returns:
            Stress test report with per-scenario and overall results
        """
        logger.info("=" * 70)
        logger.info("BiRAGAS Stress Test: 17 Differential Diagnosis Scenarios")
        logger.info("=" * 70)

        report = self.stress_test_agent.run_all_scenarios(
            disease_dags=disease_dags,
            phase_results=self.phase_results if self.phase_results else None,
        )

        overall = "PASS" if report.get('overall_passed') else "FAIL"
        passed = report.get('passed', 0)
        total = report.get('total_scenarios', 0)

        logger.info(f"Stress Test Result: {overall} ({passed}/{total} scenarios passed)")
        self.learning_engine.record_stress_test(report.get('overall_passed', False))

        return report

    def run_phase(self, phase_key: str, dag: Optional[nx.DiGraph] = None, context: Optional[Dict] = None) -> Dict:
        """Run a single phase independently."""
        if dag:
            self.dag = dag
        if context:
            self.context.update(context)

        self._execute_phase(phase_key)
        return self.phase_results.get(phase_key, {})

    def _execute_phase(self, phase_key: str) -> bool:
        """Execute a single phase with self-correction."""
        agent = self.phases.get(phase_key)
        if not agent:
            logger.error(f"Unknown phase: {phase_key}")
            return False

        phase_config = self.config.phase_configs.get(phase_key)
        max_retries = phase_config.max_retries if phase_config else 3

        self.learning_engine.start_phase(phase_key)

        # Execute phase
        result = agent.execute(self.dag, self.context, max_retries=max_retries)

        if result['success']:
            self.phase_results[phase_key] = result['result']

            # Self-correction: validate and auto-correct DAG
            is_valid, issues = self.self_corrector.validate_dag(self.dag, phase=phase_key)
            if not is_valid:
                logger.warning(f"{phase_key}: DAG validation issues detected, auto-correcting...")
                self.dag, corrections = self.self_corrector.auto_correct_dag(self.dag, phase=phase_key)
                if corrections:
                    logger.info(f"{phase_key}: Applied {len(corrections)} corrections")

            # Phase output validation
            output_valid, output_issues = self.self_corrector.validate_phase_output(
                phase_key, result['result'], self.dag
            )
            if not output_valid:
                logger.warning(f"{phase_key}: Output validation issues: {output_issues}")

            # Save intermediate results
            if self.config.save_intermediate and self.config.output_dir:
                self._save_intermediate(phase_key, result['result'])

            self.learning_engine.end_phase(phase_key, success=True)
            logger.info(
                f"{phase_key} [{agent.phase_name}]: SUCCESS ({result['duration']:.1f}s, "
                f"{result['attempts']} attempt(s))"
            )
            return True

        else:
            self.learning_engine.end_phase(phase_key, success=False)
            self.self_corrector.record_error_pattern(phase_key, "execution_failure")
            logger.error(f"{phase_key} [{agent.phase_name}]: FAILED after {result['attempts']} attempts")
            for err in result['errors']:
                logger.error(f"  Error: {err.get('error', 'unknown')}")
            return False

    def _compile_pipeline_report(self, total_duration: float) -> Dict:
        """Compile the final pipeline report."""
        report = {
            "run_id": self.config.run_id,
            "disease": self.config.disease_name,
            "total_duration_seconds": round(total_duration, 2),
            "dag_summary": {
                "nodes": self.dag.number_of_nodes(),
                "edges": self.dag.number_of_edges(),
                "layers": self._count_layers(),
            },
            "phase_results": {},
            "self_correction_summary": self.self_corrector.get_correction_summary(),
            "learning_summary": self.learning_engine.get_performance_summary(),
        }

        for phase_key, result in self.phase_results.items():
            report["phase_results"][phase_key] = {
                "completed": True,
                "summary_keys": list(result.keys()) if isinstance(result, dict) else [],
            }

        return report

    def _count_layers(self) -> Dict[str, int]:
        """Count nodes per layer."""
        layers = {}
        for n in self.dag.nodes():
            layer = self.dag.nodes[n].get('layer', 'unknown')
            layers[layer] = layers.get(layer, 0) + 1
        return layers

    def _save_intermediate(self, phase_key: str, result: Dict):
        """Save intermediate phase results to disk."""
        try:
            out_dir = os.path.join(self.config.output_dir, self.config.run_id, phase_key)
            os.makedirs(out_dir, exist_ok=True)

            # Save result summary (skip non-serializable objects)
            safe_result = {}
            for k, v in result.items():
                if isinstance(v, (str, int, float, bool, list)):
                    safe_result[k] = v
                elif isinstance(v, dict):
                    safe_result[k] = str(v)[:500]
                else:
                    safe_result[k] = f"<{type(v).__name__}>"

            filepath = os.path.join(out_dir, "result_summary.json")
            with open(filepath, 'w') as f:
                json.dump(safe_result, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save intermediate results for {phase_key}: {e}")

    def _save_outputs(self, report: Dict):
        """Save final pipeline outputs."""
        try:
            out_dir = os.path.join(self.config.output_dir, self.config.run_id)
            os.makedirs(out_dir, exist_ok=True)

            # Save report
            report_path = os.path.join(out_dir, "pipeline_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Save DAG
            dag_path = os.path.join(out_dir, "consensus_dag.json")
            from networkx.readwrite import json_graph
            dag_data = json_graph.node_link_data(self.dag)
            with open(dag_path, 'w') as f:
                json.dump(dag_data, f, indent=2, default=str)

            # Save config
            config_path = os.path.join(out_dir, "config.json")
            self.config.save_json(config_path)

            # Save correction log
            correction_path = os.path.join(out_dir, "corrections.json")
            with open(correction_path, 'w') as f:
                json.dump(self.self_corrector.get_correction_summary(), f, indent=2)

            logger.info(f"Outputs saved to: {out_dir}")

        except Exception as e:
            logger.warning(f"Failed to save outputs: {e}")

    def _setup_logging(self):
        """Configure logging for the orchestrator."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S',
        )

    # === Convenience Methods ===

    def get_dag(self) -> nx.DiGraph:
        """Return the current DAG."""
        return self.dag

    def get_phase_result(self, phase_key: str) -> Dict:
        """Return results for a specific phase."""
        return self.phase_results.get(phase_key, {})

    def get_recommendations(self) -> List[str]:
        """Get learning engine recommendations."""
        return self.learning_engine.get_recommendations()

    def get_correction_log(self) -> Dict:
        """Get self-correction summary."""
        return self.self_corrector.get_correction_summary()
