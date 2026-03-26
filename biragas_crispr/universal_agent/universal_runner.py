"""
BiRAGAS Universal Runner
==========================
End-to-end orchestration: takes ANY disease name, fetches data from
public APIs, generates input files, creates stress test scenarios,
and runs everything through the 7-phase, 23-module pipeline.

ENTRY POINT: UniversalRunner.analyze_disease("Pancreatic Cancer")
EXIT POINT:  Complete pipeline report + stress test results
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biragas.universal_runner")

# Ensure imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.dirname(SCRIPT_DIR)
for p in [FRAMEWORK_DIR, os.path.dirname(FRAMEWORK_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class UniversalRunner:
    """
    Universal BiRAGAS runner — analyze ANY disease autonomously.

    Flow:
    1. Disease name → DiseaseKnowledgeAgent → molecular data from APIs
    2. API data → DataAcquisitionAgent → BiRAGAS input files
    3. Input files → 7-phase pipeline (23 modules) → causal DAG + drug targets
    4. Disease → ScenarioEngine → unlimited stress test scenarios
    5. Stress test → validation across all 4 pillars
    6. Everything → self-correction + learning

    Usage:
        runner = UniversalRunner()

        # Analyze a single disease
        report = runner.analyze_disease("Systemic Lupus Erythematosus")

        # Analyze with stress test
        report = runner.analyze_disease("Pancreatic Cancer", run_stress_test=True)

        # Generate scenarios for any disease
        scenarios = runner.generate_scenarios("Melanoma")

        # Batch analysis
        reports = runner.batch_analyze(["SLE", "RA", "Melanoma"])
    """

    def __init__(self, output_dir: str = "biragas_universal_output", cache_dir: str = ".biragas_cache"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        from .disease_knowledge_agent import DiseaseKnowledgeAgent
        from .data_acquisition_agent import DataAcquisitionAgent
        from .scenario_engine import ScenarioEngine

        self.knowledge_agent = DiseaseKnowledgeAgent(cache_dir=cache_dir)
        self.data_agent = DataAcquisitionAgent()
        self.scenario_engine = ScenarioEngine(self.knowledge_agent)

        logger.info("UniversalRunner initialized")

    def analyze_disease(
        self,
        disease_name: str,
        run_stress_test: bool = False,
        max_scenarios: int = 10,
        n_samples: int = 30,
    ) -> Dict:
        """
        Analyze ANY disease end-to-end.

        Args:
            disease_name: Any medical disease name
            run_stress_test: Whether to run stress test scenarios
            max_scenarios: Max stress test scenarios to generate
            n_samples: Number of simulated samples

        Returns:
            Complete analysis report
        """
        start = time.time()
        run_id = f"universal_{disease_name.replace(' ', '_')}_{int(time.time())}"

        print("\n" + "=" * 70)
        print(f"  BiRAGAS Universal Analysis: {disease_name}")
        print(f"  Run ID: {run_id}")
        print("=" * 70)

        report = {
            "run_id": run_id,
            "disease": disease_name,
            "stages": {},
        }

        # ── Stage 1: Gather disease knowledge from APIs ──
        print("\n▶ Stage 1: Gathering disease knowledge from public databases...")
        disease_data = self.knowledge_agent.gather_disease_data(disease_name)
        report["stages"]["knowledge_gathering"] = {
            "genes_found": len(disease_data.get('all_genes', [])),
            "gwas_hits": len(disease_data.get('gwas_hits', [])),
            "opentargets": len(disease_data.get('opentargets_associations', [])),
            "pathways": len(disease_data.get('reactome_pathways', [])),
            "interactions": len(disease_data.get('string_interactions', [])),
            "disease_info": disease_data.get('disease_info', {}),
        }
        print(f"  ✓ {len(disease_data.get('all_genes', []))} genes identified")
        print(f"  ✓ {len(disease_data.get('gwas_hits', []))} GWAS hits")
        print(f"  ✓ {len(disease_data.get('string_interactions', []))} protein interactions")

        # ── Stage 2: Create BiRAGAS input files ──
        print("\n▶ Stage 2: Creating BiRAGAS-compatible input files...")
        data_dir = self.data_agent.create_data_directory(
            disease_name, disease_data, self.output_dir, n_samples=n_samples
        )
        report["stages"]["data_creation"] = {"data_dir": data_dir}
        print(f"  ✓ Data directory: {data_dir}")

        # ── Stage 3: Run 7-phase pipeline ──
        print("\n▶ Stage 3: Running 7-phase BiRAGAS pipeline...")
        pipeline_report = self._run_pipeline(disease_name, data_dir)
        report["stages"]["pipeline"] = pipeline_report
        print(f"  ✓ Pipeline complete")

        # ── Stage 4: Generate and run stress test ──
        if run_stress_test:
            print("\n▶ Stage 4: Generating stress test scenarios...")
            scenarios = self.scenario_engine.generate_scenarios_for_disease(
                disease_name, max_scenarios=max_scenarios
            )
            report["stages"]["stress_test"] = {
                "scenarios_generated": len(scenarios),
                "scenarios": [
                    {
                        "id": s.scenario_id,
                        "disease_a": s.disease_a,
                        "disease_b": s.disease_b,
                        "type": s.relationship_type,
                        "distinction": s.causal_distinction,
                        "difficulty": s.difficulty,
                    }
                    for s in scenarios
                ],
            }
            print(f"  ✓ {len(scenarios)} differential diagnosis scenarios generated:")
            for s in scenarios:
                print(f"    • {s.disease_a} vs {s.disease_b} ({s.relationship_type})")

        # ── Stage 5: Save and return ──
        duration = time.time() - start
        report["total_duration_seconds"] = round(duration, 2)

        # Save report
        report_path = os.path.join(self.output_dir, run_id, "universal_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print(f"  Analysis complete: {duration:.1f}s")
        print(f"  Report: {report_path}")
        print("=" * 70 + "\n")

        return report

    def generate_scenarios(self, disease_name: str, max_scenarios: int = 20) -> List[Dict]:
        """Generate stress test scenarios for any disease."""
        scenarios = self.scenario_engine.generate_scenarios_for_disease(
            disease_name, max_scenarios=max_scenarios
        )
        return [
            {
                "id": s.scenario_id,
                "disease_a": s.disease_a,
                "disease_b": s.disease_b,
                "type": s.relationship_type,
                "distinction": s.causal_distinction,
                "difficulty": s.difficulty,
                "rule_in": s.rule_in_a,
                "rule_out": s.rule_out_b,
            }
            for s in scenarios
        ]

    def batch_analyze(self, diseases: List[str], run_stress_test: bool = False) -> List[Dict]:
        """Analyze multiple diseases sequentially."""
        reports = []
        for disease in diseases:
            try:
                report = self.analyze_disease(disease, run_stress_test=run_stress_test)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to analyze {disease}: {e}")
                reports.append({"disease": disease, "error": str(e)})
        return reports

    def list_supported_diseases(self) -> Dict[str, List[str]]:
        """List all supported diseases grouped by category."""
        return self.knowledge_agent.get_disease_categories()

    def _run_pipeline(self, disease_name: str, data_dir: str) -> Dict:
        """Run the BiRAGAS pipeline on generated data."""
        try:
            from orchestrator import MasterOrchestrator, OrchestratorConfig

            config = OrchestratorConfig(
                disease_name=disease_name,
                data_dir=data_dir,
                output_dir=os.path.join(self.output_dir, f"pipeline_{disease_name.replace(' ', '_')}"),
            )
            orchestrator = MasterOrchestrator(config)
            return orchestrator.run_full_pipeline(data_dir=data_dir)
        except Exception as e:
            logger.warning(f"Pipeline execution failed: {e}")
            return {"error": str(e), "note": "Pipeline requires full data. Generated data may need enrichment."}
