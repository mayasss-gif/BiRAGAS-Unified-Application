"""
CRISPRSupervisor — Master Agentic AI Orchestrator
====================================================
Autonomous supervisor that coordinates all CRISPR agents
through the complete pipeline from raw data to ranked targets.

ENTRY POINT: supervisor.run("/path/to/CRISPR/", dag)
EXIT POINT:  Complete results + CSV exports + JSON cache

Pipeline:
    1. DataDiscoveryAgent → finds all CRISPR files automatically
    2. QualityAgent → validates DAG + input data
    3. ScreeningAgent → loads MAGeCK/BAGEL2 results
    4. PerturbSeqAgent → loads h5ad Perturb-seq data
    5. SuperiorACEAgent → 15-stream scoring
    6. MultiKnockoutAgent → 7-method ensemble (19,091 genes)
    7. CombinationAgent → 6-model synergy (40,000 pairs)
    8. SafetyAgent → tissue-specific safety profiles
    9. Export results (CSV, JSON, summary)
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import networkx as nx

logger = logging.getLogger("biragas.crispr.supervisor")


class CRISPRSupervisor:
    """
    Master agent orchestrating the complete CRISPR agentic pipeline.

    Usage:
        supervisor = CRISPRSupervisor()
        report = supervisor.run(
            crispr_dir="/path/to/CRISPR/",
            dag=my_dag,  # from BiRAGAS Phase 1
            output_dir="./crispr_results/"
        )
    """

    def __init__(self):
        from .data_discovery import DataDiscoveryAgent
        from .screening_agent import ScreeningAgent
        from .perturbseq_agent import PerturbSeqAgent
        from .knockout_agent import MultiKnockoutAgent
        from .combination_agent import CombinationAgent
        from .ace_agent import SuperiorACEAgent
        from .safety_agent import SafetyAgent
        from .quality_agent import QualityAgent

        self.discovery_agent = DataDiscoveryAgent()
        self.screening_agent = ScreeningAgent()
        self.perturbseq_agent = PerturbSeqAgent()
        self.quality_agent = QualityAgent()

        # These are initialized during run() with the DAG
        self._knockout_agent = None
        self._combination_agent = None
        self._ace_agent = None
        self._safety_agent = None

        self.results = {}
        logger.info("CRISPRSupervisor initialized with 9 agents")

    def run(
        self,
        crispr_dir: str,
        dag: nx.DiGraph,
        output_dir: str = "./crispr_agentic_output",
        max_knockout_genes: int = 0,
        max_combination_pairs: int = 200,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete autonomous CRISPR pipeline.

        Args:
            crispr_dir: Root directory containing CRISPR data
            dag: BiRAGAS causal DAG (from Phase 1)
            output_dir: Where to save results
            max_knockout_genes: Limit knockouts (0 = all)
            max_combination_pairs: Top N singles for combination
            verbose: Print progress

        Returns:
            Complete pipeline report
        """
        start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "pipeline": "BiRAGAS CRISPR Agentic Engine v2.0",
            "crispr_dir": crispr_dir,
            "stages": {},
            "errors": [],
        }

        if verbose:
            print("\n" + "=" * 70)
            print("  BiRAGAS CRISPR Agentic Multi-Knockout Engine v2.0")
            print("  Autonomous Pipeline: Discovery → Screening → Knockout → Combinations")
            print("=" * 70)

        # ── Stage 1: Data Discovery ──
        if verbose: print("\n▶ Stage 1: Autonomous Data Discovery...")
        try:
            discovery = self.discovery_agent.discover(crispr_dir)
            report["stages"]["discovery"] = {
                "total_files": discovery.total_files,
                "has_screening": discovery.has_screening,
                "has_simulator": discovery.has_simulator,
                "has_targeted": discovery.has_targeted,
                "modes": discovery.modes_available,
                "brunello": bool(discovery.brunello_library),
            }
            self.discovery_agent.export_report(discovery, os.path.join(output_dir, "discovery.json"))
            if verbose: print(f"  ✓ Discovered {discovery.total_files} files | Screening: {discovery.has_screening} | Simulator: {discovery.has_simulator}")
        except Exception as e:
            report["errors"].append(f"Discovery failed: {e}")
            if verbose: print(f"  ✗ Discovery failed: {e}")

        # ── Stage 2: Quality Validation ──
        if verbose: print("\n▶ Stage 2: Quality Validation...")
        try:
            dag_valid, dag_issues = self.quality_agent.validate_dag(dag)
            if not dag_valid:
                if verbose: print(f"  ⟳ Auto-fixing DAG issues...")
                dag, fixes = self.quality_agent.auto_fix(dag)
                if verbose: print(f"  ✓ Applied {len(fixes)} fixes")
            else:
                if verbose: print(f"  ✓ DAG valid: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            report["stages"]["quality"] = {"dag_valid": dag_valid, "issues": dag_issues}
        except Exception as e:
            report["errors"].append(f"Quality check failed: {e}")

        # ── Stage 3: Load Screening Data ──
        if verbose: print("\n▶ Stage 3: Loading Screening Data...")
        try:
            screening_status = self.screening_agent.load_from_discovery(discovery)
            screening_summary = self.screening_agent.get_summary()
            report["stages"]["screening"] = screening_summary
            if verbose:
                print(f"  ✓ {screening_summary['total_genes']} genes loaded")
                print(f"  ✓ {screening_summary['strong_drivers']} strong drivers, {screening_summary['safe_drivers']} safe drivers")
        except Exception as e:
            report["errors"].append(f"Screening failed: {e}")
            if verbose: print(f"  ✗ Screening failed: {e}")

        # ── Stage 4: Load Perturb-seq Data ──
        if verbose: print("\n▶ Stage 4: Loading Perturb-seq Data...")
        try:
            ps_status = self.perturbseq_agent.load_from_discovery(discovery)
            report["stages"]["perturbseq"] = {"loaded": len(ps_status.get("loaded", [])), "status": ps_status}
            if verbose: print(f"  ✓ {len(self.perturbseq_agent.results)} gene signatures loaded")
        except Exception as e:
            report["errors"].append(f"Perturb-seq failed: {e}")
            if verbose: print(f"  ⊘ Perturb-seq: {e}")

        # ── Stage 5: Superior ACE Scoring ──
        if verbose: print("\n▶ Stage 5: Computing 15-Stream Superior ACE Scores...")
        try:
            from .ace_agent import SuperiorACEAgent
            self._ace_agent = SuperiorACEAgent(
                dag=dag,
                screening_results=self.screening_agent.genes,
                perturbseq_results=self.perturbseq_agent.results,
            )
            ace_results = self._ace_agent.score_all()
            n_strong = sum(1 for r in ace_results.values() if r.driver_class == "Strong Driver")
            report["stages"]["ace"] = {"genes_scored": len(ace_results), "strong_drivers": n_strong}
            if verbose: print(f"  ✓ {len(ace_results)} genes scored | {n_strong} Strong Drivers")
        except Exception as e:
            report["errors"].append(f"ACE scoring failed: {e}")
            if verbose: print(f"  ✗ ACE failed: {e}")

        # ── Stage 6: Multi-Knockout Predictions ──
        if verbose: print("\n▶ Stage 6: 7-Method Ensemble Knockout Predictions...")
        try:
            from .knockout_agent import MultiKnockoutAgent
            self._knockout_agent = MultiKnockoutAgent(dag)
            ko_results = self._knockout_agent.predict_all(max_genes=max_knockout_genes)
            report["stages"]["knockout"] = {
                "genes_predicted": len(ko_results),
                "top_5": [{"gene": r.gene, "ensemble": round(r.ensemble, 4), "direction": r.direction} for r in ko_results[:5]],
            }
            self._knockout_agent.export_csv(os.path.join(output_dir, "knockout_results.csv"), top_n=500)
            self._knockout_agent.save_cache(os.path.join(output_dir, "knockout_cache.json"))
            if verbose:
                print(f"  ✓ {len(ko_results)} knockout predictions")
                if ko_results:
                    print(f"  ✓ Top target: {ko_results[0].gene} (ensemble={ko_results[0].ensemble:.4f}, {ko_results[0].direction})")
        except Exception as e:
            report["errors"].append(f"Knockout failed: {e}")
            if verbose: print(f"  ✗ Knockout failed: {e}")
            ko_results = []

        # ── Stage 7: Combination Predictions ──
        if verbose: print("\n▶ Stage 7: 6-Model Combination Synergy Predictions...")
        try:
            from .combination_agent import CombinationAgent
            ko_dict = {r.gene: r for r in ko_results}
            self._combination_agent = CombinationAgent(dag, ko_dict, top_n=max_combination_pairs)
            pairs = self._combination_agent.predict_pairs()
            report["stages"]["combination"] = {
                "pairs_evaluated": len(pairs),
                "top_3": [{"genes": r.genes, "synergy": round(r.synergy_score, 4), "type": r.interaction} for r in pairs[:3]],
            }
            self._combination_agent.export_csv(pairs, os.path.join(output_dir, "combination_results.csv"))
            if verbose:
                print(f"  ✓ {len(pairs)} combinations evaluated")
                if pairs:
                    print(f"  ✓ Best: {'+'.join(pairs[0].genes)} ({pairs[0].interaction}, synergy={pairs[0].synergy_score:.3f})")
        except Exception as e:
            report["errors"].append(f"Combination failed: {e}")
            if verbose: print(f"  ✗ Combination failed: {e}")

        # ── Stage 8: Safety Assessment ──
        if verbose: print("\n▶ Stage 8: Safety Assessment...")
        try:
            from .safety_agent import SafetyAgent
            self._safety_agent = SafetyAgent(dag)
            safety = self._safety_agent.assess_all()
            n_safe = sum(1 for p in safety.values() if p.risk_level in ("Safe", "Acceptable"))
            n_risky = sum(1 for p in safety.values() if p.risk_level == "High Risk")
            report["stages"]["safety"] = {"assessed": len(safety), "safe": n_safe, "high_risk": n_risky}
            if verbose: print(f"  ✓ {len(safety)} genes assessed | {n_safe} safe, {n_risky} high risk")
        except Exception as e:
            report["errors"].append(f"Safety failed: {e}")

        # ── Final: Save Report ──
        duration = time.time() - start
        report["duration_seconds"] = round(duration, 1)
        report["dag"] = {"nodes": dag.number_of_nodes(), "edges": dag.number_of_edges()}

        with open(os.path.join(output_dir, "pipeline_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Export BiRAGAS-compatible files
        try:
            self.screening_agent.export_biragas_csv(os.path.join(output_dir, "perturbation_data"))
        except Exception:
            pass

        if verbose:
            print("\n" + "=" * 70)
            print(f"  PIPELINE COMPLETE: {duration:.1f}s")
            print(f"  Results: {output_dir}")
            print(f"  Errors: {len(report['errors'])}")
            print("=" * 70 + "\n")

        self.results = report
        return report
