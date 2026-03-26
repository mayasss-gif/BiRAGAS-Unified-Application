"""
UnifiedCRISPRPipeline — Master Integration Entry Point
=========================================================
Single command runs everything:
    BiRAGAS 23 modules + CRISPR v1.0 + CRISPR v2.0 + Mega Engine

Usage:
    from integration import UnifiedCRISPRPipeline

    pipeline = UnifiedCRISPRPipeline()
    report = pipeline.run(
        data_dir="/path/to/transcriptomic/data",
        crispr_dir="/path/to/CRISPR/",
        disease_name="Melanoma",
        output_dir="./unified_output/"
    )
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import networkx as nx

logger = logging.getLogger("biragas.integration.unified")


class UnifiedCRISPRPipeline:
    """
    Master pipeline integrating ALL systems:

    System A: BiRAGAS Causality Framework (23 modules, 7 phases)
    System B: CRISPR MultiKnockout v1.0 (5-method, 12-stream)
    System C: CRISPR Agentic v2.0 (7-method, 15-stream, 10 agents)
    System D: Mega-Knockout Engine (177K, sparse matrix O(1))

    Pipeline:
    1. BiRAGAS Phase 1-3 → build + validate causal DAG
    2. Engine Selector → pick optimal CRISPR engine
    3. CRISPR Knockout → predict all knockouts
    4. CRISPR Combinations → predict synergistic pairs
    5. Result Merger → unify cross-engine results
    6. BiRAGAS Phase 5-7 → rank + report with CRISPR evidence
    7. Export → CSV + JSON + report
    """

    def __init__(self):
        from .biragas_bridge import BiRAGASBridge
        from .engine_selector import EngineSelector
        from .result_merger import ResultMerger

        self.bridge = BiRAGASBridge()
        self.selector = EngineSelector()
        self.merger = ResultMerger()

    def run(
        self,
        data_dir: str = "",
        crispr_dir: str = "",
        disease_name: str = "Disease",
        dag: Optional[nx.DiGraph] = None,
        engine: str = "auto",
        output_dir: str = "./unified_output",
        run_all_engines: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the complete unified pipeline.

        Args:
            data_dir: BiRAGAS transcriptomic data directory
            crispr_dir: CRISPR data directory
            disease_name: Disease name
            dag: Pre-built DAG (skip Phase 1-3 if provided)
            engine: "v1", "v2", "mega", or "auto"
            output_dir: Output directory
            run_all_engines: Run ALL engines and merge (slower but most comprehensive)
            verbose: Print progress

        Returns:
            Complete integration report
        """
        start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "pipeline": "BiRAGAS × CRISPR Unified Integration v1.0",
            "disease": disease_name,
            "stages": {},
            "errors": [],
        }

        if verbose:
            print("\n" + "=" * 70)
            print("  BiRAGAS × CRISPR UNIFIED INTEGRATION PIPELINE")
            print(f"  Disease: {disease_name}")
            print(f"  Engine: {engine} | All engines: {run_all_engines}")
            print("=" * 70)

        # ── Stage 1: Build DAG (BiRAGAS Phases 1-3) ──
        if dag is None and data_dir:
            if verbose: print("\n▶ Stage 1: BiRAGAS Phases 1-3 (Build + Validate DAG)...")
            try:
                dag = self.bridge.run_biragas_phases_1_to_3(data_dir, disease_name)
                report["stages"]["biragas_dag"] = {"nodes": dag.number_of_nodes(), "edges": dag.number_of_edges()}
                if verbose: print(f"  ✓ DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            except Exception as e:
                report["errors"].append(f"DAG construction failed: {e}")
                if verbose: print(f"  ✗ Failed: {e}")
                return report
        elif dag is not None:
            if verbose: print(f"\n▶ Stage 1: Using pre-built DAG ({dag.number_of_nodes()} nodes)")
            report["stages"]["biragas_dag"] = {"nodes": dag.number_of_nodes(), "edges": dag.number_of_edges(), "pre_built": True}

        if dag is None:
            report["errors"].append("No DAG available — provide data_dir or dag parameter")
            return report

        # ── Stage 2: Select Engine ──
        selected_engine, reason = self.selector.select(dag, crispr_dir, force=None if engine == "auto" else engine)
        report["stages"]["engine_selection"] = {"selected": selected_engine, "reason": reason}
        if verbose: print(f"\n▶ Stage 2: Engine Selection → {selected_engine}\n  {reason}")

        # ── Stage 3: Run CRISPR Knockouts ──
        if verbose: print(f"\n▶ Stage 3: CRISPR Knockout Predictions...")

        all_ko_results = {}

        if run_all_engines:
            # Run ALL engines for maximum coverage
            for eng in ["v1", "v2", "mega"]:
                if verbose: print(f"  Running engine: {eng}...")
                try:
                    eng_output = os.path.join(output_dir, f"engine_{eng}")
                    results = self.bridge.run_crispr_knockout(dag, crispr_dir, eng, eng_output)
                    if 'ko_results' in results:
                        all_ko_results[eng] = results['ko_results']
                    report["stages"][f"knockout_{eng}"] = {
                        "knockouts": results.get('knockouts', len(results.get('ko_results', []))),
                        "engine": eng,
                    }
                    if verbose: print(f"    ✓ {eng}: {results.get('knockouts', 0)} predictions")
                except Exception as e:
                    if verbose: print(f"    ✗ {eng}: {e}")
                    report["errors"].append(f"Engine {eng}: {e}")
        else:
            # Run selected engine only
            try:
                results = self.bridge.run_crispr_knockout(dag, crispr_dir, selected_engine, output_dir)
                if 'ko_results' in results:
                    all_ko_results[selected_engine] = results['ko_results']
                report["stages"]["knockout"] = {
                    "knockouts": results.get('knockouts', len(results.get('ko_results', []))),
                    "engine": selected_engine,
                }
                if verbose: print(f"  ✓ {results.get('knockouts', 0)} knockout predictions")
            except Exception as e:
                report["errors"].append(f"Knockout failed: {e}")
                if verbose: print(f"  ✗ Failed: {e}")

        # ── Stage 4: Merge Results ──
        if verbose: print(f"\n▶ Stage 4: Merging Results from {len(all_ko_results)} engine(s)...")
        try:
            unified = self.merger.merge_knockout_results(all_ko_results)
            report["stages"]["merge"] = {
                "total_genes": len(unified),
                "multi_engine": sum(1 for u in unified if u.get('n_engines', 0) > 1),
                "top_5": unified[:5],
            }
            self.merger.export_unified_csv(unified, os.path.join(output_dir, "unified_knockouts.csv"))
            if verbose:
                print(f"  ✓ {len(unified)} genes merged")
                if unified:
                    print(f"  ✓ Top: {unified[0]['gene']} (consensus={unified[0]['consensus_score']:.4f})")
        except Exception as e:
            report["errors"].append(f"Merge failed: {e}")
            unified = []

        # ── Stage 5: BiRAGAS Phases 5-7 with CRISPR ──
        if verbose: print(f"\n▶ Stage 5: BiRAGAS Phases 5-7 (Rank + Report with CRISPR)...")
        try:
            # Package results for bridge
            crispr_package = {
                'ko_results': all_ko_results.get(selected_engine, []),
                'knockouts': len(unified),
            }
            final_report = self.bridge.run_biragas_phases_5_to_7(dag, crispr_package, disease_name)
            report["stages"]["biragas_report"] = {"status": "complete"}
            if verbose: print(f"  ✓ Final report generated")
        except Exception as e:
            report["errors"].append(f"Phases 5-7 failed: {e}")
            if verbose: print(f"  ✗ Failed: {e}")

        # ── Stage 6: Save Everything ──
        duration = time.time() - start
        report["duration_seconds"] = round(duration, 1)
        report["output_dir"] = output_dir

        with open(os.path.join(output_dir, "unified_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  UNIFIED PIPELINE COMPLETE: {duration:.1f}s")
            print(f"  {len(unified)} genes ranked across {len(all_ko_results)} engine(s)")
            print(f"  Results: {output_dir}")
            print(f"  Errors: {len(report['errors'])}")
            print(f"{'=' * 70}\n")

        return report
