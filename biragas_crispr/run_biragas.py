#!/usr/bin/env python3
"""
BiRAGAS — Run the Full Causality Inference Pipeline
=====================================================
Ayass Bioscience LLC | Version 2.0.0

Entry point for the BiRAGAS 7-phase, 23-module Causality Framework
with autonomous orchestration, self-correction, and stress test validation.

Usage:
    # Full pipeline with data directory
    python run_biragas.py --data-dir /path/to/data --disease "SLE" --output-dir ./output

    # Stress test only (no data required)
    python run_biragas.py --stress-test-only

    # Full pipeline + stress test
    python run_biragas.py --data-dir /path/to/data --disease "SLE" --run-stress-test

    # Custom configuration
    python run_biragas.py --config config.json
"""

import argparse
import json
import logging
import os
import sys

# Ensure the framework is importable
FRAMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)
PARENT_DIR = os.path.dirname(FRAMEWORK_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from orchestrator import MasterOrchestrator, OrchestratorConfig


def main():
    parser = argparse.ArgumentParser(
        description="BiRAGAS Causality Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_biragas.py --stress-test-only
  python run_biragas.py --data-dir ./data --disease "SLE"
  python run_biragas.py --data-dir ./data --disease "Pancreatic Cancer" --run-stress-test
        """
    )

    parser.add_argument("--data-dir", type=str, default="",
                        help="Path to input data directory")
    parser.add_argument("--disease", type=str, default="Disease",
                        help="Disease name (e.g., 'SLE', 'Pancreatic Cancer')")
    parser.add_argument("--output-dir", type=str, default="biragas_output",
                        help="Output directory for results")
    parser.add_argument("--config", type=str, default="",
                        help="Path to JSON configuration file")
    parser.add_argument("--stress-test-only", action="store_true",
                        help="Run only the 17-scenario stress test")
    parser.add_argument("--run-stress-test", action="store_true",
                        help="Run stress test after full pipeline")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Build configuration
    if args.config:
        config = OrchestratorConfig.from_json(args.config)
    else:
        config = OrchestratorConfig()

    config.disease_name = args.disease
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.verbose = args.verbose
    config.log_level = args.log_level

    # Banner
    print()
    print("=" * 70)
    print("  BiRAGAS — Biological Retrieval-Augmented Generation Assessment System")
    print("  Causality Inference Framework v2.0.0")
    print("  Ayass Bioscience LLC")
    print("=" * 70)
    print(f"  Disease:    {config.disease_name}")
    print(f"  Run ID:     {config.run_id}")
    print(f"  Output:     {config.output_dir}")
    print("=" * 70)
    print()

    # Initialize orchestrator
    orchestrator = MasterOrchestrator(config)

    if args.stress_test_only:
        # Run stress test only
        print("Running 17-scenario differential diagnosis stress test...")
        print()
        report = orchestrator.run_stress_test()
        _print_stress_test_report(report)

    else:
        # Run full pipeline
        if not args.data_dir:
            print("WARNING: No --data-dir provided. Pipeline will fail at Phase 1.")
            print("Use --stress-test-only to run without data.")
            print()

        print(f"Running full 7-phase pipeline for {config.disease_name}...")
        print()
        report = orchestrator.run_full_pipeline(data_dir=args.data_dir)
        _print_pipeline_report(report)

        if args.run_stress_test:
            print()
            print("Running stress test...")
            stress_report = orchestrator.run_stress_test()
            _print_stress_test_report(stress_report)

    # Print recommendations
    recommendations = orchestrator.get_recommendations()
    if recommendations:
        print()
        print("Learning Engine Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print()
    print("Done.")


def _print_pipeline_report(report: dict):
    """Pretty-print pipeline report."""
    print()
    print("-" * 50)
    print("PIPELINE REPORT")
    print("-" * 50)
    print(f"  Run ID:     {report.get('run_id', 'N/A')}")
    print(f"  Disease:    {report.get('disease', 'N/A')}")
    print(f"  Duration:   {report.get('total_duration_seconds', 0):.1f}s")

    dag = report.get('dag_summary', {})
    print(f"  DAG Nodes:  {dag.get('nodes', 0)}")
    print(f"  DAG Edges:  {dag.get('edges', 0)}")
    print(f"  Layers:     {dag.get('layers', {})}")

    phases = report.get('phase_results', {})
    print(f"  Phases:     {len(phases)} completed")
    for phase, info in phases.items():
        status = "OK" if info.get('completed') else "FAIL"
        print(f"    {phase}: {status}")

    corrections = report.get('self_correction_summary', {})
    print(f"  Corrections: {corrections.get('total_corrections', 0)}")
    print("-" * 50)


def _print_stress_test_report(report: dict):
    """Pretty-print stress test report."""
    print()
    print("-" * 70)
    print("STRESS TEST REPORT — 17 Differential Diagnosis Scenarios")
    print("-" * 70)

    overall = "PASS" if report.get('overall_passed') else "FAIL"
    passed = report.get('passed', 0)
    total = report.get('total_scenarios', 0)
    print(f"  Overall: {overall} ({passed}/{total} scenarios)")
    print()

    pillars = report.get('pillar_statistics', {})
    print(f"  Pillar Results (out of {total}):")
    print(f"    DAG Topology Separation:      {pillars.get('topology_pass', 0)}/{total}")
    print(f"    Genetic Anchor Separation:    {pillars.get('genetic_pass', 0)}/{total}")
    print(f"    Propagation Separation:       {pillars.get('propagation_pass', 0)}/{total}")
    print(f"    Drug Target Separation:       {pillars.get('targets_pass', 0)}/{total}")
    print()

    for scenario in report.get('scenarios', []):
        status = "PASS" if scenario['passed'] else "FAIL"
        print(
            f"  [{status}] Scenario {str(scenario['id']):>2s}: "
            f"{scenario['disease_a']} vs {scenario['disease_b']}"
        )
        if not scenario['passed']:
            pillars_detail = scenario.get('pillars', {})
            failed_pillars = [k for k, v in pillars_detail.items() if not v]
            if failed_pillars:
                print(f"         Failed pillars: {', '.join(failed_pillars)}")

    print("-" * 70)


if __name__ == "__main__":
    main()
