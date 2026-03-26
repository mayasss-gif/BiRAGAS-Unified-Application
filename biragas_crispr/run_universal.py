#!/usr/bin/env python3
"""
BiRAGAS Universal Disease Analyzer
=====================================
Analyze ANY medical disease — from name to causal drug targets.

Usage:
    # Analyze a single disease
    python run_universal.py --disease "Pancreatic Cancer"

    # Analyze with stress test scenarios
    python run_universal.py --disease "SLE" --stress-test

    # Generate scenarios only
    python run_universal.py --disease "Melanoma" --scenarios-only

    # List all supported diseases
    python run_universal.py --list-diseases

    # Batch analyze multiple diseases
    python run_universal.py --batch "SLE,RA,Melanoma,Pancreatic Cancer"

    # Interactive mode
    python run_universal.py --interactive
"""

import argparse
import json
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from universal_agent import UniversalRunner


def main():
    parser = argparse.ArgumentParser(description="BiRAGAS Universal Disease Analyzer")
    parser.add_argument("--disease", type=str, help="Disease name to analyze")
    parser.add_argument("--stress-test", action="store_true", help="Include stress test scenarios")
    parser.add_argument("--scenarios-only", action="store_true", help="Only generate scenarios")
    parser.add_argument("--list-diseases", action="store_true", help="List all supported diseases")
    parser.add_argument("--batch", type=str, help="Comma-separated disease list")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--output-dir", type=str, default="biragas_universal_output")
    parser.add_argument("--max-scenarios", type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    runner = UniversalRunner(output_dir=args.output_dir)

    if args.list_diseases:
        categories = runner.list_supported_diseases()
        print("\nSupported Diseases by Category:")
        print("=" * 50)
        total = 0
        for parent, diseases in sorted(categories.items()):
            print(f"\n  {parent.upper()} ({len(diseases)}):")
            for d in sorted(diseases):
                print(f"    • {d}")
                total += 1
        print(f"\n  Total: {total} diseases")

    elif args.scenarios_only and args.disease:
        scenarios = runner.generate_scenarios(args.disease, max_scenarios=args.max_scenarios)
        print(f"\nStress Test Scenarios for: {args.disease}")
        print("=" * 60)
        for s in scenarios:
            print(f"  [{s['difficulty']:>8s}] {s['disease_a']} vs {s['disease_b']}")
            print(f"           {s['distinction']}")
        print(f"\n  Total: {len(scenarios)} scenarios")

    elif args.batch:
        diseases = [d.strip() for d in args.batch.split(",")]
        reports = runner.batch_analyze(diseases, run_stress_test=args.stress_test)
        print(f"\nBatch analysis complete: {len(reports)} diseases")

    elif args.disease:
        report = runner.analyze_disease(
            args.disease,
            run_stress_test=args.stress_test,
            max_scenarios=args.max_scenarios,
        )

    elif args.interactive:
        _interactive_mode(runner)

    else:
        parser.print_help()


def _interactive_mode(runner):
    print("\n" + "=" * 70)
    print("  BiRAGAS Universal Disease Analyzer — Interactive Mode")
    print("  Analyze ANY disease from name to causal drug targets")
    print("=" * 70)

    while True:
        print("\n  Options:")
        print("  1. Analyze a disease")
        print("  2. Generate stress test scenarios")
        print("  3. List all supported diseases")
        print("  4. Batch analyze multiple diseases")
        print("  5. Exit")

        choice = input("\n  Select [1-5]: ").strip()

        if choice == "1":
            disease = input("  Enter disease name: ").strip()
            if disease:
                stress = input("  Include stress test? (y/n): ").strip().lower() == 'y'
                runner.analyze_disease(disease, run_stress_test=stress)

        elif choice == "2":
            disease = input("  Enter disease name: ").strip()
            if disease:
                scenarios = runner.generate_scenarios(disease)
                print(f"\n  Scenarios for {disease}:")
                for s in scenarios:
                    print(f"    [{s['difficulty']:>8s}] vs {s['disease_b']}: {s['distinction']}")

        elif choice == "3":
            categories = runner.list_supported_diseases()
            total = 0
            for parent, diseases in sorted(categories.items()):
                print(f"\n  {parent.upper()} ({len(diseases)}):")
                for d in sorted(diseases):
                    print(f"    • {d}")
                    total += 1
            print(f"\n  Total: {total} diseases")

        elif choice == "4":
            diseases_str = input("  Enter diseases (comma-separated): ").strip()
            if diseases_str:
                diseases = [d.strip() for d in diseases_str.split(",")]
                runner.batch_analyze(diseases)

        elif choice == "5":
            print("  Goodbye!")
            break


if __name__ == "__main__":
    main()
