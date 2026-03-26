#!/usr/bin/env python3
"""
BiRAGAS Merged Causality System — Entry Point
================================================
COMPLETE SYSTEM: LLM Intelligence + 23 Implemented Science Modules

This merges:
- Causality_agent's brain (intent, literature, clarification, eligibility)
- 4-Layer BiRAGAS's body (23 working science modules)

Usage:
    # Interactive mode (recommended)
    python run.py

    # With a specific query
    python run.py --query "Find causal drivers of Pancreatic Cancer"

    # With data files
    python run.py --query "Find causal drivers of SLE" --files data/*.csv

    # Skip literature search
    python run.py --query "Find causal drivers of Melanoma" --skip-lit

    # Non-interactive
    python run.py --query "What causes RA?" --no-interact
"""

from __future__ import annotations
import os
import sys

# Ensure all modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.dirname(SCRIPT_DIR)
for p in [SCRIPT_DIR, FRAMEWORK_DIR, os.path.join(FRAMEWORK_DIR, "modules")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import SupervisorAgent
from cli import save_result, _run_clarification


# ============================================================================
# CONFIGURATION — Edit these for your analysis
# ============================================================================
QUERY = "Find the causal drivers of Systemic Lupus Erythematosus in this cohort"
FILES = []  # Add file paths here, or pass via --files
SKIP_LITERATURE = False
NO_INTERACT = False
OUTPUT_DIR = "./merged_results"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BiRAGAS Merged Causality System v3.0")
    parser.add_argument("--query", type=str, default=QUERY, help="Research query")
    parser.add_argument("--files", nargs="*", default=FILES, help="Input data files")
    parser.add_argument("--skip-lit", action="store_true", default=SKIP_LITERATURE)
    parser.add_argument("--no-interact", action="store_true", default=NO_INTERACT)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  BiRAGAS Merged Causality System v3.0")
    print("  Ayass Bioscience LLC")
    print("  LLM Intelligence + 23 Implemented Science Modules")
    print("=" * 70)
    print(f"  Query: {args.query}")
    print(f"  Files: {len(args.files)} provided")
    print(f"  Literature: {'SKIP' if args.skip_lit else 'ENABLED'}")
    print("=" * 70)
    print()

    # Initialize
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    agent = SupervisorAgent(api_key=api_key)

    # Clarification
    clarification = None
    if not args.no_interact:
        clarification = _run_clarification(agent, args.query, args.files or [])

    # Run pipeline
    result = agent.run(
        query=args.query,
        file_paths=args.files or [],
        clarification_result=clarification,
        skip_literature=args.skip_lit,
        verbose=True,
        output_dir=args.output_dir,
    )

    # Save
    if result:
        path = save_result(result, args.output_dir)
        print(f"\nResult saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
