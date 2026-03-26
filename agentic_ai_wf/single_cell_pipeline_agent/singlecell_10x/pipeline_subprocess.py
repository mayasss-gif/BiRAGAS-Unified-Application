#!/usr/bin/env python3
"""
Subprocess wrapper for entire single-cell pipeline (Celery fork-safe).

This module runs the complete single-cell pipeline in an isolated subprocess
to avoid Celery prefork deadlocks with matplotlib/R resources.

Usage:
    python -m agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.pipeline_subprocess \
        --single-10x-dir <input_dir> \
        --output-dir <output_dir> \
        [--sample-label <label>] \
        [--group-label <label>] \
        [--do-pathway-clustering] \
        [--do-groupwise-de] \
        [--do-dpt] \
        [--batch-key <key>] \
        [--integration-method <method>] \
        [--geo-json-path <path>] \
        [--logos-dir <dir>] \
        [--generate-report] \
        [--prepare-for-bisque]
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Configure logging to ensure output is flushed immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from .main_single import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run single-cell pipeline in isolated subprocess (Celery fork-safe)"
    )
    parser.add_argument(
        "--single-10x-dir",
        required=True,
        help="Path to 10x Genomics data directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--sample-label",
        default=None,
        help="Sample label (optional)",
    )
    parser.add_argument(
        "--group-label",
        default=None,
        help="Group label (optional)",
    )
    parser.add_argument(
        "--do-pathway-clustering",
        action="store_true",
        help="Enable pathway clustering",
    )
    parser.add_argument(
        "--do-groupwise-de",
        action="store_true",
        help="Enable group-wise differential expression",
    )
    parser.add_argument(
        "--do-dpt",
        action="store_true",
        help="Enable diffusion pseudotime computation",
    )
    parser.add_argument(
        "--batch-key",
        default=None,
        help="Batch key for integration (optional)",
    )
    parser.add_argument(
        "--integration-method",
        default=None,
        help="Integration method (e.g., 'bbknn')",
    )
    parser.add_argument(
        "--geo-json-path",
        default=None,
        help="Path to GEO JSON metadata file (optional)",
    )
    parser.add_argument(
        "--logos-dir",
        default=None,
        help="Directory containing logo files (optional)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML/PDF report",
    )
    parser.add_argument(
        "--prepare-for-bisque",
        action="store_true",
        help="Prepare output for Bisque deconvolution",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Path to write result JSON (optional)",
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Starting single-cell pipeline subprocess...", flush=True)
        print(f"Input directory: {args.single_10x_dir}", flush=True)
        print(f"Output directory: {args.output_dir}", flush=True)
        print(f"Python executable: {sys.executable}", flush=True)
        print(f"Working directory: {Path.cwd()}", flush=True)
        
        # Run pipeline
        print("Calling run_pipeline()...", flush=True)
        result_path = run_pipeline(
            single_10x_dir=args.single_10x_dir,
            sample_label=args.sample_label,
            group_label=args.group_label,
            out_name=args.output_dir,
            do_pathway_clustering=args.do_pathway_clustering,
            do_groupwise_de=args.do_groupwise_de,
            do_dpt=args.do_dpt,
            batch_key=args.batch_key,
            integration_method=args.integration_method,
            geo_json_path=args.geo_json_path,
            logos_dir=args.logos_dir,
            generate_report=args.generate_report,
            prepare_for_bisque=args.prepare_for_bisque,
        )
        
        result = {
            "status": "completed",
            "output_dir": str(result_path),
            "ok": True,
        }
        
        # Write result JSON if requested
        if args.result_json:
            with open(args.result_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✅ Result JSON written to: {args.result_json}", flush=True)
        
        print(f"✅ Pipeline completed successfully. Output: {result_path}", flush=True)
        print(json.dumps(result), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)
        
    except Exception as e:
        error_result = {
            "status": "failed",
            "error": str(e),
            "ok": False,
        }
        
        print(f"❌ Pipeline failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        
        # Write error result JSON if requested
        if args.result_json:
            try:
                with open(args.result_json, "w") as f:
                    json.dump(error_result, f, indent=2)
            except:
                pass
        
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

