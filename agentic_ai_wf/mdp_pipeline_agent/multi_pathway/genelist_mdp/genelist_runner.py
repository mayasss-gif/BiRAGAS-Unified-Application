#!/usr/bin/env python3
"""
Unified runner for GC (GeneCards JSON) and GL (gene-list files).
Robust to different function names in GC_enrich/GL_enrich by trying multiple
entry points and falling back to a module subprocess if needed.
"""

import argparse
import importlib
import os
import sys
import subprocess
from typing import Any


def _import_runner(runner: str) -> Any:
    pkg = __package__ or "genelist_mdp"
    if runner.upper() == "GL":
        return importlib.import_module(f"{pkg}.GL_enrich")
    elif runner.upper() == "GC":
        return importlib.import_module(f"{pkg}.GC_enrich")
    raise ValueError(f"Unknown runner: {runner}. Use 'GC' or 'GL'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified runner for GC (GeneCards JSON) and GL (gene-list files). No default paths; all via CLI."
    )
    p.add_argument("--runner", required=True, choices=["GC", "GL"],
                   help="GC = GeneCards JSON inputs (folder of .json). GL = gene-list inputs (folder of .txt/.csv/.tsv).")
    p.add_argument("--input", required=True, help="Input folder (GC: .json files; GL: .txt/.csv/.tsv files).")
    p.add_argument("--out-root", required=True, help="Output root directory.")

    # GC options (ignored by GL)
    p.add_argument("--gene-col", default="gene_symbol",
                   help="(GC) JSON['data'] key for gene symbol (default: gene_symbol).")
    p.add_argument("--score-col", default="gene_score",
                   help="(GC) JSON['data'] key for gene score (default: gene_score).")

    # Shared enrichment knobs
    p.add_argument("--fdr", type=float, default=0.05, help="FDR threshold (default 0.05)")
    p.add_argument("--topk", type=int, default=500, help="Top K genes for Enrichr (default 500)")
    p.add_argument("--perms", type=int, default=100, help="GSEA prerank permutations (default 100)")
    p.add_argument("--minset", type=int, default=10, help="Min set size for GSEA (default 10)")
    p.add_argument("--maxset", type=int, default=2000, help="Max set size for GSEA (default 2000)")
    p.add_argument("--seed", type=int, default=13, help="Random seed (default 13)")
    p.add_argument("--jaccard", type=float, default=0.7, help="Title Jaccard collapse threshold (default 0.7)")
    p.add_argument("--no-collapse", action="store_true", help="Disable redundancy collapse of similar term names")
    return p.parse_args()


def _ensure_dirs(in_dir: str, out_root: str) -> None:
    if not os.path.isdir(in_dir):
        print(f"Root folder not found or not a directory: {in_dir}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(out_root, exist_ok=True)


def _fallback_subprocess(module_qualname: str, runner: str, args: argparse.Namespace) -> int:
    """Fallback: run the runner module as `python -m <module>` with equivalent flags."""
    cmd = [sys.executable, "-m", module_qualname,
           "--input", os.path.abspath(args.input),
           "--out-root", os.path.abspath(args.out_root),
           "--fdr", str(args.fdr),
           "--topk", str(args.topk),
           "--perms", str(args.perms),
           "--minset", str(args.minset),
           "--maxset", str(args.maxset),
           "--seed", str(args.seed)]
    if not args.no_collapse:
        cmd.append("--collapse")
        cmd += ["--jaccard", str(args.jaccard)]

    # GC-only extra flags if we’re running GC_enrich as a module
    if runner.upper() == "GC":
        cmd += ["--gene-col", args.gene_col, "--score-col", args.score_col]

    print(f"[genelist_runner] Falling back to module execution: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _call_with_best_effort(mod: Any, runner: str, args: argparse.Namespace) -> int:
    """
    Try a cascade of possible entry points in GL_enrich / GC_enrich, then fallback to subprocess.
    """
    in_dir = os.path.abspath(args.input)
    out_root = os.path.abspath(args.out_root)
    _ensure_dirs(in_dir, out_root)

    # Candidate signatures (name, kwargs_builder)
    def common_kwargs():
        return dict(
            fdr=args.fdr, topk=args.topk, perms=args.perms,
            minset=args.minset, maxset=args.maxset, seed=args.seed,
            collapse=not args.no_collapse, jaccard_thresh=args.jaccard,
        )

    tried = []

    if runner.upper() == "GL":
        candidates = [
            ("run_gl_folder", lambda: dict(input_dir=in_dir, out_root=out_root, **common_kwargs())),
            ("run_folder",    lambda: dict(input_dir=in_dir, out_root=out_root, **common_kwargs())),
            ("run",           lambda: dict(input_dir=in_dir, out_root=out_root, **common_kwargs())),
            ("main",          lambda: dict()),  # often argparse-based; if exists, we’ll fallback to subprocess instead
        ]
        module_name = (mod.__package__ or "genelist_mdp") + ".GL_enrich"
    else:
        candidates = [
            ("run_gc_folder", lambda: dict(input_dir=in_dir, out_root=out_root,
                                           gene_key=args.gene_col, score_key=args.score_col, **common_kwargs())),
            ("run_folder",    lambda: dict(input_dir=in_dir, out_root=out_root,
                                           gene_key=args.gene_col, score_key=args.score_col, **common_kwargs())),
            ("run",           lambda: dict(input_dir=in_dir, out_root=out_root,
                                           gene_key=args.gene_col, score_key=args.score_col, **common_kwargs())),
            ("main",          lambda: dict()),
        ]
        module_name = (mod.__package__ or "genelist_mdp") + ".GC_enrich"

    for name, kwargs_fn in candidates:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            fn = getattr(mod, name)
            tried.append(name)
            if name == "main":
                # Likely argparse-based; prefer subprocess run to avoid signature mismatch.
                break
            try:
                ret = fn(**kwargs_fn())
                # Normalize truthy/falsey to 0/1 exit code
                return 0 if (ret is None or ret is True) else int(not ret)
            except TypeError as e:
                print(f"[genelist_runner] Signature mismatch calling {mod.__name__}.{name}: {e}", file=sys.stderr)
                # Try next candidate
            except Exception as e:
                print(f"[genelist_runner] Error in {mod.__name__}.{name}: {e}", file=sys.stderr)
                return 1

    # Fallback to running as a module
    return _fallback_subprocess(module_name, runner, args)


def main() -> int:
    args = parse_args()
    mod = _import_runner(args.runner)
    return _call_with_best_effort(mod, args.runner, args)


if __name__ == "__main__":
    sys.exit(main())
