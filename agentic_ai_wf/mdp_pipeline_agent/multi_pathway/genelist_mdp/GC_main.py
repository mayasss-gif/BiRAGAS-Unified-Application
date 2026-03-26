#!/usr/bin/env python3
"""Thin wrapper so the mdp-gc console script can import GC_enricher.GC_main:main"""
from __future__ import annotations

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..GC_enricher import GC_enrich as impl

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run GC pipeline on a folder of GeneCards-style JSON inputs."
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input folder with GeneCards JSON files (one per disease).")
    ap.add_argument("--out", dest="out", required=True,
                    help="Output root directory.")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--minset", type=int, default=10)
    ap.add_argument("--maxset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-collapse", action="store_true",
                    help="Disable redundancy collapse of similar term names.")
    ap.add_argument("--jaccard", type=float, default=0.7)
    ap.add_argument("--gene-col", default="gene_symbol",
                    help="Key in JSON['data'] for gene symbols.")
    ap.add_argument("--score-col", default="gene_score",
                    help="Key in JSON['data'] for scores (fallbacks handled in code).")
    return ap.parse_args()

def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    in_dir = Path(args.inp)
    out_root = Path(args.out)
    if not in_dir.is_dir():
        print(f"[GC_main] Input is not a directory: {in_dir}", file=sys.stderr)
        return 2

    

    # Try run function(s) in preferred order.
    runners = [
        "run_gc_folder",                 # preferred
        "run_gc_enrichment_for_folder",  # alt name
        "run_folder",                    # very generic
        "main",                          # module-level main
    ]

    for fn in runners:
        if hasattr(impl, fn):
            runner = getattr(impl, fn)
            try:
                ok = runner(
                    input_dir=str(in_dir),
                    out_root=str(out_root),
                    fdr=args.fdr,
                    topk=args.topk,
                    perms=args.perms,
                    minset=args.minset,
                    maxset=args.maxset,
                    seed=args.seed,
                    collapse=(not args.no_collapse),
                    jaccard=args.jaccard,
                    gene_col=args.gene_col,
                    score_col=args.score_col,
                )
                return 0 if (ok is True or ok is None) else 1
            except TypeError:
                # Older signature (just input_dir/out_root). Try minimal form.
                ok = runner(str(in_dir), str(out_root))
                return 0 if (ok is True or ok is None) else 1
            except Exception as e:
                print(f"[GC_main] ERROR while running {fn}: {e}", file=sys.stderr)
                return 1

    print("[GC_main] No compatible runner found in GC_enricher.GC_enrich. "
          "Expected one of: run_gc_folder, run_gc_enrichment_for_folder, run_folder, main",
          file=sys.stderr)
    return 3

if __name__ == "__main__":
    raise SystemExit(main())
