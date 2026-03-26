#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

def _gather_jsons(src: Path) -> list[Path]:
    if not src or not src.exists() or not src.is_dir():
        return []
    return sorted(src.glob("*.json"))

def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _prefix_name(src: Path, prefix: str) -> str:
    stem, ext = src.stem, (src.suffix or ".json")
    return f"{prefix}_{stem}{ext}"

def merge_and_compare(
    counts_dir: str | Path | None,
    genelist_dir: str | Path | None,
    out_root: str | Path,
    compare_prefix: str = "merged_pipeline_comparison",
    clear: bool = True,
    verbose: bool = True,
) -> int:
    """
    - Copies JSONs from counts_dir and genelist_dir into <out_root>/jsons_all_folder
      with filename prefixes 'counts_' and 'genelist_'.
    - Runs: mdp-json-compare --input <jsons_all_folder> --outdir <out_root>/json_comparison --prefix <compare_prefix> [-v]
    """
    out_root = Path(out_root).expanduser().resolve()
    merged_dir = _safe_mkdir(out_root / "jsons_all_folder")
    cmp_outdir = _safe_mkdir(out_root / "json_comparison")

    if clear:
        for p in merged_dir.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)

    counts = _gather_jsons(Path(counts_dir).expanduser().resolve()) if counts_dir else []
    gls    = _gather_jsons(Path(genelist_dir).expanduser().resolve()) if genelist_dir else []

    if not counts and not gls:
        print("[merge_and_compare] No input JSONs found in either source.", file=sys.stderr)
        return 2

    # Copy with prefixes
    added = 0
    for src in counts:
        dst = merged_dir / _prefix_name(src, "counts")
        shutil.copy2(src, dst)
        added += 1
    for src in gls:
        dst = merged_dir / _prefix_name(src, "genelist")
        shutil.copy2(src, dst)
        added += 1

    if added == 0:
        print("[merge_and_compare] Nothing copied.", file=sys.stderr)
        return 2

    # Prefer CLI; fallback to module call if CLI missing
    cmd = [
        "mdp-json-compare",
        "--input", str(merged_dir),
        "--outdir", str(cmp_outdir),
        "--prefix", compare_prefix,
    ]
    if verbose:
        cmd.append("-v")

    try:
        print("[merge_and_compare] Running:", " ".join(cmd))
        return subprocess.call(cmd)
    except FileNotFoundError:
        # Fallback to python -m json_comparison.json_compare
        fallback = [
            sys.executable, "-m", "json_comparison.json_compare",
            "--input", str(merged_dir),
            "--outdir", str(cmp_outdir),
            "--prefix", compare_prefix,
        ]
        if verbose:
            fallback.append("-v")
        print("[merge_and_compare] CLI not found; fallback:", " ".join(fallback))
        return subprocess.call(fallback)

def main():
    ap = argparse.ArgumentParser(
        description="Merge two JSON bundles (counts, genelist) into jsons_all_folder and run mdp-json-compare."
    )
    ap.add_argument("--counts", help="Path to counts pipeline JSON folder (e.g., .../results/jsons_all_folder).")
    ap.add_argument("--genelist", help="Path to genelist/GC/GL JSON folder (e.g., .../GC_enrich/jsons_all_folder).")
    ap.add_argument("--out-root", required=True, help="Output root where merged jsons_all_folder and json_comparison will be created.")
    ap.add_argument("--prefix", default="merged_pipeline_comparison", help="Comparison file prefix.")
    ap.add_argument("--no-clear", action="store_true", help="Do not clear merged folder before copying.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose compare.")
    args = ap.parse_args()

    rc = merge_and_compare(
        counts_dir=args.counts,
        genelist_dir=args.genelist,
        out_root=args.out_root,
        compare_prefix=args.prefix,
        clear=(not args.no_clear),
        verbose=args.verbose,
    )
    sys.exit(rc)

if __name__ == "__main__":
    main()
