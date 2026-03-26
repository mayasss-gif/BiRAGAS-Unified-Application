#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
merge_json_bundles.py — merge JSON outputs from two sources into one folder.

Use cases:
- Merge counts/DEGs pipeline JSONs with genelist/GC pipeline JSONs.
- On filename collisions, auto-append a source label (e.g., __counts, __degs, __gl, __gc).
- Optional content-hash de-duplication to skip identical files.
- Collision policy: overwrite | keep-first | rename.
- Symlink mode for fast, space-saving merges (falls back to copy if not supported).

Examples:
  python merge_json_bundles.py \
    --counts-dir /path/to/jsons_all_counts \
    --genelist-dir /path/to/jsons_all_folder \
    --out /path/to/merged_jsons

  # With explicit labels and dedupe:
  python merge_json_bundles.py \
    --counts-dir /path/to/jsons_all_counts --counts-label degs \
    --genelist-dir /path/to/jsons_all_folder --genelist-label gc \
    --out /path/to/merged_jsons --dedupe

  # If you only have one side:
  python merge_json_bundles.py --genelist-dir ./genelist/jsons_all_folder --out ./merged
"""

from __future__ import annotations
import argparse, hashlib, shutil, sys, os
from pathlib import Path
from typing import Iterable, Dict, Optional, Tuple, List

# ---------------- Utilities ----------------

def sha1_of_file(p: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with p.open('rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def iter_jsons(d: Optional[Path], pattern: str) -> List[Path]:
    if not d:
        return []
    if not d.exists() or not d.is_dir():
        print(f"[warn] skip (not a dir): {d}", flush=True)
        return []
    files = sorted(d.glob(pattern))
    if not files:
        print(f"[warn] no matches in: {d} (pattern={pattern})", flush=True)
    return files

def ensure_out_dir(out_dir: Path, clear: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for p in out_dir.glob("*"):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception as e:
                print(f"[warn] failed to clear {p}: {e}", flush=True)

def with_source_tag(basename: str, source_tag: str) -> str:
    stem = Path(basename).stem
    ext = Path(basename).suffix or ".json"
    tag = source_tag.strip("_")
    return f"{stem}__{tag}{ext}"

def plan_target_name(base: str, out_dir: Path, policy: str, used: set) -> Path:
    """If collision and policy=rename, produce unique name; respects 'used' names in-session."""
    target = out_dir / base
    if (not target.exists()) and (base not in used):
        return target
    if policy == "overwrite":
        return target
    if policy == "keep-first":
        return target  # caller will skip
    # rename policy
    stem, ext = Path(base).stem, Path(base).suffix or ".json"
    i = 2
    while True:
        candidate = out_dir / f"{stem}__{i}{ext}"
        if (not candidate.exists()) and (candidate.name not in used):
            return candidate
        i += 1

def copy_or_link(src: Path, dst: Path, use_symlink: bool) -> None:
    if use_symlink:
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
            return
        except Exception:
            pass  # fallback to copy
    shutil.copy2(src, dst)

# ---------------- Core merge logic ----------------

def add_file(
    src: Path,
    out_dir: Path,
    source_tag: Optional[str],
    used_names: set,
    collision: str,
    use_symlink: bool,
    seen_hashes: Dict[str, Path] | None,
    dedupe: bool
) -> Tuple[bool, bool, bool, bool]:
    """
    Returns tuple: (added, overwritten, skipped_identical, skipped_keep)
    """
    base = src.name

    # Compute hash for dedupe
    try:
        h = sha1_of_file(src) if dedupe else None
    except Exception as e:
        print(f"[warn] hash failed for {src}: {e}", flush=True)
        h = None

    if dedupe and h and h in (seen_hashes or {}):
        prev = (seen_hashes or {})[h]
        print(f"[skip-identical] {src.name} == {prev.name}", flush=True)
        return (False, False, True, False)

    # If name already exists in output, try tagging with source before applying policy
    candidate_name = base
    candidate_path = out_dir / candidate_name
    if candidate_path.exists() or (candidate_name in used_names):
        if source_tag:
            tagged = with_source_tag(base, source_tag)
            if not (out_dir / tagged).exists() and tagged not in used_names:
                candidate_name = tagged
                candidate_path = out_dir / candidate_name

    # Apply collision policy if still colliding
    if candidate_path.exists() or (candidate_name in used_names):
        if collision == "keep-first":
            print(f"[keep-first] {src.name} (existing kept as {candidate_path.name})", flush=True)
            if dedupe and h:
                seen_hashes[h] = candidate_path
            return (False, False, False, True)
        elif collision == "overwrite":
            try:
                if candidate_path.is_file() or candidate_path.is_symlink():
                    candidate_path.unlink()
                else:
                    shutil.rmtree(candidate_path)
            except Exception:
                pass
            overwritten = True
        else:
            # rename
            candidate_path = plan_target_name(candidate_name, out_dir, collision, used_names)
            overwritten = False
    else:
        overwritten = False

    # Copy or link
    try:
        copy_or_link(src, candidate_path, use_symlink)
        used_names.add(candidate_path.name)
        if dedupe and h:
            if seen_hashes is not None:
                seen_hashes[h] = candidate_path
        print(f"[add] {src} -> {candidate_path.name}", flush=True)
        return (True, overwritten, False, False)
    except Exception as e:
        print(f"[warn] failed to add {src}: {e}", flush=True)
        return (False, False, False, False)

def main():
    ap = argparse.ArgumentParser(
        description="Merge JSON outputs from two sources (counts/DEGs and genelist/GC) into one folder."
    )
    ap.add_argument("--counts-dir", help="Input directory for counts/DEGs JSONs (e.g., jsons_all_counts).")
    ap.add_argument("--genelist-dir", help="Input directory for genelist/GC JSONs (e.g., jsons_all_folder).")
    ap.add_argument("--pattern", default="*.json", help="Glob for JSON files (default: *.json).")
    ap.add_argument("--out", "-o", required=True, help="Output directory to write merged JSONs.")
    ap.add_argument("--clear", action="store_true", help="Clear output directory before merging.")
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copying (fallback to copy if not supported).")
    ap.add_argument("--collision", choices=["overwrite", "keep-first", "rename"], default="rename",
                    help="On filename collision: overwrite | keep-first | rename (default).")
    ap.add_argument("--dedupe", action="store_true", help="Content-hash de-duplication across sources.")
    # Source labels (what to append on collision)
    ap.add_argument("--counts-label", default="counts", help="Label to append for counts/DEGs source on collision (e.g., counts|degs).")
    ap.add_argument("--genelist-label", default="genelist", help="Label to append for genelist/GC source on collision (e.g., gl|gc).")

    args = ap.parse_args()

    counts_dir = Path(args.counts_dir).expanduser().resolve() if args.counts_dir else None
    genelist_dir = Path(args.genelist_dir).expanduser().resolve() if args.genelist_dir else None
    out_dir = Path(args.out).expanduser().resolve()

    if not counts_dir and not genelist_dir:
        print("[error] Provide at least one of --counts-dir or --genelist-dir.", flush=True)
        sys.exit(1)

    ensure_out_dir(out_dir, clear=args.clear)

    # Gather files (stable order: counts first, then genelist)
    counts_files = iter_jsons(counts_dir, args.pattern)
    genelist_files = iter_jsons(genelist_dir, args.pattern)

    if not counts_files and not genelist_files:
        print("[error] No JSON files found in the provided inputs.", flush=True)
        sys.exit(1)

    used_names: set = set()  # track names we place during this run
    seen_hashes: Dict[str, Path] = {} if args.dedupe else {}
    added = 0
    overwritten = 0
    skipped_identical = 0
    skipped_keep = 0
    errors = 0

    # Process counts/DEGs
    for src in counts_files:
        ok_added, ok_overw, ok_skip_id, ok_skip_keep = add_file(
            src=src,
            out_dir=out_dir,
            source_tag=(args.counts_label or "counts"),
            used_names=used_names,
            collision=args.collision,
            use_symlink=args.symlink,
            seen_hashes=seen_hashes,
            dedupe=args.dedupe
        )
        added += int(ok_added)
        overwritten += int(ok_overw)
        skipped_identical += int(ok_skip_id)
        skipped_keep += int(ok_skip_keep)

    # Process genelist/GC
    for src in genelist_files:
        ok_added, ok_overw, ok_skip_id, ok_skip_keep = add_file(
            src=src,
            out_dir=out_dir,
            source_tag=(args.genelist_label or "genelist"),
            used_names=used_names,
            collision=args.collision,
            use_symlink=args.symlink,
            seen_hashes=seen_hashes,
            dedupe=args.dedupe
        )
        added += int(ok_added)
        overwritten += int(ok_overw)
        skipped_identical += int(ok_skip_id)
        skipped_keep += int(ok_skip_keep)

    # Summary
    print("\n=== merge summary ===")
    print(f"counts files:      {len(counts_files)}")
    print(f"genelist files:    {len(genelist_files)}")
    print(f"added:             {added}")
    print(f"overwritten:       {overwritten}")
    print(f"skipped identical: {skipped_identical}")
    print(f"skipped (keep):    {skipped_keep}")
    print(f"output dir:        {out_dir}")

    # Exit code: nonzero only if nothing was added and there were no inputs
    if added == 0 and (len(counts_files) + len(genelist_files)) == 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
