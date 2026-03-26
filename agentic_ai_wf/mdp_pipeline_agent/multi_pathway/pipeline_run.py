#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


from counts_mdp.mdp_config import CONFIG as MDP_CONFIG  # type: ignore
from counts_mdp.mdp_orchestrator import run_all as mdp_run_all  # type: ignore

def log(msg: str) -> None:
    print(f"[pipeline] {msg}")

def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------- GC helpers ----------------

def _run_gc_and_capture(py: str, gc_enricher_dir: Path, gc_in: Path, gc_out_req: Path, fdr: float) -> Tuple[str, str, int]:
    """
    Runs GC_enricher/GC_main.py with cwd set to gc_enricher_dir so any relative paths
    (like 'GC_out') resolve inside that folder. Returns (stdout, stderr, returncode).
    """
    cmd = [
        py,
        str(gc_enricher_dir / "GC_main.py"),
        "--in", str(gc_in),
        "--out", str(gc_out_req),
        "--fdr", str(fdr),
    ]
    log("running GC_enricher → " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(gc_enricher_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    log("[ok] GC enricher finished")
    return proc.stdout or "", proc.stderr or "", proc.returncode

def _parse_gc_base_from_output(stdout: str, stderr: str, cwd: Path) -> Optional[Path]:
    """
    Parse 'All outputs in: <DIR>' from GC logs. Return absolute path if found.
    """
    text = (stdout or "") + "\n" + (stderr or "")
    m = re.search(r"All outputs in:\s*([^\r\n]+)", text)
    if not m:
        return None
    raw = m.group(1).strip()
    p = Path(raw)
    return (p if p.is_absolute() else (cwd / p)).resolve()

# ---------------- generic discovery helpers ----------------

def _scan_best_all_jsons(start_dirs: List[Path]) -> Optional[Path]:
    """
    Scan under start_dirs and return the 'all_jsons' directory with the most *.json files.
    """
    best_dir = None
    best_count = -1
    seen = set()
    for base in start_dirs:
        if not base.exists():
            continue
        for d in base.rglob("all_jsons"):
            rd = d.resolve()
            if rd in seen:
                continue
            seen.add(rd)
            count = sum(1 for _ in rd.glob("*.json"))
            if count > best_count:
                best_count = count
                best_dir = rd
    return best_dir if best_count > 0 else None

def _discover_all_jsons(
    mode: str,
    project_root: Path,
    gc_enricher_dir: Path,
    gc_out_req: Path,
    mdp_out_root: Path,
    gc_base_from_logs: Optional[Path],
) -> Tuple[Optional[Path], List[Path]]:
    """
    Build a candidate list and return (chosen_dir, checked_list).
    For GC: prefer log-parsed base/results/all_jsons, else canonical, else scan.
    For MDP: prefer OUT_ROOT/results/all_jsons, else canonical under project, else scan.
    """
    checked: List[Path] = []
    candidates: List[Path] = []

    if mode == "gc":
        # 0) If GC printed a base, try <base>/results/all_jsons first
        if gc_base_from_logs:
            candidates.append(gc_base_from_logs / "results" / "all_jsons")

        # 1) project canonical
        candidates.append(project_root / "results" / "all_jsons")

        # 2) requested GC out
        candidates.append(gc_out_req / "results" / "all_jsons")

        # 3) common fallbacks relative to repo
        candidates.append(gc_enricher_dir / "GC_out" / "results" / "all_jsons")
        candidates.append(Path.cwd() / "results" / "all_jsons")
        candidates.append(Path.cwd() / "GC_out" / "results" / "all_jsons")

    else:  # mdp
        # counts_mdp typically writes under OUT_ROOT; prefer that first
        candidates.append(mdp_out_root / "results" / "all_jsons")

        # project canonical
        candidates.append(project_root / "results" / "all_jsons")

        # some fallbacks
        candidates.append(Path.cwd() / "results" / "all_jsons")

    # Deduplicate while preserving order
    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    # Try direct candidates
    for c in uniq:
        checked.append(c)
        if c.is_dir() and any(c.glob("*.json")):
            return c, checked

    # Deep scan (pick 'all_jsons' with most files)
    best = _scan_best_all_jsons([project_root, gc_enricher_dir, Path.cwd(), mdp_out_root, gc_out_req])
    if best:
        checked.append(best)
        return best, checked

    return None, checked

def _require_all_jsons_or_die(chosen: Optional[Path], checked: List[Path], hints: List[str]) -> Path:
    if chosen:
        return chosen
    msg = (
        "Could not locate 'results/all_jsons' directory with JSON files.\n"
        "Checked the following locations (in order):\n  - " + "\n  - ".join(str(x) for x in checked) + "\n"
        "Hints:\n  - " + "\n  - ".join(hints)
    )
    sys.exit(msg)

# ---------------- comparison ----------------

def run_comparison(py: str, cmp_script: Path, all_jsons_dir: Path, out_dir: Path, prefix: str, verbosity: int = 1) -> None:
    safe_mkdir(out_dir)
    cmd = [
        py,
        str(cmp_script),
        "--input", str(all_jsons_dir),
        "--pattern", "*.json",
        "--cap", "30",
        "--outdir", str(out_dir),
        "--prefix", prefix,
        "--label-source", "filename",
    ]
    if verbosity >= 1:
        cmd.append("-v")
    if verbosity >= 2:
        cmd.append("-v")
    log("compare → " + " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)

# ---------------- MDP runner ----------------

def _run_mdp(counts_dir: Path, out_root: Path, disease_name: Optional[str], tissue: str) -> None:
    """
    Runs counts-based pipeline via counts_mdp, assuming the package is present as counts_mdp/*.
    It sets CONFIG minimally and calls mdp_orchestrator.run_all().
    """

    name = disease_name or counts_dir.name.replace(" ", "_")
    MDP_CONFIG["OUT_ROOT"] = str(out_root.resolve())
    MDP_CONFIG["COHORTS"] = [{
        "name": name,
        "counts_dir": str(counts_dir.resolve()),
        "id_col": "Gene",
        "lfc_col": "log2FoldChange",
        "q_col": "",
        "q_max": 0.05,
        "tissue": tissue or "",
    }]
    log(f"running MDP for '{name}' → {out_root}")
    mdp_run_all()

# ---------------- CLI main ----------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified runner for GC (GeneCards JSON enricher) OR MDP (counts) → compare.",
    )
    ap.add_argument("--mode", choices=["gc", "mdp"], required=True)
    ap.add_argument("--in", dest="inp", required=True,
                    help="GC: folder with GeneCards JSONs (filenames = disease names). "
                         "MDP: counts folder (or top folder containing counts).")
    ap.add_argument("--out", required=True, help="Project output root.")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR/P-value cutoff (default: 0.05)")
    ap.add_argument("--prefix", default="compare", help="Comparison output prefix (default: compare).")
    ap.add_argument("--py", default=sys.executable, help="Python to run sub-steps (default: current).")
    # Optional for MDP
    ap.add_argument("--name", default="", help="(MDP) disease/cohort name override")
    ap.add_argument("--tissue", default="", help="(MDP) tissue label")
    args = ap.parse_args()

    mode = args.mode
    inp_dir = Path(args.inp).resolve()
    project_root = Path(args.out).resolve()
    py = args.py
    fdr = float(args.fdr)
    prefix = args.prefix.strip() or "compare"

    if not inp_dir.is_dir():
        sys.exit(f"[ERROR] input folder not found: {inp_dir}")

    safe_mkdir(project_root)

    # Paths inside repo layout
    repo_root = Path(__file__).resolve().parent
    gc_enricher_dir = repo_root / "GC_enricher"
    cmp_script = repo_root / "comparison" / "json_comparison.py"

    if not cmp_script.exists():
        sys.exit(f"[ERROR] comparison script not found: {cmp_script}")

    # Defaults
    gc_out_req = project_root / "GC_enrich"
    mdp_out_root = project_root  # counts pipeline should respect OUT_ROOT here
    cmp_out = project_root / "results" / f"comparison_{mode}"

    # --- Execute the chosen pipeline
    if mode == "gc":
        # run GC with cwd so 'GC_out' (if used) lands under GC_enricher/
        try:
            stdout, stderr, _ = _run_gc_and_capture(py, gc_enricher_dir, inp_dir, gc_out_req, fdr)
        except subprocess.CalledProcessError as e:
            sys.exit(f"[ERROR] GC enricher failed with exit code {e.returncode}")

        # parse base from logs if present
        gc_base = _parse_gc_base_from_output(stdout, stderr, gc_enricher_dir)
        if gc_base:
            log(f"[gc-base] {gc_base}")
        else:
            log("[gc-base] not found in logs; will search typical locations")

        # discover all_jsons
        chosen, checked = _discover_all_jsons(
            mode="gc",
            project_root=project_root,
            gc_enricher_dir=gc_enricher_dir,
            gc_out_req=gc_out_req,
            mdp_out_root=mdp_out_root,
            gc_base_from_logs=gc_base,
        )
        hints = [
            "GC_main prints 'All outputs in: <DIR>' — this runner resolves it under GC_enricher/ if relative.",
            "Verify your GC input folder actually produced per-disease JSONs.",
            "If GC writes to a custom location, tell me and I’ll add it.",
        ]
        all_jsons = _require_all_jsons_or_die(chosen, checked, hints)

    else:  # mode == "mdp"
        try:
            _run_mdp(inp_dir, mdp_out_root, args.name.strip() or None, args.tissue.strip())
        except Exception as e:
            sys.exit(f"[ERROR] MDP pipeline failed: {e}")

        # discover all_jsons from counts output
        chosen, checked = _discover_all_jsons(
            mode="mdp",
            project_root=project_root,
            gc_enricher_dir=gc_enricher_dir,
            gc_out_req=gc_out_req,
            mdp_out_root=mdp_out_root,
            gc_base_from_logs=None,
        )
        hints = [
            "counts_mdp writes under OUT_ROOT; ensure results/all_jsons exists with JSONs.",
            "If your counts pipeline writes JSONs somewhere else, tell me and I’ll add it.",
        ]
        all_jsons = _require_all_jsons_or_die(chosen, checked, hints)

    log(f"[found] all_jsons: {all_jsons}")

    # --- Compare
    try:
        run_comparison(py, cmp_script, all_jsons, cmp_out, prefix, verbosity=1)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[ERROR] comparison failed with exit code {e.returncode}")

    # --- Summary
    summary = {
        "mode": mode,
        "input": str(inp_dir),
        "out_root": str(project_root),
        "gc_out_requested": (str(gc_out_req) if mode == "gc" else None),
        "all_jsons_used": str(all_jsons),
        "comparison_dir": str(cmp_out),
    }
    (project_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[DONE] Project outputs -> {project_root}")

if __name__ == "__main__":
    main()
