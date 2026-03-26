#!/usr/bin/env python3
# run_counts_from_dict.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


from . import mdp_config as _cfg
from .mdp_cli import main as mdp_main


def _eprint(msg: str) -> None:
    sys.stderr.write(str(msg).rstrip() + "\n")
    sys.stderr.flush()


def _load_spec(spec_arg: str) -> Dict[str, Any]:
    """
    spec_arg can be:
      - path to a JSON file
      - inline JSON string

    Expected minimal schema:
      {
        "out_root": "/abs/path/for/outputs",   # or pass via --out-root
        "cohorts": [
          {
            "name": "DiseaseA",
            # EITHER:
            "counts_dir": "/abs/path/to/DiseaseA",  # files directly inside (no subfolders)
            # OR:
            "degs_file": "/abs/path/to/degs.csv",

            "id_col": "Gene",                # optional (default "Gene")
            "lfc_col": "log2FoldChange",     # optional (default "log2FoldChange")
            "q_col": "padj",                 # optional ("", null, or a column name)
            "q_max": 0.05,                   # optional (default 0.05)
            "tissue": "stomach"              # optional
          },
          ...
        ]
      }
    """
    p = Path(spec_arg)
    if p.exists() and p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file: {p} :: {e}")
    # treat as inline JSON
    try:
        return json.loads(spec_arg)
    except Exception as e:
        raise ValueError(f"Provided --spec is neither a file nor valid JSON string: {e}")


def _validate_and_normalize(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Top-level JSON must be an object.")

    cohorts = spec.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("JSON must contain a non-empty 'cohorts' list.")

    norm_cohorts: List[Dict[str, Any]] = []
    for i, c in enumerate(cohorts, start=1):
        if not isinstance(c, dict):
            raise ValueError(f"cohorts[{i}] must be an object.")

        name = (c.get("name") or "").strip()
        if not name:
            raise ValueError(f"cohorts[{i}] requires 'name'.")

        counts_dir = c.get("counts_dir")
        degs_file = c.get("degs_file")

        has_counts = isinstance(counts_dir, str) and counts_dir.strip()
        has_degs = isinstance(degs_file, str) and degs_file.strip()

        if not (has_counts or has_degs):
            raise ValueError(
                f"cohorts[{i}] ('{name}') requires either 'counts_dir' or 'degs_file'."
            )

        norm: Dict[str, Any] = {
            "name": name,
            "id_col": (c.get("id_col") or "Gene"),
            "lfc_col": (c.get("lfc_col") or "log2FoldChange"),
            "q_col": "" if c.get("q_col") is None else str(c.get("q_col")),
            "q_max": 0.05 if c.get("q_max") is None else c.get("q_max"),
        }

        # Validate provided sources
        if has_counts:
            counts_dir_path = Path(counts_dir).expanduser().resolve()
            if not counts_dir_path.exists() or not counts_dir_path.is_dir():
                raise ValueError(f"[{name}] counts_dir not found or not a directory: {counts_dir_path}")
            # must have at least one file inside
            if not any(p.is_file() for p in counts_dir_path.iterdir()):
                raise ValueError(f"[{name}] counts_dir is empty: {counts_dir_path}")
            norm["counts_dir"] = str(counts_dir_path)

        if has_degs:
            degs_file_path = Path(degs_file).expanduser().resolve()
            if not degs_file_path.exists() or not degs_file_path.is_file():
                raise ValueError(f"[{name}] degs_file not found or not a file: {degs_file_path}")
            norm["degs_file"] = str(degs_file_path)

        # Optional hints
        if c.get("tissue"):
            norm["tissue"] = str(c["tissue"])
        if c.get("expr_file"):  # optional explicit counts table
            norm["expr_file"] = str(Path(c["expr_file"]).expanduser().resolve())

        norm_cohorts.append(norm)

    out_root = spec.get("out_root")
    if out_root:
        out_root = str(Path(out_root).expanduser().resolve())

    return {"out_root": out_root, "cohorts": norm_cohorts}


def _apply_config_and_run(out_root: Optional[str], cohorts: List[Dict[str, Any]], dry_run: bool = False) -> None:
    """
    Apply the dynamic config and run the full pipeline (mdp_cli).
    Enforces explicit OUT_ROOT to avoid hidden hardcoded defaults.
    """
    # Make local package imports robust regardless of how this script is invoked.
    base_dir = Path(__file__).resolve().parent       # counts_mdp/
    proj_root = base_dir.parent                      # project root
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    # Require explicit OUT_ROOT (either CLI or JSON). No hidden defaults.
    if not out_root:
        raise ValueError(
            "No OUT_ROOT provided. Supply it via --out-root or in the JSON spec under 'out_root'."
        )

    _cfg.CONFIG["OUT_ROOT"] = str(out_root)
    Path(_cfg.CONFIG["OUT_ROOT"]).mkdir(parents=True, exist_ok=True)

    # Overwrite cohorts with our dynamic spec
    _cfg.CONFIG["COHORTS"] = cohorts

    if dry_run:
        _eprint("[dry-run] Parsed/normalized configuration:")
        _eprint(f"  OUT_ROOT:  {_cfg.CONFIG['OUT_ROOT']}")
        for c in _cfg.CONFIG["COHORTS"]:
            src = "counts_dir" if "counts_dir" in c else "degs_file"
            _eprint(
                f"  - {c['name']}: {src}={c.get(src)} "
                f"id_col={c['id_col']} lfc_col={c['lfc_col']} "
                f"q_col={c.get('q_col')!s} q_max={c.get('q_max')!s} tissue={c.get('tissue','')}"
            )
        return

    # Run the full pipeline (orchestrator + overlap JSONs via mdp_cli)
    mdp_main()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run counts_mdp pipeline using cohorts provided as a JSON file or inline JSON string. "
            "Each cohort may specify either 'counts_dir' OR 'degs_file'."
        )
    )
    ap.add_argument(
        "--spec",
        required=True,
        help="Path to a JSON file OR an inline JSON string with { out_root?, cohorts: [...] }.",
    )
    ap.add_argument(
        "--out-root",
        default=None,
        help="Override OUT_ROOT from CLI. If omitted, uses the JSON's out_root.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the parsed configuration without running the pipeline.",
    )
    args = ap.parse_args()

    try:
        raw = _load_spec(args.spec)
        spec = _validate_and_normalize(raw)

        # precedence: CLI --out-root > JSON out_root
        out_root = args.out_root or spec.get("out_root")
        _apply_config_and_run(out_root, spec["cohorts"], dry_run=args.dry_run)
    except Exception as e:
        _eprint(f"ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
