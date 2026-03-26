#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging, sys, traceback, difflib, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

from pathway_compare.config import PCConfig
from pathway_compare.io_utils import (
    ensure_dir, load_jsons, read_pathway_list,
)
from pathway_compare.compute import (
    build_tidy_entities, build_views_from_entities,
    run_individual_pathway_analysis, run_multi_pathway_comparison,
)
import pathway_compare.compute as pc_compute  # to pass cfg for entity cleaning


# ---------------- logging ----------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%-Y-%m-%d %H:%M:%S",
    )


# ---------------- argparse ----------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pathway_compare",
        description="Compare one or more pathways across all diseases using a folder of JSONs (directional / non-directional / mixed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json-root", required=True, help="Folder containing per-disease JSON files.")
    p.add_argument("--out-root", required=True, help="Output folder (module will create pathway_compare/*).")

    # IMPORTANT: allow multiple flags; we’ll join before parsing
    p.add_argument("--pathways", action="append", default=[], help="Pathway name/ID (repeat flag to add more).")
    p.add_argument("--pathways-file", type=str, default="", help="Text/CSV file with pathway names (one per line or a column).")
    p.add_argument("--pathway-column", type=str, default="", help="If CSV, column name to read pathways from.")

    p.add_argument("--direction-mode", type=str, default="both", choices=["any","up","down","both"])
    p.add_argument("--borrow-any-into-directional", type=str, default="up", choices=["none","up","down","both"])
    p.add_argument("--sig-cap", type=float, default=300.0)

    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max-upset-intersections", type=int, default=100)

    p.add_argument("--fail-on-missing-pathway", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args(argv)


# ---------------- utilities ----------------

_ILIB_PREFIX = re.compile(r"^(HALLMARK_|KEGG_|REACTOME_)", re.I)
_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = str(s or "").strip()
    s = _ILIB_PREFIX.sub("", s)
    s = s.replace("_", " ").replace("-", " ")
    s = _WS.sub(" ", s)
    return s.lower()

def _select_view(views: Dict[str, Any], direction_mode: str):
    dm = (direction_mode or "both").lower()
    if dm == "any":
        return views.get("ANY")
    if dm == "up":
        return views.get("UP")
    if dm == "down":
        return views.get("DOWN")
    # both: prefer ANY; else union of UP,DOWN with best rows kept
    anyv = views.get("ANY")
    if anyv is not None and not anyv.empty:
        return anyv
    up = views.get("UP"); down = views.get("DOWN")
    up = up if up is not None else pd.DataFrame()
    down = down if down is not None else pd.DataFrame()
    ud = pd.concat([up, down], ignore_index=True)
    if ud.empty:
        return ud
    return (ud.sort_values(["qval","pval","sig"], ascending=[True, True, False], na_position="last")
              .drop_duplicates(["disease","pathway","entity_type","entity"], keep="first"))

def _catalog_labels(df) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return (all_labels, norm_index: normalized -> [originals])."""
    if df is None or df.empty or "pathway" not in df.columns:
        return [], {}
    labels = [str(x) for x in df["pathway"].dropna().unique().tolist()]
    idx: Dict[str, List[str]] = {}
    for lab in labels:
        idx.setdefault(_norm(lab), []).append(lab)
    return labels, idx

def _closest(cands: List[str], target: str, n: int = 6) -> List[str]:
    return difflib.get_close_matches(target, cands, n=n, cutoff=0.5)

def _resolve_against_labels(requested: List[str], all_labels: List[str], norm_index: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Resolve requested names to actual labels using robust matching."""
    labels_cs = set(all_labels)
    labels_ci = {lab.lower(): lab for lab in all_labels}
    norm_keys = list(norm_index.keys())

    resolved: List[str] = []
    unresolved: List[str] = []

    for raw in requested:
        q_raw = str(raw).strip()
        q_norm = _norm(q_raw)

        # 1) exact CS
        if q_raw in labels_cs:
            resolved.append(q_raw); continue
        # 2) exact CI
        if q_raw.lower() in labels_ci:
            resolved.append(labels_ci[q_raw.lower()]); continue
        # 3) exact normalized
        if q_norm in norm_index:
            candidates = sorted(norm_index[q_norm], key=len)
            resolved.append(candidates[0]); continue
        # 4) unique substring on normalized
        subs = [k for k in norm_keys if q_norm in k]
        if len(subs) == 1 and subs[0] in norm_index:
            candidates = sorted(norm_index[subs[0]], key=len)
            resolved.append(candidates[0]); continue

        unresolved.append(q_raw)

    return resolved, unresolved


# ---------------- main ----------------

def _write_run_config(run_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        ensure_dir(run_dir)
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Could not write run_config.json: {e}")


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        json_root = Path(args.json_root).expanduser().resolve()
        out_root = Path(args.out_root).expanduser().resolve() / "pathway_compare"
        ensure_dir(out_root); ensure_dir(out_root / "logs")

        raw = load_jsons(json_root)
        if not raw:
            logging.error(f"No valid JSON files found under {json_root}")
            return 2
        logging.info(f"Loaded {len(raw)} JSON file(s)")

        # Support multiple --pathways flags (append) AND comma-separated values
        inline_joined = ",".join(args.pathways or [])
        requested: List[str] = read_pathway_list(
            inline=inline_joined,
            file_path=args.pathways_file,
            csv_col=args.pathway_column
        )
        requested = [s for s in requested if str(s).strip()]
        if not requested:
            logging.error("No pathways specified. Use --pathways (repeatable) or --pathways-file.")
            return 2

        cfg = PCConfig(
            top_k=int(args.top_k),
            dpi=int(args.dpi),
            sig_cap=float(args.sig_cap),
            direction_mode=str(args.direction_mode).lower(),
            borrow_any_into_directional=str(args.borrow_any_into_directional).lower(),
            max_upset_intersections=int(args.max_upset_intersections),
            fail_on_missing_pathway=bool(args.fail_on_missing_pathway),
        )

        # build tidy (with entity cleaning controlled by cfg flags)
        pc_compute.cfg = cfg  # provide cfg to cleaner closure in compute.py
        entities = build_tidy_entities(raw, cfg.sig_cap)
        if entities is None or entities.empty:
            logging.error("No entities could be constructed from input JSONs.")
            return 2

        views = build_views_from_entities(entities, borrow=cfg.borrow_any_into_directional)

        # Resolve pathways **against actual labels present in the selected view**
        view_df = _select_view(views, cfg.direction_mode)
        if view_df is None or view_df.empty:
            logging.error("Selected view is empty after direction-mode filtering.")
            return 2

        all_labels, norm_index = _catalog_labels(view_df)
        resolved, unresolved = _resolve_against_labels(requested, all_labels, norm_index)

        if unresolved:
            logging.warning(f"Unresolved pathways ignored: {unresolved}")
            # helpful hints
            hints = {u: _closest(all_labels, u, n=6) for u in unresolved}
            for u, sugg in hints.items():
                if sugg:
                    logging.warning(f"Closest matches for '{u}': {sugg}")

        if not resolved:
            logging.error("None of the requested pathways could be resolved to JSON labels.")
            return 2

        logging.info(f"Resolved pathways (n={len(resolved)}): {resolved}")

        # per-pathway (always)
        run_individual_pathway_analysis(
            requested_pathways=resolved,
            views=views,
            out_root=out_root / "individual_analysis",
            cfg=cfg,
        )

        # combined (always)
        run_multi_pathway_comparison(
            requested_pathways=resolved,
            views=views,
            out_root=out_root / "comparison_analysis",
            cfg=cfg,
        )

        _write_run_config(out_root / "logs", {
            "json_root": str(json_root),
            "out_root": str(out_root),
            "requested": requested,
            "resolved": resolved,
            "missing": unresolved,
            "cfg": cfg.__dict__,
            "version": "pathway_compare.run/1.1.2",
        })

        logging.info("pathway_compare completed successfully.")
        return 0

    except Exception as e:
        logging.error("Fatal error in pathway_compare:\n" + "".join(traceback.format_exception(e)))
        return 2


if __name__ == "__main__":
    sys.exit(main())
