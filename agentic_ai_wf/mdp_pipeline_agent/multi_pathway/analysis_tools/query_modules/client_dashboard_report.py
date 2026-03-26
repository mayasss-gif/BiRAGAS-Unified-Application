#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .category_landscape_compare import run_category_landscape_compare
from .enzyme_signaling_compare import run_enzyme_signaling_compare
from .pathway_entity_compare import run_pathway_entity_compare
from .tf_activity_compare import run_tf_activity_compare


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _keep_existing_paths(d: Dict[str, str]) -> Dict[str, str]:
    """
    Keep only outputs that actually exist on disk.
    This prevents 'success' messages that point to nowhere.
    """
    out: Dict[str, str] = {}
    for k, v in d.items():
        try:
            if v and Path(v).expanduser().exists():
                out[k] = v
        except Exception:
            continue
    return out


def run_client_dashboard_report(root: str, out: str, sig: float = 0.10, cap: int = 300) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = _ensure_dir(Path(out).expanduser().resolve())

    res: Dict[str, str] = {}

    # 1) Category landscape
    try:
        cat_out = _ensure_dir(out_p / "category_landscape")
        r1 = run_category_landscape_compare(
            root=str(root_p),
            out=str(cat_out),
            sig=sig,
            cap=cap,
            cluster=True,
        )
        r1 = _keep_existing_paths({f"category.{k}": v for k, v in (r1 or {}).items()})
        if not r1:
            print("[warn] category_landscape_compare produced no outputs (check classified files + sig/cap).")
        res.update(r1)
    except Exception as e:
        print(f"[warn] category_landscape_compare failed: {e}")

    # 2) Enzyme + signaling
    try:
        enz_out = _ensure_dir(out_p / "enzyme_signaling")
        r2 = run_enzyme_signaling_compare(
            root=str(root_p),
            out=str(enz_out),
            top_n=30,
            cluster=True,
        )
        r2 = _keep_existing_paths({f"enzyme.{k}": v for k, v in (r2 or {}).items()})
        if not r2:
            print("[warn] enzyme_signaling_compare produced no outputs (maybe no Enzyme_and_Signaling folder).")
        res.update(r2)
    except Exception as e:
        print(f"[warn] enzyme_signaling_compare failed: {e}")

    # 3) Pathway-entity overlap dashboard (JSONs)
    try:
        pe_out = _ensure_dir(out_p / "pathway_entity")
        r3 = run_pathway_entity_compare(
            root=str(root_p),
            out=str(pe_out),
            fdr_max=sig,
            top_pathways=200,
        )
        r3 = _keep_existing_paths({f"pathway.{k}": v for k, v in (r3 or {}).items()})
        if not r3:
            print("[warn] pathway_entity_compare produced no outputs (check JSON bundle rules).")
        res.update(r3)
    except Exception as e:
        print(f"[warn] pathway_entity_compare failed: {e}")

    # 4) TF activity compare (VIPER/ULM)
    try:
        tf_out = _ensure_dir(out_p / "tf_activity")
        r4 = run_tf_activity_compare(
            root=str(root_p),
            out=str(tf_out),
            top_n=30,   # FIXED: was top_k
            cluster=True,
        )
        r4 = _keep_existing_paths({f"tf.{k}": v for k, v in (r4 or {}).items()})
        if not r4:
            print("[warn] tf_activity_compare produced no outputs (no viper_tf_scores.tsv / ulm_collectri_tf_scores.tsv).")
        res.update(r4)
    except Exception as e:
        print(f"[warn] tf_activity_compare failed: {e}")

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sig", type=float, default=0.10)
    ap.add_argument("--cap", type=int, default=300)
    args = ap.parse_args()

    res = run_client_dashboard_report(args.root, args.out, sig=args.sig, cap=args.cap)

    print("[DONE] Wrote dashboard outputs:")
    for k, v in sorted(res.items()):
        print(f"  - {k}: {v}")

    if not res:
        raise SystemExit(
            "No outputs produced.\n"
            "Most common causes:\n"
            "  - wrong --root\n"
            "  - no classified files (gsea_prerank_classified.tsv / core_enrich_*_classified.csv)\n"
            "  - JSON bundle not found per your strict rules\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
