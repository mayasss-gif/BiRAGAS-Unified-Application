from __future__ import annotations
import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from .config import PCConfig
from .io_utils import (
    ensure_dir, load_jsons,
    read_pathways_from_cli_or_file, safe_write_csv, safe_write_xlsx,
    build_pathway_alias_map, resolve_requested_pathways_fuzzy
)
from .compute import build_pathway_gene_sets_per_disease, single_pathway_tables, multi_pathway_compare
from .plotting import jaccard_heatmap, coverage_heatmap, upset_like_from_sets
from .report import write_single_report, safe_name

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%-Y-%m-%d %H:%M:%S",
    )

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pathway_compare",
        description="Compare one or more pathways across diseases from MDP JSONs (direction-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json-root", required=True, help="Folder with per-disease JSON files")
    p.add_argument("--out-root",  required=True, help="Output folder")
    p.add_argument("--pathways", type=str, default="", help="Comma-separated pathway names")
    p.add_argument("--pathways-file", type=str, default="", help="TXT/CSV with pathway names (first column)")
    p.add_argument("--direction-mode", type=str, default="both", help="any|up|down|both")
    p.add_argument("--borrow-any-into-directional", type=str, default="up", help="none|up|down|both")
    p.add_argument("--top-intersections", type=int, default=30)
    p.add_argument("--min-intersection-size", type=int, default=1)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = PCConfig(
            dpi=int(args.dpi),
            verbose=int(args.verbose),
            direction_mode=str(args.direction_mode).lower(),
            borrow_any_into_directional=str(args.borrow_any_into_directional).lower(),
            top_intersections=int(args.top_intersections),
            min_intersection_size=int(args.min_intersection_size),
        )
        json_root = Path(args.json_root).expanduser().resolve()
        out_root  = Path(args.out_root).expanduser().resolve()

        ensure_dir(out_root)
        tbl_indiv   = out_root / "tables" / "pathway_compare" / "individual"
        tbl_compare = out_root / "tables" / "pathway_compare" / "compare"
        fig_indiv   = out_root / "figures" / "pathway_compare" / "individual"
        fig_compare = out_root / "figures" / "pathway_compare" / "compare"
        for d in (tbl_indiv, tbl_compare, fig_indiv, fig_compare):
            ensure_dir(d)

        # Load JSONs & pathway list
        raw = load_jsons(json_root)
        pathways = read_pathways_from_cli_or_file(args.pathways, args.pathways_file)
        if not pathways:
            logging.error("No pathways provided. Use --pathways or --pathways-file.")
            return 2

        # Resolve user-entered names robustly → canonical JSON labels
        alias_by_name, alias_by_id = build_pathway_alias_map(raw)
        resolved, unresolved, suggestions, mapping = resolve_requested_pathways_fuzzy(
            pathways, alias_by_name, alias_by_id,
            min_score=0.66, topk=5, enable_id_matching=True
        )

        if not resolved:
            logging.error("None of the requested pathways could be resolved to JSON labels.")
            for q, opts in (suggestions or {}).items():
                if opts:
                    logging.warning(f"Closest matches for '{q}': {opts[:3]}")
            return 2

        # Log mapping e.g. 'MAPK signaling pathway' → 'MAPK signaling pathway (KEGG:04010)'
        for q, lab in (mapping or {}).items():
            if q != lab:
                logging.info(f"[resolver] '{q}' → '{lab}'")
        if unresolved:
            logging.warning(f"Unresolved (ignored): {unresolved}")

        # Use resolved canonical labels from here on
        pathways = resolved

        # Build disease→pathway→genes mapping
        per_dis_pwgenes = build_pathway_gene_sets_per_disease(
            raw_json=raw,
            direction_mode=cfg.direction_mode,
            borrow_any_into_directional=cfg.borrow_any_into_directional
        )

        # Warn on very large disease count
        if len(per_dis_pwgenes) > cfg.max_diseases_warning:
            logging.warning(f"Large cohort ({len(per_dis_pwgenes)} diseases). Plots may be dense.")

        # ========== INDIVIDUAL ANALYSIS ==========
        for pw in pathways:
            logging.info(f"Analyzing pathway: {pw}")
            try:
                tabs = single_pathway_tables(pw, per_dis_pwgenes)
                # Write tables
                safe_write_csv(tabs["coverage"],        tbl_indiv / f"{safe_name(pw)}_coverage.csv")
                safe_write_csv(tabs["shared_counts"],   tbl_indiv / f"{safe_name(pw)}_shared_counts.csv")
                safe_write_csv(tabs["jaccard"],         tbl_indiv / f"{safe_name(pw)}_jaccard.csv")
                safe_write_csv(tabs["pvals"],           tbl_indiv / f"{safe_name(pw)}_overlap_pvals.csv")
                safe_write_csv(tabs["pairwise_differences"], tbl_indiv / f"{safe_name(pw)}_pairwise_differences.csv")

                # Figures
                figs = []
                fp = jaccard_heatmap(tabs["jaccard"], out=fig_indiv / f"{safe_name(pw)}_jaccard_heatmap", cfg=cfg,
                                     title=f"{pw}: Pairwise Jaccard (genes)"); fp and figs.append(fp)
                disease_sets = {d: per_dis_pwgenes[d].get(pw, set()) for d in per_dis_pwgenes.keys()}
                fp = upset_like_from_sets(disease_sets, title=f"{pw}: UpSet of gene intersections",
                                          out=fig_indiv / f"{safe_name(pw)}_upset", cfg=cfg,
                                          top_k=cfg.top_intersections, min_size=cfg.min_intersection_size)
                fp and figs.append(fp)

                # Report
                write_single_report(fig_indiv, pw,
                                    tabs["coverage"], tabs["shared_counts"], tabs["jaccard"], tabs["pvals"],
                                    figures=figs)
            except Exception as ee:
                logging.error(f"Failed pathway {pw}: {ee}")

        # ========== COMPARED ANALYSIS (ALL INPUT PATHWAYS) ==========
        try:
            comp = multi_pathway_compare(pathways, per_dis_pwgenes)
            # Tables
            safe_write_csv(comp["pathway_disease_matrix"], tbl_compare / "pathway_disease_matrix.csv")
            safe_write_csv(comp["disease_cosine_similarity"], tbl_compare / "disease_cosine_similarity.csv")

            # Figures
            _ = coverage_heatmap(comp["pathway_disease_matrix"], axis="pathway",
                                  out=fig_compare / "multi_pathway_heatmap",
                                  cfg=cfg, title="Coverage (# genes) per pathway×disease")
            _ = jaccard_heatmap(comp["disease_cosine_similarity"], out=fig_compare / "disease_cosine_heatmap",
                                 cfg=cfg, title="Disease Cosine Similarity (multi-pathway coverage)")
        except Exception as e:
            logging.error(f"Compared analysis failed: {e}")

        logging.info("Pathway comparison completed.")
        return 0
    except Exception as e:
        logging.error("Fatal error in pathway_compare:\n" + "".join(traceback.format_exception(e)))
        return 2

if __name__ == "__main__":
    sys.exit(main())
