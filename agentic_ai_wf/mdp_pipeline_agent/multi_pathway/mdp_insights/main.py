from __future__ import annotations
import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import InsightsConfig
from .io_utils import ensure_dir, load_jsons, safe_write_csv
from .processing import (
    tidy_from_json_obj, build_views, presence_from_entities,
    summarize_shared, summarize_individual, build_presence_matrix,
    top200_pathways_cross_disease, top200_targets_by_type, directional_concordance_table,
    disease_interconnection_table,
    apply_epigenetic_suffix_filter,          # already added before
    apply_tf_mouse_filter                     # ← NEW
)

from .plotting import (
    bar_top, heatmap_matrix,
    shared_presence_heatmap, shared_upset_like, shared_entity_mix_bars,
    shared_genes_count_bar, disease_similarity_heatmap, pathway_cooccurrence_heatmap,
    grouped_entity_mix_per_pathway, heatmap_entity_mix_per_pathway, bar_genes_per_pathway,
    hist_metric_per_type, box_metric_per_type, scatter_pss_vs_n,
    # NEW shared visuals:
    shared_up_significance_heatmap, shared_pathway_leaderboard, shared_target_volcano
)
from .report import build_report


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # Windows strftime does not support "%-Y"; use portable variant
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_insights(json_root: Path, out_root: Path, cfg: InsightsConfig) -> None:
    # ---- ensure output structure
    ensure_dir(out_root)
    ensure_dir(out_root / "tables")
    ensure_dir(out_root / "figures" / "individual")
    ensure_dir(out_root / "figures" / "shared")

    # -------- Load & normalize --------
    raw = load_jsons(json_root)  # {disease: dict}
    all_entities = []
    for disease, js in raw.items():
        df = tidy_from_json_obj(disease, js, cfg.sig_cap)
        if not df.empty:
            all_entities.append(df)
    entities_df = (
        pd.concat(all_entities, ignore_index=True)
        if all_entities else
        pd.DataFrame(columns=[
            "disease","pathway","direction","entity_type","entity",
            "overlap_genes","OR","qval","pval","sig","k","a","b","N","Jaccard"
        ])
    )
    # AFTER entities_df is created
    entities_df = apply_epigenetic_suffix_filter(entities_df, cfg)
    entities_df = apply_tf_mouse_filter(entities_df, cfg)
    # presence (UP/DOWN/ANY) from entities
    presence_df = presence_from_entities(entities_df)

    # persist raw long tables
    safe_write_csv(entities_df, out_root / "tables" / "entities_long.csv")
    safe_write_csv(presence_df, out_root / "tables" / "presence_long.csv")

    # -------- Build views (ANY, UP, DOWN) --------
    # Tip: set cfg.borrow_any_into_directional="up" to treat non-directional as UP.
    views = build_views(entities_df, borrow=cfg.borrow_any_into_directional)
    any_df  = views["ANY"]
    up_df   = views["UP"]
    down_df = views["DOWN"]

    # -------- Shared summary & individual summaries --------
    shared_tbl = summarize_shared(any_df, presence_df, cfg)
    safe_write_csv(shared_tbl, out_root / "tables" / "shared_pathway_summary.csv")

    indiv_tables = summarize_individual(any_df, cfg)

    # -------- Top-200 intelligence tables --------
    mat_any = build_presence_matrix(presence_df, "ANY")

    top200_pw = top200_pathways_cross_disease(
        views_any=any_df[["pathway","disease"]].drop_duplicates(),
        entities_any=any_df, cfg=cfg
    )
    safe_write_csv(top200_pw, out_root / "tables" / "top200_pathways_cross_disease.csv")

    for et in ("tf","epigenetic","metabolites"):
        tbl = top200_targets_by_type(any_df, et, cfg)
        safe_write_csv(tbl, out_root / "tables" / f"top200_targets_{et}.csv")

    if not up_df.empty or not down_df.empty:
        dir_df = pd.concat([up_df, down_df], ignore_index=True)
        dir_conc = directional_concordance_table(dir_df, cfg)
        safe_write_csv(dir_conc, out_root / "tables" / "directional_concordance_by_pathway.csv")

    inter_tbl = disease_interconnection_table(mat_any)
    safe_write_csv(inter_tbl, out_root / "tables" / "disease_interconnection.csv")

    # -------- Figures containers --------
    all_figs: Dict[str, List[Path]] = {"shared": []}
    shared_dir = out_root / "figures" / "shared"
    indiv_dir = out_root / "figures" / "individual"

    # -------- Figures (shared) --------
    # ANY presence heatmap (baseline)
    fp = shared_presence_heatmap(mat_any, out=shared_dir / "shared_presence_heatmap", cfg=cfg);  fp and all_figs["shared"].append(fp)

    # NEW: UP significance heatmap (ANY→UP borrowing applied via up_df if configured)
    fp = shared_up_significance_heatmap(up_df, out=shared_dir / "shared_up_significance_heatmap", cfg=cfg);  fp and all_figs["shared"].append(fp)

    # UpSet-like (by # diseases) and entity mix bars
    memberships: Dict[str, set] = {}
    if not shared_tbl.empty:
        for _, r in shared_tbl.iterrows():
            memberships[str(r["pathway"])] = set(
                s.strip() for s in str(r["diseases"]).split(",") if s.strip()
            )
    fp = shared_upset_like(memberships, out=shared_dir / "shared_upset_like", cfg=cfg);  fp and all_figs["shared"].append(fp)
    fp = shared_entity_mix_bars(shared_tbl, out=shared_dir / "shared_entity_mix_bars", cfg=cfg);  fp and all_figs["shared"].append(fp)
    fp = shared_genes_count_bar(shared_tbl, out=shared_dir / "shared_genes_count", cfg=cfg);  fp and all_figs["shared"].append(fp)

    # Disease similarity & pathway co-occurrence (ANY view)
    fp = disease_similarity_heatmap(mat_any, out=shared_dir / "shared_disease_similarity", cfg=cfg);  fp and all_figs["shared"].append(fp)
    fp = pathway_cooccurrence_heatmap(mat_any, shared_tbl, out=shared_dir / "shared_pathway_cooccurrence", cfg=cfg);  fp and all_figs["shared"].append(fp)

    # NEW: Pathway Intelligence Leaderboard (Top by PIS)
    fp = shared_pathway_leaderboard(top200_pw, out=shared_dir / "shared_pathway_leaderboard", cfg=cfg);  fp and all_figs["shared"].append(fp)

    # NEW: Target volcano plots (per type) from ANY view
    for et in ("tf","epigenetic","metabolites"):
        fp = shared_target_volcano(any_df, et, out=shared_dir / f"shared_target_volcano_{et}", cfg=cfg)
        fp and all_figs["shared"].append(fp)

    # -------- Figures (individual) --------
    for disease, parts in indiv_tables.items():
        figs_d: List[Path] = []

        fp = bar_top(parts.get("top_pathways", pd.DataFrame()),
                     x="pathway", y="pss",
                     title=f"{disease}: Top Pathways by PSS",
                     out=indiv_dir / f"{disease}_top_pathways_pss",
                     cfg=cfg);  fp and figs_d.append(fp)

        for name, key in [("TFs","top_tf"), ("Epigenetic","top_epigenetic"), ("Metabolites","top_metabolites")]:
            fp = bar_top(parts.get(key, pd.DataFrame()),
                         x="entity", y="count",
                         title=f"{disease}: Top {name} (count)",
                         out=indiv_dir / f"{disease}_top_{name.lower()}",
                         cfg=cfg);  fp and figs_d.append(fp)

        if "gene_leaders" in parts and isinstance(parts["gene_leaders"], pd.DataFrame):
            fp = bar_top(parts["gene_leaders"], x="gene", y="count",
                         title=f"{disease}: Top Overlap Genes",
                         out=indiv_dir / f"{disease}_top_genes",
                         cfg=cfg);  fp and figs_d.append(fp)

        mix = parts.get("pathway_type_counts", pd.DataFrame())
        fp = grouped_entity_mix_per_pathway(mix,
                title=f"{disease}: Entity-Type Mix per Pathway",
                out=indiv_dir / f"{disease}_mix_groupedbars",
                cfg=cfg);  fp and figs_d.append(fp)
        fp = heatmap_entity_mix_per_pathway(mix,
                title=f"{disease}: Entity-Type Mix Heatmap",
                out=indiv_dir / f"{disease}_mix_heatmap",
                cfg=cfg);  fp and figs_d.append(fp)

        gene_counts = parts.get("pathway_gene_counts", pd.DataFrame())
        fp = bar_genes_per_pathway(gene_counts,
                title=f"{disease}: Union(genes) per Pathway",
                out=indiv_dir / f"{disease}_genes_per_pathway",
                cfg=cfg);  fp and figs_d.append(fp)

        qvals = parts.get("qvals_per_type", pd.DataFrame())
        fp = hist_metric_per_type(qvals, value_col="qval",
                title=f"{disease}: q-value Distribution by Entity Type",
                out=indiv_dir / f"{disease}_qval_hist",
                cfg=cfg);  fp and figs_d.append(fp)
        fp = box_metric_per_type(qvals, value_col="qval",
                title=f"{disease}: q-value Boxplot by Entity Type",
                out=indiv_dir / f"{disease}_qval_box",
                cfg=cfg);  fp and figs_d.append(fp)

        orvals = parts.get("or_per_type", pd.DataFrame())
        fp = hist_metric_per_type(orvals, value_col="OR",
                title=f"{disease}: OR Distribution by Entity Type",
                out=indiv_dir / f"{disease}_or_hist",
                cfg=cfg);  fp and figs_d.append(fp)
        fp = box_metric_per_type(orvals, value_col="OR",
                title=f"{disease}: OR Boxplot by Entity Type",
                out=indiv_dir / f"{disease}_or_box",
                cfg=cfg);  fp and figs_d.append(fp)

        top_pw = parts.get("top_pathways", pd.DataFrame())
        fp = scatter_pss_vs_n(top_pw,
                title=f"{disease}: PSS vs n_entities",
                out=indiv_dir / f"{disease}_pss_vs_nentities",
                cfg=cfg);  fp and figs_d.append(fp)

        all_figs[disease] = figs_d

    # -------- Report --------
    build_report(out_root=out_root,
                 shared_tbl=shared_tbl,
                 indiv=indiv_tables,
                 figures=all_figs)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mdp_insights",
        description="Generate publication-ready MDP insights (direction-aware, significance-aware) from JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json-root", required=True, help="Folder with per-disease JSON files")
    p.add_argument("--out-root",  required=True, help="Output folder")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--exclude-hmdb-in-shared", type=str, default="false")
    p.add_argument("--hub-cap", type=int, default=30)
    p.add_argument("--min-shared-gene-intersection", type=int, default=2)
    # NEW:
    p.add_argument("--sig-cap", type=float, default=300.0)
    p.add_argument("--min-sig-for-label", type=float, default=2.0)
    p.add_argument("--direction-mode", type=str, default="both")
    p.add_argument("--borrow-any-into-directional", type=str, default="up")  # default UP borrowing as requested
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args(argv)


def str2bool(v: str) -> bool:
    try:
        return str(v).strip().lower() in {"1","true","t","yes","y"}
    except Exception:
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = InsightsConfig(
            top_k=int(args.top_k),
            exclude_hmdb_in_shared=str2bool(args.exclude_hmdb_in_shared),
            hub_cap=int(args.hub_cap),
            min_shared_gene_intersection=int(args.min_shared_gene_intersection),
            dpi=300,
            sig_cap=float(args.sig_cap),
            min_sig_for_label=float(args.min_sig_for_label),
            direction_mode=str(args.direction_mode).lower(),
            borrow_any_into_directional=str(args.borrow_any_into_directional).lower(),
            top_n=int(args.top_n),
        )
        run_insights(
            json_root=Path(args.json_root).expanduser().resolve(),
            out_root=Path(args.out_root).expanduser().resolve(),
            cfg=cfg,
        )
        logging.info("MDP Insights completed successfully.")
        return 0
    except Exception as e:
        logging.error("Fatal error in MDP Insights:\n" + "".join(traceback.format_exception(e)))
        return 2


if __name__ == "__main__":
    sys.exit(main())
