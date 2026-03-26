# mdp_orchestrator.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .mdp_config import CONFIG, LICENSE_MODE, out_root
from .mdp_logging import info, warn, err, trace
from .mdp_io import load_table_auto, save_barh
from .mdp_enrichr import (
    latest_libs_by_patterns_any,
    latest_libs_by_patterns_human,
    filter_libs_for_ORA,
    enrichr_ora_human,
)

from .mdp_counts_deg import build_degs_from_counts_dir
from .mdp_deg_gsea import (
    build_rank_from_degs,
    split_up_down_from_degs,
    write_delta_vs_consensus,
)
from .mdp_enrichr import gsea_prerank, leading_edge_from_gsea
from .mdp_baseline import build_consensus_expectations
from .mdp_viper_ulm import viper_from_counts, viper_from_degs_fallback, run_ulm_panels
from .mdp_cross import compare_two, multi_cohort_union_and_shared


# ---------- helper: auto-pick a DEG file from a folder if counts are missing ----------
def _find_deg_file_in_dir(degs_dir: Path) -> Path | None:
    """
    Return the first plausible DEG file under 'degs_dir'.
    Accepts .csv/.tsv/.tab/.xlsx/.xls and requires at least:
      - a gene column, and
      - a statistic/score column (e.g., log2FC, stat, pvalue, etc.)
    """
    try:
        degs_dir = Path(degs_dir)
        if not degs_dir.exists() or not degs_dir.is_dir():
            return None

        candidates: list[Path] = []
        for ext in ("*.csv", "*.tsv", "*.tab", "*.xlsx", "*.xls"):
            candidates.extend(sorted(degs_dir.glob(ext)))

        for p in candidates:
            try:
                if p.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(p, nrows=100)
                else:
                    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
                    df = pd.read_csv(p, sep=sep, nrows=100)
            except Exception:
                continue
            if df is None or df.empty:
                continue

            cols = {c.lower(): c for c in df.columns}
            gene_col = next((cols[k] for k in (
                "gene", "symbol", "gene_symbol", "hgnc_symbol", "ensembl",
                "ensembl_id", "id"
            ) if k in cols), None)
            stat_col = next((cols[k] for k in (
                "log2fc", "log2foldchange", "score", "stat", "t", "wald",
                "beta", "pvalue", "p-value", "padj", "fdr"
            ) if k in cols), None)

            if gene_col and stat_col:
                return p
    except Exception:
        return None

    return None
# -------------------------------------------------------------------------------------


def _ora_and_plot_human(
    outdir: Path,
    genes: List[str],
    libs: list[str],
    tag: str,
    plot_title: str,
    plot_file: str,
    topn: int,
) -> pd.DataFrame:
    if not genes or not libs:
        return pd.DataFrame()
    df = enrichr_ora_human(genes, libs)
    if not df.empty:
        out_csv = outdir / f"{tag}.csv"
        try:
            df.to_csv(out_csv, index=False)
        except Exception as e:
            warn(f"Failed to write {out_csv.name}: {trace(e)}")
        save_barh(
            df,
            "term",
            "qval",
            plot_title,
            outdir / f"plots/{plot_file}",
            top_n=topn,
            dpi=CONFIG["FIG_DPI"],
        )
    return df


def process_one_cohort(
    coh: dict,
    baseline_expectations_by_tissue: dict[str, Path],
) -> dict:
    
    t_start = time.perf_counter()
    dt_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = coh["name"]
    root = out_root()
    outdir = root / name
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    n_total = n_case = n_ctrl = 0
    groups = None

    # 1) Preferred path: counts_dir -> build DEGs automatically
    if (not coh.get("degs_file")) and coh.get("counts_dir"):
        try:
            auto_degs, auto_counts, cc_path, degs_path, groups = build_degs_from_counts_dir(coh)
            coh["degs_file"] = str(degs_path)
            coh["expr_file"] = str(cc_path)
            info(f"[{name}] Auto-DE complete; using {degs_path} and {cc_path}")
            if groups is not None:
                n_total = len(groups)
                n_case = int((groups == "case").sum())
                n_ctrl = int((groups == "control").sum())
        except Exception as e:
            warn(f"[{name}] Auto-DE from counts_dir failed: {trace(e)}")

    # 2) NEW fallback: if no counts and no explicit degs_file, look inside degs_dir
    if not coh.get("counts_dir") and not coh.get("degs_file") and coh.get("degs_dir"):
        try:
            candidate = _find_deg_file_in_dir(Path(coh["degs_dir"]))
            if candidate:
                coh["degs_file"] = str(candidate)
                info(f"[{name}] Using DEG file discovered in degs_dir: {candidate}")
            else:
                warn(f"[{name}] No suitable DEG file found under degs_dir={coh['degs_dir']}")
        except Exception as e:
            warn(f"[{name}] degs_dir fallback failed: {trace(e)}")

    rnk = None
    df_degs = None
    df_counts = None
    degs_path = coh.get("degs_file")
    expr_path = coh.get("expr_file")
    id_col = coh.get("id_col") or "Gene"
    lfc_col = coh.get("lfc_col") or "log2FC"
    q_col = (coh.get("q_col") or None) or None
    q_max = coh.get("q_max")

    # Build ranked list from DEGs if available
    if degs_path and Path(degs_path).exists():
        try:
            df_degs = load_table_auto(degs_path)
            info(f"[{name} DEGs] OK: shape={df_degs.shape} first_cols={list(df_degs.columns)[:6]}")
            rnk = build_rank_from_degs(df_degs, id_col, lfc_col, q_col, q_max)
            if rnk is not None and not rnk.dropna().empty:
                rnk.sort_values(ascending=False).reset_index().to_csv(
                    outdir / "ranked_from_DEGs.csv",
                    index=False,
                )
                info(f"[{name}] Ranked list from DEGs: {rnk.shape[0]} genes.")
        except Exception as e:
            warn(f"[{name}] DEGs rank build failed: {trace(e)}")

    # Load counts (if present) for VIPER
    if expr_path and Path(expr_path).exists():
        try:
            df_counts = load_table_auto(expr_path)
        except Exception as e:
            warn(f"[{name}] Could not load counts: {trace(e)}")

    if rnk is None or rnk.dropna().empty:
        raise RuntimeError(f"[{name}] Could not build any ranked list.")

    # GSEA prerank on core pathway libs
    core_libs_any = CONFIG["CORE_PATHWAY_LIBS"][:]
    gsea_df, gsea_path = gsea_prerank(
        rnk, core_libs_any, outdir, figdpi=CONFIG["FIG_DPI"]
    )

    # Split up/down (if DEG table is available)
    up_genes: List[str] = []
    down_genes: List[str] = []
    if df_degs is not None and not df_degs.empty:
        try:
            up_genes, down_genes, _, _ = split_up_down_from_degs(
                df_degs, id_col, lfc_col, q_col, q_max
            )
            info(f"[{name}] UP genes: {len(up_genes)} | DOWN genes: {len(down_genes)}")
        except Exception as e:
            warn(f"[{name}] split_up_down_from_degs failed: {trace(e)}")

    # ORA per category (human)
    immune_libs_h = latest_libs_by_patterns_human(CONFIG["IMMUNE_PATTERNS"], max_libs=6)
    epi_libs_h = latest_libs_by_patterns_human(CONFIG["EPIGENETIC_PATTERNS"], max_libs=6)
    tf_libs_h = latest_libs_by_patterns_human(CONFIG["TF_PATTERNS"], max_libs=6)
    met_libs_h = filter_libs_for_ORA(CONFIG["METABOLIC_LIBS"])
    core_libs_h = filter_libs_for_ORA(CONFIG["CORE_PATHWAY_LIBS"])

    _ora_and_plot_human(
        outdir,
        up_genes,
        core_libs_h,
        "core_enrich_up",
        f"Pathways (UP, Human) — {name}",
        "core_pathways_up.png",
        CONFIG["TOP_N_PATHWAYS"],
    )
    _ora_and_plot_human(
        outdir,
        down_genes,
        core_libs_h,
        "core_enrich_down",
        f"Pathways (DOWN, Human) — {name}",
        "core_pathways_down.png",
        CONFIG["TOP_N_PATHWAYS"],
    )
    _ora_and_plot_human(
        outdir,
        up_genes,
        immune_libs_h,
        "immune_enrich_up",
        f"Immune (UP, Human) — {name}",
        "immune_up.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        down_genes,
        immune_libs_h,
        "immune_enrich_down",
        f"Immune (DOWN, Human) — {name}",
        "immune_down.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        up_genes,
        epi_libs_h,
        "epigenetic_enrich_up",
        f"Epigenetic (UP, Human) — {name}",
        "epigenetic_up.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        down_genes,
        epi_libs_h,
        "epigenetic_enrich_down",
        f"Epigenetic (DOWN, Human) — {name}",
        "epigenetic_down.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        up_genes,
        met_libs_h,
        "metabolite_enrich_up",
        f"Metabolites (UP, HMDB/Human) — {name}",
        "metabolite_up.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        down_genes,
        met_libs_h,
        "metabolite_enrich_down",
        f"Metabolites (DOWN, HMDB/Human) — {name}",
        "metabolite_down.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        up_genes,
        tf_libs_h,
        "tf_enrich_up",
        f"TF (UP, Human) — {name}",
        "tf_up.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )
    _ora_and_plot_human(
        outdir,
        down_genes,
        tf_libs_h,
        "tf_enrich_down",
        f"TF (DOWN, Human) — {name}",
        "tf_down.png",
        CONFIG["TOP_N_IMMUNO_EPI"],
    )

    # Leading-edge ORA per pathway (optional)
    try:
        le = leading_edge_from_gsea(gsea_df)
        rows_imm, rows_epi, rows_met, rows_tf = [], [], [], []
        for term, genes in le.items():
            if immune_libs_h:
                _imm = enrichr_ora_human(genes, immune_libs_h)
                if not _imm.empty:
                    rows_imm.append(_imm.assign(pathway=term))
            if epi_libs_h:
                _epi = enrichr_ora_human(genes, epi_libs_h)
                if not _epi.empty:
                    rows_epi.append(_epi.assign(pathway=term))
            if met_libs_h:
                _met = enrichr_ora_human(genes, met_libs_h)
                if not _met.empty:
                    rows_met.append(_met.assign(pathway=term))
            if tf_libs_h:
                _tf = enrichr_ora_human(genes, tf_libs_h)
                if not _tf.empty:
                    rows_tf.append(_tf.assign(pathway=term))
        if rows_imm:
            pd.concat(rows_imm, ignore_index=True).to_csv(
                outdir / "immune_enrich_per_pathway.tsv", sep="\t", index=False
            )
        if rows_epi:
            pd.concat(rows_epi, ignore_index=True).to_csv(
                outdir / "epigenetic_enrich_per_pathway.tsv", sep="\t", index=False
            )
        if rows_met:
            pd.concat(rows_met, ignore_index=True).to_csv(
                outdir / "metabolite_enrich_per_pathway.tsv", sep="\t", index=False
            )
        if rows_tf:
            pd.concat(rows_tf, ignore_index=True).to_csv(
                outdir / "tf_enrich_per_pathway.tsv", sep="\t", index=False
            )
    except Exception as e:
        warn(f"[{name}] leading-edge ORA failed: {trace(e)}")

    # VIPER
    tf_scores = pd.DataFrame()
    did_viper = False
    if df_counts is not None and not df_counts.empty:
        try:
            gcol = coh.get("id_col") or "Gene"
            gcol_resolved = None
            for c in df_counts.columns:
                if c.lower() == gcol.lower():
                    gcol_resolved = c
                    break
            if gcol_resolved is None:
                gcol_resolved = df_counts.columns[0]
            d = df_counts.copy()
            d[gcol_resolved] = d[gcol_resolved].astype(str).str.upper().str.strip()
            d = d.dropna(subset=[gcol_resolved]).drop_duplicates(subset=[gcol_resolved], keep="first")
            sample_cols = [c for c in d.columns if c != gcol_resolved]
            for c in sample_cols:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
            counts = d.set_index(gcol_resolved)
            tf_scores = viper_from_counts(
                counts,
                license_mode=CONFIG["VIPER_LICENSE"],
                tmin=CONFIG["VIPER_TMIN"],
                pleiotropy=CONFIG["VIPER_PLEIOTROPY"],
            )
            did_viper = not tf_scores.empty
        except Exception as e:
            warn(f"[{name}] VIPER (counts) failed: {trace(e)}")

    if not did_viper and df_degs is not None and not df_degs.empty:
        try:
            tf_scores = viper_from_degs_fallback(rnk.dropna())
            did_viper = not tf_scores.empty
        except Exception as e:
            warn(f"[{name}] VIPER (DEG fallback) failed: {trace(e)}")

    if did_viper and not tf_scores.empty:
        try:
            tf_scores.to_csv(outdir / "viper_tf_scores.tsv", sep="\t")
            sub = tf_scores.copy()
            sub = (sub - sub.mean(axis=0)) / (sub.std(axis=0).replace(0, np.nan))
            var = sub.var(axis=0).sort_values(ascending=False)
            keep = var.head(min(CONFIG["TF_HEATMAP_TOP_N"], len(var))).index
            sub = sub.loc[:, keep].fillna(0.0)
            if not sub.empty:
                plt.figure(figsize=(max(6, sub.shape[1] * 0.3), max(4, sub.shape[0] * 0.25)))
                plt.imshow(sub.values, aspect="auto", interpolation="nearest")
                plt.colorbar(label="z-score")
                plt.yticks(range(sub.shape[0]), sub.index, fontsize=7)
                plt.xticks(
                    range(sub.shape[1]),
                    [c.replace("TF:", "") for c in sub.columns],
                    rotation=90,
                    fontsize=7,
                )
                plt.title(f"TF activity (VIPER) — {name}")
                plt.tight_layout()
                plt.savefig(outdir / "plots/tf_heatmap.png", dpi=CONFIG["FIG_DPI"])
                plt.close()
        except Exception as e:
            warn(f"[{name}] TF heatmap failed: {trace(e)}")

    # ULM panels
    try:
        run_ulm_panels(outdir, df_degs, lfc_col, license_mode=LICENSE_MODE)
    except Exception as e:
        warn(f"[ULM] panels skipped: {trace(e)}")

    # Delta vs consensus (optional)
    delta_consensus_path = None
    tissue = coh.get("tissue")
    if tissue and tissue in baseline_expectations_by_tissue:
        try:
            delta_consensus_path = write_delta_vs_consensus(
                gsea_path=Path(gsea_path),
                expectations_tsv=baseline_expectations_by_tissue[tissue],
                out_path=outdir / "delta_vs_consensus.tsv",
            )
        except Exception as e:
            warn(f"[{name}] delta vs consensus failed: {trace(e)}")

    elapsed = time.perf_counter() - t_start
    info(
        f"[metrics] {name}: samples total={n_total} (case={n_case}, control={n_ctrl}) "
        f"| elapsed={elapsed:.2f}s"
    )

    return {
        "name": name,
        "outdir": str(outdir),
        "gsea_summary": gsea_path,
        "core_up": str(outdir / "core_enrich_up.csv"),
        "core_down": str(outdir / "core_enrich_down.csv"),
        "immune_up": str(outdir / "immune_enrich_up.csv"),
        "immune_down": str(outdir / "immune_enrich_down.csv"),
        "epigenetic_up": str(outdir / "epigenetic_enrich_up.csv"),
        "epigenetic_down": str(outdir / "epigenetic_enrich_down.csv"),
        "metabolite_up": str(outdir / "metabolite_enrich_up.csv"),
        "metabolite_down": str(outdir / "metabolite_enrich_down.csv"),
        "tf_up": str(outdir / "tf_enrich_up.csv"),
        "tf_down": str(outdir / "tf_enrich_down.csv"),
        "delta_vs_consensus": str(delta_consensus_path) if delta_consensus_path else "",
        "n_total": n_total,
        "n_case": n_case,
        "n_ctrl": n_ctrl,
        "start_time": dt_start,
        "elapsed_sec": f"{elapsed:.2f}",
    }


def run_all() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    root = out_root()
    root.mkdir(parents=True, exist_ok=True)

    # Build baseline expectations (optional)
    try:
        tissues = sorted({c.get("tissue") for c in CONFIG["COHORTS"] if c.get("tissue")})
        baseline_expectations_by_tissue = build_consensus_expectations(
            out_root=root,
            data_dir=Path(CONFIG["DATA_DIR"]),
            hpa_file=CONFIG["HPA_FILE"],
            gtex_file=CONFIG["GTEX_FILE"],
            fantom_file=CONFIG["FANTOM_FILE"],
            pathway_libs=CONFIG["CORE_PATHWAY_LIBS"],
            tissues=tissues,
        )
    except Exception as e:
        warn(f"Consensus baseline step failed: {trace(e)}")
        baseline_expectations_by_tissue = {}

    # Report which libs the GSEA scope will touch
    immune_libs_any = latest_libs_by_patterns_any(CONFIG["IMMUNE_PATTERNS"], max_libs=6)
    epi_libs_any = latest_libs_by_patterns_any(CONFIG["EPIGENETIC_PATTERNS"], max_libs=6)
    tf_libs_any = latest_libs_by_patterns_any(CONFIG["TF_PATTERNS"], max_libs=6)
    met_libs_any = CONFIG["METABOLIC_LIBS"][:]
    info(f"[immune] (GSEA scope) libs: {immune_libs_any or '(none)'}")
    info(f"[epigenetic] (GSEA scope) libs: {epi_libs_any or '(none)'}")
    info(f"[tf] (GSEA scope) libs: {tf_libs_any or '(none)'}")
    info(f"[metabolite] (GSEA scope) libs: {met_libs_any or '(none)'}")

    results: List[dict] = []
    metrics_rows: List[Dict] = []
    for coh in CONFIG["COHORTS"]:
        try:
            res = process_one_cohort(coh, baseline_expectations_by_tissue)
            results.append(res)
            metrics_rows.append(
                {
                    "cohort": res["name"],
                    "start_time": res["start_time"],
                    "elapsed_sec": res["elapsed_sec"],
                    "n_samples_total": res["n_total"],
                    "n_case": res["n_case"],
                    "n_control": res["n_ctrl"],
                }
            )
            info(f"[cohort] done -> {coh['name']}")
        except Exception as e:
            warn(f"[cohort] {coh.get('name', '?')} failed but continuing: {trace(e)}")

    # Pairwise compare (optional)
    try:
        if len(results) >= 2:
            comp_dir = root / "comparison"
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    A, B = results[i], results[j]
                    compare_two(
                        A["gsea_summary"],
                        B["gsea_summary"],
                        A["name"],
                        B["name"],
                        comp_dir,
                    )
            info(f"[compare] outputs -> {comp_dir}")
        else:
            info("[compare] Single cohort; skipping.")
    except Exception as e:
        warn(f"Pairwise comparison failed: {trace(e)}")

    # Multi-cohort union
    try:
        comp_dir = root / "comparison"
        multi_cohort_union_and_shared(
            results,
            comp_dir,
            require_significant=CONFIG.get("PRESENCE_REQUIRE_SIGNIFICANT", False),
            fdr_threshold=CONFIG.get("FDR_Q", 0.05),
        )
    except Exception as e:
        warn(f"All-cohort union failed: {trace(e)}")

    # Metrics
    try:
        if metrics_rows:
            mdf = pd.DataFrame(metrics_rows)
            mdf.to_csv(root / "cohort_metrics.tsv", sep="\t", index=False)
            info(f"[metrics] wrote: {root / 'cohort_metrics.tsv'}")
    except Exception as e:
        warn(f"Writing metrics failed: {trace(e)}")

    info(f"[DONE] All outputs in: {root}")
