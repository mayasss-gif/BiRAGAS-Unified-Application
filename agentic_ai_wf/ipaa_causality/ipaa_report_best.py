# report/ipaa_report_best.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .ipaa_report_data import (
    CohortData,
    discover_cohorts,
    load_cohort,
    read_json,
    safe_mkdir,
    write_text,
    deep_find_first,
)
from .ipaa_report_case_studies import (
    build_case_studies_html,
    build_static_case_studies_markdown,
    rank_shared_pathways,
    significant_set,
)


def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    if max_rows and df.shape[0] > max_rows:
        df = df.head(int(max_rows))
    return df.to_markdown(index=False) + "\n"


def generate_report(out_root: Path, sig_fdr: float, top_n: int, top_limit: int = 50) -> None:
    out_root = Path(out_root).expanduser().resolve()
    report_dir = out_root / "report"
    safe_mkdir(report_dir)

    # discover cohorts
    names = discover_cohorts(out_root)
    if len(names) < 2:
        raise RuntimeError(f"Need at least 2 cohorts. Found: {names}")

    # load cohorts
    cohorts: Dict[str, CohortData] = {n: load_cohort(out_root, n) for n in names}

    # run-level knobs (best effort)
    pipe_manifest = read_json(out_root / "PIPELINE_MANIFEST.json") or {}
    gsea_perm = deep_find_first(pipe_manifest, ["gsea_permutations", "permutation_num", "permutations"])
    threads = deep_find_first(pipe_manifest, ["threads", "n_threads", "threads_per_cohort"])

    # cohort overview table
    overview_rows = []
    diag_rows = []
    for n in names:
        c = cohorts[n]
        cm = read_json(c.cohort_dir / "COHORT_MANIFEST.json") or {}
        tissue = ""
        tissue_method = ""
        counts_mode = ""
        n_samples = ""
        n_genes = ""

        if isinstance(cm, dict):
            n_samples = str(cm.get("n_samples", cm.get("samples", "")) or "")
            n_genes = str(cm.get("n_genes", cm.get("genes", "")) or "")
            counts_mode = str(cm.get("counts_mode_used", cm.get("counts_mode", "")) or "")
            tblock = cm.get("tissue", {})
            if isinstance(tblock, dict):
                tissue = str(tblock.get("selected_tissue", "") or "")
                tissue_method = str(tblock.get("method", "") or "")

        tsel = read_json(c.cohort_dir / "TISSUE_SELECTION.json") or {}
        if not tissue and isinstance(tsel, dict):
            tissue = str(tsel.get("selected_tissue", "") or "")
            tissue_method = str(tsel.get("method", "") or tissue_method)

        overview_rows.append({
            "Cohort": n,
            "n_samples": n_samples,
            "n_genes": n_genes,
            "counts_mode": counts_mode,
            "tissue": tissue,
            "tissue_method": tissue_method,
            "pathway_table_used": c.pathway_table_label,
        })

        diag_rows.append({
            "Cohort": n,
            "ResolvedDir": str(c.cohort_dir),
            "PathwayTablePath": str(c.pathway_table_path) if c.pathway_table_path else "",
            "TableLabel": c.pathway_table_label,
            "n_rows": int(c.pathway_df.shape[0]) if c.pathway_df is not None else 0,
            "pathway_col": "pathway" if (c.pathway_df is not None and "pathway" in c.pathway_df.columns) else "",
            "fdr_col": c.cols.fdr or "",
            "t_col": c.cols.t or "",
            "p_col": c.cols.p or "",
            "overlap_loaded": "yes" if c.overlap_canon else "no",
            "overlap_path": str(c.overlap_path) if c.overlap_path else "",
        })

    overview_df = pd.DataFrame(overview_rows)
    diag_df = pd.DataFrame(diag_rows)

    # comparison figures
    compare_dir = out_root / "compare"
    spearman_png = compare_dir / "pathway_t_spearman.png"
    chi2_png = compare_dir / "directional_agreement_chi2.png"

    # Default pair = first two cohorts
    a = cohorts[names[0]]
    b = cohorts[names[1]]

    # Rank shared pathways for default pair by activity+evidence score (limit top 50)
    top_shared = rank_shared_pathways(a, b, sig_fdr=sig_fdr, top_n=top_n, limit=top_limit)
    top3 = top_shared[:3]

    # Build HTML interactive case studies
    html_path = report_dir / "ipaa_case_studies.html"
    build_case_studies_html(
        out_html=html_path,
        cohorts=cohorts,
        sig_fdr=sig_fdr,
        top_n=top_n,
        top_limit=top_limit,
        entity_limit=200,
    )

    # Build static top-3 case study blocks for Markdown/PDF
    static_blocks = build_static_case_studies_markdown(a, b, top3, sample_k=12, entity_limit=200)

    # Build report markdown (lives in OUT_ROOT/report/, so images are ../compare/...)
    lines: List[str] = []
    lines.append("# IPAA Disease Comparison Report\n")
    lines.append(f"- Generated from: `{out_root}`\n")
    lines.append(f"- Cohorts (diseases): {', '.join(names)}\n")
    lines.append(f"- Significance: FDR ≤ {sig_fdr} (and top_n cap={top_n})\n")
    lines.append(f"- GSEA permutations: {gsea_perm if gsea_perm is not None else 'unknown'}\n")
    lines.append(f"- Threads per cohort: {threads if threads is not None else 'unknown'}\n")

    lines.append("\n## 1) Cohort overview\n\n")
    lines.append(md_table(overview_df, max_rows=200))

    lines.append("\n## 2) Cross-cohort benchmarking\n\n")
    if spearman_png.exists():
        lines.append("- Figure: `../compare/pathway_t_spearman.png`\n\n")
        lines.append("![](../compare/pathway_t_spearman.png)\n")
    else:
        lines.append("_Missing `compare/pathway_t_spearman.png`._\n")

    if chi2_png.exists():
        lines.append("\n- Figure: `../compare/directional_agreement_chi2.png`\n\n")
        lines.append("![](../compare/directional_agreement_chi2.png)\n")
    else:
        lines.append("\n_Missing `compare/directional_agreement_chi2.png`._\n")

    lines.append("\n## 3) Pathway sets used for case studies\n\n")
    lines.append(f"Default pair: **{a.name} vs {b.name}**\n\n")
    lines.append(f"Top shared pathways (ranked by activity + evidence, capped at {top_limit}):\n\n")
    if top_shared:
        show = pd.DataFrame({"pathway": top_shared})
        lines.append(md_table(show, max_rows=50))
    else:
        lines.append("_(No shared pathways found under current thresholds.)_\n")

    lines.append("\n## 4) Mechanistic case studies (interactive compare)\n\n")
    lines.append(
        "Pick any two diseases and see shared vs unique entities update live for each pathway.\n\n"
        f"- HTML (interactive): `ipaa_case_studies.html`\n\n"
        "PDF shows the default comparison (first two diseases) for the top 3 shared pathways.\n\n"
    )

    if not static_blocks:
        lines.append("_(No case study blocks available; likely no shared pathways under thresholds.)_\n")
    else:
        for blk in static_blocks:
            lines.append(blk.full_md + "\n")

    lines.append("\n## 8) Diagnostics (what the report actually loaded)\n\n")
    lines.append(md_table(diag_df, max_rows=200))

    lines.append("\n## 9) Files produced\n\n")
    lines.append(f"- This report: `report/IPAA_disease_comparison_report.md`\n")
    lines.append(f"- Interactive HTML: `report/ipaa_case_studies.html`\n")
    lines.append("- Comparison folder: `compare/`\n")
    lines.append("- Cohort folders: `<OUT_ROOT>/<CohortName>/`\n")

    report_path = report_dir / "IPAA_disease_comparison_report.md"
    write_text(report_path, "\n".join(lines).strip() + "\n")

    # Root pointer
    pointer = (
        "# IPAA Report\n\n"
        "- Open the report here:\n"
        "  - `report/IPAA_disease_comparison_report.md`\n"
        "- Interactive case studies:\n"
        "  - `report/ipaa_case_studies.html`\n"
    )
    write_text(out_root / "REPORT.md", pointer)

    print(f"[ipaa_report_best] Cohorts: {', '.join(names)}")
    print(f"[ipaa_report_best] Wrote: {report_path}")
    print(f"[ipaa_report_best] Wrote: {html_path}")
    print(f"[ipaa_report_best] Link:  {out_root / 'REPORT.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="IPAA robust report + interactive mechanistic case studies (HTML).")
    ap.add_argument("--outdir", required=True, help="OUT_ROOT directory produced by IPAA pipeline")
    ap.add_argument("--sig-fdr", type=float, default=0.05, help="FDR threshold for significant pathways")
    ap.add_argument("--top-n", type=int, default=200, help="Cap significant set size by abs(t) (0 disables)")
    ap.add_argument("--top-limit", type=int, default=50, help="Max pathways in dropdown (top shared per disease pair)")
    args = ap.parse_args()

    generate_report(
        out_root=Path(args.outdir),
        sig_fdr=float(args.sig_fdr),
        top_n=int(args.top_n),
        top_limit=int(args.top_limit),
    )


if __name__ == "__main__":
    main()
