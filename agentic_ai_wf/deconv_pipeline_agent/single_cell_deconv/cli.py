# src/bisque_deconv/cli.py
from __future__ import annotations

import argparse
import sys

from .config import USER_DEFAULTS
from .utils import abspath_any, timestamped_run_root


def _D(key: str, fallback):
    """pull from USER_DEFAULTS with a safe fallback"""
    return USER_DEFAULTS.get(key, fallback)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "bisque-deconv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Bisque deconvolution S1→S2→S3 with independent IO paths.",
    )

    # ---------- Inputs ----------
    ap.add_argument("--ref_h5ad",      default=_D("ref_h5ad", ""))
    ap.add_argument("--root",          default=_D("root", ""))
    ap.add_argument("--bulk",          default=_D("bulk", ""))
    ap.add_argument("--bulk_gene_col", default=_D("bulk_gene_col", "gene"))
    ap.add_argument("--metadata",      default=_D("metadata", ""))
    ap.add_argument("--species",       default=_D("species", "human"), choices=["human", "mouse"])

    # ---------- Outputs ----------
    ap.add_argument("--outdir",     default=_D("outdir", ""))
    ap.add_argument("--s2_outdir",  default=_D("s2_outdir", ""))
    ap.add_argument("--enh_outdir", default=_D("enh_outdir", ""))

    # ---------- Tissue filter ----------
    ap.add_argument("--tissue_include",    nargs="*", default=_D("tissue_include", []))
    ap.add_argument("--tissue_keys",       nargs="*", default=_D("tissue_keys", ["tissue"]))
    ap.add_argument("--tissue_match_mode", default=_D("tissue_match_mode", "substring"),
                    choices=["exact", "substring"])

    # ---------- S2 tuning (strings on purpose; drivers normalize) ----------
    ap.add_argument("--drop_mito",             default=_D("drop_mito", "true"))
    ap.add_argument("--allow_duplicate_genes", default=_D("allow_duplicate_genes", "true"))
    ap.add_argument("--warn_min_overlap",      default=_D("warn_min_overlap", "200"))
    ap.add_argument("--max_prop_deviation",    default=_D("max_prop_deviation", "0.05"))
    ap.add_argument("--top_markers_per_ct",    default=_D("top_markers_per_ct", "25"))
    ap.add_argument("--top_genes_per_sample",  default=_D("top_genes_per_sample", "40"))
    ap.add_argument("--auto_install",          default=_D("auto_install", "true"))

    # ---------- S3 labels ----------
    ap.add_argument("--sample_col",     default=_D("sample_col", "sample_id"))
    ap.add_argument("--condition_col",  default=_D("condition_col", "condition"))
    ap.add_argument("--control_label",  default=_D("control_label", "Control"))
    ap.add_argument("--patient_label",  default=_D("patient_label", "Disease"))

    # ---------- Orchestration toggles ----------
    ap.add_argument("--skip_s1", action="store_true", help="Skip S1")
    ap.add_argument("--skip_s2", action="store_true", help="Skip S2")
    ap.add_argument("--skip_s3", action="store_true", help="Skip S3")

    return ap.parse_args()


def main() -> None:
    a = _parse_args()

    # --- resolve paths ---
    ref_h5ad_abs = abspath_any(a.ref_h5ad) if a.ref_h5ad else None
    root_abs     = abspath_any(a.root)     if a.root else None
    bulk_abs     = abspath_any(a.bulk)     if a.bulk else None
    meta_abs     = abspath_any(a.metadata) if a.metadata else None

    if not bulk_abs:
        raise SystemExit("Bulk file required (--bulk).")
    if not (ref_h5ad_abs or root_abs):
        raise SystemExit("Provide --ref_h5ad (file) OR --root (folder).")

    out_s1 = abspath_any(a.outdir)     if a.outdir else None
    out_s2 = abspath_any(a.s2_outdir)  if a.s2_outdir else None
    out_s3 = abspath_any(a.enh_outdir) if a.enh_outdir else None

    if not (out_s1 and out_s2 and out_s3):
        rr = timestamped_run_root()
        out_s1 = out_s1 or f"{rr}/s1"
        out_s2 = out_s2 or f"{rr}/s2"
        out_s3 = out_s3 or f"{rr}/s3"

    root_for_s1 = ref_h5ad_abs or root_abs

    # --- lazy imports (clear error if missing) ---
    try:
        from .drivers import run_s1, run_s2
    except Exception as e:
        raise SystemExit(
            "Import error: drivers.py not found or bad. Expected src/bisque_deconv/drivers.py\n" + str(e)
        )

    try:
        from .s3.run import run_s3  # expected at src/bisque_deconv/s3/run.py
    except Exception as e:
        raise SystemExit(
            "Import error: S3 run not found. Expected src/bisque_deconv/s3/run.py with run_s3\n" + str(e)
        )

    # --- S1 ---
    if not a.skip_s1:
        run_s1(
            root_for_s1=root_for_s1,
            out_s1=out_s1,
            bulk_abs=bulk_abs,
            bulk_gene_col=a.bulk_gene_col,
            species=a.species,
            tissue_include=a.tissue_include,
            tissue_keys=a.tissue_keys,
            tissue_match_mode=a.tissue_match_mode,
        )
    else:
        print("[CLI] Skipping S1")

    # --- S2 ---
    if not a.skip_s2:
        run_s2(
            out_s1=out_s1,
            out_s2=out_s2,
            drop_mito=a.drop_mito,
            allow_duplicate_genes=a.allow_duplicate_genes,
            warn_min_overlap=a.warn_min_overlap,
            max_prop_deviation=a.max_prop_deviation,
            top_markers_per_ct=a.top_markers_per_ct,
            top_genes_per_sample=a.top_genes_per_sample,
            auto_install=a.auto_install,
        )
    else:
        print("[CLI] Skipping S2")

    # --- S3 ---
    if not a.skip_s3:
        run_s3(
            out_s2=out_s2,
            metadata_candidate=meta_abs or "",
            sample_col=a.sample_col,
            condition_col=a.condition_col,
            control_label=a.control_label,
            patient_label=a.patient_label,
            out_s3=out_s3,
        )
    else:
        print("[CLI] Skipping S3")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
