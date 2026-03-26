#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


# -----------------------------
# Utilities
# -----------------------------
def _abs_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return str(Path(p).expanduser().resolve())


def _add_flag(argv: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _add_opt(argv: List[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    argv.extend([flag, str(value)])


def _parse_item_str(s: str) -> dict:
    """Parse --item string 'name=X,input=Y,meta=Z,tissue=W' to dict."""
    d: Dict[str, str] = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def _run_pipeline_via_agent(args: argparse.Namespace) -> int:
    """
    Run pipeline via IPAAAgent (no subprocess).
    Production path: direct Python calls.
    """
    from .agent import IPAAAgent
    from .config.models import IPAAConfig

    items_raw = getattr(args, "item", []) or []
    items = [_parse_item_str(x) for x in items_raw] if isinstance(items_raw, list) else []

    config = IPAAConfig(
        outdir=str(args.outdir),
        spec=getattr(args, "spec", None),
        items=items,
        counts_default=bool(getattr(args, "counts_default", False)),
        verbose=bool(getattr(args, "verbose", True)),
        refresh_omnipath=bool(getattr(args, "refresh_omnipath", False)),
        workers=int(getattr(args, "workers", 0)),
        threads_per_cohort=int(getattr(args, "threads_per_cohort", 0)),
        report_top=int(getattr(args, "report_top", 20)),
        threads=int(getattr(args, "threads", 6)),
        gsea_permutations=int(getattr(args, "gsea_permutations", 200)),
        sig_fdr=float(getattr(args, "sig_fdr", 0.05)),
        sig_top_n=int(getattr(args, "sig_top_n", 200)),
        msigdb_dbver=str(getattr(args, "msigdb_dbver", "2024.1.Hs")),
        run_baseline=bool(getattr(args, "run_baseline", True)),
        baseline_dir=str(getattr(args, "baseline_dir", "")),
        auto_select_tissue=bool(getattr(args, "auto_select_tissue", True)),
        tissue_top_k=int(getattr(args, "tissue_top_k", 3)),
        skip_engine1=bool(getattr(args, "skip_engine1", False)),
        engine1_strict=bool(getattr(args, "engine1_strict", False)),
        engine1_license_mode=str(getattr(args, "engine1_license_mode", "academic")),
        engine1_tf_method=str(getattr(args, "engine1_tf_method", "viper")),
        engine1_tmin=int(getattr(args, "engine1_tmin", 5)),
        engine1_no_overwrite=bool(getattr(args, "engine1_no_overwrite", False)),
        engine1_no_regulators_evidence=bool(getattr(args, "engine1_no_regulators_evidence", False)),
        skip_ipaa=bool(getattr(args, "skip_ipaa", False)),
        skip_causality=bool(getattr(args, "skip_causality", False)),
        skip_aggregator=bool(getattr(args, "skip_aggregator", False)),
        skip_html_report=bool(getattr(args, "skip_html_report", False)),
        diseases=getattr(args, "disease", None) or None,
        run_all=bool(getattr(args, "all", True)),
        aggregator_diseases=getattr(args, "aggregator_diseases", None),
        aggregator_out_subdir=getattr(args, "aggregator_out_subdir", None),
        no_engine0=bool(getattr(args, "no_engine0", False)),
        no_engine1=bool(getattr(args, "no_engine1", False)),
        no_engine2=bool(getattr(args, "no_engine2", False)),
        no_engine3=bool(getattr(args, "no_engine3", False)),
        no_omnipath_layer=bool(getattr(args, "no_omnipath_layer", False)),
        strict=bool(getattr(args, "strict", False)),
        log_level=str(getattr(args, "log_level", "INFO")),
        engine23_script=getattr(args, "engine23_script", None),
        no_refresh_omnipath_cache=bool(getattr(args, "no_refresh_omnipath_cache", False)),
        no_build_pkn=bool(getattr(args, "no_build_pkn", False)),
        no_refresh_pkn=bool(getattr(args, "no_refresh_pkn", False)),
        no_force_engine1=bool(getattr(args, "no_force_engine1", False)),
        no_force_engine0=bool(getattr(args, "no_force_engine0", False)),
        signor_edges=getattr(args, "signor_edges", None),
        ptm_min_substrates=int(getattr(args, "ptm_min_substrates", 5)),
        ptm_n_perm=int(getattr(args, "ptm_n_perm", 200)),
        engine1_method=str(getattr(args, "engine1_method", "mean")),
        engine1_min_size=int(getattr(args, "engine1_min_size", 10)),
        engine1_max_pathways=int(getattr(args, "engine1_max_pathways", 1500)),
        engine1_max_tfs=int(getattr(args, "engine1_max_tfs", 300)),
        corr_method=str(getattr(args, "corr_method", "spearman")),
        corr_flag_threshold=float(getattr(args, "corr_flag_threshold", 0.40)),
        min_markers=int(getattr(args, "min_markers", 5)),
        pkn_edges=getattr(args, "pkn_edges", None),
        max_steps=int(getattr(args, "max_steps", 3)),
        top_tfs=int(getattr(args, "top_tfs", 30)),
        confound_penalty_threshold=float(getattr(args, "confound_penalty_threshold", 0.40)),
        report_topn=int(getattr(args, "html_report_topn", 10)),
        report_fdr_cutoff=float(getattr(args, "html_report_fdr_cutoff", 0.10)),
        report_llm_selector=str(getattr(args, "html_report_llm_selector", "auto")),
        report_llm_narrative=str(getattr(args, "html_report_llm_narrative", "auto")),
        report_llm_model=str(getattr(args, "html_report_llm_model", "gpt-4.1")),
        report_api_key_env=str(getattr(args, "html_report_api_key_env", "OPENAI_API_KEY")),
        generate_pdf_report=bool(getattr(args, "generate_pdf_report", False)),
    )

    agent = IPAAAgent(config)
    result = agent.run_full()

    if result.status == "ok":
        print("\n[DONE] IPAA + Causality + Aggregator + HTML Report completed successfully.\n", flush=True)
        return 0
    if result.status == "partial":
        print("\n[DONE] Pipeline completed with partial success.\n", flush=True)
        return 0
    return 1


def _run_cmd(argv: List[str], *, stage_name: str, cwd: Optional[str] = None) -> int:
    """DEPRECATED: Use _run_pipeline_via_agent. Kept for backward compatibility."""
    try:
        print(f"\n[RUN] {stage_name}: {' '.join(argv)}\n", flush=True)
        r = subprocess.run(argv, cwd=cwd, check=False)
        return int(r.returncode)
    except FileNotFoundError as e:
        print(f"[ERROR] {stage_name} failed: executable/script not found: {e}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        print(f"[ERROR] {stage_name} interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[ERROR] {stage_name} failed with unexpected exception: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


@dataclass(frozen=True)
class ItemSpec:
    name: str
    input: str
    meta: Optional[str] = None
    tissue: Optional[str] = None


def _item_to_string(item: Union[str, ItemSpec, Dict[str, str]]) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, ItemSpec):
        parts = [f"name={item.name}", f"input={item.input}"]
        if item.meta:
            parts.append(f"meta={item.meta}")
        if item.tissue:
            parts.append(f"tissue={item.tissue}")
        return ",".join(parts)
    if isinstance(item, dict):
        name = (item.get("name") or "").strip()
        inp = (item.get("input") or item.get("path") or "").strip()
        if not name or not inp:
            raise ValueError("Item dict must include 'name' and 'input'.")
        parts = [f"name={name}", f"input={inp}"]
        meta = (item.get("meta") or "").strip()
        tissue = (item.get("tissue") or "").strip()
        if meta:
            parts.append(f"meta={meta}")
        if tissue:
            parts.append(f"tissue={tissue}")
        return ",".join(parts)
    raise TypeError("item must be str, ItemSpec, or dict.")


def _normalize_items(items: Optional[Union[str, ItemSpec, Dict[str, str], Sequence[Union[str, ItemSpec, Dict[str, str]]]]]) -> List[str]:
    if items is None:
        return []
    if isinstance(items, (str, ItemSpec, dict)):
        return [_item_to_string(items)]
    return [_item_to_string(it) for it in items]


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Master runner: run IPAA (main_ipaa.py) to generate OUT_ROOT, then run causality (run_causality_all.py), "
            "then run pathway summary aggregator (pathway_summary_aggregator.py).\n\n"
            "Design goals:\n"
            "  - No lost functionality: exposes all IPAA + causality args.\n"
            "  - No surprise changes: we only forward what you explicitly specify.\n"
            "  - Robust: clear errors, stage exit codes, optional continue-on-failure.\n"
            "  - One-sample cohorts: we do NOT force strict mode; stages can skip gracefully unless you set strict flags."
        ),
    )

    # --- Script locations (robust + explicit) ---
    ap.add_argument(
        "--ipaa-script",
        default=None,
        help="Path to IPAA main runner script (main_ipaa.py). If omitted, uses ./main_ipaa.py relative to this file.",
    )
    ap.add_argument(
        "--causality-script",
        default=None,
        help="Path to causality runner script (run_causality_all.py). If omitted, uses ./run_causality_all.py relative to this file.",
    )
    ap.add_argument(
        "--aggregator-script",
        default=None,
        help=(
            "Path to pathway summary aggregator (pathway_summary_aggregator.py). "
            "If omitted, uses ./pathway_summary_aggregator.py relative to this file (same directory)."
        ),
    )

    # --- Stage toggles ---
    ap.add_argument("--skip-ipaa", action="store_true", help="Skip IPAA stage (assumes outdir already exists).")
    ap.add_argument("--skip-causality", action="store_true", help="Skip causality stage.")
    ap.add_argument("--skip-aggregator", action="store_true", help="Skip aggregator stage.")
    ap.add_argument(
        "--continue-after-ipaa-failure",
        action="store_true",
        help="If IPAA exits non-zero, continue to causality anyway (default: stop).",
    )
    ap.add_argument(
        "--continue-after-causality-failure",
        action="store_true",
        help="If causality exits non-zero, continue to aggregator and exit 0 at end (default: stop on causality failure).",
    )
    ap.add_argument(
        "--continue-after-aggregator-failure",
        action="store_true",
        help="If aggregator exits non-zero, still exit 0 at the end (default: exit non-zero).",
    )

    # ==========================================================
    # IPAA ARGS (copied 1:1 from your main_ipaa.py interface)
    # ==========================================================
    ipaa = ap.add_argument_group("IPAA arguments (forwarded to main_ipaa.py)")

    ipaa.add_argument("--outdir", required=True, help="Output directory root for IPAA (also used as causality --out-root).")
    ipaa.add_argument("--spec", help="JSON spec describing cohorts and global settings (optional).")
    ipaa.add_argument(
        "--item",
        action="append",
        default=[],
        help=(
            "Cohort item (repeatable): "
            "\"name=<Disease>, input=<counts_csv_or_degs>, tissue=<optional>, meta=<optional_metadata>\"."
        ),
    )

    ipaa.add_argument("--counts-default", action="store_true", help="Default: treat cohorts as counts unless cohort overrides.")
    ipaa.add_argument("--verbose", action="store_true")

    ipaa.add_argument("--refresh-omnipath", action="store_true", help="Force refresh OmniPath/decoupler.op caches.")
    ipaa.add_argument("--workers", type=int, default=0, help="Override cohort process workers.")
    ipaa.add_argument("--threads-per-cohort", type=int, default=0, help="Override threads-per-cohort budget used for worker auto-sizing.")

    ipaa.add_argument("--report-top", type=int, default=20, help="Top pathways to show per cohort in the report.")

    ipaa.add_argument("--threads", type=int, default=6, help="Threads per cohort (GSEApy).")
    ipaa.add_argument("--gsea-permutations", type=int, default=200, help="GSEA permutations.")
    ipaa.add_argument("--sig-fdr", type=float, default=0.05, help="Significance FDR threshold.")
    ipaa.add_argument("--sig-top-n", type=int, default=200, help="Top-N pathways cap for comparisons.")
    ipaa.add_argument("--msigdb-dbver", type=str, default="2024.1.Hs", help="MSigDB version for c2.cp.")

    ipaa.add_argument("--run-baseline", action="store_true", help="Enable baseline consensus build.")
    ipaa.add_argument("--baseline-dir", type=str, default="", help="Folder containing baseline sources if --run-baseline is enabled.")
    ipaa.add_argument("--auto-select-tissue", action="store_true", help="Let IPAA auto-select tissue when cohort tissue is missing/invalid.")
    ipaa.add_argument("--tissue-top-k", type=int, default=3, help="Top-K candidate tissues considered by IPAA auto-selection.")

    ipaa.add_argument("--skip-engine1", action="store_true", help="Skip IPAA's Engine 1 stage (causal_engine_features.py).")
    ipaa.add_argument("--engine1-strict", action="store_true", help="If IPAA Engine1 fails, exit non-zero.")
    ipaa.add_argument("--engine1-license-mode", type=str, default="academic", help="Engine1 decoupler license mode.")
    ipaa.add_argument("--engine1-tf-method", type=str, default="viper", choices=["viper", "ulm"], help="Engine1 TF activity method.")
    ipaa.add_argument("--engine1-tmin", type=int, default=5, help="Engine1 VIPER min targets threshold (tmin).")
    ipaa.add_argument("--engine1-no-overwrite", action="store_true", help="Engine1: do not overwrite existing outputs.")
    ipaa.add_argument("--engine1-no-regulators-evidence", action="store_true", help="Engine1: do not write regulators_evidence.tsv.")

    # ==========================================================
    # CAUSALITY ARGS (forwarded to run_causality_all.py)
    # ==========================================================
    caus = ap.add_argument_group("Causality arguments (forwarded to run_causality_all.py)")

    caus.add_argument("--disease", action="append", default=None, help="Disease name (repeatable). If omitted, use --all.")
    caus.add_argument("--all", action="store_true", help="Run on all discovered diseases under OUT_ROOT.")
    caus.add_argument("--strict", action="store_true", help="If set, missing inputs raise errors instead of SKIPPED.")
    caus.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    caus.add_argument("--engine23-script", default=None, help="Path to run_engine2_3_mechanistic.py (used by run_causality_all.py).")

    caus.add_argument("--no-engine0", action="store_true")
    caus.add_argument("--no-engine1", action="store_true")
    caus.add_argument("--no-engine2", action="store_true")
    caus.add_argument("--no-engine3", action="store_true")
    caus.add_argument("--no-omnipath-layer", action="store_true")

    # default-on flags in causality runner; allow turning off explicitly
    caus.add_argument("--no-refresh-omnipath-cache", action="store_true")
    caus.add_argument("--no-build-pkn", action="store_true")
    caus.add_argument("--no-refresh-pkn", action="store_true")
    caus.add_argument("--no-force-engine1", action="store_true")
    caus.add_argument("--no-force-engine0", action="store_true")

    caus.add_argument("--signor-edges", default=None)
    caus.add_argument("--ptm-min-substrates", type=int, default=5)
    caus.add_argument("--ptm-n-perm", type=int, default=200)

    caus.add_argument("--engine1-method", default="mean", choices=["mean", "zscore", "pca1", "ssgsea"])
    caus.add_argument("--engine1-min-size", type=int, default=10)
    caus.add_argument("--engine1-max-pathways", type=int, default=1500)
    caus.add_argument("--engine1-max-tfs", type=int, default=300)

    caus.add_argument("--corr-method", default="spearman", choices=["spearman", "pearson"])
    caus.add_argument("--corr-flag-threshold", type=float, default=0.40)
    caus.add_argument("--min-markers", type=int, default=5)

    caus.add_argument("--pkn-edges", default=None)
    caus.add_argument("--max-steps", type=int, default=3)
    caus.add_argument("--top-tfs", type=int, default=30)
    caus.add_argument("--confound-penalty-threshold", type=float, default=0.40)

    # ==========================================================
    # AGGREGATOR ARGS (forwarded to pathway_summary_aggregator.py)
    # ==========================================================
    agg = ap.add_argument_group("Pathway summary aggregator arguments (forwarded to pathway_summary_aggregator.py)")
    # NOTE: We do NOT change your working command; aggregator runs with sensible defaults.
    agg.add_argument(
        "--aggregator-diseases",
        default=None,
        help="Comma-separated diseases for aggregator --diseases. If omitted, aggregator auto-discovers under cohorts/.",
    )
    agg.add_argument(
        "--aggregator-out-subdir",
        default=None,
        help="Aggregator --out-subdir override (e.g. engines/pathway_summary). If omitted, aggregator uses its default.",
    )

    # ==========================================================
    # HTML REPORT ARGS (generate_ipaa_html_reports.py)
    # ==========================================================
    rep = ap.add_argument_group("HTML report generation (generate_ipaa_html_reports.py)")
    rep.add_argument("--skip-html-report", action="store_true", help="Skip HTML report generation.")
    rep.add_argument("--html-report-topn", type=int, default=10, help="Top N pathways to include in HTML report.")
    rep.add_argument("--html-report-fdr-cutoff", type=float, default=0.10, help="FDR cutoff for HTML report pathway selection.")
    rep.add_argument("--html-report-llm-selector", choices=["off", "on", "auto"], default="auto", help="LLM top-pathway selection for HTML report.")
    rep.add_argument("--html-report-llm-narrative", choices=["off", "on", "auto"], default="auto", help="LLM narrative for HTML report.")
    rep.add_argument("--html-report-llm-model", default="gpt-4.1", help="LLM model for HTML report generation.")
    rep.add_argument("--html-report-api-key-env", default="OPENAI_API_KEY", help="Env var for OpenAI API key for HTML report.")
    rep.add_argument("--generate-pdf-report", action="store_true", help="Generate PDF reports alongside HTML (requires weasyprint).")

    return ap.parse_args(args=argv)


# -----------------------------
# Arg forwarding builders
# -----------------------------
def _build_ipaa_argv(args: argparse.Namespace, ipaa_script: str) -> List[str]:
    argv: List[str] = [sys.executable, ipaa_script]

    _add_opt(argv, "--outdir", args.outdir)
    _add_opt(argv, "--spec", args.spec)
    if args.item:
        for it in args.item:
            argv.extend(["--item", it])

    _add_flag(argv, "--counts-default", bool(args.counts_default))
    _add_flag(argv, "--verbose", bool(args.verbose))

    _add_flag(argv, "--refresh-omnipath", bool(args.refresh_omnipath))
    _add_opt(argv, "--workers", args.workers)
    _add_opt(argv, "--threads-per-cohort", args.threads_per_cohort)

    _add_opt(argv, "--report-top", args.report_top)

    _add_opt(argv, "--threads", args.threads)
    _add_opt(argv, "--gsea-permutations", args.gsea_permutations)
    _add_opt(argv, "--sig-fdr", args.sig_fdr)
    _add_opt(argv, "--sig-top-n", args.sig_top_n)
    _add_opt(argv, "--msigdb-dbver", args.msigdb_dbver)

    _add_flag(argv, "--run-baseline", bool(args.run_baseline))
    if isinstance(args.baseline_dir, str) and args.baseline_dir.strip():
        _add_opt(argv, "--baseline-dir", args.baseline_dir)

    _add_flag(argv, "--auto-select-tissue", bool(args.auto_select_tissue))
    _add_opt(argv, "--tissue-top-k", args.tissue_top_k)

    _add_flag(argv, "--skip-engine1", bool(args.skip_engine1))
    _add_flag(argv, "--engine1-strict", bool(args.engine1_strict))
    _add_opt(argv, "--engine1-license-mode", args.engine1_license_mode)
    _add_opt(argv, "--engine1-tf-method", args.engine1_tf_method)
    _add_opt(argv, "--engine1-tmin", args.engine1_tmin)
    _add_flag(argv, "--engine1-no-overwrite", bool(args.engine1_no_overwrite))
    _add_flag(argv, "--engine1-no-regulators-evidence", bool(args.engine1_no_regulators_evidence))

    return argv


def _build_causality_argv(args: argparse.Namespace, causality_script: str) -> List[str]:
    argv: List[str] = [sys.executable, causality_script]

    _add_opt(argv, "--out-root", args.outdir)

    if bool(args.all):
        argv.append("--all")
    else:
        if args.disease:
            for d in args.disease:
                argv.extend(["--disease", d])
        else:
            argv.append("--all")

    _add_flag(argv, "--strict", bool(args.strict))
    _add_opt(argv, "--log-level", args.log_level)
    _add_opt(argv, "--engine23-script", args.engine23_script)

    _add_flag(argv, "--no-engine0", bool(args.no_engine0))
    _add_flag(argv, "--no-engine1", bool(args.no_engine1))
    _add_flag(argv, "--no-engine2", bool(args.no_engine2))
    _add_flag(argv, "--no-engine3", bool(args.no_engine3))
    _add_flag(argv, "--no-omnipath-layer", bool(args.no_omnipath_layer))

    _add_flag(argv, "--no-refresh-omnipath-cache", bool(args.no_refresh_omnipath_cache))
    _add_flag(argv, "--no-build-pkn", bool(args.no_build_pkn))
    _add_flag(argv, "--no-refresh-pkn", bool(args.no_refresh_pkn))
    _add_flag(argv, "--no-force-engine1", bool(args.no_force_engine1))
    _add_flag(argv, "--no-force-engine0", bool(args.no_force_engine0))

    _add_opt(argv, "--signor-edges", args.signor_edges)
    _add_opt(argv, "--ptm-min-substrates", args.ptm_min_substrates)
    _add_opt(argv, "--ptm-n-perm", args.ptm_n_perm)

    _add_opt(argv, "--engine1-method", args.engine1_method)
    _add_opt(argv, "--engine1-min-size", args.engine1_min_size)
    _add_opt(argv, "--engine1-max-pathways", args.engine1_max_pathways)
    _add_opt(argv, "--engine1-max-tfs", args.engine1_max_tfs)

    _add_opt(argv, "--corr-method", args.corr_method)
    _add_opt(argv, "--corr-flag-threshold", args.corr_flag_threshold)
    _add_opt(argv, "--min-markers", args.min_markers)

    _add_opt(argv, "--pkn-edges", args.pkn_edges)
    _add_opt(argv, "--max-steps", args.max_steps)
    _add_opt(argv, "--top-tfs", args.top_tfs)
    _add_opt(argv, "--confound-penalty-threshold", args.confound_penalty_threshold)

    return argv


def _build_aggregator_argv(args: argparse.Namespace, aggregator_script: str) -> List[str]:
    """
    Forward aggregator args WITHOUT changing existing behavior:
      - always pass --out-root
      - pass --diseases only if user explicitly provided --aggregator-diseases
      - pass --out-subdir only if user explicitly provided --aggregator-out-subdir
    """
    argv: List[str] = [sys.executable, aggregator_script]
    _add_opt(argv, "--out-root", args.outdir)
    _add_opt(argv, "--diseases", args.aggregator_diseases)
    _add_opt(argv, "--out-subdir", args.aggregator_out_subdir)
    return argv


def _build_html_report_argv(args: argparse.Namespace, html_script: str) -> List[str]:
    """
    Build argv for generate_ipaa_html_reports.py.
    """
    argv: List[str] = [sys.executable, html_script]
    _add_opt(argv, "--outdir", args.outdir)
    _add_opt(argv, "--topn", args.html_report_topn)
    _add_opt(argv, "--fdr-cutoff", args.html_report_fdr_cutoff)
    _add_opt(argv, "--llm-selector", args.html_report_llm_selector)
    _add_opt(argv, "--llm-narrative", args.html_report_llm_narrative)
    _add_opt(argv, "--llm-model", args.html_report_llm_model)
    _add_opt(argv, "--api-key-env", args.html_report_api_key_env)
    _add_flag(argv, "--generate-pdf", bool(args.generate_pdf_report))
    return argv


def _run_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline via IPAAAgent (no subprocess)."""
    return _run_pipeline_via_agent(args)


def _run_pipeline_subprocess(args: argparse.Namespace) -> int:
    """DEPRECATED: Legacy subprocess-based pipeline. Use _run_pipeline_via_agent."""
    here = Path(__file__).resolve().parent

    ipaa_script = _abs_path(args.ipaa_script) or str((here / "main_ipaa.py").resolve())
    causality_script = _abs_path(args.causality_script) or str((here / "run_causality_all.py").resolve())
    aggregator_script = _abs_path(args.aggregator_script) or str((here / "pathway_summary_aggregator.py").resolve())

    print(f"[INFO] Using IPAA script:       {ipaa_script}", flush=True)
    print(f"[INFO] Using causality script:  {causality_script}", flush=True)
    print(f"[INFO] Using aggregator script: {aggregator_script}", flush=True)

    if not Path(ipaa_script).exists() and not bool(args.skip_ipaa):
        print(f"[ERROR] IPAA script not found: {ipaa_script}", file=sys.stderr)
        print("        Use --ipaa-script to point to your main_ipaa.py.", file=sys.stderr)
        return 2

    if not Path(causality_script).exists() and not bool(args.skip_causality):
        print(f"[ERROR] Causality script not found: {causality_script}", file=sys.stderr)
        print("        Use --causality-script to point to your run_causality_all.py.", file=sys.stderr)
        return 2

    if not Path(aggregator_script).exists() and not bool(args.skip_aggregator):
        print(f"[ERROR] Aggregator script not found: {aggregator_script}", file=sys.stderr)
        print("        Use --aggregator-script to point to your pathway_summary_aggregator.py.", file=sys.stderr)
        return 2

    outdir = Path(str(args.outdir)).expanduser()

    if bool(args.skip_ipaa) and not outdir.exists():
        print(f"[ERROR] --skip-ipaa set, but outdir does not exist: {outdir}", file=sys.stderr)
        return 2

    # -------------------------
    # Stage 1: IPAA
    # -------------------------
    ipaa_rc = 0
    if not bool(args.skip_ipaa):
        ipaa_argv = _build_ipaa_argv(args, ipaa_script)
        ipaa_rc = _run_cmd(ipaa_argv, stage_name="IPAA")
        if ipaa_rc != 0:
            msg = f"[ERROR] IPAA exited non-zero (rc={ipaa_rc})."
            if not bool(args.continue_after_ipaa_failure):
                print(msg + " Stopping.", file=sys.stderr)
                return ipaa_rc
            print(msg + " Continuing to causality due to --continue-after-ipaa-failure.", file=sys.stderr)
    else:
        print("[SKIP] IPAA stage skipped.", flush=True)

    # Ensure OUT_ROOT exists before downstream stages if IPAA ran (best-effort; no forced create)
    if not outdir.exists() and not bool(args.skip_ipaa):
        print(f"[ERROR] OUT_ROOT not found after IPAA: {outdir}", file=sys.stderr)
        return 2

    # -------------------------
    # Stage 2: Causality
    # -------------------------
    caus_rc = 0
    if not bool(args.skip_causality):
        caus_argv = _build_causality_argv(args, causality_script)
        caus_rc = _run_cmd(caus_argv, stage_name="CAUSALITY_ALL")
        if caus_rc != 0:
            if bool(args.continue_after_causality_failure):
                print(
                    f"[WARN] Causality exited non-zero (rc={caus_rc}) but continuing due to --continue-after-causality-failure.",
                    file=sys.stderr,
                )
            else:
                print(f"[ERROR] Causality exited non-zero (rc={caus_rc}).", file=sys.stderr)
                return caus_rc
    else:
        print("[SKIP] Causality stage skipped.", flush=True)

    # -------------------------
    # Stage 3: Aggregator
    # -------------------------
    agg_rc = 0
    if not bool(args.skip_aggregator):
        if not outdir.exists():
            print(f"[ERROR] Aggregator cannot run because outdir does not exist: {outdir}", file=sys.stderr)
            agg_rc = 2
        else:
            agg_argv = _build_aggregator_argv(args, aggregator_script)
            agg_rc = _run_cmd(agg_argv, stage_name="PATHWAY_SUMMARY_AGGREGATOR")

        if agg_rc != 0:
            if bool(args.continue_after_aggregator_failure):
                print(
                    f"[WARN] Aggregator exited non-zero (rc={agg_rc}) but continuing due to --continue-after-aggregator-failure.",
                    file=sys.stderr,
                )
            else:
                print(f"[ERROR] Aggregator exited non-zero (rc={agg_rc}).", file=sys.stderr)
                return agg_rc
    else:
        print("[SKIP] Aggregator stage skipped.", flush=True)

    # -------------------------
    # Stage 4: HTML Report Generation
    # -------------------------
    html_rc = 0
    if not bool(args.skip_html_report):
        html_script = str((here / "generate_ipaa_html_reports.py").resolve())
        if not Path(html_script).exists():
            print(f"[WARN] HTML report script not found: {html_script}. Skipping HTML report generation.", file=sys.stderr)
        else:
            html_argv = _build_html_report_argv(args, html_script)
            html_rc = _run_cmd(html_argv, stage_name="HTML_REPORT_GENERATION")
            if html_rc != 0:
                print(f"[WARN] HTML report generation exited non-zero (rc={html_rc}). Continuing.", file=sys.stderr)
    else:
        print("[SKIP] HTML report generation skipped.", flush=True)

    # -------------------------
    # Final exit logic (legacy subprocess path)
    # -------------------------
    if ipaa_rc == 0 and caus_rc == 0 and agg_rc == 0 and html_rc == 0:
        print("\n[DONE] IPAA + Causality + Aggregator + HTML Report completed successfully.\n", flush=True)
        return 0
    if caus_rc != 0 and bool(args.continue_after_causality_failure):
        print("\n[DONE] Pipeline completed with non-fatal causality failure (exiting 0 due to --continue-after-causality-failure).\n", flush=True)
        return 0
    if agg_rc != 0 and bool(args.continue_after_aggregator_failure):
        print("\n[DONE] Pipeline completed with non-fatal aggregator failure (exiting 0 due to --continue-after-aggregator-failure).\n", flush=True)
        return 0
    if caus_rc != 0:
        return caus_rc
    if agg_rc != 0:
        return agg_rc
    return ipaa_rc if ipaa_rc != 0 else 1


def run_ipaa_pipeline(
    *,
    outdir: str,
    spec: Optional[str] = None,
    item: Optional[Union[str, ItemSpec, Dict[str, str], Sequence[Union[str, ItemSpec, Dict[str, str]]]]] = None,
    counts_default: bool = False,
    verbose: bool = True,
    refresh_omnipath: bool = False,
    workers: int = 0,
    threads_per_cohort: int = 0,
    report_top: int = 20,
    threads: int = 6,
    gsea_permutations: int = 200,
    sig_fdr: float = 0.05,
    sig_top_n: int = 200,
    msigdb_dbver: str = "2024.1.Hs",
    run_baseline: bool = True,
    baseline_dir: str = "",
    auto_select_tissue: bool = True,
    tissue_top_k: int = 3,
    skip_engine1: bool = False,
    engine1_strict: bool = False,
    engine1_license_mode: str = "academic",
    engine1_tf_method: str = "viper",
    engine1_tmin: int = 5,
    engine1_no_overwrite: bool = False,
    engine1_no_regulators_evidence: bool = False,
    disease: Optional[Union[str, List[str]]] = None,
    run_all: bool = True,
    strict: bool = False,
    log_level: str = "INFO",
    engine23_script: Optional[str] = None,
    no_engine0: bool = False,
    no_engine1: bool = False,
    no_engine2: bool = False,
    no_engine3: bool = False,
    no_omnipath_layer: bool = False,
    no_refresh_omnipath_cache: bool = False,
    no_build_pkn: bool = False,
    no_refresh_pkn: bool = False,
    no_force_engine1: bool = False,
    no_force_engine0: bool = False,
    signor_edges: Optional[str] = None,
    ptm_min_substrates: int = 5,
    ptm_n_perm: int = 200,
    engine1_method: str = "mean",
    engine1_min_size: int = 10,
    engine1_max_pathways: int = 1500,
    engine1_max_tfs: int = 300,
    corr_method: str = "spearman",
    corr_flag_threshold: float = 0.40,
    min_markers: int = 5,
    pkn_edges: Optional[str] = None,
    max_steps: int = 3,
    top_tfs: int = 30,
    confound_penalty_threshold: float = 0.40,
    aggregator_diseases: Optional[str] = None,
    aggregator_out_subdir: Optional[str] = None,
    ipaa_script: Optional[str] = None,
    causality_script: Optional[str] = None,
    aggregator_script: Optional[str] = None,
    skip_ipaa: bool = False,
    skip_causality: bool = False,
    skip_aggregator: bool = False,
    continue_after_ipaa_failure: bool = False,
    continue_after_causality_failure: bool = False,
    continue_after_aggregator_failure: bool = False,
) -> int:
    """
    Run the full pipeline programmatically with explicit parameters.
    """
    if not outdir:
        raise ValueError("outdir is required.")

    argv: List[str] = []

    _add_opt(argv, "--ipaa-script", ipaa_script)
    _add_opt(argv, "--causality-script", causality_script)
    _add_opt(argv, "--aggregator-script", aggregator_script)

    _add_flag(argv, "--skip-ipaa", bool(skip_ipaa))
    _add_flag(argv, "--skip-causality", bool(skip_causality))
    _add_flag(argv, "--skip-aggregator", bool(skip_aggregator))
    _add_flag(argv, "--continue-after-ipaa-failure", bool(continue_after_ipaa_failure))
    _add_flag(argv, "--continue-after-causality-failure", bool(continue_after_causality_failure))
    _add_flag(argv, "--continue-after-aggregator-failure", bool(continue_after_aggregator_failure))

    _add_opt(argv, "--outdir", outdir)
    _add_opt(argv, "--spec", spec)

    for it in _normalize_items(item):
        argv.extend(["--item", it])

    _add_flag(argv, "--counts-default", bool(counts_default))
    _add_flag(argv, "--verbose", bool(verbose))

    _add_flag(argv, "--refresh-omnipath", bool(refresh_omnipath))
    _add_opt(argv, "--workers", workers)
    _add_opt(argv, "--threads-per-cohort", threads_per_cohort)

    _add_opt(argv, "--report-top", report_top)
    _add_opt(argv, "--threads", threads)
    _add_opt(argv, "--gsea-permutations", gsea_permutations)
    _add_opt(argv, "--sig-fdr", sig_fdr)
    _add_opt(argv, "--sig-top-n", sig_top_n)
    _add_opt(argv, "--msigdb-dbver", msigdb_dbver)

    _add_flag(argv, "--run-baseline", bool(run_baseline))
    _add_opt(argv, "--baseline-dir", baseline_dir)
    _add_flag(argv, "--auto-select-tissue", bool(auto_select_tissue))
    _add_opt(argv, "--tissue-top-k", tissue_top_k)

    _add_flag(argv, "--skip-engine1", bool(skip_engine1))
    _add_flag(argv, "--engine1-strict", bool(engine1_strict))
    _add_opt(argv, "--engine1-license-mode", engine1_license_mode)
    _add_opt(argv, "--engine1-tf-method", engine1_tf_method)
    _add_opt(argv, "--engine1-tmin", engine1_tmin)
    _add_flag(argv, "--engine1-no-overwrite", bool(engine1_no_overwrite))
    _add_flag(argv, "--engine1-no-regulators-evidence", bool(engine1_no_regulators_evidence))

    if bool(run_all):
        argv.append("--all")
    else:
        if isinstance(disease, str):
            argv.extend(["--disease", disease])
        elif disease:
            for d in disease:
                argv.extend(["--disease", d])

    _add_flag(argv, "--strict", bool(strict))
    _add_opt(argv, "--log-level", log_level)
    _add_opt(argv, "--engine23-script", engine23_script)

    _add_flag(argv, "--no-engine0", bool(no_engine0))
    _add_flag(argv, "--no-engine1", bool(no_engine1))
    _add_flag(argv, "--no-engine2", bool(no_engine2))
    _add_flag(argv, "--no-engine3", bool(no_engine3))
    _add_flag(argv, "--no-omnipath-layer", bool(no_omnipath_layer))

    _add_flag(argv, "--no-refresh-omnipath-cache", bool(no_refresh_omnipath_cache))
    _add_flag(argv, "--no-build-pkn", bool(no_build_pkn))
    _add_flag(argv, "--no-refresh-pkn", bool(no_refresh_pkn))
    _add_flag(argv, "--no-force-engine1", bool(no_force_engine1))
    _add_flag(argv, "--no-force-engine0", bool(no_force_engine0))

    _add_opt(argv, "--signor-edges", signor_edges)
    _add_opt(argv, "--ptm-min-substrates", ptm_min_substrates)
    _add_opt(argv, "--ptm-n-perm", ptm_n_perm)

    _add_opt(argv, "--engine1-method", engine1_method)
    _add_opt(argv, "--engine1-min-size", engine1_min_size)
    _add_opt(argv, "--engine1-max-pathways", engine1_max_pathways)
    _add_opt(argv, "--engine1-max-tfs", engine1_max_tfs)

    _add_opt(argv, "--corr-method", corr_method)
    _add_opt(argv, "--corr-flag-threshold", corr_flag_threshold)
    _add_opt(argv, "--min-markers", min_markers)

    _add_opt(argv, "--pkn-edges", pkn_edges)
    _add_opt(argv, "--max-steps", max_steps)
    _add_opt(argv, "--top-tfs", top_tfs)
    _add_opt(argv, "--confound-penalty-threshold", confound_penalty_threshold)

    _add_opt(argv, "--aggregator-diseases", aggregator_diseases)
    _add_opt(argv, "--aggregator-out-subdir", aggregator_out_subdir)

    args = parse_args(argv)
    return _run_pipeline(args)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    args = parse_args()
    return _run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
