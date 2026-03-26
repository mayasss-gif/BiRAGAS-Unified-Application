#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""IPAA/main_ipaa_best.py

Main runner for IPAA M6 (multi-cohort IPAA) using m6_processing.py.

Adds metadata support (optional):
- In --item mode you can pass: meta=<metadata_file>
- Metadata must have columns: sample_id, condition
- If metadata is provided, we will:
  1) Pass meta into CohortSpec if m6_processing supports it (backward compatible)
  2) ALWAYS write/overwrite labels_used.tsv from metadata after cohort run,
     so downstream Engine1 regulators evidence uses the correct grouping.

Fallback:
- If metadata is not given, existing inference logic remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import logging as _logging
import openai

from . import m6_processing as m6

LOG = logging.getLogger("IPAA_MAIN")


# =============================================================================
# Optional helper import (new)
# =============================================================================
try:
    from .ipaa_metadata_labels import align_labels_to_samples, pick_two_groups, write_labels_used  # type: ignore
except Exception:
    align_labels_to_samples = None  # type: ignore
    pick_two_groups = None  # type: ignore
    write_labels_used = None  # type: ignore


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run IPAA M6 on multiple cohorts, with optional post-stages and report.",
    )

    # --- Inputs ---
    ap.add_argument("--outdir", required=True, help="Output directory root.")
    ap.add_argument("--spec", help="JSON spec describing cohorts and global settings (optional).")
    ap.add_argument(
        "--item",
        action="append",
        default=[],
        help=(
            "Cohort item (repeatable): "
            "\"name=<Disease>, input=<counts_csv_or_degs>, tissue=<optional>, meta=<optional_metadata>\". "
            "Metadata file must have columns: sample_id, condition. "
            "Example: --item \"name=SLE, input=/mnt/d/temp/SLE.csv, meta=/mnt/d/temp/SLE_meta.tsv, tissue=spleen\""
        ),
    )

    # --- Behavior ---
    ap.add_argument(
        "--counts-default",
        action="store_true",
        help="Default: treat cohorts as counts unless cohort overrides (spec mode).",
    )
    ap.add_argument("--verbose", action="store_true")

    # parallel + caching controls
    ap.add_argument(
        "--refresh-omnipath",
        action="store_true",
        help="Force refresh OmniPath/decoupler.op caches (otherwise load from disk).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override cohort process workers. Default=auto (based on CPU and threads-per-cohort).",
    )
    ap.add_argument(
        "--threads-per-cohort",
        type=int,
        default=0,
        help="Override threads-per-cohort budget used for worker auto-sizing. If 0, uses spec.threads.",
    )

    # report controls
    ap.add_argument(
        "--report-top",
        type=int,
        default=20,
        help="Top pathways to show per cohort (up and down) in the report.",
    )

    # --- Optional spec overrides (useful in --item mode) ---
    ap.add_argument("--threads", type=int, default=6, help="Threads per cohort (GSEApy).")
    ap.add_argument("--gsea-permutations", type=int, default=200, help="GSEA permutations.")
    ap.add_argument("--sig-fdr", type=float, default=0.05, help="Significance FDR threshold.")
    ap.add_argument("--sig-top-n", type=int, default=200, help="Top-N pathways cap for comparisons.")
    ap.add_argument("--msigdb-dbver", type=str, default="2024.1.Hs", help="MSigDB version for c2.cp.")
    ap.add_argument(
        "--run-baseline",
        action="store_true",
        help="Enable baseline consensus build (required for consensus tissue list & expectations).",
    )
    ap.add_argument(
        "--baseline-dir",
        type=str,
        default="",
        help="Folder containing baseline sources (HPA/GTEx/FANTOM) if --run-baseline is enabled.",
    )
    ap.add_argument(
        "--auto-select-tissue",
        action="store_true",
        help="Let IPAA auto-select tissue when cohort tissue is missing/invalid and expectations exist.",
    )
    ap.add_argument(
        "--tissue-top-k",
        type=int,
        default=3,
        help="Top-K candidate tissues considered by IPAA auto-selection (spec field).",
    )

    # --- Engine 1 (causal features) integration ---
    ap.add_argument(
        "--skip-engine1",
        action="store_true",
        help="Skip Engine 1 (causal_engine_features) stage.",
    )
    ap.add_argument(
        "--engine1-strict",
        action="store_true",
        help="If Engine 1 fails, exit non-zero (otherwise warn and continue).",
    )
    ap.add_argument(
        "--engine1-license-mode",
        type=str,
        default="academic",
        help="Engine1 decoupler license mode (academic|commercial).",
    )
    ap.add_argument(
        "--engine1-tf-method",
        type=str,
        default="viper",
        choices=["viper", "ulm"],
        help="Engine1 TF activity method.",
    )
    ap.add_argument(
        "--engine1-tmin",
        type=int,
        default=5,
        help="Engine1 VIPER min targets threshold (tmin).",
    )
    ap.add_argument(
        "--engine1-no-overwrite",
        action="store_true",
        help="Engine1: do not overwrite existing outputs if they already exist.",
    )
    ap.add_argument(
        "--engine1-no-regulators-evidence",
        action="store_true",
        help="Engine1: do not write regulators_evidence.tsv.",
    )

    args = ap.parse_args()

    # sanity
    if args.spec and args.item:
        ap.error("Use either --spec OR --item (not both).")
    if not args.spec and not args.item:
        ap.error("Provide --spec or at least one --item.")

    return args


# =============================================================================
# Small helpers
# =============================================================================

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _win_to_wsl(path_str: str) -> str:
    s = (path_str or "").strip().strip('"').strip("'")
    if not s:
        return s
    # Respect absolute POSIX paths as provided
    if s.startswith("/"):
        return s
    # Only translate Windows paths when running inside WSL
    if os.name == "nt" and not os.environ.get("WSL_DISTRO_NAME"):
        return s
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if not m:
        return s
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _parse_item(item: str) -> Dict[str, str]:
    raw = (item or "").strip()
    if not raw:
        raise ValueError("Empty --item value")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip().lower()] = v.strip().strip('"').strip("'")

    if "name" not in out or "input" not in out:
        raise ValueError(f"--item must include name=... and input=... (got: {item})")

    out["input"] = _win_to_wsl(out["input"])
    if "meta" in out and out["meta"]:
        out["meta"] = _win_to_wsl(out["meta"])
    if "tissue" in out and out["tissue"]:
        out["tissue"] = out["tissue"].strip()

    return out


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _best_fuzzy_match(query: str, options: List[str]) -> Tuple[str, float]:
    best = ("", 0.0)
    for opt in options:
        s = _similarity(query, opt)
        if s > best[1]:
            best = (opt, s)
    return best


def _llm_pick_tissue(disease: str, options: List[str], model: str = "gpt-4o-mini") -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None


    prompt = (
        "Pick the single best matching tissue from the list for this disease. "
        "Return ONLY the tissue name exactly as it appears in the list.\n\n"
        f"Disease: {disease}\n\n"
        "Tissue list:\n"
        + "\n".join(f"- {t}" for t in options)
        + "\n"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        txt = (res.choices[0].message.content or "").strip()
        return txt.splitlines()[0].strip()
    except Exception:
        return None


def _normalize_tissue(
    disease: str,
    provided_tissue: Optional[str],
    consensus_tissues: List[str],
) -> Optional[str]:
    if not consensus_tissues:
        return None

    canon = {t.lower(): t for t in consensus_tissues}

    if provided_tissue:
        t0 = provided_tissue.strip()
        if t0.lower() in canon:
            return canon[t0.lower()]

        best, score = _best_fuzzy_match(t0, consensus_tissues)
        if score >= 0.80:
            return best

    best, score = _best_fuzzy_match(disease, consensus_tissues)
    if score >= 0.70:
        return best

    llm_choice = _llm_pick_tissue(disease, consensus_tissues)
    if llm_choice:
        if llm_choice.lower() in canon:
            return canon[llm_choice.lower()]
        best2, score2 = _best_fuzzy_match(llm_choice, consensus_tissues)
        if score2 >= 0.70:
            return best2

    return None


def _ensure_legacy_root(out_root: Path) -> None:
    legacy_root = out_root / "cohorts"
    _safe_mkdir(legacy_root)


def _find_ipaa_dir_for_disease(out_root: Path, disease: str) -> Optional[Path]:
    """
    Prefer <out_root>/<disease>/ if it contains expression_used.*
    Else fall back to <out_root>/cohorts/<disease>/
    """
    out_root = Path(out_root)
    disease = str(disease)

    primary = out_root / disease
    alt = out_root / "cohorts" / disease

    for base in (primary, alt):
        for nm in ("expression_used.tsv", "expression_used.csv", "expression_used.txt"):
            if (base / nm).exists():
                return base
    return None


def _read_expr_used_for_sample_axis(expr_path: Path) -> List[str]:
    """
    Robustly extract sample IDs from expression_used.* by choosing axis with better overlap later.
    Here we return BOTH axes as strings; caller decides using metadata overlap.
    """
    expr_path = Path(expr_path)
    suf = expr_path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    df = pd.read_csv(expr_path, sep=sep, index_col=0)
    idx = [str(x).strip() for x in df.index.astype(str).tolist()]
    cols = [str(x).strip() for x in df.columns.astype(str).tolist()]
    # return both; caller will decide
    return idx, cols


def _apply_metadata_labels_after_cohort_runs(
    *,
    out_root: Path,
    cohort_runs: List[Dict[str, str]],
    meta_by_disease: Dict[str, str],
) -> None:
    """
    Ensures labels_used.tsv is written/overwritten from metadata for each disease that has meta.
    This guarantees Engine1 regulators evidence uses metadata grouping even if older m6_processing
    doesn't support meta internally.
    """
    if not meta_by_disease:
        return
    if align_labels_to_samples is None or write_labels_used is None:
        LOG.warning("Metadata helper module not available; cannot apply meta labels.")
        return

    for cr in cohort_runs:
        disease = (cr.get("name") or "").strip()
        if not disease:
            continue
        meta_path = meta_by_disease.get(disease)
        if not meta_path:
            continue

        ddir = _find_ipaa_dir_for_disease(out_root, disease)
        if ddir is None:
            LOG.warning("[meta] Could not find disease output dir for '%s' to write labels_used.tsv", disease)
            continue

        # find expression_used.*
        expr_file = None
        for nm in ("expression_used.tsv", "expression_used.csv", "expression_used.txt"):
            p = ddir / nm
            if p.exists():
                expr_file = p
                break
        if expr_file is None:
            LOG.warning("[meta] expression_used.* not found for '%s' (dir=%s)", disease, ddir)
            continue

        try:
            idx, cols = _read_expr_used_for_sample_axis(expr_file)
            # decide sample axis by metadata overlap
            meta_series = None
            try:
                from ipaa_metadata_labels import load_metadata_labels  # type: ignore
                meta_series = load_metadata_labels(Path(meta_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load metadata for {disease}: {e}") from e

            meta_ids = set(meta_series.index.astype(str).tolist())
            ov_idx = len([s for s in idx if s in meta_ids])
            ov_cols = len([s for s in cols if s in meta_ids])

            sample_ids = cols if ov_cols >= ov_idx else idx

            labels = align_labels_to_samples(meta_path=Path(meta_path), sample_ids=sample_ids, strict=True)

            # if metadata has >2 groups, we keep it (but Engine1 uses top-2)
            if labels.nunique() > 2:
                vc = labels.value_counts()
                LOG.warning(
                    "[meta] '%s' has >2 conditions (%s). Keeping all in labels_used.tsv; "
                    "downstream two-group stats will typically use the top-2 by frequency.",
                    disease, ", ".join([f"{k}:{v}" for k, v in vc.items()])
                )

            # write to primary dir
            write_labels_used(labels, ddir / "labels_used.tsv")
            LOG.info("[meta] wrote labels_used.tsv for '%s' -> %s", disease, ddir / "labels_used.tsv")

            # also write to the mirror dir if it exists
            mirror = (Path(out_root) / "cohorts" / disease)
            if mirror.exists():
                write_labels_used(labels, mirror / "labels_used.tsv")
                LOG.info("[meta] wrote labels_used.tsv mirror for '%s' -> %s", disease, mirror / "labels_used.tsv")

            # (Optional) log group ordering
            try:
                a, b = pick_two_groups(labels)
                LOG.info("[meta] group ordering for '%s': group_A=%s | group_B=%s", disease, a, b)
            except Exception:
                pass

        except Exception as e:
            LOG.warning("[meta] Failed to apply metadata labels for '%s': %s", disease, e)


# =============================================================================
# Engine 1 integration (causal_engine_features.py)
# =============================================================================

def _import_engine1_module():
    """
    Import causal_engine_features as part of ipaa_causality package so relative
    imports (e.g. from .mdp_engine...) work. Fallback to bare import for CLI/legacy.
    """
    try:
        from agentic_ai_wf.ipaa_causality import causal_engine_features as eng1
        return eng1, "causal_engine_features"
    except Exception:
        try:
            import causal_engine_features as eng1  # type: ignore
            return eng1, "causal_engine_features"
        except Exception as e:
            raise RuntimeError(
                "Engine1 import failed. causal_engine_features requires package context. "
                "Run from project root or ensure agentic_ai_wf is on PYTHONPATH."
            ) from e


def _run_engine1_all_cohorts(
    *,
    out_root: Path,
    cohort_runs: List[Dict[str, str]],
    license_mode: str,
    tf_method: str,
    tmin: int,
    overwrite: bool,
    also_write_regulators_evidence: bool,
) -> Dict[str, Dict[str, str]]:
    eng1, module_name = _import_engine1_module()

    fn = getattr(eng1, "run_engine1_causal_features", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            f"{module_name} imported but run_engine1_causal_features(...) not found/callable."
        )

    artifacts_by_disease: Dict[str, Dict[str, str]] = {}
    for cr in cohort_runs:
        disease = (cr.get("name") or "").strip()
        if not disease:
            LOG.warning("[Engine1] cohort_runs entry missing 'name'; skipping one cohort.")
            continue

        LOG.info("[Engine1] Running causal features for: %s", disease)
        arts = fn(
            out_root=Path(out_root),
            disease=str(disease),
            license_mode=str(license_mode),
            tf_method=str(tf_method),
            tmin=int(tmin),
            overwrite=bool(overwrite),
            also_write_regulators_evidence=bool(also_write_regulators_evidence),
        )
        if isinstance(arts, dict):
            artifacts_by_disease[str(disease)] = {str(k): str(v) for k, v in arts.items()}
        else:
            artifacts_by_disease[str(disease)] = {}

    return artifacts_by_disease


# =============================================================================
# ProcessPool worker wrapper (must be top-level for pickling)
# =============================================================================

def _worker_run_cohort(payload: Dict[str, Any]) -> Dict[str, str]:
    
    _logging.basicConfig(level=_logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    import m6_processing as _m6

    return _m6.run_one_cohort(
        cohort=payload["cohort"],
        spec=payload["spec"],
        out_root=payload["out_root"],
        gmt=payload["gmt"],
        counts_default=payload["counts_default"],
        tissue_expect=payload["tissue_expect"],
        seed=payload["seed"],
        refresh_omnipath=payload["refresh_omnipath"],
    )


# =============================================================================
# Config-based entry (no argparse; used by ipaa_service)
# =============================================================================


def _items_to_arg_strings(items: List[Any]) -> List[str]:
    """Convert ItemSpec/dict list to --item string format."""
    out: List[str] = []
    for it in items:
        if isinstance(it, dict):
            name = (it.get("name") or "").strip() or "UnnamedDisease"
            inp = (it.get("input") or it.get("path") or "").strip()
            meta = (it.get("meta") or "").strip()
            tissue = (it.get("tissue") or "").strip()
        else:
            name = getattr(it, "name", "") or "UnnamedDisease"
            inp = getattr(it, "input", "") or ""
            meta = (getattr(it, "meta", None) or "") or ""
            tissue = (getattr(it, "tissue", None) or "") or ""
        parts = [f"name={name}", f"input={inp}"]
        if meta:
            parts.append(f"meta={meta}")
        if tissue:
            parts.append(f"tissue={tissue}")
        out.append(",".join(parts))
    return out


def _get_cfg_val(cfg: Any, key: str, default: Any) -> Any:
    """Safe config value access: dict.get() or getattr(). Pydantic models lack .get()."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _config_to_args(cfg: Any) -> "argparse.Namespace":
    """Build args-like object from IPAAPhaseConfig or dict."""
    import argparse
    items = cfg.items if hasattr(cfg, "items") else _get_cfg_val(cfg, "items", [])
    outdir = _get_cfg_val(cfg, "outdir", "")
    spec = _get_cfg_val(cfg, "spec", None)
    return argparse.Namespace(
        outdir=outdir,
        spec=spec,
        item=_items_to_arg_strings(items),
        counts_default=_get_cfg_val(cfg, "counts_default", False),
        verbose=_get_cfg_val(cfg, "verbose", True),
        refresh_omnipath=_get_cfg_val(cfg, "refresh_omnipath", False),
        workers=_get_cfg_val(cfg, "workers", 0),
        threads_per_cohort=_get_cfg_val(cfg, "threads_per_cohort", 0),
        report_top=_get_cfg_val(cfg, "report_top", 20),
        threads=_get_cfg_val(cfg, "threads", 6),
        gsea_permutations=_get_cfg_val(cfg, "gsea_permutations", 200),
        sig_fdr=_get_cfg_val(cfg, "sig_fdr", 0.05),
        sig_top_n=_get_cfg_val(cfg, "sig_top_n", 200),
        msigdb_dbver=_get_cfg_val(cfg, "msigdb_dbver", "2024.1.Hs"),
        run_baseline=_get_cfg_val(cfg, "run_baseline", True),
        baseline_dir=_get_cfg_val(cfg, "baseline_dir", ""),
        auto_select_tissue=_get_cfg_val(cfg, "auto_select_tissue", True),
        tissue_top_k=_get_cfg_val(cfg, "tissue_top_k", 3),
        skip_engine1=_get_cfg_val(cfg, "skip_engine1", False),
        engine1_strict=_get_cfg_val(cfg, "engine1_strict", False),
        engine1_license_mode=_get_cfg_val(cfg, "engine1_license_mode", "academic"),
        engine1_tf_method=_get_cfg_val(cfg, "engine1_tf_method", "viper"),
        engine1_tmin=_get_cfg_val(cfg, "engine1_tmin", 5),
        engine1_no_overwrite=_get_cfg_val(cfg, "engine1_no_overwrite", False),
        engine1_no_regulators_evidence=_get_cfg_val(cfg, "engine1_no_regulators_evidence", False),
    )


def run_ipaa_from_config(config: Any) -> int:
    """
    Run IPAA from config object (IPAAPhaseConfig or dict).
    No argparse, no sys.argv. Returns exit code. Used by ipaa_service.
    """
    args = _config_to_args(config)
    return _main_impl(args)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()
    return _main_impl(args)


def _main_impl(args: argparse.Namespace) -> int:
    # ensure script directory is importable
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Use m6.normalize_path for safe Windows-path-under-WSL handling
    out_root = m6.normalize_path(_win_to_wsl(args.outdir)).expanduser()
    _safe_mkdir(out_root)

    # ensure legacy root exists BEFORE running cohorts (enables mirroring)
    _ensure_legacy_root(out_root)

    # Logging
    m6.setup_logging(out_root, verbose=bool(args.verbose))
    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    meta_by_disease: Dict[str, str] = {}

    # ---------------------------------------------------------------------
    # Build/load spec
    # ---------------------------------------------------------------------
    if args.spec:
        spec_path = m6.normalize_path(args.spec).expanduser()
        if not spec_path.exists():
            LOG.error("Spec file not found: %s", spec_path)
            return 2
        spec = m6.load_spec(spec_path)
        spec_origin = str(spec_path)
        LOG.info("Loaded spec with %d cohorts.", len(spec.cohorts))

        # collect meta if present in spec cohorts
        for c in spec.cohorts:
            mp = getattr(c, "meta", None)
            if mp:
                meta_by_disease[str(c.name)] = str(mp)

    else:
        cohorts: List[Any] = []
        try:
            import auto_tissue_baseline as atb  # optional
        except Exception:
            atb = None  # type: ignore

        for item in args.item:
            d = _parse_item(item)
            name = d.get("name") or "UnnamedDisease"
            inp_raw = d.get("input") or d.get("path") or ""
            if not inp_raw:
                raise SystemExit(f"--item missing input for cohort '{name}'")

            inp = m6.normalize_path(_win_to_wsl(inp_raw)).expanduser()
            if not inp.exists():
                raise SystemExit(f"Input not found for '{name}': {inp}  (from: {inp_raw})")

            meta_raw = (d.get("meta") or "").strip()
            meta_path: Optional[Path] = None
            if meta_raw:
                mp = m6.normalize_path(_win_to_wsl(meta_raw)).expanduser()
                if not mp.exists():
                    raise SystemExit(f"Metadata not found for '{name}': {mp}  (from: {meta_raw})")
                meta_path = mp
                meta_by_disease[str(name)] = str(mp)

            # Detect counts vs DEGs (best effort).
            if atb is not None:
                try:
                    kind = str(atb.detect_input_kind(inp)).strip().lower()
                except Exception:
                    kind = "counts" if bool(args.counts_default) else "degs"
            else:
                kind = "counts" if bool(args.counts_default) else "degs"

            counts_flag = True if kind == "counts" else False

            tissue_override = (d.get("tissue") or "").strip()
            tissue_final: Optional[str] = None
            if tissue_override:
                if atb is not None:
                    try:
                        tissue_final = atb.resolve_tissue_or_die(
                            disease=name,
                            tissue_in=tissue_override,
                            hint=str(inp),
                            allow_llm=bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
                        )
                    except Exception as e:
                        LOG.warning(
                            "Tissue '%s' not usable for '%s' (%s). Falling back to auto tissue selection.",
                            tissue_override, name, e
                        )
                        tissue_final = None
                else:
                    tissue_final = tissue_override

            # Backward compatible CohortSpec construction:
            # only pass meta if the installed CohortSpec supports it.
            cohort_kwargs: Dict[str, Any] = dict(
                name=name,
                expr=str(inp),
                counts=counts_flag,
                tissue=tissue_final,
            )
            annotations = getattr(m6.CohortSpec, "__annotations__", {}) or {}
            if meta_path is not None and "meta" in annotations:
                cohort_kwargs["meta"] = str(meta_path)

            cohorts.append(m6.CohortSpec(**cohort_kwargs))

        spec = m6.PipelineSpec(
            cohorts=cohorts,
            run_baseline=bool(args.run_baseline),
            baseline_dir=(args.baseline_dir or None),
            auto_select_tissue=bool(args.auto_select_tissue) or any(getattr(c, "tissue", None) is None for c in cohorts),
            tissue_top_k=int(args.tissue_top_k),
            threads=int(args.threads),
            gsea_permutations=int(args.gsea_permutations),
            sig_fdr=float(args.sig_fdr),
            sig_top_n=int(args.sig_top_n),
            msigdb_dbver=str(args.msigdb_dbver),
        )
        spec_origin = "--item"
        LOG.info("Built spec from %d --item entries.", len(cohorts))

    # ---------------------------------------------------------------------
    # Load GMT
    # ---------------------------------------------------------------------
    LOG.info("Fetching MSigDB C2 canonical pathways (c2.cp)...")
    gmt = m6.fetch_msigdb_c2_cp_gmt(dbver=spec.msigdb_dbver)

    # ---------------------------------------------------------------------
    # Optional baseline build (once)
    # ---------------------------------------------------------------------
    tissue_expect: Optional[pd.DataFrame] = None
    consensus_tissues: List[str] = []

    if bool(spec.run_baseline):
        base_dir = m6.normalize_path(spec.baseline_dir).expanduser() if spec.baseline_dir else (Path(__file__).parent / "data")
        LOG.info("Building baseline consensus from: %s", base_dir)
        try:
            Z_cons = m6.build_consensus_baseline(base_dir, spec.hpa_file, spec.gtex_file, spec.fantom_file)
            base_out = out_root / "baseline_consensus"
            _safe_mkdir(base_out)
            Z_cons.to_csv(base_out / "consensus_Z_tissue_x_gene.tsv", sep="\t")

            tissue_expect = m6.consensus_pathway_expectations(Z_cons, gmt, min_genes=5)
            tissue_expect.to_csv(base_out / "baseline_pathway_expectations_c2cp.tsv", sep="\t", index=False)

            if hasattr(m6, "write_per_tissue_expectations"):
                m6.write_per_tissue_expectations(out_root, tissue_expect)

            LOG.info("Baseline consensus + pathway expectations written under: %s", base_out)

            if tissue_expect is not None and not tissue_expect.empty and "tissue" in tissue_expect.columns:
                consensus_tissues = sorted(set(tissue_expect["tissue"].astype(str).tolist()))
        except Exception as e:
            LOG.warning("Baseline build failed (non-fatal): %s", e)
            tissue_expect = None
            consensus_tissues = []
    else:
        LOG.info("Baseline disabled (run_baseline=false).")

    # ---------------------------------------------------------------------
    # If any cohort tissue is missing/invalid -> map to consensus tissue list
    # ---------------------------------------------------------------------
    if consensus_tissues:
        changed = 0
        for c in spec.cohorts:
            before = getattr(c, "tissue", None)
            after = _normalize_tissue(getattr(c, "name", ""), before, consensus_tissues)
            if after and after != before:
                c.tissue = after
                changed += 1
                LOG.info("Tissue mapped: %s | %s -> %s", c.name, before or "AUTO", after)
        if changed:
            LOG.info("Tissue mapping applied to %d cohort(s).", changed)

    # ---------------------------------------------------------------------
    # Parallel planning
    # ---------------------------------------------------------------------
    threads_per_cohort = int(args.threads_per_cohort) if int(args.threads_per_cohort) > 0 else int(spec.threads)
    threads_per_cohort = max(1, threads_per_cohort)

    refresh = bool(args.refresh_omnipath)

    if int(args.workers) > 0:
        workers = int(args.workers)
        cpu = os.cpu_count() or 1
        LOG.info(
            "Cohort execution: workers=%d (manual override), cpu=%d, threads_per_cohort=%d",
            workers,
            cpu,
            threads_per_cohort,
        )
    else:
        workers, cpu, tpc = m6.choose_workers(len(spec.cohorts), threads_per_cohort)
        LOG.info("Cohort execution: workers=%d (cpu=%d, threads_per_cohort=%d)", workers, cpu, tpc)

    # ---------------------------------------------------------------------
    # Run cohorts
    # ---------------------------------------------------------------------
    cohort_runs: List[Dict[str, str]] = []

    if workers <= 1 or len(spec.cohorts) <= 1:
        for cohort in spec.cohorts:
            cr = m6.run_one_cohort(
                cohort=cohort,
                spec=spec,
                out_root=out_root,
                gmt=gmt,
                counts_default=bool(args.counts_default),
                tissue_expect=tissue_expect,
                seed=0,
                refresh_omnipath=refresh,
            )
            cohort_runs.append(cr)
    else:
        payloads: List[Dict[str, Any]] = []
        for cohort in spec.cohorts:
            payloads.append(
                {
                    "cohort": cohort,
                    "spec": spec,
                    "out_root": out_root,
                    "gmt": gmt,
                    "counts_default": bool(args.counts_default),
                    "tissue_expect": tissue_expect,
                    "seed": 0,
                    "refresh_omnipath": refresh,
                }
            )

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_run_cohort, p) for p in payloads]
            for fut in as_completed(futs):
                try:
                    cr = fut.result()
                except Exception as e:
                    LOG.exception("Cohort worker failed: %s", e)
                    raise
                cohort_runs.append(cr)

        cohort_runs.sort(key=lambda d: d.get("name", ""))

    # ---------------------------------------------------------------------
    # Apply metadata -> labels_used.tsv (guaranteed for Engine1)
    # ---------------------------------------------------------------------
    try:
        _apply_metadata_labels_after_cohort_runs(
            out_root=out_root,
            cohort_runs=cohort_runs,
            meta_by_disease=meta_by_disease,
        )
    except Exception as e:
        LOG.warning("[meta] post-run label application failed (non-fatal): %s", e)

    # ---------------------------------------------------------------------
    # Post-stage 1: TF/Epigenetic/Metabolite enrichments + overlap JSONs
    # ---------------------------------------------------------------------
    try:
        import ipaa_entities_overlap as eo

        eo.run_entities_and_overlap_all(
            out_root=out_root,
            cohort_runs=cohort_runs,
            gmt=gmt,
            expressed_genes_by_cohort=None,
            gene_fdr_for_updown=0.05,
            reduce_overlap=True,
            max_pathway_jaccard=0.50,
            min_genes_expressed=10,
            run_enrichments_if_missing=True,
        )
        LOG.info("Entity enrichments + overlap JSONs complete (mirrored to out_root/jsons_all).")
    except Exception as e:
        LOG.warning("Entity/overlap stage failed (non-fatal): %s", e)

    # ---------------------------------------------------------------------
    # Post-stage 2: Pathway classification
    # ---------------------------------------------------------------------
    try:
        import ipaa_pathway_classify as pc

        pc.classify_ipaa_pathways_all(
            out_root=out_root,
            cohort_runs=cohort_runs,
            gmt=gmt,
            reduce_overlap=True,
            max_pathway_jaccard=0.50,
            min_genes_expressed=10,
            sig_fdr=float(spec.sig_fdr),
            sig_top_n=int(spec.sig_top_n),
        )
        LOG.info("Pathway classification complete (classified pathway tables written per cohort).")
    except Exception as e:
        LOG.warning("Pathway classification stage failed (non-fatal): %s", e)

    # ---------------------------------------------------------------------
    # ENGINE 1: causal features
    # ---------------------------------------------------------------------
    engine1_ok = False
    engine1_artifacts: Dict[str, Dict[str, str]] = {}

    if bool(args.skip_engine1):
        LOG.info("Engine 1 skipped (--skip-engine1).")
    else:
        LOG.info("Running Engine 1 (causal_engine_features) for all cohorts...")
        try:
            engine1_artifacts = _run_engine1_all_cohorts(
                out_root=out_root,
                cohort_runs=cohort_runs,
                license_mode=str(args.engine1_license_mode),
                tf_method=str(args.engine1_tf_method),
                tmin=int(args.engine1_tmin),
                overwrite=(not bool(args.engine1_no_overwrite)),
                also_write_regulators_evidence=(not bool(args.engine1_no_regulators_evidence)),
            )
            engine1_ok = True
            LOG.info("Engine 1 complete for %d cohort(s).", len(engine1_artifacts))
        except Exception as e:
            LOG.error("Engine 1 failed: %s", e, exc_info=bool(args.verbose))
            if bool(args.engine1_strict):
                LOG.error("Exiting because --engine1-strict is set.")
                return 3
            LOG.warning("Continuing despite Engine 1 failure (non-fatal).")

    # ---------------------------------------------------------------------
    # Compare stage
    # ---------------------------------------------------------------------
    LOG.info("Running cross-cohort comparator stage...")
    try:
        universe_n = len(
            set().union(
                *[
                    set(pd.read_csv(cr["pathway_stats"], sep="\t")["pathway"].astype(str).tolist())
                    for cr in cohort_runs
                    if cr.get("pathway_stats")
                ]
            )
        )
    except Exception:
        universe_n = 0

    m6.run_compare_stage(cohort_runs, spec, out_root, universe_n=universe_n)

    # ---------------------------------------------------------------------
    # Report stage
    # ---------------------------------------------------------------------
    report_path: Optional[Path] = None
    # try:
    #     from ipaa_report_best import build_ipaa_style_report

    #     report_path = build_ipaa_style_report(
    #         out_root=out_root,
    #         cohort_runs=cohort_runs,
    #         spec=spec,
    #         report_top=int(args.report_top),
    #     )
    #     LOG.info("IPAA report written: %s", report_path)
    # except Exception as e:
    #     LOG.exception("Report build failed: %s", e)

    # ---------------------------------------------------------------------
    # Pipeline manifest
    # ---------------------------------------------------------------------
    manifest: Dict[str, Any] = {
        "timestamp": _now_utc(),
        "spec_origin": spec_origin,
        "outdir": str(out_root),
        "counts_default": bool(args.counts_default),
        "refresh_omnipath": bool(refresh),
        "execution": {
            "workers": int(workers),
            "threads_per_cohort_budget": int(threads_per_cohort),
            "spec_threads": int(spec.threads),
        },
        "metadata": {
            "used": bool(bool(meta_by_disease)),
            "meta_by_disease": meta_by_disease,
            "required_columns": ["sample_id", "condition"],
        },
        "engine1": {
            "module": "causal_engine_features (or casual_engine_features fallback)",
            "attempted": (not bool(args.skip_engine1)),
            "ok": bool(engine1_ok),
            "strict": bool(args.engine1_strict),
            "params": {
                "license_mode": str(args.engine1_license_mode),
                "tf_method": str(args.engine1_tf_method),
                "tmin": int(args.engine1_tmin),
                "overwrite": (not bool(args.engine1_no_overwrite)),
                "write_regulators_evidence": (not bool(args.engine1_no_regulators_evidence)),
            },
            "artifacts_by_disease": engine1_artifacts,
        },
        "report": {
            "report_md": str(report_path) if report_path else "",
            "root_link": str(out_root / "REPORT.md"),
        },
        "spec": {
            **{k: v for k, v in asdict(spec).items() if k != "cohorts"},
            "n_cohorts": len(spec.cohorts),
        },
        "cohorts": [asdict(c) for c in spec.cohorts],
    }
    (out_root / "PIPELINE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOG.info("DONE. Outputs at: %s", out_root)
    LOG.info("Open report: %s", out_root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
