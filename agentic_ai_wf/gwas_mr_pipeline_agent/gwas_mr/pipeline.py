"""End-to-end GWAS retrieval + Mendelian Randomization orchestrator.

The single entry-point :func:`run_full_pipeline` chains GWAS data
retrieval with MR analysis and writes all results under one output
directory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .retrieval import log, retrieve_gwas_data, safe_name
from .mr_runner import run_mr, _find_rscript, find_r_script, _find_plink, _resolve_plink_ref
from .report import generate_report


# ── Preflight dependency check ─────────────────────────────────────────────

def _check_python_imports() -> List[str]:
    """Return a list of missing Python packages required by the pipeline."""
    missing = []
    required = {
        "pandas": "pandas",
        "requests": "requests",
        "openpyxl": "openpyxl",
    }
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def _check_openai_key() -> bool:
    """Return True if the OpenAI API key is discoverable."""
    try:
        from decouple import config, UndefinedValueError
        config("OPENAI_API_KEY")
        return True
    except Exception:
        return False


def preflight_check(
    *,
    tsv_path: Optional[str] = None,
    eqtl_root: Optional[str] = None,
    gtex_biosample_csv: Optional[str] = None,
    r_script_path: Optional[str] = None,
    plink_bin: Optional[str] = None,
    plink_ref: Optional[str] = None,
    use_llm: bool = True,
    run_mr_analysis: bool = True,
) -> bool:
    """Validate that all dependencies are in place before running.

    Prints a status report and returns ``True`` if everything required
    is present.  Missing *optional* items produce warnings but do not
    cause the check to fail.
    """
    from .defaults import (
        DEFAULT_TSV_PATH,
        DEFAULT_EQTL_ROOT,
        DEFAULT_GTEX_BIOSAMPLE_CSV,
        DEFAULT_PLINK_DIR,
    )

    ok_sym = "[OK]   "
    warn_sym = "[WARN] "
    fail_sym = "[FAIL] "

    all_ok = True

    log("=" * 60)
    log("PREFLIGHT DEPENDENCY CHECK")
    log("=" * 60)

    # ── 1. Python packages ─────────────────────────────────────────────
    log("\n--- Python packages ---")
    missing_py = _check_python_imports()
    if missing_py:
        all_ok = False
        log(f"{fail_sym}Missing Python packages: {', '.join(missing_py)}")
        log(f"       Fix: pip install {' '.join(missing_py)}")
    else:
        log(f"{ok_sym}All required Python packages installed (pandas, requests, openpyxl)")

    if use_llm:
        try:
            __import__("openai")
            log(f"{ok_sym}openai package installed")
        except ImportError:
            all_ok = False
            log(f"{fail_sym}openai package not installed (needed when use_llm=True)")
            log("       Fix: pip install openai")

        try:
            __import__("decouple")
            log(f"{ok_sym}python-decouple package installed")
        except ImportError:
            all_ok = False
            log(f"{fail_sym}python-decouple package not installed")
            log("       Fix: pip install python-decouple")

    # ── 2. OpenAI API key ──────────────────────────────────────────────
    log("\n--- OpenAI API key ---")
    if use_llm:
        if _check_openai_key():
            log(f"{ok_sym}OPENAI_API_KEY found")
        else:
            all_ok = False
            log(f"{fail_sym}OPENAI_API_KEY not found")
            log("       Fix: Create a .env file with OPENAI_API_KEY=sk-...")
            log("            or set it as a system environment variable.")
    else:
        log(f"{ok_sym}LLM disabled (use_llm=False) — API key not required")

    # ── 3. Reference data files ────────────────────────────────────────
    log("\n--- Reference data files ---")
    _tsv = tsv_path or DEFAULT_TSV_PATH
    _eqtl = eqtl_root or DEFAULT_EQTL_ROOT
    _gtex = gtex_biosample_csv or DEFAULT_GTEX_BIOSAMPLE_CSV

    if Path(_tsv).is_file():
        log(f"{ok_sym}GWAS catalog TSV: {_tsv}")
    else:
        all_ok = False
        log(f"{fail_sym}GWAS catalog TSV not found: {_tsv}")
        log("       Fix: Download from the Google Drive link and place in gwas_mr_reference/")

    if Path(_eqtl).is_dir():
        biosample_count = len(list(Path(_eqtl).glob("*.txt")))
        log(f"{ok_sym}GTEx eQTL directory: {_eqtl} ({biosample_count} biosample files)")
    else:
        all_ok = False
        log(f"{fail_sym}GTEx eQTL directory not found: {_eqtl}")
        log("       Fix: Download from the Google Drive link and extract into gwas_mr_reference/")

    if Path(_gtex).is_file():
        log(f"{ok_sym}GTEx biosample CSV: {_gtex}")
    else:
        all_ok = False
        log(f"{fail_sym}GTEx biosample sample-count CSV not found: {_gtex}")
        log("       Fix: Download from the Google Drive link and extract into gwas_mr_reference/")

    # ── 4. R / Rscript (only if MR is enabled) ─────────────────────────
    if run_mr_analysis:
        log("\n--- R / Rscript ---")
        rscript = _find_rscript()
        if rscript:
            log(f"{ok_sym}Rscript found: {rscript}")
        else:
            all_ok = False
            log(f"{fail_sym}Rscript not found on PATH or via R_HOME")
            log("       Fix: Install R and add to PATH, or set R_HOME env var")

        # ── 5. mr_pipeline.R ───────────────────────────────────────────
        log("\n--- mr_pipeline.R ---")
        try:
            r_script = find_r_script(r_script_path)
            log(f"{ok_sym}R script: {r_script}")
        except FileNotFoundError:
            all_ok = False
            log(f"{fail_sym}mr_pipeline.R not found")
            log("       Fix: Ensure it exists in the gwas_mr/ package directory")

        # ── 6. PLINK ───────────────────────────────────────────────────
        log("\n--- PLINK 1.9 ---")
        resolved_plink = _find_plink(plink_bin)
        if resolved_plink:
            log(f"{ok_sym}PLINK binary: {resolved_plink}")
        else:
            all_ok = False
            log(f"{fail_sym}PLINK not found")
            log("       Fix: Download from https://www.cog-genomics.org/plink/")
            log("            Place in gwas_mr_reference/plink/ or set PLINK_BIN env var")

        resolved_ref = _resolve_plink_ref(plink_ref)
        ref_bed = Path(resolved_ref + ".bed")
        if ref_bed.is_file():
            log(f"{ok_sym}PLINK reference panel: {resolved_ref}")
        else:
            all_ok = False
            log(f"{fail_sym}PLINK reference panel not found: {resolved_ref}")
            log("       Fix: Download 1000G EUR panel and place .bed/.bim/.fam in")
            log(f"            gwas_mr_reference/plink/ as 1000G_EUR_hg38.*")

        # ── 7. R packages (check via Rscript) ─────────────────────────
        if rscript:
            log("\n--- R packages ---")
            r_check_code = (
                'all_pkgs <- c("data.table","TwoSampleMR","ggplot2",'
                '"coloc","susieR","pheatmap","jsonlite");'
                'missing <- all_pkgs[!sapply(all_pkgs,requireNamespace,quietly=TRUE)];'
                'if(length(missing)) cat("MISSING:",paste(missing,collapse=","),"\\n") '
                'else cat("ALL_OK\\n")'
            )
            import subprocess
            try:
                proc = subprocess.run(
                    [rscript, "-e", r_check_code],
                    capture_output=True, text=True, timeout=60,
                )
                r_out = proc.stdout.strip()
                for line in r_out.splitlines():
                    if line.startswith("MISSING:"):
                        pkgs = line.split(":", 1)[1].strip()
                        all_ok = False
                        log(f"{fail_sym}Missing R packages: {pkgs}")
                        log("       Fix: In R console run:")
                        log('         install.packages(c("data.table","ggplot2","coloc",')
                        log('                           "susieR","pheatmap","jsonlite"))')
                        log('         install.packages("remotes")')
                        log('         remotes::install_github("MRCIEU/TwoSampleMR")')
                    elif line == "ALL_OK":
                        log(f"{ok_sym}All R packages installed (data.table, TwoSampleMR,")
                        log("         ggplot2, coloc, susieR, pheatmap, jsonlite)")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                all_ok = False
                log(f"{fail_sym}Could not verify R packages (Rscript timed out or failed)")
    else:
        log("\n--- R / PLINK ---")
        log(f"{ok_sym}MR analysis disabled (run_mr_analysis=False) — R/PLINK not required")

    # ── Summary ────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    if all_ok:
        log("PREFLIGHT CHECK PASSED — all required dependencies are in place.")
    else:
        log("PREFLIGHT CHECK FAILED — some required dependencies are missing.")
        log("See [FAIL] items above for details.")
    log("=" * 60 + "\n")

    return all_ok


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(
    disease_name: str,
    biosample_type: str,
    output_dir: str,
    gwas_data_dir: str,
    *,
    tsv_path: Optional[str] = None,
    eqtl_root: Optional[str] = None,
    gtex_biosample_csv: Optional[str] = None,
    r_script_path: Optional[str] = None,
    plink_bin: Optional[str] = None,
    plink_ref: Optional[str] = None,
    use_llm: bool = True,
    top_k: int = 1,
    run_mr_analysis: bool = True,
    skip_preflight: bool = False,
) -> List[Dict[str, Any]]:
    """
    End-to-end pipeline: retrieve GWAS data, then run MR analysis.

    Before execution a **preflight check** validates that all required
    dependencies (Python packages, API keys, reference data, R, PLINK)
    are available.  If any required dependency is missing the pipeline
    aborts early with a clear report.  Pass ``skip_preflight=True`` to
    bypass this check.

    All data-path parameters are optional and default to the
    ``gwas_mr_reference/`` directory at the repository root.

    Parameters
    ----------
    disease_name : str
        Disease or phenotype name (e.g. ``"Breast Cancer"``).
    biosample_type : str
        GTEx biosample type (e.g. ``"Whole Blood"``, ``"Pancreas"``, ``"PBMC"``).
    output_dir : str
        Root directory for MR results.  MR output goes to
        ``output_dir/mr_results/<trait>/``.
    gwas_data_dir : str
        Directory for downloaded GWAS summary statistics and cached
        files.  Kept separate from ``output_dir`` so that large GWAS
        downloads can be reused across multiple pipeline runs.
    tsv_path : str, optional
        Path to the GWAS catalog TSV.  Defaults to
        ``gwas_mr_reference/GWAS-DATABASE-FTP.tsv``.
    eqtl_root : str, optional
        Path to the GTEx eQTL biosample expression directory.
        Defaults to ``gwas_mr_reference/GTEx_eQTL_TISSUE_EXPRESSION/``.
    gtex_biosample_csv : str, optional
        Path to the GTEx biosample sample-count CSV.  Defaults to
        ``gwas_mr_reference/GTEx_EQTL_SAMPLE_COUNTS/EQTL-SAMPLE-COUNT-GTEx Portal.csv``.
    r_script_path : str, optional
        Path to ``mr_pipeline.R``.  Defaults to the copy bundled
        inside the ``gwas_mr`` package.
    plink_bin : str, optional
        Path to the PLINK 1.9 executable.  Auto-detected from ``PLINK_BIN``
        env var, system PATH, or ``gwas_mr_reference/plink/``.
    plink_ref : str, optional
        Prefix for the PLINK LD reference panel (``.bed/.bim/.fam``).
        Auto-detected from ``PLINK_REF`` env var or
        ``gwas_mr_reference/plink/1000G_EUR_hg38``.
    use_llm : bool
        Use OpenAI LLM for disease-name normalisation (default ``True``).
        The API key is read via ``python-decouple`` (checks ``.env``,
        environment variables, and ``settings.ini`` automatically).
    top_k : int
        Number of top GWAS rows per trait by sample size (default ``1``).
    run_mr_analysis : bool
        Whether to execute the MR step after retrieval (default ``True``).
    skip_preflight : bool
        Skip the preflight dependency check (default ``False``).

    Returns
    -------
    list[dict]
        One dict per processed trait.  Each dict contains all retrieval
        metadata plus ``mr_output_dir`` and ``mr_status`` when MR was run.
    """
    # ── Preflight check ────────────────────────────────────────────────
    if not skip_preflight:
        passed = preflight_check(
            tsv_path=tsv_path,
            eqtl_root=eqtl_root,
            gtex_biosample_csv=gtex_biosample_csv,
            r_script_path=r_script_path,
            plink_bin=plink_bin,
            plink_ref=plink_ref,
            use_llm=use_llm,
            run_mr_analysis=run_mr_analysis,
        )
        if not passed:
            raise RuntimeError(
                "Preflight check failed — required dependencies are missing. "
                "See the report above for details and fixes. "
                "Pass skip_preflight=True to bypass this check."
            )

    output_path = Path(output_dir)
    gwas_data_path = Path(gwas_data_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    gwas_data_path.mkdir(parents=True, exist_ok=True)
    mr_results_dir = output_path 

    log("=" * 60)
    log("GWAS-eQTL Full Pipeline")
    log(f"  Disease:    {disease_name}")
    log(f"  Biosample:  {biosample_type}")
    log(f"  GWAS data:  {gwas_data_path}")
    log(f"  MR output:  {output_dir}")
    log("=" * 60)

    # ── Step 1: Retrieve GWAS data ─────────────────────────────────────────
    log("\n[STEP 1/2] Retrieving GWAS data...")

    datasets = retrieve_gwas_data(
        disease_name=disease_name,
        biosample_type=biosample_type,
        tsv_path=tsv_path,
        eqtl_root=eqtl_root,
        gtex_biosample_csv=gtex_biosample_csv,
        out_root=str(gwas_data_path),
        use_llm=use_llm,
        top_k=top_k,
    )

    if not datasets:
        log("\n[DONE] No GWAS datasets retrieved. Pipeline stopped.")
        return datasets

    log(f"\n[OK] Retrieved {len(datasets)} dataset(s).")

    # ── Step 2: Run MR for each dataset ────────────────────────────────────
    if not run_mr_analysis:
        log("\n[SKIP] MR analysis skipped (run_mr_analysis=False).")
        for ds in datasets:
            ds["mr_status"] = "skipped"
            ds["mr_output_dir"] = None
        return datasets

    log("\n[STEP 2/2] Running Mendelian Randomization...")

    for ds in datasets:
        trait = ds.get("trait", "unknown")
        gwas_path = (
            ds.get("gwas_path")
            or ds.get("gwas_path_tsv")
            or ds.get("gwas_path_gz")
        )
        eqtl_path = ds.get("eqtl_path")
        n_eqtl = ds.get("n_eqtl_for_mr")
        n_gwas = ds.get("N_gwas")

        if not gwas_path or not eqtl_path:
            log(f"\n[SKIP] MR for '{trait}': missing GWAS or eQTL path.")
            ds["mr_status"] = "skipped_missing_paths"
            ds["mr_output_dir"] = None
            continue

        if not n_eqtl or not n_gwas:
            log(f"\n[SKIP] MR for '{trait}': missing sample size(s).")
            ds["mr_status"] = "skipped_missing_sample_sizes"
            ds["mr_output_dir"] = None
            continue

        mr_out = str(mr_results_dir / safe_name(trait))
        ds["mr_output_dir"] = mr_out

        log(f"\n{'─' * 50}")
        log(f"Running MR for: {trait}")
        log(f"  GWAS path:  {gwas_path}")
        log(f"  eQTL path:  {eqtl_path}")
        log(f"  N_eQTL:     {n_eqtl}")
        log(f"  N_GWAS:     {n_gwas}")
        log(f"  Output:     {mr_out}")

        try:
            run_mr(
                base_dir=str(output_path),
                eqtl_path=eqtl_path,
                gwas_path=gwas_path,
                n_eqtl=int(n_eqtl),
                n_gwas=int(n_gwas),
                out_dir=mr_out,
                disease_name=disease_name,
                biosample_type=biosample_type,
                r_script_path=r_script_path,
                plink_bin=plink_bin,
                plink_ref=plink_ref,
            )
            ds["mr_status"] = "success"
            log(f"[OK] MR completed for '{trait}'")
        except Exception as e:
            ds["mr_status"] = f"failed: {e}"
            log(f"[ERROR] MR failed for '{trait}': {e}")

        if ds["mr_status"] == "success":
            try:
                rpt = generate_report(
                    mr_output_dir=mr_out,
                    disease_name=disease_name,
                    biosample_type=biosample_type,
                    gwas_accession=ds.get("accession"),
                    n_gwas=n_gwas,
                    n_eqtl=n_eqtl,
                )
                if rpt:
                    log(f"[OK] Summary report: {rpt}")
            except Exception as rpt_err:
                log(f"[WARN] Report generation failed: {rpt_err}")

    # ── Summary ────────────────────────────────────────────────────────────
    succeeded = sum(1 for d in datasets if d.get("mr_status") == "success")
    total = len(datasets)

    log(f"\n{'=' * 60}")
    log(f"Pipeline complete: {succeeded}/{total} trait(s) processed successfully.")
    log(f"Results directory: {output_dir}")
    log("=" * 60)

    return datasets
