#!/usr/bin/env python3
"""
run_crispr_targeted_full.py

Orchestrates the full targeted CRISPR-seq analysis pipeline:
  1) Optional metadata extraction from SRA
  2) Optional FASTQ download
  3) Samplesheet generation (non-interactive)
  4) nf-core/crisprseq targeted pipeline execution
  5) Master HTML report generation

Designed as a standalone package — no hardcoded paths, no interactive
prompts. Call ``run_targeted_pipeline(...)`` with all required
parameters and the full flow runs end-to-end.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = Path("targeted_crispr_reference_data")
DEFAULT_HG38 = str(REFERENCE_DIR / "hg38.fa")
DEFAULT_GTF = str(REFERENCE_DIR / "gencode.v44.annotation.gtf")
NEXTFLOW_CONFIG = str(SCRIPT_DIR / "nextflow.config")
PLOTTER_PATCH_R = str(SCRIPT_DIR / "plotter_patch.R")


# ============================================================
# .env LOADER
# ============================================================
def _load_dotenv(*search_paths: str) -> None:
    """Load KEY=VALUE pairs from .env files into os.environ.

    Searches in the given paths first, then falls back to the current
    working directory. Already-set env vars are NOT overwritten.
    """
    candidates = [Path(p) / ".env" for p in search_paths]
    candidates.append(Path.cwd() / ".env")

    for env_file in candidates:
        if not env_file.is_file():
            continue
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
        return


# ============================================================
# COLOR + FORMAT HELPERS
# ============================================================
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
RESET = "\033[0m"

def red(msg): print(f"{RED}{msg}{RESET}")
def green(msg): print(f"{GREEN}{msg}{RESET}")
def yellow(msg): print(f"{YELLOW}{msg}{RESET}")
def cyan(msg): print(f"{CYAN}{msg}{RESET}")


# ============================================================
# INTERNAL HELPERS
# ============================================================
def _setup_environment(output_dir: str, work_dir: str = ""):
    """Create output sub-directories and export Nextflow env vars.

    Run-specific dirs (work, tmp) go inside output_dir so each run is
    self-contained.  The Singularity image cache stays in the user's
    home directory since container images are reusable across all runs.
    """
    out = Path(output_dir)
    results_dir = out / "results"
    nxf_work = Path(work_dir) if work_dir else out / "work"
    tmp_dir = out / "tmp"

    # Singularity images are identical across runs — cache them globally
    singularity_cache = Path.home() / "nxf_singularity_cache"

    for d in [nxf_work, singularity_cache, tmp_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["NXF_WORK"] = str(nxf_work)
    os.environ["NXF_SINGULARITY_CACHEDIR"] = str(singularity_cache)
    os.environ["NXF_CONDA_CACHEDIR"] = str(out / "conda_cache")
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["PLOTTER_PATCH_R"] = PLOTTER_PATCH_R

    return nxf_work, singularity_cache, tmp_dir, results_dir


def _run_extractor(input_dir: str, project_id: str):
    """Fetch SRA metadata into input_dir."""
    from .extractor import extract_metadata

    green("RUNNING METADATA EXTRACTION")
    csv_path = extract_metadata(project_id, input_dir)
    green(f"METADATA SAVED: {csv_path}")


def _run_downloader(input_dir: str):
    """Download FASTQs listed in input_dir/runs.csv."""
    from .downloading import download_fastqs

    green("DOWNLOADING FASTQ FILES")
    fastq_path = download_fastqs(input_dir)
    green(f"FASTQ FILES SAVED: {fastq_path}")


def _check_plotter_output(results_dir: Path):
    """Report whether CRISPRSEQ_PLOTTER produced plots."""
    plots_dir = results_dir / "plots"
    png_files = list(results_dir.rglob("*_delAlleles_plot.png"))
    html_files = list(results_dir.rglob("*_Deletions.html"))

    if png_files or html_files:
        green(f"CRISPRSEQ_PLOTTER: {len(png_files)} PNG + {len(html_files)} HTML plots generated")
    elif plots_dir.exists() and any(plots_dir.iterdir()):
        green("CRISPRSEQ_PLOTTER: plots directory found")
    else:
        yellow(
            "WARNING: CRISPRSEQ_PLOTTER did not produce plot files.\n"
            "  The pipeline completed but the R-based plotter may have failed.\n"
            "  Check the Nextflow log for CRISPRSEQ_PLOTTER errors."
        )


def _run_nextflow(
    samplesheet: str,
    results_dir: Path,
    output_dir: Path,
    profile: str,
    revision: str = "",
):
    """Execute nf-core/crisprseq targeted and auto-patch plotter on failure."""
    log_file = output_dir / ".nextflow.log"
    cmd = [
        "nextflow", "-log", str(log_file),
        "run", "nf-core/crisprseq",
        "-r", revision or "2.3.0",
        "-c", NEXTFLOW_CONFIG,
        "-profile", profile,
        "--analysis", "targeted",
        "--input", samplesheet,
        "--outdir", str(results_dir),
        "-resume",
    ]

    cyan("\nRUNNING COMMAND:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        red("\nPIPELINE FAILED (exit code %d)" % result.returncode)
        sys.exit(1)

    _check_plotter_output(results_dir)


# ============================================================
# PUBLIC API
# ============================================================
def run_targeted_pipeline(
    input_dir: str,
    output_dir: str,
    protospacer: str,
    hg38: str = DEFAULT_HG38,
    gtf: str = DEFAULT_GTF,
    target_gene: str = "",
    region: str = "",
    reference_seq: str = "",
    template_seq: str = "",
    gene_flank: int = 0,
    work_dir: str = "",
    project_id: str = "",
    samplesheet: str = "",
    profile: str = "singularity",
    revision: str = "2.3.0",
    extract_metadata: bool = False,
    download_fastq: bool = False,
    prepare_samplesheet: bool = True,
    run_nextflow: bool = True,
    generate_report: bool = True,
    no_llm: bool = False,
    openai_model: str = "gpt-4.1-mini",
) -> None:
    """Run the full targeted CRISPR-seq pipeline end-to-end.

    No interactive prompts — every parameter is provided up front.

    The reference sequence is resolved automatically based on which
    argument is supplied (checked in order):

    1. **reference_seq** → used directly (equivalent to Mode C).
    2. **region** (``chr:start-end``) → extracted from *hg38*
       (equivalent to Mode B).
    3. **target_gene** → looked up in *gtf*, then extracted from *hg38*
       (equivalent to Mode A).

    Parameters
    ----------
    input_dir : str
        Project data directory. Must contain ``fastq/`` with paired
        FASTQ files. May also contain ``runs.csv`` / ``runs.tsv``
        for automatic control classification.

    output_dir : str
        Base output directory.  Sub-directories are created
        automatically: ``results/``, ``work/``, ``conda_cache/``,
        ``tmp/``.

    protospacer : str
        Guide RNA (gRNA) sequence, e.g. ``"GGTGGATCCTATTCTAAACG"``.

    hg38 : str, optional
        Path to hg38 reference FASTA (indexed). Defaults to the copy
        bundled inside the ``targeted/reference/`` package directory.

    gtf : str, optional
        Path to GTF gene annotation. Defaults to the copy bundled
        inside the ``targeted/reference/`` package directory.

    target_gene : str, optional
        Gene symbol (e.g. ``"RAB11A"``). The gene coordinates are
        looked up in the GTF and the sequence is extracted from hg38.

    region : str, optional
        Genomic region ``chr:start-end``. The sequence is extracted
        from hg38.

    reference_seq : str, optional
        A DNA sequence to use directly as the amplicon reference.

    template_seq : str, optional
        HDR template sequence. Left blank if not provided.

    gene_flank : int, default 0
        Extra flanking bases around the gene body (max 100, Mode A
        only).

    work_dir : str, optional
        Nextflow work directory. Defaults to ``<output_dir>/work`` so
        each run is self-contained and easy to clean up.

    project_id : str, optional
        SRA accession. Only needed when *extract_metadata* or
        *download_fastq* is True.

    samplesheet : str, optional
        Path to an existing ``.csv`` samplesheet. When provided,
        samplesheet generation is skipped entirely.

    profile : str, default ``"conda"``
        Nextflow profile (``"conda"``, ``"singularity"``, ``"docker"``).

    revision : str, default ``"2.3.0"``
        nf-core/crisprseq pipeline revision to run.

    extract_metadata : bool, default False
        Run the SRA metadata extractor before the pipeline.

    download_fastq : bool, default False
        Download FASTQs from SRA before the pipeline.

    prepare_samplesheet : bool, default True
        Generate the samplesheet from FASTQ data in *input_dir*.

    run_nextflow : bool, default True
        Execute nf-core/crisprseq targeted.

    generate_report : bool, default True
        Generate the master HTML report from pipeline results after
        the Nextflow run completes.

    no_llm : bool, default False
        Disable LLM-based control classification even if an API key is
        available.

    openai_model : str, default ``"gpt-4.1-mini"``
        OpenAI model for LLM-based control classification.

    Raises
    ------
    SystemExit
        If a required parameter is missing or a subprocess fails.

    Examples
    --------
    Gene-based reference (Mode A)::

        run_targeted_pipeline(
            input_dir="extracted_data/PRJNA1240319",
            output_dir="results",
            protospacer="GGTGGATCCTATTCTAAACG",
            target_gene="RAB11A",
        )

    Region-based reference (Mode B)::

        run_targeted_pipeline(
            input_dir="extracted_data/PRJNA1240319",
            output_dir="results",
            protospacer="GGTGGATCCTATTCTAAACG",
            region="chr15:65869459-65891991",
        )

    Direct reference sequence (Mode C)::

        run_targeted_pipeline(
            input_dir="extracted_data/PRJNA1240319",
            output_dir="results",
            protospacer="GGTGGATCCTATTCTAAACG",
            reference_seq="ACGTACGT...",
        )

    Skip samplesheet generation, use an existing file::

        run_targeted_pipeline(
            input_dir="extracted_data/PRJNA1240319",
            output_dir="results",
            protospacer="",
            samplesheet="extracted_data/PRJNA1240319/samplesheet_final.csv",
            prepare_samplesheet=False,
        )
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    # Load .env so OPENAI_API_KEY etc. are available
    _load_dotenv(str(input_path), str(input_path.parent))

    if (extract_metadata or download_fastq) and not project_id:
        red("--project-id is required when extract_metadata or download_fastq is enabled")
        sys.exit(1)

    # --- banner + environment ---
    red("\n[ STARTED PIPELINE ]\n")
    green("SETTING UP ENVIRONMENT")

    nxf_work, singularity_cache, tmp_dir, results_dir = _setup_environment(
        str(output_path), work_dir=work_dir,
    )

    cyan("\nLOCATION SUMMARY")
    print(f"  INPUT DIR:          {input_path}")
    print(f"  OUTPUT DIR:         {output_path}")
    print(f"  RESULTS:            {results_dir}")
    print(f"  NXF WORK:           {nxf_work}")
    print(f"  SINGULARITY CACHE:  {singularity_cache}")
    print(f"  TMP DIR:            {tmp_dir}")
    if hg38:
        print(f"  HG38 FASTA:       {hg38}")
    if gtf:
        print(f"  GTF ANNOTATION:   {gtf}")
    print()

    # --- optional pre-steps ---
    if extract_metadata:
        _run_extractor(str(input_path), project_id)

    if download_fastq:
        _run_downloader(str(input_path))

    # --- samplesheet ---
    if prepare_samplesheet:
        from .prepare_samplesheet3 import generate_samplesheet

        _, csv_path = generate_samplesheet(
            project_dir=str(input_path),
            protospacer=protospacer,
            hg38=hg38,
            gtf=gtf,
            target_gene=target_gene,
            region=region,
            reference_seq=reference_seq,
            template_seq=template_seq,
            gene_flank=gene_flank,
            output_dir=str(output_path),
            openai_model=openai_model,
            no_llm=no_llm,
        )
        if not samplesheet:
            samplesheet = csv_path

    # --- nf-core/crisprseq ---
    if run_nextflow:
        sheet = samplesheet
        if not sheet:
            sheet = str(output_path / "samplesheet_final.csv")

        if not sheet.endswith(".csv"):
            red("nf-core/crisprseq REQUIRES a .csv samplesheet")
            sys.exit(1)

        _run_nextflow(sheet, results_dir, output_path, profile, revision)

    # --- master report ---
    if generate_report:
        cyan("\nGENERATING MASTER REPORT")
        try:
            from .make_crispr_master_report_final import generate_report as _gen_report

            report_path = _gen_report(
                results_dir=str(results_dir),
                output_dir=str(output_path),
            )
            green(f"REPORT WRITTEN: {report_path}")
        except ImportError as exc:
            yellow(
                f"WARNING: Could not generate report ({exc}).\n"
                "  Install pandas to enable: pip install pandas"
            )
        except Exception as exc:
            yellow(f"WARNING: Report generation failed: {exc}")

    green("\n[ PIPELINE COMPLETED SUCCESSFULLY ]")
    green(f"[ CHECK RESULTS DIRECTORY: {results_dir} ]")
    green("[ IF YOU WANT TO RUN AGAIN, FEEL FREE TO RESTART PIPELINE ]\n")
