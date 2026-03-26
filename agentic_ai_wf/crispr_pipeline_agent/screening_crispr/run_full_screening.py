#!/usr/bin/env python3
"""
CRISPR screening pipeline runner.

Provides ``run_screening()`` — a callable entry-point that executes one or more
nf-core/crisprseq screening modes without interactive prompts.

Can also be executed directly for the original interactive CLI::

    python -m screening_crispr.run_full_screening
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from agentic_ai_wf.crispr_pipeline_agent.screening_crispr.generate_report import generate_report as _gen_report

# =========================================================
# Pipeline defaults
# =========================================================
NXF_VER = "24.10.6"
PIPELINE = "nf-core/crisprseq"
REVISION = "2.3.0"
PROFILE = "singularity"

# =========================================================
# Terminal colours
# =========================================================
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

# =========================================================
# Input file specifications
# =========================================================
@dataclass(frozen=True)
class InputFileSpec:
    """Describes a required input file's location and expected format."""
    relative_path: str
    description: str
    format_hint: str


INPUT_FILE_SPECS: Dict[str, InputFileSpec] = {
    "count_table": InputFileSpec(
        "counts/count_table.tsv",
        "sgRNA count matrix",
        "Tab-separated with columns: sgRNA, Gene, <sample_1>, <sample_2>, … "
        "(integer read counts per sgRNA per sample).",
    ),
    "rra_contrasts": InputFileSpec(
        "counts/rra_contrasts.txt",
        "RRA contrast definitions",
        "Semicolon-separated file, one contrast per line: <treatment>;<control>.",
    ),
    "design_matrix": InputFileSpec(
        "counts/design_matrix.txt",
        "MLE design matrix",
        "Tab-separated with columns: Samples, baseline, <condition>. "
        "Rows are samples with 0/1 indicator values.",
    ),
    "samplesheet": InputFileSpec(
        "metadata/samplesheet_test.csv",
        "FASTQ sample sheet",
        "CSV with columns: sample, fastq_1, fastq_2, condition.",
    ),
    "library": InputFileSpec(
        "metadata/brunello_library.tsv",
        "sgRNA guide library",
        "Tab-separated with columns: id, sequence, gene.",
    ),
    "essential_genes": InputFileSpec(
        "metadata/essential_genes.txt",
        "Reference essential gene list (for BAGEL2)",
        "Plain-text file, one gene symbol per line.",
    ),
    "nonessential_genes": InputFileSpec(
        "metadata/nonessential_genes.txt",
        "Reference non-essential gene list (for BAGEL2)",
        "Plain-text file, one gene symbol per line.",
    ),
    "samplesheet_full": InputFileSpec(
        "full_test/samplesheet_full.csv",
        "Full FASTQ sample sheet (drug comparison)",
        "CSV with columns: sample, fastq_1, fastq_2, condition.",
    ),
    "drug_design_matrix": InputFileSpec(
        "full_test/drugA_drugB_vs_treatment.txt",
        "Drug-comparison design matrix",
        "Tab-separated with columns: Samples, baseline, <drug>. "
        "Rows are samples with 0/1 indicator values.",
    ),
}

# =========================================================
# Mode definitions
# =========================================================
def _build_cmd_1(inp: Path, out: Path) -> List[str]:
    return [
        "--count_table",  str(inp / "counts/count_table.tsv"),
        "--contrasts",    str(inp / "counts/rra_contrasts.txt"),
        "--rra",
        "--outdir",       str(out / "mode1_counts_rra"),
    ]

def _build_cmd_2(inp: Path, out: Path) -> List[str]:
    return [
        "--count_table",       str(inp / "counts/count_table.tsv"),
        "--mle_design_matrix", str(inp / "counts/design_matrix.txt"),
        "--outdir",            str(out / "mode2_counts_mle"),
    ]

def _build_cmd_3(inp: Path, out: Path) -> List[str]:
    return [
        "--count_table", str(inp / "counts/count_table.tsv"),
        "--contrasts",   str(inp / "counts/rra_contrasts.txt"),
        "--mle", "--rra",
        "--outdir",      str(out / "mode3_counts_full_mageck"),
    ]

def _build_cmd_4(inp: Path, out: Path) -> List[str]:
    return [
        "--input",   str(inp / "metadata/samplesheet_test.csv"),
        "--library", str(inp / "metadata/brunello_library.tsv"),
        "--outdir",  str(out / "mode4_fastq_to_counts"),
    ]

def _build_cmd_5(inp: Path, out: Path) -> List[str]:
    return [
        "--input",             str(inp / "full_test/samplesheet_full.csv"),
        "--library",           str(inp / "metadata/brunello_library.tsv"),
        "--mle_design_matrix", str(inp / "full_test/drugA_drugB_vs_treatment.txt"),
        "--outdir",            str(out / "mode5_fastq_drugA_vs_drugB"),
    ]

def _build_cmd_6(inp: Path, out: Path) -> List[str]:
    return [
        "--count_table", str(inp / "counts/count_table.tsv"),
        "--contrasts",   str(inp / "counts/rra_contrasts.txt"),
        "--mle", "--rra", "--bagel2",
        "--bagel_reference_essentials",    str(inp / "metadata/essential_genes.txt"),
        "--bagel_reference_nonessentials", str(inp / "metadata/nonessential_genes.txt"),
        "--hitselection",
        "--outdir", str(out / "mode6_full_screen"),
    ]


@dataclass(frozen=True)
class _ModeSpec:
    name: str
    description: str
    required_files: List[str]
    build_cmd: object  # callable(Path, Path) -> List[str]


MODES: Dict[int, _ModeSpec] = {
    1: _ModeSpec(
        "Counts → RRA",
        "Gene ranking using MAGeCK RRA (fast sanity check)",
        ["count_table", "rra_contrasts"],
        _build_cmd_1,
    ),
    2: _ModeSpec(
        "Counts → MLE",
        "Model-based gene effects using MAGeCK MLE",
        ["count_table", "design_matrix"],
        _build_cmd_2,
    ),
    3: _ModeSpec(
        "Counts → RRA + MLE",
        "Cross-validated MAGeCK analysis (recommended core)",
        ["count_table", "rra_contrasts"],
        _build_cmd_3,
    ),
    4: _ModeSpec(
        "FASTQs → Counts",
        "Recompute sgRNA counts from raw FASTQ files",
        ["samplesheet", "library"],
        _build_cmd_4,
    ),
    5: _ModeSpec(
        "FASTQs → Drug Comparison (MLE)",
        "Chemogenetic screen using custom design matrix",
        ["samplesheet_full", "library", "drug_design_matrix"],
        _build_cmd_5,
    ),
    6: _ModeSpec(
        "Full Screening (RRA + MLE + BAGEL2 + HitSelection)",
        "DepMap-style gene essentiality and hit prioritization",
        ["count_table", "rra_contrasts", "essential_genes", "nonessential_genes"],
        _build_cmd_6,
    ),
}

# =========================================================
# Result types
# =========================================================
@dataclass
class ModeResult:
    """Outcome of a single mode execution."""
    mode: int
    name: str
    success: bool
    return_code: int
    command: str


@dataclass
class ScreeningResult:
    """Aggregate outcome returned by :func:`run_screening`."""
    success: bool
    message: str
    mode_results: List[ModeResult] = field(default_factory=list)
    report_path: Optional[Path] = None


# =========================================================
# Validation
# =========================================================
def _validate_count_table(path: Path) -> Optional[str]:
    """Light-weight header check for the count table."""
    try:
        with open(path) as fh:
            header = fh.readline().strip().split("\t")
        if len(header) < 3:
            return (
                f"  Count table has only {len(header)} column(s) — expected at least 3 "
                "(sgRNA, Gene, and one or more sample columns)."
            )
        expected_first_two = {"sgRNA", "sgrna", "sgRNA".lower()}
        expected_second = {"Gene", "gene"}
        if header[0].lower() not in expected_first_two:
            return (
                f"  Count table first column is '{header[0]}' — expected 'sgRNA'."
            )
        if header[1].lower() not in {g.lower() for g in expected_second}:
            return (
                f"  Count table second column is '{header[1]}' — expected 'Gene'."
            )
    except Exception as exc:
        return f"  Could not read count table: {exc}"
    return None


def validate_inputs(
    input_dir: Path,
    modes: Sequence[int],
) -> Optional[str]:
    """Return a human-readable error string if validation fails, else ``None``.

    Checks that *input_dir* exists, all requested modes are valid, and every
    file required by those modes is present and non-empty.  Also performs a
    lightweight header check on the count table when relevant.
    """
    if not input_dir.exists():
        return (
            f"Input directory does not exist: {input_dir}\n\n"
            "Please provide a directory containing the screening input data.\n"
            + _expected_layout_hint()
        )

    if not input_dir.is_dir():
        return f"Input path is not a directory: {input_dir}"

    invalid = [m for m in modes if m not in MODES]
    if invalid:
        available = ", ".join(str(k) for k in sorted(MODES.keys()))
        return f"Invalid mode(s): {invalid}. Available modes: {available}"

    errors: List[str] = []

    for mode_num in modes:
        mode = MODES[mode_num]
        missing: List[InputFileSpec] = []
        empty: List[InputFileSpec] = []

        for file_key in mode.required_files:
            spec = INPUT_FILE_SPECS[file_key]
            full_path = input_dir / spec.relative_path
            if not full_path.exists():
                missing.append(spec)
            elif full_path.stat().st_size == 0:
                empty.append(spec)

        if missing or empty:
            lines = [f"\n  Mode {mode_num} — {mode.name}:"]
            for spec in missing:
                lines.append(f"    MISSING : {spec.relative_path}")
                lines.append(f"              {spec.description}")
                lines.append(f"              Expected: {spec.format_hint}")
            for spec in empty:
                lines.append(f"    EMPTY   : {spec.relative_path}")
                lines.append(f"              {spec.description}")
                lines.append(f"              Expected: {spec.format_hint}")
            errors.append("\n".join(lines))

    if errors:
        header = f"Input validation failed for: {input_dir}\n"
        return header + "\n".join(errors) + "\n" + _expected_layout_hint()

    # Deeper content check for count table (only if it was required)
    needs_count_table = any(
        "count_table" in MODES[m].required_files for m in modes
    )
    if needs_count_table:
        ct_err = _validate_count_table(input_dir / "counts/count_table.tsv")
        if ct_err is not None:
            return f"Count table format error:\n{ct_err}"

    return None


def _expected_layout_hint() -> str:
    """Return a formatted string showing the expected input directory tree."""
    lines = [
        "\nExpected input directory layout:",
        "  <input_dir>/",
        "  ├── counts/",
        "  │   ├── count_table.tsv          (sgRNA count matrix)",
        "  │   ├── rra_contrasts.txt        (contrast definitions for RRA)",
        "  │   └── design_matrix.txt        (design matrix for MLE)",
        "  ├── metadata/",
        "  │   ├── samplesheet_test.csv     (sample sheet for FASTQs)",
        "  │   ├── brunello_library.tsv     (sgRNA guide library)",
        "  │   ├── essential_genes.txt      (essential gene list for BAGEL2)",
        "  │   └── nonessential_genes.txt   (non-essential gene list for BAGEL2)",
        "  └── full_test/",
        "      ├── samplesheet_full.csv     (full sample sheet for drug comparison)",
        "      └── drugA_drugB_vs_treatment.txt (drug comparison design matrix)",
        "",
        "Not all files are needed for every mode. Only the files listed under a",
        "given mode's requirements must be present to run that mode.",
    ]
    return "\n".join(lines)


# =========================================================
# Core runner
# =========================================================
def run_screening(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    modes: Sequence[int] = (3,),
    *,
    generate_report: bool = True,
    report_filename: str = "CRISPR_Based_Genetic_Screening_Report.html",
    nxf_ver: str = NXF_VER,
    pipeline: str = PIPELINE,
    revision: str = REVISION,
    profile: str = PROFILE,
) -> ScreeningResult:
    """Run one or more nf-core/crisprseq screening modes.

    Parameters
    ----------
    input_dir:
        Directory containing input data (replaces the ``data/`` prefix).
    output_dir:
        Directory where pipeline results will be written.
    modes:
        Mode numbers to execute (1–6).  Default ``(3,)``.
    generate_report:
        If ``True`` (default), automatically generate an HTML report in
        *output_dir* after at least one mode succeeds.
    report_filename:
        Name of the HTML report file written inside *output_dir*.
    nxf_ver / pipeline / revision / profile:
        Override Nextflow / pipeline settings when needed.

    Returns
    -------
    ScreeningResult
        ``.success`` is ``True`` only when every requested mode succeeds.
        ``.message`` contains a human-readable summary.
        ``.mode_results`` holds per-mode details.
        ``.report_path`` is the path to the generated HTML report (if any).
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    error = validate_inputs(input_dir, modes)
    if error is not None:
        return ScreeningResult(success=False, message=error)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "nextflow", "-log", str(output_dir / ".nextflow.log"),
        "run", pipeline,
        "-r", revision,
        "-profile", profile,
        "-work-dir", str(output_dir / "work"),
        "--analysis", "screening",
    ]

    env = os.environ.copy()
    env["NXF_VER"] = nxf_ver

    results: List[ModeResult] = []

    for mode_num in modes:
        mode = MODES[mode_num]
        extra = mode.build_cmd(input_dir, output_dir)
        cmd = base_cmd + extra
        cmd_str = " ".join(cmd)

        print(f"\n{YELLOW}{BOLD}▶ Running Mode {mode_num}: {mode.name}{RESET}")
        print(f"{CYAN}Command:{RESET} {cmd_str}\n")

        proc = subprocess.run(cmd, env=env)
        ok = proc.returncode == 0

        if ok:
            print(f"{GREEN}✔ Mode {mode_num} completed successfully.{RESET}\n")
        else:
            print(
                f"{RED}✖ Mode {mode_num} failed "
                f"(exit code {proc.returncode}).{RESET}\n"
            )

        results.append(ModeResult(
            mode=mode_num,
            name=mode.name,
            success=ok,
            return_code=proc.returncode,
            command=cmd_str,
        ))

    all_ok = all(r.success for r in results)
    any_ok = any(r.success for r in results)

    summary_lines = []
    for r in results:
        mark = f"{GREEN}✔{RESET}" if r.success else f"{RED}✖{RESET}"
        status = "succeeded" if r.success else f"failed (exit {r.return_code})"
        summary_lines.append(f"  {mark} Mode {r.mode} ({r.name}): {status}")

    # ── Report generation ──────────────────────────────────
    report_path: Optional[Path] = None

    if generate_report and any_ok:

        print(f"\n{CYAN}{BOLD}▶ Generating HTML report …{RESET}")
        report_out = output_dir / report_filename
        rpt = _gen_report(results_dir=output_dir, output_path=report_out)
        if rpt.success:
            report_path = rpt.report_path
            summary_lines.append(
                f"  {GREEN}✔{RESET} Report: {report_path}"
            )
        else:
            summary_lines.append(
                f"  {RED}✖{RESET} Report generation failed: {rpt.message}"
            )
    elif generate_report and not any_ok:
        summary_lines.append(
            f"  {YELLOW}–{RESET} Report skipped (no successful modes)"
        )

    message = "Screening run summary:\n" + "\n".join(summary_lines)
    return ScreeningResult(
        success=all_ok,
        message=message,
        mode_results=results,
        report_path=report_path,
    )


# =========================================================
# Interactive CLI  (preserved for standalone / legacy use)
# =========================================================
def _box(title: str, lines: List[str], color: str = GREEN) -> None:
    width = max(len(title), *(len(ln) for ln in lines)) + 4
    print(color + "┌" + "─" * (width - 2) + "┐" + RESET)
    print(color + f"│ {title.ljust(width - 4)} │" + RESET)
    print(color + "├" + "─" * (width - 2) + "┤" + RESET)
    for ln in lines:
        print(color + f"│ {ln.ljust(width - 4)} │" + RESET)
    print(color + "└" + "─" * (width - 2) + "┘" + RESET)


def _print_modes() -> None:
    print(f"\n{CYAN}{BOLD}Available CRISPRseq Screening Modes:{RESET}\n")
    for k, mode in MODES.items():
        required = [INPUT_FILE_SPECS[f].relative_path for f in mode.required_files]
        _box(
            f"[ MODE {k} ] {mode.name}",
            [mode.description, "Requires:", *[f"- {r}" for r in required]],
            GREEN,
        )


def _recommend_modes(input_dir: Path) -> None:
    available = {}
    for k, mode in MODES.items():
        available[k] = all(
            (input_dir / INPUT_FILE_SPECS[f].relative_path).exists()
            for f in mode.required_files
        )

    print(f"\n{CYAN}{BOLD}Data availability check & recommendations:{RESET}\n")
    for k, ok in available.items():
        status = f"{GREEN}READY{RESET}" if ok else f"{YELLOW}MISSING FILES{RESET}"
        print(f"  Mode {k}: {MODES[k].name} → {status}")

    print(f"\n{BOLD}Recommended next steps:{RESET}")
    if available.get(6):
        print("  ✔ You can run MODE 6 for full DepMap-style screening")
    elif available.get(3):
        print("  ✔ MODE 3 is recommended as core analysis")
        print("  ➕ Add essential / non-essential gene lists to enable MODE 6")
    elif available.get(1):
        print("  ✔ MODE 1 can be used for quick sanity check")
    else:
        print("  ⚠ No complete mode available — check missing files")


def _interactive_cli() -> None:
    """Original interactive menu for standalone execution."""
    input_dir = Path("data")
    output_dir = Path("results")

    _print_modes()
    _recommend_modes(input_dir)

    choice = input(
        f"\n{CYAN}Select mode numbers (e.g. 1,3,6), "
        f"'all' to run everything, or 'q' to quit:{RESET} "
    ).strip().lower()

    if choice == "q":
        sys.exit(0)

    if choice == "all":
        selected = list(MODES.keys())
    else:
        selected = [
            int(c.strip()) for c in choice.split(",")
            if c.strip().isdigit() and int(c.strip()) in MODES
        ]

    if not selected:
        print(f"{RED}No valid modes selected. Exiting.{RESET}")
        sys.exit(1)

    result = run_screening(input_dir, output_dir, modes=selected)
    print(f"\n{result.message}")

    if not result.success:
        sys.exit(1)


# =========================================================
if __name__ == "__main__":
    _interactive_cli()
