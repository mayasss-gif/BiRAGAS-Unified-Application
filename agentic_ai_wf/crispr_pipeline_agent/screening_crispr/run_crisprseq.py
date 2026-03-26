#!/usr/bin/env python3
import subprocess
import sys
import os

# ---------- CONFIG ----------
NXF_VER = "24.10.6"
PIPELINE = "nf-core/crisprseq"
REVISION = "2.3.0"
PROFILE = "singularity"

BASE_CMD = [
    "nextflow", "run", PIPELINE,
    "-r", REVISION,
    "-profile", PROFILE,
    "--analysis", "screening"
]

# ---------- COLORS ----------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ---------- MODES ----------
MODES = {
    "1": {
        "name": "Counts → RRA",
        "desc": "Gene ranking using MAGeCK RRA (fast sanity check)",
        "requires": [
            "data/counts/count_table.tsv",
            "data/counts/rra_contrasts.txt"
        ],
        "cmd": [
            "--count_table", "data/counts/count_table.tsv",
            "--contrasts", "data/counts/rra_contrasts.txt",
            "--rra",
            "--outdir", "results/mode1_counts_rra"
        ]
    },
    "2": {
        "name": "Counts → MLE (design matrix)",
        "desc": "Model-based gene effects using MAGeCK MLE",
        "requires": [
            "data/counts/count_table.tsv",
            "data/counts/design_matrix.txt"
        ],
        "cmd": [
            "--count_table", "data/counts/count_table.tsv",
            "--mle_design_matrix", "data/counts/design_matrix.txt",
            "--outdir", "results/mode2_counts_mle"
        ]
    },
    "3": {
        "name": "Counts → RRA + MLE (recommended)",
        "desc": "Full MAGeCK analysis with cross-validation",
        "requires": [
            "data/counts/count_table.tsv",
            "data/counts/rra_contrasts.txt"
        ],
        "cmd": [
            "--count_table", "data/counts/count_table.tsv",
            "--contrasts", "data/counts/rra_contrasts.txt",
            "--mle",
            "--rra",
            "--outdir", "results/mode3_counts_full_mageck"
        ]
    },
    "4": {
        "name": "FASTQs → Counts",
        "desc": "Recompute sgRNA counts from raw sequencing data",
        "requires": [
            "data/metadata/samplesheet_test.csv",
            "data/metadata/brunello_library.tsv"
        ],
        "cmd": [
            "--input", "data/metadata/samplesheet_test.csv",
            "--library", "data/metadata/brunello_library.tsv",
            "--outdir", "results/mode4_fastq_to_counts"
        ]
    },
    "5": {
        "name": "FASTQs → Drug comparison (MLE)",
        "desc": "Chemogenetic screen using custom design matrix",
        "requires": [
            "data/full_test/samplesheet_full.csv",
            "data/metadata/brunello_library.tsv",
            "data/full_test/drugA_drugB_vs_treatment.txt"
        ],
        "cmd": [
            "--input", "data/full_test/samplesheet_full.csv",
            "--library", "data/metadata/brunello_library.tsv",
            "--mle_design_matrix", "data/full_test/drugA_drugB_vs_treatment.txt",
            "--outdir", "results/mode5_fastq_drugA_vs_drugB"
        ]
    }
}

# ---------- FUNCTIONS ----------
def print_modes():
    print(f"\n{CYAN}Available CRISPRseq SCREENING modes:{RESET}\n")
    for k, v in MODES.items():
        print(f"{GREEN}[{k}] {v['name']}{RESET}")
        print(f"    ↳ {v['desc']}")
        print(f"    ↳ Requires:")
        for r in v["requires"]:
            print(f"       - {r}")
        print()

def run_mode(key):
    mode = MODES[key]
    print(f"{YELLOW}\n▶ Running mode {key}: {mode['name']}{RESET}\n")

    env = os.environ.copy()
    env["NXF_VER"] = NXF_VER

    cmd = BASE_CMD + mode["cmd"]
    print(f"{CYAN}Command:{RESET} {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"{RED}✖ Mode {key} failed.{RESET}")
        retry = input("Try again? [y/n]: ").strip().lower()
        if retry == "y":
            return run_mode(key)
        else:
            print(f"{YELLOW}Skipping mode {key}.{RESET}")
    else:
        print(f"{GREEN}✔ Mode {key} completed successfully.{RESET}")

# ---------- MAIN ----------
def main():
    print_modes()
    choice = input(
        f"{CYAN}Enter mode numbers (e.g. 1,3), 'all' to run everything, or 'q' to quit:{RESET} "
    ).strip().lower()

    if choice == "q":
        sys.exit(0)

    if choice == "all":
        keys = MODES.keys()
    else:
        keys = [c.strip() for c in choice.split(",") if c.strip() in MODES]

    if not keys:
        print(f"{RED}No valid modes selected. Exiting.{RESET}")
        sys.exit(1)

    for k in keys:
        run_mode(k)

    print(f"\n{GREEN}All selected runs finished.{RESET}")

if __name__ == "__main__":
    main()

