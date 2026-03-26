# mdp_pipeline_3/counts_mdp/mdp_config.py
from __future__ import annotations

from pathlib import Path
import os

# --- keep BLAS from oversubscribing (speed + stability) ---
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- ensure rpy2 sees user R libs (prefer system-provided env) ---
# We do NOT hardcode a Linux-only path; we only set a fallback if nothing is provided.
def _maybe_set_r_env() -> None:
    # Honor user/system-provided R_HOME and R_LIBS_USER as-is.
    if os.environ.get("R_HOME"):
        os.environ["R_HOME"] = os.environ["R_HOME"]
    if os.environ.get("R_LIBS_USER"):
        return

    # If not provided, try a few common locations; set only when they exist.
    guesses = [
        Path.home() / "R" / "win-library",
        Path.home() / "R" / "win-library" / "4.3",
        Path.home() / "R" / "x86_64-w64-mingw32-library" / "4.3",
        Path.home() / "R" / "x86_64-pc-linux-gnu-library" / "4.3",
    ]
    for g in guesses:
        if g.exists():
            os.environ["R_LIBS_USER"] = str(g)
            break


_maybe_set_r_env()

# -----------------------------------------------------------------------------
# Project-relative paths
# -----------------------------------------------------------------------------
# This file lives in:  mdp_pipeline_3/counts_mdp/mdp_config.py
# So:
#   HERE          = .../mdp_pipeline_3/counts_mdp
#   PROJECT_ROOT  = .../mdp_pipeline_3
#   DATA_DIR      = .../mdp_pipeline_3/data
#   OUT_ROOT      = .../mdp_pipeline_3/OUT_ROOT   (default; can override)
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Default locations (can be overridden by environment variables)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "OUT_ROOT"

DATA_DIR = Path(os.getenv("MDP_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
OUT_ROOT = Path(os.getenv("MDP_OUT_ROOT", str(DEFAULT_OUT_ROOT))).resolve()

CONFIG = {
    # Baseline inputs (paths to HPA/GTEx/FANTOM; wide or long accepted)
    # Now project-relative by default: mdp_pipeline_3/data/{hpa,gtex,fantom}.tsv
    "DATA_DIR": str(DATA_DIR),
    "HPA_FILE": "hpa.tsv",
    "GTEX_FILE": "gtex.tsv",
    "FANTOM_FILE": "fantom.tsv",

    # Where outputs land (project-relative by default; override via MDP_OUT_ROOT)
    "OUT_ROOT": str(OUT_ROOT),

    # Cohorts
    "COHORTS": [
        {
            "name": "Rehumatoid_Arthritis",
            "counts_dir": "/mnt/c/Users/ssaba/Documents/AyassBiosciences_3/A58_new_disease_comparison/RA/",
            "id_col": "Gene",
            "lfc_col": "log2FoldChange",
            "q_col": "",
            "q_max": 0.05,
            "tissue": "lymph node",
        },
        # add more cohorts here as needed
    ],

    # Pathway libs (Enrichr / GSEA names), now including GO BP/CC/MF
    "CORE_PATHWAY_LIBS": [
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
        "GO_Cellular_Component_2023",
        "GO_Molecular_Function_2023",
    ],

    # Keep TF and epigenetic libraries separate
    "TF_PATTERNS": [r"ENCODE_TF_ChIP|ChEA|JASPAR|TRANSFAC|ChIP-X|DoRothEA|TF"],
    "EPIGENETIC_PATTERNS": [r"Histone|H3K|Chromatin|Enhancer|Epigenetic"],

    # Immune + Metabolic
    "IMMUNE_PATTERNS": [r"PanglaoDB", r"Azimuth", r"Immunologic|Immune|LM22|CIBERSORT"],
    "METABOLIC_LIBS": ["HMDB_Metabolites"],

    # VIPER / ULM resources
    "VIPER_LICENSE": "academic",
    "VIPER_TMIN": 5,
    "VIPER_PLEIOTROPY": False,
    "REGULON_CSV": "",  # optional external regulon CSV; leave "" to use decoupler resources

    # decoupler ULM panels
    "RUN_ULM_TF_COLLECTRI": True,
    "RUN_ULM_PROGENY": True,
    "RUN_ULM_HALLMARK": True,

    # GSEApy tuning (WSL safe)
    "GSEA_PROCESSES": 1,
    "GSEA_PERMUTATIONS": 100,

    # Plotting
    "FIG_DPI": 150,
    "TOP_N_PATHWAYS": 20,
    "TOP_N_IMMUNO_EPI": 20,
    "TF_HEATMAP_TOP_N": 40,

    # FDR threshold used in multi-cohort union
    "FDR_Q": 0.05,

    # Combined output
    "NORMALIZE_TERMS_DROP_PREFIX": True,
    "PRESENCE_REQUIRE_SIGNIFICANT": False,

    # Human-only ORA
    "HUMAN_ONLY_ORA": True,

    # Explicit locations for baseline stack (utils_io.py + baseline_expectations.py)
    "MDP2_DIR_CANDIDATES": [
        "/home/sabahatjamil/Wokspace/Manhuas/Multi_Disease_Pathway/MDP_2",
        str(PROJECT_ROOT / "MDP_2"),
        str(PROJECT_ROOT),
    ],

    # Replicate handling to avoid bias
    "COLLAPSE_TECH_REPS": True,       # collapse columns mapping to same base sample id
    "TECH_REP_METHOD": "sum",         # "sum" (recommended for counts) or "mean"
    "DROP_IDENTICAL_COLUMNS": True,   # drop bit-for-bit identical columns after collapse
}

# --- derived settings used by other modules ---

LICENSE_MODE = (CONFIG.get("VIPER_LICENSE", "academic") or "academic").strip().lower()


def out_root() -> Path:
    """Return OUT_ROOT as a resolved Path object."""
    return OUT_ROOT
