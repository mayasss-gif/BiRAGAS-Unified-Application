"""Default paths for data files and the bundled R script.

The R script lives inside the ``gwas_mr`` package.  Large reference
data files (GWAS catalog TSV, GTEx eQTLs, PLINK panel) live in the
``gwas_mr_reference/`` directory at the repository root.
"""

from pathlib import Path

# gwas_mr package directory (contains mr_pipeline.R)
PKG_DIR: Path = Path(__file__).resolve().parent

# Repository root (parent of the gwas_mr package)
REPO_ROOT: Path = PKG_DIR.parent

# Large reference data directory at the repo root
REF_DATA_DIR: Path = Path("gwas_mr_reference")

DEFAULT_R_SCRIPT: str = str(PKG_DIR / "mr_pipeline.R")

DEFAULT_TSV_PATH: str = str(REF_DATA_DIR / "GWAS-DATABASE-FTP.tsv")

DEFAULT_EQTL_ROOT: str = str(REF_DATA_DIR / "GTEx_eQTL_TISSUE_EXPRESSION")

DEFAULT_GTEX_BIOSAMPLE_CSV: str = str(
    REF_DATA_DIR / "GTEx_EQTL_SAMPLE_COUNTS" / "EQTL-SAMPLE-COUNT-GTEx Portal.csv"
)

DEFAULT_PLINK_DIR: Path = REF_DATA_DIR / "plink"
