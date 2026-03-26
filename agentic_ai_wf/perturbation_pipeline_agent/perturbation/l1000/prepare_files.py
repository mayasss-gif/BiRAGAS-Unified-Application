# src/prepare_files.py

from pathlib import Path
import pandas as pd
from logging import Logger


# Final columns we want in prepared DEGs
DEG_OUTPUT_COLS = [
    "Gene",
    "Log2FC",
    "HGNC_Synonyms",
    "Patient_LFC_Trend",
]

# Final columns we want in prepared Pathways
PATHWAY_OUTPUT_COLS = [
    "Pathway ID",
    "number_of_genes",
    "number_of_genes_in_background",
    "Pathway associated genes",
    "p_value",
    "fdr",
    "Pathway",
    "Regulation",
    "Main_Class",
    "Sub_Class",
]


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    elif suf == ".csv":
        return pd.read_csv(path)
    elif suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    else:
        return pd.read_csv(path)


def _infer_log2fc(df: pd.DataFrame, logger):
    cols = df.columns

    if "Log2FC" in cols:
        logger.info("Using existing 'Log2FC'.")
        return df["Log2FC"]

    if "Patient_LFC_mean" in cols:
        logger.info("Creating 'Log2FC' from 'Patient_LFC_mean'.")
        return df["Patient_LFC_mean"]

    candidates = [c for c in cols if c.lower().endswith("log2fc")]
    if candidates:
        chosen = candidates[0]
        logger.info(f"Creating 'Log2FC' from '{chosen}'.")
        return df[chosen]

    logger.error("Cannot infer Log2FC column.")
    raise ValueError("Log2FC cannot be inferred from DEGs.")


def _check_required(df, cols, label, logger):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"{label} missing required cols: {missing}")
        raise ValueError(f"{label} missing required columns: {missing}")


def _build_pathway_mapped_df(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Take the raw pathways DataFrame and return a new DataFrame with
    columns renamed to the canonical PATHWAY_OUTPUT_COLS.
    """
    colmap_candidates = {
        "Pathway ID": ["Pathway ID", "Pathway_ID"],
        "number_of_genes": ["number_of_genes", "Number_of_Genes"],
        "number_of_genes_in_background": [
            "number_of_genes_in_background",
            "Number_of_Genes_in_Background",
        ],
        "Pathway associated genes": [
            "Pathway associated genes",
            "Pathway_Associated_Genes",
        ],
        "p_value": ["p_value", "P_Value"],
        "fdr": ["fdr", "FDR"],
        "Pathway": ["Pathway", "Pathway_Name"],
        "Regulation": ["Regulation"],
        "Main_Class": ["Main_Class"],
        "Sub_Class": ["Sub_Class"],
    }

    out = {}
    cols = set(df.columns)

    for target, options in colmap_candidates.items():
        found = None
        for opt in options:
            if opt in cols:
                found = opt
                break
        if found is None:
            logger.error(f"Pathways: missing column for target '{target}'. Tried: {options}")
            raise ValueError(
                f"Pathways: missing column for target '{target}'. "
                f"Available columns: {list(df.columns)}"
            )
        out[target] = df[found]

    mapped_df = pd.DataFrame({col: out[col] for col in PATHWAY_OUTPUT_COLS})
    return mapped_df


def run_prepare_files(output_dir: Path, deg_src: Path, pathway_src: Path, logger: Logger):
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR = output_dir / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


    logger.info("=== STEP 1b: preparing DEGs & Pathways ===")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- DEGs ----------------
    logger.info(f"Reading DEGs: {deg_src}")
    degs = _read_any(deg_src)
    logger.info(f"DEGs shape={degs.shape}")

    _check_required(degs, ["Gene", "HGNC_Synonyms", "Patient_LFC_Trend"], "DEGs", logger)

    degs = degs.copy()
    degs["Log2FC"] = _infer_log2fc(degs, logger)

    _check_required(degs, DEG_OUTPUT_COLS, "DEGs", logger)
    out_deg = TABLE_DIR / "degs_prepared.csv"
    degs[DEG_OUTPUT_COLS].to_csv(out_deg, index=False)
    logger.info(f"Saved prepared DEGs → {out_deg}")

    # ---------------- Pathways ----------------
    logger.info(f"Reading Pathways: {pathway_src}")
    pathways_raw = _read_any(pathway_src)
    logger.info(f"Pathways shape={pathways_raw.shape}")

    pathways_prepared = _build_pathway_mapped_df(pathways_raw, logger)
    out_path = TABLE_DIR / "pathways_prepared.csv"
    pathways_prepared.to_csv(out_path, index=False)
    logger.info(f"Saved prepared Pathways → {out_path}")

    logger.info("=== STEP 1b completed ===")

    return out_deg, out_path
