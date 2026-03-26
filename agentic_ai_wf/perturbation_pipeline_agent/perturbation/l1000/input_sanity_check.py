# src/input_sanity_check.py

import json
from pathlib import Path
import pandas as pd
from logging import Logger
from typing import Tuple


from .constants import CORE_SIG_PATH, CORE_PERT_PATH, CORE_CELL_PATH, CORE_GENE_PATH, CORE_GCTX_PATH



def _read_any(path: Path) -> pd.DataFrame:
    """
    Read Excel or delimited text based on extension.
    """
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    elif suf == ".csv":
        return pd.read_csv(path)
    elif suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    else:
        return pd.read_csv(path)


def run_input_sanity_check(output_dir: Path, deg_src: Path, path_src: Path, logger: Logger):

    logger.info("=== STEP 1: L1000 input sanity check ===")

    # ---------- 1) Resolve curated DEGs / Pathways (hard requirement) ----------
    logger.info(f"Resolved DEGs input file: {deg_src}")
    logger.info(f"Resolved Pathways input file: {path_src}")

    hard_required = [deg_src, path_src]
    missing_hard = [str(p) for p in hard_required if not p.exists()]
    if missing_hard:
        logger.error("Missing required curated input files:\n" + "\n".join(missing_hard))
        raise FileNotFoundError(
            "Missing required curated input files:\n" + "\n".join(missing_hard)
        )

    # ---------- 2) Check L1000 core files (soft requirement) ----------
    core_paths = [
        CORE_SIG_PATH,
        CORE_PERT_PATH,
        CORE_CELL_PATH,
        CORE_GENE_PATH,
    ]
    missing_core = [str(p) for p in core_paths if not p.exists()]
    gctx_present = CORE_GCTX_PATH.exists()

    if missing_core or not gctx_present:
        logger.warning(
            "Some or all L1000 core files are missing.\n"
            "Missing core files:\n" + "\n".join(missing_core or ["<none>"]) + "\n"
            f"GCTX present: {gctx_present}\n"
            "Will skip loading L1000 core metadata. "
            "You can still proceed with DEG/Pathway-only analysis."
        )
        sig = pert = cell = gene = None
        gctx_path = None
    else:
        # all core present: ensure gctx and load
        gctx_path = CORE_GCTX_PATH

        sig = pd.read_csv(CORE_SIG_PATH, sep="\t", low_memory=False)
        pert = pd.read_csv(CORE_PERT_PATH, sep="\t", low_memory=False)
        cell = pd.read_csv(CORE_CELL_PATH, sep="\t", low_memory=False)
        gene = pd.read_csv(CORE_GENE_PATH, sep="\t", low_memory=False)

        logger.info(
            f"Loaded L1000 tables | sig={sig.shape}, pert={pert.shape}, "
            f"cell={cell.shape}, gene={gene.shape}"
        )

    # ---------- 3) Load curated DEGs + Pathways (always) ----------
    degs = _read_any(deg_src)
    pathways = _read_any(path_src)
    logger.info(f"Loaded DEGs file: {deg_src.name} shape={degs.shape}")
    logger.info(f"Loaded Pathways file: {path_src.name} shape={pathways.shape}")

    # ---------- 4) Write summary JSON ----------
    summary = {
        "deg_input_file": str(deg_src),
        "pathway_input_file": str(path_src),
        "degs_shape": [int(degs.shape[0]), int(degs.shape[1])],
        "pathways_shape": [int(pathways.shape[0]), int(pathways.shape[1])],
        "l1000_core_available": sig is not None,
    }

    if sig is not None:
        summary.update(
            {
                "gctx": str(gctx_path) if gctx_path is not None else None,
                "sig_rows": int(sig.shape[0]),
                "pert_rows": int(pert.shape[0]),
                "cell_rows": int(cell.shape[0]),
                "gene_rows": int(gene.shape[0]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "input_sanity_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved sanity summary → {out_json}")
    logger.info("=== STEP 1 finished successfully ===")

