# src/check_cell_essentiality.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple, Dict

from pathlib import Path
from logging import Logger, getLogger


from .depmap_celllines_uif import run_depmap_cellline_setup, DepMapCelllineContext
from .model_selection_uif import (
    build_models_table_from_master,
    normalize_list_from_input,
    select_models,
    ModelSelectionResult,
)
from .gene_selection_uif import build_gene_lists_from_degs
from .gene_identity_uif import run_gene_identity_check
from .depmap_gene_layers_uif import build_gene_layer_matrices, GeneLayerResult


# Simple type alias to avoid importing pandas at module import time
try:
    import pandas as pd  # type: ignore
    DataFrameLike = pd.DataFrame
except Exception:  # pragma: no cover
    DataFrameLike = object  # dummy fallback



@dataclass
class CheckCellEssentialityResult:
    """
    Container for all outputs of the Check_CellEssentiality pipeline.
    """
    ctx: DepMapCelllineContext
    model_selection: ModelSelectionResult
    gene_full: "DataFrameLike"          # df_full from gene_selection
    gene_selected: "DataFrameLike"      # df_selected from gene_selection
    gene_identity: "DataFrameLike"      # df from gene_identity_uif
    gene_layers: GeneLayerResult        # matrices + presence summary


def _ensure_logger(logger: Optional[Logger]) -> Logger:
    """
    Use provided logger or fall back to 'DepMap' logger that propagates to root.
    """
    if logger is not None:
        return logger
    return getLogger("DepMap")


# ======================================================================
# CONFIG HELPERS
# ======================================================================

def _load_simple_kv_config(path: Path, logger: Logger) -> Dict[str, str]:
    """
    Load a simple key=value configuration file.

    - Blank lines and lines starting with '#' are ignored.
    - Keys are lower-cased and stripped.
    - Values are stripped but kept as-is (except whitespace).

    Returns
    -------
    dict[str, str]
        Mapping from key → value (strings).
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading config from: %s", path)
    config: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            logger.warning(
                "Skipping malformed line in config (no '='): %s",
                line,
            )
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        config[key] = value

    logger.info("Parsed config %s: %s", path.name, config)
    return config


def _split_csv_value(raw: str) -> List[str]:
    """Split a comma-separated value string into a clean list of tokens."""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


# ======================================================================
# MODEL SELECTION (CONFIG-DRIVEN, NON-INTERACTIVE)
# ======================================================================

def _model_selection_from_config(
    models: "DataFrameLike",
    logger: Logger,
    output_dir: Path,
) -> ModelSelectionResult:
    """
    Non-interactive model selection using BASE_INPUT_DIR / ModelSelection.txt

    ModelSelection.txt is expected to contain exactly one active mode, e.g.:

        mode=by_disease
        diseases=Bladder Urothelial Carcinoma

    Supported modes (same as before):
      - by_ids
      - by_lineage
      - by_disease
      - keyword
      - by_names
    """
    config_path = output_dir / "ModelSelection.txt"
    cfg = _load_simple_kv_config(config_path, logger)

    mode = cfg.get("mode", "").strip().lower()
    valid_modes = {"by_ids", "by_lineage", "by_disease", "keyword", "by_names"}

    if mode not in valid_modes:
        msg = (
            f"Invalid or missing 'mode' in {config_path}. "
            f"Expected one of {sorted(valid_modes)}, got {mode!r}."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Model selection mode from config: %s", mode)

    # Dispatch based on mode
    if mode == "by_ids":
        ids_raw = cfg.get("ids", "")
        ids = _split_csv_value(ids_raw)
        if not ids:
            raise ValueError(
                f"mode=by_ids but no 'ids=' provided in {config_path}"
            )
        logger.info("Selecting models by_ids: %s", ids)
        model_sel = select_models(
            models=models,
            mode="by_ids",
            ids=ids,
            logger=logger,
        )

    elif mode == "by_lineage":
        lineages_raw = cfg.get("lineages", "")
        lineages = _split_csv_value(lineages_raw)
        if not lineages:
            raise ValueError(
                f"mode=by_lineage but no 'lineages=' provided in {config_path}"
            )
        logger.info("Selecting models by_lineage: %s", lineages)
        model_sel = select_models(
            models=models,
            mode="by_lineage",
            lineages=lineages,
            logger=logger,
        )

    elif mode == "by_disease":
        diseases_raw = cfg.get("diseases", "")
        diseases = _split_csv_value(diseases_raw)
        if not diseases:
            raise ValueError(
                f"mode=by_disease but no 'diseases=' provided in {config_path}"
            )
        logger.info("Selecting models by_disease: %s", diseases)
        model_sel = select_models(
            output_dir=output_dir,
            models=models,
            mode="by_disease",
            diseases=diseases,
            logger=logger,
        )

    elif mode == "keyword":
        keywords_raw = cfg.get("keywords", "")
        keywords = _split_csv_value(keywords_raw)
        if not keywords:
            raise ValueError(
                f"mode=keyword but no 'keywords=' provided in {config_path}"
            )
        keyword_mode = cfg.get("keyword_mode", "any").strip().lower() or "any"
        if keyword_mode not in {"any", "all"}:
            logger.warning(
                "Invalid keyword_mode=%r in %s; defaulting to 'any'.",
                keyword_mode,
                config_path,
            )
            keyword_mode = "any"

        logger.info(
            "Selecting models by keyword: %s (mode=%s)",
            keywords,
            keyword_mode,
        )
        model_sel = select_models(
            models=models,
            mode="keyword",
            keywords=keywords,
            keyword_mode=keyword_mode,
            logger=logger,
        )

    elif mode == "by_names":
        names_raw = cfg.get("names", "")
        names = _split_csv_value(names_raw)
        if not names:
            raise ValueError(
                f"mode=by_names but no 'names=' provided in {config_path}"
            )
        logger.info("Selecting models by_names: %s", names)
        model_sel = select_models(
            models=models,
            mode="by_names",
            names=names,
            logger=logger,
        )

    else:
        # Should never get here due to valid_modes check above.
        raise ValueError(f"Unhandled model selection mode: {mode!r}")

    print("\n=== Model selection completed (from config) ===")
    print("Selected model IDs:", model_sel.selected_ids)
    logger.info("Model selection completed. n=%d", len(model_sel.selected_ids))

    return model_sel


# ======================================================================
# GENE SELECTION (CONFIG-DRIVEN, SAME-RUN ONLY)
# ======================================================================

def _gene_selection_from_config(
    output_dir: Path,
    deg_simple_path: Path,
    logger: Logger,
) -> Tuple["DataFrameLike", "DataFrameLike"]:
    """
    Non-interactive gene selection from prepared_DEGs_Simple.csv
    for THIS run only (no cross-run fallback).

    Uses:
        PREPARED_DIR / 'prepared_DEGs_Simple.csv'
    which should have been created by run_prepare_deg.py
    using the same RUN_STAMP.

    Driven by:
        BASE_INPUT_DIR / GeneSelection.txt

    GeneSelection.txt supports:
      - mode=all
      - mode=top  +  top_up=<int>  top_down=<int>

    Default behavior (if user doesn't explicitly choose top mode):
      -> mode=all → ALL genes are used downstream.
    """
    if not deg_simple_path.exists():
        msg = (
            f"DEG simple file not found for this run: {deg_simple_path}\n"
            f"Please run run_prepare_deg.py first (it will use the same RUN_STAMP)."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    config_path = output_dir / "GeneSelection.txt"
    cfg = _load_simple_kv_config(config_path, logger)

    mode_in = cfg.get("mode", "").strip().lower()
    if mode_in in {"1", "a", "all"}:
        gmode: Literal["all", "top"] = "all"
    elif mode_in in {"top", "t", "2"}:
        gmode = "top"
    else:
        # If mode is missing or unrecognized, DEFAULT to "all genes".
        logger.warning(
            "Unrecognized or missing gene selection mode %r in %s; "
            "defaulting to 'all' (use all genes).",
            mode_in,
            config_path,
        )
        gmode = "all"

    if gmode == "all":
        top_up = 0
        top_down = 0
        logger.info(
            "Gene selection mode 'all': using ALL genes from %s",
            deg_simple_path,
        )
    else:
        # Try to parse top_up / top_down; default 20 / 20
        def _parse_int(key: str, default: int) -> int:
            raw = cfg.get(key, "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
                if value <= 0:
                    raise ValueError
                return value
            except ValueError:
                logger.warning(
                    "Invalid integer for '%s' in GeneSelection.txt: %r. "
                    "Using default %d.",
                    key,
                    raw,
                    default,
                )
                return default

        top_up = _parse_int("top_up", 20)
        top_down = _parse_int("top_down", 20)
        logger.info(
            "Gene selection mode 'top': top_up=%d, top_down=%d from %s",
            top_up,
            top_down,
            deg_simple_path,
        )

    df_full, df_selected = build_gene_lists_from_degs(
        output_dir=output_dir,
        deg_simple_path=deg_simple_path,
        mode=gmode,
        top_up=top_up,
        top_down=top_down,
        logger=logger,
    )

    print("\n=== Gene selection completed (from config) ===")
    print("Full list shape:    ", df_full.shape)
    print("Selected list shape:", df_selected.shape)
    print(df_selected.head())

    logger.info(
        "Gene selection completed. full_n=%d, selected_n=%d, mode=%s, "
        "top_up=%d, top_down=%d",
        df_full.shape[0],
        df_selected.shape[0],
        gmode,
        top_up,
        top_down,
    )

    return df_full, df_selected


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_check_cell_essentiality(
    output_dir: Path,
    deg_simple_path: Path,
    logger: Optional[Logger] = None,
) -> CheckCellEssentialityResult:
    """
    Full Check_CellEssentiality pipeline combining:

      4 – DepMap cell line setup
      5 – Model selection  (NON-INTERACTIVE, from ModelSelection.txt)
      6 – Gene identity check
      7 – Gene selection   (NON-INTERACTIVE, from GeneSelection.txt)
      8 – DepMap gene layers (omics data)

    All steps share a single RUN_STAMP and therefore write
    into the same DepMap_<STAMP> run folder.

    Parameters
    ----------
    logger : logging.Logger, optional
        If None, uses the 'DepMap' logger which propagates to root.

    Returns
    -------
    CheckCellEssentialityResult
        Dataclass holding context, selections, and paths.
    """
    logger = _ensure_logger(logger)

    # ------------------------------------------------------------------
    # Global logging of context
    # ------------------------------------------------------------------
    logger.info("=== Step 04–08: Check_CellEssentiality pipeline started ===")
    logger.info("Output directory: %s", output_dir)
    logger.info("DEG simple path: %s", deg_simple_path)


    # ------------------------------------------------------------------
    # 1) DepMap cell line setup (cell line context + matrices)
    # ------------------------------------------------------------------
    ctx = run_depmap_cellline_setup(output_dir=output_dir, logger=logger)
    master = ctx.master

    # ------------------------------------------------------------------
    # 2) Model selection (CONFIG-DRIVEN, NON-INTERACTIVE)
    # ------------------------------------------------------------------
    models = build_models_table_from_master(master)
    model_sel = _model_selection_from_config(output_dir=output_dir, models=models, logger=logger)

    # NOTE: At this stage we only record model selection and save
    #       Selected_Models.csv in CELL_LINES_DIR (via select_models).
    #       We do not yet subset ctx.effect_df by selected models here.

    # ------------------------------------------------------------------
    # 3) Gene selection (CONFIG-DRIVEN, SAME-RUN ONLY)
    # ------------------------------------------------------------------
    gene_full, gene_selected = _gene_selection_from_config(output_dir=output_dir, deg_simple_path=deg_simple_path, logger=logger)

    # ------------------------------------------------------------------
    # 4) Gene identity check
    # ------------------------------------------------------------------
    gene_id_df = run_gene_identity_check(output_dir=output_dir, logger=logger)
    print("\n=== Gene identity check completed ===")
    print(gene_id_df.head())

    # ------------------------------------------------------------------
    # 5) DepMap gene-layer matrices
    # ------------------------------------------------------------------
    selected_genes_path = output_dir / "Prepared_DEGs" / "InputGenes_selected.csv"

    layer_result = build_gene_layer_matrices(
        ctx=ctx,
        selected_genes_path=selected_genes_path,
        output_dir=output_dir,
        logger=logger,
    )

    print("\n=== DepMap gene-layer matrices completed ===")
    print(f"Selected genes: {len(layer_result.selected_genes)}")
    print("Genes per layer:")
    for layer, genes in layer_result.present_per_layer.items():
        print(f"  {layer:11s}: {len(genes)}")

    print("\nSaved matrices:")
    print("  Chronos effect :", layer_result.effect_matrix_path)
    print("  Dependency     :", layer_result.dep_matrix_path)
    print("  Expression     :", layer_result.expr_matrix_path)
    print("  CNV            :", layer_result.cnv_matrix_path)
    print("Gene presence summary:", layer_result.gene_presence_path)

    logger.info("Check_CellEssentiality pipeline completed successfully.")

    # Wrap results into dataclass for programmatic use
    result = CheckCellEssentialityResult(
        ctx=ctx,
        model_selection=model_sel,
        gene_full=gene_full,
        gene_selected=gene_selected,
        gene_identity=gene_id_df,
        gene_layers=layer_result,
    )

    return result
