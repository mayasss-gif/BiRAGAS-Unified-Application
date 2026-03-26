# ---------------------------- pathway processing -------------------------- #
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import UPREG_TARGET, MAX_THREADS, DEFAULT_CONFIDENCE, DEFAULT_PRIORITY
from .utils import _signature_type, _avg_fc_and_top_genes, _sort_key_base
# Moved import inside function to avoid circular import


logger = logging.getLogger(__name__)


def _process_pathway_row(
    row: pd.Series,
    disease_name: str,
    log2fc_map: Dict[str, float],
    pathogenic_paths: List[str],
    clinical_validator,
    validation_enabled: bool,
) -> Optional[Dict]:
    """Transform a pathway row into a normalized, validated dict."""
    try:
        name = row.get("Pathway", "Unknown")
        db_id = row.get("DB_ID", "Unknown")
        sig_type = _signature_type(db_id)

        genes_str = row.get("Pathway_Associated_Genes", "")
        genes = str(genes_str).split(",") if pd.notna(genes_str) and genes_str else []
        avg_fc, top3 = _avg_fc_and_top_genes(genes, log2fc_map)

        if avg_fc is None:
            return None

        regulation = "Upregulated" if avg_fc > 0 else "Downregulated"
        validation_info = None
        if name in pathogenic_paths:
            # carry through any prior validation info if available
            validation_info = {"status": "Pathogenic"}

        confidence = DEFAULT_CONFIDENCE
        is_pathogenic = name in pathogenic_paths
        description = (
            f"The {regulation.lower()} {name} pathway is "
            f"{'pathogenic' if is_pathogenic else 'associated'} in "
            f"{disease_name} pathophysiology."
        )

        if validation_enabled and clinical_validator is not None:
            from .combined_stats import _run_validation_and_describe
            confidence, ai_pathogenic, description = _run_validation_and_describe(
                clinical_validator=clinical_validator,
                disease_name=disease_name,
                pathway_name=name,
                regulation=regulation,
                top_genes=top3,
                row=row,
                validation_info=validation_info,
            )
            # Prefer AI assessment
            is_pathogenic = ai_pathogenic
            if confidence < 0.2:
                return None

        result = {
            "pathway_name": name,
            "regulation": regulation,
            "priority_rank": int(row.get("Priority_Rank", DEFAULT_PRIORITY)),
            "top_3_genes": top3,
            "validation_status": (
                validation_info.get("status")
                if validation_info
                else ("Pathogenic" if is_pathogenic else "Non-Pathogenic")
            ),
            "validation_confidence": confidence,
            "llm_description": description,
            "avg_log2fc": round(float(avg_fc), 3),
            "is_pathogenic": is_pathogenic,
            "ai_validated": validation_enabled and clinical_validator is not None,
            "validation_quality": (
                "high" if confidence >= 0.8
                else "medium" if confidence >= 0.6
                else "low" if confidence >= 0.4
                else "insufficient"
            ),
            "signature_type": sig_type,
            "db_id": db_id,
        }
        return result
    except Exception as exc:
        logger.exception("Error processing pathway %s: %s",
                         row.get("Pathway", "Unknown"), exc)
        return None


def _process_pathways(
    pathway_df: pd.DataFrame,
    disease_name: str,
    log2fc_map: Dict[str, float],
    pathogenic_names: List[str],
    clinical_validator,
    validation_enabled: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Process all pathways (parallel when large)."""
    upregulated: List[Dict] = []
    downregulated: List[Dict] = []

    total = len(pathway_df)
    use_threads = total >= UPREG_TARGET
    logger.info("Processing %d pathways | threads=%s", total, use_threads)

    if use_threads:
        with ThreadPoolExecutor(max_workers=min(MAX_THREADS, total)) as pool:
            futures = []
            for _, row in pathway_df.iterrows():
                futures.append(
                    pool.submit(
                        _process_pathway_row,
                        row=row,
                        disease_name=disease_name,
                        log2fc_map=log2fc_map,
                        pathogenic_paths=pathogenic_names,
                        clinical_validator=clinical_validator,
                        validation_enabled=validation_enabled,
                    )
                )
            for fut in as_completed(futures):
                data = fut.result()
                if not data:
                    continue
                (upregulated if data["regulation"] == "Upregulated"
                 else downregulated).append(data)
    else:
        for _, row in pathway_df.iterrows():
            data = _process_pathway_row(
                row=row,
                disease_name=disease_name,
                log2fc_map=log2fc_map,
                pathogenic_paths=pathogenic_names,
                clinical_validator=clinical_validator,
                validation_enabled=validation_enabled,
            )
            if not data:
                continue
            (upregulated if data["regulation"] == "Upregulated"
             else downregulated).append(data)

    upregulated.sort(key=_sort_key_base)
    downregulated.sort(key=_sort_key_base)
    return upregulated, downregulated
