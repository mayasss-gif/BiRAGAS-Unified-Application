
from __future__ import annotations
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .validation_layer import ClinicalValidator
from .text_generation import llm_generate_molecular_signature_description
from .pathways_processing import _process_pathways
from .config import UPREG_TARGET, DEFAULT_CONFIDENCE
from .disease_severity import llm_calculate_disease_severity

from .utils import _infer_columns_deg, _rename_deg_columns, _filter_significant_pathways, _build_log2fc_map, _build_pathway_context, _fig_to_base64, _sort_key_final, _validate_inputs, _fill_to_ten_upregulated, _enrich_downregulated, _top_and_bottom_gene, _signature_block

logger = logging.getLogger(__name__)


def compute_deg_stats(
    df: pd.DataFrame,
    *,
    p_value_thr: float = 0.05,
    logfc_thr: float = 1.0,
    top_n: int = 10,
    patient_prefix: str = "patient",
    disease_name: str = None
) -> Dict:
    """Return global DEG metrics and bar‑plot (b64)."""
    gene_col, fc_col, p_value_col = _infer_columns_deg(df, patient_prefix)
    df = df.rename(columns={gene_col: "Gene", fc_col: "log2FC", p_value_col: "p_value"})

    sig   = df[(df.p_value < p_value_thr) & (df.log2FC.abs() >= logfc_thr)]
    up    = sig[sig.log2FC > 0]
    down  = sig[sig.log2FC < 0]

    # regulation plot
    plt.figure(figsize=(3, 2.8))
    plt.bar(["Up", "Down"], [len(up), len(down)], color=['red', 'blue'])
    plt.ylabel("Genes")
    plt.title("Differentially Expressed Genes")
    plot_b64 = _fig_to_base64()


    def _top(frame: pd.DataFrame) -> List[Dict]:
        return (
            frame.assign(abs_fc=lambda d: d.log2FC.abs())
            .sort_values("abs_fc", ascending=False)
            .head(top_n)[["Gene", "log2FC", "p_value"]]
            .to_dict(orient="records")
        )

    # top genes plot with names
    # Prepare data for horizontal bar plot
    top_up = _top(up)
    top_down = _top(down)
    
    # Combine for plotting (down first, then up)
    genes = [g['Gene'] for g in reversed(top_down)] + [g['Gene'] for g in top_up]
    log2fc = [g['log2FC'] for g in reversed(top_down)] + [g['log2FC'] for g in top_up]
    colors = ['#d62728'] * len(top_down) + ['#2ca02c'] * len(top_up)  # red for down, green for up
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(genes, log2fc, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Log2 Fold Change')
    plt.title('Extended Differential Gene Expression')
    plt.legend([
        plt.Rectangle((0,0),1,1,color='#2ca02c'),
        plt.Rectangle((0,0),1,1,color='#d62728')
    ], ['Upregulated', 'Downregulated'], loc='upper left')
    plt.tight_layout()
    deg_topgenes_plot_b64 = _fig_to_base64()
    
    # Handle confidence level counts safely
    if 'Confidence' in sig.columns:
        confidence_level_counts = sig[sig['Confidence'].notnull() & (sig['Confidence'] != '')]['Confidence'].value_counts().to_dict()
    else:
        print("⚠️  Warning: 'Confidence' column not found in DEG data, using defaults")
        confidence_level_counts = {}
    
    # Safely get confidence level counts with defaults
    high_count = confidence_level_counts.get('High', 0)
    medium_count = confidence_level_counts.get('Medium', 0)
    low_count = confidence_level_counts.get('Low', 0)
    
    total_confidence_level = high_count + medium_count + low_count
    if total_confidence_level == 0:
        severity_index = 0
    else:
        numerator = (3 * high_count) + (2 * medium_count) + (1 * low_count)
        denominator = 3 * total_confidence_level
        severity_index = round((numerator / denominator) * 100, 2)

    # Take the top 10 genes from sig with the lowest 'Rank' values (if Rank exists)
    if 'Rank' in sig.columns:
        top_genes = sig.nsmallest(10, 'Rank')['Gene'].tolist()
    else:
        # Fallback: use genes with highest absolute log2FC
        top_genes = sig.assign(abs_fc=lambda d: d.log2FC.abs()).nlargest(10, 'abs_fc')['Gene'].tolist()
    
    # Calculate disease severity using LLM
    disease_severity = llm_calculate_disease_severity(confidence_level_counts, disease_name, top_genes)

    return {
        "total_sig": len(sig),
        "up_count": len(up),
        "down_count": len(down),
        "ratio_up_down": f"{round(len(up)/len(sig), 2)}:{round(len(down)/len(sig), 2)}",
        "top_up_genes": top_up,
        "top_down_genes": top_down,
        "deg_regulation_plot": plot_b64,
        "deg_topgenes_plot_b64": deg_topgenes_plot_b64,
        "severity_index": severity_index,
        "disease_severity": disease_severity,
        "total_biomarkers": high_count + medium_count,
        "top_genes": top_genes
    }


def _run_validation_and_describe(
    clinical_validator,
    disease_name: str,
    pathway_name: str,
    regulation: str,
    top_genes: List[Dict[str, float]],
    row: Optional[pd.Series],
    validation_info: Optional[Dict],
) -> Tuple[float, bool, str]:
    """Run clinical validation and produce a concise description."""
    confidence = DEFAULT_CONFIDENCE
    is_pathogenic = False
    description = (
        f"The {regulation.lower()} {pathway_name} pathway is associated "
        f"with {disease_name} pathophysiology."
    )

    try:
        context = _build_pathway_context(row) if row is not None else {}
        result = clinical_validator.validate_pathway(
            pathway=pathway_name,
            disease=disease_name,
            use_cache=False,
            regulation_direction=regulation,
            pathway_genes=top_genes,
            pathway_data=context,
        )
        confidence = float(result.confidence)
        is_pathogenic = bool(result.is_valid)

        # Generate fresh 2-line description using your LLM helper
        description = llm_generate_molecular_signature_description(
            disease_name=disease_name,
            pathway_name=pathway_name,
            regulation_status=regulation,
            top_genes=top_genes,
            validation_status=(
                validation_info.get("status") if validation_info else None
            ),
        )
        logger.info(
            "Validated %s | conf=%.2f | pathogenic=%s",
            pathway_name,
            confidence,
            is_pathogenic,
        )
    except Exception as exc:
        logger.warning("Validation/LLM description failed for %s: %s",
                       pathway_name, exc)
        # description falls back to the default crafted above

    return confidence, is_pathogenic, description


# ------------------------------- main API --------------------------------- #

def compute_pathogenic_molecular_signatures(
    deg_df: pd.DataFrame,
    sig_pathway_df: pd.DataFrame,
    validated_pathways: Dict,
    disease_name: str,
    patient_prefix: str = "patient",
    top_downregulated_pathways: Optional[List[Dict]] = None,
    top_upregulated_pathways: Optional[List[Dict]] = None,
) -> Dict:
    """Compute molecular signatures with pathogenic prioritization.

    Returns a dict with:
    - signature_data: list of up/down signature sections
    - summary counts and top/bottom gene info
    """
    _validate_inputs(deg_df, sig_pathway_df, disease_name)

    clinical_validator = ClinicalValidator()
    validation_enabled = True
    logger.info("Clinical validation system loaded")

    # Standardize inputs
    deg_df = _rename_deg_columns(deg_df, patient_prefix)
    pathway_df = _filter_significant_pathways(sig_pathway_df)
    log2fc_map = _build_log2fc_map(deg_df)

    # Pathogenic names from prior validations (if any)
    pathogenic_names = [
        p.get("pathway_name")
        for p in validated_pathways.get("pathogenic_pathways", [])
        if p.get("pathway_name")
    ]

    # Process all pathways
    upregulated, downregulated = _process_pathways(
        pathway_df=pathway_df,
        disease_name=disease_name,
        log2fc_map=log2fc_map,
        pathogenic_names=pathogenic_names,
        clinical_validator=clinical_validator,
        validation_enabled=validation_enabled,
    )

    # Compose final selections
    top_10_up = _fill_to_ten_upregulated(upregulated)
    top_10_down = _enrich_downregulated(downregulated)

    # (Optional) Allow caller-provided extras to fill remaining upregulated slots
    if top_upregulated_pathways and len(top_10_up) < UPREG_TARGET:
        remaining = UPREG_TARGET - len(top_10_up)
        # best-effort append (assume caller provides normalized dicts)
        top_10_up.extend(top_upregulated_pathways[:remaining])
        top_10_up = sorted(top_10_up, key=_sort_key_final)[:UPREG_TARGET]

    # (Optional) Allow caller-provided extras for downregulated
    if top_downregulated_pathways:
        merged = top_10_down + top_downregulated_pathways
        merged = sorted(merged, key=_sort_key_final)[:UPREG_TARGET]
        top_10_down = merged

    # Build signature sections
    signatures: List[Dict] = []
    if top_10_up:
        signatures.append(
            _signature_block(
                name="Top Upregulated Signatures",
                regulation="Upregulated",
                pathways=top_10_up,
            )
        )
    if top_10_down:
        signatures.append(
            _signature_block(
                name="Top Downregulated Signatures",
                regulation="Downregulated",
                pathways=top_10_down,
            )
        )

    # Summary
    top_gene, top_lfc, low_gene, low_lfc = _top_and_bottom_gene(deg_df)

    pathogenic_up = sum(1 for p in top_10_up if p.get("is_pathogenic"))
    pathogenic_down = sum(1 for p in top_10_down if p.get("is_pathogenic"))

    result = {
        "signature_data": signatures,
        "top_gene": top_gene,
        "top_lfc": top_lfc,
        "lowest_gene": low_gene,
        "lowest_lfc": low_lfc,
        "upregulated_count": len(top_10_up),
        "downregulated_count": len(top_10_down),
        "total_pathogenic_pathways": pathogenic_up + pathogenic_down,
        "pathogenic_upregulated_count": pathogenic_up,
        "non_pathogenic_upregulated_count": len(top_10_up) - pathogenic_up,
        "pathogenic_downregulated_count": pathogenic_down,
        "non_pathogenic_downregulated_count": len(top_10_down) - pathogenic_down,
    }
    return result



        



