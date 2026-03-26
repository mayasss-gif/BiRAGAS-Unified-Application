"""Drug discovery node."""

import logging
from pathlib import Path

from agentic_ai_wf.drugs_extraction_evaluation.new_kegg_pipeline import main as kegg_drug_discovery_pipeline

from ..config_nodes import get_node_file
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("drug_discovery", step_number=5)
async def drug_discovery_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Drug discovery pipeline from pathways and genes."""
    prioritized_genes_path = Path(state["prioritized_genes_path"])
    pathway_consolidation_path = Path(state["pathway_consolidation_path"])

    output_file = get_node_file("drug_discovery", state["analysis_id"], "drug_discovery_result.csv")

    drug_discovery_path = kegg_drug_discovery_pipeline(
        consolidated_pathways_file=pathway_consolidation_path,
        prioritized_genes_file=prioritized_genes_path,
        output_file=output_file,
        analysis_id=state["analysis_id"],
        disease_name=state["disease_name"],
    )

    if not drug_discovery_path:
        raise Exception("Drug discovery pipeline returned None")

    return {
        "drug_discovery_path": str(Path(drug_discovery_path).resolve()),
        "current_step": 5,
    }
