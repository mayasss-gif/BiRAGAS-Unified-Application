"""Multi-omics integration node."""

import logging
from pathlib import Path

from agentic_ai_wf.multiomics_pipeline_agent.main import (
    MultiomicsLayers,
    MultiomicsPipelineArgs,
    run_multiomics_agent_with_args,
)

from ..config_nodes import get_node_dir
from ..wf_common import ensure_global_config
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("multiomics", step_number=9)
async def multiomics_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Multi-omics integration - integrates multiple omics layers."""
    ensure_global_config(state)
    analysis_id = state["analysis_id"]
    disease_name = state.get("disease_name", "unknown_disease")

    output_dir = get_node_dir("multiomics", analysis_id)
    transcriptome_dir = Path(state.get("analysis_transcriptome_dir", ""))

    layers = {}
    if state.get("multiomics_layers"):
        layers = state["multiomics_layers"]
    else:
        layer_patterns = {
            "genomics": ["*genomic*.csv", "*genomic*.tsv", "*genomic*.xlsx"],
            "transcriptomics": ["*transcriptomic*.csv", "*transcriptomic*.tsv", "*counts*.csv"],
            "epigenomics": ["*epigenomic*.csv", "*epigenomic*.tsv", "*epigenomic*.xlsx"],
            "proteomics": ["*proteomic*.csv", "*proteomic*.tsv", "*proteomic*.xlsx"],
            "metabolomics": ["*metabolomic*.csv", "*metabolomic*.tsv", "*metabolomic*.xlsx"],
        }
        if transcriptome_dir.exists():
            for layer_name, patterns in layer_patterns.items():
                for pattern in patterns:
                    found = list(transcriptome_dir.glob(pattern))
                    if found:
                        layers[layer_name] = str(found[0])
                        break

    if not layers:
        raise FileNotFoundError(
            f"No multiomics layer files found. Provide via state['multiomics_layers'] or upload to {transcriptome_dir}"
        )

    metadata_path = state.get("multiomics_metadata_path")
    label_column = state.get("multiomics_label_column")
    query_term = state.get("multiomics_query_term") or f"multi-omics integration {disease_name}"
    disease_term = state.get("multiomics_disease_term") or disease_name
    n_pcs = state.get("multiomics_n_pcs_per_layer", 20)
    integrated_dim = state.get("multiomics_integrated_dim", 50)
    top_n = state.get("multiomics_top_n_results", 20)
    up_to_step = state.get("multiomics_up_to_step")

    layers_model = MultiomicsLayers(**layers) if layers else None
    multiomics_args = MultiomicsPipelineArgs(
        output_dir=str(output_dir),
        layers=layers_model,
        metadata_path=metadata_path,
        label_column=label_column,
        n_pcs_per_layer=n_pcs,
        integrated_dim=integrated_dim,
        query_term=query_term,
        disease_term=disease_term,
        top_n_results=top_n,
        up_to_step=up_to_step,
    )

    result = await run_multiomics_agent_with_args(args=multiomics_args, max_attempts=3)

    if isinstance(result, dict):
        if result.get("status") == "failed":
            errors = result.get("errors", [])
            raise Exception(f"Multiomics failed: {'; '.join(str(e) for e in errors)}")
        result_payload = result.get("result", {})
        if isinstance(result_payload, dict) and result_payload.get("output_dir"):
            od = Path(result_payload["output_dir"])
            if od.exists():
                output_dir = od

    if not output_dir.exists():
        raise Exception("Multiomics output directory not found")

    return {"multiomics_output_dir": str(output_dir), "current_step": 9}
