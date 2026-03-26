"""LangGraph workflow construction."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .error_handler import error_handler_node
from .wf_common import should_continue_after_error
from .nodes import (
    clinical_report_node,
    cohort_retrieval_node,
    deg_analysis_node,
    drug_discovery_node,
    finalization_node,
    gene_prioritization_node,
    pathway_enrichment_node,
    perturbation_analysis_node,
    pharma_report_node,
)
from .state import TranscriptomeAnalysisState


def build_transcriptome_analysis_graph():
    """Build the complete LangGraph workflow."""
    graph = StateGraph(TranscriptomeAnalysisState)

    graph.add_node("cohort_retrieval", cohort_retrieval_node)
    graph.add_node("deg_analysis", deg_analysis_node)
    graph.add_node("gene_prioritization", gene_prioritization_node)
    graph.add_node("pathway_enrichment", pathway_enrichment_node)
    graph.add_node("perturbation_analysis", perturbation_analysis_node)
    graph.add_node("drug_discovery", drug_discovery_node)
    graph.add_node("clinical_report", clinical_report_node)
    graph.add_node("pharma_report", pharma_report_node)
    graph.add_node("finalization", finalization_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("cohort_retrieval")
    graph.add_edge("cohort_retrieval", "deg_analysis")
    graph.add_conditional_edges(
        "deg_analysis",
        should_continue_after_error,
        {"continue": "gene_prioritization", "abort": "error_handler"},
    )
    graph.add_edge("gene_prioritization", "pathway_enrichment")
    graph.add_edge("pathway_enrichment", "perturbation_analysis")
    graph.add_edge("perturbation_analysis", "drug_discovery")
    graph.add_edge("drug_discovery", "clinical_report")
    graph.add_edge("drug_discovery", "pharma_report")
    graph.add_edge("clinical_report", "finalization")
    graph.add_edge("pharma_report", "finalization")
    graph.add_edge("finalization", END)
    graph.add_edge("error_handler", END)

    return graph.compile(checkpointer=MemorySaver())
