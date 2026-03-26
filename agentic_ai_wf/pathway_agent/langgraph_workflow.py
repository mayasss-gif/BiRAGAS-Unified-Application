"""
LangGraph Workflow Definition for Pathway Analysis Pipeline

Main workflow orchestration using LangGraph StateGraph with comprehensive
retry logic, conditional routing, and state persistence.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .agent_state import PipelineState
from .query_parser import parse_user_query
from .langgraph_nodes import (
    enrichment_node,
    deduplication_node,
    categorization_node,
    literature_node,
    consolidation_node
)

logger = logging.getLogger(__name__)


def should_retry_or_continue(state: PipelineState) -> str:
    """
    Conditional routing function for retry logic and stage progression
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name to execute
    """
    current = state['current_stage']
    
    if current == 'failed':
        logger.error("-> Pipeline failed - stopping execution")
        return END
    elif current == 'completed':
        logger.info("-> Pipeline completed successfully")
        return END
    else:
        # Route to the current stage for execution or retry
        logger.info(f"-> Routing to stage: {current}")
        return current


def create_pathway_analysis_workflow():
    """
    Create the LangGraph workflow for autonomous pathway analysis
    
    Returns:
        Compiled LangGraph application with checkpointing
    """
    logger.info("-> Building LangGraph workflow for pathway analysis")
    
    # Initialize graph with state
    workflow = StateGraph(PipelineState)
    
    # Add nodes for each stage
    logger.info("-> Adding workflow nodes")
    workflow.add_node("parse_query", parse_user_query)
    workflow.add_node("enrichment", enrichment_node)
    workflow.add_node("deduplication", deduplication_node)
    workflow.add_node("categorization", categorization_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("consolidation", consolidation_node)
    
    # Set entry point
    workflow.set_entry_point("parse_query")
    
    # Add edges with conditional routing
    logger.info("-> Adding workflow edges with conditional routing")
    
    # From query parsing to enrichment
    workflow.add_edge("parse_query", "enrichment")
    
    # Enrichment stage with retry logic
    workflow.add_conditional_edges(
        "enrichment",
        should_retry_or_continue,
        {
            "enrichment": "enrichment",  # Retry same stage
            "deduplication": "deduplication",  # Move to next stage
            "categorization": "categorization",  # Causal mode shortcut
            END: END  # Stop on failure
        }
    )
    
    # Deduplication stage with retry logic
    workflow.add_conditional_edges(
        "deduplication",
        should_retry_or_continue,
        {
            "deduplication": "deduplication",  # Retry same stage
            "categorization": "categorization",  # Move to next stage
            END: END  # Stop on failure
        }
    )
    
    # Categorization stage with retry logic
    workflow.add_conditional_edges(
        "categorization",
        should_retry_or_continue,
        {
            "categorization": "categorization",  # Retry same stage
            "literature": "literature",  # Move to next stage
            END: END  # Stop on failure
        }
    )
    
    # Literature analysis stage with retry logic
    workflow.add_conditional_edges(
        "literature",
        should_retry_or_continue,
        {
            "literature": "literature",  # Retry same stage
            "consolidation": "consolidation",  # Move to next stage
            END: END  # Stop on failure
        }
    )
    
    # Consolidation stage with retry logic
    workflow.add_conditional_edges(
        "consolidation",
        should_retry_or_continue,
        {
            "consolidation": "consolidation",  # Retry same stage
            END: END  # Stop on completion or failure
        }
    )
    
    # Compile with checkpointing for state persistence
    logger.info("-> Setting up checkpointing for state persistence")
    memory = MemorySaver()
    
    try:
        app = workflow.compile(checkpointer=memory)
        logger.info("-> LangGraph workflow compiled successfully")
        return app
    except Exception as e:
        logger.error(f"-> Failed to compile LangGraph workflow: {e}")
        raise


def create_simple_workflow():
    """
    Create a simplified workflow for testing without complex routing
    
    Returns:
        Simplified compiled LangGraph application
    """
    logger.info("-> Building simplified LangGraph workflow")
    
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_user_query)
    workflow.add_node("enrichment", enrichment_node)
    workflow.add_node("deduplication", deduplication_node)
    workflow.add_node("categorization", categorization_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("consolidation", consolidation_node)
    
    # Simple linear flow
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "enrichment")
    workflow.add_edge("enrichment", "deduplication")
    workflow.add_edge("deduplication", "categorization")
    workflow.add_edge("categorization", "literature")
    workflow.add_edge("literature", "consolidation")
    workflow.add_edge("consolidation", END)
    
    # Compile without checkpointing for simplicity
    app = workflow.compile()
    logger.info("-> Simplified LangGraph workflow compiled")
    return app
