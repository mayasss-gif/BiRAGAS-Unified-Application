"""
LangGraph Node Functions for Pathway Analysis Pipeline

Individual node functions for each pipeline stage with comprehensive
error handling, retry logic, and progress tracking.
"""

import time
import logging
from pathlib import Path
from .agent_state import PipelineState
from . import enrichment_pipeline, run_deduplication, categorize_pathways, run_literature_pipeline
from .consolidation import main as consolidation_main
from .ui_logging import emit_from_state, emit_from_state_async




logger = logging.getLogger(__name__)


def enrichment_node(state: PipelineState) -> PipelineState:
    """Stage 1: Pathway Enrichment"""
    emit_from_state(state, "info", f"Starting pathway enrichment for {state['disease_name']} (Stage 1/5)")
    logger.info(f"-> Stage 1/5: Starting enrichment for {state['disease_name']}")
    state['progress_messages'].append(f"Starting enrichment for {state['disease_name']}")

    try:
        deg_path = state.get("deg_file_path")
        emit_from_state(state, "info", f"Loading DEGs from {Path(deg_path).name if deg_path else 'input file'}")
        output_path = enrichment_pipeline(
            state['disease_name'],
            state['deg_file_path'],
            state['patient_prefix'],
            state['output_dir']
        )

        state['enrichment_output'] = output_path
        state['completed_stages'].append('enrichment')

        if state.get('causal'):
            state['deduplication_output'] = output_path
            state['current_stage'] = 'categorization'
            emit_from_state(
                state, "info",
                f"Enrichment complete (causal mode): {output_path.name if output_path else 'output'}"
            )
            state['progress_messages'].append(
                f"Enrichment complete (causal mode): {output_path}"
            )
            logger.info("-> Causal mode enabled: proceeding to categorization")
            return state

        state['current_stage'] = 'deduplication'
        emit_from_state(
            state, "info",
            f"Enrichment complete: {len(state['completed_stages'])} stage(s) done — {output_path.name if output_path else 'output saved'}"
        )
        state['progress_messages'].append(f"Enrichment complete: {output_path}")

        logger.info(f"-> Enrichment complete: {output_path}")
        return state

    except Exception as e:
        emit_from_state(state, "error", f"Enrichment failed: {e}")
        logger.error(f"-> Enrichment failed: {e}")
        return handle_stage_error(state, 'enrichment', e)


def deduplication_node(state: PipelineState) -> PipelineState:
    """Stage 2: Pathway Deduplication"""
    if state.get('causal'):
        emit_from_state(state, "info", "Causal mode: skipping deduplication")
        state['deduplication_output'] = state.get('enrichment_output')
        state['current_stage'] = 'categorization'
        state['progress_messages'].append("Causal mode: skipping deduplication")
        logger.info("-> Causal mode: skipping deduplication")
        return state

    emit_from_state(state, "info", "Starting pathway deduplication (Stage 2/5)")
    logger.info(f"-> Stage 2/5: Starting deduplication")
    state['progress_messages'].append("Starting pathway deduplication")

    try:
        run_deduplication(str(state['enrichment_output']))
        state['deduplication_output'] = state['enrichment_output']
        state['completed_stages'].append('deduplication')
        state['current_stage'] = 'categorization'
        emit_from_state(state, "info", "Deduplication complete — removing redundant pathway entries")
        state['progress_messages'].append("Deduplication complete")

        logger.info(f"-> Deduplication complete")
        return state

    except Exception as e:
        emit_from_state(state, "error", f"Deduplication failed: {e}")
        logger.error(f"-> Deduplication failed: {e}")
        return handle_stage_error(state, 'deduplication', e)


def categorization_node(state: PipelineState) -> PipelineState:
    """Stage 3: Pathway Categorization"""
    if state.get('causal') and state.get('current_stage') == 'completed':
        logger.info("-> Causal mode: skipping categorization")
        return state

    emit_from_state(state, "info", "Starting pathway categorization (Stage 3/5) — assigning ontology classes")
    logger.info(f"-> Stage 3/5: Starting categorization")
    state['progress_messages'].append("Starting pathway categorization")

    try:
        deduplication_output = state['deduplication_output']
        if not deduplication_output:
            raise ValueError("No deduplication output available for categorization")

        categorized_file = categorize_pathways(str(deduplication_output))

        state['categorization_output'] = Path(categorized_file)
        state['completed_stages'].append('categorization')
        if state.get('causal'):
            emit_from_state(state, "info", "Categorization complete (causal mode) — pipeline concluding")
            state['current_stage'] = 'completed'
            state['progress_messages'].append("Categorization complete (causal mode)")
            logger.info("-> Causal mode enabled: concluding pipeline after categorization")
            return state

        state['current_stage'] = 'literature'
        emit_from_state(state, "info", "Categorization complete — pathways tagged with Main_Class and Sub_Class")
        state['progress_messages'].append("Categorization complete")

        logger.info(f"-> Categorization complete")
        return state

    except Exception as e:
        emit_from_state(state, "error", f"Categorization failed: {e}")
        logger.error(f"-> Categorization failed: {e}")
        return handle_stage_error(state, 'categorization', e)


async def literature_node(state: PipelineState) -> PipelineState:
    """Stage 4: Literature Analysis"""
    if state.get('causal') and state.get('current_stage') == 'completed':
        logger.info("-> Causal mode: skipping literature analysis")
        return state

    await emit_from_state_async(state, "info", "Starting literature analysis (Stage 4/5) — fetching PubMed evidence")
    logger.info(f"-> Stage 4/5: Starting literature analysis")
    state['progress_messages'].append("Starting literature analysis")

    try:
        await run_literature_pipeline(
            state['categorization_output'],
            state['disease_name']
        )

        state['literature_output'] = state['categorization_output']
        state['completed_stages'].append('literature')
        state['current_stage'] = 'consolidation'
        await emit_from_state_async(state, "info", "Literature analysis complete — pathways annotated with clinical/functional relevance")
        state['progress_messages'].append("Literature analysis complete")

        logger.info(f"-> Literature analysis complete")
        return state

    except Exception as e:
        await emit_from_state_async(state, "error", f"Literature analysis failed: {e}")
        logger.error(f"-> Literature analysis failed: {e}")
        return handle_stage_error(state, 'literature', e)


async def consolidation_node(state: PipelineState) -> PipelineState:
    """Stage 5: Pathway Consolidation & Prioritization"""
    if state.get('causal') and state.get('current_stage') == 'completed':
        logger.info("-> Causal mode: skipping consolidation")
        return state

    await emit_from_state_async(state, "info", "Starting consolidation and prioritization (Stage 5/5) — computing LLM scores")
    logger.info(f"-> Stage 5/5: Starting consolidation and prioritization")
    state['progress_messages'].append("Starting pathway consolidation and prioritization")

    try:
        enrichment_dir = (
            Path(state["output_dir"])
            if state.get("output_dir")
            else state["literature_output"].parent
        )
        consolidation_output_dir = enrichment_dir.parent / "pathway_consolidation"

        output_path = await consolidation_main(
            categoriezed_pathways_path=state['literature_output'],
            disease_name=state['disease_name'],
            output_dir_path=consolidation_output_dir
        )

        state['consolidation_output'] = output_path
        state['completed_stages'].append('consolidation')
        state['current_stage'] = 'completed'
        n_stages = len(state['completed_stages'])
        await emit_from_state_async(
            state, "info",
            f"Pathway pipeline complete: {n_stages} stages, output saved to {output_path.name if output_path else 'consolidated file'}"
        )
        state['progress_messages'].append(f"Consolidation complete: {output_path}")

        logger.info(f"-> Consolidation complete: {output_path}")
        return state

    except Exception as e:
        await emit_from_state_async(state, "error", f"Consolidation failed: {e}")
        logger.error(f"-> Consolidation failed: {e}")
        return handle_stage_error(state, 'consolidation', e)


def handle_stage_error(state: PipelineState, stage: str, error: Exception) -> PipelineState:
    """
    Handle errors with retry logic and exponential backoff
    
    Args:
        state: Current pipeline state
        stage: Name of the failed stage
        error: Exception that occurred
        
    Returns:
        Updated pipeline state with error handling
    """
    retry_count = state['retry_counts'].get(stage, 0)
    
    # Log error details
    error_info = {
        'stage': stage,
        'error': str(error),
        'retry_count': retry_count,
        'timestamp': time.time()
    }
    state['errors'].append(error_info)
    
    if retry_count < state['max_retries']:
        # Increment retry count
        state['retry_counts'][stage] = retry_count + 1

        # Calculate exponential backoff delay
        delay = 2 ** retry_count
        emit_from_state(state, "warning", f"Retrying {stage} ({retry_count + 1}/{state['max_retries']}) after {delay}s")
        logger.warning(f"-> Error in {stage}: {error}")
        logger.warning(f"-> Retry {retry_count + 1}/{state['max_retries']} after {delay}s delay")
        
        # Add progress message
        state['progress_messages'].append(
            f"Error in {stage}: {str(error)[:100]}... Retrying {retry_count + 1}/{state['max_retries']}"
        )
        
        # Sleep for exponential backoff
        time.sleep(delay)
        
        # Set current stage to retry the same stage
        state['current_stage'] = stage
        
    else:
        # Max retries exceeded
        emit_from_state(state, "error", f"Stage {stage} failed after {state['max_retries']} retries")
        logger.error(f"-> {stage} failed after {state['max_retries']} retries: {error}")
        state['current_stage'] = 'failed'
        state['progress_messages'].append(
            f"Stage {stage} failed permanently after {state['max_retries']} retries"
        )
    
    return state
