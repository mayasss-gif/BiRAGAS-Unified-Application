from __future__ import annotations

import os
import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict
from decouple import config

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
)

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


def multiomics_pipeline_direct(
    output_dir: str,
    layers: Optional[Dict[str, str]] = None,
    metadata_path: Optional[str] = None,
    label_column: Optional[str] = None,
    n_pcs_per_layer: int = 20,
    integrated_dim: int = 50,
    query_term: Optional[str] = None,
    disease_term: Optional[str] = None,
    top_n_results: int = 20,
    up_to_step: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Direct wrapper for multiomics pipeline (non-agent mode).
    
    For agent-based execution with self-correction, use run_multiomics_agent_with_args_sync().
    """
    return run_pipeline(
        output_dir=output_dir,
        layers=layers,
        metadata_path=metadata_path,
        label_column=label_column,
        n_pcs_per_layer=n_pcs_per_layer,
        integrated_dim=integrated_dim,
        query_term=query_term,
        disease_term=disease_term,
        top_n_results=top_n_results,
        up_to_step=up_to_step,
        enable_logging=True,
    )


class MultiomicsLayers(BaseModel):
    """
    Layer file paths for multiomics pipeline.
    All fields are optional - user can provide one or all layers.
    """
    model_config = ConfigDict(extra='forbid')
    
    genomics: Optional[str] = Field(
        default=None,
        description="Path to genomics layer file (CSV/TSV/Excel)"
    )
    transcriptomics: Optional[str] = Field(
        default=None,
        description="Path to transcriptomics layer file (CSV/TSV/Excel)"
    )
    epigenomics: Optional[str] = Field(
        default=None,
        description="Path to epigenomics layer file (CSV/TSV/Excel)"
    )
    proteomics: Optional[str] = Field(
        default=None,
        description="Path to proteomics layer file (CSV/TSV/Excel)"
    )
    metabolomics: Optional[str] = Field(
        default=None,
        description="Path to metabolomics layer file (CSV/TSV/Excel)"
    )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dict format expected by pipeline, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class MultiomicsPipelineArgs(BaseModel):
    """
    Arguments for multiomics pipeline execution.
    """
    model_config = ConfigDict(extra='forbid')
    
    output_dir: str = Field(..., description="Root output directory for all pipeline outputs")
    layers: Optional[MultiomicsLayers] = Field(
        default=None,
        description="Layer file paths. All fields optional - provide one or all layers"
    )
    metadata_path: Optional[str] = Field(
        default=None,
        description="Path to metadata CSV file with sample information and labels"
    )
    label_column: Optional[str] = Field(
        default=None,
        description="Name of the column in metadata CSV that contains sample labels/classes"
    )
    n_pcs_per_layer: int = Field(
        default=20,
        description="Number of principal components to extract per omics layer"
    )
    integrated_dim: int = Field(
        default=50,
        description="Dimensionality for the integrated multiomics representation"
    )
    query_term: Optional[str] = Field(
        default=None,
        description="Query term for literature mining (Step 6). Required if running Step 6"
    )
    disease_term: Optional[str] = Field(
        default=None,
        description="Disease term for literature search context"
    )
    top_n_results: int = Field(
        default=20,
        description="Number of top biomarkers to include in literature mining (Step 6)"
    )
    up_to_step: Optional[int] = Field(
        default=None,
        description="Run pipeline up to and including this step number (1-6). If None, runs all steps"
    )


class MultiomicsPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool.
    """
    model_config = ConfigDict(extra='forbid')
    
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]


@dataclass
class MultiomicsRunnerContext:
    """
    Context for tracking retries and diagnostics.
    """
    max_attempts: int = 3
    attempts: int = 0
    last_args: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


@function_tool
def run_multiomics_pipeline_tool(
    output_dir: str,
    layers: Optional[MultiomicsLayers] = None,
    metadata_path: Optional[str] = None,
    label_column: Optional[str] = None,
    n_pcs_per_layer: int = 20,
    integrated_dim: int = 50,
    query_term: Optional[str] = None,
    disease_term: Optional[str] = None,
    top_n_results: int = 20,
    up_to_step: Optional[int] = None,
) -> MultiomicsPipelineToolResult:
    """
    Safe wrapper around multiomics pipeline that always returns structured result.
    
    Supports optional layers - user can submit one or all files.
    """
    args: Dict[str, Any] = {
        "output_dir": output_dir,
        "layers": layers,
        "metadata_path": metadata_path,
        "label_column": label_column,
        "n_pcs_per_layer": n_pcs_per_layer,
        "integrated_dim": integrated_dim,
        "query_term": query_term,
        "disease_term": disease_term,
        "top_n_results": top_n_results,
        "up_to_step": up_to_step,
    }
    
    try:
        # Convert MultiomicsLayers model to dict if provided
        layers_dict = None
        if layers:
            layers_dict = layers.to_dict()
        
        # Validate at least one layer is provided
        if not layers_dict or len(layers_dict) == 0:
            raise ValueError("At least one layer file must be provided in 'layers'")
        
        # Validate file paths exist
        missing_files = []
        for layer_name, file_path in layers_dict.items():
            if file_path and not os.path.exists(file_path):
                missing_files.append(f"{layer_name}: {file_path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing layer files: {', '.join(missing_files)}")
        
        # Validate metadata if provided
        if metadata_path and not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Run pipeline
        result = run_pipeline(
            output_dir=output_dir,
            layers=layers_dict,
            metadata_path=metadata_path,
            label_column=label_column,
            n_pcs_per_layer=n_pcs_per_layer,
            integrated_dim=integrated_dim,
            query_term=query_term,
            disease_term=disease_term,
            top_n_results=top_n_results,
            up_to_step=up_to_step,
            enable_logging=True,
        )
        
        return MultiomicsPipelineToolResult(
            ok=True,
            payload=result,
            error=None,
            args=args,
        )
        
    except Exception:
        tb = traceback.format_exc()
        return MultiomicsPipelineToolResult(
            ok=False,
            payload=None,
            error=tb,
            args=args,
        )


async def multiomics_tool_use_behavior(
    context: RunContextWrapper[MultiomicsRunnerContext],
    results: List[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """
    Custom tool use behavior with retry logic.
    """
    ctx = context.context
    latest_result: FunctionToolResult = results[-1]
    
    tool_output: MultiomicsPipelineToolResult = latest_result.output
    
    if tool_output.ok:
        final_summary = {
            "status": "ok",
            "attempts": ctx.attempts + 1,
            "args_used": tool_output.args,
            "result": tool_output.payload,
        }
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=final_summary,
        )
    
    ctx.attempts += 1
    ctx.last_args = tool_output.args
    if tool_output.error:
        ctx.errors.append(tool_output.error)
    
    if ctx.attempts >= ctx.max_attempts:
        final_failure = {
            "status": "failed",
            "attempts": ctx.attempts,
            "last_args": ctx.last_args,
            "errors": ctx.errors,
        }
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=final_failure,
        )
    
    return ToolsToFinalOutputResult(is_final_output=False)


def build_multiomics_runner_agent() -> Agent[MultiomicsRunnerContext]:
    """
    Create the specialized Multiomics Runner Agent.
    """
    instructions = """
        You are a Multi-Omics Pipeline Runner Agent for bioinformatics data integration.

        Your job:
        1. Read the user-provided arguments for the multiomics pipeline.
        2. Validate and adjust arguments before calling the tool:
           - Ensure output_dir exists or can be created
           - Verify at least one layer file is provided in 'layers' (genomics, transcriptomics, epigenomics, proteomics, or metabolomics)
           - Validate file paths exist for provided layers
           - Handle optional parameters (metadata_path, query_term, disease_term)
           - Fix relative paths using absolute paths or resolve relative to current working directory
           - Construct 'layers' as an object with optional fields: genomics, transcriptomics, epigenomics, proteomics, metabolomics
        3. Call the `run_multiomics_pipeline_tool` to execute the pipeline.
        4. If the tool returns an error:
           - Carefully read the stacktrace
           - Infer which argument(s) caused the problem (file paths, directory paths, permissions, etc.)
           - Modify only those arguments, keeping others unchanged
           - Try again, up to the maximum number of attempts
        5. When the pipeline succeeds or attempts are exhausted, return a concise,
           structured JSON summary with:
           - status: "ok" or "failed"
           - attempts
           - args_used (final args)
           - result (the tool's payload if ok)
           - errors (if any)

        Always reason step-by-step internally, but output only the final structured result.
        """
    
    agent = Agent[MultiomicsRunnerContext](
        name="Multiomics Runner Agent",
        instructions=instructions.strip(),
        model="gpt-4o-mini",
        tools=[run_multiomics_pipeline_tool],
        tool_use_behavior=multiomics_tool_use_behavior,
        model_settings=ModelSettings(
            tool_choice="required",
            temperature=0.0,
        ),
    )
    return agent


async def run_multiomics_agent_with_args(
    args: MultiomicsPipelineArgs,
    max_attempts: int = 3,
) -> Any:
    """
    High-level async entrypoint: runs the Multiomics Runner Agent with given args.
    """
    agent = build_multiomics_runner_agent()
    
    context = MultiomicsRunnerContext(max_attempts=max_attempts)
    
    user_input = (
        f"Run multiomics integration pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
    )
    
    return result.final_output


def run_multiomics_agent_with_args_sync(
    args: MultiomicsPipelineArgs,
    max_attempts: int = 3,
) -> Any:
    """
    Synchronous wrapper for run_multiomics_agent_with_args.
    
    Handles event loop creation for Django/Gunicorn thread pool environments.
    """
    logger.info(
        f"Running multiomics agent synchronously: "
        f"output_dir={args.output_dir}, "
        f"layers={list(args.layers.keys()) if args.layers else []}, "
        f"max_attempts={max_attempts}"
    )
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop for multiomics agent")
    
    agent = build_multiomics_runner_agent()
    context = MultiomicsRunnerContext(max_attempts=max_attempts)
    
    user_input = (
        f"Run multiomics integration pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    try:
        result = Runner.run_sync(
            starting_agent=agent,
            input=user_input,
            context=context,
        )
        
        status = result.final_output.get('status') if isinstance(result.final_output, dict) else 'unknown'
        logger.info(f"Multiomics agent completed: status={status}")
        return result.final_output
    except Exception as e:
        logger.exception(f"Multiomics agent execution failed: {e}")
        raise

