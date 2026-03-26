from __future__ import annotations

import os
import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field
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

# Import harmonization functions
from .harmonizer.pipeline_agent import harmonize_from_local, harmonize_single_paths
from .harmonizer.harmonizer import harmonize_single, harmonize_local

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
# ---------------------------------------------------------------------------
# 0. Simple wrapper functions for direct pipeline execution (non-agent)
# ---------------------------------------------------------------------------

def harmonization_pipeline_single(
    counts_path: str,
    meta_path: str,
    output_dir: str,
    out_mode: str = "co_locate",
    create_zip: bool = False,
) -> Dict[str, Any]:
    """
    Simple wrapper for direct single-dataset harmonization (non-agent mode).
    
    For agent-based execution with self-correction, use run_harmonization_agent_with_args_sync().
    
    Args:
        counts_path: Path to counts/expression table (csv/tsv/xlsx)
        meta_path: Path to metadata table (csv/tsv/xlsx)
        output_dir: Directory where all results will be stored
        out_mode: Output mode ("co_locate", "default", etc.)
        create_zip: Whether to create a zip file of results
        
    Returns:
        Dictionary with harmonization results
    """
    return harmonize_single(
        counts_path=counts_path,
        meta_path=meta_path,
        output_dir=output_dir,
    )


def harmonization_pipeline_local(
    data_root: str,
    output_dir: str,
    combine: bool = True,
) -> Dict[str, Any]:
    """
    Simple wrapper for direct local discovery harmonization (non-agent mode).
    
    For agent-based execution with self-correction, use run_harmonization_agent_with_args_sync().
    
    Args:
        data_root: Root directory to crawl for 'prep' folders
        output_dir: Directory where all results will be stored
        combine: If True, attempt to combine multiple datasets
        
    Returns:
        Dictionary with harmonization results
    """
    return harmonize_local(
        data_root=data_root,
        output_dir=output_dir,
        combine=combine,
    )


# ---------------------------------------------------------------------------
# 1. Pydantic models for arguments and results
# ---------------------------------------------------------------------------


class HarmonizationSingleArgs(BaseModel):
    """
    Arguments for single-dataset harmonization mode.
    """
    mode: Literal["single"] = Field(default="single", description="Harmonization mode")
    counts_path: str = Field(..., description="Path to counts/expression table (csv/tsv/xlsx)")
    meta_path: str = Field(..., description="Path to metadata table (csv/tsv/xlsx)")
    output_dir: str = Field(..., description="Directory where all results will be written")
    out_mode: str = Field(
        default="default",
        description="Output mode: 'default' (uses output_dir), 'co_locate' (saves next to input files, ignores output_dir). Default is 'default' to respect output_dir."
    )
    create_zip: bool = Field(
        default=False,
        description="Whether to create a zip file of results"
    )


class HarmonizationLocalArgs(BaseModel):
    """
    Arguments for local discovery harmonization mode.
    """
    mode: Literal["local"] = Field(default="local", description="Harmonization mode")
    data_root: str = Field(..., description="Root directory to crawl for 'prep' folders")
    output_dir: str = Field(..., description="Directory where all results will be written")
    combine: bool = Field(
        default=True,
        description="If True, attempt to combine multiple datasets"
    )


class HarmonizationArgs(BaseModel):
    """
    Unified arguments model that supports both single and local modes.
    The agent will determine which mode to use based on provided arguments.
    """
    mode: Literal["single", "local"] = Field(..., description="Harmonization mode: 'single' or 'local'")
    
    # Single mode fields
    counts_path: Optional[str] = Field(
        default=None,
        description="Path to counts/expression table (required for 'single' mode)"
    )
    meta_path: Optional[str] = Field(
        default=None,
        description="Path to metadata table (required for 'single' mode)"
    )
    out_mode: str = Field(
        default="default",
        description="Output mode: 'default' (uses output_dir), 'co_locate' (saves next to input files, ignores output_dir). Default is 'default' to respect output_dir."
    )
    create_zip: bool = Field(
        default=False,
        description="Whether to create a zip file (for 'single' mode)"
    )
    
    # Local mode fields
    data_root: Optional[str] = Field(
        default=None,
        description="Root directory to crawl for 'prep' folders (required for 'local' mode)"
    )
    combine: bool = Field(
        default=True,
        description="If True, attempt to combine multiple datasets (for 'local' mode)"
    )
    
    # Common fields
    output_dir: str = Field(..., description="Directory where all results will be written")


class HarmonizationPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool so the agent and
    custom tool_use_behavior can make decisions.
    """
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]
    mode: str


# ---------------------------------------------------------------------------
# 2. Context object to track retries / diagnostics
# ---------------------------------------------------------------------------


@dataclass
class HarmonizationRunnerContext:
    """
    This is the context type passed into Runner.run(..., context=...).

    It is NOT visible to the LLM. It's only for your Python code:
    - custom tool_use_behavior
    - future hooks / guardrails
    """
    max_attempts: int = 3
    attempts: int = 0
    last_args: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Function tool that wraps harmonization functions
# ---------------------------------------------------------------------------


@function_tool
def run_harmonization_pipeline_tool(
    mode: str,
    output_dir: str,
    counts_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    data_root: Optional[str] = None,
    out_mode: str = "co_locate",
    create_zip: bool = False,
    combine: bool = True,
) -> HarmonizationPipelineToolResult:
    """
    Safe wrapper around harmonization functions that always returns a structured result.
    
    Supports two modes:
    - 'single': Harmonize a single dataset with explicit file paths
    - 'local': Auto-discover and harmonize multiple datasets from a root directory
    
    The LLM sees this as a tool with strongly-typed arguments.
    """
    args: Dict[str, Any] = {
        "mode": mode,
        "output_dir": output_dir,
        "counts_path": counts_path,
        "meta_path": meta_path,
        "data_root": data_root,
        "out_mode": out_mode,
        "create_zip": create_zip,
        "combine": combine,
    }

    try:
        if mode == "single":
            # Validate required fields for single mode
            if not counts_path or not meta_path:
                raise ValueError(
                    "For 'single' mode, both 'counts_path' and 'meta_path' are required"
                )
            
            # Use 'default' mode to respect output_dir parameter
            # 'co_locate' mode ignores out_root and saves next to input files
            # If user explicitly wants 'co_locate', they can set it, but default respects output_dir
            effective_out_mode = out_mode if out_mode != "co_locate" else "default"
            if out_mode == "co_locate":
                logger.warning(
                    f"out_mode='co_locate' ignores output_dir. Using 'default' mode to respect "
                    f"output_dir={output_dir}. Set out_mode='co_locate' explicitly if you want outputs "
                    f"next to input files."
                )
            
            # Delegate to single harmonization
            result = harmonize_single_paths(
                counts_path=counts_path,
                meta_path=meta_path,
                out_mode=effective_out_mode,
                out_root=output_dir,
                create_zip=create_zip,
            )
            
            # Wrap result in expected format
            wrapped_result = {
                "result": result,
                "outputs": [
                    result.get("outdir"),
                    result.get("figdir"),
                    result.get("zip"),
                ],
                "summary_path": None,  # Can be added if needed
            }
            
            return HarmonizationPipelineToolResult(
                ok=True,
                payload=wrapped_result,
                error=None,
                args=args,
                mode="single",
            )
            
        elif mode == "local":
            # Validate required fields for local mode
            if not data_root:
                raise ValueError(
                    "For 'local' mode, 'data_root' is required"
                )
            
            # Delegate to local discovery harmonization
            result = harmonize_from_local(
                data_root=data_root,
                combine=combine,
                out_root=output_dir,
            )
            
            # Wrap result in expected format
            wrapped_result = {
                "result": result,
                "outputs": [],
                "summary_path": None,
            }
            
            # Extract outputs from result structure
            if result.get("mode") == "single":
                single_result = result.get("result", {})
                wrapped_result["outputs"] = [
                    single_result.get("outdir"),
                    single_result.get("figdir"),
                    single_result.get("zip"),
                ]
            elif result.get("mode") == "multi":
                runs = result.get("runs", {})
                combined = result.get("combined", {})
                outputs = []
                for run in runs.values():
                    outputs.extend([
                        run.get("outdir"),
                        run.get("figdir"),
                        run.get("zip"),
                    ])
                if combined:
                    outputs.extend([
                        combined.get("outdir"),
                        combined.get("figdir"),
                        combined.get("zip"),
                    ])
                wrapped_result["outputs"] = [o for o in outputs if o]
            
            return HarmonizationPipelineToolResult(
                ok=True,
                payload=wrapped_result,
                error=None,
                args=args,
                mode="local",
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single' or 'local'")
            
    except Exception:
        tb = traceback.format_exc()
        return HarmonizationPipelineToolResult(
            ok=False,
            payload=None,
            error=tb,
            args=args,
            mode=mode,
        )


# ---------------------------------------------------------------------------
# 4. Custom tool_use_behavior to implement self-correcting loop
# ---------------------------------------------------------------------------


async def harmonization_tool_use_behavior(
    context: RunContextWrapper[HarmonizationRunnerContext],
    results: List[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """
    Custom tool use behavior that:
    - Stops immediately on success and returns pipeline outputs.
    - On failure, allows the LLM to see the error and try again with updated args,
      but only up to context.max_attempts.
    """
    ctx = context.context
    latest_result: FunctionToolResult = results[-1]

    # Our tool returns a HarmonizationPipelineToolResult
    tool_output: HarmonizationPipelineToolResult = latest_result.output  # type: ignore[assignment]

    if tool_output.ok:
        # Success: finalize
        final_summary = {
            "status": "ok",
            "attempts": ctx.attempts + 1,  # +1 because we succeeded
            "args_used": tool_output.args,
            "mode": tool_output.mode,
            "result": tool_output.payload,
        }
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=final_summary,
        )

    # Failed execution path:
    ctx.attempts += 1
    ctx.last_args = tool_output.args
    if tool_output.error:
        ctx.errors.append(tool_output.error)

    # If we've exhausted the allowed attempts, stop and surface the errors
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

    # Otherwise, let the LLM see the tool output + stacktrace and try again
    # with new arguments. Returning is_final_output=False tells the Agent loop
    # to keep going.
    return ToolsToFinalOutputResult(is_final_output=False)


# ---------------------------------------------------------------------------
# 5. Agent factory
# ---------------------------------------------------------------------------


def build_harmonization_runner_agent() -> Agent[HarmonizationRunnerContext]:
    """
    Create the specialized Harmonization Runner Agent.

    - Forces tool use with tool_choice="required"
    - Uses our custom harmonization_tool_use_behavior
    - Uses run_harmonization_pipeline_tool as the only tool
    """
    instructions = """
        You are a Dataset Harmonization Runner Agent for RNA-seq bioinformatics data.

        Your job:
        1. Read the user-provided arguments for the harmonization pipeline.
        2. Determine the correct mode ('single' or 'local') based on provided arguments:
           - If 'counts_path' and 'meta_path' are provided → use 'single' mode
           - If 'data_root' is provided → use 'local' mode
        3. Validate and, if needed, adjust the arguments before calling the tool:
           - Fix relative paths using absolute paths or resolve them relative to current working directory
           - Ensure output_dir exists or can be created
           - For 'single' mode: verify counts_path and meta_path exist
           - For 'single' mode: If output_dir is provided, use out_mode='default' (not 'co_locate') to respect the output directory
           - For 'local' mode: verify data_root exists and contains discoverable datasets
           - Handle file format variations (csv, tsv, xlsx, etc.)
        4. Call the `run_harmonization_pipeline_tool` to execute the pipeline.
        5. If the tool returns an error:
           - Carefully read the stacktrace.
           - Infer which argument(s) caused the problem (file paths, directory paths, permissions, etc.)
           - Modify only those arguments, keeping others unchanged.
           - Try again, up to the maximum number of attempts.
        6. When the pipeline finally succeeds or attempts are exhausted, return a concise,
           structured JSON summary with:
           - status: "ok" or "failed"
           - attempts
           - mode: "single" or "local"
           - args_used (final args)
           - result (the tool's payload if ok)
           - errors (if any).

        Always reason step-by-step internally, but output only the final structured result.
        """

    agent = Agent[HarmonizationRunnerContext](
        name="Harmonization Runner Agent",
        instructions=instructions.strip(),
        model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
        tools=[run_harmonization_pipeline_tool],
        tool_use_behavior=harmonization_tool_use_behavior,
        model_settings=ModelSettings(
            # Force the model to always use a tool on each LLM call
            tool_choice="required",
            temperature=0.0,
        ),
    )
    return agent


# ---------------------------------------------------------------------------
# 6. High-level helper to run the agent from Python/CLI
# ---------------------------------------------------------------------------


async def run_harmonization_agent_with_args(
    args: HarmonizationArgs,
    max_attempts: int = 3,
) -> Any:
    """
    High-level async entrypoint: runs the Harmonization Runner Agent with the given args.
    Returns the final_output from Runner.run(..).
    """
    agent = build_harmonization_runner_agent()

    context = HarmonizationRunnerContext(max_attempts=max_attempts)

    # We pass a structured input describing the task and the args.
    # The LLM will parse this and call run_harmonization_pipeline_tool with
    # whatever arguments it decides (usually matching these).
    # Runner.run() expects input to be a string, not a dict
    user_input = (
        f"Run dataset harmonization pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )

    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
    )

    # result.final_output is what we set in harmonization_tool_use_behavior
    return result.final_output


def run_harmonization_agent_with_args_sync(
    args: HarmonizationArgs,
    max_attempts: int = 3,
) -> Any:
    """
    Synchronous wrapper for run_harmonization_agent_with_args.
    
    Handles event loop creation for Django/Gunicorn thread pool environments.
    This is the production-ready version for EC2 Django servers.
    
    Uses Runner.run_sync() which handles event loop management internally.
    Similar pattern to temporal_pipeline_agent.py and deconvolution_insights_agent.py for EC2 compatibility.
    
    Args:
        args: HarmonizationArgs with pipeline parameters
        max_attempts: Maximum retry attempts
        
    Returns:
        Final output from the agent execution
    """
    logger.info(
        f"Running harmonization agent synchronously: "
        f"mode={args.mode}, "
        f"output_dir={args.output_dir}, "
        f"max_attempts={max_attempts}"
    )
    
    # Ensure event loop exists (for multi-threaded Django environments)
    # This matches the pattern from temporal_pipeline_agent.py and deconvolution_insights_agent.py
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop for harmonization agent")
    
    agent = build_harmonization_runner_agent()
    context = HarmonizationRunnerContext(max_attempts=max_attempts)
    
    # We pass a structured input describing the task and the args.
    # Runner.run_sync() expects input to be a string, not a dict
    user_input = (
        f"Run dataset harmonization pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    # Use Runner.run_sync() for thread-safe execution
    try:
        result = Runner.run_sync(
            starting_agent=agent,
            input=user_input,
            context=context,
        )
        
        # result.final_output is what we set in harmonization_tool_use_behavior
        status = result.final_output.get('status') if isinstance(result.final_output, dict) else 'unknown'
        logger.info(f"Harmonization agent completed: status={status}")
        return result.final_output
    except Exception as e:
        logger.exception(f"Harmonization agent execution failed: {e}")
        raise


# ---------------------------------------------------------------------------
# 7. CLI entrypoint
# ---------------------------------------------------------------------------


async def _main_async() -> None:
    """
    Example async CLI entrypoint.
    """
    # Example 1: Single dataset harmonization
    single_args = HarmonizationArgs(
        mode="single",
        counts_path="agentic_ai_wf/harmonization_pipeline_agent/data/counts.csv",
        meta_path="agentic_ai_wf/harmonization_pipeline_agent/data/metadata.csv",
        output_dir="agentic_ai_wf/harmonization_pipeline_agent/output",
        create_zip=False,
    )

    final_output = await run_harmonization_agent_with_args(
        args=single_args,
        max_attempts=3,
    )

    # For CLI usage, just print the final structured result
    print(json.dumps(final_output, indent=2))
    
    # Example 2: Local discovery mode (commented out)
    # local_args = HarmonizationArgs(
    #     mode="local",
    #     data_root="path/to/data/root",
    #     output_dir="path/to/output",
    #     combine=True,
    # )
    # 
    # final_output = await run_harmonization_agent_with_args(
    #     args=local_args,
    #     max_attempts=3,
    # )
    # 
    # print(json.dumps(final_output, indent=2))


def main() -> None:
    """
    Main CLI entrypoint.
    """
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
