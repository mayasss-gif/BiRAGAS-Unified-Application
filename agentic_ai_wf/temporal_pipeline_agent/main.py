from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputFunction,
    ToolsToFinalOutputResult,
    function_tool,
)

from .temporal_bulk.runner import run_temporal_analysis

logger = logging.getLogger(__name__)

AGENT_NAME = "Temporal Analysis Agent"
STEP = "temporal_analysis"


def _temporal_artifacts_present(output_dir: Path) -> bool:
    """True when the bulk runner finished (DONE marker or main TSV present)."""
    if not output_dir.is_dir():
        return False
    if (output_dir / "DONE").is_file():
        return True
    if (output_dir / "temporal_gene_fits.tsv").is_file():
        return True
    return False


def _normalize_temporal_run_result(
    args: "TemporalAnalysisArgs",
    final_output: Any,
) -> Any:
    """
    The OpenAI Agents Runner may set final_output to a string or omit status even when
    run_temporal_analysis completed and wrote files. Align with on-disk truth so LangGraph
    does not see status=unknown after a successful long run.
    """
    out = Path(args.output_dir)

    if isinstance(final_output, dict):
        st = final_output.get("status")
        if st == "failed":
            return final_output
        if st in ("ok", "success"):
            return final_output
        if _temporal_artifacts_present(out):
            merged = dict(final_output)
            merged["status"] = "ok"
            merged["inferred_from_artifacts"] = True
            return merged
        return final_output

    if _temporal_artifacts_present(out):
        logger.info(
            "Temporal: final_output type=%s; artifacts under %s — coercing status=ok",
            type(final_output).__name__,
            out,
        )
        return {
            "status": "ok",
            "inferred_from_artifacts": True,
            "note": "Runner final_output was not a structured dict",
        }

    logger.error(
        "Temporal: missing artifacts under %s and no structured agent result",
        out,
    )
    return {
        "status": "failed",
        "errors": [
            "Agent did not return a structured result and expected output files were not found.",
        ],
    }


def _emit_temporal_log(
    workflow_logger: Any,
    event_loop: Optional[asyncio.AbstractEventLoop],
    level: str,
    message: str,
) -> None:
    """Emit log to UI from sync pipeline (thread-safe via run_coroutine_threadsafe)."""
    if not workflow_logger or not event_loop:
        return
    try:
        async def _do_log() -> None:
            try:
                if level == "info":
                    await workflow_logger.info(agent_name=AGENT_NAME, message=message, step=STEP)
                elif level == "warning":
                    await workflow_logger.warning(agent_name=AGENT_NAME, message=message, step=STEP)
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(_do_log(), event_loop)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 0. Simple wrapper function for direct pipeline execution (non-agent)
# ---------------------------------------------------------------------------

def temporal_pipeline(
    counts: str,
    metadata: str,
    input_dir: str,
    treatment_level: str,
    genes_list: str,
    deconv_csv: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simple wrapper for direct temporal analysis execution (non-agent mode).
    
    For agent-based execution with self-correction, use run_temporal_agent_with_args_sync().
    
    Args:
        counts: Path to counts matrix file
        metadata: Path to metadata file
        input_dir: Base directory for resolving paths
        treatment_level: Treatment/condition level
        genes_list: Gene list (file path or comma-separated string)
        deconv_csv: Path to deconvolution CSV file
        output_dir: Output directory (optional)
        
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = str(Path(input_dir) / "temporal_output")
    
    return run_temporal_analysis(
        counts=counts,
        metadata=metadata,
        input_dir=input_dir,
        treatment_level=treatment_level,
        genes_list=genes_list,
        deconv_csv=deconv_csv,
        output_dir=output_dir
    )


# ---------------------------------------------------------------------------
# 1. Pydantic models for arguments and results
# ---------------------------------------------------------------------------


class TemporalAnalysisArgs(BaseModel):
    """
    Minimal argument surface for the temporal pipeline that the agent will control.

    You can extend this model with more parameters from run_temporal_analysis
    if you want the agent to have direct control over them.
    """
    output_dir: str = Field(..., description="Directory where all results will be written.",)
    counts: str = Field(..., description="Path to counts matrix (rows=genes, columns=samples).")
    metadata: str = Field(..., description="Path to metadata CSV/TSV with sample_id column.")
    input_dir: Optional[str] = Field(
        default=None,
        description="Base directory for resolving relative file paths."
    )
    treatment_level: str = Field(
        default="",
        description="Treatment/condition level used for DE/interaction."
    )
    # Accept both a file path or literal list of gene names
    genes_list: Optional[List[str] | str] = Field(
        default=None,
        description="Either a list of gene IDs or a path to a gene list file."
    )
    deconv_csv: Optional[str] = Field(
        default=None,
        description="Optional cell-type deconvolution proportions file."
    )


class TemporalPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool so the agent and
    custom tool_use_behavior can make decisions.
    """
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]


# ---------------------------------------------------------------------------
# 2. Context object to track retries / diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TemporalRunnerContext:
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
    workflow_logger: Optional[Any] = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# 3. Function tool that wraps run_temporal_analysis
# ---------------------------------------------------------------------------


def _create_run_temporal_pipeline_tool(
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
):
    """Factory for run_temporal_pipeline_tool with optional UI logging."""

    @function_tool
    def run_temporal_pipeline_tool(
        output_dir: str,
        counts: str,
        metadata: str,
        input_dir: Optional[str] = None,
        treatment_level: str = "",
        genes_list: Optional[List[str] | str] = None,
        deconv_csv: Optional[str] = None,
    ) -> TemporalPipelineToolResult:
        """
        Safe wrapper around run_temporal_analysis that always returns a structured result.

        The LLM sees this as a tool with strongly-typed arguments.
        """
        args: Dict[str, Any] = {
            "output_dir": output_dir,
            "counts": counts,
            "metadata": metadata,
            "input_dir": input_dir,
            "treatment_level": treatment_level,
            "genes_list": genes_list,
            "deconv_csv": deconv_csv,
        }

        if workflow_logger and event_loop:
            _emit_temporal_log(workflow_logger, event_loop, "info", "Executing temporal pipeline...")

        try:
            result = run_temporal_analysis(
                output_dir=output_dir,
                counts=counts,
                metadata=metadata,
                input_dir=input_dir,
                treatment_level=treatment_level,
                genes_list=genes_list if genes_list is not None else "",
                deconv_csv=deconv_csv or "",
                workflow_logger=workflow_logger,
                event_loop=event_loop,
            )
            return TemporalPipelineToolResult(ok=True, payload=result, error=None, args=args)
        except Exception:
            tb = traceback.format_exc()
            return TemporalPipelineToolResult(ok=False, payload=None, error=tb, args=args)

    return run_temporal_pipeline_tool


# Default tool (no UI logging) for CLI and non-LangGraph usage
run_temporal_pipeline_tool = _create_run_temporal_pipeline_tool()


# ---------------------------------------------------------------------------
# 4. Custom tool_use_behavior to implement self-correcting loop
# ---------------------------------------------------------------------------


async def temporal_tool_use_behavior(
    context: RunContextWrapper[TemporalRunnerContext],
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
    wl = ctx.workflow_logger
    tool_output: TemporalPipelineToolResult = latest_result.output  # type: ignore[assignment]

    if tool_output.ok:
        attempt_msg = f"Pipeline completed successfully (attempt {ctx.attempts + 1})"
        if wl:
            await wl.info(agent_name=AGENT_NAME, message=attempt_msg, step=STEP)
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
        err_preview = (tool_output.error or "")[:300].replace("\n", " ")
        if wl:
            await wl.warning(
                agent_name=AGENT_NAME,
                message=f"All {ctx.max_attempts} attempts exhausted. Last error: {err_preview}...",
                step=STEP,
            )
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

    if wl:
        await wl.info(
            agent_name=AGENT_NAME,
            message=f"Attempt {ctx.attempts} failed — agent will retry with adjusted parameters",
            step=STEP,
        )
    return ToolsToFinalOutputResult(is_final_output=False)


# ---------------------------------------------------------------------------
# 5. Agent factory
# ---------------------------------------------------------------------------


def build_temporal_runner_agent(
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Agent[TemporalRunnerContext]:
    """
    Create the specialized Temporal Runner Agent.

    - Forces tool use with tool_choice="required"
    - Uses our custom temporal_tool_use_behavior
    - Uses run_temporal_pipeline_tool (with optional UI logging)
    """
    instructions = """
        You are a Temporal Bulk RNA-seq Runner Agent.

        Your job:
        1. Read the user-provided arguments for the temporal bulk pipeline.
        2. Validate and, if needed, adjust the arguments before calling the tool:
        - Fix relative paths using `input_dir` if paths don't exist.
        - If the pipeline fails due to time or gene column detection, adjust parameters
            to rely on the built-in LLM helpers in the Python code.
        - If treatment_level is invalid, inspect the metadata columns and choose a valid
            value or fall back to an empty treatment.
        3. Call the `run_temporal_pipeline_tool` to execute the pipeline.
        4. If the tool returns an error:
        - Carefully read the stacktrace.
        - Infer which argument(s) caused the problem.
        - Modify only those arguments, keeping others unchanged.
        - Try again, up to the maximum number of attempts.
        5. When the pipeline finally succeeds or attempts are exhausted, return a concise,
        structured JSON summary with:
        - status: "ok" or "failed"
        - attempts
        - args_used (final args)
        - result (the tool's payload if ok)
        - errors (if any).
        Always reason step-by-step internally, but output only the final structured result.
        """

    tool = _create_run_temporal_pipeline_tool(workflow_logger, event_loop)
    agent = Agent[TemporalRunnerContext](
        name="Temporal Runner Agent",
        instructions=instructions.strip(),
        model="gpt-4.1-mini",
        tools=[tool],
        tool_use_behavior=temporal_tool_use_behavior,
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


async def run_temporal_agent_with_args(
    args: TemporalAnalysisArgs,
    max_attempts: int = 3,
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Any:
    """
    High-level async entrypoint: runs the Temporal Runner Agent with the given args.
    Returns the final_output from Runner.run(..).
    """
    agent = build_temporal_runner_agent(workflow_logger, event_loop)
    context = TemporalRunnerContext(
        max_attempts=max_attempts,
        workflow_logger=workflow_logger,
        event_loop=event_loop,
    )

    # We pass a structured input describing the task and the args.
    # The LLM will parse this and call run_temporal_pipeline_tool with
    # whatever arguments it decides (usually matching these).
    # Runner.run() expects input to be a string, not a dict
    import json
    user_input = (
        f"Run temporal bulk RNA-seq analysis with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )

    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
    )

    normalized = _normalize_temporal_run_result(args, result.final_output)
    logger.info(
        "Temporal agent (async) completed: status=%s",
        normalized.get("status") if isinstance(normalized, dict) else "unknown",
    )
    return normalized


def run_temporal_agent_with_args_sync(
    args: TemporalAnalysisArgs,
    max_attempts: int = 3,
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Any:
    """
    Synchronous wrapper for run_temporal_agent_with_args.
    
    Handles event loop creation for Django/Gunicorn thread pool environments.
    This is the production-ready version for EC2 Django servers.
    
    Uses Runner.run_sync() which handles event loop management internally.
    Similar pattern to deconvolution_insights_agent.py for EC2 compatibility.
    
    When called from LangGraph via asyncio.to_thread(), workflow_logger and event_loop
    allow UI logs to stream in real-time: the main loop stays free to process
    run_coroutine_threadsafe-scheduled log emissions from the sync pipeline.
    
    Args:
        args: TemporalAnalysisArgs with pipeline parameters
        max_attempts: Maximum retry attempts
        workflow_logger: Optional logger for UI streaming (pass when run via to_thread)
        event_loop: Event loop to schedule log coroutines on (pass with workflow_logger)
        
    Returns:
        Final output from the agent execution
    """
    logger.info(
        f"Running temporal agent synchronously: "
        f"output_dir={args.output_dir}, "
        f"counts={args.counts}, "
        f"max_attempts={max_attempts}"
    )
    
    # Ensure event loop exists (for multi-threaded Django environments)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop for temporal agent")
    
    # Use provided workflow_logger/event_loop for UI streaming when run from LangGraph
    agent = build_temporal_runner_agent(workflow_logger, event_loop)
    context = TemporalRunnerContext(
        max_attempts=max_attempts,
        workflow_logger=workflow_logger,
        event_loop=event_loop,
    )
    
    # We pass a structured input describing the task and the args.
    # Runner.run_sync() expects input to be a string, not a dict
    import json
    user_input = (
        f"Run temporal bulk RNA-seq analysis with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    # Use Runner.run_sync() for thread-safe execution
    try:
        result = Runner.run_sync(
            starting_agent=agent,
            input=user_input,
            context=context,
        )

        normalized = _normalize_temporal_run_result(args, result.final_output)
        status = normalized.get("status") if isinstance(normalized, dict) else "unknown"
        logger.info("Temporal agent completed: status=%s", status)
        return normalized
    except Exception as e:
        logger.exception(f"Temporal agent execution failed: {e}")
        raise


# ---------------------------------------------------------------------------
# 7. CLI entrypoint
# ---------------------------------------------------------------------------


async def _main_async() -> None:
    # Instead of parsing CLI arguments, define variables directly here:

    # Define your variables here (edit these as needed for your run)
    output_dir = "agentic_ai_wf/temporal_pipeline_agent/temporal_output"
    counts = "agentic_ai_wf/temporal_pipeline_agent/data/counts.csv"
    metadata = "agentic_ai_wf/temporal_pipeline_agent/data/metadata.csv"
    input_dir = "agentic_ai_wf/temporal_pipeline_agent/data/"
    
    output_dir = output_dir
    counts = counts
    metadata = metadata
    input_dir = input_dir


    temporal_args = TemporalAnalysisArgs(
        output_dir=output_dir,
        counts=counts,
        metadata=metadata,
        input_dir=input_dir,
    )

    final_output = await run_temporal_agent_with_args(
        args=temporal_args,
        max_attempts=3,
    )

    # For CLI usage, just print the final structured result
    import json

    print(json.dumps(final_output, indent=2))


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
