from __future__ import annotations

import os
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
from decouple import config
# Import optimized pipeline (parallel execution + early cleanup)
from .perturbation.run_full_pipeline_optimized import run_full_pipeline

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
# ---------------------------------------------------------------------------
# 0. Simple wrapper function for direct pipeline execution (non-agent)
# ---------------------------------------------------------------------------

def perturbation_pipeline(
    deg_path: str,
    pathway_path: str,
    output_dir: str,
    disease: str,
    dep_map_addons: Optional[Dict[str, Any]] = None,
    l1000_addons: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
) -> Dict[str, Any]:
    """
    Simple wrapper for OPTIMIZED perturbation analysis execution (non-agent mode).
    
    OPTIMIZATIONS:
    - Parallel execution: DEPMAP + L1000 run simultaneously (30-40% faster)
    - Early resource cleanup: Chrome/Kaleido killed after each step
    - Graceful degradation: If one pipeline fails, others continue
    - Memory efficiency: 40-50% reduction
    
    Args:
        deg_path: Path to prioritized DEGs CSV file
        pathway_path: Path to Pathways Consolidated CSV file
        output_dir: Output directory
        disease: Disease name
        dep_map_addons: Optional DEPMAP configuration
        l1000_addons: Optional L1000 configuration
        parallel: Enable parallel execution (default: True)
        
    Returns:
        Dictionary with analysis results (status, message, output_dir, results, parallel)
    """
    logger.info("[Optimized Direct Pipeline] Running perturbation analysis WITHOUT agent (parallel mode)")
    logger.info(f"  deg_path: {deg_path}")
    logger.info(f"  pathway_path: {pathway_path}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  disease: {disease}")
    logger.info(f"  parallel: {parallel}")
    
    # Ensure addons are dicts with defaults, not None
    if dep_map_addons is None:
        dep_map_addons = {
            "mode_model": None,
            "genes_selection": "all",
            "top_up": None,
            "top_down": None,
        }
    if l1000_addons is None:
        l1000_addons = {
            "tissue": None,
            "drug": None,
            "time_points": None,
            "cell_lines": None,
        }
    
    result = run_full_pipeline(
        raw_deg_path=Path(deg_path),
        pathway_path=Path(pathway_path),
        output_dir=Path(output_dir),
        disease=disease,
        dep_map_addons=dep_map_addons,
        l1000_addons=l1000_addons,
        parallel=parallel,  # Enable parallel execution
    )
    
    logger.info(
        f"[Optimized Direct Pipeline] Completed: status={result.get('status')}, "
        f"parallel={result.get('parallel')}"
    )
    return result


# ---------------------------------------------------------------------------
# 1. Pydantic models for arguments and results
# ---------------------------------------------------------------------------

class PerturbationAnalysisArgs(BaseModel):
    """
    Argument model for perturbation pipeline that the agent will control.
    """
    output_dir: str = Field(..., description="Directory where all results will be written.")
    deg_path: str = Field(..., description="Path to prioritized DEGs CSV file.")
    pathway_path: str = Field(..., description="Path to Pathways Consolidated CSV file.")
    disease: str = Field(..., description="Disease name for analysis context.")
    dep_map_addons: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional DEPMAP configuration (mode_model, genes_selection, top_up, top_down)."
    )
    l1000_addons: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional L1000 configuration (tissue, drug, time_points, cell_lines)."
    )


class PerturbationPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool.
    """
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]


# ---------------------------------------------------------------------------
# 2. Context object to track retries / diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PerturbationRunnerContext:
    """
    Context type passed into Runner.run(..., context=...).
    Not visible to the LLM, only for Python code.
    """
    max_attempts: int = 3
    attempts: int = 0
    last_args: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Function tool that wraps run_full_pipeline
# ---------------------------------------------------------------------------

@function_tool
def run_perturbation_pipeline_tool(
    output_dir: str,
    deg_path: str,
    pathway_path: str,
    disease: str,
) -> PerturbationPipelineToolResult:
    """
    Safe wrapper around run_full_pipeline that always returns a structured result.
    
    The LLM sees this as a tool with strongly-typed arguments.
    
    Note: dep_map_addons and l1000_addons use default values (None) internally.
    For advanced configuration, modify the pipeline directly.
    """
    logger.info(f"[Tool Called] run_perturbation_pipeline_tool invoked!")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  deg_path: {deg_path}")
    logger.info(f"  pathway_path: {pathway_path}")
    logger.info(f"  disease: {disease}")
    
    args: Dict[str, Any] = {
        "output_dir": output_dir,
        "deg_path": deg_path,
        "pathway_path": pathway_path,
        "disease": disease,
    }

    try:
        # Validate paths exist
        if not Path(deg_path).exists():
            raise FileNotFoundError(f"DEGs file not found: {deg_path}")
        if not Path(pathway_path).exists():
            raise FileNotFoundError(f"Pathways file not found: {pathway_path}")
        
        # Use default addons (empty dicts with default values) - required by run_full_pipeline
        # These match the defaults defined in run_full_pipeline.py
        # When None is passed, run_full_pipeline uses its own defaults, but we need to pass dicts
        # to avoid TypeError when accessing keys
        dep_map_addons = {
            "mode_model": None,
            "genes_selection": "all",
            "top_up": None,
            "top_down": None,
        }
        l1000_addons = {
            "tissue": None,
            "drug": None,
            "time_points": None,
            "cell_lines": None,
        }
        
        # Delegate to optimized pipeline (parallel execution)
        result = run_full_pipeline(
            raw_deg_path=Path(deg_path),
            pathway_path=Path(pathway_path),
            output_dir=Path(output_dir),
            disease=disease,
            dep_map_addons=dep_map_addons,
            l1000_addons=l1000_addons,
            parallel=True,  # Enable parallel execution (30-40% faster)
        )
        
        # Return payload with optimization metadata
        payload = {
            "output_dir": str(output_dir),
            "status": result.get("status"),
            "parallel": result.get("parallel"),
            "results": result.get("results")
        }
        
        return PerturbationPipelineToolResult(ok=True, payload=payload, error=None, args=args)
    except Exception:
        tb = traceback.format_exc()
        return PerturbationPipelineToolResult(ok=False, payload=None, error=tb, args=args)


# ---------------------------------------------------------------------------
# 4. Custom tool_use_behavior to implement self-correcting loop
# ---------------------------------------------------------------------------

async def perturbation_tool_use_behavior(
    context: RunContextWrapper[PerturbationRunnerContext],
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

    tool_output: PerturbationPipelineToolResult = latest_result.output  # type: ignore[assignment]

    if tool_output.ok:
        final_summary = {
            "status": "ok",
            "attempts": ctx.attempts,
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


# ---------------------------------------------------------------------------
# 5. Agent factory
# ---------------------------------------------------------------------------

def build_perturbation_runner_agent() -> Agent[PerturbationRunnerContext]:
    """
    Create the specialized Perturbation Runner Agent.
    
    - Forces tool use with tool_choice="required"
    - Uses custom perturbation_tool_use_behavior
    - Uses run_perturbation_pipeline_tool as the only tool
    """
    instructions = """
        You are a Perturbation Analysis Runner Agent.

        Your job:
        1. Read the user-provided arguments for the perturbation pipeline.
        2. Validate and, if needed, adjust the arguments before calling the tool:
        - Verify file paths exist and are accessible.
        - If paths are relative, resolve them using output_dir as base.
        - If dep_map_addons or l1000_addons are invalid, use defaults (None).
        3. Call the `run_perturbation_pipeline_tool` to execute the pipeline.
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

    # Verify tool is properly decorated
    logger.info(f"[Agent Build] Tool object type: {type(run_perturbation_pipeline_tool)}")
    logger.info(f"[Agent Build] Tool object: {run_perturbation_pipeline_tool}")
    
    agent = Agent[PerturbationRunnerContext](
        name="Perturbation Runner Agent",
        instructions=instructions.strip(),
        model="gpt-4.1-mini",
        tools=[run_perturbation_pipeline_tool],
        tool_use_behavior=perturbation_tool_use_behavior,
        model_settings=ModelSettings(
            tool_choice="required",
            temperature=0.0,
        ),
    )
    
    logger.info(f"[Agent Build] Agent created with {len(agent.tools)} tool(s)")
    
    return agent


# ---------------------------------------------------------------------------
# 6. High-level helper to run the agent from Python/CLI
# ---------------------------------------------------------------------------

async def run_perturbation_agent_with_args(
    args: PerturbationAnalysisArgs,
    max_attempts: int = 3,
) -> Any:
    """
    High-level async entrypoint: runs the Perturbation Runner Agent with the given args.
    Returns the final_output from Runner.run(..).
    """
    agent = build_perturbation_runner_agent()
    context = PerturbationRunnerContext(max_attempts=max_attempts)

    import json
    user_input = (
        f"Run perturbation analysis pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )

    logger.info(f"[Perturbation Agent] Starting with user_input:\n{user_input}")
    logger.info(f"[Perturbation Agent] Agent config: model={agent.model}, tool_choice={agent.model_settings.tool_choice if agent.model_settings else 'None'}")
    logger.info(f"[Perturbation Agent] Tools: {[t.name if hasattr(t, 'name') else str(t) for t in agent.tools]}")
    
    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
    )

    logger.info(f"[Perturbation Agent] Runner result type: {type(result)}")
    logger.info(f"[Perturbation Agent] Runner result.final_output type: {type(result.final_output)}")
    logger.info(f"[Perturbation Agent] Runner result.final_output content: {result.final_output}")
    
    # Check if we got unexpected string response
    if isinstance(result.final_output, str):
        logger.error(
            f"[Perturbation Agent] ❌ Agent returned STRING instead of calling tool!\n"
            f"This usually means:\n"
            f"  1. Tool calling failed (check API key, model compatibility)\n"
            f"  2. Model doesn't support function calling\n"
            f"  3. tool_choice='required' not working\n"
            f"Response: {result.final_output[:500]}"
        )
        # Return error dict instead of string
        return {
            "status": "failed",
            "attempts": 0,
            "errors": [f"Agent returned text instead of calling tool: {result.final_output[:200]}"],
            "last_args": args.model_dump(),
        }

    return result.final_output

# ---------------------------------------------------------------------------
# 7. CLI entrypoint
# ---------------------------------------------------------------------------

async def _main_async() -> None:
    perturbation_args = PerturbationAnalysisArgs(
        output_dir="agentic_ai_wf/perturbation_pipeline_agent/output",
        deg_path="agentic_ai_wf/perturbation_pipeline_agent/input_data/ige-mediated_asthma_DEGs_prioritized.csv",
        pathway_path="agentic_ai_wf/perturbation_pipeline_agent/input_data/ige-mediated_asthma_Pathways_Consolidated.csv",
        disease="Ige-mediated Asthma",
    )

    final_output = await run_perturbation_agent_with_args(
        args=perturbation_args,
        max_attempts=3,
    )

    import json
    print(json.dumps(final_output, indent=2))


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
