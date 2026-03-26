"""Logging decorator for workflow nodes."""

import logging
import time
from typing import Any, Dict

from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

from .state import TranscriptomeAnalysisState
from .validation import validate_state_and_autoplan

logger = logging.getLogger(__name__)


def create_logged_node(node_name: str, step_number: int):
    """Decorator adding logging and error handling to nodes."""

    def decorator(node_func):
        async def wrapped_node(
            state: TranscriptomeAnalysisState, config: Dict[str, Any] = None
        ) -> TranscriptomeAnalysisState:
            step_start = time.time()
            workflow_logger = None
            if config:
                workflow_logger = config.get("configurable", {}).get("workflow_logger")
            if not workflow_logger:
                workflow_logger = _get_workflow_logger()
            agent_name = node_name.replace("_", " ").title() + " Agent"

            try:
                if workflow_logger:
                    await workflow_logger.start_step(
                        step=node_name, agent_name=agent_name, step_number=step_number
                    )
                else:
                    logger.info(f"[{node_name}] Starting (no logger)")

                planner_agent = state.get("planner_agent")
                if planner_agent:
                    new_plan = await validate_state_and_autoplan(
                        node_name, state, planner_agent
                    )
                    if new_plan:
                        if workflow_logger:
                            await workflow_logger.warning(
                                agent_name=agent_name,
                                message=f"Missing dependencies. Triggering replan: {new_plan}",
                                step=node_name,
                            )
                        return {"trigger_replan": True, "new_plan": new_plan}

                nodes_with_config = (
                    "deg_analysis",
                    "gene_prioritization",
                    "pathway_enrichment",
                    "deconvolution",
                    "temporal_analysis",
                )
                if node_name in nodes_with_config and config:
                    result = await node_func(state, config)
                else:
                    result = await node_func(state)

                duration = time.time() - step_start
                elapsed_ms = int(duration * 1000)
                step_durations = state.get("step_durations", {})
                step_durations[node_name] = duration

                if workflow_logger:
                    await workflow_logger.complete_step(
                        step=node_name,
                        agent_name=agent_name,
                        step_number=step_number,
                        elapsed_ms=elapsed_ms,
                    )
                else:
                    logger.info(f"[{node_name}] Completed in {duration:.2f}s")

                return {**result, "step_durations": step_durations}

            except Exception as e:
                if workflow_logger:
                    await workflow_logger.error(
                        agent_name=agent_name,
                        message=f"Failed: {str(e)}",
                        step=node_name,
                        exception=e,
                        error_message=str(e),
                    )
                else:
                    logger.error(f"[{node_name}] Failed: {str(e)}", exc_info=True)

                errors = state.get("errors", [])
                errors.append({"step": node_name, "error": str(e), "timestamp": time.time()})
                failed_steps = state.get("failed_steps", [])
                failed_steps.append(node_name)
                return {"errors": errors, "failed_steps": failed_steps}

        return wrapped_node

    return decorator
