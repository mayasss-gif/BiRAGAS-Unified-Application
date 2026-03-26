"""Node dependency validation and auto-replan."""

import logging
from typing import Any

from .state import NODE_DEPENDENCIES

logger = logging.getLogger(__name__)


async def validate_state_and_autoplan(
    node_name: str, state: dict, planner_agent: Any
) -> list:
    """Validate state satisfies node prerequisites; replan if missing."""
    deps = NODE_DEPENDENCIES.get(node_name, {})
    missing = [req for req in deps.get("requires", []) if req not in state or not state[req]]

    if missing:
        plan_prompt = f"""
        The node '{node_name}' cannot start because these required state keys are missing: {missing}.
        Please expand the plan to include any nodes that produce them, respecting dependency order.
        Current state keys: {list(state.keys())}
        Return only the correct node sequence as a JSON list.
        """
        if planner_agent:
            new_plan = await planner_agent.run(plan_prompt)
            return new_plan
        logger.warning(f"No planner agent provided — cannot replan for missing {missing}.")
    return []
