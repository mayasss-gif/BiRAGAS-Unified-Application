"""Dynamic subgraph builder from plan steps."""

import logging

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)
from langgraph.checkpoint.memory import MemorySaver

from .state import AVAILABLE_NODES, MetaState
from ..langgraph_wf import should_continue_after_error


def build_subgraph_from_plan(plan: list[str]):
    """
    Build minimal runnable graph for selected steps.

    ``error_handler`` is only reachable from ``deg_analysis`` when
    ``should_continue_after_error`` returns ``abort``. It must not appear in the
    main linear chain — otherwise the last real step (e.g. ``perturbation_analysis``)
    would always route to ``error_handler``, which sets ``workflow_failed`` even on success.
    """
    plan = list(plan)
    if "deg_analysis" in plan and "error_handler" not in plan:
        plan = plan + ["error_handler"]

    logger.info("Building subgraph for plan: %s", plan)
    g = StateGraph(MetaState)
    for p in plan:
        g.add_node(p, AVAILABLE_NODES[p])
    g.set_entry_point(plan[0])

    # Main chain: user steps only (never wire the last step -> error_handler sequentially)
    linear_steps = [p for p in plan if p != "error_handler"]

    i = 0
    while i < len(linear_steps) - 1:
        curr, nxt = linear_steps[i], linear_steps[i + 1]
        if curr == "deg_analysis":
            g.add_conditional_edges(
                "deg_analysis",
                should_continue_after_error,
                {"continue": nxt, "abort": "error_handler"},
            )
        else:
            g.add_edge(curr, nxt)
        i += 1

    # Terminate after the last analysis step
    if linear_steps:
        last = linear_steps[-1]
        if len(linear_steps) == 1 and last == "deg_analysis":
            g.add_conditional_edges(
                "deg_analysis",
                should_continue_after_error,
                {"continue": END, "abort": "error_handler"},
            )
        else:
            g.add_edge(last, END)

    if "drug_discovery" in plan:
        reports = [p for p in plan if p in {"clinical_report", "pharma_report"}]
        if set(reports) == {"clinical_report", "pharma_report"}:
            g.add_edge("drug_discovery", "clinical_report")
            g.add_edge("drug_discovery", "pharma_report")
            if "finalization" in plan:
                g.add_edge("clinical_report", "finalization")
                g.add_edge("pharma_report", "finalization")

    if "error_handler" in plan:
        g.add_edge("error_handler", END)

    return g.compile(checkpointer=MemorySaver())
