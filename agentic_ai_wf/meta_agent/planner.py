"""AI-driven plan generation from user queries."""

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .prompts import HUMAN_PLANNER_APPEND, SYSTEM_PROMPT_META_ORCHESTRATOR
from .state import AVAILABLE_NODES, OUTPUT_TO_NODE
from ..langgraph_wf import NODE_DEPENDENCIES

logger = logging.getLogger(__name__)


async def plan_steps(
    user_query: str,
    current_state: Optional[dict] = None,
) -> list[str]:
    """
    AI-driven planner that infers analytical intent and expands dependencies.
    Uses reasoning from the LLM (not heuristics) to form minimal, dependency-complete plans.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_META_ORCHESTRATOR),
        ("human", HUMAN_PLANNER_APPEND),
    ])
    chain = prompt | llm | StrOutputParser()

    state_keys_str = json.dumps(list(current_state.keys()) if current_state else [])
    deps_str = json.dumps(NODE_DEPENDENCIES, indent=2)
    outputs_str = json.dumps(OUTPUT_TO_NODE, indent=2)

    raw = await chain.ainvoke({
        "user_query": user_query,
        "state_keys": state_keys_str,
        "dependencies": deps_str,
        "output_map": outputs_str,
    })

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list) or not all(isinstance(x, str) for x in plan):
            raise ValueError("Invalid JSON array")
        logger.info("LLM-generated plan: %s", plan)
        return plan
    except Exception as e:
        logger.warning("Invalid LLM plan output: %s", e)
        try:
            corrected = await llm.ainvoke(
                f"The previous output was invalid JSON: {raw}. "
                "Return ONLY a valid JSON array of node names (strings), no prose."
            )
            plan = json.loads(corrected.content)
            logger.info("Corrected plan: %s", plan)
            return plan
        except Exception:
            logger.warning("Fallback: using default safe chain")
            return ["deg_analysis", "gene_prioritization", "pathway_enrichment"]
