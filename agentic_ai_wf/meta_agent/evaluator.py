"""LLM-driven workflow evaluation and reflection."""

import json
import logging
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .prompts import evaluator_prompt
from .state import MetaState

logger = logging.getLogger(__name__)

STEP_OUTPUTS = {
    "cohort_retrieval": "cohort_output_dir",
    "deg_analysis": "deg_analysis_result",
    "gene_prioritization": "prioritized_genes_path",
    "pathway_enrichment": "pathway_consolidation_path",
    "drug_discovery": "drug_discovery_path",
    "deconvolution": "deconvolution_output_dir",
    "temporal_analysis": "temporal_analysis_output_dir",
    "perturbation_analysis": "perturbation_analysis_output_dir",
    "harmonization": "harmonization_output_dir",
    "multiomics": "multiomics_output_dir",
    "mdp_analysis": "mdp_output_dir",
    "ipaa_analysis": "ipaa_output_dir",
    "fastq_analysis": "fastq_analysis_output_dir",
    "crispr_analysis": "crispr_output_dir",
    "gwas_mr_analysis": "gwas_mr_output_dir",
    "clinical_report": "clinical_report_path",
    "pharma_report": "pharma_report_path",
}

VALID_STEPS = [
    "cohort_retrieval", "deg_analysis", "gene_prioritization", "pathway_enrichment",
    "drug_discovery", "deconvolution", "temporal_analysis", "perturbation_analysis",
    "harmonization", "multiomics", "mdp_analysis", "ipaa_analysis", "single_cell",
    "fastq_analysis", "crispr_analysis", "clinical_report", "pharma_report", "finalization",
]


def summarize_state_for_eval(plan: list[str], state: MetaState) -> Dict[str, Any]:
    """Build summary of completed/failed steps and available artifacts for evaluator."""
    failed = set(state.get("failed_steps", []))
    completed = []
    for step in plan:
        if step in failed:
            continue
        output_key = STEP_OUTPUTS.get(step)
        if output_key and state.get(output_key):
            completed.append(step)
        elif step in ("finalization", "error_handler"):
            completed.append(step)

    return {
        "errors": state.get("errors", []),
        "failed_steps": list(failed),
        "completed_steps": completed,
        "have": {
            "cohort_output_dir": bool(state.get("cohort_output_dir")),
            "deg": bool(state.get("deg_analysis_result")),
            "prioritized_genes": bool(state.get("prioritized_genes_path")),
            "pathway_consolidation": bool(state.get("pathway_consolidation_path")),
            "drug_discovery": bool(state.get("drug_discovery_path")),
            "deconvolution": bool(state.get("deconvolution_output_dir")),
            "temporal_analysis": bool(state.get("temporal_analysis_output_dir")),
            "perturbation_analysis": bool(state.get("perturbation_analysis_output_dir")),
            "harmonization": bool(state.get("harmonization_output_dir")),
            "multiomics": bool(state.get("multiomics_output_dir")),
            "mdp_analysis": bool(state.get("mdp_output_dir")),
            "ipaa_analysis": bool(state.get("ipaa_output_dir")),
            "single_cell": bool(state.get("single_cell_output_dir")),
            "fastq_analysis": bool(state.get("fastq_analysis_output_dir")),
            "crispr_analysis": bool(state.get("crispr_output_dir")),
            "gwas_mr_analysis": bool(state.get("gwas_mr_output_dir")),
            "clinical_report": bool(state.get("clinical_report_path")),
            "pharma_report": bool(state.get("pharma_report_path")),
        },
    }


def _parse_json_strict(text: str) -> dict:
    """Parse JSON with minimal repair for LLM noise."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Invalid JSON from evaluator")


def augment_state_summary_for_eval(plan: list[str], state: MetaState) -> dict:
    """Add last_error to state summary."""
    state_summary = summarize_state_for_eval(plan, state)
    last_err = None
    if state.get("errors"):
        last_err = state["errors"][-1]
    elif state.get("failure_reason"):
        last_err = state["failure_reason"]
    state_summary["failed_steps"] = state.get("failed_steps", [])
    state_summary["last_error"] = last_err
    return state_summary


async def evaluate_run(user_query: str, plan: list[str], state: MetaState) -> Dict[str, Any]:
    """LLM-only reflection. No rule heuristics for dependencies."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chain = evaluator_prompt() | llm | StrOutputParser()
    state_summary = augment_state_summary_for_eval(plan, state)

    raw = await chain.ainvoke({
        "user_query": user_query,
        "plan": json.dumps(plan, indent=2),
        "state_summary": json.dumps(state_summary, indent=2),
    })

    try:
        result = _parse_json_strict(raw)
    except Exception:
        strict_chain = ChatPromptTemplate.from_template(
            "Return ONLY valid JSON per this schema, no prose:\n{schema}\n\nGiven plan:\n{plan}\n\nGiven state_summary:\n{state_summary}\n"
        ) | llm | StrOutputParser()
        schema = '{ "ok": true|false, "reason": "string", "proposed_changes": ["valid step names, non-empty when ok=false"] }'
        raw2 = await strict_chain.ainvoke({
            "schema": schema,
            "plan": json.dumps(plan, indent=2),
            "state_summary": json.dumps(state_summary, indent=2),
        })
        result = _parse_json_strict(raw2)

    if not result.get("ok", False):
        pcs = result.get("proposed_changes", [])
        if not isinstance(pcs, list) or len(pcs) == 0:
            enforce_chain = ChatPromptTemplate.from_template("""
                Return ONLY JSON. Provide a minimal NON-EMPTY 'proposed_changes' array of valid step names
                from ["cohort_retrieval","deg_analysis","gene_prioritization","pathway_enrichment","drug_discovery","deconvolution","temporal_analysis","perturbation_analysis","harmonization","multiomics","mdp_analysis","ipaa_analysis","single_cell","fastq_analysis","crispr_analysis","clinical_report","pharma_report","finalization"]
                that fixes the failure. JSON only:
                {{"ok": false, "reason": "brief", "proposed_changes": [...]}}
                State summary: {state_summary}
                Current plan: {plan}
            """) | llm | StrOutputParser()
            raw3 = await enforce_chain.ainvoke({
                "state_summary": json.dumps(state_summary, indent=2),
                "plan": json.dumps(plan, indent=2),
            })
            result = _parse_json_strict(raw3)

    return result
