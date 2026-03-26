"""Meta agent orchestration loop: plan → execute → evaluate → replan."""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .evaluator import (
    augment_state_summary_for_eval,
    evaluate_run,
    _parse_json_strict,
)
from .extractions import (
    extract_deconvolution_technique,
    extract_biosample_from_query,
    extract_gse_ids_from_query,
    extract_disease_from_query,
    extract_is_causal_from_query,
)
from .graph_builder import build_subgraph_from_plan
from .planner import plan_steps
from .state import AVAILABLE_NODES, MetaState
from .state_initializer import discover_files_from_analysis_dir
from ..workflow_context import set_workflow_logger, reset_workflow_logger

logger = logging.getLogger(__name__)


async def run_meta_agent(
    user_query: str,
    *,
    analysis_id: Optional[str] = None,
    disease_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    analysis_transcriptome_dir: Optional[str] = None,
    user_id: str = "system",
    max_attempts: int = 3,
    approved_plan: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Main meta orchestration loop: plan, execute, evaluate, replan until success or max_attempts."""
    from agentic_ai_wf.helpers.logging_utils import AsyncWorkflowLogger

    workflow_id = f"META-{uuid.uuid4().hex[:8]}"
    analysis_id = analysis_id or str(uuid.uuid4())
    workflow_logger = AsyncWorkflowLogger(
        user_id=user_id,
        analysis_id=analysis_id,
        workflow_name="Meta Agent Planning & Execution",
        total_steps=25,
        task_id=kwargs.get("task_id", ""),
        correlation_id=workflow_id,
    )

    await workflow_logger.start_workflow()
    await workflow_logger.info(
        agent_name="Meta Agent",
        message=f"User query: {user_query}",
        step="planning",
    )

    prioritized_genes_path_from_files, pathway_consolidation_path_from_files = (
        await discover_files_from_analysis_dir(
            analysis_transcriptome_dir,
            kwargs.get("turn_id"),
            workflow_logger,
        )
    )

    if approved_plan:
        plan = approved_plan
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Using approved plan: {plan}",
            step="planning",
        )
        logger.info("Using pre-approved plan: %s", plan)
    else:
        plan = await plan_steps(user_query)
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Generated plan via LLM: {plan}",
            step="planning",
        )
        logger.info("LLM-generated plan: %s", plan)

    plan_history = [plan]
    attempt = 0

    deconvolution_technique = None
    if "deconvolution" in plan:
        deconvolution_technique = await extract_deconvolution_technique(user_query)
        if deconvolution_technique:
            await workflow_logger.info(
                agent_name="Meta Agent",
                message=f"Detected deconvolution technique: {deconvolution_technique}",
                step="planning",
            )
        else:
            await workflow_logger.info(
                agent_name="Meta Agent",
                message="No specific deconvolution technique, will use default (xcell)",
                step="planning",
            )

    if not disease_name or (isinstance(disease_name, str) and not disease_name.strip()):
        extracted_disease = await extract_disease_from_query(user_query)
        if extracted_disease:
            disease_name = extracted_disease
            await workflow_logger.info(
                agent_name="Meta Agent",
                message=f"Extracted disease: {disease_name}",
                step="planning",
            )
        else:
            await workflow_logger.info(
                agent_name="Meta Agent",
                message="No disease mentioned in query",
                step="planning",
            )

    extracted_gse_ids = await extract_gse_ids_from_query(user_query)
    if extracted_gse_ids:
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Extracted GSE IDs: {extracted_gse_ids}",
            step="planning",
        )

    crispr_targeted_params = None
    if "crispr_targeted_analysis" in plan:
        from agentic_ai_wf.crispr_targeted_extraction import extract_crispr_targeted_params_async

        crispr_targeted_params = await extract_crispr_targeted_params_async(user_query)
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Extracted targeted CRISPR params: project={crispr_targeted_params.get('project_id')}, gene={crispr_targeted_params.get('target_gene')}",
            step="planning",
        )

    gwas_mr_biosample = None
    if "gwas_mr_analysis" in plan:
        gwas_mr_biosample = extract_biosample_from_query(user_query)
        if gwas_mr_biosample:
            await workflow_logger.info(
                agent_name="Meta Agent",
                message=f"Extracted GWAS-MR biosample: {gwas_mr_biosample}",
                step="planning",
            )

    crispr_screening_params = None
    if "crispr_screening_analysis" in plan:
        from agentic_ai_wf.crispr_screening_extraction import extract_crispr_screening_params_async

        crispr_screening_params = await extract_crispr_screening_params_async(user_query)
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Extracted CRISPR screening params: modes={crispr_screening_params.get('modes')}",
            step="planning",
        )

    is_causal = False
    if any(step in plan for step in ["gene_prioritization", "pathway_enrichment"]):
        is_causal = await extract_is_causal_from_query(user_query)
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Causal analysis detected: {is_causal}",
            step="planning",
        )

    meta_state: MetaState = {
        "user_query": user_query,
        "plan": plan,
        "plan_history": plan_history,
        "attempt": attempt,
        "done": False,
        "analysis_id": analysis_id,
        "disease_name": disease_name or "",
        "patient_id": patient_id or "",
        "patient_name": patient_name or "",
        "user_id": user_id,
        "analysis_transcriptome_dir": analysis_transcriptome_dir or "",
        "workflow_id": workflow_id,
        "run_id": str(uuid.uuid4()),
        "start_time": time.time(),
        "current_step": 0,
        "total_steps": 8,
        "errors": [],
        "failed_steps": [],
        "retry_count": 0,
        "step_start_times": {},
        "step_durations": {},
        "enable_cleanup": kwargs.get("enable_cleanup", True),
        "enable_progress_tracking": kwargs.get("enable_progress_tracking", True),
        "max_retries": kwargs.get("max_retries", 1),
        "workflow_failed": False,
        "failure_reason": None,
        "memory": [],
        "deconvolution_technique": deconvolution_technique,
        "biosample_type": gwas_mr_biosample,
        "prioritized_genes_path": prioritized_genes_path_from_files,
        "pathway_consolidation_path": pathway_consolidation_path_from_files,
        "gse_ids": extracted_gse_ids if extracted_gse_ids else None,
        "is_causal": is_causal,
    }
    if crispr_targeted_params:
        meta_state["crispr_targeted_project_id"] = crispr_targeted_params.get("project_id") or ""
        meta_state["crispr_targeted_target_gene"] = crispr_targeted_params.get("target_gene") or ""
        meta_state["crispr_targeted_protospacer"] = crispr_targeted_params.get("protospacer") or "GGTGGATCCTATTCTAAACG"
        meta_state["crispr_targeted_region"] = crispr_targeted_params.get("region") or ""
        meta_state["crispr_targeted_reference_seq"] = crispr_targeted_params.get("reference_seq") or ""
        meta_state["crispr_targeted_extract_metadata"] = crispr_targeted_params.get("extract_metadata", True)
        meta_state["crispr_targeted_download_fastq"] = crispr_targeted_params.get("download_fastq", True)
    if crispr_screening_params:
        meta_state["crispr_screening_modes"] = crispr_screening_params.get("modes", [3])
        meta_state["crispr_screening_input_dir"] = crispr_screening_params.get("input_dir") or ""
        meta_state["crispr_screening_generate_report"] = crispr_screening_params.get("generate_report", True)

    while attempt < max_attempts:
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Attempt {attempt + 1}: executing plan → {plan}",
            step="execution",
        )
        logger.info("Attempt %d: executing plan → %s", attempt + 1, plan)

        if not plan:
            logger.warning("Plan is empty. Requesting LLM to normalize.")
            eval_result = await evaluate_run(user_query, ["finalization"], meta_state)
            plan = [p for p in eval_result.get("proposed_changes", []) if p in AVAILABLE_NODES]
            if not plan:
                raise ValueError("Evaluator returned no executable steps; aborting.")

        app = build_subgraph_from_plan(plan)
        config = {"configurable": {"thread_id": meta_state["analysis_id"], "workflow_logger": workflow_logger}}
        set_workflow_logger(workflow_logger)
        try:
            final_state: MetaState = await app.ainvoke(meta_state, config=config)
        finally:
            reset_workflow_logger()

        if final_state.get("trigger_replan"):
            plan = final_state["new_plan"]
            plan_history.append(plan)
            meta_state["plan"] = plan
            logger.info("Auto-replan triggered → New plan: %s", plan)
            continue

        attempt += 1
        final_state["attempt"] = attempt

        if "memory" not in final_state:
            final_state["memory"] = meta_state.get("memory", [])
        final_state["memory"].append({
            "attempt": attempt,
            "plan": plan,
            "errors": final_state.get("errors", []),
            "failed_steps": final_state.get("failed_steps", []),
            "failure_reason": final_state.get("failure_reason"),
        })
        meta_state["memory"] = final_state["memory"]

        if approved_plan:
            await workflow_logger.info(
                agent_name="Meta Agent",
                message="Using approved plan - skipping evaluation",
                step="completion",
            )
            await workflow_logger.complete_workflow()
            return {"state": final_state, "plan": plan, "attempts": attempt, "status": "completed"}

        await workflow_logger.info(
            agent_name="Meta Agent",
            message="Evaluating run outcome via LLM reasoning",
            step="evaluation",
        )
        eval_result = await evaluate_run(user_query, plan, final_state)
        ok = eval_result.get("ok", False)
        reason = eval_result.get("reason", "unknown")
        proposed = eval_result.get("proposed_changes") or []

        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Evaluation → ok={ok}, reason={reason}",
            step="evaluation",
        )
        await workflow_logger.info(
            agent_name="Meta Agent",
            message=f"Proposed next plan → {proposed}",
            step="evaluation",
        )
        logger.info("Evaluation → ok=%s, reason=%s", ok, reason)

        if ok:
            await workflow_logger.complete_workflow(
                elapsed_ms=int((time.time() - meta_state["start_time"]) * 1000),
            )
            final_state["done"] = True
            return {
                "success": True,
                "plan": plan,
                "plan_history": plan_history,
                "state": final_state,
                "attempts": attempt,
            }

        next_plan = [p for p in proposed if p in AVAILABLE_NODES]
        if not next_plan:
            logger.warning("No valid steps from evaluator. Requesting normalization.")
            norm_chain = (
                ChatPromptTemplate.from_template(
                    "Return ONLY JSON with a non-empty 'proposed_changes' array containing only valid step names: "
                    '["cohort_retrieval","deg_analysis","gene_prioritization","pathway_enrichment","drug_discovery",'
                    '"deconvolution","temporal_analysis","perturbation_analysis","harmonization","multiomics",'
                    '"mdp_analysis","ipaa_analysis","single_cell","fastq_analysis","crispr_analysis",'
                    '"clinical_report","pharma_report","finalization"]. Input plan: {plan} State summary: {state_summary} JSON only.'
                )
                | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
                | StrOutputParser()
            )
            state_summary = json.dumps(augment_state_summary_for_eval(plan, final_state), indent=2)
            raw = await norm_chain.ainvoke({"plan": json.dumps(plan), "state_summary": state_summary})
            normalized = _parse_json_strict(raw)
            next_plan = [p for p in normalized.get("proposed_changes", []) if p in AVAILABLE_NODES]

        if not next_plan:
            logger.error("Evaluator could not propose executable plan. Terminating.")
            break

        if next_plan in plan_history:
            logger.warning("Plan repetition detected — terminating.")
            break

        plan = next_plan
        plan_history.append(plan)
        meta_state = final_state
        meta_state["plan"] = plan
        meta_state["plan_history"] = plan_history

    failure_reason = meta_state.get("failure_reason", "Maximum attempts reached")
    await workflow_logger.fail_workflow(error_message=failure_reason, error_code="MAX_ATTEMPTS_REACHED")
    return {
        "success": False,
        "plan": plan,
        "plan_history": plan_history,
        "state": meta_state,
        "reason": failure_reason,
        "attempts": attempt,
    }
