from __future__ import annotations

import os
import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import agentic_ai_wf.mdp_pipeline_agent.multi_pathway.full_main_orchestrator as fmo
import subprocess
import sys
from threading import Thread

from pydantic import BaseModel, Field, ConfigDict
from decouple import config
from langchain_openai import ChatOpenAI

from agents import (
    Agent,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
)

from .multi_pathway import run_full_pipeline

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


async def extract_mdp_diseases_and_files(
    user_query: str,
    available_files: List[str],
) -> Dict[str, Any]:
    """
    Extract multiple diseases from query and match files using LLM agent.
    
    Returns structured data with diseases and file assignments.
    
    Args:
        user_query: User's natural language query
        available_files: List of available file paths
        
    Returns:
        Dict with:
            - diseases: List[str] - extracted disease names
            - file_assignments: Dict[str, str] - disease -> file_path mapping
            - items: List[str] - formatted items for run_full_pipeline
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prepare file context
        file_context = ""
        if available_files:
            file_names = [Path(f).name for f in available_files]
            file_context = f"""
Available Files:
{chr(10).join(f"- {name} (path: {path})" for name, path in zip(file_names, available_files))}
"""
        else:
            file_context = "No files available."
        
        extraction_prompt = f"""Extract ALL diseases/conditions from the query and match them to available files.

            TASK: 
            1. Identify ALL diseases, conditions, or phenotypes mentioned in the query
            2. Match files to diseases based on filename similarity and context
            3. Return structured JSON

            RULES:
            - Extract ALL diseases mentioned (not just primary)
            - Use standard medical terminology
            - Match files to diseases using filename keywords and context
            - If a file matches a disease, assign it; otherwise disease uses GeneCards/KG
            - Return JSON format: {{"diseases": ["disease1", "disease2"], "assignments": {{"disease1": "file_path_or_null", "disease2": "file_path_or_null"}}}}

            EXAMPLES:

            Query: "Run MDP for lupus using uploaded file and breast cancer"
            Files: ["lupus_counts.csv", "metadata.csv"]
            Answer:
            {{
            "diseases": ["lupus", "breast cancer"],
            "assignments": {{
                "lupus": "lupus_counts.csv",
                "breast cancer": null
            }}
            }}

            Query: "Analyze pathways for asthma and vasculitis"
            Files: ["asthma_data.csv", "vasculitis_genes.txt"]
            Answer:
            {{
            "diseases": ["asthma", "vasculitis"],
            "assignments": {{
                "asthma": "asthma_data.csv",
                "vasculitis": "vasculitis_genes.txt"
            }}
            }}

            Query: "MDP analysis for Alzheimer's disease"
            Files: ["alzheimer_deg.csv", "control_data.csv"]
            Answer:
            {{
            "diseases": ["Alzheimer's disease"],
            "assignments": {{
                "Alzheimer's disease": "alzheimer_deg.csv"
            }}
            }}

            USER QUERY: "{user_query}"
            {file_context}

            Return ONLY valid JSON (no markdown, no code blocks):"""

        response = await llm.ainvoke(extraction_prompt)
        result_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        if result_text.endswith("```"):
            result_text = result_text.rsplit("```", 1)[0].strip()
        
        # Parse JSON
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {result_text}")
            return {"diseases": [], "assignments": {}, "items": []}
        
        diseases = result.get("diseases", [])
        assignments = result.get("assignments", {})
        
        # Build items list
        items = []
        file_path_map = {Path(f).name: f for f in available_files}
        
        for disease in diseases:
            assigned_file = assignments.get(disease)
            if assigned_file:
                # Find full path from filename
                file_path = file_path_map.get(assigned_file) or assigned_file
                if Path(file_path).exists():
                    items.append(f"name={disease},input={file_path}")
                    logger.info(f"Assigned file {file_path} to disease '{disease}'")
                else:
                    items.append(f"name={disease}")
                    logger.warning(f"File {assigned_file} not found, using GeneCards for '{disease}'")
            else:
                items.append(f"name={disease}")
                logger.info(f"Disease '{disease}' without file (will use GeneCards/KG)")
        
        logger.info(f"MDP agent extracted {len(diseases)} diseases: {diseases}")
        
        return {
            "diseases": diseases,
            "assignments": assignments,
            "items": items
        }
    
    except Exception as e:
        logger.exception(f"MDP disease/file extraction failed: {e}")
        return {"diseases": [], "assignments": {}, "items": []}


def detect_file_type(file_path: Path) -> str:
    """Detect file type: csv_gene, txt_gene, json_gene, or unknown."""
    if not file_path.exists():
        return "unknown"
    
    ext = file_path.suffix.lower()
    
    if ext == ".json":
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and ("gene_symbol" in data or "genes" in data or "data" in data):
                    return "json_gene"
        except Exception:
            pass
        return "json_gene"
    
    if ext == ".txt":
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line and not first_line.startswith("#"):
                    return "txt_gene"
        except Exception:
            pass
        return "txt_gene"
    
    if ext in [".csv", ".tsv"]:
        try:
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(file_path, nrows=5, sep=sep)
            gene_cols = [c for c in df.columns if c.lower() in ["gene", "gene_id", "gene_symbol", "symbol", "gene_name"]]
            if gene_cols:
                return "csv_gene"
        except Exception:
            pass
    
    return "unknown"


def parse_gene_file(file_path: Path, file_type: str) -> List[str]:
    """Parse gene file and return list of gene symbols."""
    genes = []
    
    try:
        if file_type == "txt_gene":
            with open(file_path, "r") as f:
                for line in f:
                    gene = line.strip()
                    if gene and not gene.startswith("#"):
                        genes.append(gene)
        
        elif file_type == "json_gene":
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "gene_symbol" in data:
                        genes = data["gene_symbol"] if isinstance(data["gene_symbol"], list) else [data["gene_symbol"]]
                    elif "genes" in data:
                        genes = data["genes"] if isinstance(data["genes"], list) else [data["genes"]]
                    elif "data" in data and isinstance(data["data"], dict):
                        if "gene_symbol" in data["data"]:
                            genes = data["data"]["gene_symbol"] if isinstance(data["data"]["gene_symbol"], list) else [data["data"]["gene_symbol"]]
        
        elif file_type == "csv_gene":
            try:
                import pandas as pd
                sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(file_path, sep=sep)
                gene_cols = [c for c in df.columns if c.lower() in ["gene", "gene_id", "gene_symbol", "symbol", "gene_name"]]
                if gene_cols:
                    genes = df[gene_cols[0]].dropna().astype(str).tolist()
            except Exception as e:
                logger.warning(f"Error parsing CSV gene file {file_path}: {e}")
    
    except Exception as e:
        logger.warning(f"Error parsing gene file {file_path}: {e}")
    
    return genes


def _resolve_mdp_path(path: str) -> str:
    """Fix Agent path corruption: insert 'media/' before 'agentic_uploads' when missing."""
    p = Path(path).expanduser().resolve()
    if p.exists():
        return str(p)
    s = path.replace("\\", "/")
    # Wrong: .../agenticaib_dev/agentic_uploads/...  Correct: .../agenticaib_dev/media/agentic_uploads/...
    if "/agentic_uploads/" in s and "/media/agentic_uploads/" not in s:
        fixed = s.replace("/agentic_uploads/", "/media/agentic_uploads/")
        if Path(fixed).exists():
            logger.info(f"MDP path fixed (agent corruption): {path} -> {fixed}")
            return fixed
    return path


def group_files_by_disease(
    items: List[str],
    uploaded_files: Optional[List[str]] = None
) -> List[str]:
    """
    Group files by disease name and create item list for run_full_pipeline.
    
    Handles:
    - Multiple diseases with multiple files (groups files by disease)
    - One file with multiple diseases (assigns file to first disease, others name-only)
    - Disease names without files (keeps as name-only)
    - Different file types (CSV, TXT, JSON)
    
    Args:
        items: List of items in format "name=disease,input=path" or "name=disease"
        uploaded_files: Optional list of uploaded file paths to auto-assign
    
    Returns:
        List of formatted items for run_full_pipeline
    """
    disease_files: Dict[str, List[str]] = {}
    disease_only: List[str] = []
    unassigned_files: List[str] = []
    
    # Parse items
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        if "=" in item and "name" in item:
            parts = {}
            for kv in item.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    parts[k.strip()] = v.strip()
            
            disease = parts.get("name", "").strip()
            file_path = parts.get("input", "").strip()
            if file_path:
                file_path = _resolve_mdp_path(file_path)
            if disease:
                if file_path:
                    if disease not in disease_files:
                        disease_files[disease] = []
                    disease_files[disease].append(file_path)
                else:
                    disease_only.append(disease)
        else:
            disease_only.append(item)
    
    # Auto-assign uploaded files to diseases
    if uploaded_files:
        for file_path in uploaded_files:
            file_path = _resolve_mdp_path(file_path)
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                continue
            
            # Try to infer disease from filename or directory
            disease_guess = file_path_obj.stem.replace("_", " ").replace("-", " ")
            
            # Check if any disease matches
            assigned = False
            for disease in list(disease_files.keys()) + disease_only:
                if disease.lower() in disease_guess.lower() or disease_guess.lower() in disease.lower():
                    if disease not in disease_files:
                        disease_files[disease] = []
                    disease_files[disease].append(str(file_path_obj))
                    assigned = True
                    break
            
            if not assigned:
                unassigned_files.append(str(file_path_obj))
    
    # Handle one file with multiple diseases
    if len(unassigned_files) == 1 and disease_only:
        file_path = unassigned_files[0]
        first_disease = disease_only.pop(0)
        if first_disease not in disease_files:
            disease_files[first_disease] = []
        disease_files[first_disease].append(file_path)
        logger.info(f"Auto-assigned file {file_path} to disease '{first_disease}'")
    
    # Build final item list
    result_items = []
    
    # Add diseases with files (use name=disease,input=path format)
    for disease, files in disease_files.items():
        for file_path in files:
            result_items.append(f"name={disease},input={file_path}")
    
    # Add diseases without files (use plain disease name string, not name=disease format)
    # The run_auto_mode parser expects plain strings for disease-only items
    for disease in disease_only:
        result_items.append(disease)  # Plain string, not "name=disease"
    
    # Add unassigned files as standalone (use name=disease,input=path format)
    for file_path in unassigned_files:
        file_path_obj = Path(file_path)
        disease_guess = file_path_obj.stem.replace("_", " ").replace("-", " ")
        result_items.append(f"name={disease_guess},input={file_path}")
    
    return result_items


def _patch_run_cmd_for_logging():
    """
    Patch run_cmd function and print statements in full_main_orchestrator to use logging.
    This ensures Celery workers capture all output in real-time.
    """
    
    # Patch print statements to use logger
    original_print = print
    
    def patched_print(*args, **kwargs):
        """Redirect print to logger for Celery compatibility."""
        # Check if this is stderr output
        if kwargs.get('file') == sys.stderr:
            logger.warning(' '.join(str(arg) for arg in args))
        else:
            logger.info(' '.join(str(arg) for arg in args))
    
    # Patch print in the module
    fmo.print = patched_print
    
    # Patch run_cmd function
    original_run_cmd = fmo.run_cmd
    
    def patched_run_cmd(argv, cwd=None, env=None):
        """Patched run_cmd that streams subprocess output to logger in real-time."""
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        
        logger.info(f"[MDP] Running: {' '.join(argv)}")
        
        # Use Popen for real-time streaming instead of run
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=merged_env,
        )
        
        def stream_output(pipe, log_level):
            """Stream pipe output to logger."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.rstrip()
                        if line:
                            if log_level == 'info':
                                logger.info(f"[MDP subprocess] {line}")
                            else:
                                logger.warning(f"[MDP subprocess] {line}")
                pipe.close()
            except Exception as e:
                logger.error(f"[MDP subprocess] Error reading {log_level}: {e}")
        
        # Start threads to stream stdout and stderr
        stdout_thread = Thread(target=stream_output, args=(proc.stdout, 'info'), daemon=True)
        stderr_thread = Thread(target=stream_output, args=(proc.stderr, 'warning'), daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        returncode = proc.wait()
        
        # Wait for threads to finish reading remaining output
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        
        if returncode != 0:
            logger.error(f"[MDP] Subprocess failed with return code {returncode}")
        
        # Return object-like structure for backward compatibility with existing code
        # that expects res.code attribute access
        return SimpleNamespace(
            code=returncode,
            argv=argv,
            stdout="",  # Already logged
            stderr="",  # Already logged
        )
    
    # Patch the function
    fmo.run_cmd = patched_run_cmd
    logger.debug("Patched run_cmd and print for real-time logging in Celery")


# Patch run_cmd when module is imported
_patch_run_cmd_for_logging()


def mdp_pipeline_direct(
    mode: Optional[str] = None,
    input_path: Optional[str] = None,
    output_dir: str = "",
    disease_name: str = "",
    items: Optional[List[str]] = None,
    cohort: Optional[List[str]] = None,
    tissue: Optional[str] = "",
    degs_extra: Optional[List[str]] = None,
    run_enzymes: Optional[bool] = True,
    run_analyses: Optional[bool] = True,
    analysis: Optional[List[str]] = None,
    sig: Optional[float] = 0.10,
    cap: Optional[int] = 300,
    pathways: Optional[List[str]] = None,
    direction_mode: Optional[str] = "both",
    pc_out: Optional[str] = "PC_out",
    skip_report: Optional[bool] = None,
    report_no_llm: Optional[bool] = None,
    report_q_cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Direct wrapper for MDP pipeline (non-agent mode).
    
    Supports both single-disease and multi-disease (auto) modes.
    For agent-based execution with self-correction, use run_mdp_agent_with_args_sync().
    
    Args:
        mode: Pipeline mode ('counts', 'degs', 'gl', 'gc', or 'auto'). If items provided, defaults to 'auto'.
        input_path: Single input path (single-disease mode)
        output_dir: Output directory
        disease_name: Disease name (single-disease mode)
        items: Multi-disease items list ['name=disease1,input=path1', 'name=disease2', ...]
    """
    # Auto-detect mode: if items provided, use auto mode
    if items:
        mode = "auto"
    
    if mode == "auto" and items:
        return run_full_pipeline(
            mode="auto",
            out_root=Path(output_dir),
            item=items,
            run_enzymes=run_enzymes,
            pathways=pathways or [],
            direction_mode=direction_mode,
            pc_out=pc_out,
            skip_report=skip_report,
            report_no_llm=report_no_llm,
            report_q_cutoff=report_q_cutoff,
        )
    else:
        if not input_path:
            raise ValueError("input_path is required for single-disease mode")
    return run_full_pipeline(
            mode=mode or "counts",
        input=Path(input_path),
        out_root=Path(output_dir),
        disease_name=disease_name,
        cohort=cohort,
        tissue=tissue or "",
        run_enzymes=run_enzymes,
        pathways=pathways or [],
        direction_mode=direction_mode,
        pc_out=pc_out,
        skip_report=skip_report,
        report_no_llm=report_no_llm,
        report_q_cutoff=report_q_cutoff,
    )


class MDPPipelineArgs(BaseModel):
    """
    Arguments for MDP pipeline execution.
    """
    model_config = ConfigDict(extra='forbid')
    
    mode: str = Field(
        ...,
        description="Pipeline mode: 'counts', 'degs', 'gl', 'gc', or 'auto'"
    )
    input_path: Optional[str] = Field(
        default=None,
        description="Input directory or file path for the pipeline (single-disease mode)"
    )
    output_dir: str = Field(
        ...,
        description="Root output directory for all pipeline outputs"
    )
    disease_name: str = Field(
        default="",
        description="Disease name for the pipeline (required for counts mode, optional for auto mode)"
    )
    items: Optional[List[str]] = Field(
        default=None,
        description="Multi-disease mode: list of items in format 'name=disease,input=path' or 'name=disease'"
    )
    uploaded_files: Optional[List[str]] = Field(
        default=None,
        description="List of uploaded file paths to auto-assign to diseases"
    )
    cohort: Optional[List[str]] = Field(
        default=None,
        description="List of cohort identifiers (for counts multi-cohort mode)"
    )
    tissue: Optional[str] = Field(
        default="",
        description="Tissue type (required for counts mode)"
    )
    degs_extra: Optional[List[str]] = Field(
        default_factory=list,
        description="Extra parameters for DEGs mode"
    )
    run_enzymes: bool = Field(
        default=True,
        description="Run enzyme and signaling analysis (counts/degs only)"
    )
    run_analyses: bool = Field(
        default=True,
        description="Run post-classification analysis modules"
    )
    analysis: List[str] = Field(
        default_factory=lambda: ["client_dashboard", "category_landscape", "enzyme_signaling", "pathway_entity", "tf_activity"],
        description="List of analysis modules to run"
    )
    sig: float = Field(
        default=0.10,
        description="Significance threshold for analyses"
    )
    cap: int = Field(
        default=300,
        description="Cap for analysis results"
    )
    pathways: Optional[List[str]] = Field(
        default_factory=list,
        description="List of pathways for pathway_compare (if empty, runs mdp_insights)"
    )
    direction_mode: str = Field(
        default="both",
        description="Direction mode for pathway_compare: 'up', 'down', or 'both'"
    )
    pc_out: str = Field(
        default="PC_out",
        description="Output directory for pathway_compare"
    )
    skip_report: Optional[bool] = Field(
        default=None,
        description="Skip report generation if True"
    )
    report_no_llm: Optional[bool] = Field(
        default=None,
        description="Generate report without LLM if True"
    )
    report_q_cutoff: Optional[float] = Field(
        default=None,
        description="Q-value/FDR cutoff for report theme logic"
    )


class MDPPipelineToolResult(BaseModel):
    """
    Structured result returned by the function tool.
    """
    model_config = ConfigDict(extra='forbid')
    
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    args: Dict[str, Any]


@dataclass
class MDPRunnerContext:
    """
    Context for tracking retries and diagnostics.
    """
    max_attempts: int = 3
    attempts: int = 0
    last_args: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


@function_tool
def run_mdp_pipeline_tool(
    mode: str,
    output_dir: str,
    input_path: Optional[str] = None,
    disease_name: str = "",
    items: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None,
    cohort: Optional[List[str]] = None,
    tissue: Optional[str] = "",
    degs_extra: Optional[List[str]] = None,
    run_enzymes: bool = True,
    run_analyses: bool = True,
    analysis: Optional[List[str]] = None,
    sig: float = 0.10,
    cap: int = 300,
    pathways: Optional[List[str]] = None,
    direction_mode: str = "both",
    pc_out: str = "PC_out",
    skip_report: Optional[bool] = None,
    report_no_llm: Optional[bool] = None,
    report_q_cutoff: Optional[float] = None,
) -> MDPPipelineToolResult:
    """
    Safe wrapper around MDP pipeline that always returns structured result.
    Supports both single-disease and multi-disease (auto) modes.
    """
    args: Dict[str, Any] = {
        "mode": mode,
        "input_path": input_path,
        "output_dir": output_dir,
        "disease_name": disease_name,
        "items": items,
        "uploaded_files": uploaded_files,
        "cohort": cohort,
        "tissue": tissue,
        "degs_extra": degs_extra or [],
        "run_enzymes": run_enzymes,
        "run_analyses": run_analyses,
        "analysis": analysis or ["client_dashboard", "category_landscape", "enzyme_signaling", "pathway_entity", "tf_activity"],
        "sig": sig,
        "cap": cap,
        "pathways": pathways or [],
        "direction_mode": direction_mode,
        "pc_out": pc_out,
        "skip_report": skip_report,
        "report_no_llm": report_no_llm,
        "report_q_cutoff": report_q_cutoff,
    }
    
    try:
        # Validate mode
        valid_modes = ["counts", "degs", "gl", "gc", "auto"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        
        # Ensure output directory exists
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # Multi-disease auto mode
        if mode == "auto" or items:
            if items:
                # Group files by disease and format items
                formatted_items = group_files_by_disease(items, uploaded_files)
                logger.info(f"Multi-disease mode: {len(formatted_items)} items after grouping")
            else:
                formatted_items = []
            
            exit_code = run_full_pipeline(
                mode="auto",
                out_root=output_dir_obj,
                item=formatted_items,
                run_enzymes=run_enzymes,
                pathways=pathways or [],
                direction_mode=direction_mode,
                pc_out=pc_out,
                skip_report=skip_report,
                report_no_llm=report_no_llm,
                report_q_cutoff=report_q_cutoff,
            )
        
        # Single-disease mode
        else:
            if not input_path:
                raise ValueError(f"input_path is required for mode '{mode}'")
            
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        if mode == "counts":
            if not disease_name or not disease_name.strip():
                raise ValueError("disease_name is required for counts mode")
            if not tissue or not tissue.strip():
                logger.warning("tissue is empty for counts mode; continuing but consensus tissue adjustment may skip")
        
            exit_code = run_full_pipeline(
            mode=mode,
            input=input_path_obj,
            out_root=output_dir_obj,
            disease_name=disease_name,
            cohort=cohort,
            tissue=tissue or "",
            run_enzymes=run_enzymes,
            pathways=pathways or [],
            direction_mode=direction_mode,
            pc_out=pc_out,
            skip_report=skip_report,
            report_no_llm=report_no_llm,
            report_q_cutoff=report_q_cutoff,
        )
        
        # Check exit code (0 = success, non-zero = failure)
        if exit_code != 0:
            raise RuntimeError(f"MDP pipeline exited with non-zero code: {exit_code}")
        
        return MDPPipelineToolResult(
            ok=True,
            payload={
                "output_dir": str(output_dir_obj),
                "mode": mode,
                "disease_name": disease_name,
                "exit_code": exit_code,
                "result": "Pipeline completed successfully"
            },
            error=None,
            args=args,
        )
        
    except Exception:
        tb = traceback.format_exc()
        return MDPPipelineToolResult(
            ok=False,
            payload=None,
            error=tb,
            args=args,
        )


async def mdp_tool_use_behavior(
    context: RunContextWrapper[MDPRunnerContext],
    results: List[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """
    Custom tool use behavior with retry logic.
    """
    ctx = context.context
    latest_result: FunctionToolResult = results[-1]
    
    tool_output: MDPPipelineToolResult = latest_result.output
    
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


def build_mdp_runner_agent() -> Agent[MDPRunnerContext]:
    """
    Create the specialized MDP Runner Agent.
    """
    instructions = """
        You are a Multi-Disease Pathways (MDP) Pipeline Runner Agent for bioinformatics pathway analysis.

        Your job:
        1. Read the user-provided arguments for the MDP pipeline.
        2. Determine mode:
           - If 'items' list provided OR multiple diseases mentioned: use mode='auto'
           - Otherwise: use single-disease mode ('counts', 'degs', 'gl', or 'gc')
        3. For auto/multi-disease mode:
           - Format items as: ['name=disease1,input=path1', 'name=disease2', ...]
           - Group files by disease name (same disease = multiple files)
           - If one file + multiple diseases: assign file to first disease, others name-only
           - Auto-detect file types (CSV with gene column, TXT one gene per line, JSON)
        4. Validate and adjust arguments before calling the tool:
           - Ensure mode is valid: 'counts', 'degs', 'gl', 'gc', or 'auto'
           - For single-disease: verify input_path exists and is accessible
           - For auto mode: verify items list is non-empty
           - Ensure output_dir exists or can be created
           - For counts mode: verify disease_name and tissue are provided
           - Handle optional parameters (cohort, pathways, analysis modules, etc.)
           - Fix relative paths using absolute paths or resolve relative to current working directory
        5. Call the `run_mdp_pipeline_tool` to execute the pipeline.
        6. If the tool returns an error:
           - Carefully read the stacktrace
           - Infer which argument(s) caused the problem (file paths, directory paths, permissions, mode selection, etc.)
           - Modify only those arguments, keeping others unchanged
           - Try again, up to the maximum number of attempts
        7. When the pipeline succeeds or attempts are exhausted, return a concise,
           structured JSON summary with:
           - status: "ok" or "failed"
           - attempts
           - args_used (final args)
           - result (the tool's payload if ok)
           - errors (if any)

        Always reason step-by-step internally, but output only the final structured result.
        """
    
    agent = Agent[MDPRunnerContext](
        name="MDP Runner Agent",
        instructions=instructions.strip(),
        model="gpt-4o-mini",
        tools=[run_mdp_pipeline_tool],
        tool_use_behavior=mdp_tool_use_behavior,
        model_settings=ModelSettings(
            tool_choice="required",
            temperature=0.0,
        ),
    )
    return agent


async def run_mdp_agent_with_args(
    args: MDPPipelineArgs,
    max_attempts: int = 3,
) -> Any:
    """
    High-level async entrypoint: runs the MDP Runner Agent with given args.
    """
    agent = build_mdp_runner_agent()
    
    context = MDPRunnerContext(max_attempts=max_attempts)
    
    user_input = (
        f"Run Multi-Disease Pathways (MDP) pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    result = await Runner.run(
        starting_agent=agent,
        input=user_input,
        context=context,
    )
    
    return result.final_output


def run_mdp_agent_with_args_sync(
    args: MDPPipelineArgs,
    max_attempts: int = 3,
) -> Any:
    """
    Synchronous wrapper for run_mdp_agent_with_args.
    
    Handles event loop creation for Django/Gunicorn thread pool environments.
    """
    logger.info(
        f"Running MDP agent synchronously: "
        f"mode={args.mode}, "
        f"input_path={args.input_path}, "
        f"output_dir={args.output_dir}, "
        f"disease_name={args.disease_name}, "
        f"max_attempts={max_attempts}"
    )
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new event loop for MDP agent")
    
    agent = build_mdp_runner_agent()
    context = MDPRunnerContext(max_attempts=max_attempts)
    
    user_input = (
        f"Run Multi-Disease Pathways (MDP) pipeline with robust self-correction.\n\n"
        f"Initial arguments:\n{json.dumps(args.model_dump(), indent=2)}"
    )
    
    try:
        result = Runner.run_sync(
            starting_agent=agent,
            input=user_input,
            context=context,
        )
        
        status = result.final_output.get('status') if isinstance(result.final_output, dict) else 'unknown'
        logger.info(f"MDP agent completed: status={status}")
        return result.final_output
    except Exception as e:
        logger.exception(f"MDP agent execution failed: {e}")
        raise
