from __future__ import annotations

import os,re, json, asyncio
from typing import Any, Dict, Optional, TypedDict, Tuple, List, Union
from langgraph.graph import StateGraph, START, END 
from openai import AsyncOpenAI
from agents import Agent, Runner
from pathlib import Path
from dotenv import load_dotenv

# node input analysis need to be fixed 
# GSE ID need to be checked 

# Project Imports
from .modulecalling import cohort_call, geo_ontology_fallback  
from .array_express_agent_pipeline.aeagent import AgentAE, need_ae_fallback, ae_ontology_fallback
from .array_express_agent_pipeline.utils import sanitize_folder_name, summarize_analysis
from .array_express_agent_pipeline.evaluation import (
    LLMEvaluatorAE,
    build_ae_datasets_from_summary,
    save_ae_evaluation_results
)
from  .geo_agent_pipeline.tools.decomposer import decompose_cohort_query


api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

# LangSmith for logging
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT")

# Pipeline State
class PipelineState(TypedDict, total=False):
    # Inputs
    query: str
    disease_name: str
    tissue_filter: Any  
    experiment_filter : Any
    no_of_dataset : int
    other_filter: str  

    # Outputs 
    result: Any
    geo_details: Dict[str, Any]
    arrayexpress_result: Optional[Any]

    # Outputs from fallback
    geo_fallback_result: Optional[Dict[str, Any]]
    ae_fallback_result : Optional[Dict[str,Any]]

    # Output Directory for Source
    geo_output_dir : Optional[str]
    ae_output_dir : Optional[str]

    # Final summary text
    summary_text: str
    
# Nodes 
_TISSUE_MAP = {
    "pbmc": "blood", "whole blood": "blood", "peripheral blood": "blood",
    "serum": "blood", "plasma": "blood", "buffy coat": "blood",
}

def _norm_filter(f: Optional[str]) -> Optional[str]:
    if not f: return None
    f = _TISSUE_MAP.get(f.strip().lower(), f.strip().lower())
    return "blood" if "blood" in f else f

def _normalize_tissue(tissue: str | None) -> str | None:
    """Normalize tissue names to consistent labels (e.g., PBMC -> blood)."""
    if not tissue:
        return None
    t = tissue.lower().strip()
    for k, v in _TISSUE_MAP.items():
        if k in t:
            return v
    return t

def _heuristic(query: str) -> Tuple[Optional[str], Optional[str]]:
    ql = query.lower()
    filt = "blood" if any(w in ql for w in ["blood","serum","plasma","pbmc","buffy coat"]) else None
    # naive disease: first capitalized phrase of 1–4 words
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", query)
    disease = m.group(1) if m else None
    return disease, filt

async def _extract(query: str):
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                    "You are an extractor that identifies disease_name/condition/phrase , tissue_filter, "
                    "experiment_type, no_of_dataset, and other_filter from biomedical queries. "

                    "Possible disease/condition/phrase: colon, pancreatic cancer, cervical cancer, lupus, stress response, inflammation, hypoxia, oxidative stress, etc. "

                    "Possible tissue values: pbmc, whole blood, plasma, serum, tissue, fibroblast, epithelial cells, etc. "
                    "Possible experiment types: rna seq, bulk rna, single cell, microRNA, microarray, ATAC-seq, etc. "

                    "If both disease and condition are mentioned, prioritize the disease_name. "

                    "If any of disease_name/condition/phrase, tissue_filter, experiment_type is not given, return 'Any' for that field. "

                    "other_filter should capture any additional filters not recognized as disease_name, "
                    "tissue_filter, experiment_type, or no_of_dataset. Examples: 'Illumina platform', "
                    "'male samples', 'age > 50', 'Homo sapiens', 'drug-treated', 'stimulated', 'time course', "
                    "'GSE only', 'platform GPL570'. "

                    "For no_of_dataset return numeric form. If not provided, return 5. "

                    "Return JSON: {"
                    "\"disease_name\": str|Any, "
                    "\"tissue_filter\": str|Any, "
                    "\"experiment_type\": str|Any, "
                    "\"no_of_dataset\": int|5, "
                    "\"other_filter\": str|Any"
                    "}."
                )
                ,
                },
                # --- Few-shot examples ---
                {
                    "role": "user",
                    "content": "Query: RNA-seq data for HER2 breast cancer in blood samples, atleast 3 datasets",
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "disease_name": "HER2 breast cancer",
                        "tissue_filter": "blood",
                        "experiment_type": "rna seq",
                        "no_of_dataset" : 3,
                        "other_filter": "Any"
                    }),
                },
                {
                    "role": "user",
                    "content": "Query: single-cell transcriptomics of colon's PBMCs",
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "disease_name": "colon",
                        "tissue_filter": "pbmc",
                        "experiment_type": "single cell",
                        "no_of_dataset" : 5,
                        "other_filter": "Any"
                    }),
                },
               {
                    "role": "user",
                    "content": "Query: gene expression profiling in whole blood of lupus patients",
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "disease_name": "lupus",
                        "tissue_filter": "whole blood",
                        "experiment_type": "rna seq",
                        "no_of_dataset": 5,
                        "other_filter": "Any"
                    }),
                },
                {
                    "role": "user",
                    "content": "Query: Find 2 dataset of microRNA analysis for stress response patient with tissue sample having platform Illumina RNA-seq",
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "disease_name": "stress response",
                        "tissue_filter": "tissue",
                        "experiment_type": "microRNA",
                        "no_of_dataset" : 2,
                        "other_filter" : "Illumina RNA-seq"
                    }),
                },
                # --- Actual user query ---
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        disease = data.get("disease_name")
        tissue = _normalize_tissue(data.get("tissue_filter"))
        experiment = data.get("experiment_type")
        no_of_dataset = data.get("no_of_dataset")
        other_filter = data.get("other_filter", "Any")

        print(disease, tissue, experiment, no_of_dataset)
        return disease, tissue, experiment, no_of_dataset, other_filter

    except Exception as e:
        print("EXC TYPE:", type(e).__name__)
        print("EXC MSG :", e)
        return _heuristic(query)

async def run_arrayexpress_agent(state: PipelineState) -> str:
    """
    Runs the ArrayExpress agent for the disease and filter provided in the state.
    """
    print("=" * 60)
    print("Cohort Call for Array Express")
    print("=" * 60)

    disease_name = state.get("disease_name")
    tissue_filter = state.get("tissue_filter")
    experiment_filter = state.get("experiment_filter")
    other_filter = state.get("other_filter")
    query = state.get("query")
    output_dir = state.get('ae_output_dir')
    print("Inside normal array express agent call")
    print(output_dir)
    if not disease_name or not tissue_filter:
        print("Error: Missing disease_name or tissue_filter")
        return []

    print(f"Starting ArrayExpress Agent for disease: {disease_name} and tissue filter: {tissue_filter} and experiment filter: {experiment_filter}")
    
    if not output_dir:
        from .array_express_agent_pipeline.utils import _norm_experiment_filter
        import hashlib

        # Optional: create clean output directory name
        safe_disease = disease_name.replace(" ", "_").lower()
        experiment_filter = _norm_experiment_filter(experiment_filter)

        file_saving = {
            "disease" : safe_disease,
            "tissue_filter": tissue_filter,
            "experiment_filter": experiment_filter,
            "other_filter": other_filter
        }

        def build_cache_key(filters: dict) -> str:
            # Ensure deterministic ordering
            normalized = json.dumps(filters, sort_keys=True)
            cache_hash = hashlib.sha256(normalized.encode()).hexdigest()

            # Shorten for readability (first 12 chars are enough)
            return f"cache_{cache_hash[:12]}"

        output_dir = build_cache_key(file_saving)
        
        state["ae_output_dir"] = output_dir

    # input_data = [
    #     {"role": "user", "content": disease_name},
    #     {"role": "user", "content": tissue_filter},
    #     {"role": "user", "content": experiment_filter},
    #     {"role":"user","content":query},
    #     {"role":"user","content":output_dir}
    # ]


    input_data = [
        {"role": "user", "content": disease_name},
        {"role": "user", "content": tissue_filter},
        {"role": "user", "content": experiment_filter},
        {"role":"user","content":query},
        {"role":"user","content":f"OUTPUT_DIRECTORY:{output_dir}"}
    ]
    print(output_dir)
    # Run the agent with the input data
    result = await Runner.run(AgentAE, input=input_data)
    print("Before Calling Summarie", output_dir)
    # Run the ArrayExpress agent and collect the result
    final_output = summarize_analysis(output_dir, disease_name,tissue_filter, experiment_filter)
    print("✅ ArrayExpress Agent completed. Proceeding to evaluation...")
    print("Error after this")
    if final_output is not None and getattr(final_output, "filtered_experiment_ids", None):
        ae_datasets = build_ae_datasets_from_summary(final_output)
        if not ae_datasets:
            print("⚠️ No datasets found for evaluation.")
            return ""

        print(f"📦 Loaded {len(ae_datasets)} dataset(s) for evaluation")

        evaluator = LLMEvaluatorAE()
        filters = {
            "tissue_filter": tissue_filter,
            "experiment_filter": experiment_filter,
        }

        results = await evaluator.evaluate(ae_datasets, disease_name, filters)
        print(f"✅ Evaluation completed for {len(results)} dataset(s)")

        saved_path = save_ae_evaluation_results(results, disease_name,tissue_filter,experiment_filter)

        print(f"💾 Evaluation results saved at: {saved_path}")

        return final_output
    else:
        return "No dataset found"
   
# -------- DROP-IN NODE  --------
async def node_input_analysis(state: PipelineState) -> PipelineState:
    query = (state.get("query") or "").strip()

    if query:
        d, f, e, n, o = await _extract(query)
        disease = d or state.get("disease_name") 
        tissue_filter  = _norm_filter(f) or _norm_filter(state.get("tissue_filter")) 

        experiment_filter = _norm_filter(e) or _norm_filter(state.get("experiment_filter"))
    else:
        disease = state.get("disease_name") 
        tissue_filter   = _norm_filter(state.get("tissue_filter")) 
    print("Output Directory inside node_input_analysis",state.get("ae_output_dir"))

    return {**state, "query": query, "disease_name": disease, "tissue_filter": tissue_filter, "experiment_filter" : experiment_filter, "no_of_dataset":n, "other_filter":o}

async def node_cohort_usage(state: PipelineState) -> PipelineState:
    """
    Calls your cohort_call
    (), then extracts geo agent details (robustly).
    """

    import hashlib
    def build_cache_key(filters: dict) -> str:
        # Ensure deterministic ordering
        normalized = json.dumps(filters, sort_keys=True)
        cache_hash = hashlib.sha256(normalized.encode()).hexdigest()

        # Shorten for readability (first 12 chars are enough)
        return f"cache_{cache_hash[:12]}"
    
      # 1) Resolve output_dir
    output_dir = state.get("geo_output_dir")
    print("Inside cohort usage")
    print("Output Directory : ", output_dir)
    if not output_dir:
        # Compute deterministically from filters
        file_saving = {
            "disease": (state.get("disease_name") or "").strip().lower().replace(" ", "_"),
            "tissue_filter": state.get("tissue_filter"),
            "experiment_filter": state.get("experiment_filter"),
            "other_filter": state.get("other_filter"),
            "no_of_dataset": state.get("no_of_dataset"),
        }
        

        output_dir = build_cache_key(file_saving)
        state["geo_output_dir"] = output_dir  # persist for later nodes
    print("Output Directory : ", output_dir)


    # print(state.get("query"))
    result, d_name, filt = await cohort_call(state.get("disease_name"), state.get("tissue_filter"), state.get("experiment_filter"), state.get("query"), state.get("no_of_dataset"), state.get("other_filter"), output_dir)

  
    # Try to extract geo details defensively
    geo_details: Dict[str, Any] = {}
    try:
        agent_results = getattr(result, "agent_results", {}) or {}
        geo_res = agent_results.get("geo")
        if geo_res and getattr(geo_res, "details", None):
            geo_details = geo_res.details
    except Exception:
        geo_details = {}

    return {
        "result": result,
        "geo_details": geo_details,
        "geo_output_dir": output_dir,

    }

async def node_arrayexpress_agent(state: PipelineState) -> PipelineState:
    """
    Runs the ArrayExpress agent in parallel with the GEO agent.
    """
    disease_name = state.get("disease_name")
    tissue_filter = state.get("tissue_filter")
    experiment_filter = state.get("experiment_filter")
    print(f"Starting ArrayExpress Agent for disease: {disease_name} and tissue filter: {tissue_filter} and experiment filter {experiment_filter}")
    print("State inside arrayagent : ", state)
    # Run the ArrayExpress agent and collect the result
    result = await run_arrayexpress_agent(state)
    # IMPORTANT: compute output_dir here (node scope), not inside a nested helper
    output_dir = state.get("ae_output_dir")

    return {
        "arrayexpress_result": result,
        "ae_output_dir": output_dir
    }

async def node_geo_ontology_fallback(state: PipelineState) -> PipelineState:
    """
    Runs your geo_ontology_fallback() ONLY when needed (conditional edge controls that).
    """
    disease_name = state["disease_name"]
    tissue_filter = state["tissue_filter"]
    experiment_filter = state["experiment_filter"]
    no_of_dataset = state["no_of_dataset"]
    query = state["query"]
    other_filter = state.get("other_filter")
    output_dir = state.get("geo_output_dir")
    filters = {
        "tissue_filter": tissue_filter,
        "experiment_filter": experiment_filter,
        "no_of_dataset" : no_of_dataset,
        "other filter" : other_filter
    }
    
    fallback = await geo_ontology_fallback(disease_name, filters, query, output_dir)
    print("Oustide the fallbad")
    return {
        # **state,
        "geo_fallback_result": fallback,
    }

async def node_ae_ontology_fallback(state:PipelineState) ->PipelineState:
    """
    Runs your ae_ontology_fallback() ONLY when needed (conditional edge controls that).
    """
    disease_name = sanitize_folder_name(state["disease_name"])
    tissue_filter = sanitize_folder_name(state["tissue_filter"])
    experiment_filter = sanitize_folder_name(state["experiment_filter"])
    print("State :", state)

    fallback = await ae_ontology_fallback(state, disease_name, tissue_filter, experiment_filter)
    return {
        # **state,
        "ae_fallback_result": fallback
    }

async def node_summary_tool(state: PipelineState) -> PipelineState:
    """
    
    Includes results from GEO, ArrayExpress, and any fallback data.
    """
    lines = []
    print("Inside summary")
    dn = state.get("disease_name")
    flt = state.get("tissue_filter")
    eflt = state.get("experiment_filter")
    data = state.get("no_of_dataset")
    oth_f = state.get("other_filter")
    lines.append(f"Summary for disease='{dn}', tissue_filter='{flt}', experiment_filter '{eflt}', no_of_dataset : {data}, Other filter : {oth_f}")

    # # Primary result (from your main result)
    # res = state.get("result")
    # if res:
    #     try:
    #         lines.append(f"- Success: {getattr(res, 'success', None)}")
    #         lines.append(f"- Datasets found: {getattr(res, 'total_datasets_found', None)}")
    #         lines.append(f"- Datasets downloaded: {getattr(res, 'total_datasets_downloaded', None)}")
    #         lines.append(f"- Files downloaded: {getattr(res, 'total_files_downloaded', None)}")
    #         lines.append(f"- Execution time (s): {getattr(res, 'execution_time', None)}")
    #         lines.append(f"- Output Directory : {getattr(res,'output_directory', None)}")
    #     except Exception:
    #         lines.append("- Result present, but unable to parse fields.")

    # # GEO details
    # geo_details = state.get("geo_details") or {}
    # if geo_details:
    #     lines.append("- GEO agent returned valid details; fallback skipped.")
    # else:
    #     lines.append("- GEO agent returned no valid details; checking fallback…")
    #     fb = state.get("geo_fallback_result")
       
    #     if fb is None:
    #         lines.append("  No fallback results found.")
    #     else:
    #         lines.append("\n--- GEO Fallback Summary ---")
    #         for idx, fres in enumerate(fb, 1):
    #             lines.append(f"  [{idx}] Disease: {fres.disease_name}")
    #             lines.append(f"      • Success: {fres.success}")
    #             lines.append(f"      • Total Datasets Found: {fres.total_datasets_found}")
    #             lines.append(f"      • Total Datasets Downloaded: {fres.total_datasets_downloaded}")
    #             lines.append(f"      • Total Files Downloaded: {fres.total_files_downloaded}")
    #             lines.append(f"      • Execution Time: {fres.execution_time:.2f}s")

    #             geo = fres.agent_results.get("geo")
    #             if geo:
    #                 lines.append("      • GEO Agent Details:")
    #                 lines.append(f"          - Valid Datasets: {geo.details.get('valid_datasets', 0)}")
    #                 lines.append(f"          - Downloaded Datasets: {geo.details.get('downloaded_datasets', [])}")
    #                 lines.append(f"          - Output Directory: {geo.details.get('output_directory')}")
    #                 lines.append(f"          - Execution Time: {geo.execution_time:.2f}s")

    # # ArrayExpress results
    # arrayexpress_result = state.get("arrayexpress_result")
    # if arrayexpress_result:
    #     lines.append("\n\n- ArrayExpress agent results:")
    #     try:
    #         lines.append(f"- Disease: {getattr(arrayexpress_result, 'disease', None)}")
    #         lines.append(f"- Datasets found: {getattr(arrayexpress_result, 'total_experiment', None)}")
    #         lines.append(f"- Datasets Ids: {getattr(arrayexpress_result, 'experiment_ids', None)}")

    #         lines.append(f"- Valid Datasets: {getattr(arrayexpress_result, 'total_experiment_filtered', None)}")
    #         lines.append(f"- Valid Datasets Ids: {getattr(arrayexpress_result, 'filtered_experiment_ids', None)}")

    #         lines.append(f"- Meta Data Path: {getattr(arrayexpress_result, 'metadata_path', None)}")
    #         lines.append(f"- Filtered Meta Data Path: {getattr(arrayexpress_result, 'filtered_path', None)}")

    #     except Exception:
    #         lines.append("- Result present, but unable to parse fields.")
    # else:
    #     lines.append("- No results from ArrayExpress agent.")

    # Combine all lines into the final summary text
    print("State inside the summary tool")
    print(state)
    summary_text = state

    # Return the updated state with the summary text added
    return {
        # **state,
             "summary_text": summary_text}

# ---------- Conditional Routing ----------
def needs_geo_fallback(state: PipelineState) -> str:
    """
    Returns the next node name based on whether geo_details is empty.
    """
    geo = state.get("geo_details") or {}
    return "ontology_fallback" if len(geo) == 0 else "summary_tool"

def needs_ae_fallback(state:PipelineState) -> str:
    """
    Returns the next node name based on whatever filtered_meta_file is empty
    """
    print(state)
    print("Output Dir", state.get("ae_output_dir"))
    data = need_ae_fallback(state.get("ae_output_dir"),state.get("disease_name"), state.get("tissue_filter"),state.get("experiment_filter"))
    return "ae_ontology_fallback" if not data or len(data) == 0 else "summary_tool"


# --- Build graph wiring ---
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("input_analysis", node_input_analysis)
    graph.add_node("cohort_usage", node_cohort_usage)
    # graph.add_node("arrayexpress_agent", node_arrayexpress_agent)

    # GEO
    graph.add_node("ontology_fallback", node_geo_ontology_fallback)

    # AE
    # graph.add_node("ae_ontology_fallback", node_ae_ontology_fallback)

    graph.add_node("summary_tool", node_summary_tool)


    # Start → input analysis → fork to GEO & AE
    graph.add_edge(START, "input_analysis")
    graph.add_edge("input_analysis", "cohort_usage")
    # graph.add_edge("input_analysis", "arrayexpress_agent")

    # GEO conditional
    graph.add_conditional_edges(
        "cohort_usage",
        needs_geo_fallback,
        {
            "ontology_fallback": "ontology_fallback",
            "summary_tool": "summary_tool",
        },
    )

    # # AE conditional (uses YOUR needs_ae_fallback)
    # graph.add_conditional_edges(
    #     "arrayexpress_agent",
    #     needs_ae_fallback,
    #     {
    #         "ae_ontology_fallback": "ae_ontology_fallback",
    #         "summary_tool": "summary_tool",
    #     },
    # )

    # Both fallbacks converge to summary
    graph.add_edge("ontology_fallback", "summary_tool")
    # graph.add_edge("ae_ontology_fallback", "summary_tool")
    graph.add_edge("summary_tool", END)

    return graph.compile()

# ---------- Runner ----------
async def run_pipeline(
    query: Optional[str] = None,
    output_dir: Optional[str] = None

) -> "PipelineState":
    """
    Entry point for cohort retrieval pipeline (GEO + ArrayExpress).
    
    Runs a LangGraph workflow that:
    1. Parses user query to extract disease/tissue/experiment filters
    2. Calls GEO agent (modulecalling.cohort_call)
    3. Calls ArrayExpress agent (aeagent.AgentAE)
    4. Falls back to ontology search if no results
    5. Generates summary report
    
    Args:
        query: Natural language query (e.g., "Find single cell datasets for lupus")
               If provided, disease/tissue/experiment are extracted automatically.
        disease_name: Disease/condition name (e.g., "lupus", "cervical cancer")
                     Required if query not provided.
        tissue_filter: Tissue type filter (e.g., "blood", "pbmc", "tissue")
                      Optional. Normalized internally (pbmc->blood, etc).
        experiment_filter: Experiment type (e.g., "single cell", "bulk rna", "rna seq")
                          Optional.
    
    Returns:
        PipelineState: Dict containing:
            - result: CohortResult object (success, output_directory, error, etc.)
            - geo_details: Dict of GEO dataset details
            - arrayexpress_result: AnalysisSummary object from ArrayExpress
            - geo_fallback_result: Dict from ontology fallback (if triggered)
            - ae_fallback_result: Dict from AE ontology fallback (if triggered)
            - summary_text: Human-readable summary of entire retrieval
    
    Example:
        # Natural language query
        result = await run_pipeline(query="Find single cell datasets for cervical cancer in blood")
        
        # Explicit parameters
        result = await run_pipeline(
            disease_name="lupus",
            tissue_filter="pbmc",
            experiment_filter="single cell"
        )
        
        # Check success
        if result["result"].success:
            print(f"Downloaded to: {result['result'].output_directory}")
            print(f"Summary: {result['summary_text']}")
    """
    app = build_graph()

    initial_state: "PipelineState" = {}
    if query is not None:
        initial_state["query"] = query

    if output_dir:  # user provided
        initial_state["geo_output_dir"] = output_dir
        initial_state['ae_output_dir'] = output_dir

    final_state = await app.ainvoke(
        initial_state
    )
    print("Inside the pipeline ", final_state)
    return final_state

async def cohortagent(user_query: str, output_dir: str | None = None):
    """
    Top-level API wrapper that:
      1) Detects single vs multiple disease/condition requests
      2) Runs the retrieval pipeline for each sub-query
      3) Returns combined structured results
    """
    # 1) Decompose query
    decomp = await decompose_cohort_query(user_query)

    # 2) Prepare sub-queries list
    if decomp["request_type"] == "single":
        query_list = [("Full query", user_query)]
    else:
        query_list = list(zip(decomp["conditions"], decomp["sub_queries"]))

    all_outputs = []

    # 3) Run pipeline on each sub-query
    for label, sub_q in query_list:
        result = await run_pipeline(query=sub_q, output_dir=output_dir)

        print("Inside the runner function", result)
        # Keep a clean data record in memory
        all_outputs.append({
            "condition": label,
            "query": sub_q,
            "result": result,
            "summary": result.get("summary_text", "(no summary)")
        })

    # 4) Final standardized output format
    if len(all_outputs) == 1:
        # Single request → return primary output as-is
        return {
            "request_type": "single",
            "original_query": user_query,
            "result": all_outputs[0]["result"],
            "summary": all_outputs[0]["summary"]
        }
    else:
        # Multiple requests → return all in one dictionary
        return {
            "request_type": "multi",
            "original_query": user_query,
            "conditions": [
                {
                    "condition": item["condition"],
                    "sub_query": item["query"],
                    "result": item["result"],
                    "summary": item["summary"]
                }
                for item in all_outputs
            ]
        }

# # ---------- CLI (script use) ----------
# if __name__ == "__main__":

#     async def main():
#         query = (
#             "Automatically find 2 RNA-seq cohorts in GEO matching HER2+ breast cancer"
#         )
        
# # cell line negative filter
# # breast cancer RNA Seq, lung cancer, diabetes, 
# # HER2 Breast cancer RNA Seq
# # Rare disease (Ontology Tool)

# # Compound request ( need to see this as well )
#         output = await cohortagent(query)
#         print(output)

#     asyncio.run(main())
