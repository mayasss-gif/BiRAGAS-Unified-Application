from __future__ import annotations
from agents import Agent, Runner
import asyncio
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os, json, re
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any

# Force re-load with override
load_dotenv(override=True)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Function Tool ----------------
from   .tools import arrayexpress_query, parse_biostudies_json, filter_experiments_with_llm, download_experiment_ids
from   .constants import Defaults
from   .utils import sanitize_folder_name, summarize_analysis, _norm_experiment_filter
from   .evaluation import (
    LLMEvaluatorAE,
    build_ae_datasets_from_summary,
    save_ae_evaluation_results
)

# AgentAE = Agent(
#     name="ArrayExpressAgent",
#     instructions=(
#     "You are a bioinformatics assistant that helps extract and filter experiment data "
#     "from the ArrayExpress database.\n\n"

#     "You may be provided with either:\n"
#     "- A structured disease name and tissue type, OR\n"
#     "- A free-form, instruction-style biomedical query.\n\n"

#     "Your goal is to retrieve relevant experiment metadata. If the input is a free-form "
#     "query, you must first extract the biomedical intent (disease, tissue, assay, study type) "
#     "and generate a valid ArrayExpress search query before querying the database.\n\n"

#     "If the query does not clearly specify biological details, apply the following defaults:\n"
#     "- Study type: high throughput sequencing\n"
#     "- Organism: Homo sapiens\n"
#     "- Experiment type: RNA-seq\n"
#     "- Assay by molecule: RNA assay\n\n"

#     "Use the available tools to:\n"
#     "- Query the ArrayExpress API for experiment accession IDs\n"
#     "- Fetch and parse JSON metadata from BioStudies\n"
#     "- Filter experiments based on disease and tissue relevance and user query\n"
#     "- Download valid experiments using the download tool\n\n"

#     "Decide when and how to apply each tool by reading their docstrings. "
#     "Complete the task end-to-end without being given the order of tools explicitly."

#     "Important:\n"
#     "- You may receive an output directory path as part of the input.\n"
#     "- This path is not biological information.\n"
#     "- Always forward the output directory unchanged to tools that write files or download data.\n\n"

# ),
#     tools=[arrayexpress_query, parse_biostudies_json, filter_experiments_with_llm, download_experiment_ids],

# )

AgentAE = Agent(
    name="ArrayExpressAgent",
    instructions=(
    "You are a bioinformatics assistant that helps extract and filter experiment data "
    "from the ArrayExpress database.\n\n"

    "You may receive TWO inputs:\n"
        "1. A structured disease name and tissue type, \n"
        "2. A free-form biomedical query (biological intent)\n"
        "3. An output directory path (filesystem parameter)\n\n"

    "Your goal is to retrieve relevant experiment metadata. If the input is a free-form "
    "query, you must first extract the biomedical intent (disease, tissue, assay, study type) "
    "and generate a valid ArrayExpress search query before querying the database.\n\n"

    "If the query does not clearly specify biological details, apply the following defaults:\n"
    "- Study type: high throughput sequencing\n"
    "- Organism: Homo sapiens\n"
    "- Experiment type: RNA-seq\n"
    "- Assay by molecule: RNA assay\n\n"

    "Use the available tools to:\n"
    "- Query the ArrayExpress API for experiment accession IDs\n"
    "- Fetch and parse JSON metadata from BioStudies\n"
    "- Filter experiments based on disease and tissue relevance and user query\n"
    "- Download valid experiments using the download tool\n\n"

    "Decide when and how to apply each tool by reading their docstrings. "
    "Complete the task end-to-end without being given the order of tools explicitly."

    "Important:\n"
    "- You may receive an output directory path as part of the input.\n"
    "- This path is not biological information.\n"
    "- Always forward the output directory unchanged to tools that write files or download data.\n\n"

),
    tools=[arrayexpress_query, parse_biostudies_json, filter_experiments_with_llm, download_experiment_ids],

)

from typing import Any, Dict, Optional, TypedDict, Tuple, List, Union

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
    


# async def run_arrayexpress_agent(disease_name : str, tissue_filter : str, experiment_filter : str) -> List[str]:
#     """Run the agent for a sample disease input and return final output."""
#     print("Starting ArrayExpress Agent for disease query...")

#     input_data = [
#         {"role": "user", "content": disease_name},
#         {"role": "user", "content": tissue_filter},
#         {"role": "user", "content": experiment_filter}    
#         ]
    
#     # Run the agent with the input data
#     result = await Runner.run(AgentAE, input=input_data)
#     final_output = summarize_analysis(disease_name,tissue_filter, experiment_filter)
#     print(final_output)
#     print("✅ ArrayExpress Agent completed. Proceeding to evaluation...")
    
#     if final_output is not None and getattr(final_output, "filtered_experiment_ids", None):
#         ae_datasets = build_ae_datasets_from_summary(final_output)
#         if not ae_datasets:
#             print("⚠️ No datasets found for evaluation.")
#             return ""
#         print(f"📦 Loaded {len(ae_datasets)} dataset(s) for evaluation")

#         evaluator = LLMEvaluatorAE()
#         filters = {
#             "tissue_filter": tissue_filter,
#             "experiment_filter": experiment_filter,
#         }

#         results = await evaluator.evaluate(ae_datasets, disease_name, filters)
#         print(f"✅ Evaluation completed for {len(results)} dataset(s)")

#         saved_path = save_ae_evaluation_results(results, disease_name,tissue_filter,experiment_filter)

#         print(f"💾 Evaluation results saved at: {saved_path}")

#         return final_output
#     else:
#         return "No dataset found"



async def run_arrayexpress_agent(state: PipelineState,disease_name : str, tissue_filter : str, experiment_filter : str, output_dir: str | None = None,) -> str:
    """
    Runs the ArrayExpress agent for the disease and filter provided in the state.
    """
    print("=" * 60)
    print("Cohort Call for Array Express")
    print("=" * 60)

    other_filter = state.get("other_filter")
    query = state.get("query")
    print(query)
    if not disease_name or not tissue_filter:
        print("Error: Missing disease_name or tissue_filter")
        return []

    print(f"Starting ArrayExpress Agent for disease: {disease_name} and tissue filter: {tissue_filter} and experiment filter: {experiment_filter}")
    
        # -----------------------------
    # Resolve + persist output_dir
    # -----------------------------
    if output_dir:
        # highest priority: explicit parameter
        state["output_dir"] = output_dir
        print(f"Using provided output_dir: {output_dir}")
    # elif state.get("output_dir"):
    #     # second priority: already present in state
    #     output_dir = state["output_dir"]
    #     print(f"Using existing output_dir from state: {output_dir}")
    else:
        from .utils import _norm_experiment_filter
        import hashlib

        # otherwise compute
        safe_disease = disease_name.replace(" ", "_").lower()
        exp_norm = _norm_experiment_filter(experiment_filter)

        file_saving = {
            "disease": safe_disease,
            "tissue_filter": tissue_filter,
            "experiment_filter": exp_norm,
            "other_filter": other_filter,
        }

        def build_cache_key(filters: dict) -> str:
            normalized = json.dumps(filters, sort_keys=True)
            cache_hash = hashlib.sha256(normalized.encode()).hexdigest()
            return f"cache_{cache_hash[:12]}"

        output_dir = build_cache_key(file_saving)
        state["output_dir"] = output_dir
        print(f"Computed and stored output_dir: {output_dir}")


    input_data = [
        {"role": "user", "content": f"DISEASE:{disease_name}"},
        {"role": "user", "content": f"TISSUE_FILTER: {tissue_filter}"},
        {"role": "user", "content": f"EXPERIMENT_FILTER: {experiment_filter}"},
        {"role":"user","content":f"QUERY: {query}"},
        {"role":"user","content":f"OUTPUT_DIRECTORY:{output_dir}"}
    ]
    print(output_dir)
    print("Input DAta",input_data)
    
    # Run the agent with the input data
    result = await Runner.run(AgentAE, input=input_data)
    print("Before Calling Summarie", output_dir)

    # Run the ArrayExpress agent and collect the result
    final_output = summarize_analysis(state['ae_output_dir'],disease_name,tissue_filter, experiment_filter)
    print("✅ ArrayExpress Agent completed. Proceeding to evaluation...")
    print("Final Output is", final_output)
    if final_output is not None and getattr(final_output, "filtered_experiment_ids", None):
        ae_datasets = build_ae_datasets_from_summary(final_output)
        print(ae_datasets)
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

        saved_path = save_ae_evaluation_results(results, disease_name,tissue_filter,experiment_filter,output_dir)

        print(f"💾 Evaluation results saved at: {saved_path}")

        return final_output
    else:
        return "No dataset found"

def need_ae_fallback(output_dir: str, disease:str, tissue_filter : str, experiment_filter : str) -> str:

    safe_disease = sanitize_folder_name(disease)
    safe_tissue = sanitize_folder_name(tissue_filter)
    safe_experiment = _norm_experiment_filter(experiment_filter)

    filepath = Path(Defaults.META_FILEPATH) / Path(output_dir) / safe_disease

    filepath.mkdir(parents=True, exist_ok=True)

    json_file_path = f"{filepath}/{safe_disease}_filtered_experiments.json"

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

    except Exception as e:
        print(f"❌ Failed to read JSON file at {json_file_path}: {e}")
        return
    return data


async def ae_ontology_fallback(state:PipelineState, disease:str, tissue_filter:str, experiment_filter : str) -> List[str]:
    print("Inside Ontology Fallback")
    print(state["ae_output_dir"])
    print("State",state)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    SAFE = re.compile(r"[^a-z0-9._-]+")
    def _safe_name(s: str) -> str:
        return SAFE.sub("_", (s or "").strip().lower())

    _ARRAY_FIELDS = [
        "synonyms",
        "children",
        "parents",
        "siblings",
        "primary_tissues",
        "associated_phenotypes",
        "cross_disease_drivers",
    ]

    def _write_json(path: Path, obj: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

    def get_disease_ontology(disease_name: str) -> Dict[str, Any]:
        system_msg = (
            "You are a biomedical ontology assistant. "
            "Given a disease name, return ONLY a JSON object that matches the schema: "
            "{canonical: str, synonyms: [str], children: [str], parents: [str], "
            "siblings: [str], primary_tissues: [str], associated_phenotypes: [str], "
            "cross_disease_drivers: [str]} . No extra text."
        )

        fewshot_user_1 = "Disease: asthma"
        fewshot_assistant_1 = {
            "canonical": "asthma",
            "synonyms": ["bronchial asthma"],
            "children": ["allergic asthma", "nonallergic asthma", "exercise-induced bronchoconstriction"],
            "parents": ["obstructive lung disease", "chronic inflammatory airway disease"],
            "siblings": ["chronic obstructive pulmonary disease", "bronchiectasis"],
            "primary_tissues": ["bronchial epithelium", "airway smooth muscle", "lung parenchyma"],
            "associated_phenotypes": ["wheezing", "airway hyperresponsiveness", "reversible airflow obstruction"],
            "cross_disease_drivers": ["allergy", "air pollution", "respiratory infections"]
        }

        fewshot_user_2 = "Disease: type 2 diabetes mellitus"
        fewshot_assistant_2 = {
            "canonical": "type 2 diabetes mellitus",
            "synonyms": ["T2DM", "adult-onset diabetes", "non–insulin-dependent diabetes mellitus"],
            "children": ["diabetic nephropathy", "diabetic retinopathy", "diabetic neuropathy"],
            "parents": ["diabetes mellitus", "metabolic disease"],
            "siblings": ["type 1 diabetes mellitus", "gestational diabetes"],
            "primary_tissues": ["pancreatic islet", "liver", "skeletal muscle", "adipose tissue"],
            "associated_phenotypes": ["hyperglycemia", "insulin resistance", "impaired glucose tolerance"],
            "cross_disease_drivers": ["obesity", "sedentary lifestyle"]
        }

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": fewshot_user_1},
            {"role": "assistant", "content": json.dumps(fewshot_assistant_1, ensure_ascii=False)},
            {"role": "user", "content": fewshot_user_2},
            {"role": "assistant", "content": json.dumps(fewshot_assistant_2, ensure_ascii=False)},
            {"role": "user", "content": f"Disease: {disease_name}"}
        ]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=700,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content

            if content.startswith("```"):
                content = content.strip().strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].lstrip()

            data: Dict[str, Any] = json.loads(content)

            base_dir = Path(Defaults.ONTOLOGY_FILEPATH)     
            disease_slug = _safe_name(disease_name)
            tissue_slug = _safe_name(tissue_filter)
            experiment_slug = _safe_name(experiment_filter)
            disease_dir = base_dir / disease_slug        
            disease_dir.mkdir(parents=True, exist_ok=True)

            _write_json(disease_dir / f"{disease_slug}.json", data)

            fallback_dir = disease_dir / "fallback_attempt"
            fallback_dir.mkdir(parents=True, exist_ok=True)

            canonical = data.get("canonical", disease_name)
            for field in _ARRAY_FIELDS:
                values = data.get(field, [])

                _write_json(fallback_dir / f"{field}.json", values)

                for v in values:
                    item_slug = _safe_name(str(v))

                    file_name = f"{field}__{item_slug}.json"

                    payload = {
                        "disease": disease_name,
                        "canonical": canonical,
                        "field": field,
                        "value": v
                    }
                    if len(file_name) > 10:
                        print(f"⚠️ Skipping unusually large payload (keys={len(payload)}): {file_name}")
                        continue
                    _write_json(fallback_dir / file_name, payload)

            print(f"[Saved] {disease_name} -> {disease_dir.resolve()}")
            return data

        except Exception as e:
            try:
                base_dir = Path(Defaults.ONTOLOGY_FILEPATH)
                base_dir.mkdir(parents=True, exist_ok=True)
                err_path = base_dir / f"{_safe_name(disease_name)}__error.json"
                _write_json(err_path, {"input_disease": disease_name, "error": str(e)})
                print(f"[ErrorSaved] {disease_name} -> {err_path.resolve()}")

            except Exception:
                pass
            return {"error": f"Failed to fetch ontology: {e}"}

    ontology_result = get_disease_ontology(disease)
    all_diseases = []

    for field in ["synonyms", "siblings","primary_tissues", "children", "parents","associated_phenotypes","cross_disease_drivers"]:
        print(field)
        for name in ontology_result.get(field, []):
            print(name)
            all_diseases.append(name)
    
    print("Onotology Disease been run")
    print(all_diseases)

    ontology_results = []
    print("State :", state)
    for idx, disease in enumerate(all_diseases, start=1):
        print(f"Calling the AE Agent for the disease{disease}")
        state["query"] = disease + " and " + tissue_filter +" and " + experiment_filter
        state["disease"] = disease
        print("State inside ontology", state)
        print(disease)
        result = await run_arrayexpress_agent(state, disease,tissue_filter,experiment_filter,state["ae_output_dir"])
        ontology_results.append(result)
    return ontology_result

# For testing individual modules
if __name__ == "__main__":
    async def _main():
       
        input_data = [
            {"role": "user", "content": "cervical cancer"},
            {"role": "user", "content": "any"},
            {"role": "user", "content": "Rna-seq"},
            {"role":"user","content":"Find rna seq cohort for cervical cancer"}
        ]
        query = "Find rna seq cohort for cervical cancer"
        # Run the agent with the input data
        # result = await Runner.run(AgentAE, input=input_data)
        # print(result)
        # node = need_ae_fallback("pancreatic cancer","tissue")
        # print(node)

        pipeline_state: PipelineState = {
    # =====================
            # Inputs
            # =====================
            "query": "pancreatic cancer blood rna seq",
            "disease_name": "pancreatic cancer",
            "tissue_filter": ["blood", "pbmc"],
            "experiment_filter": ["RNA-seq", "bulk RNA-seq"],
            "no_of_dataset": 3,
            "other_filter": "human",
            "ae_output_dir": "cache_452aa2f5af555"}

        from cohort_retrieval_agent.cohortagent import node_geo_ontology_fallback
        node = "ontology_fallback"
        if node == "ontology_fallback":
            result = await ae_ontology_fallback(pipeline_state,"pancreatic cancer","blood","rna seq")
            print("Ontology Result:", json.dumps(result, indent=4))
    asyncio.run(_main())
