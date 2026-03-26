import os
import pickle
import json
from openai import OpenAI
from decouple import config

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------
# 1. LLM CALL — Convert natural language → GEO query
# ----------------------------------------------------

def _query_llm_for_api(prompt, schema, system_template):
    """
    Minimal GPT call. Converts prompt → JSON.
    Handles non-string prompts robustly.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Safely stringify schema and inject into template without .format() breaking on {}
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
    system_prompt = system_template.replace("{schema}", schema_str)

    # Ensure user content is always a string
    if isinstance(prompt, (dict, list)):
        user_content = json.dumps(prompt, indent=2, ensure_ascii=False)
    else:
        user_content = str(prompt)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    print("Raw LLM response:", raw)

    # Try direct JSON parse
    try:
        return {"success": True, "data": json.loads(raw), "raw": raw}
        # return{"success":True, "data":raw}
    except Exception:
        # Attempt to extract JSON inside text
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return {"success": True, "data": json.loads(raw[start:end]), "raw": raw}
        except Exception as e:
            return {"success": False, "error": f"JSON parse error: {e}", "raw": raw}


# ----------------------------------------------------
# 2. MAIN FUNCTION — ALWAYS RETURNS GPT-GENERATED QUERY
# ----------------------------------------------------

def query_geo(prompt):
    """
    Always generate a GEO search query using GPT.
    Includes disease/condition synonyms + abbreviations as an OR-group when applicable.
    """

    if not prompt:
        return {"error": "prompt is required"}

    # Load GEO schema
    schema_path = os.path.join(os.path.dirname(__file__), "geo.pkl")
    with open(schema_path, "rb") as f:
        geo_schema = pickle.load(f)

    # (Optional but recommended) augment schema guidance so the LLM returns disease_terms
    # If your geo.pkl is not a JSON schema, this still works as a guidance object.
    geo_schema_guidance = {
        "type": "object",
        "properties": {
            "search_term": {"type": "string"},
            "database": {"type": "string", "description": "Use 'gds' unless user explicitly requests otherwise"},
            "disease_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Disease/condition synonyms + abbreviations used in the query"
            }
        },
        "required": ["search_term", "database"]
    }

    # Prefer your loaded schema if it is already a dict-like schema; otherwise use guidance
    schema_for_llm = geo_schema if isinstance(geo_schema, dict) else geo_schema_guidance

    # System template for LLM
    system_template = """
You convert natural language into GEO (Gene Expression Omnibus) search queries.

OUTPUT MUST BE VALID JSON and match this schema:
{schema}

Your job has TWO steps:

1) Extract ALL key biological concepts from the user query:
- Organism (e.g. Homo sapiens)
- Tissue / sample type
- Platform / technology (e.g. Illumina)
- Disease / condition / biological process (e.g. immune response)
- Dataset type ONLY if explicitly stated (Series, Samples, Platforms, Datasets)

2) Build a single GEO search_term string using ALL concepts.

IMPORTANT TYPE RULES
- DO NOT add gse[ETYP], gsm[ETYP], gpl[ETYP], gds[ETYP] unless the user explicitly
  mentions the dataset type (Series, Samples, Platforms, or Datasets).
- If dataset type is NOT mentioned, DO NOT include any [ETYP] tag.

GENERAL RULES
- NEVER drop diseases/conditions/biological processes.
- Don't include samples characteristics like disease samples, normal samples
- Use AND / OR / NOT (uppercase).
- Use AND for multi-word terms, e.g. "immune"[All Fields] AND "response"[All Fields].
- Use correct GEO fields when known (e.g., Homo sapiens[ORGN]).
- Use parentheses for grouping.

DISEASE SYNONYM RULES (MANDATORY WHEN DISEASE/CONDITION EXISTS)
- If a disease/condition is present, generate 3–8 widely-used synonyms and/or abbreviations
  (include US/UK spelling variants if relevant).
- DO NOT invent obscure terms; keep only common equivalents.
- In the final search_term, represent the disease/condition as a single parenthesized OR-group:
  ("primary term"[All Fields] OR "synonym 1"[All Fields] OR "synonym 2"[All Fields] OR "abbr"[All Fields])
- Also return the exact list you used in a JSON array field called "disease_terms".

EXAMPLES

    1) User query:
    Find RNA-seq datasets having normal samples for breast cancer in humans.

    CORRECT (only disease + experiment type; ignore samples and organism):
    {
    "search_term": "(breast[All Fields] AND cancer[All Fields] OR mammary[All Fields] AND carcinoma[All Fields] OR breast[All Fields] AND carcinoma[All Fields]) AND RNA[All Fields] AND Seq[All Fields]",
    "database": "gds",
    "disease_terms": ["breast cancer", "mammary carcinoma", "breast carcinoma"]
    }

    2) User query:
    Get GEO Series related to inflammation disease samples in humans.

    CORRECT (dataset type ignored by scope rule; only disease retained):
    {
    "search_term": "(inflammation[All Fields] OR inflammatory[All Fields] AND response[All Fields])",
    "database": "gds",
    "disease_terms": ["inflammation", "inflammatory response"]
    }

    3) User query:
    Circular RNA Seq data for Lung cancer in 2020.

    CORRECT (ignore year and circular detail; keep disease + RNA-seq only):
    {
    "search_term": "(lung[All Fields] AND cancer[All Fields] OR lung[All Fields] AND carcinoma[All Fields] OR pulmonary[All Fields] AND carcinoma[All Fields]) AND RNA[All Fields] AND Seq[All Fields]",
    "database": "gds",
    "disease_terms": ["lung cancer", "lung carcinoma", "pulmonary carcinoma"]
    }

    4) User query:
    Retrieve and download treatment-naïve or baseline human lung cancer bulk RNA-seq expression datasets suitable for differential expression analysis, prioritizing tumor vs control cohort designs from primary tissue, and excluding drug/therapy response arms and cultured samples.

    CORRECT (only disease + experiment type; all other constraints ignored):
    {
    "search_term": "(lung[All Fields] AND cancer[All Fields] OR lung[All Fields] AND carcinoma[All Fields] OR pulmonary[All Fields] AND carcinoma[All Fields]) AND RNA[All Fields] AND Seq[All Fields]",
    "database": "gds",
    "disease_terms": ["lung cancer", "lung carcinoma", "pulmonary carcinoma"]
    }
"""

    # Hit GPT
    llm_res = _query_llm_for_api(prompt, schema_for_llm, system_template)
    print("LLM result:", llm_res)

    if not llm_res.get("success"):
        return llm_res

    data = llm_res["data"]
    print("Parsed data:", data)
   
    return {
        "search_term": data.get("search_term", ""),
        "database": data.get("database", "gds"),
        # NEW: expose synonyms list for debugging/traceability
        "disease_terms": data.get("disease_terms", []),
        "source": "prompt",
        "raw": llm_res["raw"],
    }