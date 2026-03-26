"""
Extract CRISPR screening parameters from user query.

Supports: modes (1-6), input_dir override.
Default: modes=[3] (RRA+MLE), input_dir from default or uploaded data.
Used by meta_agent and agentic_views for crispr_screening_analysis.
"""
import re
import json
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

DEFAULT_MODES = [3]
DEFAULT_INPUT_DIR = "agentic_ai_wf/crispr_pipeline_agent/input_data/screening_data"

# Mode descriptions for NL matching
MODE_KEYWORDS = {
    1: ["mode 1", "mode1", "rra only", "counts to rra", "gene ranking", "mageck rra", "sanity check"],
    2: ["mode 2", "mode2", "mle only", "counts to mle", "model-based", "mageck mle"],
    3: ["mode 3", "mode3", "rra + mle", "rra and mle", "rra mle", "core", "recommended", "mageck"],
    4: ["mode 4", "mode4", "fastq to counts", "fastqs to counts", "recompute counts"],
    5: ["mode 5", "mode5", "drug", "drug comparison", "druga drugb", "chemogenetic"],
    6: ["mode 6", "mode6", "full", "bagel", "hitselection", "depmap", "essentiality", "full screen"],
}


def extract_crispr_screening_params_sync(user_query: str) -> Dict[str, Any]:
    """
    Extract CRISPR screening params from user query. Sync version.
    Returns: modes (List[int]), input_dir (str), generate_report (bool).
    """
    query = (user_query or "").lower()
    out: Dict[str, Any] = {
        "modes": list(DEFAULT_MODES),
        "input_dir": "",
        "generate_report": True,
    }

    # Extract explicit mode numbers (e.g., "mode 1", "modes 3,6", "run mode 1 and 3")
    mode_nums = []
    for m in [1, 2, 3, 4, 5, 6]:
        if re.search(rf"\bmode\s*{m}\b", query, re.I) or re.search(rf"\bmodes?\s*[,\s]*{m}\b", query, re.I):
            mode_nums.append(m)
    if mode_nums:
        out["modes"] = sorted(set(mode_nums))

    # Keyword-based mode detection
    for mode_id, keywords in MODE_KEYWORDS.items():
        if any(kw in query for kw in keywords) and mode_id not in out["modes"]:
            out["modes"].append(mode_id)
    if len(out["modes"]) > 1:
        out["modes"] = sorted(set(out["modes"]))

    # "all" or "full" → modes 1-6
    if "run all" in query or "all modes" in query or "run mode 1 to 6" in query:
        out["modes"] = [1, 2, 3, 4, 5, 6]

    # input_dir: if user specifies custom path (rare in NL)
    input_match = re.search(r"input\s*(?:dir|path|folder)[:\s]+[\"']?([^\s\"']+)[\"']?", query, re.I)
    if input_match:
        out["input_dir"] = input_match.group(1).strip()

    if "no report" in query or "skip report" in query:
        out["generate_report"] = False

    return out


async def extract_crispr_screening_params_async(user_query: str) -> Dict[str, Any]:
    """
    Async: regex first, LLM fallback for ambiguous mode requests.
    """
    sync_result = extract_crispr_screening_params_sync(user_query)
    if len(sync_result["modes"]) > 0:
        return sync_result

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Extract CRISPR screening mode numbers from this query.

Return JSON only: {{"modes": [3]}}

Modes: 1=Counts→RRA, 2=Counts→MLE, 3=RRA+MLE (recommended), 4=FASTQ→Counts, 5=Drug comparison, 6=Full+BAGEL2+HitSelection.
Default if unclear: [3]. Use specific modes only if user clearly requests them.

Query: "{user_query}"

JSON:"""
        resp = await llm.ainvoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```\w*\n?", "", raw).strip()
        data = json.loads(raw)
        if data.get("modes"):
            sync_result["modes"] = [int(m) for m in data["modes"] if 1 <= int(m) <= 6]
    except Exception as e:
        logger.warning(f"LLM extraction failed for crispr_screening: {e}")
        sync_result["modes"] = list(DEFAULT_MODES)

    return sync_result


def validate_crispr_screening_params(params: Dict[str, Any]) -> Optional[str]:
    """Return error message if params invalid, else None."""
    modes = params.get("modes", DEFAULT_MODES)
    if not modes:
        return "CRISPR screening requires at least one mode (1-6)"
    invalid = [m for m in modes if m not in (1, 2, 3, 4, 5, 6)]
    if invalid:
        return f"Invalid mode(s): {invalid}. Supported: 1-6"
    return None
