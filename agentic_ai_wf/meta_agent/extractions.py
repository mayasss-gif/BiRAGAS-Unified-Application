"""Parameter extraction from user queries (deconvolution, GSE, disease, biosample, causal)."""

import json
import logging
import os
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

BIOSAMPLE_PHRASES = [
    ("whole blood", "Whole Blood"),
    ("blood", "Whole Blood"),
    ("pancreas", "Pancreas"),
    ("pancreatic", "Pancreas"),
    ("brain", "Brain"),
    ("cerebral", "Brain"),
    ("breast", "Breast"),
    ("mammary", "Breast"),
    ("colon", "Colon"),
    ("colorectal", "Colon"),
    ("pbmc", "PBMC"),
    ("peripheral blood", "Whole Blood"),
    ("liver", "Liver"),
    ("kidney", "Kidney"),
    ("lung", "Lung"),
    ("heart", "Heart"),
    ("adipose", "Adipose Tissue"),
    ("fat", "Adipose Tissue"),
]


async def extract_deconvolution_technique(user_query: str) -> Optional[str]:
    """Extract deconvolution technique (xcell, cibersortx, bisque) from query."""
    try:
        query_lower = user_query.lower()
        if "xcell" in query_lower or "x-cell" in query_lower:
            logger.info("Detected deconvolution technique (regex): xcell")
            return "xcell"
        if "cibersort" in query_lower:
            logger.info("Detected deconvolution technique (regex): cibersortx")
            return "cibersortx"
        if "bisque" in query_lower:
            logger.info("Detected deconvolution technique (regex): bisque")
            return "bisque"

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Extract the deconvolution technique from the following query.
TASK: Identify which deconvolution method the user wants to use.
SUPPORTED: xcell, cibersortx, bisque
RULES: Return ONLY the technique name or "null" if not mentioned.
USER QUERY: "{user_query}"
Answer:"""
        response = await llm.ainvoke(prompt)
        result = response.content.strip().lower()
        valid = ["xcell", "cibersortx", "bisque"]
        if result in valid:
            logger.info("Detected deconvolution technique (LLM): %s", result)
            return result
        if result not in ["null", "none", "n/a", "unknown"]:
            logger.warning("Unexpected deconvolution technique from LLM: %s", result)
        return None
    except Exception as e:
        logger.warning("Failed to extract deconvolution technique: %s", e)
        return None


def extract_biosample_from_query(user_query: str) -> Optional[str]:
    """Extract GTEx biosample type from query for GWAS-MR."""
    q = (user_query or "").lower()
    for phrase, canonical in BIOSAMPLE_PHRASES:
        if phrase in q:
            return canonical
    return None


async def extract_gse_ids_from_query(user_query: str) -> List[str]:
    """Extract GSE IDs from user query (regex + optional LLM fallback)."""
    gse_pattern = r"\bGSE\d{6,}\b"
    matches = re.findall(gse_pattern, user_query, re.IGNORECASE)
    gse_ids = list(set(gse.upper() for gse in matches))
    if gse_ids:
        logger.info("Detected GSE IDs from query (regex): %s", gse_ids)
        return gse_ids

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Extract all GEO Series (GSE) IDs. Return ONLY a JSON array or [].
Examples: ["GSE174367"], ["GSE123456","GSE789012"], []
USER QUERY: "{user_query}"
Answer:"""
        response = await llm.ainvoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r"```json\n?", "", raw)
        raw = re.sub(r"```\n?", "", raw)
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            normalized = [g for g in parsed if isinstance(g, str) and g.upper().startswith("GSE")]
            normalized = list(set(g.upper() for g in normalized))
            if normalized:
                logger.info("Detected GSE IDs from query (LLM): %s", normalized)
                return normalized
    except Exception as e:
        logger.warning("GSE ID extraction failed: %s", e)
    return []


async def extract_disease_from_query(user_query: str) -> Optional[str]:
    """Extract disease/condition from user query."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Extract the primary disease/condition. Return ONLY the name (lowercase) or "null".
Examples: breast cancer, lupus, alzheimer's disease
USER QUERY: "{user_query}"
Answer:"""
        response = await llm.ainvoke(prompt)
        result = response.content.strip().lower()
        invalid = ("null", "none", "n/a", "unknown", "not mentioned", "not specified")
        if result and result not in invalid:
            logger.info("Detected disease from query: %s", result)
            return result
        return None
    except Exception as e:
        logger.warning("Disease extraction failed: %s", e)
        return None


async def extract_is_causal_from_query(user_query: str) -> bool:
    """Detect whether user requests causal-focused analysis."""
    if not user_query:
        return False
    q = user_query.lower()
    neg = ["noncausal", "non-causal", "not causal", "no causal", "without causal"]
    if any(m in q for m in neg):
        return False
    pos = ["causal", "causality", "causal inference", "causal discovery", "causal effect", "cause effect", "cause-effect", "counterfactual"]
    if any(m in q for m in pos):
        return True
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Does the user request causal analysis? Return ONLY "true" or "false".
USER QUERY: "{user_query}"
Answer:"""
        response = await llm.ainvoke(prompt)
        result = response.content.strip().lower()
        if result in ("true", "yes"):
            return True
        if result in ("false", "no", "null", "none"):
            return False
    except Exception as e:
        logger.warning("Causal analysis detection failed: %s", e)
    return False


# Backward-compat aliases (plan_handler/core.py and others)
_extract_deconvolution_technique = extract_deconvolution_technique
_extract_biosample_from_query = extract_biosample_from_query
_extract_gse_ids_from_query = extract_gse_ids_from_query
_extract_disease_from_query = extract_disease_from_query
_extract_is_causal_from_query = extract_is_causal_from_query
