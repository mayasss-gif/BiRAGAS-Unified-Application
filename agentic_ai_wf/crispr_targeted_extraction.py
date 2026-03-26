"""
Extract targeted CRISPR-seq parameters from user query.

Supports: project_id (PRJNA/SRA), target_gene, protospacer, region (chr:start-end),
reference_seq. Used by meta_agent and agentic_views for crispr_targeted_analysis.
"""
import re
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROTOSPACER = "GGTGGATCCTATTCTAAACG"

# PRJNA, SRP, ERP, etc. + alphanumeric
PROJECT_PATTERN = re.compile(r"\b(PRJNA\d+|SRP\d+|ERP\d+)\b", re.I)
# Gene symbol: 2-10 uppercase letters (human)
GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
# Protospacer: 18-24 nt DNA (A/C/G/T only)
PROTOSPACER_PATTERN = re.compile(r"\b([ACGT]{18,24})\b")
# Region: chr12:34567890-34567900
REGION_PATTERN = re.compile(r"\b(chr[\w\d]+:\d+-\d+)\b", re.I)


def extract_crispr_targeted_params_sync(user_query: str) -> Dict[str, Any]:
    """
    Extract targeted CRISPR params from user query. Sync version for agentic_views.
    Returns dict with: project_id, target_gene, protospacer, region, reference_seq, mode, extract_metadata, download_fastq.
    """
    query = user_query or ""
    out: Dict[str, Any] = {
        "project_id": "",
        "target_gene": "",
        "protospacer": DEFAULT_PROTOSPACER,
        "region": "",
        "reference_seq": "",
        "mode": "gene",
        "extract_metadata": True,
        "download_fastq": True,
    }

    project_m = PROJECT_PATTERN.search(query)
    if project_m:
        out["project_id"] = project_m.group(1).upper()

    region_m = REGION_PATTERN.search(query)
    if region_m:
        out["region"] = region_m.group(1)
        out["mode"] = "region"

    _skip_genes = frozenset(("DNA", "RNA", "PCR", "CDNA", "MRNA", "CDS", "UTR", "CRISPR", "SRA", "GSE"))
    gene_matches = GENE_PATTERN.findall(query)
    for g in gene_matches:
        if g.upper() not in _skip_genes and len(g) <= 10:
            out["target_gene"] = g.upper()
            if out["mode"] != "region":
                out["mode"] = "gene"
            break

    proto_m = PROTOSPACER_PATTERN.search(query)
    if proto_m:
        out["protospacer"] = proto_m.group(1).upper()

    if "reference_seq" in query.lower() or "reference sequence" in query.lower():
        out["mode"] = "reference_seq"

    if "no download" in query.lower() or "skip download" in query.lower():
        out["download_fastq"] = False
    if "no metadata" in query.lower() or "skip metadata" in query.lower():
        out["extract_metadata"] = False

    return out


async def extract_crispr_targeted_params_async(user_query: str) -> Dict[str, Any]:
    """
    Async wrapper: tries regex first, falls back to LLM if project_id or target_gene missing.
    """
    sync_result = extract_crispr_targeted_params_sync(user_query)
    if sync_result["project_id"] and sync_result["target_gene"]:
        return sync_result
    if sync_result["project_id"] and sync_result["region"]:
        return sync_result
    if sync_result["project_id"] and sync_result["reference_seq"]:
        return sync_result

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""Extract targeted CRISPR-seq parameters from this query.

Return JSON only: {{"project_id":"","target_gene":"","protospacer":"","region":"","reference_seq":""}}

Rules:
- project_id: SRA/PRJNA ID (e.g. PRJNA1240319, SRP123456)
- target_gene: Gene symbol (e.g. RAB11A, TP53)
- protospacer: 18-24 nt gRNA sequence (A/C/G/T). Default: GGTGGATCCTATTCTAAACG
- region: chr:start-end (e.g. chr15:65869459-65891991)
- reference_seq: Direct DNA sequence if user provides one

Return empty string for missing. Only fill fields you can confidently infer.

Query: "{user_query}"

JSON:"""
        resp = await llm.ainvoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```\w*\n?", "", raw).strip()
        data = json.loads(raw)
        if data.get("project_id"):
            sync_result["project_id"] = str(data["project_id"]).upper()
        if data.get("target_gene"):
            sync_result["target_gene"] = str(data["target_gene"]).upper()
        if data.get("protospacer"):
            sync_result["protospacer"] = str(data["protospacer"]).upper()
        if data.get("region"):
            sync_result["region"] = str(data["region"])
            sync_result["mode"] = "region"
        if data.get("reference_seq"):
            sync_result["reference_seq"] = str(data["reference_seq"])
            sync_result["mode"] = "reference_seq"
        if sync_result["target_gene"] and sync_result["mode"] not in ("region", "reference_seq"):
            sync_result["mode"] = "gene"
    except Exception as e:
        logger.warning(f"LLM extraction failed for crispr_targeted: {e}")

    return sync_result


def validate_crispr_targeted_params(params: Dict[str, Any]) -> Optional[str]:
    """Return error message if params invalid, else None."""
    if not params.get("project_id"):
        return "targeted CRISPR requires a project/SRA ID (e.g. PRJNA1240319)"
    mode = params.get("mode", "gene")
    if mode == "gene" and not params.get("target_gene"):
        return "targeted CRISPR in gene mode requires target_gene (e.g. RAB11A)"
    if mode == "region" and not params.get("region"):
        return "targeted CRISPR in region mode requires region (chr:start-end)"
    if mode == "reference_seq" and not params.get("reference_seq"):
        return "targeted CRISPR in reference_seq mode requires reference_seq"
    if not params.get("protospacer"):
        return "protospacer (gRNA sequence) is required"
    return None
