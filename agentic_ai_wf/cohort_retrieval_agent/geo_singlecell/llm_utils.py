from .constants import DEFAULT_MODEL

import os
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI



def init_openai_client(logger) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Put it in .env or env vars.")
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def validate_biomedical_query(client: OpenAI, user_query: str,
                              logger) -> bool:
    """
    Use LLM to decide if user_query is a biomedical / disease / scRNA-type
    query. If not, we don't hit GEO at all.
    """
    system_prompt = (
        "You are a strict classifier. Classify whether a query is a valid "
        "biomedical / omics / disease / tissue / cell-type / single-cell "
        "analysis query.\n\n"
        "Return exactly one token:\n"
        "- BIO if it is clearly about biology, disease, genes, cells, tissues, "
        "  cancer, single-cell or RNA-seq data.\n"
        "- NON_BIO if it is about sports, time, movies, generic chat, or "
        "  anything not related to biomedical data.\n"
        "DO NOT add explanations.\n"
    )

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
        max_tokens=4,
    )
    label = (resp.choices[0].message.content or "").strip().upper()
    logger.info("Query classification result for %r: %s", user_query, label)
    return label == "BIO"


def build_entrez_query_with_llm(client: OpenAI, user_query: str,
                                logger) -> str:
    """
    LLM → Entrez query string for GEO DataSets (db=gds).
    Enforces Homo sapiens + expression profiling by HTS.
    """
    system_prompt = (
        "You are an expert NCBI GEO query builder. "
        "Given a free-text biomedical description, you must output EXACTLY ONE "
        "Entrez query string suitable for the GEO DataSets database (db=gds).\n\n"
        "Rules:\n"
        "1. Always restrict to Homo sapiens.\n"
        '   Use: \"Homo sapiens\"[organism]\n'
        "2. Always restrict to datasets with expression profiling by high throughput sequencing.\n"
        '   Use: \"expression profiling by high throughput sequencing\"[DataSet Type]\n'
        "3. Include the main disease / condition keywords from the input.\n"
        "4. Do NOT include any explanations, comments, or markdown. Output ONLY the query.\n"
        "5. You can use boolean operators AND / OR and quotes where needed.\n"
    )

    user_prompt = f"Free-text description: {user_query}\n\nReturn only the Entrez query string."

    logger.info("Requesting LLM to build Entrez query for GEO search...")
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    query = (resp.choices[0].message.content or "").strip()

    if not query or len(query) < 5:
        logger.warning("LLM returned short/empty query, using fallback.")
        query = (
            f"({user_query}) AND \"Homo sapiens\"[organism] "
            f"AND \"expression profiling by high throughput sequencing\"[DataSet Type]"
        )

    logger.info("LLM-derived Entrez query: %s", query)
    return query


def normalize_biomedical_query(client: OpenAI, user_query: str,
                               logger) -> str:
    """
    Fixes spelling mistakes, expands abbreviations, and ensures a clean biomedical query.
    Returns a cleaned version of the query without changing meaning.
    """
    system_prompt = (
        "You are an expert biomedical text normalizer. "
        "Your job is to correct spelling mistakes, fix typos, expand abbreviations, "
        "and normalize disease/tissue/cell-type names WITHOUT changing the meaning.\n\n"
        "Rules:\n"
        "1. Only fix spelling and formatting.\n"
        "2. Do NOT add new concepts.\n"
        "3. Do NOT rewrite or summarize the text.\n"
        "4. Output ONLY the corrected query text.\n"
    )

    user_prompt = f"Normalize the biomedical query:\n{user_query}"

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        normalized = (resp.choices[0].message.content or "").strip()
        if not normalized:
            return user_query
        return normalized
    except Exception as e:
        logger.error("Error normalizing query with LLM: %s", e)
        return user_query


def rank_gse_records_with_llm(client: OpenAI, user_query: str,
                              records: List[Dict[str, Any]],
                              logger) -> List[Dict[str, Any]]:
    if not records:
        return records

    lines = []
    for idx, r in enumerate(records, start=1):
        snippet = (r.get("summary") or "").replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"{idx}. {r['accession']} | {r['title']} | {snippet}")
    records_text = "\n".join(lines)

    system_prompt = (
        "You are an expert curator of GEO single-cell datasets. "
        "Given a user query and a list of GEO Series (accession, title, summary), "
        "you must rank them from most to least relevant.\n\n"
        "Return ONLY a comma-separated list of indices (e.g., '2,1,3')."
    )
    user_prompt = (
        f"User query: {user_query}\n\n"
        "Candidate GEO Series (index, accession, title, summary snippet):\n"
        f"{records_text}\n\n"
        "Return a comma-separated list of the indices in best-to-worst order."
    )

    logger.info("Requesting LLM to rank %d candidate GSEs by relevance...", len(records))
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    content = (resp.choices[0].message.content or "").strip()
    logger.info("LLM ranking response: %s", content)

    indices: List[int] = []
    for part in re.split(r"[,\s]+", content):
        if part.isdigit():
            i = int(part)
            if 1 <= i <= len(records):
                indices.append(i)

    seen = set()
    ordered: List[Dict[str, Any]] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(records[i - 1])

    remaining = [r for r in records if r not in ordered]
    ordered.extend(remaining)
    return ordered
