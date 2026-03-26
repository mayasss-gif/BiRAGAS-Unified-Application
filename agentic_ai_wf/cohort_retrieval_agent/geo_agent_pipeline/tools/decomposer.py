"""
decomposer.py

Utility to detect whether a natural-language cohort request
is SINGLE-condition or MULTIPLE-condition, and to split it
into sub-queries accordingly.

Usage example (inside your CLI script):

    from decomposer import decompose_cohort_query

    decomp = await decompose_cohort_query(user_query)
    if decomp["request_type"] == "single":
        query_list = [("Full query", user_query)]
    else:
        query_list = list(zip(decomp["conditions"], decomp["sub_queries"]))
"""

import json
from typing import List, Literal, TypedDict
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file in this folder
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment!")

client = AsyncOpenAI(api_key=API_KEY)

class DecomposedQuery(TypedDict):
    request_type: Literal["single", "multiple"]
    conditions: List[str]
    sub_queries: List[str]



async def decompose_cohort_query(
    user_query: str,
    model: str = "gpt-4.1",  # use a stronger model by default
) -> DecomposedQuery:
    """
    Call the OpenAI Responses API to decide if the user_query is:
      - a single-condition request, or
      - a multi-condition request (and then split it accordingly).

    Returns:
        DecomposedQuery dict:
        {
            "request_type": "single" | "multiple",
            "conditions": [...],
            "sub_queries": [...]
        }

    In case of any parsing/model error, this will fall back to treating the
    query as a single-condition request with the original query.
    """

    DECOMPOSER_SYSTEM_PROMPT = """
    You are a query decomposition assistant for a GEO cohort retrieval system.

    Your task:
    - Decide if the user's request is about ONE condition/disease/topic or MULTIPLE.
    - If MULTIPLE, split it into separate sub-queries, each focused on a SINGLE condition/disease/topic.
    - Preserve the original intent, wording, and structure as much as possible, only substituting the condition phrase.

    DEFINITIONS
    - A "condition" can be:
    - a disease (e.g., "breast cancer"),
    - a biological state (e.g., "inflammation"),
    - an outcome/topic (e.g., "drug response")
    that the user wants cohorts for.
    - The user may list multiple conditions using commas, "and", "or", or inside patterns like:
    "e.g.", "such as", "for example", "like".

    STRICT RULES FOR MULTIPLE CONDITIONS
    - If you detect MORE THAN ONE distinct condition phrase, you MUST set "request_type" to "multiple".
    - This is true EVEN IF the user only wrote ONE sentence.
    - Conditions separated by commas, slashes, the word "and",
    or listed inside parentheses after "e.g."/"such as"/"for example"/"like"
    MUST be treated as separate conditions.

    HOW TO BUILD SUB-QUERIES (VERY IMPORTANT)
    - For "multiple" requests:
    - Extract each distinct condition as a clean phrase,
        e.g., "breast cancer", "drug response", "inflammation".
    - For each condition, build a sub_query that:
        * is a valid stand-alone natural-language query, AND
        * preserves all other details from the original query
        (e.g., "Automatically find RNA-seq cohorts in GEO matching specific keywords"),
        changing ONLY the condition phrase.

    - For "single" requests:
    - If there is ONLY ONE distinct condition, set "request_type" to "single".
    - The "sub_queries" array MUST contain exactly one element which is the
        original query (or a very close equivalent).

    DO NOT
    - Do NOT treat the entire user query sentence as a single "condition".
    - Do NOT invent conditions that are not clearly present.
    - Do NOT merge multiple conditions into a single condition string.

    EXAMPLES (FOLLOW THESE EXACTLY)

    Example 1 (MULTIPLE):
    User query:
    "Automatically find RNA-seq cohorts in GEO matching specific keywords (e.g., 'breast cancer', 'drug response', 'inflammation')"

    Your output MUST be:
    {
    "request_type": "multiple",
    "conditions": [
        "breast cancer",
        "drug response",
        "inflammation"
    ],
    "sub_queries": [
        "Automatically find RNA-seq cohorts in GEO matching specific keywords 'breast cancer'.",
        "Automatically find RNA-seq cohorts in GEO matching specific keywords 'drug response'.",
        "Automatically find RNA-seq cohorts in GEO matching specific keywords 'inflammation'."
    ]
    }

    Example 2 (SINGLE):
    User query:
    "Automatically find RNA-seq cohorts in GEO matching specific keywords 'breast cancer'"

    Your output MUST be:
    {
    "request_type": "single",
    "conditions": [
        "breast cancer"
    ],
    "sub_queries": [
        "Automatically find RNA-seq cohorts in GEO matching specific keywords 'breast cancer'"
    ]
    }

    OUTPUT FORMAT (MANDATORY)
    - You MUST output ONLY valid JSON with this exact schema:

    {
    "request_type": "single or multiple",
    "conditions": ["list of condition strings you detected"],
    "sub_queries": ["list of rewritten sub-queries, one per condition"]
    }

    - "request_type" MUST be exactly either "single" or "multiple" (lowercase).
    """


    try:
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0,
        )
        print(resp)
        # Adjust this access pattern if your SDK version differs
        raw = resp.output[0].content[0].text
        data = json.loads(raw)

        # Minimal validation / normalization
        request_type_raw = data.get("request_type", "")
        request_type = str(request_type_raw).strip().lower()
        if request_type not in {"single", "multiple"}:
            request_type = "single"

        conditions = data.get("conditions") or []
        sub_queries = data.get("sub_queries") or []

        # If model says "multiple", it MUST return aligned lists
        if request_type == "multiple":
            if not conditions or not sub_queries or len(conditions) != len(sub_queries):
                # If it's inconsistent, degrade gracefully to single
                request_type = "single"

        # Final safety: if single but no sub_queries, just use original query
        if request_type == "single" and not sub_queries:
            conditions = [user_query]
            sub_queries = [user_query]

        return DecomposedQuery(
            request_type=request_type,  # type: ignore[arg-type]
            conditions=conditions,
            sub_queries=sub_queries,
        )

    except Exception as e:
        print("Inside error",e)
        # Fallback: treat as single query if anything goes wrong
        return DecomposedQuery(
            request_type="single",
            conditions=[user_query],
            sub_queries=[user_query],
        )


# Optional: quick manual test
if __name__ == "__main__":
    import asyncio

    async def _test():
        q = (
            "Automatically find 1 RNA-seq cohorts in GEO matching specific keywords "
            "in 'breast cancer'"
        )
        print("Before function call")
        decomp = await decompose_cohort_query(q)
        print(json.dumps(decomp, indent=2))

    asyncio.run(_test())
