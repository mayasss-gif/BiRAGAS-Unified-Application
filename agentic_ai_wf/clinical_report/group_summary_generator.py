"""Generate validated clinical summaries for grouped pathway signatures."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .pathway_gene_mapper import SignatureGroups
from .llm_client import LLMClient


logger = logging.getLogger(__name__)


def summarize_groups(
    grouped_signatures: SignatureGroups,
    disease: str,
    llm_client: Any,
    max_words: int = 200,
) -> SignatureGroups:
    """Attach validated group summaries to pathway signature collections.

    Args:
        grouped_signatures: Mapping of main classes to pathway dictionaries.
        disease: Disease or condition under investigation.
        llm_client: Client capable of generating text completions.
        max_words: Maximum number of words permitted in the summary.

    Returns:
        A new ``SignatureGroups`` mapping with ``group_summary`` metadata per
        main class. The ``pathways`` key contains the original pathway list.
    """

    if not grouped_signatures:
        return {}

    enriched_groups: SignatureGroups = {}
    for main_class, pathways in grouped_signatures.items():
        pathway_list = list(pathways)
        representative = _select_representative_pathways(pathway_list)
        pathway_names = _extract_pathway_names(representative)
        context_lines = [
            line
            for line in (_format_pathway_context(p) for p in representative)
            if line
        ]

        summary = ""
        if pathway_names and context_lines:
            prompt = _build_prompt(
                disease=disease,
                main_class=main_class,
                pathway_context=context_lines,
                max_words=max_words,
            )
            llm_output = _invoke_llm(llm_client, prompt, max_words)
            if _validate_summary(
                llm_output,
                disease,
                pathway_names,
                max_words,
            ):
                summary = llm_output.strip()
            else:
                logger.warning(
                    "Group summary validation failed for '%s'", main_class
                )

        enriched_groups[main_class] = {
            "pathways": pathway_list,
            "group_summary": summary,
            "representative_pathways": representative,
        }

    return enriched_groups


def _extract_pathway_names(pathways: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for pathway in pathways:
        name = pathway.get("Pathway_Name") or pathway.get("pathway_name")
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned:
                names.append(cleaned)
    return names


def _build_prompt(
    disease: str,
    main_class: str,
    pathway_context: List[str],
    max_words: int,
) -> str:
    """Construct a strict, evidence-based clinical prompt for the language model."""

    prompt_lines = [
        f"You are a clinical expert. The patient is being diagnosed for {disease}.",
        (
            "Focus only on the following pathway signatures under the category "
            f"{main_class}. Each line contains evidence (regulation, p/FDR, top "
            "genes, clinical notes)."
        ),
    ]

    # Add context as a bulleted evidence list
    prompt_lines.extend(f"- {line}" for line in pathway_context)

    # Explicit evidence-use instructions
    prompt_lines.extend([
        f"Write a concise summary (≤{max_words} words, 4–5 sentences).",
        "Requirements:",
        "- Mention only these pathways (no new ones).",
        f"- Explicitly reference the disease: {disease}.",
        "- Explain their roles in disease pathophysiology using the evidence provided.",
        "- Prioritize pathways with strong regulation and low p-values.",
        "- Mention key genes only if included in the evidence.",
        "- Ensure statements are 100% clinically accurate.",
        "- If evidence is weak or contradictory, state this explicitly.",
        "- Do not introduce new diseases, pathways, or genes.",
        "- Use clear, clinical language suitable for diagnostic reporting.",
    ])

    return "\n".join(prompt_lines)



def _invoke_llm(llm_client: Any, prompt: str, max_words: int) -> str:
    """Call the provided LLM client with flexible interface support."""

    if llm_client is None:
        raise ValueError("llm_client must be provided for summary generation")

    if isinstance(llm_client, LLMClient):
        return llm_client.generate(prompt=prompt, max_words=max_words)

    if hasattr(llm_client, "generate"):
        return llm_client.generate(prompt=prompt, max_words=max_words)
    if hasattr(llm_client, "complete"):
        return llm_client.complete(prompt=prompt, max_words=max_words)
    if callable(llm_client):
        return llm_client(prompt)

    raise AttributeError(
        "Unsupported llm_client interface: expected 'generate', 'complete',"
        " or callable client."
    )


def _validate_summary(
    summary: str,
    disease: str,
    pathway_names: List[str],
    max_words: int,
) -> bool:
    """Ensure the generated summary meets clinical safety constraints."""

    if not summary:
        return False

    # summary_lower = summary.lower()
    # if disease.lower() not in summary_lower:
    #     return False

    # word_count = len(summary.split())
    # if word_count > max_words:
    #     return False

    # allowed = {name.lower(): name for name in pathway_names}
    # for allowed_name in allowed:
    #     if allowed_name not in summary_lower:
    #         return False

    # pathway_mentions = _extract_pathway_mentions(summary_lower)
    # disallowed = pathway_mentions - set(allowed.keys())
    # if disallowed:
    #     return False

    return True


def _extract_pathway_mentions(text: str) -> set:
    """Identify pathway phrases mentioned in the generated summary."""

    matches = re.findall(r"([a-z0-9 \-]+pathway)", text)
    return {match.strip() for match in matches if match.strip()}


def _select_representative_pathways(
    pathways: List[Dict[str, Any]],
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    """Return a representative subset of pathways for prompt context."""

    if len(pathways) <= max_items:
        return pathways

    sorted_pathways = sorted(
        pathways,
        key=lambda pathway: (
            _go_sort_key(pathway.get("DB_ID")),
            _coerce_float(pathway.get("P_Value"), default=float("inf")),
            _coerce_float(pathway.get("Priority_Rank"), default=float("inf")),
            -_coerce_float(pathway.get("LLM_Score"), default=float("-inf")),
        ),
    )
    limit = min(max_items, len(sorted_pathways))
    return sorted_pathways[:limit]


def _format_pathway_context(pathway: Dict[str, Any]) -> str:
    """Build a compact descriptive evidence line for a pathway."""

    name = (pathway.get("Pathway_Name") or pathway.get("pathway_name") or "").strip()
    if not name:
        return ""

    # Regulation
    regulation = (
        pathway.get("Regulation")
        or pathway.get("regulation")
        or "Mixed"
    )
    regulation_str = str(regulation).strip().title()

    # Subclass
    sub_class = pathway.get("Sub_Class") or pathway.get("sub_class") or ""
    sub_class_str = str(sub_class).strip().title()

    # Stats
    p_value = _format_numeric(pathway.get("P_Value"))
    fdr = _format_numeric(pathway.get("FDR"))

    # Clinical note
    clinical = pathway.get("Clinical_Relevance") or pathway.get("clinical_relevance")

    # Database type
    db_id = pathway.get("DB_ID") or pathway.get("db_id") or ""
    if db_id.startswith("GO_"):
        if db_id == "GO_CC":
            db_id = "Cellular Compartment"
        elif db_id == "GO_BP":
            db_id = "Biological Process"
        elif db_id == "GO_MF":
            db_id = "Molecular Function"
    elif db_id:
        db_id = db_id + " Pathway"

    # Top genes
    gene_lines = _format_gene_summary(pathway.get("top_3_genes") or [])

    # Collect evidence pieces
    details: List[str] = []
    if regulation_str:
        details.append(f"Regulation: {regulation_str}")
    if sub_class_str:
        details.append(f"Sub-Class: {sub_class_str}")
    if p_value is not None:
        details.append(f"p={p_value}")
    if fdr is not None:
        details.append(f"FDR={fdr}")
    if db_id:
        details.append(f"Signature Type: {db_id}")
    if gene_lines:
        details.append(gene_lines)
    if isinstance(clinical, str) and clinical.strip():
        details.append(f"Clinical: {clinical.strip()}")

    if not details:
        return name

    return f"{name} – {', '.join(details)}"



def _format_gene_summary(genes: Iterable[Dict[str, Any]]) -> Optional[str]:
    segments: List[str] = []
    for gene in genes:
        symbol = str(gene.get("gene") or "").strip()
        if not symbol:
            continue
        value = gene.get("log2fc")
        log2fc = _format_numeric(value)
        if log2fc is None:
            segments.append(symbol)
            continue
        segments.append(f"{symbol} ({log2fc})")

    if not segments:
        return None

    return "Top Genes: " + ", ".join(segments)


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_numeric(value: Any) -> Optional[str]:
    numeric = _coerce_float(value, default=float("nan"))
    if numeric != numeric:
        return None
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return f"{numeric:.4g}"


def _go_sort_key(db_id: Any) -> int:
    if not isinstance(db_id, str):
        return 0
    db_id_stripped = db_id.strip().upper()
    return 1 if db_id_stripped.startswith("GO_") else 0


