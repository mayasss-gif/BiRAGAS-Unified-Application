#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch sample metadata from the NCBI GEO API.

Uses two complementary approaches:
  1. NCBI E-utilities (esearch + esummary) for structured metadata.
  2. Direct GEO SOFT-format query for the full sample record.

Results are cached to disk so repeated report builds don't re-hit the API.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_QUERY_BASE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

REQUEST_TIMEOUT = 30


# ── E-utilities helpers ──────────────────────────────────────────────────────

def esearch_geo(gsm_id: str) -> Optional[str]:
    """Search GEO DataSets for a GSM accession and return the internal UID."""
    params = {
        "db": "gds",
        "term": f"{gsm_id}[ACCN]",
        "retmode": "json",
    }
    resp = requests.get(
        f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    return id_list[0] if id_list else None


def esummary_geo(uid: str) -> dict:
    """Retrieve the document summary for a GEO UID."""
    params = {
        "db": "gds",
        "id": uid,
        "retmode": "json",
    }
    resp = requests.get(
        f"{NCBI_BASE}/esummary.fcgi", params=params, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("result", {})
    return results.get(uid, results)


# ── Direct GEO SOFT-format fetch ────────────────────────────────────────────

def fetch_geo_soft(gsm_id: str) -> str:
    """Download the full SOFT-format text record for a GSM accession."""
    params = {
        "acc": gsm_id,
        "targ": "self",
        "form": "text",
        "view": "full",
    }
    resp = requests.get(
        GEO_QUERY_BASE, params=params, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text


def parse_soft_record(soft_text: str) -> dict:
    """Parse key=value pairs from a SOFT-format text block into a dict."""
    record: Dict[str, List[str]] = {}
    for line in soft_text.splitlines():
        if line.startswith("!"):
            line = line[1:]
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                record.setdefault(key, []).append(value)
    return {k: (v[0] if len(v) == 1 else v) for k, v in record.items()}


# ── Main public API ──────────────────────────────────────────────────────────

def get_gsm_info(gsm_id: str) -> dict:
    """
    Return all available information for the given GSM accession.

    Returns a dict with two top-level keys:
      - "esummary":    structured metadata from E-utilities (may be None)
      - "soft_record": parsed SOFT-format fields
    """
    gsm_id = gsm_id.strip().upper()
    if not gsm_id.startswith("GSM"):
        raise ValueError(f"Expected a GSM accession, got: {gsm_id}")

    info: dict = {}

    uid = esearch_geo(gsm_id)
    if uid:
        info["esummary"] = esummary_geo(uid)
    else:
        info["esummary"] = None
        logger.info(
            "No GDS record found via esearch for %s; "
            "this is normal for samples not linked to a DataSet.",
            gsm_id,
        )

    soft_text = fetch_geo_soft(gsm_id)
    info["soft_record"] = parse_soft_record(soft_text)

    return info


def extract_gsm_id(sample_name: str) -> Optional[str]:
    """
    Extract the bare GSM accession from a sample directory name.

    Examples:
        "GSM2406675_10X001" -> "GSM2406675"
        "GSM2406675"        -> "GSM2406675"
        "some_other_name"   -> None
    """
    parts = sample_name.split("_")
    if parts and parts[0].upper().startswith("GSM"):
        return parts[0].upper()
    return None


def soft_record_to_text(soft: dict) -> str:
    """
    Convert a parsed SOFT record dict into a readable text summary
    suitable for feeding into an LLM prompt.
    """
    lines = []
    for key, value in soft.items():
        clean_key = key.replace("Sample_", "").replace("_", " ").title()
        if isinstance(value, list):
            lines.append(f"{clean_key}: {'; '.join(value)}")
        else:
            lines.append(f"{clean_key}: {value}")
    return "\n".join(lines)


# ── Cached fetch (avoids hitting API on every report rebuild) ────────────────

def fetch_sample_info_cached(
    sample_dir: Path,
    gsm_id: Optional[str] = None,
) -> str:
    """
    Return a text description of the GEO sample for use in LLM prompts.

    1. Checks for a cached file in ``<sample_dir>/.geo_sample_info.txt``.
    2. If absent, fetches from NCBI, caches the result, and returns it.
    3. Returns an empty string on any network/parsing failure.

    Parameters
    ----------
    sample_dir : Path
        The sample output directory (e.g., ``crispr_output/GSE90546/GSM2406675_10X001``).
    gsm_id : str, optional
        Override the GSM accession.  If None, extracted from ``sample_dir.name``.
    """
    cache_path = sample_dir / ".geo_sample_info.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    if gsm_id is None:
        gsm_id = extract_gsm_id(sample_dir.name)
    if gsm_id is None:
        logger.warning(
            "Could not determine GSM ID from directory name '%s'",
            sample_dir.name,
        )
        return ""

    try:
        logger.info("Fetching GEO metadata for %s ...", gsm_id)
        info = get_gsm_info(gsm_id)
        soft = info.get("soft_record", {})

        if not soft:
            return ""

        text = soft_record_to_text(soft)

        cache_path.write_text(text, encoding="utf-8")
        logger.info("Cached GEO metadata to %s", cache_path)
        return text

    except Exception as exc:
        logger.warning("Failed to fetch GEO metadata for %s: %s", gsm_id, exc)
        return ""
