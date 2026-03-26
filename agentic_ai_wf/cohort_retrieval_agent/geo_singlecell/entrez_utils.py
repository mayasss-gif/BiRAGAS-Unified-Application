import os
import time
from typing import List, Dict, Any

import requests

from .constants import EUTILS_BASE


def esearch_gds(query: str, retmax: int, logger) -> Dict[str, Any]:
    params = {
        "db": "gds",
        "term": query,
        "retmax": retmax,
        "usehistory": "y",
        "retmode": "json",
    }
    ncbi_key = os.getenv("NCBI_API_KEY")
    if ncbi_key:
        params["api_key"] = ncbi_key

    url = f"{EUTILS_BASE}/esearch.fcgi"
    logger.info("Calling ESearch at %s", url)
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def esummary_gds(webenv: str, query_key: str, retstart: int, retmax: int,
                 logger) -> Dict[str, Any]:
    params = {
        "db": "gds",
        "WebEnv": webenv,
        "query_key": query_key,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
    }
    ncbi_key = os.getenv("NCBI_API_KEY")
    if ncbi_key:
        params["api_key"] = ncbi_key

    url = f"{EUTILS_BASE}/esummary.fcgi"
    logger.info("Calling ESummary at %s (retstart=%d, retmax=%d)",
                url, retstart, retmax)
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def extract_gse_records_from_summary(summary_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From one ESummary JSON response, extract a list of GSE-like records
    and also keep the full esummary item as 'esummary'.
    """
    result = summary_json.get("result", {})
    uids = [uid for uid in result.get("uids", [])]

    records: List[Dict[str, Any]] = []
    for uid in uids:
        item = result.get(uid, {})
        acc = item.get("accession")
        if not acc or not str(acc).upper().startswith("GSE"):
            continue

        rec = {
            "uid": uid,
            "accession": acc,
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "gdstype": item.get("gdstype", ""),
            "taxon": item.get("taxon", ""),
            "entrytype": item.get("entrytype", ""),
            "esummary": item,  # raw esummary
        }
        records.append(rec)
    return records


def collect_all_gse_records(webenv: str, query_key: str, total: int,
                            logger,
                            batch_size: int = 100) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    for start in range(0, total, batch_size):
        this_max = min(batch_size, total - start)
        js = esummary_gds(webenv, query_key, retstart=start,
                          retmax=this_max, logger=logger)
        all_records.extend(extract_gse_records_from_summary(js))
        time.sleep(0.34)
    return all_records


def filter_gse_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep:
      - Homo sapiens
      - Expression profiling by high throughput sequencing
      - entrytype that looks like a Series.
    """
    out = []
    for r in records:
        tax = r.get("taxon", "").lower()
        gdstype = r.get("gdstype", "").lower()
        entrytype = (r.get("entrytype") or "").upper()

        if "homo sapiens" not in tax:
            continue
        if "expression profiling by high throughput sequencing" not in gdstype:
            continue
        if "GSE" not in entrytype and not r["accession"].upper().startswith("GSE"):
            continue
        out.append(r)
    return out


def filter_single_cell_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Single-cell / 10x heuristic using title/summary keywords.
    """
    sc_keywords = [
        "single-cell", "single cell", "single-cell rna", "scrna",
        "scrna-seq", "scrna seq", "sc-rna", "snrna", "snrna-seq",
        "single nucleus", "10x", "10x genomics", "droplet", "chromium"
    ]
    out = []
    for r in records:
        txt = (r.get("title", "") + " " + r.get("summary", "")).lower()
        if any(kw in txt for kw in sc_keywords):
            out.append(r)
    return out
