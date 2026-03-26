#!/usr/bin/env python3
"""
pathway_classifier.py

Two modes:

1) Single-file mode:
   python pathway_classifier.py --input <file>

   - Detect schema (legacy Pathway+DB_ID or term-based)
   - Run KEGG + LLM-based classification
   - Write classifications back to the SAME file (in-place).

2) Root-dir mode:
   python pathway_classifier.py --root-dir <MDP_OUT_ROOT>

   - Treat <root-dir> as an mdp_pipeline_3 OUT_ROOT
   - For each disease folder (excluding baseline_consensus, comparison, results, agentic_analysis):
       * Look for:
           - gsea_prerank.tsv
           - core_enrich_up.csv
           - core_enrich_down.csv
       * Apply mild significance filters:
           - core_enrich_*: qval <= q_thresh_core (default 0.2), cap max_rows
           - gsea_prerank: FDR q-val <= q_thresh_gsea (default 0.3), cap max_rows
       * Run classification ONLY on filtered rows
       * Save new classified files:
           - gsea_prerank_classified.tsv
           - core_enrich_up_classified.csv
           - core_enrich_down_classified.csv

   - Original files remain unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import argparse
import openai
import pandas as pd
import requests
from rich import print
from tqdm import tqdm

# ============================================================
# SELF-CONTAINED CONFIG + LOGGER (no external module imports)
# ============================================================

# You can set DATASET_PATH via environment variable, or it defaults to current folder.
DATASET_PATH = os.getenv("DATASET_PATH", str(Path(__file__).resolve().parent))
CATEGORIZER_CSV_PATH = str(Path(DATASET_PATH) / "classification_memory.csv")

logger = logging.getLogger("pathway_categorizer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(levelname)s] %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# === DATABASE SETUP ===
if os.path.dirname(CATEGORIZER_CSV_PATH):
    os.makedirs(os.path.dirname(CATEGORIZER_CSV_PATH), exist_ok=True)


def init_csv_cache() -> None:
    """Initialize the CSV cache file with proper headers if it doesn't exist."""
    try:
        if not Path(CATEGORIZER_CSV_PATH).exists():
            cache_df = pd.DataFrame(
                columns=["pathway", "main_class", "sub_class", "source", "confidence", "created_at"]
            )
            cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)
            print("[yellow]Created new CSV cache file[/yellow]")
        else:
            cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
            changed = False

            required_cols = ["pathway", "main_class", "sub_class", "source", "confidence", "created_at"]
            for col in required_cols:
                if col not in cache_df.columns:
                    if col == "confidence":
                        cache_df[col] = 1.0
                    elif col == "created_at":
                        cache_df[col] = datetime.now().isoformat()
                    else:
                        cache_df[col] = ""
                    changed = True

            if changed:
                cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)

    except Exception as e:
        print(f"[red]Error initializing CSV cache: {e}[/red]")


# Initialize the cache
init_csv_cache()


# === TERM/SCHEMA HELPERS ===
def parse_term_to_source_and_pathway(term: str) -> Tuple[str, str]:
    """
    Parse a GSEA-style 'term' value like:
      'Reactome_2022:Expression And Translocation Of Olfactory Receptors R-HSA-9752946'
    Returns: (source_token, pathway_name)
    """
    term = (term or "").strip()
    if not term:
        return "UNKNOWN", ""

    if ":" in term:
        src, name = term.split(":", 1)  # split ONLY on first colon
        return src.strip(), name.strip()

    return "UNKNOWN", term


def normalize_ontology(source_token: str) -> str:
    """
    Map source token to a canonical ontology label for grouping/routing.
    This is what gets passed into the LLM as `ontology`.
    """
    s = (source_token or "").strip().upper()

    if s.startswith("REACTOME"):
        return "REACTOME"
    if s.startswith("GO_BIOLOGICAL_PROCESS"):
        return "GO_BP"
    if s.startswith("GO_MOLECULAR_FUNCTION"):
        return "GO_MF"
    if s.startswith("GO_CELLULAR_COMPONENT"):
        return "GO_CC"
    if "WIKIPATHWAY" in s or s.startswith("WIKIPATHWAYS"):
        return "WIKIPATHWAY"
    if s.startswith("KEGG"):
        return "KEGG"
    if "HALLMARK" in s:
        return "HALLMARK"

    return source_token.strip() or "GO"


def detect_schema(df: pd.DataFrame) -> str:
    """
    Returns:
      - 'legacy' if df has Pathway + DB_ID
      - 'term' if df has term (case-insensitive)
    Raises ValueError if neither schema is supported.
    """
    cols_lower = {c.lower() for c in df.columns}

    if "pathway" in cols_lower and "db_id" in cols_lower:
        return "legacy"
    if "term" in cols_lower:
        return "term"

    raise ValueError("Unsupported schema: need ['Pathway','DB_ID'] or at least ['term'].")


def _resolve_col(df: pd.DataFrame, wanted_lower: str) -> str:
    """Return the real column name in df that matches wanted_lower case-insensitively."""
    for c in df.columns:
        if c.lower() == wanted_lower:
            return c
    raise KeyError(f"Column '{wanted_lower}' not found (case-insensitive). Columns: {list(df.columns)}")


# === UTILITY FUNCTIONS ===
def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def insert_into_cache(
    pathway: str,
    main_class: str,
    sub_class: str,
    source: str,
    confidence: float = 1.0,
) -> None:
    """Insert pathway classification into cache."""
    try:
        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)

        for col in ["pathway", "main_class", "sub_class", "source", "confidence", "created_at"]:
            if col not in cache_df.columns:
                cache_df[col] = ""

        if pathway in cache_df["pathway"].values:
            cache_df.loc[cache_df["pathway"] == pathway, "main_class"] = main_class
            cache_df.loc[cache_df["pathway"] == pathway, "sub_class"] = sub_class
            cache_df.loc[cache_df["pathway"] == pathway, "source"] = source
            cache_df.loc[cache_df["pathway"] == pathway, "confidence"] = confidence
            cache_df.loc[cache_df["pathway"] == pathway, "created_at"] = datetime.now().isoformat()
        else:
            new_row = pd.DataFrame(
                {
                    "pathway": [pathway],
                    "main_class": [main_class],
                    "sub_class": [sub_class],
                    "source": [source],
                    "confidence": [confidence],
                    "created_at": [datetime.now().isoformat()],
                }
            )
            cache_df = pd.concat([cache_df, new_row], ignore_index=True)

        cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)

    except Exception as e:
        print(f"[red]Error inserting into cache: {e}[/red]")


def find_similar_pathway(pathway: str, threshold: float = 0.8) -> Optional[Tuple[str, str, str, str, float]]:
    """Find similar pathways in cache using fuzzy matching."""
    try:
        if not Path(CATEGORIZER_CSV_PATH).exists():
            return None

        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        if cache_df.empty:
            return None

        best_match: Optional[Tuple[str, str, str, str, float]] = None
        best_similarity = 0.0

        for _, row in cache_df.iterrows():
            similarity = calculate_similarity(pathway, str(row["pathway"]))
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = (
                    str(row["pathway"]),
                    str(row["main_class"]),
                    str(row["sub_class"]),
                    str(row["source"]),
                    float(similarity),
                )

        return best_match
    except Exception as e:
        print(f"[red]Error finding similar pathway: {e}[/red]")
        return None


def get_cached_pathway(pathway: str) -> Optional[Tuple[str, str, str]]:
    """Get cached pathway classification."""
    try:
        if not Path(CATEGORIZER_CSV_PATH).exists():
            return None

        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        if cache_df.empty:
            return None

        match = cache_df[cache_df["pathway"] == pathway]
        if not match.empty:
            row = match.iloc[0]
            return (str(row["main_class"]), str(row["sub_class"]), str(row["source"]))

        return None
    except Exception as e:
        print(f"[red]Error getting cached pathway: {e}[/red]")
        return None


# === ENHANCED KEGG CLASSIFICATION ===
def get_kegg_classification(pathway_name: str) -> Optional[Tuple[str, str, str]]:
    """Enhanced KEGG classification with better error handling."""
    try:
        cached_result = get_cached_pathway(pathway_name)
        if cached_result:
            return cached_result

        similar = find_similar_pathway(pathway_name, threshold=0.85)
        if similar:
            cached_pathway, main_class, sub_class, source, similarity = similar
            print(f"[yellow] Using fuzzy match ({similarity:.2f}): {cached_pathway}[/yellow]")
            insert_into_cache(pathway_name, main_class, sub_class, source, similarity)
            return main_class, sub_class, source

        main_class = ""
        sub_class = ""

        # Method 1
        try:
            url = f"http://rest.kegg.jp/find/pathway/{pathway_name}"
            response = requests.get(url, timeout=10)
            if response.ok and response.text.strip():
                first_line = response.text.strip().split("\n")[0]
                kegg_id = first_line.split("\t")[0]

                detail_url = f"http://rest.kegg.jp/get/{kegg_id}"
                detail_resp = requests.get(detail_url, timeout=10)
                if detail_resp.ok:
                    lines = detail_resp.text.split("\n")
                    class_line = [line for line in lines if line.startswith("CLASS")]
                    if class_line:
                        class_info = class_line[0].replace("CLASS       ", "").strip()
                        if ";" in class_info:
                            parts = class_info.split(";")
                            main_class = parts[0].strip()
                            sub_class = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            main_class = class_info
                            sub_class = "General Process"
        except Exception as e:
            print(f"[yellow] KEGG Method 1 failed for {pathway_name}: {e}[/yellow]")

        # Method 2 (simplify name)
        if not main_class:
            try:
                simplified_name = pathway_name.replace("pathway", "").replace("signaling", "").strip()
                url = f"http://rest.kegg.jp/find/pathway/{simplified_name}"
                response = requests.get(url, timeout=10)
                if response.ok and response.text.strip():
                    first_line = response.text.strip().split("\n")[0]
                    kegg_id = first_line.split("\t")[0]

                    detail_url = f"http://rest.kegg.jp/get/{kegg_id}"
                    detail_resp = requests.get(detail_url, timeout=10)
                    if detail_resp.ok:
                        lines = detail_resp.text.split("\n")
                        class_line = [line for line in lines if line.startswith("CLASS")]
                        if class_line:
                            class_info = class_line[0].replace("CLASS       ", "").strip()
                            if ";" in class_info:
                                parts = class_info.split(";")
                                main_class = parts[0].strip()
                                sub_class = parts[1].strip() if len(parts) > 1 else ""
                            else:
                                main_class = class_info
                                sub_class = "General Process"
            except Exception as e:
                print(f"[yellow] KEGG Method 2 failed for {pathway_name}: {e}[/yellow]")

        if main_class:
            main_class = main_class.strip()
            sub_class = sub_class.strip() if sub_class else "General Process"
            if not sub_class:
                sub_class = "General Process"

            insert_into_cache(pathway_name, main_class, sub_class, "KEGG", 1.0)
            time.sleep(0.5)
            return main_class, sub_class, "KEGG"

        print(f"[red] KEGG classification failed completely for: {pathway_name}[/red]")
        return None

    except Exception as e:
        print(f"[red] KEGG error for {pathway_name}: {e}[/red]")
        return None


# === BATCH LLM CLASSIFICATION ===
def batch_classify_pathways_llm(
    pathways: List[Tuple[str, str]],
    batch_size: int = 10,
) -> Dict[str, Tuple[str, str, str]]:
    """Classify multiple pathways in a single LLM call to minimize API usage."""
    if not pathways:
        return {}

    results: Dict[str, Tuple[str, str, str]] = {}

    for i in range(0, len(pathways), batch_size):
        batch = pathways[i : i + batch_size]
        pathways_text = "\n".join(
            [f"{j+1}. {pathway} ({ontology})" for j, (pathway, ontology) in enumerate(batch)]
        )

        prompt = f"""
You are a biomedical ontology expert. Classify each of the following pathways into a **Main_Class** and **Sub_Class** based on biological function.

Use one of the following functional categories for Main_Class (choose only one per pathway):
- Metabolism
- Signal Transduction
- Immune System
- Cell Cycle
- Cancer
- Apoptosis
- Developmental Biology
- Genetic Information Processing
- Environmental Information Processing
- Cellular Processes
- Disease Mechanisms

Guidelines for Sub_Class:
- Sub_Class must describe a more specific role/pathway type/molecular mechanism under the Main_Class.
- Sub_Class must not simply repeat or rephrase the pathway name.
- DO NOT leave any Main_Class or Sub_Class empty.

Pathways to classify:
{pathways_text}

Return format (strictly one line per pathway):
1. Main_Class: [Main_Class] | Sub_Class: [Sub_Class]
...
up to {len(batch)} total
"""

        try:
            client = openai.OpenAI()
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

            text = res.choices[0].message.content
            lines = text.strip().split("\n")

            parsed_count = 0
            for line in lines:
                if parsed_count >= len(batch):
                    break

                if not line.strip() or "|" not in line:
                    continue

                try:
                    parts = line.split("|")
                    if len(parts) < 2:
                        continue

                    main_part = parts[0].strip()
                    sub_part = parts[1].strip()

                    if ":" in main_part:
                        main_class = main_part.split(":", 1)[1].strip()
                    else:
                        main_class = main_part.strip()

                    if ":" in sub_part:
                        sub_class = sub_part.split(":", 1)[1].strip()
                    else:
                        sub_class = sub_part.strip()

                    main_class = re.sub(r"^\d+\.?\s*", "", main_class).strip()
                    sub_class = re.sub(r"^\d+\.?\s*", "", sub_class).strip()

                    if main_class and sub_class:
                        pathway, ontology = batch[parsed_count]
                        results[pathway] = (main_class, sub_class, ontology)
                        insert_into_cache(pathway, main_class, sub_class, ontology, 0.9)
                        parsed_count += 1

                except Exception as e:
                    print(f"[red] Error parsing line: {line} - {e}[/red]")
                    continue

            if parsed_count < len(batch):
                print(
                    f"[yellow] Only parsed {parsed_count}/{len(batch)} pathways from batch, "
                    f"using fallback for remaining[/yellow]"
                )
                for j in range(parsed_count, len(batch)):
                    pathway, ontology = batch[j]
                    fallback_result = classify_single_pathway_llm(pathway, ontology)
                    if fallback_result[0] and fallback_result[1]:
                        results[pathway] = fallback_result
                    else:
                        results[pathway] = ("Cellular Processes", "Unknown Process", ontology)
                        insert_into_cache(pathway, "Cellular Processes", "Unknown Process", ontology, 0.5)

            print(f"[green] Batch processed: {len(results)} pathways classified[/green]")
            time.sleep(1)

        except Exception as e:
            print(f"[red] Batch LLM error: {e}[/red]")
            for pathway, ontology in batch:
                fallback_result = classify_single_pathway_llm(pathway, ontology)
                if fallback_result[0] and fallback_result[1]:
                    results[pathway] = fallback_result
                else:
                    results[pathway] = ("Cellular Processes", "Unknown Process", ontology)
                    insert_into_cache(pathway, "Cellular Processes", "Unknown Process", ontology, 0.5)

    return results


def classify_single_pathway_llm(pathway_name: str, ontology: str) -> Tuple[str, str, str]:
    """Fallback single pathway LLM classification with better error handling."""
    try:
        prompt = f"""
You are a biomedical ontology expert. Classify the following pathway from the {ontology} database into a Main_Class and Sub_Class.

Use one of the following fixed categories for Main_Class:
- Metabolism
- Signal Transduction
- Immune System
- Cell Cycle
- Cancer
- Apoptosis
- Developmental Biology
- Genetic Information Processing
- Environmental Information Processing
- Cellular Processes
- Disease Mechanisms

IMPORTANT:
- Provide BOTH Main_Class and Sub_Class (never empty).
- DO NOT use the exact pathway name as Sub_Class.

Return format:
Main_Class: ...
Sub_Class: ...

Pathway: {pathway_name}
Source: {ontology}
"""

        client = openai.OpenAI()
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )

        text = res.choices[0].message.content
        main_class = ""
        sub_class = ""

        for line in text.split("\n"):
            line = line.strip()
            if "Main_Class" in line or "main_class" in line.lower():
                if ":" in line:
                    main_class = line.split(":", 1)[1].strip()
            elif "Sub_Class" in line or "sub_class" in line.lower():
                if ":" in line:
                    sub_class = line.split(":", 1)[1].strip()

        main_class = re.sub(r"^\d+\.?\s*", "", main_class).strip()
        sub_class = re.sub(r"^\d+\.?\s*", "", sub_class).strip()

        if not main_class:
            main_class = "Cellular Processes"
        if not sub_class:
            sub_class = "Unknown Process"

        insert_into_cache(pathway_name, main_class, sub_class, ontology, 0.8)
        time.sleep(1)

        return main_class, sub_class, ontology

    except Exception as e:
        print(f"[red] Single LLM error for {pathway_name}: {e}[/red]")
        main_class = "Cellular Processes"
        sub_class = "Unknown Process"
        insert_into_cache(pathway_name, main_class, sub_class, ontology, 0.5)
        return main_class, sub_class, ontology


def process_pathways_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing function with optimized LLM usage.
    Supports BOTH:
      - legacy schema: Pathway + DB_ID
      - term schema: term column in format 'SOURCE:Pathway Name ...'
    """
    schema = detect_schema(df)

    pathway_col: Optional[str] = None
    dbid_col: Optional[str] = None
    term_col: Optional[str] = None

    if schema == "legacy":
        pathway_col = _resolve_col(df, "pathway")
        dbid_col = _resolve_col(df, "db_id")
    else:
        term_col = _resolve_col(df, "term")

    row_items: Dict[int, List[Dict[str, str]]] = {}
    all_pathways: set = set()
    pathway_to_row_mapping: DefaultDict[str, List[int]] = defaultdict(list)
    pathway_to_sources: DefaultDict[str, set] = defaultdict(set)

    for idx, row in df.iterrows():
        items: List[Dict[str, str]] = []

        if schema == "legacy":
            pathway_block = str(row[pathway_col]).splitlines()
            db_id_raw = str(row[dbid_col])
            db_sources = [x.strip().upper() for x in db_id_raw.split(",") if x.strip()]

            source_token = ", ".join(db_sources) if db_sources else "UNKNOWN"

            ontology = "GO"
            for src in db_sources:
                if src in ["REACTOME", "GO_BP", "GO_MF", "GO_CC", "WIKIPATHWAY", "KEGG"]:
                    ontology = src
                    break

            for p in pathway_block:
                p = p.strip()
                if not p:
                    continue
                items.append({"pathway": p, "source_token": source_token, "ontology": ontology})

        else:
            term_block = str(row[term_col]).splitlines()
            for t in term_block:
                src_token, p_name = parse_term_to_source_and_pathway(t)
                if not p_name:
                    continue
                ontology = normalize_ontology(src_token)
                items.append({"pathway": p_name, "source_token": src_token, "ontology": ontology})

        row_items[idx] = items

        for it in items:
            p = it["pathway"]
            all_pathways.add(p)
            pathway_to_row_mapping[p].append(idx)
            pathway_to_sources[p].add(it["source_token"])

    print(f"[bold blue] Found {len(all_pathways)} unique pathways to classify[/bold blue]")

    cached_results: Dict[str, Tuple[str, str, str]] = {}
    uncached_pathways: List[str] = []

    for pathway in all_pathways:
        cached_result = get_cached_pathway(pathway)
        if cached_result:
            cached_results[pathway] = cached_result
        else:
            uncached_pathways.append(pathway)

    print(f"[green] Found {len(cached_results)} pathways in cache[/green]")
    print(f"[yellow] Need to classify {len(uncached_pathways)} new pathways[/yellow]")

    kegg_candidates: List[str] = []
    llm_candidates: List[str] = []

    for pathway in uncached_pathways:
        is_kegg_pathway = False
        if schema == "term":
            for src_token in pathway_to_sources[pathway]:
                if "KEGG" in (src_token or "").upper() or "KEGG" in pathway.upper():
                    is_kegg_pathway = True
                    break
        else:
            # legacy heuristic based on DB_ID list
            for idx in pathway_to_row_mapping[pathway]:
                db_sources = str(df.iloc[idx][dbid_col]).split(",")
                for src in db_sources:
                    src = src.strip().upper()
                    if "KEGG" in src or "KEGG" in pathway.upper():
                        is_kegg_pathway = True
                        break
                if is_kegg_pathway:
                    break

        if is_kegg_pathway:
            kegg_candidates.append(pathway)
        else:
            llm_candidates.append(pathway)

    print(f"[blue] KEGG pathways to classify: {len(kegg_candidates)}[/blue]")
    print(f"[magenta] Non-KEGG pathways to classify: {len(llm_candidates)}[/magenta]")

    kegg_failed: List[str] = []
    for pathway in kegg_candidates:
        print(f"[blue] Trying KEGG classification for: {pathway}[/blue]")
        kegg_result = get_kegg_classification(pathway)
        if kegg_result and kegg_result[0] and kegg_result[1]:
            cached_results[pathway] = kegg_result
            print(f"[green] KEGG classified: {pathway} -> {kegg_result[0]} | {kegg_result[1]}[/green]")
        else:
            print(f"[yellow] KEGG classification failed for: {pathway}, will use LLM[/yellow]")
            kegg_failed.append(pathway)

    llm_candidates.extend(kegg_failed)
    print(f"[magenta] Total pathways for LLM classification: {len(llm_candidates)}[/magenta]")

    if llm_candidates:
        ontology_groups: DefaultDict[str, List[str]] = defaultdict(list)

        for pathway in llm_candidates:
            ontology = "GO"
            for idx in pathway_to_row_mapping[pathway]:
                for it in row_items.get(idx, []):
                    if it["pathway"] == pathway:
                        ontology = it.get("ontology") or "GO"
                        break
                if ontology != "GO":
                    break
            ontology_groups[ontology].append(pathway)

        for ontology, pathways in ontology_groups.items():
            print(f"[magenta] Processing {len(pathways)} pathways from {ontology} with LLM[/magenta]")
            batch_data = [(p, ontology) for p in pathways]
            llm_results = batch_classify_pathways_llm(batch_data, batch_size=8)

            for pathway, (main_class, sub_class, source) in llm_results.items():
                if main_class and sub_class:
                    cached_results[pathway] = (main_class, sub_class, source)
                    print(f"[green] LLM classified: {pathway} -> {main_class} | {sub_class}[/green]")
                else:
                    print(f"[red] LLM classification incomplete for: {pathway}[/red]")
                    cached_results[pathway] = ("Cellular Processes", "Unknown Process", source or "LLM")

    unclassified_count = 0
    for pathway in all_pathways:
        if pathway not in cached_results:
            print(f"[red] WARNING: Pathway not classified: {pathway}[/red]")
            cached_results[pathway] = ("Cellular Processes", "Unknown Process", "FALLBACK")
            unclassified_count += 1

    if unclassified_count > 0:
        print(f"[yellow] Applied fallback classification to {unclassified_count} pathways[/yellow]")

    new_sources: List[str] = []
    new_main_classes: List[str] = []
    new_sub_classes: List[str] = []

    for idx, _row in tqdm(df.iterrows(), total=len(df), desc="Applying classifications"):
        items = row_items.get(idx, [])
        block_sources: List[str] = []
        block_main: List[str] = []
        block_sub: List[str] = []

        for it in items:
            pathway = it["pathway"]
            src_token = it.get("source_token") or "UNKNOWN"

            if pathway in cached_results:
                main_class, sub_class, _classifier_source = cached_results[pathway]
            else:
                main_class, sub_class, _classifier_source = "Cellular Processes", "Unknown Process", "FALLBACK"

            block_sources.append(src_token)
            block_main.append(main_class or "Cellular Processes")
            block_sub.append(sub_class or "Unknown Process")

        new_sources.append("\n".join(block_sources))
        new_main_classes.append("\n".join(block_main))
        new_sub_classes.append("\n".join(block_sub))

    df["Ontology_Source"] = new_sources
    df["Main_Class"] = new_main_classes
    df["Sub_Class"] = new_sub_classes

    total_classified = len([p for p in all_pathways if p in cached_results])
    print(f"[bold green] Classification complete: {total_classified}/{len(all_pathways)} pathways classified[/bold green]")

    return df


def categorize_pathways(input_csv: str) -> Path:
    """
    Single-file mode: classify all pathways in a single CSV/TSV, in-place.

    - Detect schema
    - Run KEGG/LLM classification
    - Overwrite the same file with extra columns:
        Ontology_Source, Main_Class, Sub_Class
    """
    logger.info(f"Categorizing pathways from {input_csv}")
    input_path = Path(input_csv)

    if input_path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(input_path, sep="\t", engine="python")
    else:
        df = pd.read_csv(input_path, engine="python")

    logger.info(f"Loaded {len(df)} rows from {input_csv}")

    _ = detect_schema(df)

    df = process_pathways_optimized(df)
    logger.info(f"Processed {len(df)} pathways")

    # Preserve delimiter style on write
    if input_path.suffix.lower() in [".tsv", ".tab"]:
        df.to_csv(input_path, sep="\t", index=False)
    else:
        df.to_csv(input_path, index=False)

    logger.info(f"Categorized pathways saved to {input_path}")
    logger.info("Classification complete!")
    logger.info(f"Cached memory saved to {CATEGORIZER_CSV_PATH}")

    try:
        if Path(CATEGORIZER_CSV_PATH).exists():
            cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
            logger.info(f"Total pathways in cache: {len(cache_df)}")
        else:
            logger.info("No cache file found")
    except Exception as e:
        logger.error(f"Error reading cache statistics: {e}")

    return input_path


# =====================================================================
# NEW: mdp-style ROOT DIR SUPPORT (per-disease, filtered classified files)
# =====================================================================

def list_disease_folders(
    root_dir: Path,
    exclude: Tuple[str, ...] = ("baseline_consensus", "comparison", "results", "agentic_analysis"),
) -> List[Path]:
    """
    List disease subfolders under an mdp_pipeline_3 OUT_ROOT.

    A disease folder is any subdirectory not in `exclude`.
    """
    folders: List[Path] = []
    for entry in root_dir.iterdir():
        if entry.is_dir() and entry.name not in exclude:
            folders.append(entry)
    return sorted(folders, key=lambda p: p.name)


def _filter_enrichment_for_classification(
    df: pd.DataFrame,
    kind: str,
    q_thresh_core: float,
    q_thresh_gsea: float,
    max_rows: int,
) -> pd.DataFrame:
    """
    Apply mild significance filtering for enrichment tables.

    kind:
      - "core" for core_enrich_up/down
      - "gsea" for gsea_prerank

    Returns a possibly smaller df suitable for classification.
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    if kind == "gsea":
        # Expect: term, NES, FDR q-val
        if "fdr q-val" not in cols or "term" not in cols:
            logger.warning("gsea_prerank missing 'FDR q-val' or 'term' – skipping filter.")
            return df.head(max_rows)

        qcol = cols["fdr q-val"]
        termcol = cols["term"]

        df = df.dropna(subset=[qcol, termcol])
        df_f = df[df[qcol] <= q_thresh_gsea].copy()
        if df_f.empty:
            # fallback: best by smallest q
            df_f = df.nsmallest(max_rows, qcol)
        else:
            if len(df_f) > max_rows:
                # prioritize by q then |NES| if NES present
                if "nes" in cols:
                    nescol = cols["nes"]
                    df_f["__abs_nes__"] = df_f[nescol].abs()
                    df_f = df_f.sort_values([qcol, "__abs_nes__"], ascending=[True, False])
                    df_f = df_f.head(max_rows)
                    df_f = df_f.drop(columns=["__abs_nes__"])
                else:
                    df_f = df_f.nsmallest(max_rows, qcol)
        return df_f

    else:
        # core_enrich_up/down
        # Expect: term + qval (or pval fallback)
        if "term" not in cols:
            logger.warning("core_enrich file missing 'term' – skipping filter.")
            return df.head(max_rows)

        termcol = cols["term"]
        if "qval" in cols:
            qcol = cols["qval"]
        elif "pval" in cols:
            qcol = cols["pval"]
        else:
            logger.warning("core_enrich file missing 'qval'/'pval' – skipping filter.")
            return df.head(max_rows)

        df = df.dropna(subset=[qcol, termcol])
        df_f = df[df[qcol] <= q_thresh_core].copy()
        if df_f.empty:
            df_f = df.nsmallest(max_rows, qcol)
        else:
            if len(df_f) > max_rows:
                df_f = df_f.nsmallest(max_rows, qcol)
        return df_f


def _classify_enrichment_file_with_filter(
    path: Path,
    kind: str,
    q_thresh_core: float,
    q_thresh_gsea: float,
    max_rows: int,
) -> Optional[Path]:
    """
    Read one enrichment file, apply mild filtering, classify, and
    write a new *_classified file next to it.

    Returns the output path, or None if skipped.
    """
    if not path.exists():
        return None

    logger.info(f"Classifying {kind} enrichment file: {path}")

    if path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t", engine="python")
    else:
        df = pd.read_csv(path, engine="python")

    if df.empty:
        logger.warning(f"{path} is empty – skipping.")
        return None

    try:
        df_f = _filter_enrichment_for_classification(
            df,
            kind=kind,
            q_thresh_core=q_thresh_core,
            q_thresh_gsea=q_thresh_gsea,
            max_rows=max_rows,
        )
    except Exception as e:
        logger.error(f"Error filtering {path}: {e}")
        return None

    if df_f.empty:
        logger.warning(f"No rows left after filtering for {path} – skipping.")
        return None

    # Ensure we have at least a term/Pathway schema
    try:
        _ = detect_schema(df_f)
    except Exception as e:
        logger.warning(f"Schema detection failed for {path}: {e} – skipping.")
        return None

    df_class = process_pathways_optimized(df_f)

    out_path = path.with_name(f"{path.stem}_classified{path.suffix}")
    if out_path.suffix.lower() in [".tsv", ".tab"]:
        df_class.to_csv(out_path, sep="\t", index=False)
    else:
        df_class.to_csv(out_path, index=False)

    logger.info(f"  wrote classified file: {out_path}")
    return out_path


def categorize_disease_enrichment_files(
    root_dir: Path,
    q_thresh_core: float = 0.2,
    q_thresh_gsea: float = 0.3,
    max_rows: int = 300,
) -> List[Path]:
    """
    ROOT-DIR MODE:

    For each disease folder under `root_dir`, look for:
      - gsea_prerank.tsv
      - core_enrich_up.csv
      - core_enrich_down.csv

    For each file that exists:
      - apply mild significance filter (q thresholds, max_rows)
      - classify with KEGG+LLM
      - write *_classified files next to originals

    Returns a list of all output classified file paths.
    """
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    written: List[Path] = []

    disease_dirs = list_disease_folders(root_dir)
    if not disease_dirs:
        logger.warning(f"No disease folders found under {root_dir}")
        return written

    logger.info(f"Found {len(disease_dirs)} disease folder(s) under {root_dir}")

    for ddir in disease_dirs:
        logger.info(f"[disease] {ddir.name}")
        gsea_path = ddir / "gsea_prerank.tsv"
        core_up_path = ddir / "core_enrich_up.csv"
        core_down_path = ddir / "core_enrich_down.csv"

        out_gsea = _classify_enrichment_file_with_filter(
            gsea_path,
            kind="gsea",
            q_thresh_core=q_thresh_core,
            q_thresh_gsea=q_thresh_gsea,
            max_rows=max_rows,
        )
        if out_gsea is not None:
            written.append(out_gsea)

        out_core_up = _classify_enrichment_file_with_filter(
            core_up_path,
            kind="core",
            q_thresh_core=q_thresh_core,
            q_thresh_gsea=q_thresh_gsea,
            max_rows=max_rows,
        )
        if out_core_up is not None:
            written.append(out_core_up)

        out_core_down = _classify_enrichment_file_with_filter(
            core_down_path,
            kind="core",
            q_thresh_core=q_thresh_core,
            q_thresh_gsea=q_thresh_gsea,
            max_rows=max_rows,
        )
        if out_core_down is not None:
            written.append(out_core_down)

    return written


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Pathway classifier:\n"
            "  • Single file mode: classify one CSV/TSV in-place (--input)\n"
            "  • Root mode: classify filtered gsea_prerank/core_enrich files\n"
            "    under an mdp_pipeline_3 OUT_ROOT (--root-dir)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root-dir",
        help=(
            "Root directory containing disease subfolders "
            "(each with gsea_prerank.tsv and/or core_enrich_up/down.csv)."
        ),
    )
    parser.add_argument(
        "--input",
        help="Single enrichment file (CSV/TSV) to classify in-place.",
    )
    parser.add_argument(
        "--q-core",
        type=float,
        default=0.2,
        help="Mild q-value threshold for core_enrich_* files.",
    )
    parser.add_argument(
        "--q-gsea",
        type=float,
        default=0.3,
        help="Mild FDR threshold for gsea_prerank.tsv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300,
        help="Maximum number of rows per file to send to KEGG/LLM.",
    )

    args = parser.parse_args()

    # Safety checks
    if args.root_dir and args.input:
        raise SystemExit("Use either --root-dir OR --input, not both.")
    if not args.root_dir and not args.input:
        raise SystemExit("You must provide either --root-dir or --input.")

    if args.root_dir:
        root = Path(args.root_dir).resolve()
        logger.info(f"Running ROOT mode on: {root}")
        written = categorize_disease_enrichment_files(
            root_dir=root,
            q_thresh_core=args.q_core,
            q_thresh_gsea=args.q_gsea,
            max_rows=args.max_rows,
        )
        logger.info(f"Pathway classification done for {len(written)} file(s).")
        for p in written:
            logger.info(f"  wrote: {p}")
    else:
        inp = Path(args.input).resolve()
        logger.info(f"Running single-file mode on: {inp}")
        out_path = categorize_pathways(str(inp))
        logger.info(f"Single-file classification finished: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
