# tools.py
import json,math,aiohttp,asyncio,os,requests
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from agents import function_tool
from openai import AsyncOpenAI
import json, math, aiohttp, asyncio, os, requests, logging
from pathlib import Path
import logging, os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from openai import OpenAI

# Project Imports
from   .constants import Defaults
from   .utils import (
    filter_quantification_files,process_single_id,save_metadata_list,
    normalize,contains_high_throughput,
    sanitize_folder_name,save_invalid_ids, 
    tissue_based_filter, load_invalid_ids,
    _norm_experiment_filter)


import re
import aiohttp

FTP_SOURCES = [
    "https://ftp.ebi.ac.uk/biostudies/fire/",
    "https://ftp.ebi.ac.uk/pub/databases/biostudies/",
]

ACCESSION_PATTERN = re.compile(r"E-[A-Z]+-\d+")

async def search_biostudies_ftp(session) -> set[str]:
    """
    Fallback search: scan BioStudies FTP directories
    and extract experiment accessions.
    """
    accession_ids = set()

    for base_url in FTP_SOURCES:
        try:
            async with session.get(base_url) as resp:
                if resp.status != 200:
                    logger.warning(f"FTP source unavailable: {base_url}")
                    continue

                text = await resp.text()
                matches = ACCESSION_PATTERN.findall(text)
                accession_ids.update(matches)

        except Exception as e:
            logger.error(f"FTP search failed for {base_url}: {e}")

    return accession_ids


def get_logger(name: str = "AE-Agent"):
    import os
    from pathlib import Path
    import logging

    logger = logging.getLogger(name)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not logger.handlers:
        
        # Stable absolute log path
        logs_dir = Path("./agentic_ai_wf/shared/cohort_data/logs/arrayexpress")
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / "arrayexpress_agent.log"


        # Console handler
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        stream_handler.setFormatter(stream_formatter)

        # File handler
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setFormatter(stream_formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Prevent double-logging via root logger
        logger.propagate = False

        # Emit one line so you can see where the file is
        logger.info(f"File logging enabled → {log_path}")

    return logger

logger = get_logger()
load_dotenv(override=True)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@function_tool
async def arrayexpress_query(user_query: str) -> List[str]:
    """
    Generates an ArrayExpress search query using an LLM (few-shot),
    then searches ArrayExpress and returns a list of experiment IDs.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # -------------------------------
    # STEP 1: LLM → Search Query
    # -------------------------------
    print("Inside query generatio")
    print(user_query)

    system_prompt = """
You are a bioinformatics query parser.

Your task:
- Extract ONLY biomedical search intent from the user query
- Ignore task verbs such as: search, retrieve, rank, return
- Ignore platform names such as: GEO, ArrayExpress, ENA

Rules:
- Always include: collection:arrayexpress
- Always include: organism:"homo sapiens"
- Use quoted phrases
- Use OR for synonymous concepts
- Return ONLY the final search query
- Do NOT explain anything
"""

    few_shot_examples = """
User Query:
Search GEO using the keywords "single cell breast cancer RNA-seq"

Search Query:
collection:arrayexpress AND organism:"homo sapiens" AND
("breast cancer" OR "breast carcinoma") AND
("single-cell rna-seq" OR "scRNA-seq")

User Query:
Retrieve RNA sequencing datasets for Alzheimer disease in human brain.

Search Query:
collection:arrayexpress AND organism:"homo sapiens" AND
("alzheimer's disease" OR "alzheimer disease") AND
("rna-seq" OR "rna sequencing") AND
("brain" OR "cortex")
"""

    user_content = f"""
{few_shot_examples}

User Query:
{user_query}

Search Query:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    search_query = response.choices[0].message.content.strip()

    # ---------------------------------
    # STEP 2: Query ArrayExpress
    # ---------------------------------
    print("Before querying array express")
    base_url = "https://www.ebi.ac.uk/biostudies/api/v1/search"
    size = 100
    accession_ids = set()

    async def fetch_page(session, page):
        try:
            file_types = ["csv", "tsv", "tpm", "fpkm"]
            params = [
                ("query", search_query),
                ("page", page),
                ("size", size),
            ]
            params += [("facet.file_type", ft) for ft in file_types]

            async with session.get(base_url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Page {page} failed: {resp.status}")
                    return [], 0

                data = await resp.json()
                hits = data.get("hits", [])
                accessions = [
                    hit["accession"] for hit in hits if hit.get("accession")
                ]
                return accessions, data.get("totalHits", 0)

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            return [], 0

    async with aiohttp.ClientSession() as session:
        first_page, total_hits = await fetch_page(session, 1)
        accession_ids.update(first_page)

        total_pages = math.ceil(total_hits / size)
        tasks = [fetch_page(session, p) for p in range(2, total_pages + 1)]
        results = await asyncio.gather(*tasks)

        for accs, _ in results:
            accession_ids.update(accs)

    logger.info(
        f"ArrayExpress query complete | "
        f"Query='{search_query}' | "
        f"Experiments={len(accession_ids)}"
    )

    return list(accession_ids)

# @function_tool
# async def arrayexpress_query(user_query: str) -> List[str]:
#     """
#     Generates an ArrayExpress search query using an LLM (few-shot),
#     then searches BioStudies API.
#     Falls back to FTP path resolution if API returns 0 experiments.
#     """

#     import os
#     import math
#     import asyncio
#     import aiohttp
#     import logging
#     from typing import List
#     from openai import OpenAI

#     logger = logging.getLogger(__name__)

#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     # ------------------------------------------------------------------
#     # STEP 1: LLM → ArrayExpress Search Query
#     # ------------------------------------------------------------------

#     system_prompt = """
# You are a bioinformatics query parser.

# Your task:
# - Extract ONLY biomedical search intent from the user query
# - Ignore task verbs such as: search, retrieve, rank, return
# - Ignore platform names such as: GEO, ArrayExpress, ENA

# Rules:
# - Always include: collection:arrayexpress
# - Always include: organism:"homo sapiens"
# - Use quoted phrases
# - Use OR for synonymous concepts
# - Return ONLY the final search query
# - Do NOT explain anything
# """

#     few_shot_examples = """
# User Query:
# Search GEO using the keywords "single cell breast cancer RNA-seq"

# Search Query:
# collection:arrayexpress AND organism:"homo sapiens" AND
# ("breast cancer" OR "breast carcinoma") AND
# ("single-cell rna-seq" OR "scRNA-seq")
# """

#     user_content = f"""
# {few_shot_examples}

# User Query:
# {user_query}

# Search Query:
# """

#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0,
#     )

#     search_query = response.choices[0].message.content.strip()

#     # ------------------------------------------------------------------
#     # STEP 2: Primary Search → BioStudies API
#     # ------------------------------------------------------------------

#     BASE_API_URL = "https://www.ebi.ac.uk/biostudies/api/v1/search"
#     PAGE_SIZE = 100
#     accession_ids: set[str] = set()

#     async def fetch_page(session, page: int):
#         params = [
#             ("query", search_query),
#             ("page", page),
#             ("size", PAGE_SIZE),
#         ]

#         async with session.get(BASE_API_URL, params=params) as resp:
#             if resp.status != 200:
#                 return [], 0

#             data = await resp.json()
#             hits = data.get("hits", [])
#             accs = [h["accession"] for h in hits if h.get("accession")]
#             return accs, data.get("totalHits", 0)

#     async with aiohttp.ClientSession() as session:
#         first_page, total_hits = await fetch_page(session, 1)
#         accession_ids.update(first_page)

#         if total_hits > PAGE_SIZE:
#             total_pages = math.ceil(total_hits / PAGE_SIZE)
#             tasks = [fetch_page(session, p) for p in range(2, total_pages + 1)]
#             results = await asyncio.gather(*tasks)
#             for accs, _ in results:
#                 accession_ids.update(accs)

#         # ------------------------------------------------------------------
#         # STEP 3: FTP Fallback (ONLY if API returns 0)
#         # ------------------------------------------------------------------

#         if not accession_ids:
#             logger.warning("No API results found. Falling back to FTP resolution...")

#             def build_ftp_paths(accession: str) -> list[str]:
#                 """
#                 Build all known BioStudies FTP paths for an accession.
#                 """
#                 prefix, num = accession.rsplit("-", 1)
#                 shard = num[-3:]  # last 3 digits

#                 return [
#                     # FIRE layout (sometimes empty)
#                     f"https://ftp.ebi.ac.uk/biostudies/fire/{accession}/",

#                     # Canonical sharded BioStudies layout
#                     f"https://ftp.ebi.ac.uk/pub/databases/biostudies/{prefix}-/{shard}/{accession}/",
#                 ]

#             async def ftp_exists(url: str) -> bool:
#                 try:
#                     async with session.head(url, allow_redirects=True) as resp:
#                         return resp.status == 200
#                 except Exception:
#                     return False

#             async def validate_accession(accession: str) -> bool:
#                 for path in build_ftp_paths(accession):
#                     if await ftp_exists(path):
#                         logger.info(f"FTP found: {accession} → {path}")
#                         return True
#                 return False

#             # 🔍 Last-resort: extract candidate E-MTAB accessions from shard index
#             candidate_accessions = set()

#             shard_index_url = "https://ftp.ebi.ac.uk/pub/databases/biostudies/E-MTAB-/"
#             async with session.get(shard_index_url) as resp:
#                 if resp.status == 200:
#                     text = await resp.text()
#                     candidate_accessions.update(
#                         line.strip("/")
#                         for line in text.split()
#                         if line.startswith("E-MTAB-")
#                     )

#             validated = set()
#             for acc in candidate_accessions:
#                 if await validate_accession(acc):
#                     validated.add(acc)

#             accession_ids.update(validated)

#     logger.info(
#         f"ArrayExpress search complete | "
#         f"Query='{search_query}' | "
#         f"Experiments={len(accession_ids)}"
#     )

#     return sorted(accession_ids)


@function_tool  
def parse_biostudies_json(accession_ids: List[str], disease:str, tissue_filter : str, experiment_filter : str, output_dir : str) -> str:
    """
    For each accession, call your helper `process_single_id` (threaded),
    accumulate parsed metadata, and save with `save_metadata_list` at the given output_dir.
    Returns the path written by `save_metadata_list`.
    """
    print(output_dir)
    print("Inside Parse_biostudies")
    if not accession_ids:
        raise ValueError("No accession IDs provided to parse_biostudies_json().")

    logger.info("Fetching metadata for %d accessions...", len(accession_ids))
    metadata_list = []
    errors = 0

    # Use a bounded pool to avoid hammering remote APIs
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_id, acc): acc for acc in accession_ids}
        for fut in as_completed(futures):
            acc = futures[fut]
            try:
                result = fut.result()
                if result:
                    metadata_list.append(result)
                else:
                    logger.warning("Empty metadata for %s", acc)
            except Exception as e:
                errors += 1
                logger.error("Error processing %s: %s", acc, e)

    if not metadata_list:
        # Surface a clear error so the agent doesn’t proceed with empty files
        raise RuntimeError(
            "Failed to fetch metadata for all accessions (0 results). "
            "Check network connectivity or update process_single_id()."
        )
    print("Before Metadata list")
    # print(metadata_list)
    print(disease)
    path = save_metadata_list(metadata_list, disease, tissue_filter, experiment_filter, output_dir)
    print("After saving")
    if not path or not os.path.exists(path):
        raise RuntimeError("save_metadata_list() did not return a valid file path.")
    
    logger.info("Metadata saved to %s (%d records, %d errors)", path, len(metadata_list), errors)
    return path

@function_tool
async def filter_experiments_with_llm(json_file_path: str, disease: str, tissue_filter :str, experiment_filter : str, query:str, output_dir :str) -> List[str]:
    """"
    Filters RNA-seq experiment metadata for a given disease using rule-based and LLM-assisted criteria.

    The function:
      - Loads experiment metadata from JSON.
      - Excludes previously invalid experiments.
      - Ensures valid files, Homo sapiens organism, high-throughput sequencing, and RNA assays.
      - Applies an LLM-based tissue filter.
      - Records rejection reasons and saves valid/invalid results to disk.

    Args:
        json_file_path (str): Path to the experiment metadata file.
        disease (str): Disease name (used for logging and output structure).
        tissue_filter (str): Tissue filter possible values can be blood, tissue or any.
        experiment_filter (str): Experiment filter possible values include single cell, cell line, rna and others
        output_dir(str): Output directory to store filter experiment results

    Returns:
        List[str]: Accessions that pass all filters.
    """
    # invalid_ids = load_invalid_ids(disease,tissue_filter,experiment_filter)
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    except FileNotFoundError:
        return []
    
    matching_ids = []
    matching_records = []
    rejection_reasons = []

    for record in metadata:
        accession = record.get("accession")
        # if accession in invalid_ids:
        #     # print(f"Skipping {accession} because it is in the invalid list.")
        #     # logger.info(f"Skipping {accession} because it is in the invalid list.")
            
        #     continue

        file_entries = record.get("files", "")
        file_valid = filter_quantification_files(file_entries)

        # If valid files are not present, reject the experiment
        # if file_valid == []:
        #     rejection_reason = "Valid files not present"
        #     rejection_reasons.append({
        #         "experiment_id": accession,
        #         "rejection_reason": rejection_reason
        #     })
        #     continue  # Skip this experiment and move to the next

        organism = record.get("organism", "").lower()
        study_type = record.get("study_type", "")

        assay = normalize(record.get("assay_by_molecule", ""))
        title = record.get("title", "")
        description = record.get("description", "")


        # --- Inclusion Filters ---
        # rejection_reason = None  # Initialize rejection_reason for each record

        if "homo sapiens" not in organism:
            rejection_reason = "Organism is not Homo sapiens"
            print("Not right Homo Sapiens")
            logger.warning("Not right Homo Sapiens")

        
        if not contains_high_throughput(study_type):
            rejection_reason = "Study type is not high throughput"
            print("Not Right study type")
            logger.warning("Not Right study type")
 
        # if rejection_reason:
        #     rejection_reasons.append({
        #         "experiment_id": accession,
        #         "rejection_reason": rejection_reason
        #     })
        #     continue  
        
        # Run the filter experiment logic
        filter_result = tissue_based_filter(record, tissue_filter, experiment_filter, query)
        # rejection_reason = "LLM Rejected, experiment doesn't follow the given conditions "
        if not filter_result["validity"]:
            # rejection_reasons.append({
            #     "experiment_id": accession,
            #     "rejection_reason": filter_result["reason"]
            # })
            continue  

        matching_ids.append(accession)
        matching_records.append(record)

    # Save invalid experiments (those with rejection reasons)
    # print(rejection_reasons)
    # if rejection_reasons:
    #     save_invalid_ids(rejection_reasons, disease, tissue_filter, experiment_filter)

    # Save matching records
    safe_disease = sanitize_folder_name(disease)
    tissue_filter =  sanitize_folder_name(tissue_filter)
    experiment_filter = _norm_experiment_filter(experiment_filter)

    # Base directory for saving data
    filepath = Path(Defaults.META_FILEPATH) / Path(output_dir) / Path(safe_disease)
    filepath.mkdir(parents=True, exist_ok=True)

    output_file = f"{filepath}/{safe_disease}_filtered_experiments.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(matching_records, f_out, indent=2)
        # print(f"[SAVED] Filtered records written to {output_file}")
        logger.info(f"[SAVED] Filtered records written to {output_file}")

    except Exception as e:
        print(f"[ERROR] Could not save filtered data: {e}")
        logger.error(f"[ERROR] Could not save filtered data: {e}")
    return matching_ids
    
@function_tool
async def download_experiment_ids(disease:str, tissue_filter :str, experiment_filter : str, output_dir : str) -> List[str]:
    """
    Download (or record) the experiment present inside the filtered. Return a path (e.g., a CSV)
    containing the final list of accession IDs for reproducibility.
    """

    logger.info("Calling the download tool to download the valid experiments")
    # Save matching records
    safe_disease = sanitize_folder_name(disease)
    tissue_filter = sanitize_folder_name(tissue_filter)
    experiment_filter = _norm_experiment_filter(experiment_filter)

    # Base directory for saving data
    base_dir = Path(Defaults.META_FILEPATH) / output_dir / Path(safe_disease)
    print(base_dir)
    # Make sure the directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    json_file_path = f"{base_dir}/{safe_disease}_filtered_experiments.json"

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON file at {json_file_path}: {e}")
        return

    # Make sure the directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    if data!= []:
        for entry in data:
            accession = entry.get("accession")
            files = entry.get("files", [])
            save_path = base_dir / accession
            save_path.mkdir(parents=True, exist_ok=True)
            for file_name in files:
                url = f"https://www.ebi.ac.uk/arrayexpress/files/{accession}/{file_name}"
                file_path = os.path.join(save_path, file_name)
                try:
                    logger.info(f"Downloading: {url}")
                    with requests.get(url, stream=True, timeout=120) as response:
                        response.raise_for_status()
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    logger.info(f"Saved to: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to download {file_name}: {e}")
        return accession
    else:
        logger.warning("No experiment to download")
        return "No experiment to download"