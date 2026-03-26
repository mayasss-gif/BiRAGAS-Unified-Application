from __future__ import annotations
import re
import json
import requests
from typing import Optional, List, Dict, Union
import os
from pathlib import Path
import openai
import pandas as pd
from dataclasses import dataclass, asdict
import logging
from decouple import config
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

# Project Imports
from   .constants import BioStudyMetadata, BioStudyKeys, Defaults, LLMFilterConstants


def get_logger(name: str = "AE-Agent"):
    """
    Ensures both console + rotating file logging.
    Log file goes to <this_file_dir>/logs/ae_agent.log unless AE_LOG_DIR is set.
    """
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger(name)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Stable absolute log path
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "ae_agent.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Ensure at least one console handler
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                     for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Ensure a file handler
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_file:
        fh = RotatingFileHandler(str(log_path), mode="a", maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Announce once where the file is
        logger.info(f"File logging enabled → {log_path}")

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

logger = get_logger()

@dataclass
class AnalysisSummary:
    disease: str
    tissue_filter: str
    experiment_filter : str
    total_experiment: int
    total_experiment_filtered: int
    experiment_ids: list[str]
    filtered_experiment_ids: list[str]
    metadata_path: str
    filtered_path: str

    def to_dict(self):
        return asdict(self)

def safe_name(name: str) -> str:
    """Convert disease/filter names into safe folder names."""
    return name.lower().replace(" ", "_").replace("/", "_")

def _norm_experiment_filter(exp: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """
    Normalize experiment filter(s):
      - 'single cell' or any variant → 'sc'
      - RNA-seq variants → 'rna'
      - scATAC-seq → 'scatac'
      - spatial transcriptomics → 'st'
    Dedup + prefer 'sc' over 'rna' when both appear.
    """
    
    def _to_list(x: Optional[Union[str, List[str]]]) -> List[str]:
        return [] if x is None else (x if isinstance(x, list) else [x])

    def _dedup(items: List[str]) -> List[str]:
        return list(dict.fromkeys(items))
    # Canonical experiment labels (short form)
    _CANON_RNA     = "rna"
    _CANON_SCRNA   = "sc"
    _CANON_SCATAC  = "scatac"
    _CANON_SPATIAL = "st"

    # Patterns grouped by canonical labels
    _PATTERN_MAP = {
    _CANON_SCRNA: [
                r"\b(sn|sc)\s*-?\s*rna\s*-?\s*seq\b",
                r"\bsingle[-\s]*cell(?:ular)?\s*(rna)?\s*(seq(uencing)?)?\b",
                r"\bsingle[-\s]*cell\s*sequencing\b",
                r"\bsc\s*transcriptomics\b",
                r"\b(single[-\s]*nucleus|single[-\s]*nuclei)\s*rna\s*-?\s*seq\b",
                r"\bsingle[-\s]*cell\b",
            ],
    _CANON_SCATAC: [
                r"\bsc\s*-?\s*atac\s*-?\s*seq\b",
                r"\bsingle[-\s]*cell\s*atac\s*-?\s*seq\b",
            ],
    _CANON_RNA: [
                r"\brna\s*-?\s*seq(uencing)?\b",
                r"\btranscriptome\s*sequencing\b",
                r"\bbulk\s*rna\s*-?\s*seq\b",
            ],
    _CANON_SPATIAL: [
                r"\b(visium|spatial\s*tx|spatial\s*transcriptomics)\b",
            ],
        }

        # Compile regexes once
    _PATTERN_MAP = {k: [re.compile(p, re.I) for p in v] for k, v in _PATTERN_MAP.items()}

    def _canon_from_text(txt: str) -> Optional[str]:
        for canon, patterns in _PATTERN_MAP.items():
            if any(p.search(txt) for p in patterns):
                return canon
        return None
    items = [s.strip() for s in _to_list(exp) if (s or "").strip()]
    if not items:
        return None

    normalized = [_canon_from_text(raw) or raw for raw in items]
    normalized = _dedup(normalized)

    # Preference: if both sc and rna present → keep only sc
    if _CANON_SCRNA in normalized and _CANON_RNA in normalized:
        normalized = [x for x in normalized if x != _CANON_RNA]

    return normalized[0] if len(normalized) == 1 else normalized

def summarize_analysis(output_dir : str, disease: str, tissue_filter: str, experiment_filter:str) -> AnalysisSummary:
    """
    Summarizes the analysis by reading the automatically resolved files:
      META_FILEPATH / f"{output_dir}" / "{disease}_metadata.json"
      META_FILEPATH / f"{output_dir}" / f"{disease}_filtered_experiments.json"

    Returns experiment counts and IDs.
    """
    print("Summarize Analysi")
    safe_disease = safe_name(disease)
    safe_tfilter = safe_name(tissue_filter)
    safe_efilter = _norm_experiment_filter(experiment_filter)

    filepath = Path(Defaults.META_FILEPATH) / Path(output_dir)
    print("Output Dire In Evaluation",output_dir)
    metadata_path = filepath /safe_disease/f"{safe_disease}_metadata.json"
    filtered_path = filepath / safe_disease/ f"{safe_disease}_filtered_experiments.json"

    try:
        if not metadata_path.exists():
            logger.warning("Metadata file missing — skipping: %s", metadata_path)
            metadata = None
        else:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in metadata file — skipping: %s", metadata_path)
        metadata = None
    except Exception as e:
        logger.warning("Unexpected error reading %s: %s", metadata_path, e)
        metadata = None

    if metadata is None:
        # safely return or skip further processing
        return None
    try:
        if not filtered_path.exists():
            logger.warning("Filtered experiments file missing — skipping: %s", filtered_path)
            filtered = None
        else:
            with filtered_path.open("r", encoding="utf-8") as f:
                filtered = json.load(f)
            print(filtered)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in filtered experiments file — skipping: %s", filtered_path)
        filtered = None

    except Exception as e:
        logger.warning("Unexpected error reading filtered experiments file %s: %s", filtered_path, e)
        filtered = None

    if filtered is None:
        # gracefully stop or skip downstream logic
        return None
    
    def extract_ids(data):
        ids = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k.lower() in ["accession", "experiment_id", "id"]:
                            ids.append(str(v))
                elif isinstance(item, str):
                    ids.append(item)
        elif isinstance(data, dict):
            if "experiments" in data and isinstance(data["experiments"], list):
                for exp in data["experiments"]:
                    if isinstance(exp, dict):
                        ids.extend([str(v) for k, v in exp.items() if k.lower() in ["accession", "experiment_id", "id"]])
                    elif isinstance(exp, str):
                        ids.append(exp)
            else:
                for k, v in data.items():
                    if k.lower() in ["accession", "experiment_id", "id"]:
                        ids.append(str(v))
        return sorted(set(ids))

    experiment_ids = extract_ids(metadata)
    filtered_experiment_ids = extract_ids(filtered)

    return AnalysisSummary(
        disease=disease,
        tissue_filter=safe_tfilter,
        experiment_filter = safe_efilter,
        total_experiment=len(experiment_ids),
        total_experiment_filtered=len(filtered_experiment_ids),
        experiment_ids=experiment_ids,
        filtered_experiment_ids=filtered_experiment_ids,
        metadata_path=str(metadata_path),
        filtered_path=str(filtered_path),
    )

# def build_ftp_url(accession_id: str) -> Optional[str]:
#     match = re.match(r'^([A-Z\-]+)-(\d+)$', accession_id)
#     if not match:
#         return None
#     prefix, number = match.groups()
#     prefix += "-"
#     last_digits = number[-3:]
#     return f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}/{last_digits}/{accession_id}/{accession_id}.json"

def build_ftp_url(accession: str) -> list[tuple[str, str]]:
    """
    Returns (base_dir, json_path) tuples
    """
    prefix, num = accession.rsplit("-", 1)
    shard = num[-3:]

    return [
        # FIRE (no shard, JSON may not exist)
        (
            f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}-/{shard}/{accession}/",
            f"https://ftp.ebi.ac.uk/biostudies/fire/{prefix}-/{shard}/{accession}/{accession}.json",
        ),

        # Canonical BioStudies (SHARDED, JSON usually exists)
        (
            f"https://ftp.ebi.ac.uk/pub/databases/biostudies/{prefix}-/{shard}/{accession}/",
            f"https://ftp.ebi.ac.uk/pub/databases/biostudies/{prefix}-/{shard}/{accession}/{accession}.json",
        ),
    ]

# def fetch_json_data(url: str, accession_id: str) -> Optional[dict]:
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         print(f"[Error] Failed to fetch JSON for {accession_id}: {e}")
#         return None

def url_exists(session, url: str) -> bool:
    print("Inside url_ evcist")
    print(url)
    print("Session ",session)
    headers = {"Range": "bytes=0-0"}  # minimal request
    resp = session.get(url, headers=headers, timeout=10)
    print(f"{url} → {resp.status_code}")
    return resp.status_code in (200, 206)

    # except Exception:
    #     return False
    
# def fetch_json_data(session, accession: str) -> dict | None:
#     print("Start checking the url")
#     for base_dir, json_url in build_ftp_url(accession):
#         try:
#             print(json_url)
#             # ✅ Step 2: try JSON (optional)
#             if url_exists(session, json_url):
#                 print(json_url)
#                 print("Insidethis ")
#                 with session.get(json_url) as resp:
#                     print(resp.json())
#                     return resp.json()
#         except:
#             continue
#         # ✅ Step 3: directory exists even if JSON doesn't
#         logger.info(f"FTP directory found (no JSON): {base_dir}")
#         return {}

#     logger.warning(f"No FTP location found for {accession}")
#     return None

def fetch_json_data(session, accession: str) -> dict | None:
    print("Start checking the url")

    found_directory = False
    # print(accession)
    # print("Session",session)
    for base_dir, json_url in build_ftp_url(accession):
        # try:
        # print(f"Checking JSON: {json_url}")

            # ✅ If JSON exists → return it immediately
        # print("Before calling url exists")
        if url_exists(session, json_url):
            # print("Inside url exits")
            with session.get(json_url, timeout=15) as resp:
                        if resp.status_code in (200, 206):
                            # print("Returning Json")
                            return resp.json()

            # ✅ JSON not present → check if directory exists
            if url_exists(session, base_dir):
                found_directory = True
                logger.info(f"FTP directory found (no JSON): {base_dir}")

        # except Exception as e:
        #     logger.warning(f"Error checking {json_url}: {e}")
        #     continue

    # ✅ After checking ALL fallbacks
    if found_directory:
        return {}   # directory exists but no JSON anywhere

    logger.warning(f"No FTP location found for {accession}")
    return None


def extract_metadata(accession_id: str, data: dict) -> Optional[BioStudyMetadata]:
    try:
        top_attrs = {a['name']: a['value'] for a in data.get("attributes", [])}
        section = data.get("section", {})
        section_attrs = {a['name']: a['value'] for a in section.get("attributes", [])}

        assay = ""
        assay_accno = f"s-assays-data-{accession_id}"
        for sub in section.get("subsections", []):
            if isinstance(sub, dict) and sub.get("accno") == assay_accno:
                for a in sub.get("attributes", []):
                    if a["name"].lower() == "assay by molecule":
                        assay = a["value"]

        study_type_list = [
            attr["value"].lower()
            for attr in section.get("attributes", [])
            if attr.get("name") == "Study type"
        ]

        study_type_clean = ', '.join(study_type_list) if study_type_list else Defaults.NA

        metadata_obj = BioStudyMetadata(
            accession=accession_id,
            title=top_attrs.get(BioStudyKeys.TITLE, Defaults.NA),
            organism=section_attrs.get(BioStudyKeys.ORGANISM, Defaults.NA).lower(),
            study_type=study_type_clean,
            assay_by_molecule=assay.lower(),
            description=' '.join(section_attrs.get(BioStudyKeys.DESCRIPTION, Defaults.NA).split())
        )

        return metadata_obj
    except Exception as e:
        print(f"[Warning] Skipping metadata for {accession_id}: {e}")
        return None

def sanitize_folder_name(name: str) -> str:
    """
    Clean a string to make it safe for use as a folder name.
    - Replaces spaces and special chars with underscores.
    - Strips leading/trailing underscores.
    - Lowercases the name for consistency.
    """
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name.strip().lower()).strip('_')

def save_metadata_list(metadata_list: List[BioStudyMetadata], disease : str, tissue_filter :str, experiment_filter :str, output_dir : str) -> str:
    print("Inside save_metadatlist function")
    print(output_dir)
    safe_disease = sanitize_folder_name(disease)
    tissue_filter = sanitize_folder_name(tissue_filter)
    experiment_filter = _norm_experiment_filter(experiment_filter)
    print("Output Direcotry", output_dir)
    filepath = Path(Defaults.META_FILEPATH) / Path(output_dir) / Path(safe_disease)/f"{safe_disease}_metadata.json"
    print(filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    print("Before Saving file")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in metadata_list], f, indent=2, ensure_ascii=False)

    return filepath

def extract_file_dicts(data: list) -> List[dict]:
    """
    Recursively extract all dictionaries from nested lists where 'type' == 'file'.
    """
    file_dicts = []
    def recursive_search(item):
        if isinstance(item, dict):
            if item.get("type", "").lower() == "file":
                file_dicts.append(item)

            for value in item.values():
                recursive_search(value)

        elif isinstance(item, list):
            for elem in item:
                recursive_search(elem)

    recursive_search(data)
    return file_dicts

def filter_quantification_files(file_list: List[Dict]) -> List[Dict]:    
    """
    Filters and returns files whose path contains quantification-related keywords,
    unless any file path contains 'hela' or 'sirna', in which case an empty list is returned.
    
    Args:
        file_list (List[Dict]): List of file dictionaries (each with a 'path' key).
    
    Returns:
        List[Dict]: Filtered list of file dictionaries based on quantification keywords.
    """
    keywords = [
        "count", "counts", "raw_counts", "readcounts", "featurecounts",
        "norm_counts", "normalized_counts", "fpkm", "rpkm", "tpm",
        "cpm", "tmm", "gene_exp.diff", "diff"
    ]

    # Filter for quantification-related keywords
    valid_paths = []
    for file in file_list:
        if any(kw in file for kw in keywords):
            valid_paths.append(file)

    return valid_paths

# def tissue_based_filter(experiment_json: Dict, tissue_filter: str, experiment_filter : str) -> Dict:
#     """
#     This function uses GPT-3 to filter the experiment data based on user criteria.
#     The experiment JSON and the filter are passed to the model to determine the validity.
    
#     Args:
#     - experiment_json: JSON object representing the experiment data.
#     - tissue_filter: The tissue's filter criteria in a string format.

#     Returns:
#     - dict: Contains validity status (True/False) and reason for rejection (if False).
#     """
#     record = experiment_json
#     organism = record.get("organism", "").lower()
#     study_type = record.get("study_type", "")

#     assay = normalize(record.get("assay_by_molecule", ""))
#     title = record.get("title", "")
#     description = record.get("description", "")

#     try:
#         prompt = f"""
#         Below is an experiment metadata JSON. Please evaluate if this experiment meets the following user criteria:
#         - Sample tissue should be : {tissue_filter}
#         - Experiment Type should be related to {experiment_filter}
        
#         Provide 'True' if it meets all criteria, otherwise provide 'False' along with the reason for rejection.

#         Experiment JSON: {json.dumps(experiment_json)}
#         """
        
#         response = openai.completions.create(
#             model="gpt-4o-mini",  
#             prompt=prompt,
#             max_tokens=250,
#             temperature=0.5
#         )
        
#         gpt_response = response.choices[0].text.strip()
#         validity = "True" in gpt_response
#         reason = gpt_response.replace("True", "").strip() if not validity else "Valid"
        
#         return {"validity": validity, "reason": reason}
    
#     except Exception as e:
#         print(f"Error while calling GPT-3: {e}")
#         return {"validity": False, "reason": str(e)}

def _extract_json_block(text: str) -> Dict:
    """
    Extract and parse the first valid JSON object from an LLM response.
    Handles cases where the model wraps output in code fences.
    """
    text = text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        # After stripping, the first newline often separates the language tag
        if "\n" in text:
            text = "\n".join(text.split("\n")[1:])
    # Try direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        # Fallback: find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {}

def tissue_based_filter(experiment_json: Dict, tissue_filter: str, experiment_filter: str, query:str) -> Dict:
    """
    LLM-based filter that:
    - Uses the *raw* experiment_filter (no local mapping).
    - Enforces strict modality matching:
        * If user asks for single-cell (e.g., 'single cell', 'scRNA', '10x', 'snRNA'),
          bulk RNA-seq is INVALID, even if it's still RNA-seq.
        * If user asks for bulk RNA-seq, single-cell or single cell rna seq is INVALID 
    - Tissue check is semantic (synonyms allowed). If ambiguous, return False with reason.
    Returns: {"validity": bool, "reason": str}
    """

    print(query)
    system_msg = (
        "You are a meticulous curation assistant for genomics experiments. "
        "You must strictly follow user constraints and output ONLY valid JSON."
    )

    # IMPORTANT: We pass experiment_filter RAW (no mapping); the model must interpret it.
#     user_prompt = f"""
# You will evaluate ONE experiment's metadata JSON against user constraints.

# CONSTRAINTS (strict):
# 1) TISSUE: Must match the user's requested tissue (“{tissue_filter}”).
#    - Allow common synonyms (e.g., PBMC == peripheral blood mononuclear cells).
#    - If {experiment_filter} indicates bulk RNA-seq (total RNA, polyA, coding RNA, standard RNA-seq) then cell line, Hela cells are not allowed. 
#    - If the tissue is not clearly indicated or ambiguous, treat as NOT matching.

# 2) EXPERIMENT TYPE: Must match the user's requested experiment type (raw string below):
#    "{experiment_filter}"

#    ENFORCE these modality rules:
#    - If the request mentions terms like: single cell, single-cell, scRNA, snRNA, single nucleus, 10x, Drop-seq, Smart-seq,
#      then ONLY accept experiments that explicitly indicate a single-cell or single-nucleus protocol or barcoding.
#      Bulk RNA-seq (total/coding/polyA/3' bulk) is INVALID.
#    - If the request explicitly indicates bulk RNA-seq (total RNA, polyA, coding RNA, standard RNA-seq) and does NOT mention single-cell,
#      then single-cell rna seq datasets are INVALID. Incase of bulk RNA-Seq then single-cell RNA Seq is not allowed. Also if the datasets mention HeLa cells, cell line marked as INVALID
#    - If the request mentions a different modality (e.g., ATAC-seq, Ribo-seq, PRO-seq, PAR-CLIP),
#      only accept that specific modality; others are INVALID.
#    - If the request is vague/ambiguous, be conservative: require an explicit match signal in the metadata; otherwise INVALID.

# EVALUATION SOURCES:
# - You may use any text fields in the JSON (e.g., study_type, assay_by_molecule, title, description, files).

# OUTPUT format (STRICT JSON ONLY; no prose, no code fences):
# {{
#   "validity": true/false,
#   "reason": "Short justification (<= 25 words)"
# }}

# Now evaluate this experiment:

# EXPERIMENT_JSON:
# {json.dumps(experiment_json, ensure_ascii=False)}
# """

    user_prompt = f"""
You will evaluate ONE experiment's metadata JSON against user constraints.

CONSTRAINTS (strict):

1) TISSUE: Must match the user's requested tissue (“{tissue_filter}”).
   - Allow common synonyms (e.g., PBMC == peripheral blood mononuclear cells).
   - If a tissue is requested, experiments using immortalized cell lines
     (e.g., HeLa, HEK293, K562) are INVALID regardless of modality.
   - If tissue is mentioned as "any", accept all kind of tissues.
   - If the tissue is not clearly indicated or ambiguous, treat as NOT matching.

2) EXPERIMENT TYPE: Must match the user's requested experiment type (raw string below):
   "{experiment_filter}"
   - If the experiment type is generic (e.g., "RNA-seq") and single-cell is not explicitly requested,
     then single-cell RNA-seq datasets are INVALID.
   - Require explicit confirmation of bulk or single-cell modality in metadata.

ENFORCE these modality rules:
- Determine the requested modality from the {query}:
  * If the query mentions single-cell keywords (single cell, scRNA, snRNA, 10x, Drop-seq, Smart-seq), requested_modality = single-cell.
  * If the query mentions bulk RNA-seq (total RNA, polyA, coding RNA, standard RNA-seq), requested_modality = bulk.
  * Otherwise, requested_modality = generic RNA-seq.

- Accept an experiment ONLY if:
  1) requested_modality == "single-cell" AND metadata explicitly indicates single-cell or single cell RNA Sequencing protocol.
  2) requested_modality == "bulk" AND metadata indicates bulk RNA-seq AND does NOT indicate single-cell.
  3) requested_modality == "RNA-seq" AND metadata indicates bulk RNA-seq. Single-cell RNA-seq is INVALID.
        - Single-cell signals in study_type, description, library_strategy, protocol, or files (e.g., barcodes, features, matrix) MUST be treated as INVALID.

- If metadata indicates a different modality (ATAC-seq, Ribo-seq, PRO-seq, PAR-CLIP), accept ONLY if requested_modality matches exactly.
- If ambiguous, missing, or conflicting signals exist, mark experiment INVALID.

ENFORCEMENT DETAILS (MANDATORY):
- Modality decisions MUST be supported by explicit evidence in metadata fields
  (e.g., assay_by_molecule, library_strategy, protocol, technology, description).
- Do NOT infer modality from title alone.
- If conflicting modality signals are present, mark INVALID.
- If no explicit modality signal is present, mark INVALID.

EVALUATION SOURCES:
- You may use any text fields in the JSON (e.g., study_type, assay_by_molecule, title, description, files).

OUTPUT format (STRICT JSON ONLY; no prose, no code fences):
{{
  "validity": true/false,
  "reason": "Short justification (<= 50 words)"
}}

Now evaluate this experiment:

EXPERIMENT_JSON:
{json.dumps(experiment_json, ensure_ascii=False)}
"""

    try:
        # If you're on the legacy openai python (matching your current style):
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )

        raw = resp.choices[0].message.content.strip()
        obj = _extract_json_block(raw)

        validity = bool(obj.get("validity") is True)
        reason = obj.get("reason") or ("Valid" if validity else "Does not meet criteria.")
        print("Reason /n")
        print(reason[:300])
        print(validity)
        # Ensure return schema matches your original
        return {"validity": validity, "reason": reason[:300]}

    except Exception as e:
        print(f"Error while calling LLM: {e}")
        return {"validity": False, "reason": str(e)}

def save_invalid_ids(rejection_reasons: List[Dict], disease: str, tissue_filter : str, experiment_filter : str):
    """
    Save rejection reasons into a CSV file.
    
    Args:
    - rejection_reasons: List of dictionaries containing rejection reasons.
    - disease: The disease name for saving the CSV file.
    """
    if rejection_reasons:
        df = pd.DataFrame(rejection_reasons)
        save_path = Path(Defaults.INVALID_FILEPATH) / f"{sanitize_folder_name(disease)}_{sanitize_folder_name(tissue_filter)}_{_norm_experiment_filter(experiment_filter)}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[SAVED] Rejection reasons written to {save_path}")
    else:
        print("No rejection reasons to save.")

def load_invalid_ids(disease: str, tissue_filter: str,experiment_filter: str) -> List:
    disease_filter_file = Path(Defaults.INVALID_FILEPATH) / f"{sanitize_folder_name(disease)}_{sanitize_folder_name(tissue_filter)}_{_norm_experiment_filter(experiment_filter)}.csv"
    
    if disease_filter_file.exists():
        print(f"[INFO] Disease filter file {disease_filter_file} found. Skipping previously invalid experiments.")
        invalid_ids_df = pd.read_csv(disease_filter_file)
        invalid_ids = invalid_ids_df["experiment_id"].tolist() 
         
    else:
        invalid_ids = []
    return invalid_ids

def process_single_id(accession_id: str) -> Optional[dict]:
    ftp_url = build_ftp_url(accession_id)
    if not ftp_url:
        print(f"[Warning] Invalid accession ID format: {accession_id}")
        return None
    print(ftp_url)
    import requests

    with requests.Session() as session:
        data = fetch_json_data(session, accession_id)
    # data = fetch_json_data(ftp_url, accession_id)
    if not data:
        return None
    # print(data)
    metadata = extract_metadata(accession_id, data)
    # print("After metadata")
    # print(metadata)
    sections = data.get("section", {}).get("subsections", [])
    file_entries = extract_file_dicts(sections)
    paths = []
    for file in file_entries:
        path = file.get("path", "")
        paths.append(path)
    metadata.files = paths
    return metadata

def normalize(text: str) -> str:
    
    """Replace special dashes with spaces and lower the case."""
    return re.sub(r"[-–—_]", " ", text.lower())

def contains_high_throughput(text: str) -> bool:
    """
    Accept if at least one segment (split by commas) contains 'high throughput sequencing'
    and none of the exclusion keywords.
    """
    lowered_text = text.lower()
    segments = [seg.strip() for seg in lowered_text.split(",")]

    pattern = r"(?<![a-zA-Z0-9_])(high[\s\-–—_]*throughput[\s\-–—_]*sequencing|rna[\s\-–—_]*seq(?:\s+of\s+(?:coding|total)\s+rna)?)(?![a-zA-Z0-9_])"

    for segment in segments:
        if re.search(pattern, segment):
            return True

    return False 
