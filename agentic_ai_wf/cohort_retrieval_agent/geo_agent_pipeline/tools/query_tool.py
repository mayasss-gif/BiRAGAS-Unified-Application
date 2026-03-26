"""
Enhanced Query tool for the Cohort Retrieval Agent system.

This tool handles querying various data sources for datasets related to diseases
with LLM-based filtering, invalid GSE ID tracking, and robust treatment study detection.
"""

import aiohttp
import logging
import xmltodict
import json
import csv
import re
import os
from typing import Dict, List, Any, Optional, Iterable, Tuple, Set,Union
from urllib.parse import urlencode
from pathlib import Path
from Bio import Entrez
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import tiktoken
from openai import OpenAI
import gzip
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

# Project Imports
from   ..base.base_tool import AsyncContextTool, ToolResult
from   ..config import CohortRetrievalConfig
from   ..exceptions import QueryError, NetworkError
from   ..config import DirectoryPathsConfig
from   ..tools.queryagent import query_geo
from   ..tools.evaluator import EvaluationTool
from   ..tools.evaluatorrag import LLMEvaluator, load_datasets_from_folder

# Configure logging for query tool
def setup_query_logging():
    """Set up logging configuration for query tool."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("query_tool")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler for query.log
    file_handler = logging.FileHandler(logs_dir / "query.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize the logger
query_logger = setup_query_logging()

class QueryTool(AsyncContextTool[List[Dict[str, Any]]]):
    """
    Enhanced tool for executing queries against various APIs and databases.
    
    Features:
    - LLM-based filtering of datasets for disease vs control analysis
    - Invalid GSE ID tracking and management
    - Treatment study detection and exclusion
    - Blood/tissue type keyword filtering
    - Supports HTTP GET/POST requests with automatic retry logic and response parsing
    """
    
    # Class constants
    BLOOD_KEYWORDS = [
        "blood", "whole blood", "peripheral blood", "pbmc", "peripheral blood mononuclear",
        "buffy coat", "plasma", "serum","Peripheral T lymphocytes"
    ]
    
    TREATMENT_KEYWORDS = [
        'drug treatment', 'compound', 'inhibitor', 'agonist', 'antagonist',
        'therapeutic', 'intervention', 'clinical trial', 'pharmacological',
        'dosage', 'dose', 'administration', 'therapy', 'anti-', 'pro-',
        'vs vehicle', 'vs dmso', 'vs placebo', 'treated vs untreated',
        'drug vs', 'treatment vs', 'compound vs', 'inhibitor vs',
        'mitoxantrone', 'fulvestrant', 'tamoxifen', 'chemotherapy',
        'drug screening', 'drug discovery', 'mechanism of action',
        'treatment response', 'therapeutic response','treatment','treated'
    ]
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config)
        self.logger = query_logger
        self.logger.info("Initializing QueryTool")
        
        # Set Entrez email for NCBI queries (required by NCBI)
        if hasattr(config, 'geo_config') and config.geo_config.entrez_email:
            try:
                Entrez.email = config.geo_config.entrez_email
                self.logger.info(f"Set Entrez email: {config.geo_config.entrez_email}")
            except ImportError:
                self.logger.warning("BioPython not available - Entrez email not set")
        else:
            self.logger.warning("No Entrez email configured - NCBI queries may fail")
        
        # Initialize OpenAI client for LLM filtering [[memory:3904167]]
        try:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                self.logger.warning("OPENAI_API_KEY not found - LLM filtering will be disabled")
                self.openai_client = None
            else:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.logger.info("OpenAI client initialized for LLM filtering")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
        
        # Setup invalid GSE tracking
        self.logger.info("QueryTool initialization completed")
    
    def _normalize_token(self, text: str) -> str:
        """
        Normalize a token for filenames:
        - strip leading/trailing spaces
        - collapse internal whitespace to single space
        - remove non-alphanumeric except spaces/underscores/hyphens

        # lower | the file name
        """
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[^\w\s-]", "", text) 
        text = text.lower() # keep letters, digits, _, space, -
        return text

    def _disease_stub(self, disease: str) -> str:
        """
        Convert disease string to the requested style like 'Pancreaticcancer':
        - Title case words, concatenate without spaces
        - Then lowercase all letters except the first character
        """
        norm = self._normalize_token(disease)

        # Title-case then remove spaces to get e.g. "PancreaticCancer"
        stub = norm.title().replace(" ", "")
        if not stub:
            return "Unknown"
        
        # Make it 'Pancreaticcancer' (first upper, rest lower)
        return stub[0].lower() + stub[1:].lower()

    def _filter_stub(self, sample_filter: str) -> str:
        """
        Keep filter lowercase (e.g., 'tissue', 'blood') and safe.
        """
        norm = self._normalize_token(sample_filter).replace(" ", "_")
        return norm.lower() or "unknown"

    def _norm_experiment_filter(self, exp: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
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

 
    def _json_safe(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable types."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        # fallback: stringify unknown objects (e.g., pydantic models, custom classes)
        return str(obj)

    def _save_selected_dataset_json(
        self,
        output_dir: Optional[Path],
        dataset_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Write a dataset selection record to disk."""
        if output_dir is None:
            return
        print("save selected dataset")
        print(output_dir)
        out = Path(output_dir) / "selected_datasets"
        out.mkdir(parents=True, exist_ok=True)

        safe_payload = self._json_safe(payload)

        # one file per dataset (overwrite-safe if rerun; include timestamp if you prefer multiple versions)
        fp = out / f"{dataset_id}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(safe_payload, f, ensure_ascii=False, indent=2)

    def _build_invalid_file_path(self, base_dir: Path, disease: str, tissue_filter: str, experiment_filter:str) -> Path:
        """
        Build CSV filename like: Pancreaticcancer_tissue.csv
        """
        disease_part = self._disease_stub(disease)
        tissue_part = self._filter_stub(tissue_filter)
        experiment_part = self._norm_experiment_filter(experiment_filter)
        
        fname = f"{disease_part}_{tissue_part}_{experiment_part}.csv"
        return base_dir / fname

    def _get_invalid_store_dir(self) -> Path:
        """
        Base directory under cohort_db for invalid GSE tracking.
        """
        self.logger.debug("Ensuring invalid GSE store directory exists")
        config = DirectoryPathsConfig()
        cohort_db_dir = getattr(config, "invalid_dir", None)
        store_dir = Path(cohort_db_dir)  # flat in cohort_db as you showed
        os.makedirs(store_dir, exist_ok=True)
        return store_dir

    def _get_invalid_gse_file_path(
        self, disease: str, filter: str
    ) -> Path:
        """
        Get the path for invalid GSE IDs CSV file for a given disease + filter,
        e.g., Pancreaticcancer_tissue.csv
        """
        store_dir = self._get_invalid_store_dir()
        tissue_filter = filter['tissue_filter']
        experiment_filter = filter['experiment_filter']

        file_path = self._build_invalid_file_path(store_dir, disease, tissue_filter,experiment_filter)
        self.logger.debug(f"Invalid GSE file path: {file_path}")
        return file_path

    def _get_invalid_gse_ids(self, disease: str, filter: str) -> Set[str]:
        """
        Return a set of invalid GSE IDs (without 'GSE' prefix) for the given disease+filter.
        Accepts files with either just `gse_id` or both `gse_id,reason`.
        """
        file_path = self._get_invalid_gse_file_path(disease, filter)
        tissue_filter = filter['tissue_filter']
        experiment_filter = filter['experiment_filter']

        self.logger.debug(f"Retrieving invalid GSE IDs from {file_path}")
        if not file_path.exists():
            self.logger.debug("Invalid GSE file does not exist, returning empty set")
            return set()

        try:
            with open(file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "gse_id" not in reader.fieldnames:
                    self.logger.warning(f"'gse_id' column not found in {file_path}, returning empty set")
                    return set()
                invalid_ids = {
                    str(row.get("gse_id", "")).strip()
                    for row in reader
                    if str(row.get("gse_id", "")).strip()
                }
            self.logger.debug(f"Retrieved {len(invalid_ids)} invalid GSE IDs for {disease}/{tissue_filter}/{experiment_filter}")
            return invalid_ids
        except Exception as e:
            self.logger.warning(f"Failed to read invalid GSE file {file_path}: {e}")
            return set()

    def _append_invalid_gse_ids(
        self,
        new_entries: Iterable,  # accepts List[str] or List[Tuple[str, str]]
        disease: str,
        filter: str
        ):
            """
            Append new invalid GSE entries to the disease+filter CSV without duplicates.
            Each entry is either:
            - a string gse_id, or
            - a tuple (gse_id, reason)
            CSV schema: gse_id, reason
            Also updates a per-disease Excel workbook (sheet per sample_filter) if pandas is available.
            """
            # Normalize incoming entries to (id, reason)
            norm: List[Tuple[str, str]] = []
            for item in (new_entries or []):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    gid = str(item[0]).strip()
                    reason = str(item[1]).strip() if len(item) >= 2 else ""
                else:
                    gid = str(item).strip()
                    reason = ""
                if gid:
                    norm.append((gid, reason))

            if not norm:
                self.logger.debug("No new invalid GSE entries to append")
                return

            # Deduplicate incoming by id (keep first reason we see)
            seen = set()
            dedup_norm = []
            for gid, reason in norm:
                if gid not in seen:
                    seen.add(gid)
                    dedup_norm.append((gid, reason))

            file_path = self._get_invalid_gse_file_path(disease, filter)

            # Load existing to avoid re-adding
            existing_ids: Set[str] = set()
            if file_path.exists():
                try:
                    with open(file_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        if reader.fieldnames and "gse_id" in reader.fieldnames:
                            for row in reader:
                                gid = str(row.get("gse_id", "")).strip()
                                if gid:
                                    existing_ids.add(gid)
                        else:
                            self.logger.warning(f"'gse_id' column missing in {file_path}; treating as empty file")
                except Exception as e:
                    self.logger.warning(f"Failed to read existing invalid GSE IDs from {file_path}: {e}")

            to_add = [(gid, reason) for gid, reason in dedup_norm if gid not in existing_ids]
            if not to_add:
                self.logger.debug("No new unique invalid GSE entries to add")
                return

            # Ensure file exists with header
            try:
                new_file = not file_path.exists()
                if new_file:
                    self.logger.debug(f"Creating new invalid GSE file with header: {file_path}")
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["gse_id", "reason"])

                # If file exists but has old header, we’ll rewrite with 2 columns
                else:
                    with open(file_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    if rows and rows[0] != ["gse_id", "reason"]:
                        # migrate: add empty reasons to existing rows
                        header = rows[0]
                        if header and "gse_id" in header and "reason" not in header:
                            gid_idx = header.index("gse_id")
                            migrated = [["gse_id", "reason"]]
                            for r in rows[1:]:
                                gid_val = r[gid_idx] if gid_idx < len(r) else ""
                                migrated.append([gid_val, ""])
                            with open(file_path, "w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerows(migrated)

                with open(file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(to_add)

                self.logger.info(f"Added {len(to_add)} new invalid GSE entries to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to append invalid GSE entries to {file_path}: {e}")
                return

    def _get_geo_query(self, disease_name: str) -> str:
        """Build GEO search query for disease."""
        self.logger.debug(f"Building GEO query for disease: {disease_name}")
        if disease_name.lower() == "any":
            query = (
            f'"Homo sapiens"[porgn] AND "gse"[Entry Type] '
            f'AND "Expression profiling by high throughput sequencing"[Filter]'
        )
        else:
            # Add all the 
            query = (
                f'"{disease_name}"[All Fields] AND "Homo sapiens"[porgn] AND "gse"[Entry Type] '
                f'AND "Expression profiling by high throughput sequencing"[Filter]'
            )
        self.logger.debug(f"Generated query: {query}")
        return query
    
    def _is_likely_treatment_study(self, dataset_info: dict) -> bool:
        """
        Returns True if dataset is likely treatment study and should be excluded,
        BUT only when the query did NOT explicitly request treatment datasets.
        """
        # If user explicitly wants treatment datasets, ignore treatment filtering.
        if getattr(self, "include_treatment", False):
            self.logger.debug("Query explicitly includes treatment -> skipping treatment filtering.")
            return False

        # Otherwise: default behavior = filter treatment studies
        self.logger.debug("Query does not explicitly include treatment -> applying treatment filtering.")

        text_to_check = ""
        if 'title' in dataset_info:
            text_to_check += str(dataset_info['title']).lower() + " "
        if 'summary' in dataset_info:
            text_to_check += str(dataset_info['summary']).lower() + " "

        samples = dataset_info.get('Samples', [])
        if isinstance(samples, list):
            for sample in samples[:10]:
                if isinstance(sample, dict) and 'Title' in sample:
                    text_to_check += str(sample['Title']).lower() + " "

        treatment_score = sum(1 for kw in self.TREATMENT_KEYWORDS if kw in text_to_check)

        if treatment_score >= 2:
            return True

        strong_treatment_indicators = [
            'drug treatment', 'vs vehicle', 'vs dmso', 'vs placebo',
            'clinical trial', 'therapeutic intervention', 'drug screening',
            'mechanism of action'
        ]
        if any(ind in text_to_check for ind in strong_treatment_indicators):
            return True

        return False

    
    import tarfile
    import io
    from collections import defaultdict

    def _tar_has_complete_samples(self, tar_bytes: bytes) -> bool:
        """
        Checks whether the tar archive contains features, matrix, and barcodes
        for EVERY sample.
        """
        REQUIRED = {"features.tsv", "genes.tsv", "matrix.mtx", "barcodes.tsv"}

        sample_files = defaultdict(set)

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                name = member.name.lower()

                # Identify sample directory (top-level folder)
                parts = name.split("/")
                if len(parts) < 2:
                    continue

                sample_id = parts[0]
                filename = parts[-1]

                if filename in REQUIRED:
                    # normalize genes.tsv -> features.tsv
                    if filename == "genes.tsv":
                        filename = "features.tsv"
                    sample_files[sample_id].add(filename)

        if not sample_files:
            return False

        for sample, files in sample_files.items():
            if not {"features.tsv", "matrix.mtx", "barcodes.tsv"} <= files:
                self.logger.info(f"[TAR INVALID] Sample '{sample}' missing required files: {files}")
                return False

        return True

    # def _has_supplementary_files(self, gse_id: str) -> bool:
    #     SUPPLEMENTARY_KEYWORDS = [
    #         "count", "counts", "raw_counts", "readcounts", "featurecounts",
    #         "norm_counts", "normalized_counts", "fpkm", "rpkm", "tpm", "cpm"
    #     ]
    #     # extensions fallback (case-insensitive)
    #     PREFERRED_FORMATS = [".txt", ".tsv", ".csv", ".gz", ".tar"]

    #     def normalize_gse_id(raw: str) -> str:
    #         """
    #         Convert inputs like '200294225' -> 'GSE294225'.
    #         Also handles 'GSE089408' -> 'GSE89408', '89408' -> 'GSE89408', etc.
    #         """
    #         if raw is None:
    #             raise ValueError("Missing dataset id.")

    #         s = str(raw).strip()

    #         # Common internal form starts with '200' — treat that as 'GSE'
    #         if re.fullmatch(r'200\d+', s):
    #             digits = s[3:]  # drop the '200' prefix
    #             return f"GSE{int(digits)}"  # int() strips any leading zeros

    #         # Generic: optional 'GSE', then digits
    #         m = re.search(r'(?:GSE)?0*(\d+)$', s, flags=re.IGNORECASE)
    #         if not m:
    #             raise ValueError(f"Could not parse a GEO series id from '{s}'.")
    #         return f"GSE{int(m.group(1))}"

    #     def geo_series_block_prefix(gse_id: str) -> str:
    #         """
    #         GEO FTP series block folder.
    #         GSE294225 -> 'GSE294nnn'
    #         GSE89408  -> 'GSE89nnn'
    #         < 1000    -> 'GSEnnn'
    #         """
    #         if not re.match(r'^GSE\d+$', gse_id):
    #             raise ValueError(f"Not a valid GSE id: '{gse_id}'")
    #         n = int(gse_id[3:])
    #         return "GSEnnn" if n < 1000 else f"GSE{n // 1000}nnn"

    #     # ---------- Download series matrix ----------
    #     # ---------- Input handling ----------
    #     gse_id = normalize_gse_id(gse_id)       
    #     if not gse_id:
    #         raise ValueError("DatasetInfo object must have a non-empty dataset_id.")

    #     # (Optional) Build URLs
    #     geo_prefix = geo_series_block_prefix(gse_id)     
    #     url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{gse_id}/suppl/"        
    #     try:
    #         r = requests.get(url, timeout=20)
    #     except Exception as e:
    #          self.logger.info(f"[HTTP] Exception for {gse_id} supplementary check: {e}")
       
    #     if r.status_code != 200:
    #          self.logger.info(f"[HTTP] {gse_id} supplementary dir not accessible at {url} (HTTP {r.status_code})")
       
    #     soup = BeautifulSoup(r.text, 'html.parser')
    #     links = [a['href'] for a in soup.find_all('a', href=True)]
    #         # Clean links: drop query/anchors, directories, and nav entries
    #     clean_files = []
    #     for href in links:
    #         if href.startswith('?') or href.endswith('/'):
    #             continue
    #         root = href.split('?', 1)[0].split('#', 1)[0]
    #         if root.lower().startswith('parent'):
    #             continue
    #         clean_files.append(root)

    #         # 1) keyword-based matches
    #         keyword_pattern = re.compile(
    #             r"(?<![A-Za-z0-9])(" + "|".join(SUPPLEMENTARY_KEYWORDS) + r")(?![A-Za-z0-9])",
    #             re.IGNORECASE
    #         )
    #         matching_files = [f for f in clean_files if keyword_pattern.search(f)]
    #         if matching_files:
    #             self.logger.info(f"[MATCH] Keyword-based matches:{matching_files}")
    #             return True

    #         # 2) extension-based matches
    #         ext_match = any(f.lower().endswith(ext) for f in clean_files for ext in PREFERRED_FORMATS)
    #         if ext_match:
    #             self.logger.info(f"[MATCH] Extension-based match found among preferred formats:{PREFERRED_FORMATS}")
    #             return True

    #         self.logger.info(f"[NO MATCH] Neither keywords nor preferred extensions found in:{url}")

    #     return False
    
    def _has_supplementary_files(self, gse_id: str) -> bool:
        import re
        import io
        import tarfile
        import requests
        from bs4 import BeautifulSoup
        from collections import defaultdict

        SUPPLEMENTARY_KEYWORDS = [
            "count", "counts", "raw_counts", "readcounts", "featurecounts",
            "norm_counts", "normalized_counts", "fpkm", "rpkm", "tpm", "cpm"
        ]

        PREFERRED_FORMATS = [".txt", ".tsv", ".csv", ".gz", ".tar", ".tar.gz"]

        def normalize_gse_id(raw: str) -> str:
            if raw is None:
                raise ValueError("Missing dataset id.")
            s = str(raw).strip()
            if re.fullmatch(r"200\d+", s):
                return f"GSE{int(s[3:])}"
            m = re.search(r"(?:GSE)?0*(\d+)$", s, re.IGNORECASE)
            if not m:
                raise ValueError(f"Could not parse GEO id from '{s}'.")
            return f"GSE{int(m.group(1))}"

        def geo_series_block_prefix(gse_id: str) -> str:
            n = int(gse_id[3:])
            return "GSEnnn" if n < 1000 else f"GSE{n // 1000}nnn"

        def tar_has_complete_samples(tar_bytes: bytes) -> bool:
            import io
            import tarfile
            from collections import defaultdict
            REQUIRED = {"features.tsv", "matrix.mtx", "barcodes.tsv"}
            sample_files = defaultdict(set)
            

            FEATURE_PATTERNS = (
                "features.tsv",
                "genes.tsv",
                "variable.tsv",
                "fc.tsv",
            )

            BARCODE_PATTERNS = (
                "barcodes.tsv",
            )

            MATRIX_PATTERNS = (
                "matrix.mtx",
            )

            REQUIRED_ROLES = {"feature", "barcode", "matrix"}

            sample_roles = defaultdict(set)

            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:

                for member in tar.getmembers():
                    if not member.isfile():
                        continue

                    name = member.name.lower()

                    # Remove .gz if present
                    filename = name[:-3] if name.endswith(".gz") else name

                    role = None

                    if any(filename.endswith(p) for p in FEATURE_PATTERNS):
                        role = "feature"
                    elif any(filename.endswith(p) for p in BARCODE_PATTERNS):
                        role = "barcode"
                    elif any(filename.endswith(p) for p in MATRIX_PATTERNS):
                        role = "matrix"

                    if not role:
                        continue

                    # Extract sample/prefix by removing role suffix
                    suffix = filename.split("_")[-1]
                    sample = filename.replace(f"_{suffix}", "")

                    sample_roles[sample].add(role)

            # Nothing valid found
            if not sample_roles:
                return False

            # Validate each sample
            for sample, roles in sample_roles.items():
                missing = REQUIRED_ROLES - roles
                if missing:
                    self.logger.info(
                        f"[TAR INVALID] Sample '{sample}' missing roles: {missing}"
                    )
                    return False

            return True

        # ---------------- Main logic ----------------

        gse_id = normalize_gse_id(gse_id)
        geo_prefix = geo_series_block_prefix(gse_id)
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{gse_id}/suppl/"

        try:
            r = requests.get(url, timeout=20)
        except Exception as e:
            self.logger.info(f"[HTTP] Exception accessing {url}: {e}")
            return False

        if r.status_code != 200:
            self.logger.info(f"[HTTP] {url} returned {r.status_code}")
            return False

        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)]

        clean_files = []
        for href in links:
            if href.startswith("?") or href.endswith("/"):
                continue
            root = href.split("?", 1)[0].split("#", 1)[0]
            if root.lower().startswith("parent"):
                continue
            clean_files.append(root)

        # 1) keyword-based matches
        keyword_pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(SUPPLEMENTARY_KEYWORDS) + r")(?![A-Za-z0-9])",
            re.IGNORECASE
        )
        keyword_matches = [f for f in clean_files if keyword_pattern.search(f)]
        if keyword_matches:
            self.logger.info(f"[MATCH] Keyword-based matches: {keyword_matches}")
            return True

        # 2) TAR inspection (strict)
        # for f in clean_files:
        #     if f.lower().endswith((".tar", ".tar.gz")):
        #         tar_url = url + f
        #         self.logger.info(f"[TAR] Inspecting {tar_url}")

        #         try:
        #             r_tar = requests.get(tar_url, stream=True, timeout=60)
        #             if r_tar.status_code != 200:
        #                 return False

        #             if not tar_has_complete_samples(r_tar.content):
        #                 self.logger.info(f"[REJECT] Incomplete samples in {f}")
        #                 return False

        #             self.logger.info(f"[ACCEPT] Valid count matrix tar: {f}")
        #             return True

        #         except Exception as e:
        #             self.logger.error(f"[TAR ERROR] {f}: {e}")
        #             return False

        # 3) fallback extension-based match (non-tar)
        for f in clean_files:
            for ext in PREFERRED_FORMATS:
                if ext not in (".tar", ".tar.gz") and f.lower().endswith(ext):
                    self.logger.info(f"[MATCH] Extension-based match: {f}")
                    return True

        self.logger.info(f"[NO MATCH] No valid supplementary files at {url}")
        return False


    async def _run_llm_check(self, info: Dict[str, Any], disease: str, filter: dict) -> Dict[str, Any]:
        """
        Runs the LLM-based dataset validation step.
        Determines whether a GEO dataset is suitable for DEG analysis
        based on species, assay type, tissue type, and control/non-control group availability.

        Args:
            info (Dict[str, Any]): Metadata or description of the dataset being checked.
            disease (str): Disease name for context.
            filter (dict): Contains keys like 'tissue_filter' and 'experiment_filter'.

        Returns:
            Dict[str, Any]: Result dictionary containing 'decision' (True/False),
                            'prompt' used, and 'reason' or 'error' if applicable.
        """
        # print(info)
            # Tissue and Experiment Control function
        def build_tissue_rule(user_tissue: str) -> str:
            """Generate clear inclusion/exclusion rules for the given tissue filter."""
            user_tissue = (user_tissue or "").lower().strip()

            if user_tissue == "blood":
                return (
                    "Sample Type: Accept if sample mentions PBMC, peripheral blood, whole blood, "
                    "buffy coat, plasma, or peripheral T lymphocytes. Reject if the dataset not derived from blood."
                )

            elif user_tissue == "tissue":
                return (
                    "Sample Type: Accept if the dataset explicitly mentions organ or primary tissues "
                    "(e.g., liver, lung, heart, tumor, biopsy). "
                )

            else:  # "any"
                return (
                    "Sample Type: Accept both tissue or blood if experiment is RNA Seq else accept"
                    "any biological material (tissue, cells, or blood) as long as "
                    "the dataset meets the assay and species criteria. Reject only if it is synthetic, "
                    "non-human, or not biological material."
                )

        def build_experiment_rule(experiment_filter: str) -> str:
            """Generate inclusion/exclusion rules for the experiment type."""
            ef = (experiment_filter or "").lower().strip()

            if ef == "rna-seq":
                return (
                    "Assay/Tech: Must clearly indicate bulk RNA-seq or transcriptome sequencing not circular RNAs or anyother type of RNA. "
                    "Accept if it mentions RNA-seq, transcriptome, Illumina HiSeq/NextSeq/NovaSeq, "
                    "featureCounts, or read counts. Reject if it is single-cell RNA-seq or cell lines or cells or microarray or circRNAs."
                )
            elif ef == "microarray":
                return (
                    "Assay/Tech: Must clearly indicate gene expression microarray. "
                    "Accept if it mentions microarray, genechip, Affymetrix, Agilent, or Illumina HT array. "
                    "Reject if it is RNA-seq or single-cell sequencing."
                )
            elif ef == "single-cell":
                return (
                    "Assay/Tech: Must clearly indicate single-cell RNA-seq. "
                    "Accept if it mentions scRNA-seq, single-cell sequencing, 10x Genomics, Drop-seq, or Smart-seq. "
                    "Reject if it is bulk RNA-seq or microarray."
                )
            else:
                return (
                    f"Assay/Tech: Must match the provided experiment type '{experiment_filter}'. "
                    "Reject if the dataset uses unrelated methods."
                )
        MIN_CTRL = 1      # set to 2 later if you want a bit more rigor
        MIN_NONCTRL = 1   # set to 2 later if you want a bit more rigor

        try:
            # ---------------------------
            # Normalize input filters
            # ---------------------------
            tissue_filter = (filter.get("tissue_filter") or "any").strip().lower()
            experiment_filter = (filter.get("experiment_filter") or "rna-seq").strip().lower()
            other_filter = (filter.get("other_filter"))
            # ---------------------------
            # Build prompt rules
            # ---------------------------
            tissue_rule = build_tissue_rule(tissue_filter)
            experiment_rule = build_experiment_rule(experiment_filter)

            # ---------------------------
            # Compose LLM prompt
            # ---------------------------
            prompt = f"""
            You are an expert in biomedical dataset curation.

            Decide if the dataset below is suitable for ***{disease} and ***{experiment_filter}** analysis
            using samples from **{tissue_filter}** type

            Return:
            - `True` only if **all** inclusion criteria below are satisfied.
            - `False` otherwise.

            ### Inclusion Criteria
            - **Species:** Human (Homo sapiens) only.
            - {experiment_rule}
            - {tissue_rule}
            - {other_filter}
            
            ### Technology vs Sample Rules
            - **Technology** describes *how sequencing was performed* (e.g., RNA-seq).
            - **Model System** describes *what was sequenced* (e.g. PBMC, tumor biopsy,
            MDA-MB-231 breast cancer cells).
            - If the technology clearly indicates **RNA-seq**, do **not** classify sample type as "cell lines" or "cells" or "single cell".
            Instead:
            • Maintain {experiment_filter} as the technology, and ignore any other type of technology
            • Classify biological source under **model system**.
            - Select only primary biological materials (tissue/blood) over immortalized models if RNA Seq is mentioned.
            unless the study design explicitly supports biological relevance to {tissue_filter}. Also ignore any other type of RNA like circular RNA, circRNAs, Single RNA etc

            ### Notes
            - If assay or biological source are ambiguous but still match the intent of 
            **{experiment_filter}** and **{tissue_filter}**, and control/non-control groups can be formed,
            return `True`; otherwise `False`.

            Dataset to evaluate:
            {info}
            """

            # ---------------------------
            # Call LLM (using async API)
            # ---------------------------

            system_msg = (
              'Respond in JSON ONLY (no code fences). Use exactly: '
              '{ "valid": <true|false>, "justification": "<one concise paragraph>" }'
          )
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
            )

            # ---------------------------
            # Normalize & interpret result
            # ---------------------------
            result = json.loads(response.choices[0].message.content)
            # self.logger.debug(f"LLM check result: {result}")
            return result
           

        except Exception as e:
            self.logger.error(f"LLM validation failed for {disease}: {e}")
            return {
                "decision": False,
                "error": str(e),
                "prompt": None
            }

    async def process_dataset_and_validate(self, dataset, tissue_filter, experiment_filter, output_dir: str = "temp") -> str:
        """
        One-shot pipeline:
        - Reads GSE ID from dataset
        - Downloads and decompresses series matrix
        - Extracts series metadata (title, summary)
        - Parses sample metadata
        - Calls LLM to validate groups/tissue and dataset suitability
        Returns:
        str: JSON string returned by the LLM (unchanged)
        Raises:
        Exception on download/parse/LLM errors.
        """
        import re
        self.logger.info(f"Starting dataset validation for: {dataset}, tissue filter: {tissue_filter}")
     
        def normalize_gse_id(raw: str) -> str:
            """
            Convert inputs like '200294225' -> 'GSE294225'.
            Also handles 'GSE089408' -> 'GSE89408', '89408' -> 'GSE89408', etc.
            """
            if raw is None:
                raise ValueError("Missing dataset id.")
            import re
            s = str(raw).strip()

            # Common internal form starts with '200' — treat that as 'GSE'
            if re.fullmatch(r'200\d+', s):
                digits = s[3:]  # drop the '200' prefix
                return f"GSE{int(digits)}"  # int() strips any leading zeros

            # Generic: optional 'GSE', then digits
            m = re.search(r'(?:GSE)?0*(\d+)$', s, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Could not parse a GEO series id from '{s}'.")
            return f"GSE{int(m.group(1))}"

        def geo_series_block_prefix(gse_id: str) -> str:
            """
            GEO FTP series block folder.
            GSE294225 -> 'GSE294nnn'
            GSE89408  -> 'GSE89nnn'
            < 1000    -> 'GSEnnn'
            """
            import re
            if not re.match(r'^GSE\d+$', gse_id):
                raise ValueError(f"Not a valid GSE id: '{gse_id}'")
            n = int(gse_id[3:])
            return "GSEnnn" if n < 1000 else f"GSE{n // 1000}nnn"

        def get_matrix_url(prefix: str, gse_id: str) -> str:
            base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/matrix/"
            # List the directory contents
            resp = requests.get(base_url)
            if resp.status_code != 200:
                raise Exception(f"Failed to access {base_url}")

            # Find a file that starts with gse_id and ends with series_matrix.txt.gz
            matches = re.findall(r'href="([^"]+)"', resp.text)
            for fname in matches:
                if fname.startswith(gse_id) and fname.endswith("series_matrix.txt.gz"):
                    return base_url + fname

            raise FileNotFoundError(f"No matching series matrix file found for {gse_id} in {base_url}")

        dataset = normalize_gse_id(dataset)       
        gse_id = dataset
        self.logger.debug(f"Normalized GSE ID: {gse_id}")
        if not gse_id:
            self.logger.error("Empty dataset ID provided")
            raise ValueError("DatasetInfo object must have a non-empty dataset_id.")

        prefix = geo_series_block_prefix(gse_id)   
        url = get_matrix_url(prefix,gse_id)
        self.logger.debug(f"Matrix URL: {url}")
        matrix_filename = f"{gse_id}_series_matrix.txt.gz"

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, matrix_filename)
        self.logger.debug(f"Output path: {output_path}")
        self.logger.info(f"Downloading series matrix from: {url}")

        with requests.get(url, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                self.logger.error(f"Failed to download file, status code: {resp.status_code}")
                raise Exception("Failed to download file")
            self.logger.info("Successfully downloaded series matrix, writing to file")
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        self.logger.info("Series matrix file written successfully")

        # ---------- Decompress ----------
        self.logger.info("Decompressing series matrix file")
        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as gz:
                data = gz.read()
            self.logger.debug(f"Decompressed data length: {len(data)} characters")
        except Exception as e:
            self.logger.error(f"Error reading gzipped file: {e}")
            raise Exception(f"Error reading gzipped file: {e}")

        # ---------- Helpers ----------
        def extract_series_metadata(data_str: str) -> dict:
            self.logger.debug("Extracting series metadata")
            title = ""
            summary = ""
            for line in data_str.splitlines():
                if line.startswith("!Series_title"):
                    title = line.split("\t", 1)[-1].strip().strip('"')
                elif line.startswith("!Series_summary"):
                    summary = line.split("\t", 1)[-1].strip().strip('"')
            
            return {"title": title, "summary": summary}

        def parse_sample_metadata(sample_section: str) -> List[dict]:
            self.logger.debug("Parsing sample metadata")
            lines = sample_section.splitlines()
            sample_dict = {}
            for line in lines:
                if line.startswith("!Sample_") and "\t" in line:
                    parts = line.split("\t")
                    key = parts[0].replace("!Sample_", "").strip()
                    values = [v.strip().strip('"') for v in parts[1:]]
                    sample_dict[key] = values

            if not sample_dict:
                self.logger.error("No sample metadata found")
                raise ValueError("No sample metadata found. Please check input format.")

            sample_lengths = [len(v) for v in sample_dict.values()]
            if len(set(sample_lengths)) > 1:
                self.logger.error(f"Inconsistent sample lengths: {sample_lengths}")
                raise ValueError(f"Inconsistent sample lengths in metadata fields: {sample_lengths}")

            num_samples = sample_lengths[0]
            self.logger.debug(f"Parsed {num_samples} samples")
            return [{key: sample_dict[key][i] for key in sample_dict} for i in range(num_samples)]

        # ---------- Extract metadata ----------
       
        def _get_encoder(model: str):
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")

        def tokens_of_text(text: str, model: str) -> int:
            enc = _get_encoder(model)
            return len(enc.encode(text or ""))

        def truncate_text_to_tokens(text: str, model: str, max_tokens: int) -> str:
            enc = _get_encoder(model)
            toks = enc.encode(text or "")
            if len(toks) <= max_tokens:
                return text or ""
            return enc.decode(toks[:max_tokens])

        def count_chat_tokens(messages, model: str) -> int:
            """
            Approximate token counter for Chat Completions with role/content pairs.
            Good enough for budgeting; avoids overflows before the API call.
            """
            total = 0
            for m in messages:
                total += tokens_of_text(m.get("content",""), model)
                total += tokens_of_text(m.get("role",""), model)  # tiny overhead
            return total

        def fit_messages_to_budget(messages, model: str, max_input_tokens: int):
            """
            Ensure the combined messages fit within max_input_tokens.
            Trim the last user message first; if still too large, trim system.
            """
            msgs = [dict(m) for m in messages]  # shallow copy

            def try_fit():
                return count_chat_tokens(msgs, model) <= max_input_tokens

            if try_fit():
                return msgs

            # 1) Trim the last user message (common case in your pattern)
            last_user_idx = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
            if last_user_idx is not None:
                over = count_chat_tokens(msgs, model) - max_input_tokens
                if over > 0:
                    # give it some slack: trim a bit more than strictly needed
                    # to avoid off-by-a-few reflows
                    target_reduce = over + 256
                    user_text = msgs[last_user_idx]["content"]
                    user_len = tokens_of_text(user_text, model)
                    new_len = max(0, user_len - target_reduce)
                    msgs[last_user_idx]["content"] = truncate_text_to_tokens(user_text, model, new_len)
            if try_fit():
                return msgs

            # 2) If still too big, trim the system message
            system_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
            if system_idx is not None:
                over = count_chat_tokens(msgs, model) - max_input_tokens
                if over > 0:
                    target_reduce = over + 256
                    sys_text = msgs[system_idx]["content"]
                    sys_len = tokens_of_text(sys_text, model)
                    new_len = max(0, sys_len - target_reduce)
                    msgs[system_idx]["content"] = truncate_text_to_tokens(sys_text, model, new_len)
            if try_fit():
                return msgs

            # 3) As a last resort, hard-trim both to half the budget each
            # (should basically never be hit if you only have system+user)
            half = max_input_tokens // 2
            if system_idx is not None:
                msgs[system_idx]["content"] = truncate_text_to_tokens(msgs[system_idx]["content"], model, half)
            if last_user_idx is not None:
                msgs[last_user_idx]["content"] = truncate_text_to_tokens(msgs[last_user_idx]["content"], model, half)
            return msgs

        series_meta = extract_series_metadata(data)
        self.logger.debug("Extracting sample data from series matrix")
        sample_data_only = "\n".join(line for line in data.splitlines() if line.startswith("!Sample_"))

        samples = parse_sample_metadata(sample_data_only)

        self.logger.debug("Setting up tissue filtering rules")
        # ---------- Tissue rule ----------
        filter = tissue_filter.lower()
        def build_tissue_rule(user_tissue: str) -> str:
            """Generate clear inclusion/exclusion rules for the given tissue filter."""
            user_tissue = (user_tissue or "").lower().strip()

            if user_tissue == "blood":
                return (
                    "Sample Type: Accept if sample mentions PBMC, peripheral blood, whole blood, "
                    "buffy coat, plasma, or peripheral T lymphocytes. Reject if the dataset not derived from blood."
                )

            elif user_tissue == "tissue":
                return (
                    "Sample Type: Accept if the dataset explicitly mentions organ or primary tissues "
                    "(e.g., liver, lung, heart, tumor, biopsy). "
                )

            else:  # "any"
                return (
                    """Sample Type: Accept both tissue or blood not any from this list ["cell line", "cell-line", "cell culture", "in vitro", "cultured cells",
                                # single-cell
                                "single cell", "single-cell", "scrna", "scrna-seq", "10x genomics", "smart-seq",
                                # common human cancer cell lines
                                "hela", "hek293"]  if experiment is RNA Seq else accept"
                    "any biological material (tissue, cells, or blood) as long as "
                    "the dataset meets the assay and species criteria. Reject only if it is synthetic, "
                    "non-human, or not biological material."""
                )

        tissue_rule = build_tissue_rule(tissue_filter)
        import json
            # ---------- LLM prompt ----------
        system_prompt = f"""
        You are a biomedical metadata validation agent.
        Classify each sample and determine dataset validity.

        Output JSON only:
        - "Groups Exist": list of distinct groups
        - "Tissue Types": list of detected tissue types
        - "SampleClassification": list of dicts with "sample_id", "group", "tissue_type"
        - "Valid_Dataset": true or false
        - "Reason": brief reason for acceptance/rejection

        Group assignment uses control/disease keywords.
        Tissue rules:
        {tissue_rule}

        ### Technology vs Sample Rules
            - **Technology** describes *how sequencing was performed* (e.g., RNA-seq).
            - **Model System** describes *what was sequenced* (e.g. PBMC, tumor biopsy,
            MDA-MB-231 breast cancer cells).
            - If the technology clearly indicates **RNA-seq**, do **not** classify sample type as "cell lines" or "cells" or "single cell".
            Instead:
            • Maintain {experiment_filter} as the technology, and
            • Classify biological source under **model system**.
            - Select only primary biological materials (tissue/blood) over immortalized models if RNA Seq is mentioned.
            unless the study design explicitly supports biological relevance to {tissue_filter}.

            ### Notes
            - If assay or biological source are ambiguous but still match the intent of 
            **{experiment_filter}** and **{tissue_filter}**, and control/non-control groups can be formed,
            return `True`; otherwise `False`.

        """
        user_prompt = f"""
        Series Title: {series_meta.get("title", "")}
        Series Summary: {series_meta.get("summary", "")}
        Sample Metadata:
        {json.dumps(samples, indent=2)}
        """
        # -------------------------------
        # Assemble messages
        # -------------------------------
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        import json
        import re

        # Canonical list of terms that indicate cell-line or single-cell studies
        FORBIDDEN_TERMS = [
            # generic
            "cell line", "cell-line", "cell culture", "in vitro", "cultured cells", "cells",
            # single-cell
            "single cell", "single-cell", "scrna", "scrna-seq", "10x genomics", "smart-seq", "circRNA", "circRNAs", "circular RNA",
            # common human cancer cell lines
            "hela", "hek293", "mda-mb", "ht29", "hct116", "k562", "a549", "mcf7", "pc3",
        ]


        def should_reject_dataset(series_meta: dict, samples: list) -> bool:
            """
            Returns True if any red-flag cell line terms appear in:
            - Series title
            - Series summary
            - Sample metadata (titles, characteristics, json text)
            """

            text = ""

            # Extract searchable text
            text += f" {series_meta.get('title', '')} "
            text += f" {series_meta.get('summary', '')} "
            text += f" {json.dumps(samples)} "

            text_lower = text.lower()
            # print("Samples Text: ", text)
            # Check forbidden terms
            for term in FORBIDDEN_TERMS:
                if term in text_lower:
                    return True

            return False

        MAX_INPUT_TOKENS = 120000   # example budget; adjust for gpt-4o context
        messages = fit_messages_to_budget(messages, "gpt-4o", MAX_INPUT_TOKENS)
        # dsx
        # -------------------------------
        # Send to API
        # -------------------------------
        # if experiment_filter == "rna-seq" or experiment_filter == "rna seq" or experiment_filter == "rna":
        #     result = should_reject_dataset(series_meta, samples)
        # # print("Series Meta : ",series_meta)
        # # print("Samples : ", json.dumps(samples))
        #     if result == True : 
        #         return result,series_meta,samples
        
        self.logger.info("Sending validation request to OpenAI API")
        resp = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        self.logger.debug("Received response from OpenAI API")
        return resp.choices[0].message.content,series_meta,samples
    
    async def _passes_sample_filter(self, dataset, tissue_filter, experiment_filter) -> bool:
        """
            Calls process_dataset_and_validate(dataset) which returns a JSON string.
            Saves the JSON to disk and returns True iff 'Valid_Dataset' is True.
        """
        self.logger.info(f"Running sample filter for dataset: {dataset}, tissue filter: {tissue_filter}")
        try:
            # Run validation
            self.logger.debug("Calling process_dataset_and_validate")
            result_str, series_meta, samples = await self.process_dataset_and_validate(dataset, tissue_filter, experiment_filter)

            if result_str == True:
                return False, series_meta, samples
            
            self.logger.debug("Received validation result")
            import re
            result_json = result_str
            self.logger.debug(f"Raw JSON result: {result_json[:200]}...")
            try:
                result = json.loads(result_json)
                self.logger.debug("Successfully parsed JSON result")
            except:
                self.logger.debug("JSON parsing failed, attempting to clean and retry")
                cleaned = result_str.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
                result = json.loads(cleaned)
                self.logger.debug("Successfully parsed cleaned JSON result")

            base_out = "downloads"
            save_dir = "validations"
           
            # Use flag to decide pass/fail
            is_valid = bool(result.get("Valid_Dataset", False))
            self.logger.info(f"Dataset validation result: {is_valid}")
            if not is_valid:
                self.logger.info(f"Dataset {dataset} failed validation")
                return False, series_meta, samples

            self.logger.info(f"Dataset {dataset} passed validation")
            return True, series_meta, samples

        except Exception as e:
            self.logger.error(f"Error in LLM filtering for {dataset}: {e}")
            return False, series_meta, samples

    def _check_group_detection_potential(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick assessment of whether dataset has potential for good group detection
        based on sample names and metadata fields.
        """
        self.logger.debug("Checking group detection potential")
        # Control group keywords
        control_keywords = ['control', 'ctrl', 'normal', 'healthy', 'wild type', 'vehicle',
                           'untreated', 'baseline', 'naive', 'sham', 'mock', 'negative', 'wildtype']
        
        # Disease group keywords
        disease_keywords = ['disease', 'tumor', 'cancer', 'melanoma', 'carcinoma', 'lesion',
                           'patient', 'treated', 'mutant', 'ko', 'kd', 'infected', 'exposed', 'positive']
        
        # Combine text for analysis
        text_to_check = ""
        if 'title' in dataset_info:
            text_to_check += dataset_info['title'].lower() + " "
        if 'summary' in dataset_info:
            text_to_check += dataset_info['summary'].lower() + " "
        
        samples = dataset_info.get('Samples', [])
        sample_names = []
        if isinstance(samples, list):
            for sample in samples[:15]:  # Check first 15 samples
                if isinstance(sample, dict) and 'Title' in sample:
                    sample_name = sample['Title'].lower()
                    sample_names.append(sample_name)
                    text_to_check += sample_name + " "
        
        # Count potential control/disease samples
        control_matches = sum(1 for sample in sample_names
                             if any(kw in sample for kw in control_keywords))
        disease_matches = sum(1 for sample in sample_names
                             if any(kw in sample for kw in disease_keywords))
        
        # Check for preferred tissues
        tissue_matches = sum(1 for tissue in self.BLOOD_KEYWORDS if tissue in text_to_check)
        
        result = {
            'potential_controls': control_matches,
            'potential_disease': disease_matches,
            'has_preferred_tissue': tissue_matches > 0,
            'tissue_types_found': [tissue for tissue in self.BLOOD_KEYWORDS if tissue in text_to_check],
            'good_group_potential': control_matches >= 3 and disease_matches >= 3,
            'total_samples': len(sample_names)
        }
        self.logger.debug(f"Group detection potential: {result}")
        return result
    
    async def create_context(self) -> aiohttp.ClientSession:
        """Create HTTP session for queries."""
        self.logger.debug("Creating HTTP session context")
        timeout = aiohttp.ClientTimeout(total=self.config.network_config.timeout)
        headers = {
            'User-Agent': self.config.network_config.user_agent
        }
        
        session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=100)
        )
        self.logger.debug("HTTP session created successfully")
        return session
    
    async def close_context(self, session: aiohttp.ClientSession):
        """Close HTTP session."""
        self.logger.debug("Closing HTTP session")
        await session.close()
        self.logger.debug("HTTP session closed")
    
    async def execute(self, 
                     base_url: str, 
                     params: Dict[str, Any],
                     method: str = "GET",
                     headers: Optional[Dict[str, str]] = None) -> ToolResult[List[Dict[str, Any]]]:
        """
        Execute a query with retry logic.
        
        Args:
            base_url: Base URL for the query
            params: Query parameters
            method: HTTP method (GET or POST)
            headers: Optional additional headers
            
        Returns:
            ToolResult containing query results
        """
        self.logger.info(f"Executing {method} query to: {base_url}")
        self.logger.debug(f"Query parameters: {params}")
        
        if not self.validate_input(base_url, params):
            self.logger.error("Input validation failed")
            return ToolResult(
                success=False,
                data=[],
                error="Invalid input parameters",
                details={"base_url": base_url, "params": params}
            )
        
        try:
            result = await self._execute_query(base_url, params, method, headers)
            self.logger.debug(f"Query executed successfully, got {len(result)} results")
            
            if not self.validate_output(result):
                self.logger.error("Output validation failed")
                return ToolResult(
                    success=False,
                    data=[],
                    error="Invalid output format",
                    details={"result_type": type(result).__name__}
                )
            
            self.logger.info("Query execution completed successfully")
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                error=str(e),
                details={"exception": type(e).__name__}
            )
    
    async def _execute_query(self, 
                           base_url: str, 
                           params: Dict[str, Any],
                           method: str = "GET",
                           headers: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Execute the actual query."""
        self.logger.debug(f"Executing {method} query with {len(params)} parameters")
        session = self.context
        additional_headers = headers or {}
        
        try:
            if method.upper() == "GET":
                url = f"{base_url}?{urlencode(params)}"
                self.logger.debug(f"GET URL: {url}")
                async with session.get(url, headers=additional_headers) as response:
                    return await self._process_response(response)
            
            elif method.upper() == "POST":
                self.logger.debug(f"POST to: {base_url}")
                async with session.post(base_url, data=params, headers=additional_headers) as response:
                    return await self._process_response(response)
            
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                raise QueryError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during query: {e}")
            raise NetworkError(f"Network error during query: {e}", endpoint=base_url)
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise QueryError(f"Query execution failed: {e}", query=str(params))
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> List[Dict[str, Any]]:
        """Process HTTP response and extract data."""
        self.logger.debug(f"Processing response with status: {response.status}")
        if response.status != 200:
            self.logger.error(f"HTTP error {response.status}: {response.reason}")
            raise NetworkError(
                f"HTTP error {response.status}: {response.reason}",
                endpoint=str(response.url),
                status_code=response.status
            )
        
        content_type = response.headers.get('content-type', '').lower()
        self.logger.debug(f"Response content type: {content_type}")
        text = await response.text()
        self.logger.debug(f"Response text length: {len(text)} characters")
        
        try:
            if 'xml' in content_type:
                self.logger.debug("Parsing as XML")
                return self._parse_xml_response(text)
            elif 'json' in content_type:
                self.logger.debug("Parsing as JSON")
                return self._parse_json_response(text)
            else:
                # Try to detect format from content
                if text.strip().startswith('<'):
                    self.logger.debug("Detected XML format from content")
                    return self._parse_xml_response(text)
                elif text.strip().startswith('{') or text.strip().startswith('['):
                    self.logger.debug("Detected JSON format from content")
                    return self._parse_json_response(text)
                else:
                    self.logger.debug("Parsing as text")
                    return self._parse_text_response(text)
                    
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            raise QueryError(f"Failed to parse response: {e}", details={"content_type": content_type})
    
    def _parse_xml_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse XML response."""
        try:
            # Add debugging to see what we're getting
            self.logger.debug(f"Parsing XML response (first 200 chars): {xml_text[:200]}")
            
            data = xmltodict.parse(xml_text)
            
            # Add debugging to see parsed structure
            self.logger.debug(f"Parsed XML structure keys: {list(data.keys()) if data else 'None'}")
            
            # Check if parsing returned None
            if data is None:
                raise QueryError("XML parsing returned None - invalid XML content")
            
            # Handle different XML structures
            if 'eSearchResult' in data:
                # NCBI eSearch format - extract the eSearchResult
                esearch_result = data['eSearchResult']
                self.logger.debug(f"eSearchResult keys: {list(esearch_result.keys()) if esearch_result else 'None'}")
                return self._parse_ncbi_esearch(esearch_result)
            elif 'response' in data:
                # Generic response format
                return self._extract_list_from_dict(data['response'])
            else:
                # Try to extract any list-like structure
                return self._extract_list_from_dict(data)
                
        except Exception as e:
            self.logger.error(f"XML parsing failed: {e}")
            self.logger.debug(f"XML content that failed: {xml_text}")
            raise QueryError(f"XML parsing failed: {e}")
    
    def _parse_json_response(self, json_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response."""
        self.logger.debug("Parsing JSON response")
        try:
            data = json.loads(json_text)
            
            if isinstance(data, list):
                self.logger.debug(f"JSON response is a list with {len(data)} items")
                return data
            elif isinstance(data, dict):
                self.logger.debug("JSON response is a dict, extracting list")
                return self._extract_list_from_dict(data)
            else:
                self.logger.debug("JSON response is a single value")
                return [{"data": data}]
                
        except Exception as e:
            self.logger.error(f"JSON parsing failed: {e}")
            raise QueryError(f"JSON parsing failed: {e}")
    
    def _parse_text_response(self, text: str) -> List[Dict[str, Any]]:
        """Parse plain text response."""
        self.logger.debug("Parsing text response")
        lines = text.strip().split('\n')
        result = [{"line": i, "content": line} for i, line in enumerate(lines) if line.strip()]
        self.logger.debug(f"Parsed {len(result)} lines from text response")
        return result
    
    def _parse_ncbi_esearch(self, esearch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse NCBI eSearch response."""
        if esearch_data is None:
            self.logger.warning("eSearch data is None")
            return []
        
        # Debug: show the full eSearch structure
        self.logger.debug(f"Full eSearchResult structure: {esearch_data}")
        
        # Get count
        count = int(esearch_data.get('Count', 0))
        self.logger.debug(f"NCBI eSearch count: {count}")
        
        if count == 0:
            self.logger.info("Count is 0, returning empty list")
            return []
        
        # Get ID list
        id_list = esearch_data.get('IdList')
        self.logger.debug(f"IdList structure: {id_list}")
        
        # Handle count-only queries (retmax=0) where IdList is None
        if id_list is None:
            self.logger.debug("IdList is None - this is a count-only query")
            # Return count information for count-only queries
            return [{"count": count, "source": "ncbi"}]
        
        # Handle empty IdList (no results)
        if not id_list:
            self.logger.warning("IdList is empty")
            return []
        
        # Get the actual IDs
        ids = id_list.get('Id', [])
        self.logger.debug(f"Raw IDs from eSearch: {ids}")
        
        # Normalize to list (same as working code)
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            ids = []
        
        self.logger.debug(f"Normalized IDs: {ids}")
        
        # Return in format expected by GEO agent
        return [{"id": id_val, "source": "ncbi", "count": count} for id_val in ids]
    
    def _extract_list_from_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract list-like data from nested dictionary."""
        self.logger.debug("Extracting list from dictionary")
        # Look for common list-containing fields
        list_fields = ['results', 'data', 'items', 'records', 'datasets', 'entries']
        
        for field in list_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    self.logger.debug(f"Found list in field '{field}' with {len(value)} items")
                    return value
                elif isinstance(value, dict):
                    self.logger.debug(f"Found dict in field '{field}', wrapping as single item")
                    return [value]
        
        # If no list found, return the dictionary as a single item
        self.logger.debug("No list fields found, returning dict as single item")
        return [data]
    
    from typing import Optional

    import json

    def _query_explicitly_includes_treatment(self, user_query: str) -> bool:
        """
        LLM-based intent classifier.
        Returns True ONLY if the query explicitly requests treatment/intervention datasets.
        Default (no mention / ambiguous) => False
        """

        if not user_query or not user_query.strip():
            return False

        # Cache per query to avoid repeated LLM calls
        if not hasattr(self, "_treatment_intent_cache"):
            self._treatment_intent_cache = {}

        cache_key = user_query.strip().lower()
        if cache_key in self._treatment_intent_cache:
            return self._treatment_intent_cache[cache_key]

        system_msg = (
            "You are a strict intent classifier for bioinformatics dataset queries.\n"
            "Determine whether the user query EXPLICITLY requests treatment/intervention datasets.\n\n"
            "Return STRICT JSON only:\n"
            '{ "include_treatment": true | false }\n\n'
            "Rules:\n"
            "- include_treatment=true ONLY if the query clearly mentions treatment/intervention/perturbation\n"
            "  (e.g., drug-treated, treated vs control, stimulation, dose response, therapy, compound exposure).\n"
            "- If treatment is not mentioned, return false.\n"
            "- If ambiguous, return false.\n"
            "- Disease cohort queries without intervention mention => false.\n"
        )

        prompt = user_query

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            include_treatment = bool(parsed.get("include_treatment", False))
        except Exception:
            # STRICT default: filter treatments unless explicitly requested
            include_treatment = False

        self._treatment_intent_cache[cache_key] = include_treatment
        return include_treatment

    import json
    from typing import Dict

    def _llm_rnaseq_cellline_intent(self, user_query: str) -> Dict[str, bool]:
        """
        Returns:
        {
            "is_rnaseq_request": bool,
            "wants_cell_line": bool,
            "exclude_cell_line": bool
        }

        Key behavior:
        - If user wants cell line => exclude_cell_line MUST be False, even if RNA-seq is True.
        - If RNA-seq is True and cell line not requested => exclude_cell_line True (default).
        - If RNA-seq is False => exclude_cell_line False.
        """
        if not user_query or not user_query.strip():
            return {"is_rnaseq_request": False, "wants_cell_line": False, "exclude_cell_line": False}

        # Cache per query
        if not hasattr(self, "_intent_cache_rnaseq_cellline"):
            self._intent_cache_rnaseq_cellline = {}

        key = user_query.strip().lower()
        if key in self._intent_cache_rnaseq_cellline:
            return self._intent_cache_rnaseq_cellline[key]

        system_msg = (
            "You are a strict intent classifier for GEO dataset search.\n"
            "Given the user query, decide:\n"
            "1) Does the user explicitly request RNA-seq datasets?\n"
            "2) Does the user explicitly request cell line / in vitro datasets?\n"
            "3) Should the pipeline exclude cell-line datasets?\n\n"
            "Return STRICT JSON only:\n"
            '{ "is_rnaseq_request": true|false, "wants_cell_line": true|false, "exclude_cell_line": true|false }\n\n'
            "Rules:\n"
            "- is_rnaseq_request=true ONLY if the query clearly mentions RNA-seq / RNA seq / transcriptome sequencing.\n"
            "- wants_cell_line=true ONLY if the query clearly asks for cell line, cell-line, in vitro, cultured cells, or named lines (HeLa, HEK293, A549, etc.).\n"
            "- If wants_cell_line=true, then exclude_cell_line MUST be false (even if RNA-seq=true).\n"
            "- exclude_cell_line=true ONLY when RNA-seq is requested AND the user did NOT request cell line.\n"
            "- If ambiguous, default all to false except exclude_cell_line which should follow the rule above.\n"
        )

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
        )

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            out = {
                "is_rnaseq_request": bool(parsed.get("is_rnaseq_request", False)),
                "wants_cell_line": bool(parsed.get("wants_cell_line", False)),
                "exclude_cell_line": bool(parsed.get("exclude_cell_line", False)),
            }
        except Exception:
            # Conservative fallback: don't exclude unless we are sure it's RNA-seq
            out = {"is_rnaseq_request": False, "wants_cell_line": False, "exclude_cell_line": False}

        # Hard enforcement (guardrail) to guarantee your rule
        if out["wants_cell_line"]:
            out["exclude_cell_line"] = False
        if not out["is_rnaseq_request"]:
            out["exclude_cell_line"] = False

        self._intent_cache_rnaseq_cellline[key] = out
        return out

    async def query_ncbi_geo(self, disease_name: str, query : str, output_dir : Optional[Path] = None, filters: Optional[Dict[str, Any]] = None, max_results : Optional[int] = None) -> ToolResult[List[Dict[str, Any]]]:
        """
        Enhanced NCBI GEO database query with LLM-based filtering.
        
        Args:
            disease_name: Disease name to search for
            filters: Additional filters to apply (legacy parameter)
            max_results: Maximum number of filtered datasets to return
            
        Returns:
            ToolResult with filtered GEO dataset IDs
        """
        geo_config = self.config.geo_config
        
        geo_query = query_geo(query)

        self.logger.info(f" Searching GEO for: {geo_query}")
        print(f" Searching GEO for: {geo_query}")
        
        tissue_filter = filters['tissue_filter']
        experiment_filter = filters['experiment_filter']

        try:
            # Use Bio.Entrez directly for enhanced control
            handle = Entrez.esearch(db=geo_query["database"], term=geo_query["search_term"], retmax=5000)
            print(geo_query["database"], geo_query["search_term"])
            record = Entrez.read(handle)
            gse_ids = record['IdList']
            
            self.logger.info(f" Found {len(gse_ids)} total datasets")
            
            if not gse_ids:
                self.logger.info(f"No datasets found for disease: {disease_name}")
                return ToolResult(success=True, data=[])
            
            # # # Filter out invalid GSE IDs
            invalid_gse_ids = self._get_invalid_gse_ids(disease=disease_name, filter=filters)

            new_gse_ids = [gse_id for gse_id in gse_ids if str(gse_id) not in invalid_gse_ids]
            
            self.logger.info(f" Found {len(new_gse_ids)} datasets after filtering invalid GSE IDs")
            print(f" Found {len(new_gse_ids)} datasets after filtering invalid GSE IDs")
            
            filtered_datasets = []
            new_invalid_entries = []
            result_data = []

            processed = 0
            treatment_studies_filtered = 0
            if gse_ids:
                for gse_id in gse_ids:
                    if len(filtered_datasets) >= max_results:
                        break
                    try:
                        processed += 1
                        self.logger.debug(f"Processing GSE ID: {gse_id}")
                        
                        # Get dataset summary from NCBI
                        summary_handle = Entrez.esummary(db="gds", id=gse_id)
                        summary = Entrez.read(summary_handle)[0]

                        self.include_treatment = self._query_explicitly_includes_treatment(query)
                        print("Treatment Request or not")
                        print(self.include_treatment)
                        # Pre-filter obvious treatment studies to save API costs
                        if self._is_likely_treatment_study(summary):
                            treatment_studies_filtered += 1
                            self.logger.info(f" {gse_id} - Pre-filtered as treatment study")
                            new_invalid_entries.append((gse_id, "Likely treatment study (keyword prefilter)"))
                            continue
                        
                        if not self._has_supplementary_files(gse_id):
                            self.logger.debug(f"Skipping {gse_id} - no supplementary count files")
                            continue
                        def is_cell_line_study(summary: str) -> bool:
                            if not summary:
                                return False
                            
                            summary_lower = summary.lower()
                            print("SUmmary :",summary_lower)
                            forbidden_terms = [
                                 # generic
                                "cell line", "cell-line", "cell culture", "in vitro", "cultured cells",
                                # single-cell
                                "single cell", "single-cell", "scrna", "scrna-seq", "10x genomics", "smart-seq", "circrna","circrnas", "circular RNA", "circular rna"
                                # common human cancer cell lines
                                "hela", "hek293", "mda-mb", "ht29", "hct116", "k562", "a549", "mcf7", "pc3",
                            ]
                            print(any(term in summary_lower for term in forbidden_terms))
                            return any(term in summary_lower for term in forbidden_terms)
                        
                        intent = self._llm_rnaseq_cellline_intent(query)

                        if intent["exclude_cell_line"]:
                            # Apply cell-line rejection only when we should exclude them
                            if is_cell_line_study(summary.get("summary", "").lower()):
                                print("❌ Reject: Cell-line or single-cell study detected")
                                continue

                            if is_cell_line_study(summary.get("title", "").lower()):
                                print("❌ Reject: Cell-line or single-cell study detected")
                                continue

                            print("✔ Eligible tissue/blood dataset (cell-line excluded)")
                        else:
                            # Either not RNA-seq request, or user explicitly wants cell lines
                            print("ℹ Skipping cell-line filter based on query intent")

                        analysis = await self._run_llm_check(summary, disease_name, filters)
                        # if experiment_filter is in ["rna seq", "RNA Seq", ""]:
                        print(analysis)
                        if analysis.get("valid") and not analysis.get("is_treatment_study", False):
                            self.logger.info(f" {gse_id} - Suitable for DEG analysis")
                            print(analysis.get("justification"))
                            s_valid,series_meta,samples = await self._passes_sample_filter(gse_id, tissue_filter, experiment_filter)
                            # s_valid = True
                            print(s_valid)
                            if s_valid:
                                config = CohortRetrievalConfig()  # Assuming you have a valid config setup
                                evaluator = LLMEvaluator()
                                evaluation_tool = EvaluationTool(config, evaluator)
                                # datasets = load_datasets_from_folder(base_path=output_dir)

                                from ..agents.geo_agent import GEORetrievalAgent
                                geo_agent = GEORetrievalAgent(config)
                                def normalize_gse_id(raw: str) -> str:
                                    if raw is None:
                                        raise ValueError("Missing dataset id.")
                                    s = str(raw).strip()
                                    if re.fullmatch(r"200\d+", s):
                                        return f"GSE{int(s[3:])}"
                                    m = re.search(r"(?:GSE)?0*(\d+)$", s, re.IGNORECASE)
                                    if not m:
                                        raise ValueError(f"Could not parse GEO id from '{s}'.")
                                    return f"GSE{int(m.group(1))}"

                                gse_id = normalize_gse_id(gse_id)
                                dataset_info = await geo_agent._get_dataset_info(gse_id, disease_name, filters)
                                results = await evaluation_tool.evaluate_datasets(dataset_info,series_meta, samples, disease_name, query, filters)
                                
                                 
                                 # Save the results to a JSON file
                                
                                for result in results:
                                       # 2. Extract all primary metric values
                                    primary_vals = list(result.primary_metrics.values())

                                    # 3. If ANY metric < 0.5 → skip
                                    if any(v < 0.5 for v in primary_vals):
                                        continue

                                    # 4. Otherwise add the GSE ID
                                    filtered_datasets.append(result.dataset_id)

                                    # Build your record (includes run_llm + sample filter + evaluation info)
                                    selection_record = {
                                        "selected_at_utc": datetime.utcnow().isoformat() + "Z",
                                        "query": query,
                                        "disease_name": disease_name,
                                        "filters": filters,
                                        "gds_id": gse_id,
                                        "dataset_id": result.dataset_id,

                                        # requested fields:
                                        "run_llm_check": analysis,
                                        "passes_sample_filter": {
                                            "valid": s_valid,
                                            "series_meta": series_meta,
                                            "samples": samples,
                                        },

                                        # optional but very useful:
                                        "evaluation": {
                                            "primary_metrics": result.primary_metrics,
                                            "composite_scores": result.composite_scores,
                                            "detailed_metrics": result.detailed_metrics,
                                            "supporting_references": result.supporting_references,
                                            "justification": result.justification,
                                            "error_message": result.error_message,
                                        },
                                    }

                                    # Save JSON on selection
                                    self._save_selected_dataset_json(output_dir, result.dataset_id, selection_record)


                                    result_data.append({
                                            "dataset_id": result.dataset_id,
                                            "primary_metrics": result.primary_metrics,
                                            "composite_scores": result.composite_scores,
                                            "detailed_metrics": result.detailed_metrics,
                                            "supporting_references": result.supporting_references,
                                            "justification": result.justification,
                                            "error_message": result.error_message
                                    })
                                

                            else:
                                self.logger.info(f" {gse_id} - Failed sample-level validation")
                                new_invalid_entries.append((gse_id, "Failed sample-level validation (Valid_Dataset=false)"))
                        else:
                            reason = str(analysis.get("justification", "Rejected by LLM")).strip()
                            short_reason = (reason[:300] + "…") if len(reason) > 300 else reason
                            if analysis.get("is_treatment_study", False):
                                msg = f"Treatment study (LLM): {short_reason}"
                                # self.logger.info(f" {gse_id} - {msg}")
                                new_invalid_entries.append((gse_id, msg))
                            else:
                                msg = f"Not suitable (LLM): {short_reason}"
                                # self.logger.info(f" {gse_id} - {msg}")
                                new_invalid_entries.append((gse_id, msg))

                    except Exception as e:
                        self.logger.warning(f" Error processing {gse_id}: {e}")
                        new_invalid_entries.append((gse_id, f"Processing error: {e}"))
              
                if new_invalid_entries:
                    self._append_invalid_gse_ids(
                        new_invalid_entries,
                        disease=disease_name,
                        filter=filters
                    )
                    # Update would be here 
            filepath = os.path.join(output_dir, "evaluation_results.json")
            with open(filepath, "w", encoding="utf-8") as json_file:
                json.dump(result_data, json_file, ensure_ascii=False, indent=4)
            # Print filtering summary
            self.logger.info(f" FILTERING SUMMARY:")
            self.logger.info(f"   Total processed: {processed}")
            self.logger.info(f"   Pre-filtered treatment studies: {treatment_studies_filtered}")
            self.logger.info(f"   Suitable datasets found: {len(filtered_datasets)}")
            
            print(f"   FILTERING SUMMARY:")
            print(f"   Total processed: {processed}")
            print(f"   Pre-filtered treatment studies: {treatment_studies_filtered}")
            print(f"   Suitable datasets found: {len(filtered_datasets)}")
            print(f" Final filtered datasets: {len(filtered_datasets)}")
            
            # Return in the format expected by the GEO agent
            result_data = [{"id": id_val, "source": "ncbi", "filtered": True} for id_val in filtered_datasets]
            return ToolResult(success=True, data=result_data)
            
        except Exception as e:
            self.logger.error(f"Enhanced GEO query failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                error=str(e),
                details={"exception": type(e).__name__}
            )
    
    def validate_input(self, base_url: str, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        self.logger.debug("Validating input parameters")
        if not base_url or not isinstance(base_url, str):
            self.logger.error("Invalid base_url")
            return False
        
        if not base_url.startswith(('http://', 'https://')):
            self.logger.error("base_url must start with http:// or https://")
            return False
        
        if not isinstance(params, dict):
            self.logger.error("params must be a dictionary")
            return False
        
        self.logger.debug("Input validation passed")
        return True
    
    def validate_output(self, result: List[Dict[str, Any]]) -> bool:
        """Validate output format."""
        self.logger.debug("Validating output format")
        if not isinstance(result, list):
            self.logger.error("Result must be a list")
            return False
        
        for item in result:
            if not isinstance(item, dict):
                self.logger.error("All result items must be dictionaries")
                return False
        
        self.logger.debug("Output validation passed")
        return True 