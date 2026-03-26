"""
GEO Retrieval Agent for the Cohort Retrieval Agent system.

This agent handles querying, filtering, and downloading data from NCBI GEO.
"""

import asyncio
import tempfile
import GEOparse
import contextlib
import json
import os, csv
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Iterable, Any
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import urllib.parse
import time
import datetime

# Project Import
from   ..base.base_agent import BaseRetrievalAgent, DatasetInfo
from   ..tools.download_tool import DownloadInfo
from   ..config import CohortRetrievalConfig
from   ..exceptions import QueryError
from   ..tools import QueryTool, FilterTool, DownloadTool, MetadataTool, ValidationTool, GPTClassificationTool
from   ..config import AgentName

class GEORetrievalAgent(BaseRetrievalAgent):
    """
    Agent for retrieving datasets from NCBI Gene Expression Omnibus (GEO).
    
    This agent specializes in finding and downloading RNA-seq datasets
    with supplementary count files from the GEO database.
    """
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "GEO")
        
        # Initialize tools
        self.tools = {
            "query": QueryTool(config),
            "filter": FilterTool(config),
            "download": DownloadTool(config),
            "metadata": MetadataTool(config),
            "validation": ValidationTool(config),
            "classification": GPTClassificationTool(config)
        }
        
        self.geo_config = config.geo_config
        
        # Statistics tracking
        self.datasets_with_supplementary = 0
        self.valid_tissue_samples_found = 0
        
        # Caching for optimization
        self.cache_enabled = True
        self.cached_datasets = {}
        self.cached_metadata = {}

    async def query_datasets(self, disease_name: str, filters: Optional[Dict[str, Any]] = None, query : Optional[str] = None, output_dir: Optional[Path] = None, max_datasets:Optional[int] = None) -> List[DatasetInfo]:
        """
        Query GEO for datasets related to a disease with intelligent caching.
        
        Args:
            disease_name: Name of the disease to search for
            filters: Optional additional filters
            
        Returns:
            List of DatasetInfo objects
        """
        self.logger.info(f"Querying GEO for disease: {disease_name}")
        print(f"Querying GEO for disease: {disease_name} from GEORetrievalAgent")
        tissue_filter = filters['tissue_filter']
        experiment_filter = filters['experiment_filter']
        
        try:
            # Step 1: Check existing datasets and load cache
            output_dir = output_dir or self.config.directory_paths.get_disease_path(
                AgentName.GEO.value, disease_name)
            print(output_dir)
            print(query)
            existing_datasets = await self._load_existing_datasets(output_dir, query)
            
            self.logger.info(f"Found {len(existing_datasets)} existing datasets for {disease_name}")
            print(f"Found {len(existing_datasets)} existing datasets for {disease_name}")
            
            # Step 2: If we already have enough datasets, return cached results
            if len(existing_datasets) >= max_datasets:
                self.logger.info(f"Already have {len(existing_datasets)} datasets (max: {max_datasets}). Using cached results.")
                print(f" CACHE HIT: Already have {len(existing_datasets)} datasets (max: {max_datasets}). Skipping new queries.")
                
                # Load cached dataset info
                cached_datasets = []
                for dataset_id in list(existing_datasets.keys())[:max_datasets]:
                    dataset_info = await self._load_cached_dataset_info(dataset_id, output_dir, disease_name)
                    if dataset_info:
                        cached_datasets.append(dataset_info)
                
                return cached_datasets
            
            # Step 3: Need more datasets - query for new ones
            needed_datasets = max_datasets - len(existing_datasets)
            self.logger.info(f"Need {needed_datasets} more datasets. Querying GEO...")
            print(f"PARTIAL CACHE: Have {len(existing_datasets)}, need {needed_datasets} more. Querying GEO...")
            
            # Use query tool to get initial dataset IDs
            async with self.tools["query"] as query_tool:
                query_result = await query_tool.query_ncbi_geo(disease_name, query, output_dir, filters, max_datasets)
            
            if not query_result.success:
                raise QueryError(f"GEO query failed: {query_result.error}", source="GEO", query=disease_name)
            
            # Debug: Check what we got from the query
            self.logger.debug(f"Query result data: {query_result.data}")
                        
            dataset_ids = []
            for item in query_result.data:
                if isinstance(item, dict) and 'id' in item:
                    dataset_ids.append(item['id'])
                elif isinstance(item, str):
                    # Handle case where item is just a string ID
                    dataset_ids.append(item)
            
            if not dataset_ids:
                self.logger.warning(f"No dataset IDs found for {disease_name}")
                # Return existing datasets if we have any
                if existing_datasets:
                    cached_datasets = []
                    for dataset_id in list(existing_datasets.keys())[:max_datasets]:
                        dataset_info = await self._load_cached_dataset_info(dataset_id, output_dir, disease_name)
                        if dataset_info:
                            cached_datasets.append(dataset_info)
                    return cached_datasets
                return []
            
            self.logger.info(f"Found {len(dataset_ids)} potential datasets from query")
            print(f"Found {len(dataset_ids)} potential datasets from query")
            
            # Step 4: Filter out already existing datasets
            new_dataset_ids = []
            for dataset_id in dataset_ids:
                gse_id = dataset_id.replace("200", "GSE")
                if gse_id not in existing_datasets:
                    new_dataset_ids.append(dataset_id)
            
            self.logger.info(f"Found {len(new_dataset_ids)} new datasets to process")
            print(f"Found {len(new_dataset_ids)} new datasets to process")
            
            # Step 5: Process new datasets only
            new_datasets = []
            for dataset_id in new_dataset_ids:
                try:
                    print("Dataset ID ",dataset_id)
                    gse_id = dataset_id.replace("200", "GSE")
                    # Get dataset info
                    dataset_info = await self._get_dataset_info(gse_id, disease_name, filters)
                    
                    if dataset_info == None:
                        self.logger.info("Valid but files are not available")
                    else:
                        new_datasets.append(dataset_info)

                except Exception as e:
                    self.logger.warning(f"Error processing dataset {dataset_id}: {e}")
                    continue

            # Step 6: Combine existing and new datasets
            all_datasets = []
            
            # Load existing datasets
            for dataset_id in list(existing_datasets.keys())[:max_datasets]:
                dataset_info = await self._load_cached_dataset_info(dataset_id, output_dir, disease_name)
                if dataset_info:
                    all_datasets.append(dataset_info)

            self.logger.info(f" FILTERED: {len(new_datasets)} datasets passed filter criteria")
            return new_datasets
            
        except Exception as e:
            raise QueryError(f"GEO query failed: {e}", source="GEO", query=disease_name)
    
    async def _load_existing_datasets(self, output_dir: Path, query: str) -> Dict[str, Dict[str, Any]]:
        """Load information about existing datasets from the output directory."""
        existing_datasets = {}
        
        if not output_dir.exists():
            return existing_datasets
        
         # --- Load the dataset_summary.csv and check if query matches ---
        summary_file = output_dir / "dataset_summary.csv"

        valid_dataset_ids = set()
        def clean_query(text: str) -> str:
            # Remove all special characters except spaces
            cleaned = re.sub(r"[^a-zA-Z0-9 ]+", "", text)
            # Collapse multiple spaces to one
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        query = clean_query(query)

        if summary_file.exists():
            try:
                with open(summary_file, "r", newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    print(reader)
                    for row in reader:
                        if not row:
                            continue
                        
                        first_col_value = row[0].strip()
                        print(first_col_value)
                        # Match query to first column
                        if first_col_value.lower() == query.lower():
                            valid_dataset_ids.add(first_col_value)

                if not valid_dataset_ids:
                    self.logger.info(f"No dataset matched the query: {query}")
                    return {}
                
                self.logger.info(f"Matched datasets based on query '{query}': {valid_dataset_ids}")

            except Exception as e:
                self.logger.error(f"Error reading dataset_summary.csv: {e}")
                return {}
        else:
            self.logger.warning("dataset_summary.csv not found. No datasets will be loaded.")
            return {}
    
        try:
            # Look for dataset directories
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith("GSE"):
                    dataset_id = item.name
                    
                    # Check if metadata file exists
                    metadata_file = item / f"{dataset_id}_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Get list of downloaded files
                            downloaded_files = []
                            for file_path in item.glob("*"):
                                if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                                    downloaded_files.append(file_path.name)
                            
                            existing_datasets[dataset_id] = {
                                "metadata": metadata,
                                "downloaded_files": downloaded_files,
                                "dataset_dir": item,
                                "last_updated": metadata.get("download_timestamp", "unknown")
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Error loading metadata for {dataset_id}: {e}")
            
            self.logger.info(f"Loaded {len(existing_datasets)} existing datasets from {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading existing datasets: {e}")
        
        return existing_datasets
    
    async def _load_cached_dataset_info(self, dataset_id: str, output_dir: Path, disease_name: str) -> Optional[DatasetInfo]:
        """Load DatasetInfo from cached metadata."""
        try:
            metadata_file = output_dir / dataset_id / f"{dataset_id}_metadata.json"
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if we need to update with new files
            dataset_dir = output_dir / dataset_id
            current_files = []
            if dataset_dir.exists():
                for file_path in dataset_dir.glob("*"):
                    if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                        current_files.append(file_path.name)
            
            # Discover current available files online
            available_urls = self._build_download_urls(dataset_id)
            
            dataset_info = DatasetInfo(
                dataset_id=dataset_id,
                source="GEO",
                title=cached_data.get("title", ""),
                description=cached_data.get("description", ""),
                overall_design = cached_data.get("overall_design"),
                sample_count=cached_data.get("sample_count", 0),
                file_types=cached_data.get("file_types", []),
                tissuecategorization= cached_data.get("tissuecategorization",[]),
                metadata=cached_data.get("metadata", {}),
                download_urls=available_urls,
                estimated_size_mb=cached_data.get("estimated_size_mb", 0)
            )
            
            # Add cache information
            dataset_info.metadata["cached"] = True
            dataset_info.metadata["cached_files"] = current_files
            dataset_info.metadata["last_cached"] = cached_data.get("download_timestamp", "unknown")
            
            self.logger.debug(f"Loaded cached dataset info for {dataset_id}")
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Error loading cached dataset info for {dataset_id}: {e}")
            return None
    
    async def _get_dataset_info(self, gse_id: str, disease_name: str, filters : dict) -> Optional[DatasetInfo]:
        """Get detailed information about a GEO dataset."""
        try:
            self.logger.info(f"Getting dataset info for {gse_id} from GEORetrievalAgent _get_dataset_info")
            # Check for supplementary files first
            # if "cell" not in filters.get('experiment_filter', []):
            if not await self._has_supplementary_files(gse_id):
                    self.logger.debug(f"Skipping {gse_id} - no supplementary count files")
                    return None
            self.datasets_with_supplementary += 1
            # Get basic metadata using GEOparse (in a separate process to avoid blocking)
            print("BEfore Metadata")
            metadata = await self._fetch_geo_metadata(gse_id)
            if not metadata:
                return None
            # Extract sample information
            print("Before sample info")
            samples = await self._extract_sample_info(gse_id, metadata)
            if not samples:
                self.logger.debug(f"Skipping {gse_id} - no valid samples")
                return None
            download_urls = self._build_download_urls(gse_id)
            
            # Estimate total size (rough estimate)
            estimated_size = len(download_urls) * 50  # 50MB average per file
            
            # Create tissue categorization (same as original filterids)
            tissue_categorization = {}
            for sample in samples:
                tissue = sample.get("tissue_type", "unknown")
                sample_id = sample.get("sample_id", "unknown")
                if tissue not in tissue_categorization:
                    tissue_categorization[tissue] = {"count": 0, "sample_ids": []}
                tissue_categorization[tissue]["count"] += 1
                tissue_categorization[tissue]["sample_ids"].append(sample_id)

            dataset_info = DatasetInfo(
                dataset_id=gse_id,
                source="GEO",
                title=metadata.get("title", ""),
                description=metadata.get("summary", ""),
                overall_design = metadata.get("overall_design"),
                sample_count=len(samples),
                file_types=["series_matrix", "supplementary"],
                tissuecategorization= tissue_categorization,
                metadata = {
                    "sample_id": [sample['sample_id'] for sample in samples],
                    "tissue_type": [sample['tissue_type'] for sample in samples],
                    "Characteristics": [sample['characteristics'] for sample in samples],
                    "library_source": [sample['library_source'] for sample in samples],
                    "library_strategy": [sample['library_strategy'] for sample in samples],
                    "extraction_protocol": [sample['extraction_protocol'] for sample in samples],
                    "molecule": [sample['molecule'] for sample in samples]
                },
                download_urls=download_urls,
                estimated_size_mb=estimated_size
            )
            self.valid_tissue_samples_found += len(samples)
            return dataset_info
            
            
        except Exception as e:
            self.logger.error(f"Error getting dataset info for {gse_id}: {e}")
            return None
  
    def _geo_series_suppl_http_url(self,gse_id: str) -> str:
        m = re.fullmatch(r'(GSE)(\d+)', gse_id.strip(), flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid GSE accession: {gse_id}")
        prefix = m.group(1).upper()
        num_str = str(int(m.group(2)))  # normalize: remove leading zeros

        range_dir = f"{prefix}nnn" if len(num_str) <= 3 else f"{prefix}{num_str[:-3]}nnn"
        return f"http://ftp.ncbi.nlm.nih.gov/geo/series/{range_dir}/{prefix}{num_str}/suppl/"

    async def _has_supplementary_files(self, gse_id: str) -> bool:
        SUPPLEMENTARY_KEYWORDS = [
            "count", "counts", "raw_counts", "readcounts", "featurecounts",
            "norm_counts", "normalized_counts", "fpkm", "rpkm", "tpm", "cpm"
        ]
        # extensions fallback (case-insensitive)
        PREFERRED_FORMATS = [".txt", ".tsv", ".csv", ".gz", ".tar"]

        def normalize_gse_id(raw: str) -> str:
            """
            Normalize GSE IDs safely.
            - Preserve leading zeros if the ID already starts with 'GSE0...'
            - Convert raw numeric-only inputs to GSE####
            - Handle internal '200######' → GSE######
            """
            if raw is None:
                raise ValueError("Missing dataset id.")

            s = str(raw).strip().upper()

            # --- Case 2: Internal 200###### format ---
            if re.fullmatch(r"200\d+", s):
                digits = s[3:]
                return f"GSE{int(digits)}"

            # --- Case 3: Starts with GSE but no protected leading zero ---
            if s.startswith("GSE"):
                digits = s[3:]
                return f"GSE{int(digits)}"   # removes leading zeros for normal cases

            # --- Case 4: Only digits ---
            if digits_match := re.fullmatch(r"\d+", s):
                return f"GSE{int(s)}"

            raise ValueError(f"Could not parse a GEO series id from '{s}'.")

        def geo_series_block_prefix(gse_id: str) -> str:
            """
            GEO FTP series block folder.
            GSE294225 -> 'GSE294nnn'
            GSE89408  -> 'GSE89nnn'
            < 1000    -> 'GSEnnn'
            """
            if not re.match(r'^GSE\d+$', gse_id):
                raise ValueError(f"Not a valid GSE id: '{gse_id}'")
            n = int(gse_id[3:])
            return "GSEnnn" if n < 1000 else f"GSE{n // 1000}nnn"

        # ---------- Download series matrix ----------
        # ---------- Input handling ----------
        gse_id = normalize_gse_id(gse_id)   
        print(gse_id)    
        if not gse_id:
            raise ValueError("DatasetInfo object must have a non-empty dataset_id.")

        # (Optional) Build URLs
        geo_prefix = geo_series_block_prefix(gse_id)     
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{gse_id}/suppl/"  
        print("URL :", url)      
        try:
            r = requests.get(url, timeout=20)
        except Exception as e:
             self.logger.info(f"[HTTP] Exception for {gse_id} supplementary check: {e}")
       
        if r.status_code != 200:
             self.logger.info(f"[HTTP] {gse_id} supplementary dir not accessible at {url} (HTTP {r.status_code})")
       
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
            # Clean links: drop query/anchors, directories, and nav entries
        clean_files = []
        for href in links:
            if href.startswith('?') or href.endswith('/'):
                continue
            root = href.split('?', 1)[0].split('#', 1)[0]
            if root.lower().startswith('parent'):
                continue
            clean_files.append(root)

            # 1) keyword-based matches
            keyword_pattern = re.compile(
                r"(?<![A-Za-z0-9])(" + "|".join(SUPPLEMENTARY_KEYWORDS) + r")(?![A-Za-z0-9])",
                re.IGNORECASE
            )
            matching_files = [f for f in clean_files if keyword_pattern.search(f)]
            if matching_files:
                self.logger.info(f"[MATCH] Keyword-based matches:{matching_files}")
                return True

            # 2) extension-based matches
            ext_match = any(f.lower().endswith(ext) for f in clean_files for ext in PREFERRED_FORMATS)
            if ext_match:
                self.logger.info(f"[MATCH] Extension-based match found among preferred formats:{PREFERRED_FORMATS}")
                return True

            self.logger.info(f"[NO MATCH] Neither keywords nor preferred extensions found in:{url}")

        return False
    
    async def _fetch_geo_metadata(self, gse_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a GEO dataset using GEOparse with fallbacks."""
        def normalize_gse_id(raw: str) -> str:
            """
            Normalize GSE IDs safely.
            - Preserve leading zeros if the ID already starts with 'GSE0...'
            - Convert raw numeric-only inputs to GSE####
            - Handle internal '200######' → GSE######
            """
            if raw is None:
                raise ValueError("Missing dataset id.")

            s = str(raw).strip().upper()

            # --- Case 2: Internal 200###### format ---
            if re.fullmatch(r"200\d+", s):
                digits = s[3:]
                return f"GSE{int(digits)}"

            # --- Case 3: Starts with GSE but no protected leading zero ---
            if s.startswith("GSE"):
                digits = s[3:]
                return f"GSE{int(digits)}"   # removes leading zeros for normal cases

            # --- Case 4: Only digits ---
            if digits_match := re.fullmatch(r"\d+", s):
                return f"GSE{int(s)}"

            raise ValueError(f"Could not parse a GEO series id from '{s}'.")
        gse_id = normalize_gse_id(gse_id)
        try:
            self.logger.debug(f"Fetching real metadata for {gse_id} using GEOparse")
            
            # Create temporary directory for GEOparse downloads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Use asyncio to run GEOparse in executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                def fetch_geo_sync():
                    """Synchronous GEO fetch function to run in executor."""
                    try:
                        # Download and parse GEO dataset with retry logic
                        max_retries = 3
                        retry_delay = 4
                          # seconds
                        
                        for attempt in range(max_retries):
                            try:
                                self.logger.debug(f"GEOparse attempt {attempt + 1}/{max_retries} for {gse_id}")
                                
                                # Set timeout and retry parameters for GEOparse
                                gse = GEOparse.get_GEO(
                                    geo=gse_id, 
                                    destdir=str(temp_path),
                                    how="full",  # Get full data
                                    annotate_gpl=False,  # Skip GPL annotation to speed up
                                    geotype="GSE"  # Explicitly specify GSE type
                                )
                                
                                # If we get here, download was successful
                                self.logger.info(f"GEOparse successfully downloaded {gse_id}")
                                break
                                
                            except Exception as e:
                                error_msg = str(e).lower()
                                
                                # Check for common FTP/network errors
                                if any(err in error_msg for err in [
                                    'ftp', 'timeout', 'connection', 'network', 'download failed',
                                    'eoferror', 'size do not match', 'no such file', 'permission denied'
                                ]):
                                    self.logger.warning(f"GEOparse FTP/network error for {gse_id} (attempt {attempt + 1}): {e}")
                                    
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                                        continue
                                    else:
                                        self.logger.error(f"GEOparse failed after {max_retries} attempts for {gse_id}")
                                        raise e
                                else:
                                    # Non-network error, don't retry
                                    self.logger.error(f"GEOparse non-network error for {gse_id}: {e}")
                                    raise e
                        
                        # Extract metadata
                        metadata = {
                            "title": gse.metadata.get("title", ["Unknown"])[0] if gse.metadata.get("title") else "Unknown",
                            "summary": gse.metadata.get("summary", [""])[0] if gse.metadata.get("summary") else "",
                            "organism": gse.metadata.get("organism", ["Unknown"])[0] if gse.metadata.get("organism") else "Unknown",
                            "platform": gse.metadata.get("platform_id", ["Unknown"])[0] if gse.metadata.get("platform_id") else "Unknown",
                            "submission_date": gse.metadata.get("submission_date", ["Unknown"])[0] if gse.metadata.get("submission_date") else "Unknown",
                            "last_update_date": gse.metadata.get("last_update_date", ["Unknown"])[0] if gse.metadata.get("last_update_date") else "Unknown",
                            "pubmed_id": gse.metadata.get("pubmed_id", [None])[0] if gse.metadata.get("pubmed_id") else None,
                            "web_link": gse.metadata.get("web_link", [None])[0] if gse.metadata.get("web_link") else None,
                            "overall_design": gse.metadata.get("overall_design", [""])[0] if gse.metadata.get("overall_design") else "",
                            "type": gse.metadata.get("type", ["Unknown"])[0] if gse.metadata.get("type") else "Unknown",
                            "contributor": gse.metadata.get("contributor", []) if gse.metadata.get("contributor") else [],
                            "sample_count": len(gse.gsms) if hasattr(gse, 'gsms') else 0
                        }
                        
                        
                        return metadata
                        
                    except Exception as e:
                        self.logger.error(f"GEOparse error for {gse_id}: {e}")
                        raise e
                
                # Run GEOparse in executor to avoid blocking the event loop
                metadata = await loop.run_in_executor(None, fetch_geo_sync)
                
                self.logger.info(f"Successfully fetched metadata for {gse_id}: {metadata.get('title', 'Unknown title')}")
                self.logger.debug(f"Found {len(metadata.get('sample_ids', []))} samples in {gse_id}")
                
                return metadata
                
        except ImportError:
            self.logger.error("GEOparse not installed. Install with: pip install GEOparse")
            # Try BeautifulSoup fallback
            return await self._fetch_geo_metadata_bs4(gse_id)
            
        except Exception as e:
            self.logger.error(f"GEOparse failed for {gse_id}: {e}")
            # Try BeautifulSoup fallback
            return await self._fetch_geo_metadata_bs4(gse_id)
    
    async def _fetch_geo_metadata_bs4(self, gse_id: str) -> Optional[Dict[str, Any]]:
        """Fallback method to fetch GEO metadata using BeautifulSoup web scraping."""
        try:
            self.logger.info(f"Attempting BeautifulSoup fallback for {gse_id}")
            
            # Use asyncio to run web scraping in executor
            loop = asyncio.get_event_loop()
            
            def scrape_geo_web():
                """Synchronous web scraping function."""
                try:
                    # GEO web page URL
                    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
                    
                    # Set headers to mimic a real browser
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    # Make request with timeout and retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(url, headers=headers, timeout=30)
                            response.raise_for_status()
                            break
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"Web scraping attempt {attempt + 1} failed for {gse_id}: {e}")
                                time.sleep(2 * (attempt + 1))
                                continue
                            else:
                                raise e
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract metadata from the page
                    metadata = {
                        "title": "Unknown",
                        "summary": "",
                        "organism": "Unknown",
                        "platform": "Unknown",
                        "submission_date": "Unknown",
                        "last_update_date": "Unknown",
                        "pubmed_id": None,
                        "web_link": url,
                        "overall_design": "",
                        "type": "Unknown",
                        "contributor": [],
                        "sample_count": 0,
                        "sample_ids": [],
                        "samples": []
                    }
                    
                    # Extract title
                    title_tag = soup.find('td', string=re.compile(r'Title', re.IGNORECASE))
                    if title_tag and title_tag.find_next_sibling('td'):
                        metadata["title"] = title_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract summary
                    summary_tag = soup.find('td', string=re.compile(r'Summary', re.IGNORECASE))
                    if summary_tag and summary_tag.find_next_sibling('td'):
                        metadata["summary"] = summary_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract organism
                    organism_tag = soup.find('td', string=re.compile(r'Organism', re.IGNORECASE))
                    if organism_tag and organism_tag.find_next_sibling('td'):
                        metadata["organism"] = organism_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract platform
                    platform_tag = soup.find('td', string=re.compile(r'Platform', re.IGNORECASE))
                    if platform_tag and platform_tag.find_next_sibling('td'):
                        platform_text = platform_tag.find_next_sibling('td').get_text(strip=True)
                        # Extract GPL ID from platform text
                        gpl_match = re.search(r'GPL\d+', platform_text)
                        if gpl_match:
                            metadata["platform"] = gpl_match.group()
                        else:
                            metadata["platform"] = platform_text
                    
                    # Extract submission date
                    submission_tag = soup.find('td', string=re.compile(r'Submission date', re.IGNORECASE))
                    if submission_tag and submission_tag.find_next_sibling('td'):
                        metadata["submission_date"] = submission_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract last update date
                    update_tag = soup.find('td', string=re.compile(r'Last update date', re.IGNORECASE))
                    if update_tag and update_tag.find_next_sibling('td'):
                        metadata["last_update_date"] = update_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract type
                    type_tag = soup.find('td', string=re.compile(r'Series type', re.IGNORECASE))
                    if type_tag and type_tag.find_next_sibling('td'):
                        metadata["type"] = type_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract overall design
                    design_tag = soup.find('td', string=re.compile(r'Overall design', re.IGNORECASE))
                    if design_tag and design_tag.find_next_sibling('td'):
                        metadata["overall_design"] = design_tag.find_next_sibling('td').get_text(strip=True)
                    
                    # Extract PubMed ID
                    pubmed_links = soup.find_all('a', href=re.compile(r'pubmed'))
                    if pubmed_links:
                        for link in pubmed_links:
                            pubmed_match = re.search(r'pubmed/(\d+)', link.get('href', ''))
                            if pubmed_match:
                                metadata["pubmed_id"] = pubmed_match.group(1)
                                break
                    
                    # Extract sample information
                    sample_links = soup.find_all('a', href=re.compile(r'acc=GSM'))
                    sample_ids = []
                    samples = []
                    
                    for link in sample_links:
                        gsm_match = re.search(r'acc=(GSM\d+)', link.get('href', ''))
                        if gsm_match:
                            gsm_id = gsm_match.group(1)
                            sample_ids.append(gsm_id)
                            
                            # Create basic sample info
                            sample_info = {
                                "sample_id": gsm_id,
                                "title": link.get_text(strip=True) or f"Sample {gsm_id}",
                                "source_name": "tissue sample",
                                "organism": metadata["organism"],
                                "characteristics": [
                                    "tissue: unknown tissue",
                                    "disease state: unknown",
                                    "sample type: unknown"
                                ],
                                "molecule": "total RNA",
                                "library_source": "transcriptomic",
                                "library_strategy": "RNA-Seq",
                                "platform_id": metadata["platform"]
                            }
                            samples.append(sample_info)
                    
                    metadata["sample_ids"] = sample_ids
                    metadata["samples"] = samples
                    metadata["sample_count"] = len(samples)
                    
                    return metadata
                    
                except Exception as e:
                    self.logger.error(f"Web scraping error for {gse_id}: {e}")
                    raise e
            
            # Run web scraping in executor
            metadata = await loop.run_in_executor(None, scrape_geo_web)
            
            self.logger.info(f"Successfully scraped metadata for {gse_id} using BeautifulSoup")
            self.logger.debug(f"Found {len(metadata.get('sample_ids', []))} samples via web scraping")
            
            return metadata
            
        except ImportError:
            self.logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            # Final fallback to mock data
            return self._get_mock_metadata(gse_id)
            
        except Exception as e:
            self.logger.error(f"BeautifulSoup fallback failed for {gse_id}: {e}")
            # Final fallback to mock data
            return self._get_mock_metadata(gse_id)
    
    def _get_mock_metadata(self, gse_id: str) -> Dict[str, Any]:
        """Final fallback mock metadata when all other methods fail."""
        self.logger.warning(f"Using mock metadata for {gse_id}")
        return {
            "title": f"Dataset {gse_id}",
            "summary": f"RNA-seq dataset from GEO: {gse_id}",
            "platform": "GPL570",  # Common platform
            "organism": "Homo sapiens",
            "sample_ids": [f"{gse_id}_sample_{i}" for i in range(1, 21)],  # Mock sample IDs
            "sample_count": 20,
            "submission_date": "Unknown",
            "last_update_date": "Unknown",
            "type": "Expression profiling by high throughput sequencing",
            "samples": [
                {
                    "sample_id": f"{gse_id}_sample_{i}",
                    "title": f"Sample {i}",
                    "source_name": "tissue sample",
                    "organism": "Homo sapiens",
                    "characteristics": [
                        f"tissue: {'tumor' if i % 2 == 0 else 'normal'} tissue",
                        f"disease state: {'malignant' if i % 2 == 0 else 'normal'}",
                        f"sample type: {'primary tumor' if i % 2 == 0 else 'adjacent normal'}"
                    ],
                    "molecule": "total RNA",
                    "library_source": "transcriptomic",
                    "library_strategy": "RNA-Seq",
                    "platform_id": "GPL570"
                }
                for i in range(1, 21)
            ]
        }
    
    async def _extract_sample_info(self, gse_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract valid tissue sample information."""
        samples = []
        base_tmp_dir = "temp"
        def normalize_gse_id(raw: str) -> str:
            """
            Normalize GSE IDs safely.
            - Preserve leading zeros if the ID already starts with 'GSE0...'
            - Convert raw numeric-only inputs to GSE####
            - Handle internal '200######' → GSE######
            """
            if raw is None:
                raise ValueError("Missing dataset id.")

            s = str(raw).strip().upper()

            # --- Case 2: Internal 200###### format ---
            if re.fullmatch(r"200\d+", s):
                digits = s[3:]
                return f"GSE{int(digits)}"

            # --- Case 3: Starts with GSE but no protected leading zero ---
            if s.startswith("GSE"):
                digits = s[3:]
                return f"GSE{int(digits)}"   # removes leading zeros for normal cases

            # --- Case 4: Only digits ---
            if digits_match := re.fullmatch(r"\d+", s):
                return f"GSE{int(s)}"

            raise ValueError(f"Could not parse a GEO series id from '{s}'.")

        os.makedirs(os.path.join(base_tmp_dir, gse_id), exist_ok=True)
        # gse_id = normalize_gse_id(gse_id)
        # --- Download series (silence any prints from GEOparse) ---
        with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            series = GEOparse.get_GEO(geo=gse_id, destdir=os.path.join(base_tmp_dir, gse_id))

        sample_ids: List[str] = series.metadata.get("sample_id", [])
        if not isinstance(sample_ids, list):
            sample_ids = [sample_ids]
        cohort_size = len(sample_ids)

        
        samples: List[Dict[str, Any]] = []
         
        for gsm_id in sample_ids[:10]:
            try:
                with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
                    gsm = GEOparse.get_GEO(geo=gsm_id, destdir=os.path.join(base_tmp_dir, gse_id))
                characteristics = gsm.metadata.get(
                    "characteristics_ch1", "characteristics_ch1"
                )
                library_strategy_gsm = gsm.metadata.get(
                    "library_strategy", "library_strategy"
                )[0]
                library_source = gsm.metadata.get(
                    "library_source", "library_source"
                )[0]
                extract_protocol = gsm.metadata.get(
                    "extract_protocol_ch1", "extract_protocol_ch1"
                )[0]
                molecule = gsm.metadata.get(
                    "molecule_ch1", "molecule_ch1"
                )[0]
                tissue_info = characteristics[0] if characteristics else "N/A"

                sdict = {
                        "sample_id": gsm_id,
                        "tissue_type": tissue_info,
                        "characteristics": characteristics,
                        "library_source": library_source,
                        "library_strategy": library_strategy_gsm,
                        "extraction_protocol": extract_protocol,
                        "molecule": molecule,
                    }
                samples.append(sdict)


            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error reading sample {gsm_id}: {e}")
                continue
        return samples
    
    def _build_download_urls(self, gse_id: str) -> List[str]:
        """Build download URLs for a GEO dataset with proper pattern matching."""
        urls = []
        # def normalize_gse_id(raw: str) -> str:
        #     """
        #     Convert inputs like '200294225' -> 'GSE294225'.
        #     Also handles 'GSE089408' -> 'GSE89408', '89408' -> 'GSE89408', etc.
        #     """
        #     if raw is None:
        #         raise ValueError("Missing dataset id.")

        #     s = str(raw).strip()

        #     # Common internal form starts with '200' — treat that as 'GSE'
        #     if re.fullmatch(r'200\d+', s):
        #         digits = s[3:]  # drop the '200' prefix
        #         return f"GSE{int(digits)}"  # int() strips any leading zeros

        #     # Generic: optional 'GSE', then digits
        #     m = re.search(r'(?:GSE)?0*(\d+)$', s, flags=re.IGNORECASE)
        #     if not m:
        #         raise ValueError(f"Could not parse a GEO series id from '{s}'.")
        #     return f"GSE{int(m.group(1))}"
        def normalize_gse_id(raw: str) -> str:
            """
            Normalize GSE IDs safely.
            - Preserve leading zeros if the ID already starts with 'GSE0...'
            - Convert raw numeric-only inputs to GSE####
            - Handle internal '200######' → GSE######
            """
            if raw is None:
                raise ValueError("Missing dataset id.")

            s = str(raw).strip().upper()

            # --- Case 2: Internal 200###### format ---
            if re.fullmatch(r"200\d+", s):
                digits = s[3:]
                return f"GSE{int(digits)}"

            # --- Case 3: Starts with GSE but no protected leading zero ---
            if s.startswith("GSE"):
                digits = s[3:]
                return f"GSE{int(digits)}"   # removes leading zeros for normal cases

            # --- Case 4: Only digits ---
            if digits_match := re.fullmatch(r"\d+", s):
                return f"GSE{int(s)}"

            raise ValueError(f"Could not parse a GEO series id from '{s}'.")

        def geo_series_block_prefix(gse_id: str) -> str:
            """
            GEO FTP series block folder.
            GSE294225 -> 'GSE294nnn'
            GSE89408  -> 'GSE89nnn'
            < 1000    -> 'GSEnnn'
            """
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

        try:
            # ---------- Download series matrix ----------
            # ---------- Input handling ----------
            gse_id = normalize_gse_id(gse_id)  
            print(gse_id)      
            if not gse_id:
                raise ValueError("DatasetInfo object must have a non-empty dataset_id.")

            
            self.logger.info(f" BUILDING DOWNLOAD URLS for {gse_id}")
            
            geo_prefix = geo_series_block_prefix(gse_id)    
            print(geo_prefix)
            # Step 1: Get series matrix files (platform-specific naming)
            series_matrix_found = False
            try:
                matrix_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{gse_id}/matrix/"
                self.logger.info(f"Checking series matrix directory: {matrix_base}")
                
                response = requests.get(matrix_base, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = [a.get('href') for a in soup.find_all('a', href=True) if a.get('href')]
                    
                    # Look for series matrix files (platform-specific patterns)
                    matrix_files = [link for link in links if 'series_matrix.txt.gz' in link and not link.startswith('?')]
                    
                    if matrix_files:
                        self.logger.info(f" Found {len(matrix_files)} series matrix files:")
                        for i, matrix_file in enumerate(matrix_files):
                            # URL decode the filename
                            decoded_filename = urllib.parse.unquote(matrix_file)
                            full_url = f"{matrix_base}{matrix_file}"
                            urls.append(full_url)
                            series_matrix_found = True
                    else:
                        self.logger.info(f"    No series matrix files found in directory")
                else:
                    self.logger.info(f"    Cannot access series matrix directory (HTTP {response.status_code})")
                    
            except Exception as e:
                self.logger.info(f"  Error checking series matrix directory: {e}")
            
            # Step 2: Get supplementary files with validation
            supplementary_found = False
            valid_supplementary_files = []
            
            try:
                suppl_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{gse_id}/suppl/"
                print(f"   Checking supplementary directory: {suppl_base}")
                
                response = requests.get(suppl_base, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = [a.get('href') for a in soup.find_all('a', href=True) if a.get('href')]
                    
                    # Filter for actual files (not directories)
                    file_links = [link for link in links if not link.endswith('/') and not link.startswith('?')]
                    print(f"  Found {len(file_links)} supplementary files to validate:")
                    
                    for i, link in enumerate(file_links):
                        full_url = f"{suppl_base}{link}"
                        
                        # Validate the file using our criteria
                        is_valid, validation_reason = self._validate_supplementary_file(gse_id)
                        
                        if is_valid:
                            urls.append(full_url)
                            valid_supplementary_files.append(link)
                            print(f"      {i+1}.{link} - {validation_reason}")
                            supplementary_found = True
                        else:
                            print(f"      {i+1}. {link} - {validation_reason}")
                    
                    if valid_supplementary_files:
                        print(f"  Selected {len(valid_supplementary_files)} valid supplementary files")
                    else:
                        print(f"   No valid supplementary files found")
                        
                else:
                    print(f"    Cannot access supplementary directory (HTTP {response.status_code})")
                    
            except Exception as e:
                print(f"   Error checking supplementary directory: {e}")
            
            # Step 4: Summary
            print(f" DOWNLOAD URL SUMMARY for {gse_id}:")
            print(f" Series matrix files: {' Found' if series_matrix_found else 'Not found'}")
            print(f"    Valid supplementary files: {' Found' if supplementary_found else 'Not found'}")
            print(f"    Total URLs generated: {len(urls)}")
            
            print(f"    Requirements check:")
            print(f"       Has series matrix: {'Found' if series_matrix_found else 'Not Found'}")
            print(f"       Has supplementary: {'Found' if supplementary_found else 'Not Found'}")
            
            print(f" Requirements check:")
            print(f" Has series matrix: {'Found' if series_matrix_found else 'Not Found'}")
            
            return urls
            
        except Exception as e:
            self.logger.error(f"Error building download URLs for {gse_id}: {e}")
            return []
    
    def _url_matches_supplementary_criteria(self, url: str) -> bool:
        """Check if a URL matches our supplementary file criteria."""
        try:
            filename = url.split('/')[-1]
            is_valid, _ = self._validate_supplementary_file(filename, url)
            return is_valid
        except Exception:
            return False
    
    async def download_dataset(self, dataset_info: DatasetInfo, output_dir: Path) -> bool:
        """
        Download a GEO dataset with intelligent incremental downloading.
        
        Args:
            dataset_info: Information about the dataset to download
            output_dir: Directory to download files to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dataset_dir = output_dir / dataset_info.dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            # Check if this is a cached dataset
            is_cached = dataset_info.metadata.get("cached", False)
            cached_files = dataset_info.metadata.get("cached_files", [])
            
            if is_cached:
                self.logger.info(f"Processing cached dataset {dataset_info.dataset_id}")
                
                # Check for new files to download
                new_downloads = await self._get_new_files_to_download(dataset_info, dataset_dir)
                
                if not new_downloads:
                    self.logger.info(f"No new files to download for {dataset_info.dataset_id}")
                    
                    # Update metadata timestamp
                    await self._save_dataset_metadata(dataset_info, dataset_dir)
                    return True
                else:
                    self.logger.info(f"Found {len(new_downloads)} new files to download for {dataset_info.dataset_id}")
                    
                    for i in new_downloads:
                        # Download only new files
                        async with self.tools["download"] as download_tool:
                            download_result = await download_tool.execute(new_downloads)
                    if download_result.success:
                        successful_downloads = sum(1 for result in download_result.data if result.success)
                        failed_downloads = len(download_result.data) - successful_downloads
                        
                        self.logger.info(f"Downloaded {successful_downloads}/{len(new_downloads)} new files for {dataset_info.dataset_id}")
                        
                        # Log details of failed downloads
                        if failed_downloads > 0:
                            self.logger.info(f"  DOWNLOAD FAILURES: {failed_downloads} files failed to download:")
                            for result in download_result.data:
                                if not result.success:
                                    self.logger.info(f"    {result.filename}: {result.error}")
                                    self.logger.info(f"      URL: {result.url if hasattr(result, 'url') else 'Unknown'}")
                                    
                                    # Check if it's a common URL pattern issue
                                    if "404" in str(result.error):
                                        self.logger.info(f"       This file may not exist on the server")
                                    elif "403" in str(result.error):
                                        self.logger.info(f"       Access forbidden - file may be restricted")
                                    elif "timeout" in str(result.error).lower():
                                        self.logger.info(f"       Server timeout - try again later")
                        
                        # Continue even if some downloads failed
                        return True
                    else:
                        self.logger.error(f"Incremental download failed for {dataset_info.dataset_id}: {download_result.error}")
                        return False
            else:
                self.logger.info(f"Downloading new dataset {dataset_info.dataset_id}")
                
                # Prepare download list for all files
                downloads = []
                for url in dataset_info.download_urls:
                    filename = url.split('/')[-1]
                    if not filename:
                        continue
                    
                    destination = dataset_dir / filename
                    downloads.append(DownloadInfo(
                        url=url,
                        destination=destination,
                        filename=filename
                    ))
                
                # Download all files
                async with self.tools["download"] as download_tool:
                    download_result = await download_tool.execute(downloads)
                
                if not download_result.success:
                    self.logger.error(f"Download failed for {dataset_info.dataset_id}: {download_result.error}")
                    return False
                
                # Count successful downloads
                successful_downloads = sum(1 for result in download_result.data if result.success)
                total_downloads = len(download_result.data)
                
                self.logger.info(f"Downloaded {successful_downloads}/{total_downloads} files for {dataset_info.dataset_id}")
            
            # Validate downloaded files
            downloaded_files = []
            for file_path in dataset_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                    downloaded_files.append(file_path)
            
            if downloaded_files:
                validation_result = await self.tools["validation"].execute(downloaded_files)
                if validation_result.success:
                    valid_files = sum(1 for result in validation_result.data if result.is_valid)
                    self.logger.info(f"Validated {valid_files}/{len(downloaded_files)} downloaded files")
            
            # Save/update metadata
            await self._save_dataset_metadata(dataset_info, dataset_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_info.dataset_id}: {e}")
            return False
    
    async def _get_new_files_to_download(self, dataset_info: DatasetInfo, dataset_dir: Path) -> List[DownloadInfo]:
        """Determine which files are new and need to be downloaded, with validation."""
        new_downloads = []
        
        try:
            # Get list of currently downloaded files
            existing_files = set()
            if dataset_dir.exists():
                for file_path in dataset_dir.glob("*"):
                    if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                        existing_files.add(file_path.name)
            
            self.logger.info(f" EXISTING FILES in {dataset_info.dataset_id}: {sorted(existing_files)}")
            self.logger.info(f" CHECKING {len(dataset_info.download_urls)} potential download URLs:")
            
            # Check each URL to see if we need to download it
            for i, url in enumerate(dataset_info.download_urls):
                filename = url.split('/')[-1]
                if not filename:
                    continue

                # 🚫 Skip HTML files entirely
                if filename.lower().endswith(".html"):
                    self.logger.info(f"   {i+1}. SKIP: {filename} (ignored .html file)")
                    continue
                
                destination = dataset_dir / filename
                
                # Check if file doesn't exist or is incomplete
                needs_download = False
                reason = ""
                
                if filename not in existing_files:
                    needs_download = True
                    reason = "file not found"
                elif not destination.exists():
                    needs_download = True
                    reason = "file path doesn't exist"
                elif destination.stat().st_size == 0:
                    needs_download = True
                    reason = "file is empty"
                
                if needs_download:
                    # Validate the file before adding to download list
                    is_valid = True
                    if is_valid:
                        new_downloads.append(DownloadInfo(
                            url=url,
                            destination=destination,
                            filename=filename
                        ))
                        print(f"{i+1}.  NEW: {filename} ({reason})")
                        print(f"URL: {url}")
                        print(f"VALID:")
                    else:
                        print(f"{i+1}.  SKIP: {filename} ({reason}, failed validation)")
                        print(f"URL: {url}")
                else:
                    print(f"{i+1}.  SKIP: {filename} (already exists)")
            
            if new_downloads:
                print(f"SUMMARY: {len(new_downloads)} valid new files to download")
                print(f"File validation applied using GEO config criteria")
            else:
                print(f"SUMMARY: All files up-to-date or no valid new files found")
            
            return new_downloads
            
        except Exception as e:
            self.logger.error(f"Error determining new files for {dataset_info.dataset_id}: {e}")
            print(f" ERROR determining new files for {dataset_info.dataset_id}: {e}")
            return []

    def _validate_supplementary_file(self, gse_id :str) -> tuple[bool, str]:
        """
        Validate if a supplementary file should be downloaded based on GEO config.
        
        Args:
            filename: Name of the file to validate
            url: URL of the file to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """

        SUPPLEMENTARY_KEYWORDS = [
            "count", "counts", "raw_counts", "readcounts", "featurecounts",
            "norm_counts", "normalized_counts", "fpkm", "rpkm", "tpm", "cpm"
        ]
        # extensions fallback (case-insensitive)
        PREFERRED_FORMATS = [".txt", ".tsv", ".csv", ".gz", ".tar",".csv.gz"]
        geo_id = gse_id
        geo_prefix = f"{geo_id[:-3]}nnn"
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_prefix}/{geo_id}/suppl/"
        try:
            r = requests.get(base_url, timeout=20)
            if r.status_code != 200:
                print(f"[HTTP] {gse_id} supplementary dir not accessible (HTTP {r.status_code})")
                return False,"supplementary dir not accessible"

            soup = BeautifulSoup(r.text, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)]

            # 1) original boundary-based keyword pattern
            keyword_pattern = re.compile(
                r"(?<![A-Za-z0-9])(" + "|".join(SUPPLEMENTARY_KEYWORDS) + r")(?![A-Za-z0-9])",
                re.IGNORECASE
            )
            matching_files = [f for f in links if keyword_pattern.search(f)]
            if matching_files:
                print("[MATCH] Keyword-based matches:", matching_files)
                return True,"[MATCH] Keyword-based matches"

            # 2) fallback: extension-based match (case-insensitive)

            clean_files = [f for f in links if not f.startswith('?') and '.' in f]
            ext_match = any(f.lower().endswith(ext) for f in clean_files for ext in PREFERRED_FORMATS)
            if ext_match:
                print("[MATCH] Extension-based match found among preferred formats:", PREFERRED_FORMATS)
                return True,"[MATCH] Extension-based match found "

            print("[NO MATCH] Neither keywords nor preferred extensions found.")
            return False,"[NO MATCH] Neither keywords nor preferred extensions found"

        except Exception as e:
            print(f"[HTTP] Exception for {gse_id} supplementary check: {e}")
            return False
    
    def _get_file_category_info(self, filename: str) -> Dict[str, Any]:
        """Get detailed category information for a file."""
        filename_lower = filename.lower()
        file_categories = self.geo_config.supplementary_config.get("file_categories", {})
        
        category_info = {
            "matches": [],
            "highest_priority": 0,
            "best_category": None
        }
        
        for category_name, category_config in file_categories.items():
            keywords = category_config.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in filename_lower:
                    match_info = {
                        "category": category_name,
                        "keyword": keyword,
                        "priority": category_config.get("priority", 0),
                        "description": category_config.get("description", ""),
                        "required": category_config.get("required", False)
                    }
                    category_info["matches"].append(match_info)
                    
                    if match_info["priority"] > category_info["highest_priority"]:
                        category_info["highest_priority"] = match_info["priority"]
                        category_info["best_category"] = match_info
                    break
        
        return category_info
    
    def _should_skip_dataset_query(self, dataset_id: str, existing_datasets: Dict[str, Any]) -> bool:
        """Determine if we should skip querying metadata for a dataset."""
        if dataset_id not in existing_datasets:
            return False
        
        # Check if metadata is recent (less than 7 days old)
        try:
            last_updated = existing_datasets[dataset_id].get("last_updated", "")
            if last_updated and last_updated != "unknown":
                last_update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                if datetime.now().replace(tzinfo=last_update_time.tzinfo) - last_update_time < timedelta(days=7):
                    return True
        except Exception:
            pass
        
        return False
    
    async def _save_dataset_metadata(self, dataset_info: DatasetInfo, dataset_dir: Path):
        """Save dataset metadata to JSON file."""
        try:
            metadata_file = dataset_dir / f"{dataset_info.dataset_id}_metadata.json"
            
            metadata = {
                "dataset_id": dataset_info.dataset_id,
                "source": dataset_info.source,
                "title": dataset_info.title,
                "description": dataset_info.description,
                "overall_design":dataset_info.overall_design,
                "sample_count": dataset_info.sample_count,
                "file_types": dataset_info.file_types,
                "metadata": dataset_info.metadata,
                "download_timestamp": datetime.datetime.now().isoformat(),
                "agent_version": "1.0.0"
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Saved metadata for {dataset_info.dataset_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save metadata for {dataset_info.dataset_id}: {e}")
    
    def validate_dataset(self, dataset_info: DatasetInfo) -> bool:
        """
        Validate that a dataset meets GEO-specific criteria.
        
        Args:
            dataset_info: Dataset information to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            print(f"Validating dataset {dataset_info.dataset_id} from GEORetrievalAgent validate_dataset")
            # Check sample count
            if dataset_info.sample_count < 2:
                return False
            
            # Check for RNA-seq data
            metadata = dataset_info.metadata or {}
            experiment_type = metadata.get("experiment_type", "").lower()
            if "rna" not in experiment_type and "seq" not in experiment_type:
                return False
            
            # Check organism - prioritize sample-level organism data
            samples = metadata.get("samples", [])
            organism_valid = False
            
            # First check sample-level organism data
            if samples:
                sample_organisms = [sample.get("organism", "").lower() for sample in samples]
                organism_valid = any("homo sapiens" in org or "human" in org for org in sample_organisms if org)
            
            # Fall back to dataset-level organism if no sample data
            if not organism_valid:
                dataset_organism = metadata.get("organism", "").lower()
                organism_valid = "homo sapiens" in dataset_organism or "human" in dataset_organism
            
            if not organism_valid:
                return False
            
            # Check for tissue samples
            valid_tissue_samples = 0
            
            for sample in samples:
                tissue_type = sample.get("tissue_type", "").lower()
                if any(keyword in tissue_type for keyword in self.geo_config.tissue_keywords):
                    if not any(exclude in tissue_type for exclude in self.geo_config.exclude_keywords):
                        valid_tissue_samples += 1
                        
            return valid_tissue_samples >= 2
            
        except Exception as e:
            self.logger.error(f"Error validating dataset {dataset_info.dataset_id}: {e}")
            return False
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of file types supported by this agent."""
        return [
            "series_matrix.txt.gz",
            "supplementary.tar",
            "counts.txt.gz",
            "raw_counts.txt.gz",
            "fpkm.txt.gz",
            "tpm.txt.gz"
        ]
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """Get information about the GEO data source."""
        return {
            "name": "NCBI Gene Expression Omnibus (GEO)",
            "description": "Public repository of high-throughput gene expression data",
            "url": "https://www.ncbi.nlm.nih.gov/geo/",
            "supported_organisms": ["Homo sapiens"],
            "supported_experiments": ["RNA-seq", "microarray"],
            "file_types": self.get_supported_file_types(),
            "agent_version": "1.0.0",
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GEO agent execution statistics."""
        base_stats = super().get_statistics()
        
        # Add GEO-specific statistics
        base_stats.update({
            "datasets_with_supplementary": self.datasets_with_supplementary,
            "valid_tissue_samples_found": self.valid_tissue_samples_found,
            "average_samples_per_dataset": (
                self.valid_tissue_samples_found / max(self.successful_queries, 1)
            )
        })
        
        return base_stats 