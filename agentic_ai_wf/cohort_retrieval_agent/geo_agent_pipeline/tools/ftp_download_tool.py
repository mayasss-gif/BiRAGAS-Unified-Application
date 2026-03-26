"""
FTP Download tool for the Cohort Retrieval Agent system.

This tool handles robust FTP downloads with retry logic, async support,
and intelligent file categorization for GEO supplementary files.
"""

import asyncio
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from ftplib import FTP, error_perm, error_temp
import aiohttp
import aiofiles
from bs4 import BeautifulSoup

# Project Imports 
from   ..base.base_tool import AsyncContextTool, ToolResult
from   ..config import CohortRetrievalConfig
from   ..exceptions import DownloadError


@dataclass
class FTPFileInfo:
    """Information about an FTP file."""
    filename: str
    size: int
    category: Optional[str] = None
    priority: int = 0
    local_path: Optional[Path] = None
    download_url: str = ""
    is_preferred_format: bool = True


@dataclass
class FTPDownloadResult:
    """Result of FTP download operation."""
    dataset_id: str
    total_files: int
    downloaded_files: int
    failed_files: int
    files_by_category: Dict[str, List[FTPFileInfo]]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    download_time_seconds: float = 0.0


class FTPDownloadTool(AsyncContextTool[FTPDownloadResult]):
    """
    Tool for downloading supplementary files from GEO FTP server.
    
    Features:
    - Async HTTP and FTP support
    - Intelligent file categorization and prioritization
    - Retry logic with exponential backoff
    - Progress tracking and validation
    - Configurable download behavior
    """
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config)
        self.geo_config = config.geo_config
        self.ftp_host = self.geo_config.ftp_host
        self.download_config = self.geo_config.supplementary_config["download_behavior"]
        self.validation_rules = self.geo_config.supplementary_config["validation_rules"]
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def create_context(self) -> aiohttp.ClientSession:
        """Create HTTP session for downloads."""
        timeout = aiohttp.ClientTimeout(total=self.download_config["timeout_seconds"])
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                'User-Agent': 'CohortRetrievalAgent/1.0 (Research Use)'
            }
        )
        return self.session
    
    async def close_context(self, session: aiohttp.ClientSession):
        """Close HTTP session."""
        if session and not session.closed:
            await session.close()
        self.session = None
    
    async def execute(self, 
                     dataset_id: str,
                     output_dir: Path,
                     prefer_http: bool = True) -> ToolResult[FTPDownloadResult]:
        """
        Download supplementary files for a GEO dataset.
        
        Args:
            dataset_id: GEO dataset ID (e.g., GSE123456)
            output_dir: Directory to save files
            prefer_http: Whether to prefer HTTP over FTP
            
        Returns:
            ToolResult with download results
        """
        if not self.validate_input(dataset_id, output_dir):
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"dataset_id": dataset_id, "output_dir": str(output_dir)}
            )
        
        try:
            start_time = time.time()
            
            # Create output directory
            dataset_dir = output_dir / dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Discover available files
            self.logger.info(f"Discovering supplementary files for {dataset_id}")
            available_files = await self._discover_supplementary_files(dataset_id)
            
            if not available_files:
                self.logger.warning(f"No supplementary files found for {dataset_id}")
                return ToolResult(
                    success=True,
                    data=FTPDownloadResult(
                        dataset_id=dataset_id,
                        total_files=0,
                        downloaded_files=0,
                        failed_files=0,
                        files_by_category={},
                        warnings=[f"No supplementary files found for {dataset_id}"]
                    )
                )
            
            # Categorize and prioritize files
            categorized_files = self._categorize_files(available_files)
            
            # Select files to download based on configuration
            selected_files = self._select_files_for_download(categorized_files)
            
            if not selected_files:
                return ToolResult(
                    success=True,
                    data=FTPDownloadResult(
                        dataset_id=dataset_id,
                        total_files=len(available_files),
                        downloaded_files=0,
                        failed_files=0,
                        files_by_category=categorized_files,
                        warnings=["No files selected for download based on criteria"]
                    )
                )
            
            # Download selected files
            self.logger.info(f"Downloading {len(selected_files)} files for {dataset_id}")
            download_results = await self._download_files(
                selected_files, 
                dataset_id, 
                dataset_dir, 
                prefer_http
            )
            
            download_time = time.time() - start_time
            
            result = FTPDownloadResult(
                dataset_id=dataset_id,
                total_files=len(available_files),
                downloaded_files=download_results["success_count"],
                failed_files=download_results["failed_count"],
                files_by_category=categorized_files,
                errors=download_results["errors"],
                warnings=download_results["warnings"],
                download_time_seconds=download_time
            )
            
            self.logger.info(f"Download completed for {dataset_id}: "
                           f"{result.downloaded_files}/{result.total_files} files, "
                           f"{download_time:.2f}s")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"FTP download failed for {dataset_id}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                details={"dataset_id": dataset_id, "exception": type(e).__name__}
            )
    
    async def _discover_supplementary_files(self, dataset_id: str) -> List[str]:
        """Discover available supplementary files using HTTP."""
        try:
            geo_prefix = f"{dataset_id[:-3]}nnn"
            base_url = f"https://{self.ftp_host}/geo/series/{geo_prefix}/{dataset_id}/suppl/"
            
            self.logger.debug(f"Checking supplementary files at: {base_url}")
            
            async with self.session.get(base_url) as response:
                if response.status != 200:
                    self.logger.warning(f"Cannot access {base_url}, status: {response.status}")
                    return []
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract file links
                links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    # Skip parent directory and other non-file links
                    if href not in ['../', '../', '/'] and not href.startswith('?'):
                        links.append(href)
                
                self.logger.debug(f"Found {len(links)} potential files for {dataset_id}")
                return links
                
        except Exception as e:
            self.logger.error(f"Error discovering files for {dataset_id}: {e}")
            return []
    
    def _categorize_files(self, filenames: List[str]) -> Dict[str, List[FTPFileInfo]]:
        """Categorize files based on configuration."""
        categorized = {}
        
        for filename in filenames:
            # Check if file format is preferred
            is_preferred = self.geo_config.is_preferred_format(filename)
            
            # Get category
            category = self.geo_config.categorize_file(filename)
            
            if category:
                # Get priority from category
                category_info = self.geo_config.supplementary_config["file_categories"][category]
                priority = category_info["priority"]
                
                file_info = FTPFileInfo(
                    filename=filename,
                    size=0,  # Will be determined during download
                    category=category,
                    priority=priority,
                    is_preferred_format=is_preferred
                )
                
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(file_info)
        
        # Sort files within each category by priority and preference
        for category in categorized:
            categorized[category].sort(
                key=lambda x: (x.priority, x.is_preferred_format, x.filename),
                reverse=True
            )
        
        return categorized
    
    def _select_files_for_download(self, categorized_files: Dict[str, List[FTPFileInfo]]) -> List[FTPFileInfo]:
        """Select files for download based on configuration."""
        selected = []
        
        # Check if we have required categories
        required_categories = self.geo_config.get_required_categories()
        preferred_categories = self.validation_rules["preferred_categories"]
        
        # First, add files from required categories
        for category in required_categories:
            if category in categorized_files:
                selected.extend(categorized_files[category])
        
        # Then, add files from preferred categories
        for category in preferred_categories:
            if category in categorized_files and category not in required_categories:
                selected.extend(categorized_files[category])
        
        # Finally, add other files if we don't have enough
        if len(selected) == 0:
            # If no required/preferred files found, take any available files
            for category, files in categorized_files.items():
                if category not in required_categories and category not in preferred_categories:
                    selected.extend(files)
        
        # Filter by format preference
        preferred_files = [f for f in selected if f.is_preferred_format]
        if preferred_files:
            selected = preferred_files
        
        # Limit number of files to avoid overwhelming downloads
        max_files = 10  # Configurable limit
        if len(selected) > max_files:
            selected = selected[:max_files]
        
        return selected
    
    async def _download_files(self, 
                            files: List[FTPFileInfo], 
                            dataset_id: str,
                            output_dir: Path,
                            prefer_http: bool = True) -> Dict[str, Any]:
        """Download files with retry logic."""
        success_count = 0
        failed_count = 0
        errors = []
        warnings = []
        
        # Create semaphore for concurrent downloads
        max_concurrent = self.download_config["max_concurrent_downloads"]
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create download tasks
        tasks = []
        for file_info in files:
            task = self._download_single_file(
                file_info, dataset_id, output_dir, prefer_http, semaphore
            )
            tasks.append(task)
        
        # Execute downloads concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                error_msg = f"Failed to download {files[i].filename}: {result}"
                errors.append(error_msg)
                self.logger.error(error_msg)
            elif result:
                success_count += 1
                files[i].local_path = result
                self.logger.debug(f"Successfully downloaded {files[i].filename}")
            else:
                failed_count += 1
                warning_msg = f"Download skipped for {files[i].filename}"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)
        
        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _download_single_file(self, 
                                  file_info: FTPFileInfo,
                                  dataset_id: str,
                                  output_dir: Path,
                                  prefer_http: bool,
                                  semaphore: asyncio.Semaphore) -> Optional[Path]:
        """Download a single file with retry logic."""
        async with semaphore:
            geo_prefix = f"{dataset_id[:-3]}nnn"
            
            # Try HTTP first if preferred
            if prefer_http:
                http_url = f"https://{self.ftp_host}/geo/series/{geo_prefix}/{dataset_id}/suppl/{file_info.filename}"
                result = await self._download_via_http(http_url, file_info, output_dir)
                if result:
                    return result
            
            # Fallback to FTP
            return await self._download_via_ftp(dataset_id, file_info, output_dir)
    
    async def _download_via_http(self, 
                               url: str, 
                               file_info: FTPFileInfo, 
                               output_dir: Path) -> Optional[Path]:
        """Download file via HTTP with retry logic."""
        max_retries = self.download_config["retry_attempts"]
        retry_delay = self.download_config["retry_delay_seconds"]
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"HTTP download attempt {attempt + 1}/{max_retries}: {file_info.filename}")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        # Check file size
                        content_length = response.headers.get('content-length')
                        if content_length:
                            file_size = int(content_length)
                            if not self._validate_file_size(file_size):
                                self.logger.warning(f"File size validation failed for {file_info.filename}: {file_size} bytes")
                                return None
                        
                        # Download file
                        local_path = output_dir / file_info.filename
                        async with aiofiles.open(local_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        self.logger.debug(f"HTTP download successful: {file_info.filename}")
                        return local_path
                    
                    elif response.status == 404:
                        self.logger.warning(f"File not found (404): {file_info.filename}")
                        return None
                    
                    else:
                        self.logger.warning(f"HTTP error {response.status} for {file_info.filename}")
                        
            except Exception as e:
                self.logger.warning(f"HTTP download error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        return None
    
    async def _download_via_ftp(self, 
                              dataset_id: str, 
                              file_info: FTPFileInfo, 
                              output_dir: Path) -> Optional[Path]:
        """Download file via FTP (synchronous, run in executor)."""
        loop = asyncio.get_event_loop()
        
        def ftp_download_sync():
            """Synchronous FTP download function."""
            max_retries = self.download_config["retry_attempts"]
            retry_delay = self.download_config["retry_delay_seconds"]
            
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"FTP download attempt {attempt + 1}/{max_retries}: {file_info.filename}")
                    
                    ftp = FTP(self.ftp_host)
                    ftp.login()
                    
                    geo_prefix = f"{dataset_id[:-3]}nnn"
                    ftp.cwd(f"/geo/series/{geo_prefix}/{dataset_id}/suppl/")
                    
                    # Check if file exists
                    files = ftp.nlst()
                    if file_info.filename not in files:
                        self.logger.warning(f"File not found on FTP: {file_info.filename}")
                        ftp.quit()
                        return None
                    
                    # Download file
                    local_path = output_dir / file_info.filename
                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {file_info.filename}", f.write)
                    
                    ftp.quit()
                    self.logger.debug(f"FTP download successful: {file_info.filename}")
                    return local_path
                    
                except (error_perm, error_temp) as e:
                    self.logger.warning(f"FTP error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
                except Exception as e:
                    self.logger.error(f"Unexpected FTP error: {e}")
                    break
            
            return None
        
        try:
            result = await loop.run_in_executor(None, ftp_download_sync)
            return result
        except Exception as e:
            self.logger.error(f"FTP download executor error: {e}")
            return None
    
    def _validate_file_size(self, size_bytes: int) -> bool:
        """Validate file size against configuration rules."""
        min_size = self.validation_rules["min_file_size_bytes"]
        max_size = self.validation_rules["max_file_size_mb"] * 1024 * 1024
        
        return min_size <= size_bytes <= max_size
    
    def validate_input(self, dataset_id: str, output_dir: Path) -> bool:
        """Validate input parameters."""
        if not dataset_id or not dataset_id.startswith("GSE"):
            self.logger.error("Invalid dataset ID format")
            return False
        
        if not output_dir:
            self.logger.error("Output directory not specified")
            return False
        
        return True
    
    def validate_output(self, result: FTPDownloadResult) -> bool:
        """Validate output result."""
        return isinstance(result, FTPDownloadResult) and result.dataset_id
    
    async def check_supplementary_files_availability(self, dataset_id: str) -> ToolResult[Dict[str, Any]]:
        """Check if supplementary files are available for a dataset."""
        try:
            files = await self._discover_supplementary_files(dataset_id)
            categorized = self._categorize_files(files)
            
            # Check if we have required categories
            required_categories = self.geo_config.get_required_categories()
            has_required = all(cat in categorized for cat in required_categories)
            
            # Check if we have preferred categories
            preferred_categories = self.validation_rules["preferred_categories"]
            has_preferred = any(cat in categorized for cat in preferred_categories)
            
            result = {
                "dataset_id": dataset_id,
                "total_files": len(files),
                "files_by_category": {cat: len(files) for cat, files in categorized.items()},
                "has_required_categories": has_required,
                "has_preferred_categories": has_preferred,
                "recommended_for_download": has_required or has_preferred or len(files) > 0
            }
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to check supplementary files: {e}",
                details={"dataset_id": dataset_id}
            ) 