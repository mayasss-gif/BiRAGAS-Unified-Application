"""
Download tool for the Cohort Retrieval Agent system.

This tool handles downloading files from various sources with retry logic,
progress tracking, and concurrent download capabilities.
"""

import aiohttp
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
from ftplib import FTP

# Project Imports
from  ..base.base_tool import AsyncContextTool, ToolResult
from  ..config import CohortRetrievalConfig
from  ..exceptions import DownloadError, NetworkError


@dataclass
class DownloadInfo:
    """Information about a file to download."""
    url: str
    destination: Path
    filename: str
    expected_size: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    url: str
    file_path: Path
    size_bytes: int = 0
    download_time: float = 0.0
    error: Optional[str] = None


class DownloadTool(AsyncContextTool[List[DownloadResult]]):
    """
    Tool for downloading files from various sources.
    
    Features:
    - Concurrent downloads with connection pooling
    - Progress tracking and callbacks
    - Automatic retry with exponential backoff
    - Checksum validation
    - Resume capability for interrupted downloads
    """
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "DownloadTool")
        self.network_config = config.network_config
        self.semaphore = asyncio.Semaphore(self.network_config.concurrent_downloads)
        # NEW: a separate semaphore to serialize very large archives (.tar/.tar.gz/.tgz)
        self.bigfile_semaphore = asyncio.Semaphore(1)

        self.progress_callback = None
        self.total_downloads = 0
        self.completed_downloads = 0
    
    async def create_context(self) -> aiohttp.ClientSession:
        """Create HTTP client session for downloads."""
        # timeout = aiohttp.ClientTimeout(total=self.network_config.timeout * 2)
        timeout = aiohttp.ClientTimeout(
                    total=None,          # never kill a long-running download
                    sock_connect=60,     # 60s to establish TCP/TLS
                    sock_read=1200        # 10 min between chunks (tune as needed)
                    )  # Longer timeout for downloads
        headers = {'User-Agent': self.network_config.user_agent}
        
        connector = aiohttp.TCPConnector(
            limit=self.network_config.concurrent_downloads,
            limit_per_host=self.network_config.concurrent_downloads // 2
        )
        
        return aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector
        )
    
    async def close_context(self, session: aiohttp.ClientSession):
        """Close HTTP client session."""
        await session.close()
    
    async def execute(self, 
                     downloads: List[DownloadInfo], 
                     progress_callback=None) -> ToolResult[List[DownloadResult]]:
        """
        Execute multiple downloads concurrently.
        
        Args:
            downloads: List of DownloadInfo objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            ToolResult with download results
        """
        if not self.validate_input(downloads):
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"downloads_count": len(downloads) if downloads else 0}
            )
        
        self.progress_callback = progress_callback
      
        # self.total_downloads = len(filtered)
        self.completed_downloads = 0

        # IMPORTANT: pass filtered list forward
        return await self.run_with_retry(self._execute_downloads, downloads)
    
    async def _execute_downloads(self, downloads: List[DownloadInfo]) -> List[DownloadResult]:
        """Internal method to execute downloads concurrently."""
        if not self.context:
            raise DownloadError("HTTP session not initialized")
        
        # Create tasks for concurrent downloads
        tasks = []
        for download in downloads:
            if download.filename.endswith(".html"):
                continue
            task = asyncio.create_task(self._download_single_file(download))
            tasks.append(task)
        
        # Wait for all downloads to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        download_results = []
        for result in results:
            if isinstance(result, Exception):
                download_results.append(DownloadResult(
                    success=False,
                    url="unknown",
                    file_path=Path("unknown"),
                    error=str(result)
                ))
            else:
                download_results.append(result)
        
        return download_results
    
    async def _download_single_file(self, download: DownloadInfo) -> DownloadResult:
        """Download a single file with retry logic."""
        # Detect very large archives by extension and serialize them
        is_big = download.filename.lower().endswith((".tar", ".tar.gz", ".tgz"))

        # Use the global bigfile semaphore for archives; normal semaphore otherwise
        sema = self.bigfile_semaphore if is_big else self.semaphore

        async with sema:  # Limit concurrent downloads
            start_time = time.time()
            
            try:
                # Ensure destination directory exists
                download.destination.parent.mkdir(parents=True, exist_ok=True)
                # Download the file
                size_bytes = await self._download_file_chunks(download)
                
                download_time = time.time() - start_time
                self.completed_downloads += 1
                
                # Update progress
                if self.progress_callback:
                    progress = self.completed_downloads / self.total_downloads
                    self.progress_callback(progress, f"Downloaded {download.filename}")
                
                self.logger.info(f"Downloaded {download.filename} ({size_bytes} bytes) in {download_time:.2f}s")
                
                return DownloadResult(
                    success=True,
                    url=download.url,
                    file_path=download.destination,
                    size_bytes=size_bytes,
                    download_time=download_time
                )
                
            except Exception as e:
                self.completed_downloads += 1
                error_msg = f"Download failed for {download.filename}: {e}"
                self.logger.error(error_msg)
                
                return DownloadResult(
                    success=False,
                    url=download.url,
                    file_path=download.destination,
                    error=error_msg
                )
    
    async def _download_file_chunks(self, download: DownloadInfo) -> int:
        """Download file in chunks with progress tracking."""
        session = self.context
        headers = download.headers or {}
        
        # Check if file already exists and get its size
        existing_size = 0
        if download.destination.exists():
            existing_size = download.destination.stat().st_size
            if download.expected_size and existing_size >= download.expected_size:
                self.logger.info(f"File {download.filename} already exists and appears complete")
                return existing_size
            elif existing_size > 0:
                # Check if this is a GEO download (GEO servers don't support range requests well)
                is_geo_download = 'ncbi.nlm.nih.gov' in download.url or 'ftp.ncbi.nlm.nih.gov' in download.url
                
                if is_geo_download:
                    # For GEO downloads, remove existing file and start fresh to avoid 416 errors
                    self.logger.info(f"Removing existing partial file {download.filename} for fresh GEO download")
                    download.destination.unlink()
                    existing_size = 0
                else:
                    # For non-GEO downloads, try to resume
                    headers['Range'] = f'bytes={existing_size}-'
                    self.logger.info(f"Resuming download of {download.filename} from byte {existing_size}")
        
        try:
            async with session.get(download.url, headers=headers) as response:
                # Handle 416 errors for GEO downloads by retrying without range
                if response.status == 416 and 'ncbi.nlm.nih.gov' in download.url:
                    self.logger.warning(f"Got 416 error for GEO download {download.filename}, retrying without range")
                    # Remove range header and existing file, then retry
                    headers.pop('Range', None)
                    if download.destination.exists():
                        download.destination.unlink()
                    existing_size = 0
                    
                    # Retry without range header
                    async with session.get(download.url, headers=headers) as retry_response:
                        if retry_response.status != 200:
                            raise NetworkError(
                                f"HTTP {retry_response.status}: {retry_response.reason}",
                                endpoint=download.url,
                                status_code=retry_response.status
                            )
                        # Use the retry response for the rest of the download
                        response = retry_response
                        await self._process_download_response(response, download, existing_size)
                        return download.destination.stat().st_size
                
                # For other status codes, check if they're acceptable
                expected_statuses = [200] if 'ncbi.nlm.nih.gov' in download.url else [200, 206]
                if response.status not in expected_statuses:
                    raise NetworkError(
                        f"HTTP {response.status}: {response.reason}",
                        endpoint=download.url,
                        status_code=response.status
                    )

                await self._process_download_response(response, download, existing_size)
                return download.destination.stat().st_size
                
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error downloading {download.filename}: {e}", endpoint=download.url)
        except Exception as e:
            raise DownloadError(f"Download error for {download.filename}: {e}", url=download.url)
    
    async def _process_download_response(self, response, download: DownloadInfo, existing_size: int):
        """Process the download response and save the file."""
        total_size = existing_size
        content_length = response.headers.get('content-length')
        if content_length:
            total_size += int(content_length)
        
        mode = 'ab' if existing_size > 0 else 'wb'
        async with aiofiles.open(download.destination, mode) as f:
            downloaded = existing_size
            async for chunk in response.content.iter_chunked(self.network_config.chunk_size):
                await f.write(chunk)
                downloaded += len(chunk)
    
    async def download_geo_files(self, 
                                dataset_id: str, 
                                output_dir: Path, 
                                file_urls: List[str]) -> ToolResult[List[DownloadResult]]:
        """
        Download GEO dataset files.
        
        Args:
            dataset_id: GEO dataset ID
            output_dir: Output directory
            file_urls: List of file URLs to download
            
        Returns:
            ToolResult with download results
        """
        downloads = []
       
        def _is_html_filename(name: Optional[str]) -> bool:
            n = (name or "").lower()

        for url in file_urls:
            fname = url.split("/")[-1] or f"{dataset_id}_file_{len(downloads)}"
            if _is_html_filename(fname):
                self.logger.info(f"Skipping HTML file in GEO list: {fname} ({url})")
                continue
            destination = output_dir / dataset_id / fname
            downloads.append(DownloadInfo(url=url, destination=destination, filename=fname))
        
        return await self.execute(downloads)
    
    async def download_with_ftp(self, 
                               ftp_host: str, 
                               ftp_path: str, 
                               local_path: Path,
                               filename: str) -> DownloadResult:
        """
        Download file using FTP.
        
        Args:
            ftp_host: FTP server hostname
            ftp_path: Path on FTP server
            local_path: Local path to save file
            filename: Name of the file
            
        Returns:
            DownloadResult
        """
        start_time = time.time()
        try:
            
            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to FTP
            ftp = FTP(ftp_host)
            ftp.login()
            ftp.cwd(ftp_path)
            
            # Download file
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)
            
            ftp.quit()
            
            size_bytes = local_path.stat().st_size
            download_time = time.time() - start_time
            
            self.logger.info(f"Downloaded {filename} via FTP ({size_bytes} bytes) in {download_time:.2f}s")
            
            return DownloadResult(
                success=True,
                url=f"ftp://{ftp_host}{ftp_path}/{filename}",
                file_path=local_path,
                size_bytes=size_bytes,
                download_time=download_time
            )
            
        except Exception as e:
            error_msg = f"FTP download failed for {filename}: {e}"
            self.logger.error(error_msg)
            
            return DownloadResult(
                success=False,
                url=f"ftp://{ftp_host}{ftp_path}/{filename}",
                file_path=local_path,
                error=error_msg
            )
    
    def validate_input(self, downloads: List[DownloadInfo]) -> bool:
        """Validate input parameters."""
        if not isinstance(downloads, list):
            self.logger.error("downloads must be a list")
            return False
        
        if not downloads:
            self.logger.error("downloads list cannot be empty")
            return False
        
        for download in downloads:
            if not isinstance(download, DownloadInfo):
                self.logger.error("All downloads must be DownloadInfo objects")
                return False
            
            if not download.url:
                self.logger.error("Download URL cannot be empty")
                return False
            
            if not download.destination:
                self.logger.error("Download destination cannot be empty")
                return False
        
        return True
    
    def validate_output(self, result: List[DownloadResult]) -> bool:
        """Validate output result."""
        if not isinstance(result, list):
            self.logger.error("Result must be a list")
            return False
        
            return n.endswith(".html") or n.endswith(".htm")
        for download_result in result:
            if not isinstance(download_result, DownloadResult):
                self.logger.error("All result items must be DownloadResult objects")
                return False
        
        return True 