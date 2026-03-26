"""
DEPRECATED: Use preprocess_counts_simplified.py instead.

This file has been replaced with a simplified, DRY-compliant architecture.
The new version eliminates redundancy and follows Django/Celery/FastAPI best practices.

For new code, import from:
    from .preprocess_counts_simplified import process_geo_file, process_for_celery

Legacy functions are maintained for backward compatibility but will be removed.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from agents import Agent
from decouple import config
import os

# Configure OpenAI API key
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileDetectionResult(BaseModel):
    """Results from file format detection."""
    model_config = ConfigDict(extra="forbid")
    
    delimiter: str = Field(..., description="Detected delimiter character")
    header_row: int = Field(..., description="Zero-based header row index")
    encoding: str = Field(default="utf-8", description="File encoding")
    file_extension: str = Field(..., description="Original file extension")


class ColumnAnalysisRequest(BaseModel):
    """Request payload for GPT column analysis."""
    model_config = ConfigDict(extra="forbid")
    
    columns: List[str] = Field(..., description="List of column names")
    sample_data: List[Dict[str, Any]] = Field(..., description="Sample rows (≤5)")
    file_info: str = Field(..., description="File context information")


class ColumnAnalysisResponse(BaseModel):
    """Response from GPT column analysis."""
    model_config = ConfigDict(extra="forbid")
    
    gene_column: str = Field(..., description="Identified gene identifier column")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of decision")


class CleaningResult(BaseModel):
    """Final result from cleaning operation."""
    model_config = ConfigDict(extra="forbid")
    
    gene_col: str = Field(..., description="Name of gene identifier column")
    numeric_cols: List[str] = Field(..., description="List of numeric sample columns")
    df_preview: List[Dict[str, Any]] = Field(..., description="First 5 rows preview")
    status: str = Field(..., pattern="^(success|error)$", description="Operation status")
    message: str = Field(..., description="Status message or error description")
    cleaned_file_path: Optional[str] = Field(default=None, description="Path to saved cleaned DataFrame")

    @field_validator('df_preview')
    def validate_preview_size(cls, v):
        """Ensure preview doesn't exceed reasonable size."""
        if len(v) > 5:
            return v[:5]
        return v


class DataQualityAnalysis(BaseModel):
    """Analysis of data quality for DEGs readiness."""
    model_config = ConfigDict(extra="forbid")
    
    sample_columns: List[str] = Field(..., description="Valid sample columns for DEGs")
    invalid_rows: List[int] = Field(default_factory=list, description="Row indices to remove")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall data quality")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistical summary")
    recommendations: List[str] = Field(default_factory=list, description="Data cleaning recommendations")


class GeoCleanerAgent:
    """Main GEO data cleaning agent."""
    
    def __init__(self):
        """Initialize the cleaning agent."""
        # Only create agent if API key is available
        if OPENAI_API_KEY:
            try:
                self.agent = Agent(
                    name="geo_cleaner",
                    model="gpt-4.1-mini-2025-04-14",
                    instructions=self._get_analysis_instructions()
                )
                logger.info("GPT agent initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPT agent: {e}")
                self.agent = None
        else:
            logger.info("No OpenAI API key provided, GPT analysis will be disabled")
            self.agent = None
        
        self.max_payload_size = 10 * 1024 * 1024  # 10MB limit
    
    def _get_analysis_instructions(self) -> str:
        """Get instructions for GPT analysis."""
        return """
        You are an expert bioinformatics data engineer preparing GEO expression data for DEGs analysis.
        
        TASK 1 - Gene Column Identification:
        Priority: Gene_Symbol > Gene > Gene_ID > Transcript_ID
        Look for columns containing gene symbols (GAPDH, TP53, etc.) - most preferred for DEGs.
        
        TASK 2 - Sample Column Detection:
        Identify numeric columns representing samples (treatment/control groups).
        Valid sample patterns: *_FPKM, *_TPM, *_counts, Sample_*, *N, *T, Control_*, Treat_*
        Exclude: Gene_ID, descriptions, chromosome info, annotations.
        
        TASK 3 - Data Quality Assessment:
        Analyze first 10 rows for DEGs readiness:
        - Detect invalid rows (all zeros, metadata rows, headers)
        - Statistical summary (mean, variance, zero-inflation)
        - Quality score based on expression distribution
        
        Return JSON with:
        {
            "gene_column": "column_name",
            "confidence": 0.95,
            "reasoning": "brief explanation",
            "sample_columns": ["col1", "col2"],
            "invalid_rows": [0, 5],
            "quality_score": 0.85,
            "statistics": {"mean_expression": 5.2, "zero_fraction": 0.15},
            "recommendations": ["Remove rows with all zeros", "Log-transform recommended"]
        }
        """
    
    async def _detect_file_format(self, file_path: str) -> FileDetectionResult:
        """Detect file format and parameters."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = path_obj.suffix.lower()
        
        if file_ext == '.xlsx':
            return FileDetectionResult(
                delimiter=",",
                header_row=0,
                file_extension=file_ext
            )
        
        # For text files, detect delimiter
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        
        # Detect delimiter by counting occurrences
        delimiters = ['\t', ',', ';', '|']
        delimiter_counts = {d: first_line.count(d) for d in delimiters}
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        # If no clear delimiter found, default to tab
        if delimiter_counts[best_delimiter] == 0:
            best_delimiter = '\t'
        
        return FileDetectionResult(
            delimiter=best_delimiter,
            header_row=0,
            file_extension=file_ext
        )
    
    def _load_dataframe(self, file_path: str, detection_result: FileDetectionResult) -> pd.DataFrame:
        """Load DataFrame based on detection results."""
        try:
            if detection_result.file_extension == '.xlsx':
                df = pd.read_excel(file_path, header=detection_result.header_row)
            else:
                df = pd.read_csv(
                    file_path,
                    sep=detection_result.delimiter,
                    header=detection_result.header_row,
                    encoding=detection_result.encoding,
                    low_memory=False
                )
            
            # Handle unnamed first column
            if df.columns[0].startswith('Unnamed:'):
                df.columns = ['Gene'] + list(df.columns[1:])
            
            # Handle duplicate column names
            df.columns = self._make_unique_columns(df.columns.tolist())
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame: {e}")
            raise
    
    def _make_unique_columns(self, columns: List[str]) -> List[str]:
        """Make column names unique by appending numbers."""
        seen = {}
        unique_cols = []
        
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        
        return unique_cols
    
    def _prepare_sample_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare sample data for GPT analysis (string columns only)."""
        # Get string columns only
        string_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                string_cols.append(col)
        
        if not string_cols:
            # If no string columns, include first column
            string_cols = [df.columns[0]]
        
        # Sample up to 5 rows from string columns only
        sample_df = df[string_cols].head(5)
        
        # Convert to list of dicts, handling NaN values
        sample_data = []
        for _, row in sample_df.iterrows():
            row_dict = {}
            for col in string_cols:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    row_dict[col] = str(value)
            sample_data.append(row_dict)
        
        return sample_data
    
    def _check_payload_size(self, request: ColumnAnalysisRequest) -> bool:
        """Check if request payload is within size limits."""
        import json
        payload_str = json.dumps(request.model_dump())
        size_bytes = len(payload_str.encode('utf-8'))
        
        if size_bytes > self.max_payload_size:
            logger.warning(f"Payload size {size_bytes} exceeds limit {self.max_payload_size}")
            return False
        
        return True
    
    async def _analyze_columns_with_gpt(self, df: pd.DataFrame, file_path: str) -> str:
        """Use GPT to analyze and identify gene column."""
        # Check if agent is available
        if self.agent is None:
            logger.info("GPT agent not available, using fallback gene detection")
            return self._fallback_gene_detection(df)
            
        try:
            sample_data = self._prepare_sample_data(df)
            
            request = ColumnAnalysisRequest(
                columns=df.columns.tolist(),
                sample_data=sample_data,
                file_info=f"File: {Path(file_path).name}, Shape: {df.shape}"
            )
            
            # Check payload size
            if not self._check_payload_size(request):
                # Fallback to simple heuristics
                return self._fallback_gene_detection(df)
            
            # Call GPT agent using proper OpenAI Agents SDK
            try:
                from agents import Runner
                
                prompt = f"""
                Analyze this GEO gene expression data and identify the gene identifier column:
                
                Columns: {request.columns}
                Sample data: {request.sample_data}
                File info: {request.file_info}
                
                Return your analysis as JSON with gene_column, confidence, and reasoning fields.
                """
                
                # Use the OpenAI Agents SDK Runner to execute the agent
                response = await Runner.run(self.agent, prompt)
                content = response.final_output if hasattr(response, 'final_output') else str(response)
                
                logger.info(f"GPT analysis successful: {content[:100]}...")
                
            except ImportError:
                logger.warning("OpenAI Agents SDK not properly installed, falling back to heuristics")
                return self._fallback_gene_detection(df)
            except Exception as runner_error:
                logger.warning(f"Agent Runner failed: {runner_error}, falling back to heuristics")
                return self._fallback_gene_detection(df)
            
            # Parse enhanced response
            import json
            try:
                result = json.loads(content)
                if 'sample_columns' in result:  # Enhanced analysis
                    quality_analysis = DataQualityAnalysis(**result)
                    return self._process_enhanced_analysis(df, quality_analysis)
                else:  # Basic analysis
                    analysis = ColumnAnalysisResponse(**result)
                    if analysis.gene_column in df.columns:
                        logger.info(f"GPT identified gene column: {analysis.gene_column} (confidence: {analysis.confidence})")
                        return analysis.gene_column
                    else:
                        logger.warning(f"GPT suggested invalid column: {analysis.gene_column}")
                        return self._fallback_gene_detection(df)
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse GPT response: {e}")
                return self._fallback_gene_detection(df)
                
        except Exception as e:
            logger.warning(f"GPT analysis failed: {e}")
            return self._fallback_gene_detection(df)
    
    def _process_enhanced_analysis(self, df: pd.DataFrame, analysis: DataQualityAnalysis) -> str:
        """Process enhanced analysis and clean data for DEGs."""
        logger.info(f"Enhanced analysis - Quality score: {analysis.quality_score:.2f}")
        
        # Remove invalid rows
        if analysis.invalid_rows:
            df.drop(analysis.invalid_rows, inplace=True)
            logger.info(f"Removed {len(analysis.invalid_rows)} invalid rows")
        
        # Log recommendations
        for rec in analysis.recommendations:
            logger.info(f"Recommendation: {rec}")
        
        # Log statistics
        stats = analysis.statistics
        logger.info(f"Expression stats - Mean: {stats.get('mean_expression', 'N/A'):.2f}, "
                   f"Zero fraction: {stats.get('zero_fraction', 'N/A'):.2f}")
        
        return analysis.sample_columns[0] if analysis.sample_columns else self._fallback_gene_detection(df)
    
    def _fallback_gene_detection(self, df: pd.DataFrame) -> str:
        """Fallback heuristic-based gene column detection with priority for gene symbols."""
        # High priority: Gene symbol columns (most preferred)
        symbol_keywords = ['gene_symbol', 'symbol', 'gene_name', 'genesymbol']
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            for keyword in symbol_keywords:
                keyword_clean = keyword.replace('_', '').replace('-', '').replace(' ', '')
                if keyword_clean in col_lower:
                    logger.info(f"Fallback detected gene symbol column: {col}")
                    return col
        
        # Medium priority: General gene identifiers  
        gene_keywords = ['gene', 'gene_id', 'geneid', 'ensembl', 'refseq']
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            for keyword in gene_keywords:
                keyword_clean = keyword.replace('_', '').replace('-', '').replace(' ', '')
                if keyword_clean in col_lower:
                    logger.info(f"Fallback detected gene ID column: {col}")
                    return col
        
        # Lower priority: Probe/transcript identifiers
        other_keywords = ['probe', 'transcript', 'id']
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            for keyword in other_keywords:
                keyword_clean = keyword.replace('_', '').replace('-', '').replace(' ', '')
                if keyword_clean in col_lower and col_lower != 'geneid':  # Avoid generic 'id'
                    logger.info(f"Fallback detected identifier column: {col}")
                    return col
        
        # Default to first column if no clear match
        first_col = df.columns[0]
        logger.info(f"Fallback using first column as gene column: {first_col}")
        return first_col
    
    def _identify_numeric_columns(self, df: pd.DataFrame, gene_col: str) -> List[str]:
        """Identify numeric sample columns."""
        numeric_cols = []
        
        for col in df.columns:
            if col == gene_col:
                continue
                
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue
        
        return numeric_cols
    
    def _create_preview(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create preview of first 5 rows."""
        preview_df = df.head(5)
        preview_data = []
        
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = None
                else:
                    # Convert to appropriate type for JSON serialization
                    if isinstance(value, (int, float)):
                        row_dict[col] = float(value) if pd.api.types.is_float_dtype(type(value)) else int(value)
                    else:
                        row_dict[col] = str(value)
            preview_data.append(row_dict)
        
        return preview_data
    
    async def clean_geo_counts(self, file_path: str) -> Dict[str, Any]:
        """
        Intelligent GEO data processing for DEGs analysis.
        Creates clean dataframe with Gene column first, followed by sample columns only.
        """
        try:
            logger.info(f"Starting GEO file cleaning: {file_path}")
            
            # Load and detect format
            detection_result = await self._detect_file_format(file_path)
            logger.info(f"Detected format - delimiter: '{detection_result.delimiter}', header_row: {detection_result.header_row}")
            
            df = self._load_dataframe(file_path, detection_result)
            logger.info(f"Loaded DataFrame with shape: {df.shape}")
            
            if df.empty:
                return CleaningResult(gene_col="", numeric_cols=[], df_preview=[], 
                                   status="error", message="File is empty").model_dump()
            
            # AI-powered analysis for DEGs readiness
            gene_col = await self._analyze_columns_with_gpt(df, file_path)
            
            # Intelligent column filtering for DEGs
            sample_cols = self._identify_sample_columns(df, gene_col)
            
            # Create DEGs-ready dataframe: Gene column first, then samples only
            if gene_col in df.columns:
                logger.info(f"Gene column found: {gene_col}, renaming to 'Gene'...")
                # Rename gene column to standardized "Gene" for DEGs pipeline
                degs_df = df[[gene_col] + sample_cols].copy()
                degs_df = degs_df.rename(columns={gene_col: 'Gene'})
                
                logger.info(f"\n\n+++ DEGs-ready dataframe +++: {degs_df.head(3)}")

                # Statistical analysis and quality check
                stats = self._compute_statistics(degs_df, sample_cols)
                
                # Remove invalid rows for DEGs
                degs_df = self._clean_expression_data(degs_df)
                
                logger.info(f"DEGs-ready data: {degs_df.shape[0]} genes × {len(sample_cols)} samples")
                
                # Save cleaned DataFrame to file for later use
                cleaned_file_path = self._save_cleaned_dataframe(degs_df, file_path)
                
                result = CleaningResult(
                    gene_col="Gene",
                    numeric_cols=sample_cols,
                    df_preview=degs_df.head(5).to_dict('records'),
                    status="success",
                    message=f"DEGs-ready: {degs_df.shape[0]} genes × {len(sample_cols)} samples. {stats}",
                    cleaned_file_path=cleaned_file_path  # Add the cleaned file path
                )
            else:
                raise ValueError(f"Gene column '{gene_col}' not found")
                
            logger.info(f"DEGs preparation completed: {result.message}")
            return result.model_dump()
            
        except Exception as e:
            error_msg = f"Error cleaning GEO file: {str(e)}"
            logger.error(error_msg)
            
            return CleaningResult(
                gene_col="",
                numeric_cols=[],
                df_preview=[],
                status="error",
                message=error_msg
            ).model_dump()
    
    def _identify_sample_columns(self, df: pd.DataFrame, gene_col: str) -> List[str]:
        """Identify sample columns: all numeric columns except gene-related ones."""
        sample_cols = []
        
        for col in df.columns:
            if col == gene_col:
                continue
                
            # Skip string/gene-related columns (case-insensitive)
            skip_patterns = ['gene_id', 'transcript_id', 'description', 'desc', 'chr', 'strand', 'length', 'symbol']
            if any(pattern in col.lower() for pattern in skip_patterns):
                continue
                
            # Include ALL numeric columns as samples (universal approach)
            if pd.api.types.is_numeric_dtype(df[col]):
                sample_cols.append(col)
                    
        logger.info(f"Identified {len(sample_cols)} sample columns from {len(df.columns)} total (all numeric columns except gene-related)")
        return sample_cols
    
    def _compute_statistics(self, df: pd.DataFrame, sample_cols: List[str]) -> str:
        """Compute statistical summary for expression data."""
        if not sample_cols:
            return "No sample data"
            
        sample_data = df[sample_cols]
        mean_expr = sample_data.mean().mean()
        zero_frac = (sample_data == 0).sum().sum() / (sample_data.shape[0] * sample_data.shape[1])
        
        return f"Mean expr: {mean_expr:.1f}, Zero%: {zero_frac:.1%}"
    
    def _clean_expression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid rows for DEGs analysis."""
        initial_rows = len(df)
        
        # Remove rows with empty gene names
        df = df.dropna(subset=['Gene'])
        df = df[df['Gene'].str.strip() != '']
        
        # Remove duplicate genes (keep first occurrence)
        df = df.drop_duplicates(subset=['Gene'], keep='first')
        
        cleaned_rows = len(df)
        if cleaned_rows < initial_rows:
            logger.info(f"Cleaned data: removed {initial_rows - cleaned_rows} invalid rows")
            
        return df
    
    def _save_cleaned_dataframe(self, df: pd.DataFrame, original_file_path: str) -> str:
        """
        Save cleaned DataFrame to file with _cleaned_counts.csv suffix.
        
        Args:
            df: Cleaned DataFrame with Gene as first column
            original_file_path: Path to original file
            
        Returns:
            Path to saved cleaned file
        """
        from pathlib import Path
        
        original_path = Path(original_file_path)
        cleaned_filename = original_path.stem + "_cleaned_counts.csv"
        cleaned_file_path = original_path.parent / cleaned_filename
        
        # Save with Gene column as first column (not as index yet)
        df.to_csv(cleaned_file_path, index=False)
        
        logger.info(f"Saved cleaned DataFrame to: {cleaned_file_path}")
        return str(cleaned_file_path)


# Global agent instance
_geo_agent = None


def _get_agent() -> GeoCleanerAgent:
    """Get or create global agent instance."""
    global _geo_agent
    if _geo_agent is None:
        _geo_agent = GeoCleanerAgent()
    return _geo_agent


async def clean_geo_counts(file_path: str) -> Dict[str, Any]:
    """
    Async function to clean GEO counts/TPM/FPKM files.
    
    Args:
        file_path: Path to the input file (.txt, .csv, .tsv, .xlsx)
        
    Returns:
        Dict containing:
        - gene_col: str - Name of gene identifier column
        - numeric_cols: List[str] - List of numeric sample columns  
        - df_preview: List[Dict] - First 5 rows for inspection
        - status: str - "success" or "error"
        - message: str - Status message
    """
    agent = _get_agent()
    return await agent.clean_geo_counts(file_path)


def clean_geo_counts_sync(file_path: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for clean_geo_counts.
    
    Args:
        file_path: Path to the input file (.txt, .csv, .tsv, .xlsx)
        
    Returns:
        Dict containing cleaning results and metadata
    """
    def run_async():
        """Helper to run async function in sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            return asyncio.run(clean_geo_counts(file_path))
        else:
            # Running in async context, use asyncio.create_task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, clean_geo_counts(file_path))
                return future.result()
    
    return run_async()


def process_geo_file_with_cleaned_df(file_path: str) -> tuple[pd.DataFrame, List[str], str]:
    """
    Process GEO file and return the actual cleaned DataFrame from saved file.
    
    Args:
        file_path: Path to the input file (.txt, .csv, .tsv, .xlsx)
        
    Returns:
        Tuple containing:
        - df: Actual cleaned DataFrame with Gene column as index
        - sample_columns: List of sample column names (numeric columns) 
        - gene_col: Gene column name (always "Gene")
    """
    try:
        # Step 1: Run intelligent analysis (this saves the cleaned file)
        result = clean_geo_counts_sync(file_path)
        if result['status'] != 'success':
            raise ValueError(f"File processing failed: {result['message']}")
        
        # Step 2: Check if cleaned file was saved
        from pathlib import Path
        cleaned_file_path = result.get('cleaned_file_path')
        if cleaned_file_path and Path(cleaned_file_path).exists():
            # Load the already cleaned DataFrame
            logger.info(f"Loading cleaned DataFrame from: {cleaned_file_path}")
            degs_df = pd.read_csv(cleaned_file_path)
            
            # Set Gene column as index for bioinformatics standard
            degs_df = degs_df.set_index('Gene')
            
            sample_columns = result['numeric_cols']
            
            logger.info(f"Loaded cleaned file: {degs_df.shape[0]} genes × {len(sample_columns)} samples (Gene set as index)")
            
            return degs_df, sample_columns, 'Gene'
        else:
            # Fallback: reconstruct if no cleaned file was saved
            logger.warning("No cleaned file found, reconstructing DataFrame")
            detection_result = _detect_file_format_sync(file_path)
            df_original = _load_dataframe_sync(file_path, detection_result)
            
            sample_columns = result['numeric_cols']
            original_gene_col = _find_gene_column_sync(df_original)
            
            degs_df = df_original[[original_gene_col] + sample_columns].copy()
            degs_df = degs_df.rename(columns={original_gene_col: 'Gene'})
            degs_df = _clean_expression_data_sync(degs_df)
            degs_df = degs_df.set_index('Gene')
            
            logger.info(f"Reconstructed file: {degs_df.shape[0]} genes × {len(sample_columns)} samples (Gene set as index)")
            
            return degs_df, sample_columns, 'Gene'
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise ValueError(f"Failed to process file: {str(e)}")


def process_geo_file(file_path: str) -> tuple[pd.DataFrame, List[str], str]:
    """
    Main function to process GEO expression files for DEGs analysis.
    Uses the SAME cleaning logic as the intelligent pipeline to ensure consistency.
    
    Args:
        file_path: Path to the input file (.txt, .csv, .tsv, .xlsx)
        
    Returns:
        Tuple containing:
        - df: Clean pandas DataFrame with Gene column as index
        - sample_columns: List of sample column names (numeric columns)
        - gene_col: Gene column name (always "Gene")
        
    Raises:
        ValueError: If file processing fails or invalid format
    """
    # Use the new function that ensures we get the actually cleaned DataFrame
    return process_geo_file_with_cleaned_df(file_path)


def _detect_file_format_sync(file_path: str) -> FileDetectionResult:
    """Synchronous file format detection for Celery compatibility."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = path_obj.suffix.lower()
    
    # Detect delimiter for text files
    if file_extension in ['.txt', '.tsv']:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            else:
                delimiter = '\t'  # default
    elif file_extension == '.csv':
        delimiter = ','
    else:
        delimiter = '\t'  # default for xlsx and others
    
    return FileDetectionResult(
        delimiter=delimiter,
        header_row=0,
        encoding='utf-8',
        file_extension=file_extension
    )


def _load_dataframe_sync(file_path: str, detection_result: FileDetectionResult) -> pd.DataFrame:
    """Synchronous DataFrame loading for Celery compatibility."""
    if detection_result.file_extension == '.xlsx':
        return pd.read_excel(file_path, header=detection_result.header_row)
    else:
        return pd.read_csv(
            file_path,
            delimiter=detection_result.delimiter,
            header=detection_result.header_row,
            encoding=detection_result.encoding
        )


def _find_gene_column_sync(df: pd.DataFrame) -> str:
    """Synchronous gene column detection for Celery compatibility."""
    # Priority-based gene column detection
    gene_patterns = [
        ['gene_symbol', 'symbol'],
        ['gene', 'gene_name', 'genesymbol'],
        ['gene_id', 'geneid'],
        ['transcript_id', 'transcriptid']
    ]
    
    for patterns in gene_patterns:
        for col in df.columns:
            if col.lower() in patterns:
                return col
    
    # Fallback to first string column
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    
    # Final fallback to first column
    return df.columns[0]


def _clean_expression_data_sync(df: pd.DataFrame) -> pd.DataFrame:
    """Synchronous data cleaning for Celery compatibility."""
    initial_rows = len(df)
    
    # Count empty gene names (NA and empty strings)
    empty_na = df['Gene'].isna().sum()
    empty_strings = (df['Gene'].str.strip() == '').sum() if not df['Gene'].isna().all() else 0
    total_empty = empty_na + empty_strings
    
    # Remove rows with empty gene names
    df_no_empty = df.dropna(subset=['Gene'])
    df_no_empty = df_no_empty[df_no_empty['Gene'].str.strip() != '']
    
    # Report detailed statistics
    if total_empty > 0:
        logger.info(f"Found {total_empty} empty gene names ({empty_na} NA + {empty_strings} empty strings)")
    
    total_removed = initial_rows - len(df_no_empty)
    if total_removed > 0:
        logger.info(f"Cleaned data: removed {total_removed} invalid rows")
        
    return df_no_empty


# Example usage (for testing purposes)
if __name__ == "__main__":


    # Create a simple test file for demonstration
    import tempfile
    test_data = """Gene,Sample1,Sample2,Sample3
ACTB,1000,950,1100
GAPDH,800,850,900
TP53,200,180,220
TP53,210,190,230
"""
    
    # Write test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_test_counts.txt', delete=False) as f:
        f.write(test_data)
        file = f.name
    
    try:
        # Test the new main function
        print("=== Testing Main Function: process_geo_file ===")
        df, sample_columns, gene_col = process_geo_file(file)
        
        print(f"\nResults:")
        print(f"DataFrame shape: {df.shape}")
        print(f"Gene column: {gene_col}")
        print(f"Sample columns ({len(sample_columns)}): {sample_columns}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print("\n=== Legacy Test (for comparison) ===")
        # Test sync function
        result = clean_geo_counts_sync(file)
        print("Sync test result keys:", list(result.keys()))
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Clean up test file
        import os
        if os.path.exists(file):
            os.unlink(file)
        
        # Clean up any generated cleaned file
        cleaned_file = file.replace('.txt', '_cleaned_counts.csv')
        if os.path.exists(cleaned_file):
            os.unlink(cleaned_file)


# =============================================================================
# COMPATIBILITY WRAPPER - Use simplified version for new code
# =============================================================================

def process_geo_file_new(file_path: str) -> tuple:
    """
    NEW SIMPLIFIED API - Use this instead of the legacy functions above.
    """
    from .preprocess_counts_simplified import process_geo_file
    return process_geo_file(file_path)


def process_for_celery_new(file_path: str) -> dict:
    """
    CELERY-OPTIMIZED API - Use this for Celery tasks.
    """
    from .preprocess_counts_simplified import process_for_celery
    return process_for_celery(file_path)