"""
Simplified GEO Data Processor - DRY Architecture
Production-ready, Celery-compatible bioinformatics data pipeline.

Author: Senior Bioinformatics Engineer
Python: 3.11+
Architecture: DRY, Single Responsibility, Celery-optimized
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
from decouple import config
import os

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI API key
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@dataclass
class ProcessedResult:
    """Simple result container following dataclass pattern."""
    dataframe: pd.DataFrame
    sample_columns: List[str] 
    gene_column: str
    cleaned_file_path: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class GEOProcessor:
    """
    Single responsibility: Process GEO files for DEGs analysis.
    DRY architecture with minimal redundancy.
    """
    
    def __init__(self, use_ai: bool = True, save_cleaned: bool = True):
        self.use_ai = use_ai and bool(OPENAI_API_KEY)
        self.save_cleaned = save_cleaned
        self.agent = self._init_agent() if self.use_ai else None
        
    def _init_agent(self):
        """Initialize AI agent if available."""
        try:
            from agents import Agent
            return Agent(
                name="geo_processor", 
                model="gpt-4.1-mini-2025-04-14",
                instructions="Identify gene columns in bioinformatics data."
            )
        except Exception as e:
            logger.warning(f"AI agent unavailable: {e}")
            return None
    
    def process(self, file_path: str) -> ProcessedResult:
        """
        Main processing function - single entry point.
        Replaces all redundant functions with one clean interface.
        """
        logger.info(f"Processing: {Path(file_path).name}")
        
        # Step 1: Load data with auto-detection
        df = self._load_data(file_path)
        
        # Step 2: Identify gene column (AI or heuristic)
        gene_col = self._find_gene_column(df)
        
        # Step 3: Extract sample columns (all numeric except gene)
        sample_cols = self._get_sample_columns(df, gene_col)
        
        # Step 4: Create clean DEGs-ready DataFrame
        clean_df = self._create_clean_dataframe(df, gene_col, sample_cols)
        
        # Step 5: Optionally save cleaned file
        saved_path = self._save_if_requested(clean_df, file_path)
        
        # Step 6: Compute basic stats
        stats = self._compute_stats(clean_df, sample_cols)
        
        logger.info(f"Processed: {len(clean_df)} genes × {len(sample_cols)} samples")
        
        return ProcessedResult(
            dataframe=clean_df,
            sample_columns=sample_cols,
            gene_column='Gene',
            cleaned_file_path=saved_path,
            stats=stats
        )
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Smart data loading with auto-detection."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Auto-detect format and load
        try:
            if path.suffix.lower() == '.xlsx':
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")
    
    def _find_gene_column(self, df: pd.DataFrame) -> str:
        """Find gene column using AI or heuristics."""
        if self.agent:
            return self._ai_gene_detection(df)
        return self._heuristic_gene_detection(df)
    
    def _ai_gene_detection(self, df: pd.DataFrame) -> str:
        """AI-powered gene column detection."""
        try:
            from agents import Runner
            prompt = f"Find gene column in: {list(df.columns)}\nSample: {df.head(2).to_dict()}"
            
            with Runner(self.agent) as runner:
                response = runner.run(prompt)
                # Parse AI response for gene column name
                for col in df.columns:
                    if col.lower() in response.lower():
                        return col
        except Exception as e:
            logger.warning(f"AI detection failed: {e}")
        
        return self._heuristic_gene_detection(df)
    
    def _heuristic_gene_detection(self, df: pd.DataFrame) -> str:
        """Fast heuristic gene column detection."""
        gene_patterns = ['gene', 'symbol', 'name', 'id']
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in gene_patterns):
                return col
        
        # Fallback: first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
                
        return df.columns[0]
    
    def _get_sample_columns(self, df: pd.DataFrame, gene_col: str) -> List[str]:
        """Extract sample columns (all numeric except gene-related)."""
        exclude_patterns = ['gene', 'symbol', 'id', 'name', 'desc', 'chr', 'pos']
        
        sample_cols = []
        for col in df.columns:
            if col == gene_col:
                continue
            
            # Skip gene-related columns
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue
                
            # Include if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                sample_cols.append(col)
        
        return sample_cols
    
    def _create_clean_dataframe(self, df: pd.DataFrame, gene_col: str, sample_cols: List[str]) -> pd.DataFrame:
        """Create clean DataFrame for DEGs analysis."""
        # Select and rename gene column
        clean_df = df[[gene_col] + sample_cols].copy()
        clean_df = clean_df.rename(columns={gene_col: 'Gene'})
        
        # Clean data
        initial_rows = len(clean_df)
        
        # Remove empty genes
        empty_genes = clean_df['Gene'].isna().sum() + (clean_df['Gene'].astype(str).str.strip() == '').sum()
        clean_df = clean_df.dropna(subset=['Gene'])
        clean_df = clean_df[clean_df['Gene'].astype(str).str.strip() != '']
        
        # Remove duplicates
        duplicates = len(clean_df) - clean_df['Gene'].nunique()
        clean_df = clean_df.drop_duplicates(subset=['Gene'], keep='first')
        
        # Set Gene as index
        clean_df = clean_df.set_index('Gene')
        
        # Log cleaning stats
        removed = initial_rows - len(clean_df)
        if empty_genes > 0:
            logger.info(f"Removed {empty_genes} empty gene names")
        if duplicates > 0:
            logger.info(f"Removed {duplicates} duplicate genes")
        if removed > 0:
            logger.info(f"Total cleaned: {removed} rows removed")
            
        return clean_df
    
    def _save_if_requested(self, df: pd.DataFrame, original_path: str) -> Optional[str]:
        """Save cleaned file if requested."""
        if not self.save_cleaned:
            return None
            
        path = Path(original_path)
        cleaned_path = path.parent / f"{path.stem}_cleaned_counts.csv"
        
        # Reset index to save Gene column
        df_to_save = df.reset_index()
        df_to_save.to_csv(cleaned_path, index=False)
        
        logger.info(f"Saved: {cleaned_path}")
        return str(cleaned_path)
    
    def _compute_stats(self, df: pd.DataFrame, sample_cols: List[str]) -> Dict[str, Any]:
        """Compute basic statistics."""
        numeric_data = df[sample_cols].values.flatten()
        return {
            'mean_expression': round(numeric_data.mean(), 1),
            'zero_percentage': round((numeric_data == 0).mean() * 100, 1),
            'genes': len(df),
            'samples': len(sample_cols)
        }


# =============================================================================
# SIMPLIFIED API - Single Entry Points
# =============================================================================

def process_geo_file(file_path: str, save_cleaned: bool = True) -> Tuple[pd.DataFrame, List[str], str]:
    """
    MAIN FUNCTION - Single entry point for GEO processing.
    
    Args:
        file_path: Path to GEO expression file
        save_cleaned: Whether to save cleaned file
        
    Returns:
        Tuple of (dataframe, sample_columns, gene_column_name)
    """
    processor = GEOProcessor(save_cleaned=save_cleaned)
    result = processor.process(file_path)
    
    return result.dataframe, result.sample_columns, result.gene_column


def get_cleaned_file_path(original_path: str) -> str:
    """Get path to cleaned file."""
    path = Path(original_path)
    return str(path.parent / f"{path.stem}_cleaned_counts.csv")


def load_cleaned_file(cleaned_path: str) -> Tuple[pd.DataFrame, List[str], str]:
    """Load previously cleaned file."""
    if not Path(cleaned_path).exists():
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_path}")
    
    df = pd.read_csv(cleaned_path, index_col='Gene')
    sample_columns = df.columns.tolist()
    
    logger.info(f"Loaded cleaned file: {len(df)} genes × {len(sample_columns)} samples")
    
    return df, sample_columns, 'Gene'


# =============================================================================
# CELERY-OPTIMIZED FUNCTIONS
# =============================================================================

def process_for_celery(file_path: str) -> Dict[str, Any]:
    """
    Celery-optimized function that returns serializable results.
    
    Returns:
        Dict with processing results and metadata
    """
    try:
        processor = GEOProcessor()
        result = processor.process(file_path)
        
        return {
            'status': 'success',
            'genes_count': len(result.dataframe),
            'samples_count': len(result.sample_columns),
            'sample_columns': result.sample_columns,
            'cleaned_file_path': result.cleaned_file_path,
            'stats': result.stats,
            'message': f"Processed {result.stats['genes']} genes × {result.stats['samples']} samples"
        }
        
    except Exception as e:
        logger.error(f"Celery processing failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'genes_count': 0,
            'samples_count': 0
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create test data
    import tempfile
    
    test_data = """Gene,Sample1,Sample2,Sample3,Sample4
ACTB,1000,950,1100,1050
GAPDH,800,850,900,875
TP53,200,180,220,210
TP53,210,190,230,220
INVALID,,100,200,150
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(test_data)
        test_file = f.name
    
    try:
        print("=== Simplified GEO Processor Demo ===")
        
        # Main processing function
        df, samples, gene_col = process_geo_file(test_file)
        
        print(f"Result: {len(df)} genes × {len(samples)} samples")
        print(f"Samples: {samples}")
        print(f"Gene column: {gene_col}")
        print(f"First 3 genes:\n{df.head(3)}")
        
        # Celery example
        print("\n=== Celery-Compatible Result ===")
        celery_result = process_for_celery(test_file)
        print(f"Status: {celery_result['status']}")
        print(f"Message: {celery_result['message']}")
        
    finally:
        # Cleanup
        os.unlink(test_file)
        cleaned_file = get_cleaned_file_path(test_file)
        if os.path.exists(cleaned_file):
            os.unlink(cleaned_file)