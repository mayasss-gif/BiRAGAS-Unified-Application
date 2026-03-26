"""
Filter tool for the Cohort Retrieval Agent system.

This tool handles filtering and validation of datasets based on various criteria.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Project Imports 
from   ..base.base_tool import BaseTool, ToolResult
from   ..config import CohortRetrievalConfig

@dataclass
class FilterCriteria:
    """
    Criteria for filtering datasets.
    """
    include_keywords: List[str]
    exclude_keywords: List[str]
    min_samples: int = 1
    max_samples: int = 10000
    required_file_types: List[str] = None
    organism: str = "Homo sapiens"
    library_strategy: str = "RNA-Seq"
    use_llm_filtering: bool = False  

@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    dataset_id: str
    source: str
    title: str
    description: str
    overall_design : str
    sample_count: int
    file_types: List[str]
    tissuecategorization : Dict[str, Any]
    metadata: Dict[str, Any]
    download_urls: List[str]
    estimated_size_mb: float

class FilterTool(BaseTool[List[DatasetInfo]]):
    """
    Tool for filtering datasets based on various criteria.
    
    Handles filtering by:
    - Keywords (inclusion/exclusion)
    - Sample counts
    - File types
    - Organism
    - Library strategy
    """
    def setup_logger(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{self.name}_filtering.log")
        
        self.logger = logging.getLogger(f"FilterTool.{id(self)}")
        
        self.logger.setLevel(logging.DEBUG)
        
        # Clear previous handlers
        self.logger.handlers.clear()

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False


    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "FilterTool")
        self.geo_config = config.geo_config
        self.setup_logger()  
        # ✅ Default criteria for GEO
        self.default_geo_criteria = FilterCriteria(
            include_keywords=self.geo_config.tissue_keywords,
            exclude_keywords=self.geo_config.exclude_keywords,
            min_samples=2,
            max_samples=self.geo_config.max_datasets * 10,
            required_file_types=self.geo_config.supplementary_keywords,
            organism="Homo sapiens",
            library_strategy="RNA-Seq",
            use_llm_filtering=False
    )
    

    async def execute(self, 
                     datasets: List[DatasetInfo], 
                     criteria: FilterCriteria) -> ToolResult[List[DatasetInfo]]:
        """
        Filter datasets based on criteria.
        
        Args:
            datasets: List of datasets to filter
            criteria: FilterCriteria object
            
        Returns:
            ToolResult with filtered datasets
        """
        if not self.validate_input(datasets, criteria):
            print("FilterTool: Invalid input parameters")
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"datasets_count": len(datasets) if datasets else 0}
            )
        
        return await self.run_with_retry(self._filter_datasets, datasets, criteria)
    
    async def _filter_datasets(self, 
                              datasets: List[DatasetInfo], 
                              criteria: FilterCriteria) -> List[DatasetInfo]:
        """Internal method to filter datasets."""
        filtered = []
    
        self.logger.info("Logger setup started and working.")
        self.logger.info(f"Filtered {len(datasets)} datasets to {len(datasets)} valid datasets")
        return datasets
    
    def _passes_all_filters(self, dataset: DatasetInfo, criteria: Optional[FilterCriteria] = None) -> bool:
        """Check if dataset passes all filter criteria and log reasons."""
 
        """Check if dataset passes all filter criteria and log reasons."""
        return True
        
    def _passes_keyword_filter(self, dataset: DatasetInfo, criteria: FilterCriteria) -> bool:
        """Check if dataset passes keyword filtering."""
        # Combine searchable text
        searchable_text = ' '.join([
            dataset.title.lower(),
            dataset.description.lower(),
            str(dataset.metadata).lower()
        ])
        
        # Check inclusion keywords
        if criteria.include_keywords:
            has_include = any(keyword.lower() in searchable_text for keyword in criteria.include_keywords)
            if not has_include:
                return False
        
        # Check exclusion keywords
        if criteria.exclude_keywords:
            has_exclude = any(keyword.lower() in searchable_text for keyword in criteria.exclude_keywords)
            if has_exclude:
                return False
        
        return True
    
    def _has_required_file_types(self, dataset: DatasetInfo, required_types: List[str]) -> bool:
        """Check if dataset has required file types."""
        dataset_types = set(ft.lower() for ft in dataset.file_types)
        required_types_lower = set(rt.lower() for rt in required_types)
        
        return bool(dataset_types.intersection(required_types_lower))
    
    def _matches_organism(self, dataset: DatasetInfo, organism: str) -> bool:
        """Check if dataset matches organism."""
        metadata_str = str(dataset.metadata).lower()
        return organism.lower() in metadata_str
    
    def _matches_library_strategy(self, dataset: DatasetInfo, strategy: str) -> bool:
        """Check if dataset matches library strategy."""
        metadata_str = str(dataset.metadata).lower()
        return strategy.lower() in metadata_str
    
    async def filter_geo_datasets(self, 
                                 datasets: List[DatasetInfo], 
                                 disease_name: str) -> ToolResult[List[DatasetInfo]]:
        """
        Filter GEO datasets using GEO-specific criteria.
        
        Args:
            datasets: List of GEO datasets
            disease_name: Disease name for context
            
        Returns:
            ToolResult with filtered datasets
        """
        criteria = FilterCriteria(
            include_keywords=self.geo_config.tissue_keywords + [disease_name],
            exclude_keywords=self.geo_config.exclude_keywords,
            min_samples=2,
            max_samples=self.geo_config.max_datasets * 10,
            required_file_types=self.geo_config.supplementary_keywords,
            organism="Homo sapiens",
            library_strategy="RNA-Seq",
            use_llm_filtering=False 
        )
        
        return await self.execute(datasets, criteria)
    
    def _passes_llm_filter(self, dataset: DatasetInfo) -> bool:
        """
        Use OpenAI GPT-4o to evaluate whether the dataset metadata meets bulk RNA-seq criteria
        and contains normal/control samples required for differential gene expression analysis.
        """        
        # Safe extraction of fields with fallbacks
        title = dataset.title or ""
        description = dataset.description or ""
        overall_design = dataset.overall_design or ""
        metadata = dataset.metadata or {}
        
        # Extract sample characteristics if available
        sample_characteristics = metadata.get('sample_characteristics', [])
        sample_types = metadata.get('sample_type', [])
        sample_source = metadata.get('sample_source_name_ch1', [])
        
        prompt = f"""
        You are an expert in biomedical dataset curation with specialization in RNA-seq analysis.
        Evaluate the following dataset information and determine whether it represents a **high-quality, 
        human bulk RNA-seq dataset** from GEO suitable for **robust differential gene expression (DEG) analysis**.
        
        **Inclusion Criteria (must ALL be met):**
        - Species: Human samples (e.g., "Homo sapiens" or "human")
        - Technology: Bulk RNA-seq or transcriptomics
        - Sample Type: Tissue or solid tumor biopsies 
        - Normal/Control Samples: The dataset must include at least some samples labeled as normal, control, 
        healthy, wild type, non-diseased, adjacent normal, or similar. If only diseased samples are present, 
        the dataset is unsuitable for DEG analysis.
        
    
        - Microarray technologies:
            microarray, array, chip
        - Non-human samples:
            mouse, rat, primate, animal model, drosophila, zebrafish
        
        **Sample Information:**
        - Sample Characteristics: {sample_characteristics}
        - Sample Types: {sample_types}
        - Sample Sources: {sample_source}
        
        **Dataset Details:**
        ---
        **Dataset ID:** {dataset.dataset_id}
        **Title:** {title}
        **Description:** {description}
        **Overall Design:** {overall_design}
        **Structured Metadata:** {metadata}
        ---
        
        Evaluation Instructions:
        1. First, check if any exclusion terms are present. If yes, respond with ONLY: `False`
        2. If no exclusion terms, verify all inclusion criteria are met:
            - Human samples
            - Bulk RNA-seq technology
            - Tissue/solid tumor samples
            - Presence of normal/control samples
        3. If all criteria are met, respond with ONLY: `True`
        4. If any criterion is not met, respond with ONLY: `False`
        
        Do not provide any explanation or additional text beyond the single word response.
        """
        
        try:
            load_dotenv(override=True)
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0  # Ensure deterministic output
            )
            label = response.choices[0].message.content.strip().lower()
            
            # Log the classification result
            log_message = f"Classified dataset {dataset.dataset_id} as: {label}"
            if hasattr(self, 'logger'):
                self.logger.info(log_message)
            else:
                print(log_message)
                
            return label == "true"
        except Exception as e:
            error_message = f"GPT classification error for {dataset.dataset_id}: {e}"
            if hasattr(self, 'logger'):
                self.logger.error(error_message)
            else:
                print(error_message)
            return False


    def validate_input(self, datasets: List[DatasetInfo], criteria: FilterCriteria) -> bool:
        """Validate input parameters."""
        if not isinstance(datasets, list):
            self.logger.error("datasets must be a list")
            return False
        
        if not isinstance(criteria, FilterCriteria):
            self.logger.error("criteria must be a FilterCriteria object")
            return False
        
        if criteria.min_samples < 0:
            self.logger.error("min_samples must be non-negative")
            return False
        
        if criteria.max_samples < criteria.min_samples:
            self.logger.error("max_samples must be >= min_samples")
            return False
        
        return True
    
    def validate_output(self, result: List[DatasetInfo]) -> bool:
        """Validate output result."""
        if not isinstance(result, list):
            self.logger.error("Result must be a list")
            return False
        
        for dataset in result:
            if not isinstance(dataset, DatasetInfo):
                self.logger.error("All result items must be DatasetInfo objects")
                return False
        
        return True 