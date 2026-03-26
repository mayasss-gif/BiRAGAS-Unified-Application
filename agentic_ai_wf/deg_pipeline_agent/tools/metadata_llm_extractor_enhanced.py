"""
Enhanced GEO Metadata Parser for Differential Gene Expression Analysis

This module provides functionality to parse Gene Expression Omnibus (GEO) 
metadata using OpenAI's API and directly generate metadata DataFrames with 
proper sample-to-group mappings for differential gene expression analysis.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import pandas as pd
import openai
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GEOParsingResult:
    """Enhanced data class to hold GEO parsing results with direct DataFrame output"""
    metadata_df: pd.DataFrame
    group_column: str
    group_values: List[str]
    metadata_source_line: str
    sample_mappings: Dict[str, str]
    error: Optional[str] = None
    raw_response: Optional[str] = None


class EnhancedGEOMetadataParser:
    """
    Enhanced GEO metadata parser that directly generates metadata DataFrames
    with proper sample-to-group mappings using OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = OPENAI_API_KEY, model: str = "gpt-4.1-mini-2025-04-14"):
        """
        Initialize the enhanced GEO metadata parser.
        
        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY environment variable
            model: OpenAI model to use for parsing (using gpt-4.1-mini-2025-04-14 for better reliability)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def _create_enhanced_parsing_prompt(self, geo_text: str, sample_ids: List[str], disease: str = "") -> str:
        """
        Create an enhanced structured prompt that includes sample IDs for direct mapping.
        
        Args:
            geo_text: Raw GEO metadata text
            sample_ids: List of sample identifiers to be mapped
            disease: Disease or condition context for better parsing
            
        Returns:
            Formatted prompt string with sample context
        """
        sample_ids_str = ", ".join(sample_ids)
        
        return f"""
            You are a specialized bioinformatics agent for transcriptome analysis. Your task is to analyze GEO metadata and create a complete sample-to-group mapping for differential gene expression (DEG) analysis.

                ## Context Information
                - Disease/Condition: {disease}
                - Sample IDs to map: {sample_ids_str}
                - Total samples: {len(sample_ids)}

                ## Input GEO Metadata:
                {geo_text.strip()}

                ## Your Tasks:
                1. **Understand the Study Design**: Read the series summary and overall design to understand the experimental setup.

                2. **Identify Group-Defining Characteristic**: Find the most informative `!Sample_characteristics_ch1` line that captures biologically meaningful group distinctions. Look for:
                - Treatment vs control conditions
                - Disease vs healthy states
                - Genotype differences (e.g., wild-type vs mutant)
                - Relevant comparisons tied to {disease}
                - Dose or time-dependent variations (only if binary)

                3. **Select Two Groups Only**: From the chosen characteristic line, select exactly **2 distinct groups**:
                - One **Control Group**: Should include keywords such as "control", "ctrl", "untreated", "vehicle", "wild type", "WT", "normal", "DMSO"
                - One **Experimental Group**: Should indicate disease, treatment, dose, time point, or experimental variation

                4. **Strict Sample Filtering**:
                - Create a mapping **only for the samples that belong to these two selected groups**
                - **Exclude** any sample ID whose group does not match exactly one of the two selected values
                - Maintain order and alignment based on the metadata structure

                5. **Mapping Sample IDs to Groups**:
                - Use the sample IDs provided in {sample_ids_str}
                - Map each sample to its corresponding group using metadata alignment
                - Do not guess or include partial matches—only return matches that align **strictly** with the two selected groups

                ## Critical Output Requirements:
                - Output **exactly one JSON object**, no additional explanation
                - Use **only** the two selected group values for mapping
                - Ensure the **control group** is listed **first** in the `group_values` array
                - Include `group_column`, `group_values`, `metadata_source_line`, and `sample_mappings` as top-level keys
                - Exclude all samples that are not members of the selected two groups

                ## Output Format (JSON only):
                {{
                    "group_column": "characteristic_name_without_colon",
                    "group_values": ["control_group_value", "experimental_group_value"],
                    "metadata_source_line": "full_original_line_used",
                    "sample_mappings": {{
                        "sample_id_1": "group_value",
                        "sample_id_2": "group_value",
                        ...
                    }}
                }}

                ## Example Output Structure:
                {{
                    "group_column": "treatment",
                    "group_values": ["DMSO", "20μM LY"],
                    "metadata_source_line": "!Sample_characteristics_ch1\t\"treatment: 20μM LY\"\t\"treatment: 20μM LY\"\t\"treatment: DMSO\"\t\"treatment: DMSO\"",
                    "sample_mappings": {{
                        "LY-1": "20μM LY",
                        "LY-2": "20μM LY",
                        "NC-1": "DMSO",
                        "NC-2": "DMSO"
                    }}
                }}

                Return only strict JSON. No explanations. No markdown. No YAML. No formatting.

            """
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean up JSON response by removing markdown formatting and extra content.
        
        Args:
            response_text: Raw response from OpenAI API
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        response_text = re.sub(r'```(?:json)?\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        
        # Extract JSON object (find first { to last })
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx + 1]
        
        return response_text.strip()
    
    def parse_geo_metadata_with_samples(
        self, 
        geo_text: str, 
        sample_ids: List[str], 
        disease: str = ""
    ) -> GEOParsingResult:
        """
        Parse GEO metadata and directly create sample-to-group mappings.
        
        Args:
            geo_text: Raw GEO metadata text containing series info and sample characteristics
            sample_ids: List of sample identifiers to be mapped to groups
            disease: Disease or condition context to improve parsing accuracy
            
        Returns:
            GEOParsingResult object containing parsed information and ready-to-use DataFrame
        """
        try:
            prompt = self._create_enhanced_parsing_prompt(geo_text, sample_ids, disease)
            
            logger.info(f"Parsing GEO metadata for {len(sample_ids)} samples...")
            
            # Call OpenAI API with enhanced parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert bioinformatics analyst specializing in GEO metadata parsing for DEG analysis. Return only valid JSON with complete sample mappings."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum consistency
                max_tokens=2000,  # Increased for larger sample sets
                response_format={"type": "json_object"}  # Enforce JSON response
            )
            
            # Extract and clean response
            response_text = response.choices[0].message.content
            cleaned_response = self._clean_json_response(response_text)
            
            logger.info("Raw LLM response received, parsing JSON...")
            
            # Parse JSON
            result_dict = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["group_column", "group_values", "metadata_source_line", "sample_mappings"]
            for field in required_fields:
                if field not in result_dict:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate sample mappings completeness
            sample_mappings = result_dict["sample_mappings"]
            if len(sample_mappings) != len(sample_ids):
                logger.warning(f"Sample mapping count mismatch: expected {len(sample_ids)}, got {len(sample_mappings)}")
            
            # Validate all sample IDs are mapped
            missing_samples = set(sample_ids) - set(sample_mappings.keys())
            if missing_samples:
                logger.warning(f"Missing sample mappings for: {missing_samples}")
            
            # Create metadata DataFrame directly
            metadata_rows = []
            for sample_id in sample_ids:
                if sample_id in sample_mappings:
                    group_value = sample_mappings[sample_id]
                    # Only include samples that belong to one of the two selected groups
                    if group_value in result_dict["group_values"]:
                        metadata_rows.append({
                            "sample": sample_id,
                            result_dict["group_column"]: group_value
                        })
                    else:
                        logger.warning(f"Sample {sample_id} has group '{group_value}' not in selected groups {result_dict['group_values']}")
                else:
                    logger.warning(f"Sample {sample_id} not found in LLM mappings")
            
            if not metadata_rows:
                raise ValueError("No valid sample-group mappings could be created")
            
            metadata_df = pd.DataFrame(metadata_rows)
            
            logger.info(f"Successfully created metadata DataFrame with {len(metadata_df)} samples")
            logger.info(f"Groups: {result_dict['group_values']}")
            logger.info(f"Group distribution: {metadata_df[result_dict['group_column']].value_counts().to_dict()}")
            
            return GEOParsingResult(
                metadata_df=metadata_df,
                group_column=result_dict["group_column"],
                group_values=result_dict["group_values"],
                metadata_source_line=result_dict["metadata_source_line"],
                sample_mappings=sample_mappings,
                raw_response=response_text
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return GEOParsingResult(
                metadata_df=pd.DataFrame(),
                group_column="",
                group_values=[],
                metadata_source_line="",
                sample_mappings={},
                error=f"JSON parsing failed: {str(e)}",
                raw_response=response_text if 'response_text' in locals() else None
            )
        except Exception as e:
            logger.error(f"API call or processing failed: {e}")
            return GEOParsingResult(
                metadata_df=pd.DataFrame(),
                group_column="",
                group_values=[],
                metadata_source_line="",
                sample_mappings={},
                error=f"Processing failed: {str(e)}"
            )
    
    def batch_parse_datasets(
        self, 
        geo_datasets: List[str], 
        sample_ids_list: List[List[str]], 
        disease: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple GEO datasets with their corresponding sample IDs.
        
        Args:
            geo_datasets: List of GEO metadata text strings
            sample_ids_list: List of sample ID lists corresponding to each dataset
            disease: Disease or condition context
            
        Returns:
            List of dictionaries containing dataset index and parsing results
        """
        if len(geo_datasets) != len(sample_ids_list):
            raise ValueError("Number of datasets must match number of sample ID lists")
        
        results = []
        total_datasets = len(geo_datasets)
        
        for i, (geo_text, sample_ids) in enumerate(zip(geo_datasets, sample_ids_list)):
            logger.info(f"Processing dataset {i+1}/{total_datasets} with {len(sample_ids)} samples...")
            result = self.parse_geo_metadata_with_samples(geo_text, sample_ids, disease)
            results.append({
                "dataset_index": i,
                "result": result,
                "success": result.error is None
            })
        
        return results


def parse_geo_and_create_metadata_enhanced(
    geo_text: str,
    sample_ids: List[str],
    disease: str = "",
    api_key: Optional[str] = OPENAI_API_KEY,
    model: str = "gpt-4.1-mini-2025-04-14"
) -> pd.DataFrame:
    """
    Main enhanced function to parse GEO metadata and create a metadata DataFrame directly.
    
    This is the primary interface function that handles everything in one LLM call.
    
    Args:
        geo_text: Raw GEO metadata text
        sample_ids: List of sample identifiers to be mapped
        disease: Disease or condition context for better parsing
        api_key: OpenAI API key (optional if set in environment)
        model: OpenAI model to use
        
    Returns:
        DataFrame with sample metadata suitable for DEG analysis
        
    Raises:
        ValueError: If parsing fails or data is inconsistent
    """
    
    logger.info(f"🧬 Starting enhanced GEO metadata parsing for {len(sample_ids)} samples...")
    
    # Initialize enhanced parser
    parser = EnhancedGEOMetadataParser(api_key=api_key, model=model)
    
    # Parse GEO metadata with direct DataFrame generation
    parsing_result = parser.parse_geo_metadata_with_samples(geo_text, sample_ids, disease)
    
    if parsing_result.error:
        logger.error(f"❌ Parsing failed: {parsing_result.error}")
        if parsing_result.raw_response:
            logger.error(f"Raw response: {parsing_result.raw_response}")
        raise ValueError(f"GEO metadata parsing failed: {parsing_result.error}")
    
    logger.info(f"✅ Successfully parsed metadata:")
    logger.info(f"   - Group column: {parsing_result.group_column}")
    logger.info(f"   - Groups: {parsing_result.group_values}")
    logger.info(f"   - Total samples mapped: {len(parsing_result.metadata_df)}")
    
    return parsing_result.metadata_df