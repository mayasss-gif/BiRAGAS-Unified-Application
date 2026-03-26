"""
GEO Metadata Parser for Differential Gene Expression Analysis

This module provides functionality to parse Gene Expression Omnibus (GEO) 
metadata using OpenAI's API and generate metadata DataFrames suitable for 
differential gene expression analysis.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import openai


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GEOParsingResult:
    """Data class to hold GEO parsing results"""
    group_column: str
    group_values: List[str]
    metadata_source_line: str
    error: Optional[str] = None
    raw_response: Optional[str] = None


class GEOMetadataParser:
    """
    A class for parsing GEO metadata using OpenAI API to extract groups 
    suitable for differential gene expression analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the GEO metadata parser.
        
        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY environment variable
            model: OpenAI model to use for parsing
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def _create_parsing_prompt(self, geo_text: str, disease: str = "") -> str:
        """
        Create a structured prompt for the OpenAI API.
        
        Args:
            geo_text: Raw GEO metadata text
            disease: Disease or condition context for better parsing
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are a Python agent specialized in bioinformatics and transcriptome analysis. Your task is to analyze metadata lines from a gene expression study, and detect if any line contains **two or more distinct group values**, which are suitable for differential gene expression (DEG) analysis.

## Input Format
The input includes:
- !Series_summary: provides study context
- !Series_overall_design: describes experimental design
- Multiple !Sample_characteristics_ch1 lines: tab-separated metadata

## Disease/Condition Context: {disease}

## Your Tasks
1. Read the series summary and overall design to understand the study purpose.
2. Parse each !Sample_characteristics_ch1 line.
3. For each line:
   - Extract the **label** (e.g., `treatment`, `genotype`, `cell type`) and the corresponding values.
   - Clean each value by removing the label prefix (e.g., `treatment:`).
4. Identify lines with **at least 2 distinct group values**.
5. Among such lines, **choose the best line for DEG analysis**:
   - Prefer biologically meaningful comparisons related to `{disease}`.
   - Ignore irrelevant labels such as `age`, `sex`, `batch`, etc.
   - Prioritize labels like `treatment`, `genotype`, `condition`, `disease`, `knockdown`, etc.

## Group Selection Rules
- From the selected line, choose **exactly 2 distinct group values** to represent two experimental groups.
- If more than two values exist, pick the **two most relevant groups** based on biological context and study design.
- **Ensure the first group value is the control or baseline**, and the second is the experimental/disease group.

### Consider as control group if it contains (case-insensitive):
- `control`, `ctrl`, `untreated`, `vehicle`, `wild type`, `WT`, `normal`

### Consider as disease/experimental if it contains:
- the disease name (e.g., `{disease}`), a dosage (e.g., `10 mM`, `treated`, `stimulated`), or a genetic modification

## Output Format (JSON only)
Return a dictionary with:
- `"group_column"`: the characteristic label (e.g., `"treatment"`)
- `"group_values"`: a list with **[control_group, disease_group]** in that order
- `"metadata_source_line"`: the full original `!Sample_characteristics_ch1` line used

## Input:
{geo_text.strip()}

## Output (JSON only):
{{
"group_column": "the_characteristic_name",
"group_values": ["control_group_value", "experimental_group_value"],
"metadata_source_line": "the_exact_line_from_metadata"
}}

Return **only valid JSON**, no extra explanations.
"""
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean up JSON response by removing markdown formatting.
        
        Args:
            response_text: Raw response from OpenAI API
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        return response_text.strip()
    
    def parse_geo_metadata(self, geo_text: str, disease: str = "") -> GEOParsingResult:
        """
        Parse GEO metadata using OpenAI API to extract exactly two groups for DEG analysis.
        
        Args:
            geo_text: Raw GEO metadata text containing series info and sample characteristics
            disease: Disease or condition context to improve parsing accuracy
            
        Returns:
            GEOParsingResult object containing parsed information or error details
        """
        try:
            prompt = self._create_parsing_prompt(geo_text, disease)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a bioinformatics expert who analyzes GEO metadata. Return only valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=1000
            )
            
            # Extract and clean response
            response_text = response.choices[0].message.content
            cleaned_response = self._clean_json_response(response_text)
            
            # Parse JSON
            result_dict = json.loads(cleaned_response)
            
            return GEOParsingResult(
                group_column=result_dict["group_column"],
                group_values=result_dict["group_values"],
                metadata_source_line=result_dict["metadata_source_line"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return GEOParsingResult(
                group_column="",
                group_values=[],
                metadata_source_line="",
                error=f"Failed to parse JSON response: {str(e)}",
                raw_response=response_text if 'response_text' in locals() else None
            )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return GEOParsingResult(
                group_column="",
                group_values=[],
                metadata_source_line="",
                error=f"API call failed: {str(e)}"
            )
    
    def batch_parse_datasets(self, geo_datasets: List[str], disease: str = "") -> List[Dict[str, Any]]:
        """
        Parse multiple GEO datasets at once.
        
        Args:
            geo_datasets: List of GEO metadata text strings
            disease: Disease or condition context
            
        Returns:
            List of dictionaries containing dataset index and parsing results
        """
        results = []
        total_datasets = len(geo_datasets)
        
        for i, geo_text in enumerate(geo_datasets):
            logger.info(f"Processing dataset {i+1}/{total_datasets}...")
            result = self.parse_geo_metadata(geo_text, disease)
            results.append({
                "dataset_index": i,
                "result": result
            })
        
        return results


def extract_group_assignments(metadata_line: str) -> List[str]:
    """
    Extract group assignments from a metadata source line.
    
    Args:
        metadata_line: The !Sample_characteristics_ch1 line from GEO metadata
        
    Returns:
        List of group assignment values
    """
    # Split by tabs and ignore the first column (the !Sample_characteristics_ch1 label)
    fields = metadata_line.strip().split('\t')[1:]
    
    # Extract values after the colon, removing quotes
    group_assignments = []
    for field in fields:
        # Remove quotes and split by colon to get the value
        cleaned_field = field.strip('"')
        if ':' in cleaned_field:
            value = cleaned_field.split(':', 1)[1].strip()
            group_assignments.append(value)
    
    return group_assignments


class SampleGroupMapper:
    """
    LLM-based intelligent mapper for associating sample IDs with experimental groups.
    Handles various scenarios and naming conventions automatically.
    """
    
    def __init__(self, client: openai.OpenAI, model: str = "gpt-4o"):
        """
        Initialize the sample group mapper.
        
        Args:
            client: OpenAI client instance
            model: Model to use for mapping
        """
        self.client = client
        self.model = model
    
    def _create_mapping_prompt(
        self,
        sample_ids: List[str],
        parsing_result: GEOParsingResult,
        geo_text: str,
        additional_context: str = ""
    ) -> str:
        """
        Create a prompt for LLM-based sample-to-group mapping.
        
        Args:
            sample_ids: List of sample identifiers
            parsing_result: Result from GEO metadata parsing
            geo_text: Original GEO metadata text for context
            additional_context: Any additional context provided by user
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are an expert bioinformatics analyst specializing in gene expression data organization. Your task is to intelligently map sample IDs to experimental groups for differential gene expression analysis.

## Context Information
**Original GEO Metadata:**
{geo_text}

**Identified Group Information:**
- Group Column: {parsing_result.group_column}
- Target Groups: {parsing_result.group_values}
- Metadata Source Line: {parsing_result.metadata_source_line}

**Additional Context:** {additional_context}

## Sample IDs to Map:
{sample_ids}

## Your Task
Analyze the sample IDs and available metadata to create intelligent mappings. Consider:

1. **Sample ID Patterns**: Look for naming conventions, abbreviations, or codes that indicate group membership
2. **Sequential Ordering**: Sometimes samples are ordered by group (e.g., first N samples = group1, next M samples = group2)
3. **Naming Conventions**: Common patterns like:
   - ctrl/control vs treated/drug
   - wt/wild vs ko/knockout vs mut/mutant
   - before/after, pre/post
   - time points (0h, 6h, 24h)
   - doses (low, high, 10mg, 100mg)
   - cell types or conditions in the names

4. **Metadata Alignment**: Use the metadata source line to understand the expected pattern
5. **Replication Patterns**: Identify biological/technical replicates (usually numbered)

## Mapping Strategies
Try these approaches in order:
1. **Direct Pattern Matching**: Sample names contain group identifiers
2. **Position-Based**: Use metadata line order if sample IDs are ordered consistently
3. **Intelligent Inference**: Use biological knowledge and naming conventions
4. **Replication Analysis**: Group samples that appear to be replicates

## Output Format
Return a JSON object with:
- `"mappings"`: List of [sample_id, group_value] pairs for samples that belong to the target groups
- `"mapping_strategy"`: String explaining how you determined the mappings
- `"confidence"`: "high", "medium", or "low" based on certainty
- `"excluded_samples"`: List of sample IDs that don't belong to either target group (if any)
- `"notes"`: Any important observations or assumptions made

## Important Rules
- Only include samples that clearly belong to one of the two target groups: {parsing_result.group_values}
- If a sample's group assignment is unclear, exclude it rather than guess
- Ensure balanced representation if possible (similar number of replicates per group)
- Consider biological replicates vs technical replicates

## Sample IDs:
{json.dumps(sample_ids, indent=2)}

## Expected Groups:
Control Group: "{parsing_result.group_values[0] if parsing_result.group_values else 'N/A'}"
Experimental Group: "{parsing_result.group_values[1] if len(parsing_result.group_values) > 1 else 'N/A'}"

Return only valid JSON:
{{
  "mappings": [["sample_id1", "group1"], ["sample_id2", "group2"], ...],
  "mapping_strategy": "explanation of mapping approach",
  "confidence": "high|medium|low",
  "excluded_samples": ["sample_id3", ...],
  "notes": "additional observations"
}}
"""
    
    def map_samples_to_groups(
        self,
        sample_ids: List[str],
        parsing_result: GEOParsingResult,
        geo_text: str,
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Use LLM to intelligently map sample IDs to experimental groups.
        
        Args:
            sample_ids: List of sample identifiers
            parsing_result: Result from GEO metadata parsing
            geo_text: Original GEO metadata text
            additional_context: Additional context for mapping
            
        Returns:
            Dictionary containing mapping results and metadata
        """
        try:
            prompt = self._create_mapping_prompt(
                sample_ids, parsing_result, geo_text, additional_context
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a bioinformatics expert who maps sample IDs to experimental groups. Return only valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            cleaned_response = re.sub(r'```json\s*', '', response_text)
            cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
            return json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"LLM mapping failed: {e}")
            return {
                "error": f"LLM mapping failed: {str(e)}"
            }


def parse_geo_and_create_metadata(
    geo_text: str,
    sample_ids: List[str],
    disease: str = "",
    additional_context: str = "",
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    fallback_to_traditional: bool = True
) -> pd.DataFrame:
    """
    Main function to parse GEO metadata and create a metadata DataFrame using LLM-based intelligent mapping.
    
    This is the primary interface function that combines parsing and DataFrame creation.
    
    Args:
        geo_text: Raw GEO metadata text
        sample_ids: List of sample identifiers
        disease: Disease or condition context for better parsing
        additional_context: Additional context to help with sample-to-group mapping
        api_key: OpenAI API key (optional if set in environment)
        model: OpenAI model to use
        fallback_to_traditional: Whether to fall back to traditional parsing if LLM fails
        
    Returns:
        DataFrame with sample metadata suitable for DEG analysis
        
    Raises:
        ValueError: If parsing fails or data is inconsistent
    """
    # Initialize parser and client
    parser = GEOMetadataParser(api_key=api_key, model=model)
    client = openai.OpenAI(api_key=api_key)
    
    # Parse GEO metadata
    parsing_result = parser.parse_geo_metadata(geo_text, disease)
    
    # Create and return metadata DataFrame using LLM-based mapping
    return create_metadata_dataframe(
        parsing_result, 
        sample_ids, 
        geo_text=geo_text,
        client=client,
        additional_context=additional_context,
        fallback_to_traditional=fallback_to_traditional
    )


# Example usage and testing
def main():
    """Example usage of the GEO metadata parser"""
    
    # Example GEO metadata
    example_geo_text = '''
!Series_summary	"This study investigates the effects of lactate treatment on pancreatic cancer cell metabolism and gene expression."
!Series_overall_design	"PANC-1 pancreatic cancer cells were treated with 30 mM lactate or vehicle control. Three biological replicates per condition were analyzed by RNA-seq."
!Sample_characteristics_ch1	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"
!Sample_characteristics_ch1	"treatment: 30 mM lactate"	"treatment: 30 mM lactate"	"treatment: 30 mM lactate"	"treatment: control"	"treatment: control"	"treatment: control"
!Sample_characteristics_ch1	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"
'''
    
    # Sample IDs
    sample_ids = [
        "p1_lac1_FPKM", "p1_lac2_FPKM", "p1_lac3_FPKM",
        "p1_nc1_FPKM", "p1_nc2_FPKM", "p1_nc3_FPKM"
    ]
    
    try:
        # Parse and create metadata DataFrame using LLM-based intelligent mapping
        metadata_df = parse_geo_and_create_metadata(
            geo_text=example_geo_text,
            sample_ids=sample_ids,
            disease="pancreatic cancer",
            additional_context="This is a lactate treatment study with PANC-1 cells"
        )
        
        print("Generated Metadata DataFrame:")
        print(metadata_df)
        
        # Save to CSV
        metadata_df.to_csv("metadata.csv", index=False)
        logger.info("Metadata saved to metadata.csv")
        
    except Exception as e:
        logger.error(f"Failed to process GEO metadata: {e}")




def create_metadata_dataframe(
    parsing_result: GEOParsingResult, 
    sample_ids: List[str],
    geo_text: str = "",
    client: Optional[openai.OpenAI] = None,
    additional_context: str = "",
    fallback_to_traditional: bool = True
) -> pd.DataFrame:
    """
    Generate a metadata DataFrame using LLM-based intelligent sample-to-group mapping.
    
    Args:
        parsing_result: Result from GEO metadata parsing
        sample_ids: List of sample identifiers
        geo_text: Original GEO metadata text for additional context
        client: OpenAI client for LLM mapping (if None, creates new one)
        additional_context: Additional context to help with mapping
        fallback_to_traditional: Whether to fall back to traditional parsing if LLM fails
        
    Returns:
        DataFrame with sample IDs and their corresponding group assignments
        
    Raises:
        ValueError: If parsing result contains errors or mapping fails
    """
    if parsing_result.error:
        raise ValueError(f"Cannot create metadata DataFrame due to parsing error: {parsing_result.error}")
    
    if not parsing_result.group_values:
        raise ValueError("No group values found in parsing result")
    
    # Initialize client if not provided
    if client is None:
        client = openai.OpenAI()
    
    # Try LLM-based mapping first
    mapper = SampleGroupMapper(client)
    mapping_result = mapper.map_samples_to_groups(
        sample_ids, parsing_result, geo_text, additional_context
    )
    
    # Check if LLM mapping was successful
    if "error" not in mapping_result and "mappings" in mapping_result:
        logger.info(f"LLM mapping successful with {mapping_result.get('confidence', 'unknown')} confidence")
        logger.info(f"Strategy used: {mapping_result.get('mapping_strategy', 'Unknown')}")
        
        if mapping_result.get("notes"):
            logger.info(f"Notes: {mapping_result['notes']}")
        
        # Create DataFrame from LLM mappings
        mappings = mapping_result["mappings"]
        if not mappings:
            raise ValueError("LLM found no valid sample-to-group mappings")
        
        df_metadata = pd.DataFrame(
            mappings, 
            columns=["sample", parsing_result.group_column]
        )
        
        # Log excluded samples if any
        if mapping_result.get("excluded_samples"):
            logger.warning(f"Excluded samples: {mapping_result['excluded_samples']}")
        
        return df_metadata
    
    # Fallback to traditional method if LLM fails and fallback is enabled
    elif fallback_to_traditional:
        logger.warning(f"LLM mapping failed: {mapping_result.get('error', 'Unknown error')}")
        logger.info("Falling back to traditional metadata parsing...")
        
        return _create_metadata_dataframe_traditional(parsing_result, sample_ids)
    
    else:
        raise ValueError(f"LLM-based mapping failed: {mapping_result.get('error', 'Unknown error')}")


def _create_metadata_dataframe_traditional(
    parsing_result: GEOParsingResult, 
    sample_ids: List[str]
) -> pd.DataFrame:
    """
    Traditional method for creating metadata DataFrame (fallback).
    
    Args:
        parsing_result: Result from GEO metadata parsing
        sample_ids: List of sample identifiers
        
    Returns:
        DataFrame with sample IDs and their corresponding group assignments
    """
    # Extract group assignments from the metadata line
    group_assignments = extract_group_assignments(parsing_result.metadata_source_line)
    
    if len(sample_ids) != len(group_assignments):
        raise ValueError(
            f"Mismatch between sample IDs count ({len(sample_ids)}) "
            f"and group assignments count ({len(group_assignments)})"
        )
    
    # Filter samples to include only those with group values we're interested in
    valid_groups = set(parsing_result.group_values)
    metadata_rows = []
    
    for sample_id, group_value in zip(sample_ids, group_assignments):
        if group_value in valid_groups:
            metadata_rows.append([sample_id, group_value])
    
    if not metadata_rows:
        raise ValueError("No samples found with the specified group values")
    
    # Create DataFrame
    df_metadata = pd.DataFrame(
        metadata_rows, 
        columns=["sample", parsing_result.group_column]
    )
    
    return df_metadata


def parse_geo_and_create_metadata(
    geo_text: str,
    sample_ids: List[str],
    disease: str = "",
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> pd.DataFrame:
    """
    Main function to parse GEO metadata and create a metadata DataFrame.
    
    This is the primary interface function that combines parsing and DataFrame creation.
    
    Args:
        geo_text: Raw GEO metadata text
        sample_ids: List of sample identifiers
        disease: Disease or condition context for better parsing
        api_key: OpenAI API key (optional if set in environment)
        model: OpenAI model to use
        
    Returns:
        DataFrame with sample metadata suitable for DEG analysis
        
    Raises:
        ValueError: If parsing fails or data is inconsistent
    """
    # Initialize parser
    parser = GEOMetadataParser(api_key=api_key, model=model)
    
    # Parse GEO metadata
    parsing_result = parser.parse_geo_metadata(geo_text, disease)
    
    # Create and return metadata DataFrame
    return create_metadata_dataframe(parsing_result, sample_ids)


# Example usage and testing
def main():
    """Example usage of the GEO metadata parser"""
    
    # Example GEO metadata
    example_geo_text = '''
!Series_summary	"This study investigates the effects of lactate treatment on pancreatic cancer cell metabolism and gene expression."
!Series_overall_design	"PANC-1 pancreatic cancer cells were treated with 30 mM lactate or vehicle control. Three biological replicates per condition were analyzed by RNA-seq."
!Sample_characteristics_ch1	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"
!Sample_characteristics_ch1	"treatment: 30 mM lactate"	"treatment: 30 mM lactate"	"treatment: 30 mM lactate"	"treatment: control"	"treatment: control"	"treatment: control"
!Sample_characteristics_ch1	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"	"batch: batch1"
'''
    
    # Sample IDs
    sample_ids = [
        "p1_lac1_FPKM", "p1_lac2_FPKM", "p1_lac3_FPKM",
        "p1_nc1_FPKM", "p1_nc2_FPKM", "p1_nc3_FPKM"
    ]
    
    try:
        # Parse and create metadata DataFrame
        metadata_df = parse_geo_and_create_metadata(
            geo_text=example_geo_text,
            sample_ids=sample_ids,
            disease="pancreatic cancer"
        )
        
        print("Generated Metadata DataFrame:")
        print(metadata_df)
        
        # Save to CSV
        metadata_df.to_csv("metadata.csv", index=False)
        logger.info("Metadata saved to metadata.csv")
        
    except Exception as e:
        logger.error(f"Failed to process GEO metadata: {e}")


if __name__ == "__main__":
    main()