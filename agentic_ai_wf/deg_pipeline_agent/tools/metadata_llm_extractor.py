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

from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")

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
    
    def __init__(self, api_key: Optional[str] = OPENAI_API_KEY, model: str = "gpt-4.1-mini-2025-04-14"):
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


def create_metadata_dataframe(
    parsing_result: GEOParsingResult, 
    sample_ids: List[str]
) -> pd.DataFrame:
    """
    Generate a metadata DataFrame from parsing results and sample IDs.
    
    Args:
        parsing_result: Result from GEO metadata parsing
        sample_ids: List of sample identifiers corresponding to the metadata
        
    Returns:
        DataFrame with sample IDs and their corresponding group assignments
        
    Raises:
        ValueError: If parsing result contains errors or invalid data
    """
    try:
        if parsing_result.error:
            raise ValueError(f"Cannot create metadata DataFrame due to parsing error: {parsing_result.error}")
        
        if not parsing_result.group_values:
            raise ValueError("No group values found in parsing result")
        
        # Extract group assignments from the metadata line
        group_assignments = extract_group_assignments(parsing_result.metadata_source_line)
        
        if len(sample_ids) != len(group_assignments):
            print(f"Mismatch between sample IDs count ({len(sample_ids)}) and group assignments count ({len(group_assignments)})")
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
    except Exception as e:
        print(f"Error creating metadata DataFrame: {e}")
        return pd.DataFrame([{"sample": "error", "group": "error"}])

def create_metadata_dataframe_with_llm_matching(
    parsing_result: GEOParsingResult,
    sample_ids: List[str]
) -> pd.DataFrame:
    """
    Generate a metadata DataFrame using LLM fallback when mismatch occurs
    between sample IDs and group assignments.
    
    Args:
        parsing_result: Result from GEO metadata parsing
        sample_ids: List of sample identifiers
        
    Returns:
        DataFrame with matched sample IDs and their corresponding group
        
    Raises:
        ValueError: If no valid matches can be constructed
    """
    if parsing_result.error:
        raise ValueError(f"Cannot create metadata DataFrame due to parsing error: {parsing_result.error}")

    if not parsing_result.group_values:
        raise ValueError("No group values found in parsing result")
    
    group_assignments = extract_group_assignments(parsing_result.metadata_source_line)
    
    # Fallback logic: if mismatch in lengths, try to align with LLM help
    if len(sample_ids) != len(group_assignments):
        print(f"🔍 Mismatch detected: {len(sample_ids)} sample_ids vs {len(group_assignments)} group_assignments.")
        # Simulate LLM matching (in production replace with actual call)
        matched_pairs = match_sample_ids_to_groups_with_llm(sample_ids, group_assignments)
    else:
        matched_pairs = list(zip(sample_ids, group_assignments))

    valid_groups = set(parsing_result.group_values)
    metadata_rows = [
        [sample, group] for sample, group in matched_pairs
        if group in valid_groups
    ]

    if not metadata_rows:
        raise ValueError("❌ No valid sample-group pairs could be matched using LLM fallback.")

    return pd.DataFrame(metadata_rows, columns=["sample", parsing_result.group_column])

def match_sample_ids_to_groups_with_llm(
    sample_ids: List[str],
    group_assignments: List[str]
) -> List[Tuple[str, str]]:
    """
    Simulate LLM-based matching of sample IDs and group assignments
    when their lengths do not match.
    
    This is a simplified placeholder version.
    
    Returns:
        List of aligned (sample_id, group_value) tuples
    """
    min_len = min(len(sample_ids), len(group_assignments))
    print(f"⚠️ Using first {min_len} items from each list for safe alignment.")
    return list(zip(sample_ids[:min_len], group_assignments[:min_len]))


def parse_geo_and_create_metadata(
    geo_text: str,
    sample_ids: List[str],
    disease: str = "",
    api_key: Optional[str] = OPENAI_API_KEY,
    model: str = "gpt-4.1-mini-2025-04-14"
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

    print(f"✅ Parsing GEO metadata ...")
    
    # Initialize parser
    parser = GEOMetadataParser(api_key=api_key, model=model)
    
    # Parse GEO metadata
    parsing_result = parser.parse_geo_metadata(geo_text, disease)

    print(f"✅ Parsing Result: {parsing_result}")
    
    # Create and return metadata DataFrame
    return create_metadata_dataframe_with_llm_matching(parsing_result, sample_ids)


# Example usage and testing
def main():
    """Example usage of the GEO metadata parser"""
    
    # Example GEO metadata
    example_geo_text = '''
!Series_summary	"Lycorine (LY) is a natural alkaloid extracted from the Amaryllidaceae plant, it has been evaluated for its anticancer effects on pancreatic cancer cells and xenograft models. Our research presents a multi-faceted investigation of LY, including assessments of its impact on cell viability, proliferation, migration, invasion, and potential mechanisms of action."
!Series_overall_design	"We divided the samples into two groups, four each: a control group and a group treated with 20μM LY. After 24 hours of treatment, we extracted RNA using Trizol for mRNA sequencing."
!Sample_characteristics_ch1	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"	"tissue: cell line"
!Sample_characteristics_ch1	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"	"cell line: PANC-1"
!Sample_characteristics_ch1	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"	"cell type: Human pancreatic cancer cell"
!Sample_characteristics_ch1	"treatment: 20μM LY"	"treatment: 20μM LY"	"treatment: 20μM LY"	"treatment: 20μM LY"	"treatment: 20μM LY"	"treatment: DMSO"	"treatment: DMSO"	"treatment: DMSO"	"treatment: DMSO"	"treatment: DMSO"
!Sample_molecule_ch1	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"	"total RNA"
'''
    
    # Sample IDs
    sample_ids = [
        "LY-1", "LY-2", "LY-3", "LY-4", "LY-5", "NC-1", "NC-2", "NC-3", "NC-4", "NC-5"
    ]
    
    try:
        # Parse and create metadata DataFrame
        metadata_df = parse_geo_and_create_metadata(
            geo_text=example_geo_text,
            sample_ids=sample_ids,
            disease="pancreatic cancer",
            model="gpt-4.1-mini-2025-04-14"
        )
        
        print("Generated Metadata DataFrame:")
        print(metadata_df)
        
        # Save to CSV
        metadata_df.to_csv("metadata_llm_extractor541.csv", index=False)
        logger.info("Metadata saved to metadata.csv")
        
    except Exception as e:
        logger.error(f"Failed to process GEO metadata: {e}")


if __name__ == "__main__":
    main()