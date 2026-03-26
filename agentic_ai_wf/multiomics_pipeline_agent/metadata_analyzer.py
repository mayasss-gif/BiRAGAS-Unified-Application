"""
Utility module for intelligently analyzing metadata to detect label columns.
Uses LLM-based analysis when available, falls back to rule-based detection.
"""

import os
from typing import Optional

import pandas as pd

# Try to import OpenAI for LLM-based analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def detect_label_column_llm(metadata_df: pd.DataFrame, api_key: Optional[str] = None) -> Optional[str]:
    """
    Use LLM to intelligently detect the label column from metadata.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        The metadata DataFrame to analyze
    api_key : str, optional
        OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable.
        
    Returns
    -------
    str or None
        The name of the detected label column, or None if detection fails.
    """
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        # Get API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Prepare metadata summary for LLM
        # Get column names, data types, and sample values
        column_info = []
        for col in metadata_df.columns:
            dtype = str(metadata_df[col].dtype)
            unique_count = metadata_df[col].nunique()
            sample_values = metadata_df[col].dropna().head(5).tolist()
            column_info.append({
                "name": col,
                "dtype": dtype,
                "unique_values": unique_count,
                "sample_values": [str(v) for v in sample_values]
            })
        
        # Create prompt for LLM
        prompt = f"""You are analyzing a metadata DataFrame for a multiomics analysis pipeline. 
Your task is to identify which column contains the sample labels/classes/conditions that should be used for supervised machine learning.

Here are the columns in the metadata:
{column_info}

Considerations:
1. The label column should contain categorical/discrete values (like "Diseased", "Healthy", "Control", "Case", etc.)
2. It should NOT be a sample ID or index column
3. It should have relatively few unique values (typically 2-10 for binary/multi-class classification)
4. Column names like "label", "status", "condition", "class", "group", "phenotype", "disease_status" are good candidates
5. Avoid columns that are clearly identifiers, dates, or continuous numeric values

Respond with ONLY the column name that should be used as the label column. If no suitable column exists, respond with "NONE".
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a cost-effective model
            messages=[
                {"role": "system", "content": "You are a bioinformatics expert helping to identify label columns in metadata."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for deterministic output
            max_tokens=50
        )
        
        detected_col = response.choices[0].message.content.strip()
        
        # Clean up the response
        detected_col = detected_col.strip('"').strip("'")
        
        # Validate that the detected column exists
        if detected_col.upper() == "NONE" or detected_col not in metadata_df.columns:
            return None
        
        return detected_col
        
    except Exception as e:
        print(f"Warning: LLM-based label detection failed: {e}")
        return None


def detect_label_column_smart(metadata_df: pd.DataFrame, use_llm: bool = True) -> Optional[str]:
    """
    Intelligently detect the label column from metadata.
    First tries LLM-based detection if available, then falls back to rule-based detection.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        The metadata DataFrame to analyze
    use_llm : bool, default=True
        Whether to attempt LLM-based detection first
        
    Returns
    -------
    str or None
        The name of the detected label column, or None if no suitable column is found.
    """
    # Try LLM-based detection first if requested and available
    if use_llm and OPENAI_AVAILABLE:
        llm_result = detect_label_column_llm(metadata_df)
        if llm_result:
            print(f"🤖 LLM detected label column: {llm_result}")
            return llm_result
    
    # Fall back to rule-based detection
    # Priority order: exact matches > keyword matches > categorical columns
    
    # 1. Check for exact common label column names
    exact_matches = ["label", "Label", "LABEL", "condition", "Condition", "CONDITION",
                     "status", "Status", "STATUS", "class", "Class", "CLASS",
                     "group", "Group", "GROUP", "phenotype", "Phenotype", "PHENOTYPE"]
    for col in metadata_df.columns:
        if col in exact_matches:
            return col
    
    # 2. Check for keyword matches (case-insensitive)
    keyword_patterns = ["label", "status", "class", "group", "phenotype", "condition",
                       "disease", "outcome", "response", "treatment"]
    for col in metadata_df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in keyword_patterns):
            # Additional check: should be categorical (not too many unique values)
            unique_count = metadata_df[col].nunique()
            if unique_count <= 20:  # Reasonable threshold for categorical
                return col
    
    # 3. Find categorical columns with few unique values (potential labels)
    # Skip obvious ID columns
    id_keywords = ["id", "sample", "patient", "subject", "index"]
    categorical_candidates = []
    
    for col in metadata_df.columns:
        col_lower = col.lower()
        # Skip ID columns
        if any(keyword in col_lower for keyword in id_keywords):
            continue
        
        unique_count = metadata_df[col].nunique()
        total_count = len(metadata_df)
        
        # Good label column: few unique values relative to total, but not just one value
        if 2 <= unique_count <= min(20, total_count * 0.5):
            # Check if values look categorical (not numeric IDs)
            sample_values = metadata_df[col].dropna().head(10)
            # If values are strings or look like categories
            if metadata_df[col].dtype == 'object' or all(str(v).isalpha() or str(v).replace('_', '').replace('-', '').isalnum() for v in sample_values):
                categorical_candidates.append((col, unique_count))
    
    # Return the column with the fewest unique values (most likely to be a label)
    if categorical_candidates:
        categorical_candidates.sort(key=lambda x: x[1])
        return categorical_candidates[0][0]
    
    return None

