#!/usr/bin/env python3
"""
Signatures Grouper Module

Groups and sorts molecular signatures from consolidated CSV files based on 
Main_Class, Priority_Rank, and LLM_Score with special handling for GO-based 
entries.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd


def validate_csv_columns(df: pd.DataFrame) -> bool:
    """
    Validate that the CSV contains all required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if all required columns are present, False otherwise
    """
    required_columns = [
        'Main_Class', 'Pathway_ID', 'Pathway_Name', 
        'Priority_Rank', 'LLM_Score', 'DB_ID'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    return True


def clean_signature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare molecular signature data for grouping.
    
    Args:
        df: Raw signature DataFrame
        
    Returns:
        Cleaned DataFrame with proper data types and handling of missing values
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle missing values
    cleaned_df['Main_Class'] = cleaned_df['Main_Class'].fillna('Unknown')
    cleaned_df['Priority_Rank'] = pd.to_numeric(
        cleaned_df['Priority_Rank'], errors='coerce'
    ).fillna(999)  # High number for missing ranks
    cleaned_df['LLM_Score'] = pd.to_numeric(
        cleaned_df['LLM_Score'], errors='coerce'
    ).fillna(0.0)
    cleaned_df['DB_ID'] = cleaned_df['DB_ID'].fillna('')
    
    # Add GO classification flag
    cleaned_df['is_go_entry'] = cleaned_df['DB_ID'].str.startswith('GO_')
    
    return cleaned_df


def sort_signatures_within_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort molecular signatures within a group by specified criteria.
    
    Args:
        group_df: DataFrame containing signatures from a single Main_Class
        
    Returns:
        Sorted DataFrame with GO entries de-prioritized
    """
    # Sort by: is_go_entry (False first), Priority_Rank (asc), LLM_Score (desc)
    sorted_df = group_df.sort_values(
        by=['is_go_entry', 'Priority_Rank', 'LLM_Score'],
        ascending=[True, True, False]
    )
    
    return sorted_df


def group_signatures(
    input_data: Union[str, pd.DataFrame]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group and sort molecular signatures from a CSV file or DataFrame.
    
    Args:
        input_data: Either a path to the signatures consolidated CSV file
                   or a pandas DataFrame containing the signature data
        
    Returns:
        Dictionary with Main_Class as keys and sorted signature lists as values
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
        pd.errors.EmptyDataError: If the CSV file is empty
        TypeError: If input_data is neither string nor DataFrame
    """
    try:
        # Handle different input types
        if isinstance(input_data, str):
            # Validate file exists
            if not Path(input_data).exists():
                raise FileNotFoundError(f"CSV file not found: {input_data}")
            # Read CSV file
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            # Use provided DataFrame
            df = input_data.copy()
        else:
            raise TypeError(
                f"input_data must be a string (file path) or pandas DataFrame, "
                f"got {type(input_data)}"
            )
        
        if df.empty:
            raise pd.errors.EmptyDataError("Input data is empty")
        
        # Validate required columns
        if not validate_csv_columns(df):
            raise ValueError("Missing required columns in input data")
        
        # Clean and prepare data
        cleaned_df = clean_signature_data(df)
        
        # Group by Main_Class and count signatures per group
        class_counts = cleaned_df.groupby('Main_Class').size().sort_values(
            ascending=False
        )

        # Normalize the index to lowercase for safe comparison
        lower_index = [str(x).lower() for x in class_counts.index]

        # Ensure "Human Diseases" appears at the end if present
        if "human diseases" in lower_index:
            # Get the actual key as it exists in the index
            hd_key = class_counts.index[lower_index.index("human diseases")]
            
            other_classes = class_counts.drop(hd_key)
            class_counts = pd.concat([other_classes, class_counts[[hd_key]]])
        
        # Initialize result dictionary
        grouped_signatures: Dict[str, List[Dict[str, Any]]] = {}
        
        # Process each Main_Class in descending order of signature count
        for main_class in class_counts.index:
            # Get signatures for this class
            class_group = cleaned_df[cleaned_df['Main_Class'] == main_class]
            
            # Sort signatures within the group
            sorted_group = sort_signatures_within_group(class_group)
            
            # Convert to list of dictionaries
            signature_list = []
            for _, row in sorted_group.iterrows():
                signature_dict = {
                    'Pathway_ID': row['Pathway_ID'],
                    'Pathway_Name': row['Pathway_Name'],
                    'Priority_Rank': int(row['Priority_Rank']),
                    'LLM_Score': float(row['LLM_Score']),
                    'DB_ID': row['DB_ID'],
                    'is_go_entry': bool(row['is_go_entry'])
                }
                
                # Add all optional fields if they exist and are not null
                optional_fields = [
                    'Confidence_Level', 'Score_Justification', 'Pathway_Source',
                    'Number_of_Genes', 'Number_of_Genes_in_Background', 
                    'P_Value', 'FDR', 'Hit_Score', 'Regulation',
                    'Clinical_Relevance', 'Functional_Relevance', 'Sub_Class',
                    'Disease_Category', 'Disease_Subcategory', 
                    'Cellular_Component', 'Subcellular_Element', 
                    'Ontology_Source', 'Input_Genes', 
                    'Pathway_Associated_Genes'
                ]
                
                for field in optional_fields:
                    if field in row and pd.notna(row[field]):
                        # Convert numeric fields to appropriate types
                        if field in ['Number_of_Genes', 
                                   'Number_of_Genes_in_Background']:
                            signature_dict[field] = int(row[field])
                        elif field in ['P_Value', 'FDR', 'Hit_Score']:
                            signature_dict[field] = float(row[field])
                        else:
                            signature_dict[field] = row[field]
                
                signature_list.append(signature_dict)
            
            grouped_signatures[main_class] = signature_list
        
        return grouped_signatures
        
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error processing input data: {e}")
    except Exception as e:
        raise ValueError(f"Error processing input data: {e}")


def print_summary(grouped_signatures: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Print a summary of the grouped molecular signatures.
    
    Args:
        grouped_signatures: The grouped signature dictionary
    """
    print("\n=== Signature Grouping Summary ===")
    print(f"Total Main_Class groups: {len(grouped_signatures)}")
    
    total_signatures = sum(len(signatures) for signatures in grouped_signatures.values())
    print(f"Total signatures: {total_signatures}")
    
    print("\nGroups by size:")
    for main_class, signatures in grouped_signatures.items():
        go_count = sum(1 for s in signatures if s.get('is_go_entry', False))
        non_go_count = len(signatures) - go_count
        print(f"  {main_class}: {len(signatures)} signatures "
              f"({non_go_count} non-GO, {go_count} GO)")


def main() -> None:
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Group and sort molecular signatures from consolidated CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python signatures_grouper.py signatures_consolidated.csv
  python signatures_grouper.py /path/to/signatures.csv --summary
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the molecular signatures consolidated CSV file'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of grouped molecular signatures'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Optional output file path to save grouped signature data as JSON'
    )
    
    args = parser.parse_args()
    
    try:
        # Group signatures
        print(f"Processing CSV file: {args.csv_file}")
        grouped_signatures = group_signatures(args.csv_file)
        
        # Print summary if requested
        if args.summary:
            print_summary(grouped_signatures)
        
        # Save to output file if specified
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(grouped_signatures, f, indent=2, ensure_ascii=False)
            print(f"\nGrouped signatures saved to: {args.output}")
        
        print("\n✅ Signature grouping completed successfully!")
        
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
