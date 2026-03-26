import pandas as pd


def drugs_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process dataframe by sorting and deduplicating based on drug_id
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"Unique drug_ids before deduplication: {df['drug_id'].nunique()}")
    
    # Sort by llm_score (descending) first, then by molecular_evidence_score (descending)
    print("Sorting dataframe...")
    df_sorted = df.sort_values(
        by=['llm_score', 'molecular_evidence_score'], 
        ascending=[False, False]  # Both descending for highest scores first
    ).reset_index(drop=True)
    
    if len(df_sorted) > 10:
        # Keep only first occurrence of each drug_id (deduplication)
        print("Deduplicating by drug_id...")
        df_deduped = df_sorted.drop_duplicates(subset=['drug_id'], keep='first').reset_index(drop=True)
    
        print(f"Final dataframe shape: {df_deduped.shape}")
        print(f"Unique drug_ids after deduplication: {df_deduped['drug_id'].nunique()}")
        
        # Display top 5 rows for verification
        print("\nTop 5 rows after processing:")
        print(df_deduped[['drug_id', 'drug_name', 'llm_score', 'molecular_evidence_score']].head())
        


        return df_deduped
    else:
        print("No deduplication performed as there are less than 10 rows.")
        return df

# Example usage:
if __name__ == "__main__":
    # Replace 'your_input_file.csv' with your actual file path
    input_file = '/home/msq/workdir/UW/ayassbs/software/backend/agenticaib/agentic_ai_wf/shared/drugs_discovery/c8fc3428-ba9f-4a46-8eb2-95fb984a3750/drug_discovery_result.csv'
    
    df = pd.read_csv(input_file)
    try:
        processed_df = drugs_deduplication(df)
        
        # Additional verification
        print(f"\nVerification:")
        print(f"All drug_ids are unique: {processed_df['drug_id'].is_unique}")
        print(f"Sorting verification - llm_score is sorted (desc): {processed_df['llm_score'].is_monotonic_decreasing}")
        
        # Check if molecular_evidence_score is properly sorted within llm_score groups
        for score in processed_df['llm_score'].unique():
            subset = processed_df[processed_df['llm_score'] == score]
            if len(subset) > 1:
                is_sorted = subset['molecular_evidence_score'].is_monotonic_decreasing
                print(f"For llm_score {score}, molecular_evidence_score sorted (desc): {is_sorted}")
                if not is_sorted:
                    break
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

