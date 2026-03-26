import os
import pandas as pd
from .utils.detect_header_data import detect_header_data_mismatch
from .utils.find_set_header import find_and_set_proper_header
from .utils.find_missing_header import fix_missing_first_header_column
from .utils.convert_strings_numeric import convert_strings_to_numeric
from .utils.analyze_data import analyze_dataframe

def analyze_file(file_path):
    """
    Analyze a single file and return processed dataframe with analysis results.
    
    Args:
        file_path (str): Path to the file to analyze
        
    Returns:
        dict: Contains 'dataframe', 'numeric_cols', 'status', and 'message'
    """
    if not os.path.exists(file_path):
        return {"dataframe": None, "numeric_cols": [], "status": "error", "message": "File not found"}
    
    file_name = os.path.basename(file_path)
    file_extension = file_name.split(".")[-1].lower()
    
    try:
        # Read file based on extension
        if file_extension == "txt":
            # Detect header data mismatch first
            header_data_mismatch = detect_header_data_mismatch(file_path)
            
            if header_data_mismatch.get('df') is not None:
                counts_df = header_data_mismatch['df']
            else:
                # Fallback: try different separators
                separators = ["\t", ",", ";", " "]
                counts_df = None
                
                for sep in separators:
                    try:
                        counts_df = pd.read_csv(file_path, sep=sep, low_memory=False, index_col=False)
                        if counts_df.shape[1] > 1:
                            break
                    except:
                        continue
                
                if counts_df is None:
                    return {"dataframe": None, "numeric_cols": [], "status": "error", "message": "Failed to read txt file"}
            
            # Fix header issues if detected
            if header_data_mismatch.get('has_mismatch', False):
                counts_df = fix_missing_first_header_column(counts_df)
                
        elif file_extension == "csv":
            counts_df = pd.read_csv(file_path, low_memory=False, index_col=False)
            
        elif file_extension in ["xlsx", "xls"]:
            counts_df = pd.read_excel(file_path, index_col=False)
            
        else:
            return {"dataframe": None, "numeric_cols": [], "status": "error", "message": f"Unsupported file extension: {file_extension}"}
        
        # Apply processing pipeline
        counts_df = find_and_set_proper_header(counts_df)
        counts_df = convert_strings_to_numeric(counts_df)
        counts_df, numeric_cols = analyze_dataframe(counts_df, file_path, file_name)
        
        return {
            "dataframe": counts_df,
            "numeric_cols": numeric_cols,
            "status": "success",
            "message": f"Successfully processed {file_name}"
        }
        
    except Exception as e:
        return {
            "dataframe": None,
            "numeric_cols": [],
            "status": "error",
            "message": f"Error processing {file_name}: {str(e)}"
        }

if __name__ == "__main__":
    # Example usage:
    result = analyze_file("/home/msq/workdir/UW/ayassbs/software/backend/agenticaib/agentic_ai_wf/shared/cohort_data/GEO/Pancreatic Cancer/GSE282795/GSE282795_LU99_raw_counts (2).txt")
    if result["status"] == "success":
        df = result["dataframe"]
        numeric_cols = result["numeric_cols"]
        print(f"Processed successfully: {result}")
    else:
        print(f"Error: {result['message']}")