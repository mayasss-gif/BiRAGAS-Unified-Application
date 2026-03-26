# src/pathway_prioritization/utils/file_utils.py
import os
import json
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Set, List, Optional
import logging

logger = logging.getLogger(__name__)

def find_existing_results(output_folder: Path, disease_name: str) -> Dict[str, Any]:
    """Find existing FINAL and intermediate result files for smart caching"""
    disease_clean = disease_name.replace(' ', '_').replace('/', '_')

    # Search for FINAL results
    final_file = output_folder / f"{disease_clean}_Pathways_Consolidated.csv"

    # Search for intermediate results
    intermediate_pattern = str(output_folder / f"intermediate_results_{disease_clean}_*.csv")
    intermediate_files = glob.glob(intermediate_pattern)

    # Sort files by modification time (newest first)
    intermediate_files.sort(key=os.path.getmtime, reverse=True)

    # Search for summary files
    summary_pattern = str(output_folder / f"pathway_prioritization_summary_{disease_clean}_*.json")
    summary_files = glob.glob(summary_pattern)
    summary_files.sort(key=os.path.getmtime, reverse=True)

    results = {
        'final_results_found': final_file.exists(),
        'latest_final_file': final_file if final_file.exists() else None,
        'intermediate_files': intermediate_files,
        'latest_intermediate_file': intermediate_files[0] if intermediate_files else None,
        'summary_files': summary_files,
        'latest_summary_file': summary_files[0] if summary_files else None
    }

    logger.info(f"Smart caching scan for {disease_name}:")
    logger.info(f"  FINAL file found: {results['final_results_found']}")
    logger.info(f"  Intermediate files found: {len(intermediate_files)}")
    logger.info(f"  Summary files found: {len(summary_files)}")

    return results

def load_processed_pathways(result_file: Path) -> Set[str]:
    """Load unique pathway identifiers from existing result files"""
    processed_pathways = set()

    try:
        df = pd.read_csv(result_file)
        logger.info(f"Loading processed pathways from {result_file.name}")
        
        # Define the key columns we need for pathway identification
        column_mappings = {
            'pathway_source': ['Pathway_Source', 'pathway_source', 'Pathway source', 'source'],
            'pathway_id': ['Pathway_ID', 'pathway_id', 'Pathway ID', 'ID'],
            'pathway_name': ['Pathway_Name', 'pathway_name', 'Pathway', 'name'],
            'db_id': ['DB_ID', 'db_id', 'database_id']
        }

        # Find the actual column names in the CSV
        found_columns = {}
        for key, possible_names in column_mappings.items():
            for col_name in possible_names:
                if col_name in df.columns:
                    found_columns[key] = col_name
                    break

        if not found_columns:
            logger.error(f"No pathway identifier columns found in {result_file}")
            return processed_pathways

        # Extract unique pathway composite identifiers
        unique_pathways = set()
        for _, row in df.iterrows():
            try:
                pathway_parts = []
                for key in ['pathway_source', 'pathway_id', 'pathway_name', 'db_id']:
                    if key in found_columns:
                        value = str(row.get(found_columns[key], '')).strip()
                        if value and value.lower() != 'nan' and value != '':
                            pathway_parts.append(f"{key}:{value}")

                if pathway_parts:
                    composite_id = "|".join(pathway_parts)
                    unique_pathways.add(composite_id)

            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue

        processed_pathways.update(unique_pathways)
        logger.info(f"Processed {len(unique_pathways)} unique pathways")

    except Exception as e:
        logger.error(f"Error loading processed pathways from {result_file}: {e}")

    return processed_pathways

def save_json_file(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data to JSON file with error handling"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def save_dataframe_to_csv(df: pd.DataFrame, output_file: Path) -> bool:
    """Save DataFrame to CSV with error handling"""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV file {output_file}: {e}")
        return False