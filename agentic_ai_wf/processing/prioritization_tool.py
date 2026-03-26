import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from agents import function_tool

from ..models.drugs_models import (
    EnrichedDrugDegInput, 
    EnrichedDrugPriority, 
    EnrichedDrugPriorityOutput
)
from agentic_ai_wf.config import global_config
from agentic_ai_wf.config.global_config import PROCESSED_PATHWAYS_DIR

# Configure logging
logger = logging.getLogger(__name__)


class PrioritizationError(Exception):
    """Custom exception for drug prioritization errors."""
    pass


def validate_prioritization_input(drugs: EnrichedDrugDegInput) -> None:
    """
    Validate input data for drug prioritization.
    
    Args:
        drugs: Input data containing drugs to prioritize
        
    Raises:
        PrioritizationError: If validation fails
    """
    if not drugs:
        raise PrioritizationError("Input drugs object is None or empty")
    
    if not hasattr(drugs, 'drugs') or not drugs.drugs:
        raise PrioritizationError("No drugs provided for prioritization")
    
    if not isinstance(drugs.drugs, list):
        raise PrioritizationError("Drugs must be provided as a list")
    
    logger.info(f"Validation passed: {len(drugs.drugs)} drugs ready for prioritization")


def calculate_priority_score(drug) -> int:
    """
    Calculate priority score for a drug based on weighted criteria.
    
    Priority Logic (highest to lowest):
    - DEG Match + Matching Status + Approved = 6 (Most favorable)
    - DEG Match + Matching Status + Not Approved = 5
    - DEG Match + Not Matching + Approved = 4
    - DEG Match + Not Matching + Not Approved = 3
    - No DEG Match + Matching Status + Approved = 2
    - No DEG Match + Matching Status + Not Approved = 1
    - No DEG Match + Not Matching + Any Approval = 0 (Least favorable)
    
    Args:
        drug: Drug object with deg_match_status, matching_status, and approved fields
        
    Returns:
        Priority score (0-6)
    """
    try:
        deg_match = getattr(drug, 'deg_match_status', False)
        matching = getattr(drug, 'matching_status', False)
        approved = getattr(drug, 'approved', False)
        
        if deg_match:
            if matching:
                return 6 if approved else 5
            else:
                return 4 if approved else 3
        else:
            if matching:
                return 2 if approved else 1
            else:
                return 0
                
    except Exception as e:
        logger.warning(f"Error calculating priority for drug {getattr(drug, 'name', 'unknown')}: {e}")
        return 0


def create_output_directory(output_dir: str, analysis_id: str) -> Path:
    """
    Create output directory structure for prioritized drugs.
    
    Args:
        output_dir: Base output directory
        analysis_id: Analysis identifier for subdirectory
        
    Returns:
        Path object for the created directory
        
    Raises:
        PrioritizationError: If directory creation fails
    """
    try:
        if not output_dir:
            raise PrioritizationError("Output directory cannot be empty")
        
        if not analysis_id:
            raise PrioritizationError("Analysis ID cannot be empty")
        
        # Create the full path: outputdir/analysis_id/
        full_path = Path(output_dir) / analysis_id
        full_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {full_path}")
        return full_path
        
    except Exception as e:
        raise PrioritizationError(f"Failed to create output directory {output_dir}/{analysis_id}: {e}")


def generate_filename(drugs_df: pd.DataFrame, analysis_id: str) -> str:
    """
    Generate appropriate filename for prioritized drugs output.
    
    Args:
        drugs_df: DataFrame containing drug data
        analysis_id: Analysis identifier
        
    Returns:
        Generated filename
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Try to get pathway info for filename
        if not drugs_df.empty and 'pathway_id' in drugs_df.columns:
            unique_pathways = drugs_df['pathway_id'].nunique()
            if unique_pathways == 1:
                pathway_id = drugs_df['pathway_id'].iloc[0]
                filename = f"prioritized_drugs_{pathway_id}_{timestamp}.csv"
            else:
                filename = f"prioritized_drugs_multiple_pathways_{timestamp}.csv"
        else:
            filename = f"prioritized_drugs_{analysis_id}_{timestamp}.csv"
        
        logger.info(f"Generated filename: {filename}")
        return filename
        
    except Exception as e:
        logger.warning(f"Error generating filename: {e}")
        # Fallback filename
        return f"prioritized_drugs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def save_prioritized_drugs(drugs_df: pd.DataFrame, output_path: Path, filename: str) -> str:
    """
    Save prioritized drugs DataFrame to CSV file.
    
    Args:
        drugs_df: DataFrame containing prioritized drugs
        output_path: Directory path to save the file
        filename: Name of the output file
        
    Returns:
        Full path to the saved file
        
    Raises:
        PrioritizationError: If saving fails
    """
    try:
        file_path = output_path / filename
        drugs_df.to_csv(file_path, index=False)
        
        logger.info(f"Successfully saved {len(drugs_df)} prioritized drugs to {file_path}")
        return str(file_path)
        
    except Exception as e:
        raise PrioritizationError(f"Failed to save prioritized drugs to {output_path}/{filename}: {e}")


def get_analysis_id() -> str:
    """
    Get analysis ID from global config or generate a default one.
    
    Returns:
        Analysis ID string
    """
    try:
        # Try to get from global config
        analysis_id = getattr(global_config, 'patient_dir', None)
        if analysis_id and analysis_id.strip():
            return analysis_id.strip()
        
        # Fallback to timestamp-based ID
        fallback_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.warning(f"No analysis ID found in global config, using fallback: {fallback_id}")
        return fallback_id
        
    except Exception as e:
        logger.warning(f"Error getting analysis ID: {e}")
        return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_prioritization_summary(drugs: List[EnrichedDrugPriority]) -> Dict[str, Any]:
    """
    Create a summary of the prioritization results.
    
    Args:
        drugs: List of prioritized drugs
        
    Returns:
        Dictionary containing prioritization summary
    """
    if not drugs:
        return {"total_drugs": 0, "message": "No drugs prioritized"}
    
    try:
        df = pd.DataFrame([d.model_dump() for d in drugs])
        
        summary = {
            "total_drugs": len(drugs),
            "priority_distribution": df['priority_status'].value_counts().to_dict(),
            "approved_drugs": len(df[df['approved'] == True]),
            "deg_matched_drugs": len(df[df['deg_match_status'] == True]),
            "condition_matched_drugs": len(df[df['matching_status'] == True]),
            "high_priority_drugs": len(df[df['priority_status'] >= 4]),
            "unique_pathways": df['pathway_id'].nunique() if 'pathway_id' in df.columns else 0
        }
        
        return summary
        
    except Exception as e:
        logger.warning(f"Error creating prioritization summary: {e}")
        return {"total_drugs": len(drugs), "error": str(e)}


@function_tool
def prioritize_drugs(
    drugs: EnrichedDrugDegInput, 
    output_dir: Optional[str] = None,
    analysis_id: Optional[str] = None
) -> EnrichedDrugPriorityOutput:
    """
    Prioritize drugs based on DEG match, matching status, and approval status.
    
    This function implements a weighted prioritization system where:
    1. DEG Match status has the highest weight
    2. Matching status (condition match) has medium weight  
    3. Approval status has the lowest weight
    
    Args:
        drugs: Input containing drugs to prioritize
        output_dir: Optional output directory (defaults to PROCESSED_PATHWAYS_DIR)
        analysis_id: Optional analysis ID (defaults to global_config.patient_dir)
        
    Returns:
        EnrichedDrugPriorityOutput containing prioritized drugs
        
    Raises:
        PrioritizationError: If prioritization fails
    """
    try:
        logger.info("Starting drug prioritization process")
        
        # Validate input
        validate_prioritization_input(drugs)
        
        # Get output directory and analysis ID
        output_dir = output_dir or PROCESSED_PATHWAYS_DIR
        analysis_id = analysis_id or get_analysis_id()
        
        # Create prioritized drugs list
        prioritized_drugs = []
        
        for drug in drugs.drugs:
            try:
                # Calculate priority score
                priority_score = calculate_priority_score(drug)
                
                # Create prioritized drug object
                prioritized_drug = EnrichedDrugPriority(
                    drug_id=drug.drug_id,
                    name=drug.name,
                    pathway_id=drug.pathway_id,
                    pathway_name=drug.pathway_name,
                    drug_class=drug.drug_class,
                    target=drug.target,
                    efficacy=drug.efficacy,
                    brite=drug.brite,
                    approved=drug.approved,
                    adv_reactions=drug.adv_reactions,
                    route=drug.route,
                    matching_status=drug.matching_status,
                    LLM_Match_Reason=drug.LLM_Match_Reason,
                    deg_match_status=drug.deg_match_status,
                    priority_status=priority_score
                )
                
                prioritized_drugs.append(prioritized_drug)
                
            except Exception as e:
                logger.error(f"Error prioritizing drug {getattr(drug, 'name', 'unknown')}: {e}")
                # Continue processing other drugs
                continue
        
        if not prioritized_drugs:
            raise PrioritizationError("No drugs were successfully prioritized")
        
        # Sort by priority score (highest first)
        sorted_drugs = sorted(prioritized_drugs, key=lambda d: d.priority_status, reverse=True)
        
        # Create DataFrame for saving
        drugs_df = pd.DataFrame([d.model_dump() for d in sorted_drugs])
        
        # Create output directory and save file
        output_path = create_output_directory(output_dir, analysis_id)
        filename = generate_filename(drugs_df, analysis_id)
        saved_file_path = save_prioritized_drugs(drugs_df, output_path, filename)
        
        # Create summary
        summary = create_prioritization_summary(sorted_drugs)
        logger.info(f"Prioritization completed successfully: {summary}")
        
        return EnrichedDrugPriorityOutput(drugs=sorted_drugs)
        
    except PrioritizationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in drug prioritization: {e}")
        raise PrioritizationError(f"Drug prioritization failed: {e}")


def get_prioritization_statistics(drugs: EnrichedDrugPriorityOutput) -> Dict[str, Any]:
    """
    Get detailed statistics about prioritization results.
    
    Args:
        drugs: Prioritized drugs output
        
    Returns:
        Dictionary containing detailed statistics
    """
    try:
        if not drugs or not drugs.drugs:
            return {"error": "No prioritized drugs provided"}
        
        df = pd.DataFrame([d.model_dump() for d in drugs.drugs])
        
        stats = {
            "total_drugs": len(drugs.drugs),
            "priority_levels": {
                "high_priority": len(df[df['priority_status'] >= 4]),
                "medium_priority": len(df[df['priority_status'].between(2, 3)]),
                "low_priority": len(df[df['priority_status'] <= 1])
            },
            "approval_status": {
                "approved": len(df[df['approved'] == True]),
                "not_approved": len(df[df['approved'] == False])
            },
            "matching_status": {
                "deg_matched": len(df[df['deg_match_status'] == True]),
                "condition_matched": len(df[df['matching_status'] == True]),
                "both_matched": len(df[(df['deg_match_status'] == True) & (df['matching_status'] == True)])
            },
            "top_drugs": df.head(5)[['name', 'priority_status', 'approved', 'deg_match_status', 'matching_status']].to_dict('records')
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating prioritization statistics: {e}")
        return {"error": str(e)}