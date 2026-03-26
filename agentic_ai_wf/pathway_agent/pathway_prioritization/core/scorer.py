# src/pathway_prioritization/core/scorer.py
import re
import logging
from typing import Dict, Any, Tuple
from ..models import PathwayData, PathwayScore

logger = logging.getLogger(__name__)

class PathwayScorer:
    """Handles pathway scoring response parsing and validation"""
    
    @staticmethod
    def parse_single_pathway_response(response: str, pathway: PathwayData) -> PathwayScore:
        """Parse LLM response for a single pathway"""
        try:
            # Extract score
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 50.0
            
            # Extract confidence
            confidence_match = re.search(r'Confidence:\s*(High|Medium|Low)', response, re.IGNORECASE)
            confidence = confidence_match.group(1).title() if confidence_match else "Medium"
            
            # Extract justification
            justification_match = re.search(
                r'Justification:\s*(.+?)(?=Disease_Category|\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
            justification = justification_match.group(1).strip() if justification_match else "No detailed justification provided"
            
            # Extract LLM-generated fields
            disease_category_match = re.search(r'Disease_Category:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            disease_category = disease_category_match.group(1).strip() if disease_category_match else ""
            
            disease_subcategory_match = re.search(r'Disease_Subcategory:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            disease_subcategory = disease_subcategory_match.group(1).strip() if disease_subcategory_match else ""
            
            cellular_component_match = re.search(r'Cellular_Component:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            cellular_component = cellular_component_match.group(1).strip() if cellular_component_match else ""
            
            subcellular_element_match = re.search(r'Subcellular_Element:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            subcellular_element = subcellular_element_match.group(1).strip() if subcellular_element_match else ""
            
            # Update pathway data with LLM-generated fields
            updated_pathway = PathwayData(
                db_id=pathway.db_id,
                pathway_source=pathway.pathway_source,
                pathway_id=pathway.pathway_id,
                pathway_name=pathway.pathway_name,
                number_of_genes=pathway.number_of_genes,
                number_of_genes_in_background=pathway.number_of_genes_in_background,
                input_genes=pathway.input_genes,
                pathway_genes=pathway.pathway_genes,
                p_value=pathway.p_value,
                fdr=pathway.fdr,
                regulation=pathway.regulation,
                clinical_relevance=pathway.clinical_relevance,
                functional_relevance=pathway.functional_relevance,
                hit_score=pathway.hit_score,
                ontology_source=pathway.ontology_source,
                relation=pathway.relation,
                main_class=pathway.main_class,
                sub_class=pathway.sub_class,
                disease_category=disease_category,
                disease_subcategory=disease_subcategory,
                cellular_component=cellular_component,
                subcellular_element=subcellular_element,
                references=pathway.references,
                audit_log=pathway.audit_log
            )
            
            # Ensure score is within valid range
            score = max(0, min(100, score))
            
            return PathwayScore(
                pathway_data=updated_pathway,
                llm_score=score,
                score_justification=justification,
                confidence_level=confidence
            )
            
        except Exception as e:
            logger.warning(f"Error parsing single pathway response: {e}")
            return PathwayScore(
                pathway_data=pathway,
                llm_score=50.0,
                score_justification="Parsing error - assigned default score",
                confidence_level="Medium"
            )