# ----------------------------------------------------------------------
# DISEASE SEVERITY ASSESSMENT                                            
# ----------------------------------------------------------------------
import os
from openai import OpenAI
from typing import Dict, List

from decouple import config
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

def llm_calculate_disease_severity(confidence_level_counts: Dict[str, int], disease_name: str = None, top_genes: List[str] = None) -> str:
    """
    Use LLM to calculate disease severity based on DEG confidence levels and disease context.
    
    Args:
        confidence_level_counts: Dictionary with counts of High, Medium, Low confidence DEGs
        disease_name: Name of the disease for context
        top_genes: List of top differentially expressed genes for additional context
        
    Returns:
        str: Severity level - "HIGH", "MODERATE", "MILD", or "LOW"
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return "MODERATE"  # Fallback severity
        
        client = OpenAI(api_key=api_key)
        
        # Prepare the prompt for severity assessment
        high_count = confidence_level_counts.get('High', 0)
        medium_count = confidence_level_counts.get('Medium', 0)
        low_count = confidence_level_counts.get('Low', 0)
        total_genes = high_count + medium_count + low_count
        
        disease_context = f" in the context of {disease_name}" if disease_name else ""
        genes_context = f" Top genes include: {', '.join(top_genes[:5])}" if top_genes else ""
        
        prompt = f"""You are a clinical bioinformatics expert analyzing transcriptomic data for disease severity assessment.

            Based on the following differentially expressed genes (DEGs) data{disease_context}:

            - High confidence DEGs: {high_count}
            - Medium confidence DEGs: {medium_count} 
            - Low confidence DEGs: {low_count}
            - Total significant DEGs: {total_genes}{genes_context}

            Analyze this data and determine the likely disease severity level. Consider:
            1. The number and proportion of high-confidence biomarkers
            2. The overall number of significant DEGs
            3. The disease context and typical gene expression patterns
            4. The potential clinical impact of the observed changes

            Respond with ONLY one of these four severity levels:
            - "HIGH" - for severe disease activity with many high-confidence biomarkers
            - "MODERATE" - for moderate disease activity with mixed confidence levels
            - "MILD" - for mild disease activity with mostly low-confidence changes
            - "LOW" - for minimal disease activity with few significant changes

            Provide only the severity level (HIGH/MODERATE/MILD/LOW) without any additional text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # Validate the response
        valid_severities = ["HIGH", "MODERATE", "MILD", "LOW"]
        if result in valid_severities:
            print(f"LLM severity assessment: {result}")
            return result
        else:
            print(f"Invalid LLM severity response: {result}, using fallback")
            return "MODERATE"
            
    except Exception as e:
        print(f"LLM severity calculation failed: {str(e)}")
        # Fallback severity calculation based on confidence levels
        if total_genes == 0:
            return "LOW"
        
        high_ratio = high_count / total_genes if total_genes > 0 else 0
        if high_ratio >= 0.6:
            return "HIGH"
        elif high_ratio >= 0.3:
            return "MODERATE"
        elif total_genes >= 50:
            return "MILD"
        else:
            return "LOW"
