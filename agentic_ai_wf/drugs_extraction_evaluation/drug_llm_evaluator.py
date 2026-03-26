"""
LLM-Based Drug Clinical Evaluation System

This module provides comprehensive LLM-based evaluation of drugs for clinical relevance,
including pathway analysis, drug interaction checking.
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass
from decouple import config
from pydantic import BaseModel, Field, ValidationError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class DrugEvaluationResult:
    """Result of LLM drug evaluation"""
    drug_id: str
    drug_name: str
    priority_rank: int
    llm_score: float
    score_justification: str
    clinical_relevance: str
    pathway_relevance: str
    interaction_warnings: List[str]
    contraindications: List[str]
    recommendation: str


class MolecularAnalysisResult(BaseModel):
    """Pydantic model for structured molecular analysis results."""
    drug_id: str = Field(description="KEGG drug identifier")
    drug_name: str = Field(description="Name of the drug")
    molecular_evidence_score: float = Field(
        ge=0, le=100, 
        description="Score from 0-100 based on molecular evidence strength"
    )
    evidence_summary: str = Field(
        max_length=2000,
        description="Brief factual summary of molecular evidence"
    )
    pathway_association_strength: str = Field(
        pattern="^(STRONG|MODERATE|WEAK)$",
        description="Strength of pathway association based on gene overlap"
    )
    target_mechanism_clarity: str = Field(
        pattern="^(CLEAR|PARTIAL|UNCLEAR)$",
        description="Clarity of target mechanism based on available data"
    )
    expression_pattern_match: str = Field(
        pattern="^(CONSISTENT|INCONSISTENT|INSUFFICIENT_DATA)$",
        description="Match between drug targets and patient expression patterns"
    )
    literature_support_level: str = Field(
        pattern="^(HIGH|MEDIUM|LOW)$",
        description="Level of literature support based on approval status"
    )
    molecular_rationale: str = Field(
        max_length=2000,
        description="Scientific basis for pathway association"
    )


class BatchAnalysisResponse(BaseModel):
    """Pydantic model for batch analysis response."""
    analysis_results: List[MolecularAnalysisResult] = Field(
        description="List of molecular analysis results for each drug"
    )
    batch_summary: str = Field(
        description="Brief summary of the batch analysis"
    )


class DrugLLMEvaluator:
    """
    Enhanced LLM-based drug evaluator with structured outputs and proper validation.
    
    Features:
    - Structured outputs with Pydantic validation
    - Configurable batch processing
    - Comprehensive error handling
    - OpenAI tracing integration
    - Production-ready logging
    """

    def __init__(self, 
                 model: str = "gpt-4.1-mini-2025-04-14",
                 batch_size: int = 5,
                 max_drugs: Optional[int] = None):
        """
        Initialize the enhanced drug evaluator.
        
        Args:
            model: OpenAI model to use
            batch_size: Number of drugs per batch
            max_drugs: Maximum number of drugs to process (None for all)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.batch_size = batch_size
        self.max_drugs = max_drugs
        
        # Evaluation statistics
        self.stats = {
            "total_processed": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "validation_errors": 0,
            "api_errors": 0
        }
        
        logger.info(f"Initialized DrugLLMEvaluator with model: {model}, batch_size: {batch_size}")

    def create_molecular_analysis_prompt(
        self,
        drugs_batch: List[Dict],
        disease_name: str = "Unknown Disease"
    ) -> str:
        """Create evidence-based molecular analysis prompt."""
        
        # Group drugs by pathway for better analysis
        pathway_drug_groups = {}
        for drug in drugs_batch:
            pathway_id = drug.get('pathway_id', '')
            if pathway_id not in pathway_drug_groups:
                pathway_drug_groups[pathway_id] = {
                    'pathway_name': drug.get('pathway_name', ''),
                    'priority_rank': drug.get('priority_rank', ''),
                    'drugs': []
                }
            pathway_drug_groups[pathway_id]['drugs'].append(drug)

        # Create molecular evidence summary
        molecular_evidence = []
        for pathway_id, group in pathway_drug_groups.items():
            evidence = f"""
            PATHWAY: {group['pathway_name']} (ID: {pathway_id})
            - Priority Rank: {group['priority_rank']}
            - Drug Count: {len(group['drugs'])}
            """
            molecular_evidence.append(evidence)
        
        prompt = f"""
        MOLECULAR EVIDENCE ANALYSIS FOR {disease_name.upper()}
        
        CONTEXT: Transcriptomic analysis pipeline: Cohort Data → DEG Analysis → Gene Prioritization → Pathway Enrichment → Drug-Pathway Mapping
        
        PATHWAY EVIDENCE:
        {chr(10).join(molecular_evidence)}
        
        DRUGS TO ANALYZE ({len(drugs_batch)} drugs):
        """
        
        # Add drug information with molecular evidence factors
        for i, drug in enumerate(drugs_batch, 1):
            evidence_factors = self._calculate_evidence_factors(drug)
            
            drug_info = f"""
            Drug {i}: {drug.get('drug_name', 'Unknown')} (KEGG: {drug.get('drug_id', 'N/A')})
            - Target Genes: {drug.get('target_genes', 'Unknown')}
            - Classes: {drug.get('drug_classes', 'Unknown')}
            - Evidence: {'; '.join(evidence_factors) if evidence_factors else 'Limited'}
            - Gene Overlap: {drug.get('gene_overlap', 'None')}
            - Expression Data: {'Available' if drug.get('patient_log2fc') else 'None'}
            """
            prompt += drug_info

        prompt += f"""
            ANALYSIS REQUIREMENTS:
            Provide factual molecular evidence analysis only. NO clinical recommendations.
            
            Focus on:
            1. Molecular pathway associations
            2. Gene expression patterns  
            3. Drug-target relationships
            4. Known mechanisms of action

            IMPORTANT SCORING RULES:
            - If drug contains "topical", "cream", "ointment", "gel", "otc", "over the counter", 
                "panadol", "vitamin", or "acetaminophen" in name or drug_classes: 
                SET molecular_evidence_score to maximum 15 points
            - These are not suitable for systemic treatment and should receive minimal scores
            
            Return a JSON response with the following structure for each drug:
            {{
                "analysis_results": [
                    {{
                        "drug_id": "drug_kegg_id",
                        "drug_name": "drug_name",
                        "molecular_evidence_score": 0-100,
                        "evidence_summary": "Factual summary (max 100 words)",
                        "pathway_association_strength": "STRONG|MODERATE|WEAK",
                        "target_mechanism_clarity": "CLEAR|PARTIAL|UNCLEAR", 
                        "expression_pattern_match": "CONSISTENT|INCONSISTENT|INSUFFICIENT_DATA",
                        "literature_support_level": "HIGH|MEDIUM|LOW",
                        "molecular_rationale": "Scientific basis (max 150 words)"
                    }}
                ],
                "batch_summary": "Brief summary of batch analysis"
            }}
            
            Base scoring on:
            - Gene overlap (40 points max)
            - Expression data (30 points max) 
            - FDA approval (20 points max)
            - Mechanism detail (10 points max)
            """
            
        return prompt

    def _calculate_evidence_factors(self, drug: Dict) -> List[str]:
        """Calculate evidence factors for a drug."""
        factors = []
        
        # Gene overlap
        gene_overlap = drug.get('gene_overlap', '')
        if gene_overlap and gene_overlap != 'None':
            overlap_count = len(gene_overlap.split(',')) if ',' in gene_overlap else 1
            factors.append(f"Gene overlap: {overlap_count} genes")
        
        # Expression data
        if drug.get('patient_log2fc', '') and 'log2FC' in drug.get('patient_log2fc', ''):
            factors.append("Expression data available")
        
        # FDA approval
        if drug.get('fda_approved_status', '') == 'Approved':
            factors.append("FDA approved")
            
        # Mechanism detail
        mechanism = drug.get('mechanism_of_action', '')
        if mechanism and mechanism != 'Unknown' and len(mechanism) > 20:
            factors.append("Detailed mechanism")
            
        return factors

    async def evaluate_drugs_molecular_evidence(
        self,
        drugs_data: List[Dict],
        disease_name: str = "Unknown Disease"
    ) -> List[Dict]:
        """
        Evaluate drugs using structured molecular evidence analysis.
        
        Args:
            drugs_data: List of drug dictionaries
            disease_name: Disease context
            
        Returns:
            List of molecular analysis results
        """
        logger.info(f"Starting molecular evidence analysis for {len(drugs_data)} drugs")
        
        # Apply max_drugs limit if specified
        if self.max_drugs and len(drugs_data) > self.max_drugs:
            drugs_data = drugs_data[:self.max_drugs]
            logger.info(f"Limited analysis to {self.max_drugs} drugs")
        
        all_results = []
        total_batches = (len(drugs_data) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(drugs_data), self.batch_size):
            batch_drugs = drugs_data[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_drugs)} drugs)")
            
            try:
                # Create molecular analysis prompt
                prompt = self.create_molecular_analysis_prompt(
                    batch_drugs, 
                    disease_name
                )
                
                # Get structured analysis using OpenAI SDK
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a molecular biology expert analyzing drug-pathway associations for research purposes. Provide only factual, evidence-based analysis."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format=BatchAnalysisResponse,
                    temperature=0.1,
                    max_tokens=4000
                )
                
                # Extract parsed results
                parsed_response = response.choices[0].message.parsed
                
                if parsed_response and parsed_response.analysis_results:
                    batch_results = [result.model_dump() for result in parsed_response.analysis_results]
                    all_results.extend(batch_results)
                    self.stats["successful_batches"] += 1
                    logger.info(f"Successfully processed batch {batch_num} - {len(batch_results)} results")
                else:
                    logger.warning(f"No results in batch {batch_num}")
                    self._add_fallback_results(batch_drugs, all_results)
                    
            except ValidationError as e:
                logger.error(f"Validation error in batch {batch_num}: {e}")
                self.stats["validation_errors"] += 1
                self._add_fallback_results(batch_drugs, all_results)
                
            except Exception as e:
                logger.error(f"API error in batch {batch_num}: {e}")
                self.stats["api_errors"] += 1
                self._add_fallback_results(batch_drugs, all_results)
                
            finally:
                self.stats["total_processed"] += len(batch_drugs)
        
        # Log final statistics
        self._log_final_stats()
        
        logger.info(f"Molecular analysis completed. {len(all_results)} results generated")
        return all_results

    def _add_fallback_results(self, batch_drugs: List[Dict], all_results: List[Dict]):
        """Add fallback results when LLM analysis fails."""
        for drug in batch_drugs:
            fallback_result = self._create_fallback_analysis(drug)
            all_results.append(fallback_result)
        self.stats["failed_batches"] += 1

    def _create_fallback_analysis(self, drug: Dict) -> Dict:
        """Create fallback molecular analysis when LLM fails."""
        # Calculate basic molecular evidence score
        evidence_score = 0
        
        # Gene overlap (40 points max)
        gene_overlap = drug.get('gene_overlap', '')
        if gene_overlap and gene_overlap != 'None':
            overlap_count = len(gene_overlap.split(',')) if ',' in gene_overlap else 1
            evidence_score += min(40, overlap_count * 10)
        
        # Expression data (30 points max)
        if drug.get('patient_log2fc', '') and 'log2FC' in drug.get('patient_log2fc', ''):
            evidence_score += 30
        
        # FDA approval (20 points max)
        if drug.get('fda_approved_status', '') == 'Approved':
            evidence_score += 20
        
        # Mechanism clarity (10 points max)
        mechanism = drug.get('mechanism_of_action', '')
        if mechanism and len(mechanism) > 50:
            evidence_score += 10
        
        return {
            "drug_id": drug.get('drug_id', ''),
            "drug_name": drug.get('drug_name', ''),
            "molecular_evidence_score": evidence_score,
            "evidence_summary": "Computed from available molecular and pathway data",
            "pathway_association_strength": "MODERATE" if evidence_score > 40 else "WEAK",
            "target_mechanism_clarity": "PARTIAL" if mechanism and len(mechanism) > 50 else "UNCLEAR",
            "expression_pattern_match": "CONSISTENT" if 'log2FC' in drug.get('patient_log2fc', '') else "INSUFFICIENT_DATA",
            "literature_support_level": "MEDIUM" if drug.get('fda_approved_status') == 'Approved' else "LOW",
            "molecular_rationale": f"Evidence based on pathway association and available molecular data"
        }

    def _log_final_stats(self):
        """Log final processing statistics."""
        logger.info("=== Molecular Analysis Statistics ===")
        for key, value in self.stats.items():
            logger.info(f"  {key}: {value}")
        
        if self.stats["total_processed"] > 0:
            success_rate = (self.stats["successful_batches"] / 
                          (self.stats["successful_batches"] + self.stats["failed_batches"])) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")

    async def evaluate_drugs_molecular_evidence_from_df(
        self,
        df: pd.DataFrame,
        disease_name: str = "Unknown Disease",
        save_results: bool = True,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Wrapper method to evaluate drugs from DataFrame.
        """
        logger.info(f"Converting DataFrame to drug list for molecular analysis ({len(df)} drugs)")
        
        # Convert DataFrame to list of dictionaries
        drugs_data = df.to_dict('records')
        
        # Perform molecular evidence analysis
        results = await self.evaluate_drugs_molecular_evidence(
            drugs_data=drugs_data,
            disease_name=disease_name
        )
        
        # Convert results back to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        if not results_df.empty and 'drug_id' in df.columns and 'drug_id' in results_df.columns:
            final_df = pd.merge(
                df, 
                results_df, 
                on='drug_id', 
                how='left',
                suffixes=('', '_analysis')
            )
        else:
            logger.warning("Could not merge results with original DataFrame")
            final_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        # Save results if requested
        if save_results and output_file:
            try:
                final_df.to_csv(output_file, index=False)
                logger.info(f"Results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
        
        logger.info(f"Analysis completed. Final dataset shape: {final_df.shape}")
        return final_df


# Convenience function for easy integration
async def evaluate_drugs_with_llm(
    df: pd.DataFrame,
    disease_name: str = "Unknown Disease",
    save_results: bool = True,
    output_file: Optional[str] = None,
    max_drugs: Optional[int] = None,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Convenience function to evaluate drugs using enhanced LLM analysis.
    
    Args:
        df: DataFrame with drug data
        disease_name: Patient's primary disease name
        save_results: Whether to save results
        output_file: Output file path
        max_drugs: Maximum number of drugs to process (None for all)
        batch_size: Number of drugs per batch
        
    Returns:
        DataFrame with molecular analysis results
    """
    evaluator = DrugLLMEvaluator(
        batch_size=batch_size,
        max_drugs=max_drugs
    )
    
    return await evaluator.evaluate_drugs_molecular_evidence_from_df(
        df=df,
        disease_name=disease_name,
        save_results=save_results,
        output_file=output_file
    )
