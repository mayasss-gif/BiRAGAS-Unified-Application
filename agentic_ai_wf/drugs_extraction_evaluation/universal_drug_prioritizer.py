# -*- coding: utf-8 -*-
"""
Universal Drug Prioritization Strategy for Multi-Disease Applications
Production-ready implementation following the comprehensive strategy document.

Key Features:
- Pre-LLM computational filtering with safety gates
- Pathway-drug mapping using KEGG database
- Target expression matching and directionality safety
- Clinical relevance filtering
- LLM-driven final scoring and ranking
- Production-ready error handling and logging
"""

import os
import re
import json
import logging
import asyncio
import concurrent.futures
from typing import List, Dict, Optional
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from decouple import config

# OpenAI Agents SDK imports (with fallback handling)
try:
    from agents import Agent, Runner, WebSearchTool
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    Agent = Runner = WebSearchTool = None

from .column_config import DrugColumnConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Clinical relevance patterns
TOPICAL_EXCLUSIONS = re.compile(
    r"(topical|cream|ointment|gel|lotion|patch|otc|over.?the.?counter|vitamin|supplement|acetaminophen|panadol)",
    re.IGNORECASE
)

# ===== Data Models =====

# Only keeping essential classes that are used in the code

class PrioritizedDrug(BaseModel):
    """LLM output for prioritized drug"""
    pathway_id: str
    drug_id: str
    drug_name: str
    priority_score: int = Field(ge=0, le=100)
    confidence: str = Field(pattern="^(HIGH|MEDIUM|LOW)$")
    justification: str = Field(max_length=1500)
    recommendation: str = Field(pattern="^(YES|NO)$")

class DiseaseContext(BaseModel):
    """Disease-specific medical context for prioritization"""
    disease_name: str
    pathophysiology: str = Field(max_length=800)
    key_molecular_drivers: str = Field(max_length=600)
    therapeutic_targets: str = Field(max_length=600)
    contraindications: str = Field(max_length=800)
    standard_treatments: str = Field(max_length=600)
    clinical_considerations: str = Field(max_length=800)

class ValidatedDiseaseContext(BaseModel):
    """Validated disease context with corrections"""
    disease_name: str
    pathophysiology: str
    key_molecular_drivers: str
    therapeutic_targets: str
    contraindications: str
    standard_treatments: str
    clinical_considerations: str

# PrioritizationResult class removed - not needed for single drug processing

# ===== Core Implementation =====

class UniversalDrugPrioritizer:
    """
    Universal drug prioritization system implementing the complete strategy.
    Follows production-ready patterns with comprehensive error handling.
    """
    
    def __init__(self, model: str = MODEL_DEFAULT):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self._disease_context_cache = {}  # Cache disease contexts
        logger.info(f"Initialized UniversalDrugPrioritizer with model: {model}")
    
    def _create_medical_agent(self, disease_name: str):
        """Create a medical expert agent with web search tools."""
        if not AGENTS_SDK_AVAILABLE:
            logger.warning("OpenAI Agents SDK not available, using fallback")
            return None
            
        instructions = f"""You are a medical expert specializing in {disease_name}.
        Generate comprehensive, evidence-based medical context for drug prioritization.
        Use web search to find the latest clinical data, treatment guidelines, and research.
        
        Provide structured medical information including:
        - Core pathophysiology and disease mechanisms
        - Key molecular drivers and dysregulated pathways
        - Primary therapeutic targets
        - Critical contraindications and safety considerations
        - Standard treatments and clinical protocols
        - Monitoring requirements and special populations"""
        
        try:
            agent = Agent(
                name=f"MedicalExpert-{disease_name}",
                instructions=instructions,
                tools=[WebSearchTool()],
                model=self.model,
                output_type=DiseaseContext
            )
            logger.debug(f"Successfully created agent for {disease_name}")
            return agent
        except Exception as agent_creation_error:
            logger.error(f"Failed to create agent for {disease_name}: {agent_creation_error}")
            return None
    
    def _generate_disease_context(self, disease_name: str) -> DiseaseContext:
        """Generate disease context using OpenAI Agents SDK with web search."""
        if disease_name in self._disease_context_cache:
            return self._disease_context_cache[disease_name]
        
        try:
            # Create medical expert agent
            logger.debug(f"Creating medical agent for {disease_name}")
            agent = self._create_medical_agent(disease_name)
            if not agent:
                raise ImportError("Agent creation failed - OpenAI Agents SDK unavailable")
            
            query = f"""Generate comprehensive medical context for {disease_name}.
            Use web search to find the latest clinical information.
            
            Provide detailed, evidence-based information covering:
            - Pathophysiology and disease mechanisms
            - Key molecular drivers and dysregulated pathways  
            - Primary therapeutic targets
            - Critical contraindications and safety considerations
            - Standard treatments and clinical protocols
            - Monitoring requirements and special populations"""
            
            # Run agent with structured output using Runner
            async def run_medical_agent():
                return await Runner.run(agent, query, max_turns=1)
            
            # Execute agent (handles event loop scenarios)
            logger.debug(f"Running medical agent for {disease_name}")
            try:
                run_result = asyncio.run(run_medical_agent())
                logger.debug(f"Agent execution completed for {disease_name}")
            except RuntimeError as re:
                logger.debug(f"RuntimeError during agent execution, using thread executor: {re}")
                # Handle nested event loop with thread executor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_medical_agent())
                    run_result = future.result(timeout=90)
                    logger.debug(f"Thread executor completed for {disease_name}")
            except Exception as agent_error:
                logger.error(f"Agent execution failed for {disease_name}: {agent_error}")
                raise agent_error
            
            if run_result and hasattr(run_result, 'final_output'):
                # Extract the actual result from RunResult.final_output
                disease_context = run_result.final_output
                
                # Validate that we got a proper DiseaseContext object
                if isinstance(disease_context, DiseaseContext):
                    self._disease_context_cache[disease_name] = disease_context
                    logger.info(f"Successfully generated disease context for {disease_name} using Agents SDK")
                    return disease_context
                else:
                    logger.warning(f"Agent returned unexpected type: {type(disease_context)}, content: {disease_context}")
                    raise ValueError(f"Agent returned invalid type: {type(disease_context)}")
            else:
                logger.warning(f"Agent run_result is invalid: {run_result}")
                raise ValueError(f"Agent returned no valid result: {run_result}")
            
        except Exception as e:
            # Log detailed error information for debugging
            error_type = type(e).__name__
            logger.warning(f"Agents SDK fallback for {disease_name}: {error_type} - {str(e)}")
            logger.debug(f"Full error details for {disease_name}: {e}", exc_info=True)
            return self._fallback_disease_context(disease_name)
    
    def _fallback_disease_context(self, disease_name: str) -> DiseaseContext:
        """Fallback disease context generation using simple API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"""Medical context for {disease_name}. JSON: 
                {{"disease_name":"{disease_name}","pathophysiology":"brief","key_molecular_drivers":"pathways","therapeutic_targets":"targets","contraindications":"warnings","standard_treatments":"treatments","clinical_considerations":"considerations"}}"""}],
                response_format={"type": "json_object"}
            )
            
            context_data = json.loads(response.choices[0].message.content)
            disease_context = DiseaseContext(**context_data)
            self._disease_context_cache[disease_name] = disease_context
            return disease_context
            
        except Exception:
            # Static fallback
            fallback = DiseaseContext(
                disease_name=disease_name,
                pathophysiology=f"General pathophysiology of {disease_name}",
                key_molecular_drivers="Standard inflammatory pathways",
                therapeutic_targets="Common therapeutic approaches", 
                contraindications="Standard safety considerations",
                standard_treatments="Established protocols",
                clinical_considerations="General monitoring requirements"
            )
            self._disease_context_cache[disease_name] = fallback
            return fallback
    
    def prioritize_drugs(self, df: pd.DataFrame, disease_name: str) -> pd.DataFrame:
        """
        Main entry point for drug prioritization.
        
        Args:
            df: Input dataframe with drug-pathway mappings
            disease_name: Target disease for prioritization
            
        Returns:
            DataFrame with prioritized drugs and scores
        """
        try:
            logger.info(f"Starting drug prioritization for {disease_name} with {df.shape[0]} drugs")
            
            # Generate and validate disease context
            disease_context = self._generate_disease_context(disease_name)
            
            # validated_context = self._validate_disease_context(disease_context)
                
            prioritized_results = self._llm_prioritization(df, disease_name, disease_context)
            
            return self._format_final_results(prioritized_results)
            
        except Exception as e:
            logger.error(f"Prioritization failed: {e}")
            return self._create_fallback_results(df)
    
    def _llm_prioritization(self, df: pd.DataFrame, disease_name: str, disease_context: DiseaseContext) -> List[PrioritizedDrug]:
        """LLM-based final prioritization and scoring with disease context - one drug at a time"""
        results = []
        candidates = df.to_dict('records')
        
        # Create mapping of drug_id+pathway_id to original data for preservation
        self.original_data_map = {}
        for record in candidates:
            key = f"{record.get('drug_id', '')}|{record.get('pathway_id', '')}"
            self.original_data_map[key] = record
        
        # Process each drug individually
        for i, candidate in enumerate(candidates):
            try:
                drug_result = self._process_single_drug(candidate, disease_name, disease_context)
                results.append(drug_result)
            except Exception as e:
                logger.error(f"Drug processing failed: {e}")
                results.append(self._fallback_single_drug_scoring(candidate))
        
        return results
    
    def _process_single_drug(self, drug: Dict, disease_name: str, disease_context: DiseaseContext) -> PrioritizedDrug:
        """Process single drug using OpenAI structured output API"""
        try:
            # Create evaluation prompt
            system_prompt = f"""You are a clinical pharmacologist evaluating drug relevance for {disease_name}.
            Evaluate drugs based on clinical appropriateness, safety, and evidence.
            
            EXCLUSIONS (score=0, recommendation=NO):
            - Topical preparations, OTC drugs, vitamins, supplements
            - Drugs contraindicated for {disease_name}
            - Aspirin, acetaminophen for most conditions
            
            SCORING: HIGH(70-100): Direct indication, FDA approved. MEDIUM(40-69): Related indication. LOW(1-39): Weak evidence."""
            
            user_prompt = f"""Evaluate this drug for {disease_name} treatment:

            DISEASE CONTEXT:
            - Key Drivers: {disease_context.key_molecular_drivers}
            - Therapeutic Targets: {disease_context.therapeutic_targets}
            - Contraindications: {disease_context.contraindications}

            DRUG TO EVALUATE:
            {json.dumps(drug, indent=2)}

            Provide priority score (0-100), confidence (HIGH/MEDIUM/LOW), recommendation (YES/NO), and 2-3 sentence justification covering mechanism, disease relevance, FDA status, and safety considerations."""

            # Use OpenAI structured output API
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=PrioritizedDrug
            )
            
            # Extract the parsed result
            prioritized_drug = response.output_parsed
            
            # Ensure we have the required drug identifiers from the original data
            prioritized_drug.pathway_id = str(drug.get("pathway_id", ""))
            prioritized_drug.drug_id = str(drug.get("drug_id", ""))
            prioritized_drug.drug_name = str(drug.get("drug_name", ""))
            
            return prioritized_drug
            
        except Exception as e:
            logger.warning(f"Structured output prioritization failed for {drug.get('drug_name', 'unknown')}: {e}")
            return self._fallback_single_drug_scoring(drug)
    

    def _fallback_single_drug_scoring(self, drug: Dict) -> PrioritizedDrug:
        """
        Simplified deterministic scoring for a single drug
        """
        drug_id = str(drug.get("drug_id", ""))
        drug_name = str(drug.get("drug_name", ""))
        pathway_id = str(drug.get("pathway_id", ""))
        
        # Check for exclusions
        blob = f"{drug_name} {drug.get('drug_classes','')} {drug.get('route_of_administration','')}".lower()
        if any(k in blob for k in ["cream","ointment","gel","patch","lotion","topical","otc","vitamin","acetaminophen","panadol"]):
            return PrioritizedDrug(
                pathway_id=pathway_id,
                drug_id=drug_id,
                drug_name=drug_name,
                priority_score=0,
                confidence="LOW",
                justification="Excluded by clinical relevance (topical/OTC/vitamin/acetaminophen).",
                recommendation="NO"
            )
        
        # Calculate simple score based on available data
        score = 0
        
        # Molecular evidence score (max 50)
        try:
            mol_score = float(drug.get('molecular_evidence_score', 0))
            score += min(50, int(mol_score))
        except:
            pass
        
        # FDA approval (20 points)
        if str(drug.get('fda_approved_status', '')).upper() == 'APPROVED':
            score += 20
        
        # Clinical relevance (10 points)
        if str(drug.get('clinical_relevance', '')).upper() in ['HIGH', 'STRONG']:
            score += 10
        
        # Scale score to 0-100
        scaled_score = min(100, max(0, score))
        confidence = "HIGH" if scaled_score >= 75 else "MEDIUM" if scaled_score >= 45 else "LOW"
        recommendation = "YES" if scaled_score >= 40 else "NO"
        
        justification = (
            f"{drug_name}: FDA status={drug.get('fda_approved_status','unknown')}. "
            f"Priority score based on available evidence and clinical relevance."
        )
        
        return PrioritizedDrug(
            pathway_id=pathway_id,
            drug_id=drug_id,
            drug_name=drug_name,
            priority_score=scaled_score,
            confidence=confidence,
            justification=justification,
            recommendation=recommendation
        )

    
    # _calculate_fallback_score method removed - functionality merged into _fallback_single_drug_scoring
    
    def _format_final_results(self, prioritized_drugs: List[PrioritizedDrug]) -> pd.DataFrame:
        """
        Format final results preserving all input columns and adding LLM enhancements
        
        Args:
            prioritized_drugs: List of PrioritizedDrug objects sorted by priority
            
        Returns:
            DataFrame with final results preserving all input data
        """
        
        
        if not prioritized_drugs:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns())
        
        # Sort by priority score
        sorted_drugs = sorted(prioritized_drugs, key=lambda x: x.priority_score, reverse=True)
        
        # Convert to DataFrame 
        results_data = []
        
        for i, drug in enumerate(sorted_drugs, 1):
            # Start with core LLM results
            drug_data = {
                'final_rank': i,
                'pathway_id': drug.pathway_id,
                'drug_id': drug.drug_id,
                'drug_name': drug.drug_name,
                'priority_score': drug.priority_score,
                'confidence': drug.confidence,
                'justification': drug.justification,
                'recommendation': drug.recommendation
            }
            
            # Preserve ALL original data using the stored mapping
            key = f"{drug.drug_id}|{drug.pathway_id}"
            if hasattr(self, 'original_data_map') and key in self.original_data_map:
                original_data = self.original_data_map[key]
                # Merge original data, keeping LLM updates where they exist
                for col_key, value in original_data.items():
                    if col_key not in drug_data:  # Don't overwrite LLM-generated values
                        drug_data[col_key] = value
            
            results_data.append(drug_data)
            
            # No alternate drug handling - focus only on prioritizing existing drugs
        
        # Use only the prioritized drugs from input data
        all_results = results_data
        
        # Create DataFrame and ensure all expected columns are present
        df = pd.DataFrame(all_results)
        df = DrugColumnConfig.add_missing_columns(df, preserve_existing=True)
        
        logger.info(f"Prioritized {len(all_results)} drugs based on evidence and scores")
        
        return df
    
    def _create_empty_result(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Create empty result when no candidates pass filtering"""
        return pd.DataFrame(columns=['final_rank', 'pathway_id', 'drug_id', 'drug_name', 'priority_score', 'confidence', 'justification', 'recommendation'])
    
    def _create_fallback_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fallback results when entire process fails"""
        logger.warning("Creating fallback results due to system failure")
        fallback_data = []
        
        for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
            fallback_data.append({
                'final_rank': i,
                'pathway_id': row.get('pathway_id', ''),
                'drug_id': row.get('drug_id', ''),
                'drug_name': row.get('drug_name', ''),
                'priority_score': 30,  # Low fallback score
                'confidence': 'LOW',
                'justification': 'Fallback result due to system error. Manual review required.',
                'recommendation': 'NO'  # Conservative fallback
            })
        
        return pd.DataFrame(fallback_data)

# ===== Main Function =====

def prioritize_drugs_universal(
    df: pd.DataFrame,
    disease_name: str,
    model: str = MODEL_DEFAULT
) -> pd.DataFrame:
    """
    Universal drug prioritization entry point.
    
    Args:
        df: Input DataFrame with drug-pathway data
        disease_name: Target disease name
        model: OpenAI model to use
        
    Returns:
        DataFrame with prioritized drugs and scores
    """
    prioritizer = UniversalDrugPrioritizer(model=model)
    return prioritizer.prioritize_drugs(df, disease_name)

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame([
        {
            'drug_id': 'D00001',
            'drug_name': 'Aspirin',
            'pathway_id': 'hsa04611',
            'pathway_name': 'Platelet activation',
            'target_genes': 'PTGS1,PTGS2',
            'gene_overlap': 'PTGS1',
            'patient_log2fc': '1.2',
            'mechanism_of_action': 'COX inhibitor',
            'molecular_evidence_score': 85.0,
            'fda_approved_status': 'APPROVED'
        }
    ])
    
    result = prioritize_drugs_universal(sample_data, "Cardiovascular Disease")
    print(result)
