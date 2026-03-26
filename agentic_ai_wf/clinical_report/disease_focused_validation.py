"""
Disease-Focused Validation System
=================================

This module provides precise, disease-specific validation with structured prompts
that classify genes, pathways, and drugs with clear clinical relevance categories.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI
import os
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Import global disease context cache
try:
    from .disease_context_cache import (
        get_disease_context_cached, 
        set_disease_context_cached
    )
    GLOBAL_CACHE_AVAILABLE = True
except ImportError:
    GLOBAL_CACHE_AVAILABLE = False
    print("⚠️ Global disease context cache not available, using local cache")

logger = logging.getLogger(__name__)

# Pydantic models for structured input/output
class PathwayValidationOutput(BaseModel):
    """Output model for pathway validation"""
    pathway: str
    status: str = Field(..., description="Must be: 'Pathogenic', 'Protective', or 'Uncertain'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(..., description="Brief explanation of biological relevance")
    clinical_significance: str = Field(..., description="Brief clinical relevance statement")
class DiseaseContextOutput(BaseModel):
    """Output model for LLM-generated disease context"""
    standard_name: str = Field(..., description="Standardized disease name")
    aliases: List[str] = Field(default_factory=list, description="Common aliases and synonyms")
    key_pathways: List[str] = Field(default_factory=list, description="3-5 most relevant biological pathways")
    pathogenic_mechanisms: List[str] = Field(default_factory=list, description="3-5 key disease-promoting mechanisms")
    protective_mechanisms: List[str] = Field(default_factory=list, description="3-5 key protective/therapeutic mechanisms")
    fda_drugs: List[str] = Field(default_factory=list, description="Major FDA-approved drugs (generic names)")
    biomarkers: List[str] = Field(default_factory=list, description="Key diagnostic/prognostic biomarkers")

@dataclass
class PathwayValidationResult:
    """Structured result for pathway validation"""
    pathway: str
    status: str  # "Pathogenic", "Protective", "Uncertain"
    justification: str
    confidence: float
    evidence_sources: List[str]


class DiseaseContextValidator:
    """Disease-specific validation with structured prompts"""
    
    def __init__(self, api_key: str = None):
        try:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.llm_available = True
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.client = None
            self.llm_available = False
        # Choose a model compatible with beta.parse structured outputs
        # Falls back to env OPENAI_MODEL or a sensible default
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Use global cache if available, otherwise fall back to local cache
        if GLOBAL_CACHE_AVAILABLE:
            self.use_global_cache = True
            # Local cache only used as fallback
            self.llm_context_cache = {}
            logger.info("✅ Using global thread-safe disease context cache")
        else:
            self.use_global_cache = False
            # Cache for LLM-generated disease contexts to avoid repeated calls
            self.llm_context_cache = {}
            logger.warning("⚠️ Using local disease context cache (not thread-safe)")
        
        # Disease-specific knowledge bases
        self.disease_contexts = {
            "Systemic Lupus Erythematosus": {
                "aliases": ["SLE", "Lupus", "systemic lupus", "lupus erythematosus"],
                "key_pathways": ["interferon signaling", "B cell receptor", "complement cascade", "immune complex"],
                "pathogenic_mechanisms": ["autoantibody production", "immune complex deposition", "interferon signature"],
                "protective_mechanisms": ["immune tolerance", "regulatory T cells", "apoptosis"],
                "fda_drugs": ["belimumab", "anifrolumab", "hydroxychloroquine"],
                "biomarkers": ["anti-dsDNA (FDA-approved diagnostic)", "anti-Sm (Clinical guideline recommended)", "complement levels (Clinical use)", "interferon score (Research validated)"]
            },
            "Rheumatoid Arthritis": {
                "aliases": ["RA", "rheumatoid arthritis"],
                "key_pathways": ["TNF signaling", "IL-6 signaling", "T cell activation", "synovial inflammation"],
                "pathogenic_mechanisms": ["synovial hyperplasia", "cartilage destruction", "bone erosion"],
                "protective_mechanisms": ["immune tolerance", "anti-inflammatory responses"],
                "fda_drugs": ["methotrexate", "adalimumab", "etanercept", "rituximab"],
                "biomarkers": ["RF (FDA-approved diagnostic)", "anti-CCP (FDA-approved diagnostic)", "CRP (Clinical use)", "ESR (Clinical use)"]
            },
            "Vasculitis": {
                "aliases": ["systemic vasculitis", "ANCA-associated vasculitis", "large vessel vasculitis", "small vessel vasculitis"],
                "key_pathways": ["neutrophil activation", "complement cascade", "TNF signaling", "interferon signaling"],
                "pathogenic_mechanisms": ["vessel wall inflammation", "neutrophil infiltration", "autoantibody-mediated damage", "complement activation"],
                "protective_mechanisms": ["immunosuppression", "anti-inflammatory response", "endothelial repair"],
                "fda_drugs": ["cyclophosphamide", "rituximab", "prednisone", "azathioprine"],
                "biomarkers": ["ANCA (FDA-approved diagnostic)", "PR3-ANCA (Clinical guideline recommended)", "MPO-ANCA (Clinical guideline recommended)", "CRP (Clinical use)", "ESR (Clinical use)"]
            },
            "Cancer": {
                "aliases": ["malignancy", "tumor", "neoplasm", "carcinoma"],
                "key_pathways": ["p53 pathway", "cell cycle", "apoptosis", "angiogenesis", "metastasis"],
                "pathogenic_mechanisms": ["oncogene activation", "tumor suppressor loss", "immune evasion"],
                "protective_mechanisms": ["DNA repair", "apoptosis", "immune surveillance"],
                "fda_drugs": ["checkpoint inhibitors", "targeted therapy", "chemotherapy"],
                "biomarkers": ["tumor markers (FDA-approved diagnostic)", "ctDNA (Clinical use)", "immune infiltration (Research validated)"]
            },
            "Hypogammaglobulinemia": {
                "aliases": ["antibody deficiency", "immunodeficiency", "low immunoglobulin"],
                "key_pathways": ["B cell development", "antibody production", "class switching", "plasma cell differentiation"],
                "pathogenic_mechanisms": ["B cell deficiency", "impaired antibody production", "recurrent infections"],
                "protective_mechanisms": ["immunoglobulin replacement", "infection prophylaxis"],
                "fda_drugs": ["intravenous immunoglobulin", "subcutaneous immunoglobulin"],
                "biomarkers": ["IgG levels (Clinical use)", "IgA levels (Clinical use)", "IgM levels (Clinical use)", "B cell counts (Clinical use)"]
            }
        }
    
    def get_disease_context(self, disease_name: str) -> Dict:
        """Get disease-specific context for validation with optimized caching"""
        # First check hardcoded disease contexts (fastest)
        for standard_name, context in self.disease_contexts.items():
            if any(alias.lower() in disease_name.lower() for alias in [standard_name] + context["aliases"]):
                logger.debug(f"🎯 Found hardcoded context for: {disease_name} -> {standard_name}")
                return {**context, "standard_name": standard_name}
        
        # Check global cache first (thread-safe, persistent)
        if self.use_global_cache and GLOBAL_CACHE_AVAILABLE:
            cached_context = get_disease_context_cached(disease_name)
            if cached_context:
                logger.debug(f"🎯 Global cache HIT for disease: {disease_name}")
                return cached_context
        
        # Check local cache (fallback)
        cache_key = disease_name.lower().strip()
        if cache_key in self.llm_context_cache:
            logger.debug(f"🎯 Local cache HIT for disease: {disease_name}")
            return self.llm_context_cache[cache_key]
        
        # Generate context using LLM for unknown diseases
        if self.llm_available:
            try:
                logger.info(f"🔄 Generating NEW disease context for: {disease_name}")
                llm_context = self.llm_generate_disease_context(disease_name)
                
                # Cache in both global and local caches
                if self.use_global_cache and GLOBAL_CACHE_AVAILABLE:
                    set_disease_context_cached(disease_name, llm_context)
                    logger.info(f"💾 Cached context globally for: {disease_name}")
                else:
                    self.llm_context_cache[cache_key] = llm_context
                    logger.info(f"💾 Cached context locally for: {disease_name}")
                
                return llm_context
                
            except Exception as e:
                logger.warning(f"LLM disease context generation failed for {disease_name}: {e}")
        
        # Default context for unknown diseases (fallback)
        logger.warning(f"⚠️ Using default context for: {disease_name}")
        return {
            "standard_name": disease_name,
            "aliases": [],
            "key_pathways": [],
            "pathogenic_mechanisms": ["inflammation", "cell death", "dysfunction"],
            "protective_mechanisms": ["repair", "homeostasis", "regulation"],
            "fda_drugs": [],
            "biomarkers": []
        }
    
    def llm_generate_disease_context(self, disease_name: str) -> Dict:
        """Generate comprehensive disease context using LLM with structured output"""
        
        # NOTE: Caching is now handled in get_disease_context() method
        # This method focuses purely on LLM generation
        
        prompt = f"""You are a clinical geneticist and biomarker specialist with access to comprehensive medical databases (OMIM, ClinVar, GWAS Catalog, FDA approvals).

            DISEASE: {disease_name}

            Task: Generate comprehensive, evidence-based disease context for clinical gene validation. Base ALL information on established medical literature and databases.

            CRITICAL REQUIREMENTS:
            - ONLY include information with strong scientific evidence
            - Reference established clinical guidelines when possible
            - For biomarkers, specify if they are FDA-approved or clinically validated
            - Use conservative assessments - if uncertain, indicate limited evidence
            - Include specific gene symbols when known

            REQUIRED JSON FORMAT:
            {{
            "standard_name": "Official disease name (ICD-10/SNOMED preferred)",
            "aliases": ["Medical synonyms", "ICD codes", "Common abbreviations"],
            "key_pathways": ["Pathway names from KEGG/Reactome databases"],
            "pathogenic_mechanisms": ["Evidence-based disease mechanisms"],
            "protective_mechanisms": ["Established therapeutic mechanisms"],
            "fda_drugs": ["FDA-approved generic drug names with approval years if known"],
            "biomarkers": ["Clinically validated biomarkers with evidence level"]
            }}

            ENHANCED REQUIREMENTS:
            1. **standard_name**: Use ICD-10, SNOMED, or OMIM official nomenclature
            2. **aliases**: Include ICD codes, medical synonyms, common abbreviations (2-4 items)
            3. **key_pathways**: Reference KEGG/Reactome pathway names (3-5 pathways)
            4. **pathogenic_mechanisms**: Evidence-based mechanisms from peer-reviewed literature (3-5 items)
            5. **protective_mechanisms**: Established therapeutic targets/mechanisms (3-5 items)
            6. **fda_drugs**: Only FDA-approved drugs with generic names (3-6 items)
            7. **biomarkers**: Specify validation level:
            - "GENE_SYMBOL (FDA-approved diagnostic)"
            - "GENE_SYMBOL (Clinical guideline recommended)"
            - "GENE_SYMBOL (Research validated)"
            - "PROTEIN/METABOLITE (Clinical use)"

            EVIDENCE STANDARDS:
            - FDA-approved: Include only drugs/tests with FDA approval
            - Clinical guidelines: Reference major medical associations (AHA, ACS, etc.)
            - Research validated: Multiple peer-reviewed studies with clinical relevance
            - Database confirmed: Present in OMIM, ClinVar, GWAS Catalog

            BIOMARKER CATEGORIES:
            Include genes/proteins used for:
            - Diagnosis (FDA-approved tests)
            - Prognosis (clinical outcome prediction)
            - Treatment monitoring (therapy response)
            - Risk assessment (genetic predisposition)

            For rare diseases: State "Limited clinical data available" rather than speculating."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical geneticist and biomarker specialist with expertise in evidence-based medicine. You have access to major medical databases (OMIM, ClinVar, GWAS Catalog, FDA databases) and clinical guidelines. Provide ONLY scientifically validated information. When evidence is limited or unclear, explicitly state this rather than speculating. Prioritize FDA-approved diagnostics and established clinical guidelines."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800
            )
            
            llm_response = response.choices[0].message.content
            parsed_response = json.loads(llm_response)
            
            # Validate and structure the response
            disease_context = DiseaseContextOutput(**parsed_response)
            
            # Convert to the expected format
            result = {
                "standard_name": disease_context.standard_name,
                "aliases": disease_context.aliases,
                "key_pathways": disease_context.key_pathways,
                "pathogenic_mechanisms": disease_context.pathogenic_mechanisms,
                "protective_mechanisms": disease_context.protective_mechanisms,
                "fda_drugs": disease_context.fda_drugs,
                "biomarkers": disease_context.biomarkers
            }
            
            logger.info(f"✅ Generated comprehensive disease context for {disease_name}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for disease context: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM disease context generation failed: {e}")
            raise
    
    def build_pathway_validation_prompt(self, pathway_name: str, direction: str, 
                                      disease: str, evidence_text: str = "") -> str:
        """Build scientifically rigorous pathway validation prompt to prevent hallucination"""
        context = self.get_disease_context(disease)
        
        return f"""You are a systems biologist and clinical pathologist analyzing pathway dysregulation in {context['standard_name']}. Provide ONLY evidence-based pathway classifications using established disease pathophysiology.

                DISEASE CONTEXT:
                - Standard Name: {context['standard_name']}
                - Known Key Pathways: {', '.join(context.get('key_pathways', []))}
                - Pathogenic Mechanisms: {', '.join(context.get('pathogenic_mechanisms', []))}
                - Protective Mechanisms: {', '.join(context.get('protective_mechanisms', []))}
                - FDA-Approved Drugs: {', '.join(context.get('fda_drugs', []))}

                PATHWAY TO ANALYZE:
                - Pathway: {pathway_name}
                - Dysregulation: {direction}
                - Evidence: {evidence_text[:200] if evidence_text else "Patient transcriptomic data only"}

                CRITICAL INSTRUCTIONS:
                - Base classifications ONLY on well-documented disease pathophysiology
                - Cross-reference against known key pathways for {context['standard_name']} above
                - If uncertain about pathway role, classify as "Uncertain" rather than guessing  
                - Use conservative confidence scores unless evidence is very strong
                - Reference established pathway databases (KEGG, Reactome, WikiPathways) when possible

                CLASSIFICATION CRITERIA:

                **Pathogenic** (Use ONLY if strong evidence exists):
                - Pathway appears in known pathogenic mechanisms list above for {context['standard_name']}
                - Multiple peer-reviewed studies confirm pathway promotes disease progression
                - {direction} dysregulation is established in {context['standard_name']} pathophysiology
                - Therapeutic targets in this pathway exist (drugs targeting pathway components)
                - Clear mechanistic role in disease promotion documented

                **Protective** (Use ONLY if strong evidence exists):
                - Pathway appears in known protective mechanisms list above for {context['standard_name']}
                - {direction} dysregulation represents therapeutic/beneficial response
                - FDA-approved drugs target this pathway for {context['standard_name']} treatment
                - Multiple studies show pathway activation improves outcomes
                - Clear mechanistic role in disease resolution/protection

                **Uncertain** (Default for unclear cases):
                - Limited or conflicting evidence for {context['standard_name']}
                - Pathway not well-characterized in this specific disease
                - Contradictory studies about pathway role
                - Novel or emerging research only
                - {direction} dysregulation significance unclear

                CONFIDENCE SCORING:
                - 0.9-1.0: Pathway central to {context['standard_name']} pathophysiology, FDA drug targets
                - 0.7-0.8: Multiple large studies, clear mechanistic evidence, clinical relevance
                - 0.5-0.6: Some evidence but limited disease-specific validation
                - 0.3-0.4: Weak or conflicting evidence for this specific disease
                - 0.1-0.2: Minimal evidence, mostly speculative

                DIRECTION-SPECIFIC ANALYSIS:
                For {direction} dysregulation, consider:
                - Is this the expected direction for {context['standard_name']}?
                - Does this direction align with known disease mechanisms?
                - Are there therapeutic implications for this direction?

                REQUIRED OUTPUT FORMAT:
                {{
                "pathway": "{pathway_name}",
                "status": "Pathogenic|Protective|Uncertain",
                "confidence": 0.0-1.0,
                "justification": "Evidence-based mechanistic explanation with disease-specific context",
                "clinical_significance": "Specific clinical relevance or therapeutic implications"
                }}

                Prioritize scientific accuracy over speculation. Reference specific mechanisms when possible."""
    
    
    def llm_generate_structured_response(self, prompt: str, response_model: Any, max_retries: int = 2) -> Dict[str, Any]:
        """
        Generate structured response from LLM using Pydantic model validation
        
        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model class for validating and structuring the response
            max_retries: Maximum number of retry attempts for failed generations
            
        Returns:
            Dictionary containing the validated structured response
        
        Raises:
            ValueError: If LLM generation fails after max retries
            Exception: For other unexpected errors
        """
        if not self.llm_available or not self.client:
            logger.error("LLM client not available for structured response generation")
            # Return minimal valid response based on model fields
            return {field: getattr(response_model, "__annotations__", {}).get(field, str) for field in response_model.__fields__}
            
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Generate response with JSON output format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a clinical expert providing structured analysis. Respond with valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,  # Lower temperature for more consistent structured outputs
                    max_tokens=800
                )
                
                # Extract and parse the JSON response
                llm_response = response.choices[0].message.content
                parsed_response = json.loads(llm_response)
                
                # Validate with Pydantic model
                validated_response = response_model(**parsed_response)
                
                # Convert Pydantic model to dictionary
                result = validated_response.dict()
                logger.info(f"✅ Successfully generated structured response with model {response_model.__name__}")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error (attempt {retry_count+1}/{max_retries+1}): {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"Structured response generation error (attempt {retry_count+1}/{max_retries+1}): {e}")
                last_error = e
                
            retry_count += 1
            
        # If we've exhausted retries, log error and raise exception
        logger.error(f"Failed to generate structured response after {max_retries} retries: {last_error}")
        raise ValueError(f"Failed to generate structured response: {last_error}")

    def validate_pathway(self, pathway_name: str, direction: str, disease: str,
                        evidence_text: str = "") -> PathwayValidationResult:
        """Validate pathway with structured prompt"""
        prompt = self.build_pathway_validation_prompt(pathway_name, direction, disease, evidence_text)
        
        result = self.llm_generate_structured_response(
            prompt=prompt,
            response_model=PathwayValidationOutput,
            max_retries=2
        )
        
        return PathwayValidationResult(
            pathway=result.get("pathway", pathway_name),
            status=result.get("status", "Uncertain"),
            justification=result.get("justification", "No justification provided"),
            confidence=result.get("confidence", 0.5),
            evidence_sources=[evidence_text] if evidence_text else []
        )
    