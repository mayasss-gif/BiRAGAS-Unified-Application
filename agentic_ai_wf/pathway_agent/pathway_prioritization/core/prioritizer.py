# src/pathway_prioritization/core/prioritizer.py
import os
import json
import re
import time
import logging
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from ..models import PathwayData, PathwayScore, DiseaseContext
from ..utils import ThreadSafeProgressTracker
from ..config import settings
from .scorer import PathwayScorer

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PathwayPrioritizer:
    """Enhanced Pathway Prioritization System for Disease-Specific Analysis"""

    def __init__(self, api_key: str = None, disease_name: str = ""):
        """Initialize the prioritizer with API key and disease context"""
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.disease_name = disease_name
        self.disease_context = self._get_disease_context(disease_name)
        self.scoring_criteria = self._initialize_scoring_criteria()
        self.apply_llm_to_kegg_only = True
        self.scorer = PathwayScorer()

    def _get_disease_context(self, disease_name: str) -> Dict[str, any]:
        """Get disease-specific context for pathway analysis (dynamic)"""
        # Check cache first
        cached_context = self._get_cached_disease_context(disease_name)
        if cached_context:
            logger.info(f"Using cached context for {disease_name}")
            return cached_context

        # Hardcoded contexts for well-known diseases
        hardcoded_contexts = {
            "Pancreatic Cancer": {
                "key_pathways": [
                    "KRAS signaling", "PI3K-AKT", "mTOR pathway", "EGFR signaling",
                    "DNA repair", "Apoptosis", "Cell cycle", "TGF-beta signaling",
                    "Wnt signaling", "Notch signaling", "JAK-STAT", "NF-kappa B"
                ],
                "molecular_features": [
                    "KRAS mutations", "TP53 mutations", "CDKN2A alterations",
                    "SMAD4 inactivation", "BRCA1/2 mutations", "PIK3CA mutations"
                ],
                "therapeutic_targets": [
                    "KRAS G12C", "mTOR inhibitors", "PARP inhibitors",
                    "Immune checkpoint inhibitors", "EGFR inhibitors"
                ],
                "disease_category": "cancer",
                "source": "hardcoded"
            },
            "Lupus Cancer": {
                "key_pathways": [
                    "Immune response", "Inflammation", "Autoimmune regulation",
                    "Cell death", "Cytokine signaling", "B cell activation",
                    "T cell signaling", "Complement system"
                ],
                "molecular_features": [
                    "Autoantibodies", "Immune complex deposition", "Type I interferon signature",
                    "Lymphocyte dysfunction", "Complement deficiency"
                ],
                "therapeutic_targets": [
                    "B cell inhibitors", "Cytokine blockers", "Immunosuppressants",
                    "Monoclonal antibodies", "JAK inhibitors"
                ],
                "disease_category": "autoimmune",
                "source": "hardcoded"
            }
        }

        # Return hardcoded context if available
        if disease_name in hardcoded_contexts:
            logger.info(f"Using hardcoded context for {disease_name}")
            return hardcoded_contexts[disease_name]

        # Generate dynamic context for unknown diseases
        logger.info(f"Generating dynamic context for {disease_name}")
        try:
            dynamic_context = self._generate_dynamic_disease_context(disease_name)
            # Cache the generated context
            self._cache_disease_context(disease_name, dynamic_context)
            return dynamic_context
        except Exception as e:
            logger.warning(f"Failed to generate dynamic context for {disease_name}: {e}")
            # Return fallback context based on disease category
            return self._get_fallback_disease_context(disease_name)

    def _get_cached_disease_context(self, disease_name: str) -> Dict[str, any]:
        """Check if disease context is cached"""
        cache_file = settings.cache_dir / "disease_context_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                return cache.get(disease_name)
            except Exception as e:
                logger.warning(f"Error reading disease context cache: {e}")
        return None

    def _cache_disease_context(self, disease_name: str, context: Dict[str, any]):
        """Cache disease context for future use"""
        cache_file = settings.cache_dir / "disease_context_cache.json"
        cache = {}

        # Load existing cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading existing cache: {e}")

        # Add new context
        cache[disease_name] = context

        # Save updated cache
        try:
            cache_file.parent.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Cached context for {disease_name}")
        except Exception as e:
            logger.warning(f"Error saving disease context cache: {e}")

    def _generate_dynamic_disease_context(self, disease_name: str) -> Dict[str, any]:
        """Generate disease context dynamically using LLM"""
        prompt = f"""
            You are a medical expert. Provide comprehensive information about {disease_name} for pathway analysis.

            Return a JSON object with the following structure:
            {{
                "key_pathways": ["pathway1", "pathway2", ...],
                "molecular_features": ["feature1", "feature2", ...],
                "therapeutic_targets": ["target1", "target2", ...],
                "disease_category": "cancer|neurological|autoimmune|metabolic|cardiovascular|infectious|other",
                "source": "generated"
            }}

            Focus on established, well-documented information. Be concise but comprehensive.
            """

        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a medical expert providing structured disease information. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                context = json.loads(json_match.group())
                logger.info(f"Successfully generated dynamic context for {disease_name}")
                return context
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            logger.error(f"Failed to generate dynamic context for {disease_name}: {e}")
            raise

    def _get_fallback_disease_context(self, disease_name: str) -> Dict[str, any]:
        """Provide fallback context based on disease category"""
        disease_category = self._detect_disease_category(disease_name)

        fallback_contexts = {
            "cancer": {
                "key_pathways": [
                    "Cell cycle", "Apoptosis", "DNA repair", "PI3K-AKT signaling",
                    "p53 pathway", "MAPK pathway", "Angiogenesis", "Immune response"
                ],
                "molecular_features": [
                    "Oncogenes", "Tumor suppressors", "DNA mutations", "Chromosomal instability",
                    "Microsatellite instability", "Growth factor receptors"
                ],
                "therapeutic_targets": [
                    "Kinase inhibitors", "Immune checkpoint inhibitors", "Hormone therapy",
                    "Targeted therapy", "Chemotherapy", "Radiation therapy"
                ],
                "disease_category": "cancer",
                "source": "fallback"
            },
            "autoimmune": {
                "key_pathways": [
                    "Immune response", "Inflammation", "T cell activation", "B cell signaling",
                    "Cytokine signaling", "Complement activation", "Antigen presentation"
                ],
                "molecular_features": [
                    "Autoantibodies", "Immune complex formation", "Tissue inflammation",
                    "Loss of self-tolerance", "Molecular mimicry"
                ],
                "therapeutic_targets": [
                    "Immunosuppressive agents", "Anti-inflammatory drugs", "Monoclonal antibodies",
                    "Cytokine inhibitors", "T cell modulators"
                ],
                "disease_category": "autoimmune",
                "source": "fallback"
            }
        }

        context = fallback_contexts.get(disease_category, {
            "key_pathways": [
                "Cell signaling", "Metabolism", "Immune response", "Inflammation",
                "Oxidative stress", "Protein synthesis", "Cell death"
            ],
            "molecular_features": [
                "Genetic variants", "Protein dysfunction", "Metabolic alterations",
                "Inflammatory markers", "Tissue damage"
            ],
            "therapeutic_targets": [
                "Small molecule drugs", "Protein therapeutics", "Gene therapy",
                "Lifestyle interventions", "Supportive care"
            ],
            "disease_category": "other",
            "source": "fallback"
        })

        logger.info(f"Using fallback context for {disease_name} (category: {disease_category})")
        return context

    def _detect_disease_category(self, disease_name: str) -> str:
        """Detect disease category from name"""
        disease_lower = disease_name.lower()

        cancer_keywords = ['cancer', 'carcinoma', 'sarcoma', 'melanoma', 'lymphoma', 'leukemia', 'tumor', 'tumour', 'neoplasm', 'oncology', 'malignancy']
        neuro_keywords = ['alzheimer', 'parkinson', 'dementia', 'sclerosis', 'epilepsy', 'stroke', 'brain', 'neurological', 'neuropathy', 'huntington']
        autoimmune_keywords = ['arthritis', 'lupus', 'diabetes', 'inflammatory', 'autoimmune', 'crohn', 'psoriasis', 'thyroiditis']
        cardio_keywords = ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'atherosclerosis', 'coronary', 'myocardial', 'arrhythmia']

        if any(keyword in disease_lower for keyword in cancer_keywords):
            return "cancer"
        elif any(keyword in disease_lower for keyword in neuro_keywords):
            return "neurological"
        elif any(keyword in disease_lower for keyword in autoimmune_keywords):
            return "autoimmune"
        elif any(keyword in disease_lower for keyword in cardio_keywords):
            return "cardiovascular"
        else:
            return "other"

    def _initialize_scoring_criteria(self) -> Dict[str, Dict]:
        """Initialize scoring criteria for pathway prioritization"""
        return {
            "statistical_significance": {
                "weight": 0.25,
                "description": "P-value and FDR significance",
                "thresholds": {"excellent": 0.001, "good": 0.01, "moderate": 0.05}
            },
            "disease_relevance": {
                "weight": 0.30,
                "description": "Direct relevance to disease pathophysiology",
                "factors": ["known_disease_pathway", "therapeutic_target", "biomarker_potential"]
            },
            "functional_importance": {
                "weight": 0.20,
                "description": "Biological importance and pathway centrality",
                "factors": ["essential_processes", "pathway_connectivity", "gene_count"]
            },
            "clinical_significance": {
                "weight": 0.15,
                "description": "Clinical relevance and therapeutic potential",
                "factors": ["druggable_targets", "prognostic_value", "diagnostic_utility"]
            },
            "evidence_strength": {
                "weight": 0.10,
                "description": "Literature support and experimental validation",
                "factors": ["publication_count", "experimental_validation", "clinical_studies"]
            }
        }

    def generate_scoring_prompt(self, disease_name: str) -> str:
        """Generate comprehensive scoring prompt for pathway prioritization"""
        disease_context = self.disease_context
        key_pathways = ", ".join(disease_context.get("key_pathways", []))
        molecular_features = ", ".join(disease_context.get("molecular_features", []))

        system_prompt = f"""
            You are a molecular biologist and bioinformatics expert specializing in {disease_name} research. 
            Your task is to evaluate biological pathways for their relevance to {disease_name} pathogenesis, 
            progression, and therapeutic potential.

            DISEASE CONTEXT - {disease_name}:
            - Key Known Pathways: {key_pathways}
            - Molecular Features: {molecular_features}
            - Therapeutic Targets: {", ".join(disease_context.get("therapeutic_targets", []))}

            SCORING CRITERIA (0-100 scale):
            1. DISEASE SPECIFICITY & RELEVANCE (35%): Direct relevance to {disease_name}
            2. STATISTICAL SIGNIFICANCE (25%): Evaluate p-value and FDR
            3. FUNCTIONAL IMPORTANCE (20%): Biological significance
            4. CLINICAL SIGNIFICANCE (15%): Therapeutic and diagnostic potential
            5. EVIDENCE STRENGTH (5%): Literature and experimental support

            Provide your analysis in this exact format:
            Score: [numerical score 0-100]
            Confidence: [High/Medium/Low] 
            Justification: [detailed 2-3 sentence explanation]
            Disease_Category: [Category]
            Disease_Subcategory: [Subcategory]
            Cellular_Component: [Component]
            Subcellular_Element: [Element]
            """
        return system_prompt

    def _score_single_pathway_threaded(self, pathway: PathwayData, worker_id: str) -> PathwayScore:
        """Score a single pathway using synchronous OpenAI API call for threading"""
        start_time = time.time()
        
        try:
            system_prompt = self.generate_scoring_prompt(self.disease_name)
            
            pathway_description = f"""
                PATHWAY:
                Name: {pathway.pathway_name}
                Source: {pathway.pathway_source}
                Genes Count: {pathway.number_of_genes}
                P-value: {pathway.p_value:.2e}
                FDR: {pathway.fdr:.2e}
                Hit Score: {pathway.hit_score}
                Regulation: {pathway.regulation}
                Clinical Relevance: {pathway.clinical_relevance}
                Functional Relevance: {pathway.functional_relevance}
                Main Class: {pathway.main_class}
                Sub Class: {pathway.sub_class}
                """
            
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        Analyze the following biological pathway for its relevance to {self.disease_name}.
                        Provide detailed scoring based on the criteria outlined:

                        {pathway_description}

                        Provide your analysis in this exact format:
                        Score: [numerical score 0-100]
                        Confidence: [High/Medium/Low] 
                        Justification: [detailed 2-3 sentence explanation]
                        Disease_Category: [Category]
                        Disease_Subcategory: [Subcategory]
                        Cellular_Component: [Component]
                        Subcellular_Element: [Element]
                    """}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p
            )
            
            response_content = response.choices[0].message.content.strip()
            score = self.scorer.parse_single_pathway_response(response_content, pathway)
            
            processing_time = time.time() - start_time
            logger.debug(f"Completed {pathway.pathway_name} in {processing_time:.2f}s (Score: {score.llm_score})")
            
            return score
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {pathway.pathway_name}: {e}")
            
            return PathwayScore(
                pathway_data=pathway,
                llm_score=0.0,
                score_justification=f"Error during processing: {str(e)}",
                confidence_level="Low"
            )

    def score_pathways_parallel(self, pathways: List[PathwayData], max_workers: int = None, 
                               progress_callback: callable = None) -> List[PathwayScore]:
        """Score pathways using parallel processing with ThreadPoolExecutor"""
        if not pathways:
            return []
            
        max_workers = max_workers or settings.max_workers
        progress_tracker = ThreadSafeProgressTracker(len(pathways))
        results_lock = Lock()
        all_scores = []
        
        logger.info(f"Starting parallel pathway scoring: {len(pathways)} pathways with {max_workers} workers")
        
        def worker_task(pathway: PathwayData) -> PathwayScore:
            """Worker function for processing a single pathway"""
            worker_id = f"Worker-{threading.current_thread().ident}"
            start_time = time.time()
            
            try:
                score = self._score_single_pathway_threaded(pathway, worker_id)
                processing_time = time.time() - start_time
                
                progress_tracker.update_worker_stats(worker_id, pathway.pathway_name, processing_time, True)
                
                with results_lock:
                    all_scores.append(score)
                
                if progress_callback:
                    try:
                        progress_report = progress_tracker.get_progress_report()
                        progress_callback(progress_report)
                    except Exception as callback_error:
                        logger.warning(f"Progress callback error: {callback_error}")
                
                return score
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Failed to process {pathway.pathway_name}: {e}")
                
                progress_tracker.update_worker_stats(worker_id, pathway.pathway_name, processing_time, False)
                
                default_score = PathwayScore(
                    pathway_data=pathway,
                    llm_score=0.0,
                    score_justification=f"Worker error: {str(e)}",
                    confidence_level="Low"
                )
                
                with results_lock:
                    all_scores.append(default_score)
                
                return default_score
        
        # Execute parallel processing
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PathwayWorker") as executor:
            future_to_pathway = {executor.submit(worker_task, pathway): pathway for pathway in pathways}
            
            # Wait for all tasks to complete
            for future in as_completed(future_to_pathway):
                pathway = future_to_pathway[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Future exception for {pathway.pathway_name}: {e}")
        
        final_report = progress_tracker.get_progress_report()
        logger.info(f"Parallel processing completed!")
        logger.info(f"Final Stats: {final_report.processed} successful, {final_report.failed} failed")
        logger.info(f"Total time: {final_report.elapsed_time}s, Avg: {final_report.average_processing_time}s/pathway")
        logger.info(f"Success rate: {final_report.success_rate}%")
        
        return all_scores