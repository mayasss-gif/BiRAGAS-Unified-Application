"""
Drugs Extraction and Evaluation Package

This package provides comprehensive drug evaluation capabilities including:
- LLM-based clinical evaluation
- Patient medication history management  
- Drug interaction and contraindication checking
- Integration with KEGG pipeline for drug discovery
"""

from .drug_llm_evaluator import (
    DrugLLMEvaluator,
    DrugEvaluationResult,
    evaluate_drugs_with_llm
)

from .drug_evaluation_pipeline import (
    DrugEvaluationPipeline,
    evaluate_kegg_drugs,
    evaluate_kegg_drugs_sync
)

__version__ = "1.0.0"
__author__ = "Agentic AI Team"

__all__ = [
    "DrugLLMEvaluator",
    "DrugEvaluationResult",
    "evaluate_drugs_with_llm",
    "DrugEvaluationPipeline",
    "evaluate_kegg_drugs",
    "evaluate_kegg_drugs_sync"
]
