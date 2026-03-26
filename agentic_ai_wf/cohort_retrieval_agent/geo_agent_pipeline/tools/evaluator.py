"""
Evaluator tool for the Cohort Retrieval Agent system.

This tool handles evaluating files from various sources with retry logic,
progress tracking, and concurrent download capabilities.
"""


from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import asyncio
from openai import OpenAI
from typing import Dict, List, Optional, Any, Union


# Project Imports
from ..config import CohortRetrievalConfig
from ..base.base_tool import BaseTool, ToolResult
from ..tools.evaluatorrag import LLMEvaluator

@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""
    dataset_id: str
    primary_metrics: Dict[str, float]
    composite_scores: Dict[str, float]
    detailed_metrics: Dict[str, float]
    supporting_references: Dict[str, Any]
    justification: str
    error_message: Optional[str] = None


class EvaluationTool(BaseTool[List[EvaluationResult]]):
    """
    Tool for evaluating datasets using the LLMEvaluator.
    
    Evaluates:
    - Dataset relevance to disease and tissue type
    - Primary, composite, and detailed metrics
    - Supporting references and justification
    """
    
    def __init__(self, config: CohortRetrievalConfig, evaluator: LLMEvaluator):
        super().__init__(config, "EvaluationTool")
        self.evaluator = evaluator
    
    async def evaluate_datasets(self, datasets: List[Dict[str, Any]], series_meta : List[Dict[str,Any]], samples : Any,disease: str, query : str, filters: Optional[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """
        Evaluates a list of datasets and returns the evaluation results.
        
        Args:
            datasets (List[Dict[str, Any]]): List of dataset metadata dictionaries to evaluate.
            disease (str): The disease being studied.
            filters (str): The expected tissue type and experiment type.
        
        Returns:
            List[EvaluationResult]: List of evaluation results for each dataset.
        """
        try:
            result = await self.evaluator.evaluate(datasets,series_meta,samples, disease, query, filters)
            # Convert results to EvaluationResult instances
            evaluation_results = []
            
            eval_result = EvaluationResult(
                    dataset_id=result["dataset_id"],
                    primary_metrics=result["primary_metrics"],
                    composite_scores=result["composite_scores"],
                    detailed_metrics=result["detailed_metrics"],
                    supporting_references=result["supporting_references"],
                    justification=result["justification"]
                )
            evaluation_results.append(eval_result)

            return evaluation_results

        except Exception as e:
            # Handle errors and return a meaningful error message
            error_message = str(e)
            return [EvaluationResult(
                dataset_id="error",
                primary_metrics={},
                composite_scores={},
                detailed_metrics={},
                supporting_references={},
                justification="",
                error_message=error_message
            )]

