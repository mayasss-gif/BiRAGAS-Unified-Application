"""
Drug Evaluation Pipeline

This module provides a complete pipeline for evaluating drugs using LLM analysis,
integrating patient medication history, and producing production-ready results.
"""
import pandas as pd
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
from datetime import datetime
import os

from .drug_llm_evaluator import DrugLLMEvaluator

logger = logging.getLogger(__name__)


class DrugEvaluationPipeline:
    """
    Production-ready drug evaluation pipeline.

    This pipeline provides:
    - Batch processing of drug evaluations
    - Patient medication history integration
    - LLM-based clinical relevance scoring
    - Drug interaction and contraindication checking
    - Comprehensive reporting and logging
    """

    def __init__(
        self,
        batch_size: int = 5,
        max_drugs: int = None,
        enable_logging: bool = True
    ):
        """
        Initialize the drug evaluation pipeline.

        Args:
            openai_api_key: OpenAI API key for LLM evaluation
            batch_size: Number of drugs to process per batch
            enable_logging: Whether to enable detailed logging
        """
        print(f"Initializing DrugEvaluationPipeline with batch_size: {batch_size} and max_drugs: {max_drugs}")

        self.evaluator = DrugLLMEvaluator(max_drugs=max_drugs, batch_size=batch_size)
        self.evaluator.batch_size = batch_size
        self.enable_logging = enable_logging

        if enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging for the pipeline."""
        # Create logs directory if it doesn't exist
        log_dir = Path("agentic_ai_wf/logs/drug_evaluation")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"drug_evaluation_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        logger.info(f"Drug evaluation pipeline logging started - {log_file}")

    async def evaluate_drugs_dataframe(
        self,
        df: pd.DataFrame,
        disease_name: str = "Unknown Disease",
        analysis_id: str = "drug_evaluation",
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate drugs in DataFrame using comprehensive LLM analysis.

        Args:
            df: DataFrame with drug data
            patient_medications_file: Path to patient medications file
            disease_name: Patient's primary disease name
            analysis_id: Analysis identifier for outputs
            output_dir: Output directory for results

        Returns:
            DataFrame with LLM evaluation results
        """
        # Initialize variables for error handling
        evaluated_df = None
        timestamp = None
        output_path = None
        
        try:
            start_time = time.time()
            logger.info(f"Starting drug evaluation for {len(df)} drugs")

            # Setup output directory
            if output_dir is None:
                output_dir = f"agentic_ai_wf/shared/drug_evaluation/{analysis_id}"

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_path / \
                f"drug_evaluation_results_{timestamp}.csv"

            # Perform LLM evaluation
            logger.info("Performing LLM-based drug evaluation...")

            evaluated_df = await self.evaluator.evaluate_drugs_molecular_evidence_from_df(
                df=df,
                disease_name=disease_name,
                save_results=True,
                output_file=str(output_file)
            )

            # Generate additional analysis
            await self._generate_evaluation_report(
                evaluated_df, analysis_id, output_path, disease_name
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Drug evaluation completed in {processing_time:.2f} seconds")

            return evaluated_df

        except Exception as e:
            logger.error(f"Error in drug evaluation pipeline: {e}")
            # Try to save partial results if available
            try:
                if evaluated_df is not None and output_path is not None and timestamp is not None:
                    logger.info("Attempting to save partial results...")
                    partial_output_file = output_path / f"partial_drug_evaluation_results_{timestamp}.csv"
                    evaluated_df.to_csv(partial_output_file, index=False)
                    logger.info(f"Partial results saved to: {partial_output_file}")
                    return evaluated_df
            except Exception as save_error:
                logger.error(f"Failed to save partial results: {save_error}")
            
            # Return original DataFrame if evaluation completely failed
            logger.warning("Returning original DataFrame due to evaluation error")
            return df

    async def _generate_evaluation_report(
        self,
        df: pd.DataFrame,
        analysis_id: str,
        output_dir: Path,
        disease_name: str
    ):
        """Generate comprehensive evaluation report."""
        try:
            logger.info("Generating evaluation report...")

            # Summary statistics
            stats = self._calculate_evaluation_statistics(df)

            # Create report
            report = {
                "analysis_id": analysis_id,
                "disease_name": disease_name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_drugs_evaluated": len(df),
                "statistics": stats,
                "top_recommended_drugs": self._get_top_drugs(df, "RECOMMEND", 10),
                "high_clinical_relevance": self._get_high_relevance_drugs(df, 10),
                "interaction_warnings": self._summarize_interactions(df),
                "summary": self._generate_executive_summary(df, stats)
            }

            # Save report
            report_file = output_dir / f"evaluation_report_{analysis_id}.json"

            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Evaluation report saved: {report_file}")

        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")

    def _calculate_evaluation_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate evaluation statistics."""
        try:
            stats = {}

            # Basic counts
            stats['total_drugs'] = len(df)

            # Clinical relevance distribution
            if 'Drug_Clinical_Relevance' in df.columns:
                relevance_counts = df['Drug_Clinical_Relevance'].value_counts(
                ).to_dict()
                stats['clinical_relevance_distribution'] = relevance_counts

            # Recommendation distribution
            if 'Drug_Recommendation' in df.columns:
                recommendation_counts = df['Drug_Recommendation'].value_counts(
                ).to_dict()
                stats['recommendation_distribution'] = recommendation_counts

            # Score statistics
            if 'Drug_LLM_Score' in df.columns:
                scores = pd.to_numeric(df['Drug_LLM_Score'], errors='coerce')
                stats['score_statistics'] = {
                    'mean': float(scores.mean()),  # type: ignore
                    'median': float(scores.median()),  # type: ignore
                    'std': float(scores.std()),  # type: ignore
                    'min': float(scores.min()),  # type: ignore
                    'max': float(scores.max())  # type: ignore
                }

            # Interaction warnings
            if 'Drug_Interaction_Warnings' in df.columns:
                interactions = df['Drug_Interaction_Warnings'].str.len().sum()
                stats['total_interaction_warnings'] = int(
                    interactions) if not pd.isna(interactions) else 0

            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}

    def _get_top_drugs(self, df: pd.DataFrame, recommendation: str, limit: int) -> List[Dict]:
        """Get top drugs by recommendation type."""
        try:
            if 'Drug_Recommendation' not in df.columns:
                return []

            filtered_df = df[df['Drug_Recommendation'] == recommendation]

            if 'Drug_LLM_Score' in filtered_df.columns:
                top_drugs = filtered_df.nlargest(
                    limit, 'Drug_LLM_Score')  # type: ignore
            else:
                top_drugs = filtered_df.head(limit)

            # Convert to records format explicitly to avoid linter issues
            columns = ['Drug_Name', 'Drug_ID',
                       'Drug_LLM_Score', 'Drug_Score_Justification']
            result_df = top_drugs[columns]
            return result_df.to_dict('records')  # type: ignore

        except Exception as e:
            logger.error(f"Error getting top drugs: {e}")
            return []

    def _get_high_relevance_drugs(self, df: pd.DataFrame, limit: int) -> List[Dict]:
        """Get drugs with high clinical relevance."""
        try:
            if 'Drug_Clinical_Relevance' not in df.columns:
                return []

            high_relevance = df[df['Drug_Clinical_Relevance'] == 'HIGH']

            if 'Drug_LLM_Score' in high_relevance.columns:
                top_relevant = high_relevance.nlargest(
                    limit, 'Drug_LLM_Score')  # type: ignore
            else:
                top_relevant = high_relevance.head(limit)

            # Convert to records format explicitly to avoid linter issues
            columns = ['Drug_Name', 'Drug_ID',
                       'Drug_Clinical_Relevance', 'Drug_LLM_Score']
            result_df = top_relevant[columns]
            return result_df.to_dict('records')  # type: ignore

        except Exception as e:
            logger.error(f"Error getting high relevance drugs: {e}")
            return []

    def _summarize_interactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize interaction warnings."""
        try:
            if 'Drug_Interaction_Warnings' not in df.columns:
                return {}

            # Count drugs with interactions
            has_interactions = df['Drug_Interaction_Warnings'].notna() & (
                df['Drug_Interaction_Warnings'] != '')
            interaction_count = has_interactions.sum()

            return {
                'drugs_with_interactions': int(interaction_count),
                'total_drugs': len(df),
                'percentage_with_interactions': round(float(interaction_count / len(df) * 100), 2)
            }

        except Exception as e:
            logger.error(f"Error summarizing interactions: {e}")
            return {}

    def _generate_executive_summary(self, df: pd.DataFrame, stats: Dict) -> str:
        """Generate executive summary of evaluation."""
        try:
            total_drugs = stats.get('total_drugs', 0)

            # Get recommendation counts
            recommendations = stats.get('recommendation_distribution', {})
            recommended = recommendations.get('RECOMMEND', 0)
            consider = recommendations.get('CONSIDER', 0)
            avoid = recommendations.get('AVOID', 0)

            # Get clinical relevance counts
            relevance = stats.get('clinical_relevance_distribution', {})
            high_relevance = relevance.get('HIGH', 0)
            medium_relevance = relevance.get('MEDIUM', 0)

            # Generate summary
            summary = f"""
                Drug Evaluation Executive Summary:

                Total drugs evaluated: {total_drugs}

                Recommendations:
                - RECOMMEND: {recommended} drugs ({recommended/total_drugs*100:.1f}%)
                - CONSIDER: {consider} drugs ({consider/total_drugs*100:.1f}%)
                - AVOID: {avoid} drugs ({avoid/total_drugs*100:.1f}%)

                Clinical Relevance:
                - HIGH: {high_relevance} drugs ({high_relevance/total_drugs*100:.1f}%)
                - MEDIUM: {medium_relevance} drugs ({medium_relevance/total_drugs*100:.1f}%)

                The evaluation identified {recommended} drugs for immediate consideration and {consider} drugs for further evaluation.
                {high_relevance} drugs demonstrated high clinical relevance for the patient's condition.
            """.strip()

            return summary

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Executive summary generation failed."


# Main integration function for KEGG pipeline
async def evaluate_kegg_drugs(
    results_df: pd.DataFrame,
    analysis_id: str = "kegg_drug_evaluation",
    disease_name: str = "Unknown Disease",
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Main function to evaluate KEGG drugs using LLM analysis.

    This function is designed to be easily integrated into the main KEGG pipeline
    after results_df is created.

    Args:
        results_df: DataFrame with KEGG drug results
        analysis_id: Analysis identifier
        disease_name: Patient's disease name
        patient_medications_file: Path to patient medications file
        openai_api_key: OpenAI API key
        output_dir: Output directory for results

    Returns:
        DataFrame with LLM evaluation results added
    """
    try:
        logger.info(
            f"Starting KEGG drug evaluation for {len(results_df)} drugs")

        # Initialize pipeline
        pipeline = DrugEvaluationPipeline(
            batch_size=5,
            max_drugs=100,
            enable_logging=True
        )

        # Perform evaluation
        evaluated_df = await pipeline.evaluate_drugs_dataframe(
            df=results_df,
            disease_name=disease_name,
            analysis_id=analysis_id,
            output_dir=output_dir
        )

        logger.info("KEGG drug evaluation completed successfully")
        return evaluated_df

    except Exception as e:
        logger.error(f"Error in KEGG drug evaluation: {e}")
        # Try to save the original DataFrame with basic information
        try:
            if output_dir is None:
                output_dir = f"agentic_ai_wf/shared/drug_evaluation/{analysis_id}"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_file = output_path / f"fallback_drug_results_{timestamp}.csv"
            results_df.to_csv(fallback_file, index=False)
            logger.info(f"Fallback results saved to: {fallback_file}")
        except Exception as save_error:
            logger.error(f"Failed to save fallback results: {save_error}")
        
        # Return original DataFrame if evaluation fails
        logger.warning("Returning original DataFrame due to evaluation error")
        return results_df


# Synchronous wrapper for backward compatibility
def evaluate_kegg_drugs_sync(
    results_df: pd.DataFrame,
    analysis_id: str = "kegg_drug_evaluation",
    disease_name: str = "Unknown Disease",
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Synchronous wrapper for evaluate_kegg_drugs.

    Args:
        results_df: DataFrame with KEGG drug results
        analysis_id: Analysis identifier
        disease_name: Patient's disease name
        patient_medications_file: Path to patient medications file
        openai_api_key: OpenAI API key
        output_dir: Output directory for results

    Returns:
        DataFrame with LLM evaluation results added
    """
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print(f"Using ThreadPoolExecutor")
                future = executor.submit(asyncio.run, evaluate_kegg_drugs(
                    results_df=results_df,
                    analysis_id=analysis_id,
                    disease_name=disease_name,
                    output_dir=output_dir
                ))
                return future.result()
        except RuntimeError:
            # No running loop, use asyncio.run
            print(f"No running loop, using asyncio.run")
            return asyncio.run(evaluate_kegg_drugs(
                results_df=results_df,
                analysis_id=analysis_id,
                disease_name=disease_name,
                output_dir=output_dir
            ))
    except Exception as e:
        logger.error(f"Error in synchronous drug evaluation: {e}")
        return results_df
