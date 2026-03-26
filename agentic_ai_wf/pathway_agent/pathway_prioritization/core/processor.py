# src/pathway_prioritization/core/processor.py
import pandas as pd
import numpy as np
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

from ..models import PathwayData, PathwayScore, ProcessingConfig
from ..utils import find_existing_results, load_processed_pathways, save_dataframe_to_csv
from .prioritizer import PathwayPrioritizer

logger = logging.getLogger(__name__)

class PathwayDataProcessor:
    """Enhanced data processor for pathway prioritization"""

    def __init__(self, prioritizer: PathwayPrioritizer):
        self.prioritizer = prioritizer

    def load_pathway_data(self, file_path: Path) -> List[PathwayData]:
        """Load and validate pathway data from CSV files"""
        all_pathways = []

        required_columns = [
            'DB_ID', 'Pathway source', 'Pathway ID', 'number_of_genes',
            'number_of_genes_in_background', 'inputGenes', 'Pathway associated genes',
            'p_value', 'fdr', 'Pathway', 'Regulation', 'clinical_relevance',
            'functional_relevance', 'hit_score', 'Ontology_Source', 'Main_Class', 'Sub_Class'
        ]

        try:
            df = pd.read_csv(file_path)
            
            # Store info about KEGG pathways for LLM processing decision
            kegg_count = len(df[df['DB_ID'] == 'KEGG'])
            self.prioritizer.apply_llm_to_kegg_only = False
            
            if self.prioritizer.apply_llm_to_kegg_only:
                logger.info(f"Found {kegg_count} KEGG pathways - will apply LLM consolidation to KEGG only")
            else:
                logger.info(f"Found {kegg_count} KEGG pathways - will apply LLM consolidation to all pathways")
            
            logger.info(f"Loaded {len(df)} total pathways from {file_path}")

            # Validate required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                return all_pathways

            # Convert to PathwayData objects
            for _, row in df.iterrows():
                try:
                    pathway = PathwayData(
                        db_id=str(row.get('DB_ID', '')),
                        pathway_source=str(row.get('Pathway source', '')),
                        pathway_id=str(row.get('Pathway ID', '')),
                        pathway_name=str(row.get('Pathway', '')),
                        number_of_genes=int(row.get('number_of_genes', 0)),
                        number_of_genes_in_background=int(row.get('number_of_genes_in_background', 0)),
                        input_genes=str(row.get('inputGenes', '')),
                        pathway_genes=str(row.get('Pathway associated genes', '')),
                        p_value=float(row.get('p_value', 1.0)),
                        fdr=float(row.get('fdr', 1.0)),
                        regulation=str(row.get('Regulation', '')),
                        clinical_relevance=str(row.get('clinical_relevance', '')),
                        functional_relevance=str(row.get('functional_relevance', '')),
                        hit_score=float(row.get('hit_score', 0.0)),
                        ontology_source=str(row.get('Ontology_Source', '')),
                        main_class=str(row.get('Main_Class', '')),
                        sub_class=str(row.get('Sub_Class', '')),
                        disease_category=str(row.get('disease_category', '')),
                        disease_subcategory=str(row.get('disease_subcategory', '')),
                        cellular_component=str(row.get('cellular_component', '')),
                        subcellular_element=str(row.get('subcellular_element', '')),
                        references=str(row.get('references', '')),
                        audit_log=str(row.get('audit_log', '')),
                        relation=str(row.get('relation', ''))
                    )
                    all_pathways.append(pathway)
                except Exception as e:
                    logger.warning(f"Error processing row in {file_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return all_pathways

        logger.info(f"Successfully loaded {len(all_pathways)} total pathways")
        return all_pathways

    def save_results(self, scores: List[PathwayScore], output_file: Path) -> bool:
        """Save prioritization results to CSV"""
        logger.info(f"Saving {len(scores)} scores to {output_file}")
        
        results_data = []

        for score in scores:
            pathway = score.pathway_data
            try:
                results_data.append({
                    "Priority_Rank": score.priority_rank,
                    "LLM_Score": score.llm_score,
                    "Confidence_Level": score.confidence_level,
                    "Score_Justification": score.score_justification,
                    "DB_ID": pathway.db_id,
                    "Pathway_Source": pathway.pathway_source,
                    "Pathway_ID": pathway.pathway_id,
                    "Pathway_Name": pathway.pathway_name,
                    "Number_of_Genes": pathway.number_of_genes,
                    "Number_of_Genes_in_Background": pathway.number_of_genes_in_background,
                    "P_Value": pathway.p_value,
                    "FDR": pathway.fdr,
                    "Hit_Score": pathway.hit_score,
                    "Regulation": pathway.regulation,
                    "Clinical_Relevance": pathway.clinical_relevance,
                    "Functional_Relevance": pathway.functional_relevance,
                    "Main_Class": pathway.main_class,
                    "Sub_Class": pathway.sub_class,
                    "Disease_Category": pathway.disease_category,
                    "Disease_Subcategory": pathway.disease_subcategory,
                    "Cellular_Component": pathway.cellular_component,
                    "Subcellular_Element": pathway.subcellular_element,
                    "Ontology_Source": pathway.ontology_source,
                    "Input_Genes": pathway.input_genes,
                    "Pathway_Associated_Genes": pathway.pathway_genes
                })
            except Exception as e:
                logger.error(f"Error processing score: {e}")
                continue

        df = pd.DataFrame(results_data)
        return save_dataframe_to_csv(df, output_file)

    async def process_pathway_prioritization(
        self,
        pathways_file: Path,
        disease_name: str,
        output_dir: Path,
        config: Optional[ProcessingConfig] = None
    ) -> Dict[str, any]:
        """Process pathway prioritization for the specified disease"""
        
        if config is None:
            config = ProcessingConfig(output_dir=output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting pathway prioritization for {disease_name}")
        logger.info(f"Output folder: {output_dir}")

        # Check for existing results
        existing_results = find_existing_results(output_dir, disease_name)

        # Load pathway data
        logger.info("Loading pathway data")
        all_pathways = self.load_pathway_data(file_path=pathways_file)
        if not all_pathways:
            raise ValueError("No valid pathway data loaded")

        # Filter pathways for LLM processing
        if self.prioritizer.apply_llm_to_kegg_only:
            pathways_for_llm = [p for p in all_pathways if p.db_id == 'KEGG']
            logger.info(f"Filtered to {len(pathways_for_llm)} KEGG pathways for LLM processing")
        else:
            pathways_for_llm = all_pathways
            logger.info(f"Using all {len(pathways_for_llm)} pathways for LLM processing")

        if not pathways_for_llm:
            logger.warning("No pathways selected for LLM processing")
            # Create default scores for all pathways
            all_scores = [
                PathwayScore(
                    pathway_data=pathway,
                    llm_score=0.0,
                    score_justification="Non-LLM pathway - assigned default score",
                    confidence_level="Medium"
                ) for pathway in all_pathways
            ]
        else:
            # Use parallel processing for scoring
            logger.info(f"Processing {len(pathways_for_llm)} pathways using parallel processing")
            
            def progress_callback(report):
                if report.processed % 10 == 0:  # Log every 10 pathways
                    logger.info(f"Progress: {report.processed}/{report.total_pathways} "
                               f"({report.progress_percentage}%) - "
                               f"Success: {report.success_rate}%")

            all_scores = self.prioritizer.score_pathways_parallel(
                pathways=pathways_for_llm,
                max_workers=config.max_workers,
                progress_callback=progress_callback
            )

        # Sort and rank results
        all_scores.sort(key=lambda x: x.llm_score, reverse=True)
        for i, score in enumerate(all_scores):
            score.priority_rank = i + 1

        # Save final results
        final_output_file = output_dir / f"{disease_name.replace(' ', '_')}_Pathways_Consolidated.csv"
        self.save_results(all_scores, final_output_file)

        # Generate summary
        summary = self._generate_summary(all_scores, disease_name, final_output_file)
        
        logger.info(f"Processing completed! Results saved to: {final_output_file}")
        
        return {
            "all_scores": all_scores,
            "summary": summary,
            "output_file": final_output_file,
            "total_pathways": len(all_scores)
        }

    def _generate_summary(self, scores: List[PathwayScore], disease_name: str, output_file: Path) -> Dict[str, any]:
        """Generate processing summary"""
        top_pathways = scores[:10]
        
        summary = {
            "disease_name": disease_name,
            "total_pathways_analyzed": len(scores),
            "average_score": float(np.mean([s.llm_score for s in scores])),
            "score_distribution": {
                "high_score_90_plus": len([s for s in scores if s.llm_score >= 90]),
                "good_score_70_89": len([s for s in scores if 70 <= s.llm_score < 90]),
                "moderate_score_50_69": len([s for s in scores if 50 <= s.llm_score < 70]),
                "low_score_below_50": len([s for s in scores if s.llm_score < 50])
            },
            "top_pathways": [
                {
                    "rank": score.priority_rank,
                    "pathway_name": score.pathway_data.pathway_name,
                    "llm_score": score.llm_score,
                    "confidence": score.confidence_level,
                    "p_value": score.pathway_data.p_value
                } for score in top_pathways
            ],
            "output_file": str(output_file),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary