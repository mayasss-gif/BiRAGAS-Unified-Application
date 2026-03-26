import logging

from .dag_builder import run_dag_pipeline, PipelineConfig
from .centrality_calculator import run_centrality

logging.basicConfig(level=logging.INFO, format='[CausalityPipeline] %(message)s')


def run_causality_pipeline(config: PipelineConfig, output_dir: str = None) -> dict:
    """Run the full causality pipeline: DAG construction (Phase 1) → Centrality analysis (Phase 2).

    Args:
        config:     A PipelineConfig with disease_name and all file paths.
        output_dir: Directory to write all outputs (JSON, CSV). If None, no files are saved.

    Returns:
        dict with keys:
            consensus_dag    – raw consensus DAG from Phase 1
            enriched_dag     – DAG with centrality metrics from Phase 2
            patient_dags     – per-patient DAGs from Phase 1
            dag_stats        – Phase 1 summary statistics
            metrics_report   – Phase 2 tier classifications and scores
    """
    logging.info(f"=== CAUSALITY PIPELINE START: {config.disease_name} ===")

    # ── Phase 1: Causal DAG Construction ────────────────────────────────────
    logging.info("── Phase 1: Building Consensus Causal DAG ──")
    dag_result = run_dag_pipeline(config, output_dir=output_dir)

    # ── Phase 2: Centrality & Hub Classification ────────────────────────────
    logging.info("── Phase 2: Centrality Analysis & Tier Classification ──")
    centrality_result = run_centrality(
        dag_result['consensus_dag'],
        config.disease_name,
        output_dir=output_dir,
    )

    logging.info(f"=== CAUSALITY PIPELINE COMPLETE: {config.disease_name} ===")

    return {
        'consensus_dag': dag_result['consensus_dag'],
        'enriched_dag': centrality_result['enriched_dag'],
        'patient_dags': dag_result['patient_dags'],
        'dag_stats': dag_result['stats'],
        'metrics_report': centrality_result['metrics_report'],
    }
