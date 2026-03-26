import networkx as nx
import json
import os
import numpy as np
import logging

from .centrality_engine import CentralityCalculator

logging.basicConfig(level=logging.INFO, format='[Centrality] %(message)s')


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def run_centrality(dag: nx.DiGraph, disease_name: str, output_dir: str = None) -> dict:
    """Run centrality analysis on a consensus DAG.

    Args:
        dag:           The consensus nx.DiGraph from Phase 1.
        disease_name:  Disease name (must match the name used in Phase 1).
        output_dir:    Optional directory to write outputs into.

    Returns:
        dict with keys:
            enriched_dag    – the DAG with centrality attributes added
            metrics_report  – tier classifications and per-gene scores
    """
    logging.info("CENTRALITY CALCULATOR: INITIATING")

    calculator = CentralityCalculator(dag, disease_name)
    enriched_dag, metrics_report = calculator.run_pipeline()

    t1 = len(metrics_report['hub_classifications']['Tier_1_Master_Regulators'])
    t2 = len(metrics_report['hub_classifications']['Tier_2_Secondary_Drivers'])
    t3 = len(metrics_report['hub_classifications']['Tier_3_Downstream_Effectors'])
    logging.info(f"  Tier 1 Master Regulators: {t1}")
    logging.info(f"  Tier 2 Secondary Drivers: {t2}")
    logging.info(f"  Tier 3 Downstream Effectors: {t3}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prefix = disease_name.replace(' ', '_')

        report_path = os.path.join(output_dir, f"{prefix}_centrality_metrics_report.json")
        with open(report_path, 'w') as f:
            json.dump(metrics_report, f, indent=2, cls=_NpEncoder)
        logging.info(f"  [Saved] {report_path}")

        enriched_json = nx.node_link_data(enriched_dag)
        enriched_json['meta'] = {
            'Disease': disease_name,
            'version': '1.0_Centrality_Enriched',
        }
        dag_path = os.path.join(output_dir, f"{prefix}_enriched_dag.json")
        with open(dag_path, 'w') as f:
            json.dump(enriched_json, f, indent=2, cls=_NpEncoder)
        logging.info(f"  [Saved] {dag_path}")

    return {
        'enriched_dag': enriched_dag,
        'metrics_report': metrics_report,
    }
