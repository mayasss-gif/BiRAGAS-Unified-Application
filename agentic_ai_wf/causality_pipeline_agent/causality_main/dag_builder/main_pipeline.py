from .data_loader import DataLoader
from .dag_engine import AdvancedDAGEngine
from .config import PipelineConfig
import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[Pipeline] %(message)s')


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


def run_dag_pipeline(config: PipelineConfig, output_dir: str = None) -> dict:
    """Run the full causal DAG construction pipeline for any disease.

    Args:
        config:     A PipelineConfig instance with disease_name
                    and all required file paths set (absolute paths).
        output_dir: Optional directory to write JSON/CSV outputs into.
                    If None, no files are saved.

    Returns:
        dict with keys:
            consensus_dag  – the final nx.DiGraph
            patient_dags   – list of per-patient nx.DiGraphs
            stats          – summary statistics dict
    """
    logging.info("PIPELINE: INITIATING")

    # 1. Data Ingestion
    loader = DataLoader(config)
    data_packet = loader.load_all()
    logging.info("DATA INGESTION COMPLETE")

    logging.info(f"  Genes:      {len(data_packet.get('gene_list', []))}")
    logging.info(f"  Pathways:   {len(data_packet.get('pathways_df', []))}")
    logging.info(f"  L-R Pairs:  {len(data_packet.get('lr_pairs', []))}")
    logging.info(f"  eQTL Links: {len(data_packet.get('eqtl_edges', []))}")
    logging.info(f"  MR Links:   {len(data_packet.get('mr_evidence', []))}")

    # 2. Engine Execution
    engine = AdvancedDAGEngine(data_packet, config)
    consensus_dag, patient_dags = engine.build_consensus_pipeline()

    # 3. Collect stats
    roots_genetic = [n for n, d in consensus_dag.nodes(data=True) if d.get('is_gwas_hit')]
    snp_nodes = [n for n, d in consensus_dag.nodes(data=True) if d.get('type') == 'snp']
    mr_edges = [u for u, v, d in consensus_dag.edges(data=True)
                if 'mendelian_randomization_causality' in d.get('evidence', [])]
    pert_drivers = [u for u, v, d in consensus_dag.edges(data=True)
                    if 'perturbation_asymmetry' in d.get('evidence', [])]

    stats = {
        'disease': config.disease_name,
        'consensus_nodes': len(consensus_dag.nodes),
        'consensus_edges': len(consensus_dag.edges),
        'validated_genetic_roots': len(roots_genetic),
        'snp_source_nodes': len(snp_nodes),
        'mr_validated_trait_edges': len(mr_edges),
        'perturbation_drivers': len(pert_drivers),
        'patient_count': len(patient_dags),
    }

    logging.info("PIPELINE COMPLETE")
    for k, v in stats.items():
        logging.info(f"  {k}: {v}")

    # 4. Optional file output
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_outputs(consensus_dag, patient_dags, config, stats, output_dir)

    return {
        'consensus_dag': consensus_dag,
        'patient_dags': patient_dags,
        'stats': stats,
    }


def _save_outputs(consensus_dag, patient_dags, config, stats, output_dir):
    """Serialize the consensus DAG to JSON and CSV files."""
    prefix = config.disease_name.replace(' ', '_')

    # A. JSON graph
    output_json = nx.node_link_data(consensus_dag)
    output_json['meta'] = {
        'Disease': config.disease_name,
        'patient_count': len(patient_dags),
        'version': '1.0_Causal_DAG',
    }
    json_path = os.path.join(output_dir, f"{prefix}_consensus_causal_dag.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2, cls=_NpEncoder)
    logging.info(f"  [Saved] {json_path}")

    # B. Nodes CSV
    nodes_df = pd.DataFrame([dict(id=n, **d) for n, d in consensus_dag.nodes(data=True)])
    nodes_path = os.path.join(output_dir, f"{prefix}_consensus_dag_nodes.csv")
    nodes_df.to_csv(nodes_path, index=False)

    # C. Edges CSV
    edges_list = []
    for u, v, d in consensus_dag.edges(data=True):
        edge_dict = {'source': u, 'target': v}
        for key, val in d.items():
            edge_dict[key] = "|".join(val) if isinstance(val, (list, set)) else val
        edges_list.append(edge_dict)

    edges_df = pd.DataFrame(edges_list)
    edges_path = os.path.join(output_dir, f"{prefix}_consensus_dag_edges.csv")
    edges_df.to_csv(edges_path, index=False)
    logging.info(f"  [Saved] {nodes_path} & {edges_path}")
