import os
from pathlib import Path

package_path = Path(__file__).resolve().parent
dataset_dir = package_path / 'datasets'
cgc_file = dataset_dir / 'CGC.txt'

jl_zscore_file = dataset_dir / 'z-scores_final.csv'

GENE_CARD_SCORES_PATH = './agentic_ai_wf/gene_prioritization/datasets/disease_to_gene.pkl'

REPORTS_DIR = "./agentic_ai_wf/reports"

GENE_PRIORITIZATION_LIMIT = None

DEG_FILTERING_OUTPUT_DIR = "./agentic_ai_wf/shared/deg_data"