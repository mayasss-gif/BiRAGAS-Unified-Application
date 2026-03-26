from dotenv import load_dotenv
import os
# from pathlib import Path

API_BASE   = "https://string-db.org/api"
SPECIES    = 9606
CALLER     = "batch_enrich_processor"
MAX_CHAR   = 1900
MAX_COUNT  = 200

API_TO_KEY = {
    'Process':       'Process',
    'Function':      'Function',
    'Component':     'Component',
    'KEGG':          'KEGG_PATHWAY',
    'RCTM':          'REACTOME',
    'WikiPathways':  'WikiPathway',
}

CATEGORY_DISPLAY = {
    'Process':       'Biological_Process',
    'Function':      'Molecular_Function',
    'Component':     'COMPARTMENTS',
    'KEGG_PATHWAY':  'kegg',
    'REACTOME':      'rctm',
    'WikiPathway':   'wikipathways',
}

ALLOWED_KEYS = set(CATEGORY_DISPLAY.keys())



go_folder_disps   = {"Biological_Process", "Molecular_Function", "COMPARTMENTS"}
main_folder_disps = {"kegg", "rctm", "wikipathways"}


load_dotenv()  # reads .env

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")

settings = Settings()

PACKAGE_PATH = "./agentic_ai_wf/pathway_agent"

# ANALYSIS_DIR = Path("ALL_ANALYSIS")

# PACKAGE_PATH = Path(os.path.abspath(__file__)).parent
 
DATASET_PATH = f"{PACKAGE_PATH}/datasets"
 
CATEGORIZER_CSV_PATH = f"{DATASET_PATH}/classification_memory.csv"

PATHWAY_LITERATURE_LIMIT = 500

PATHWAY_CONSOLIDATION_CACHE_DIR = "./agentic_ai_wf/shared/pathway_consolidation_cache/"