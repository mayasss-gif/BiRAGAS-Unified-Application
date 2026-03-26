import os
from typing import Dict, List


PACKAGE_PATH = "./agentic_ai_wf/clinical_report"
# PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(PACKAGE_PATH, "datasets")
# LOGO_PATH = os.path.join(DATASET_PATH, "logo.jpg")
LOGO_PATH = os.path.join(DATASET_PATH, "ARI-Logo.png")
HTML_TEMPLATE_FILE = "clinical_report1.html"


GO_IDS = {"GO_CC", "GO_BP", "GO_MF"}
MAX_THREADS = 10
UPREG_TARGET = 10
MIN_ABS_LFC = 1.0
DEFAULT_CONFIDENCE = 0.7
DEFAULT_PRIORITY = 999

PATHWAY_CATEGORIES: Dict[str, List[str]] = {
    "Interferon Signaling": ["interferon", "ifn"],
    "Complement": ["complement"],
    "B‑cell Dysfunction": ["b cell", "b‑cell", "cd20", "baff"],
    "T‑cell Dysregulation": ["t cell", "t‑cell", "ctla4", "pd‑1"],
    "Signal Transduction": ['PI3K', 'MAPK', 'EGFR', 'signaling', 'receptor', 'Akt', 'mTOR'],
    "Immune / Inflammatory": ['immune', 'cytokine', 'chemokine', 'TNF', 'interferon', 'PD-1','PD-L1','T cell'],
    "Metabolism & Energy": ['metabolic', 'glycolysis', 'AMPK', 'oxidative', 'fatty', 'lipid', 'HIF'],
    "DNA Repair / Cell Cycle": ['DNA', 'cell cycle', 'checkpoint', 'repair', 'mitotic', 'p53', 'RB1', 'p53', 'RB1'],
    "Proteostasis / Stress": ['proteasome', 'heat shock', 'HSP', 'protein folding', 'chaperone', 'chaperones'],
}




