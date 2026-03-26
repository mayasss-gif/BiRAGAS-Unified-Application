#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Deconvolution Helper (DRY-RUN) — recursive folder scan + safer disease parsing + sample-type prompt in files mode

Vaults
------
SC_BASE     = SC_BASE
SC_DISEASE  = SC_BASE/SC_Disease_Dataset
SC_ORGAN    = SC_BASE/SC_Normal_Organ
"""

import os, re, glob, sys, json, traceback, difflib
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(__name__)

from openai import OpenAI

import pandas as pd

# ---------- Config ----------
BULK_EXTS   = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
SCRNA_EXTS  = (".h5ad",)

SC_BASE     = "SC_BASE"
SC_DISEASE  = os.path.join(SC_BASE, "SC_Disease_Dataset")
SC_ORGAN    = os.path.join(SC_BASE, "SC_Normal_Organ")

IMMUNE_LIKE = {"blood","pbmc","plasma"}
LOG_PATH    = "./deconv_selector.log"

# Canonical aliases (map each synonym → canonical disease)
ALIASES = {
    # =========================
    # Cancers / immune
    # =========================
    "pancreatic cancer": [
        "pdac", "pancreatic ductal adenocarcinoma",
        "pancreatic adenocarcinoma", "malignant pancreatic neoplasm",
        "pancreas carcinoma", "pancreatic carcinoma"
    ],
    "glioblastoma": [
        "gbm", "glioblastoma multiforme", "astrocytoma grade iv", "high grade glioma"
    ],
    "lung adenocarcinoma": [
        "luad", "nsclc", "lung cancer", "non small cell lung cancer",
        "non-small cell lung cancer", "pulmonary adenocarcinoma", "non small cell lung carcinoma"
    ],
    "breast cancer": [
        "brca", "tnbc", "triple-negative breast cancer", "triple negative breast cancer",
        "er+ breast cancer", "luminal a breast cancer", "luminal b breast cancer",
        "her2-positive breast cancer", "mammary carcinoma", "breast carcinoma"
    ],
    "colorectal cancer": [
        "crc", "coad", "read", "colorectal carcinoma",
        "colon cancer", "rectal cancer", "bowel cancer",
        "large intestine carcinoma", "colonic adenocarcinoma"
    ],
    "melanoma": [
        "cutaneous melanoma", "skin melanoma", "malignant melanoma", "metastatic melanoma"
    ],
    "cervical cancer": [
        "cervical carcinoma", "cxca", "cervix cancer", "cervix carcinoma",
        "uterine cervix carcinoma", "cervical adenocarcinoma"
    ],

    # =========================
    # Organ / systemic diseases
    # =========================
    "acute kidney injury": [
        "aki", "kidney injury", "acute renal failure",
        "acute kidney failure", "acute renal injury", "arf",
        "acute renal insufficiency", "acute renal dysfunction"
    ],
    "kidney disease": [
        "renal disease", "ckd", "chronic kidney disease",
        "chronic renal failure", "renal insufficiency",
        "renal dysfunction", "nephropathy", "kidney disorder",
        "renal impairment", "chronic renal disease"
    ],
    "myocardial infarction": [
        "mi", "heart attack", "acute myocardial infarction",
        "cardiac infarction", "stemi", "nstemi",
        "acute coronary syndrome", "acs", "acute mi"
    ],
    "cardiomyopathy": [
        "cmp", "hf", "heart failure", "dilated cardiomyopathy",
        "hypertrophic cardiomyopathy", "restrictive cardiomyopathy",
        "ischemic cardiomyopathy", "arrhythmogenic cardiomyopathy",
        "noncompaction cardiomyopathy", "lv noncompaction",
        "cardiac muscle disease"
    ],
    "alzheimer's disease": [
        "alzheimers", "alzheimer", "alz", "alzheimer disease",
        "senile dementia of alzheimer type", "sdAT",
        "late-onset alzheimers", "early-onset alzheimers",
        "alzheimers type dementia", "dementia alzheimer type"
        # NOTE: intentionally excludes ambiguous 2-letter "ad"
    ],
    "multiple sclerosis": [
        "ms", "relapsing-remitting ms", "rrms",
        "primary progressive ms", "ppms",
        "secondary progressive ms", "spms",
        "demyelinating disease", "cns demyelination",
        "multiple sclerosis disease", "ms disorder"
    ],
    "rheumatoid arthritis": [
        "ra", "seropositive ra", "seronegative ra",
        "autoimmune arthritis", "rheumatoid disease",
        "arthritis rheumatoid", "chronic rheumatoid arthritis"
    ],
    "systemic lupus erythematosus": [
        "sle", "lupus", "lupus erythematosus",
        "systemic lupus", "sle disorder",
        "autoimmune lupus", "lupus systemic",
        "sle disease", "sle syndrome"
    ],

    # =========================
    # Respiratory / lung
    # =========================
    "chronic obstructive pulmonary disease": [
        "copd", "chronic bronchitis", "emphysema",
        "chronic obstructive airway disease", "chronic lung disease"
    ],
    "interstitial lung disease": [
        "ild", "dpld", "diffuse parenchymal lung disease", "interstitial pneumonia"
    ],
    "pulmonary fibrosis": [
        "pf", "idiopathic pulmonary fibrosis", "ipf", "lung fibrosis"
    ],
    "pneumonia": [
        "pna", "lung infection", "pulmonary infection",
        "bacterial pneumonia", "viral pneumonia", "aspiration pneumonia"
    ],

    # =========================
    # Endocrine / metabolic
    # =========================
    "type 1 diabetes mellitus": [
        "t1d", "type i diabetes", "insulin-dependent diabetes",
        "iddm", "juvenile diabetes", "type 1 dm"
    ],
    "type 2 diabetes mellitus": [
        "t2d", "type ii diabetes", "non-insulin dependent diabetes",
        "t2dm", "adult-onset diabetes", "niddm"
    ],
    "cystic fibrosis": [
        "cf", "cftr-related disorder", "cftr disease", "mucoviscidosis"
    ],

    # =========================
    # Neurological / neurodegenerative
    # =========================
    "amyotrophic lateral sclerosis": [
        "als", "lou gehrig disease", "motor neuron disease", "mnd"
    ],
    "frontotemporal dementia": [
        "ftd", "ftld", "frontotemporal lobar degeneration", "frontotemporal disorder"
    ],
    "lewy body dementia": [
        "dlb", "dementia with lewy bodies", "lewy body disease"
    ],
    "pick disease": [
        "pid", "picks disease", "pick's disease", "pick type ftd"
    ],
    "progressive supranuclear palsy": [
        "psp", "psp (steele–richardson–olszewski)"
    ],
    "temporal lobe epilepsy": [
        "tle", "mesial temporal lobe epilepsy", "temporal epilepsy"
    ],

    # =========================
    # Autoimmune / rheum / skin
    # =========================
    "sjogren syndrome": [
        "sjogren's syndrome", "sjogrens", "sicca", "sicca syndrome"
    ],
    "localized scleroderma": [
        "morphea", "morphea (localized scleroderma)"
    ],
    "keloid": [
        "keloid scar", "keloid disorder", "keloid disease"
    ],

    # =========================
    # Eye / vision
    # =========================
    "age-related macular degeneration": [
        "amd", "armd", "age related macular degeneration",
        "dry amd", "wet amd", "neovascular amd"
    ],
    "cataract": [
        "lens cataract", "senile cataract", "age-related cataract"
    ],

    # =========================
    # General / umbrella
    # =========================
    "heart disorder": [
        "cardiac disease", "cardiovascular disease", "cvd",
        "heart disease", "cardiac disorder"
    ],
    "respiratory system disorder": [
        "respiratory disease", "lung disease", "pulmonary disease"
    ],
    "digestive system disorder": [
        "gi disorder", "gastrointestinal disease", "gastrointestinal disorder"
    ],
    "hematologic disorder": [
        "blood disease", "hematologic condition", "blood disorder"
    ],
    "injury": [
        "trauma", "bodily injury", "tissue injury", "wound"
    ],
}


# organ map
# === Organ map (expanded and normalized) ===
DISEASE_TO_ORGAN = {
    # --- Brain / nervous system ---
    "glioblastoma": "brain",
    "alzheimer's disease": "brain", "alzheimers": "brain",
    "multiple sclerosis": "brain", "ms": "brain",
    "epilepsy": "brain", "temporal lobe epilepsy": "brain",
    "frontotemporal dementia": "brain", "ftd": "brain",
    "pick disease": "brain", "lewy body dementia": "brain",
    "amyotrophic lateral sclerosis": "brain", "als": "brain",

    # --- Heart / cardiovascular ---
    "myocardial infarction": "heart", "mi": "heart", "stemi": "heart", "nstemi": "heart",
    "heart attack": "heart", "cardiac infarction": "heart",
    "heart failure": "heart", "hf": "heart",
    "cardiomyopathy": "heart", "cmp": "heart", "dilated cardiomyopathy": "heart",
    "hypertrophic cardiomyopathy": "heart", "arrhythmogenic right ventricular cardiomyopathy": "heart",

    # --- Kidney / renal ---
    "acute kidney injury": "kidney", "aki": "kidney",
    "acute renal failure": "kidney", "chronic kidney disease": "kidney",
    "ckd": "kidney", "renal disease": "kidney", "kidney disease": "kidney",
    "nephropathy": "kidney", "wilms tumor": "kidney",

    # --- Lung / respiratory ---
    "lung adenocarcinoma": "lung", "nsclc": "lung", "luad": "lung",
    "lung cancer": "lung", "non-small cell lung carcinoma": "lung",
    "pulmonary fibrosis": "lung", "ipf": "lung",
    "chronic obstructive pulmonary disease": "lung", "copd": "lung",
    "pneumonia": "lung", "pna": "lung", "interstitial lung disease": "lung",
    "pulmonary emphysema": "lung",

    # --- Digestive / GI ---
    "colorectal cancer": "colon", "crc": "colon",
    "colon cancer": "colon", "rectal cancer": "rectum",
    "barrett esophagus": "esophagus",
    "crohn's disease": "intestine", "ulcerative colitis": "colon",
    "gastric cancer": "stomach", "stomach cancer": "stomach",
    "pancreatic cancer": "pancreas", "pdac": "pancreas",
    "hepatitis": "liver", "liver disease": "liver",

    # --- Endocrine / metabolic ---
    "type 1 diabetes mellitus": "pancreas",
    "type 2 diabetes mellitus": "pancreas",
    "cystic fibrosis": "lung",

    # --- Skin / soft tissue ---
    "melanoma": "skin", "cutaneous melanoma": "skin",
    "basal cell carcinoma": "skin", "bcc": "skin",
    "squamous cell carcinoma": "skin", "scc": "skin",
    "keloid": "skin",

    # --- Reproductive ---
    "breast cancer": "breast", "brca": "breast",
    "prostate cancer": "prostate",
    "cervical cancer": "cervix", "cervix cancer": "cervix",

    # --- Immune / rheumatic ---
    "rheumatoid arthritis": "synovium", "ra": "synovium",
    "systemic lupus erythematosus": "blood", "sle": "blood",
    "sjogren syndrome": "salivary gland",

    # --- Eye / vision ---
    "age-related macular degeneration": "eye",
    "cataract": "eye",
}

ORGAN_KEYWORDS = {
    "brain": [
        "cns", "central nervous system", "neural", "neuronal", "cerebral",
        "cerebrum", "cerebellum", "brainstem", "spinal cord", "spine", "gyrus",
        "gray matter", "white matter", "glia", "astrocyte", "microglia"
    ],
    "eye": [
        "ocular", "retina", "retinal", "choroid", "cornea", "lens", "macula",
        "uvea", "optic", "conjunctiva", "vitreous"
    ],
    "ear": [
        "inner ear", "utricle", "saccule", "crista ampullaris", "cochlea", "vestibular"
    ],
    "tongue": [
        "lingual", "taste bud", "oral tongue"
    ],
    "salivary gland": [
        "salivary_gland", "saliva gland", "parotid", "submandibular", "sublingual"
    ],
    "blood": [
        "peripheral blood", "pbmc", "whole blood", "leukocyte", "immune blood"
    ],
    "bone marrow": [
        "bone_marrow", "marrow", "hematopoietic bone marrow", "bm"
    ],
    "thymus": [
        "thymic"
    ],
    "spleen": [
        "splenic"
    ],
    "lymph node": [
        "lymph_node", "lymph nodes", "ln", "lymphoid"
    ],
    "vasculature": [
        "vascular", "blood vessel", "vessel", "artery", "vein", "endothelium",
        "endothelial", "microvasculature", "capillary"
    ],
    "heart": [
        "cardiac", "myocardium", "myocardial", "endocardium", "epicardium",
        "ventricle", "atria", "atrial", "sinoatrial", "sa node", "av node"
    ],
    "lung": [
        "pulmonary", "airway", "alveoli", "alveolar", "bronchus", "bronchi",
        "bronchioles", "trachea", "tracheal"
    ],
    "trachea": [
        "windpipe"
    ],
    "skin": [
        "cutaneous", "dermis", "epidermis", "integument", "keratinocyte"
    ],
    "fat": [
        "adipose", "subcutaneous fat", "visceral fat", "white adipose", "brown adipose"
    ],
    "muscle": [
        "skeletal muscle", "striated muscle", "smooth muscle", "myocyte", "myofiber",
        "diaphragm", "pelvic diaphragm", "abdominal muscle", "psoas"
    ],
    "breast": [
        "mammary", "mammary gland", "lactiferous", "ductal epithelium"
    ],
    "liver": [
        "hepatic", "hepatocyte", "bile duct", "cholangiocyte"
    ],
    "pancreas": [
        "exocrine pancreas", "endocrine pancreas", "islet", "islets of langerhans",
        "acinar", "ductal", "beta cell", "alpha cell"
    ],
    "stomach": [
        "gastric", "antrum", "fundus", "pylorus"
    ],
    "small intestine": [
        "small_intestine", "ileum", "jejunum", "duodenum", "enteric", "intestinal (small)"
    ],
    "large intestine": [
        "large_intestine", "colon", "colonic", "proximal colon", "distal colon",
        "ascending colon", "transverse colon", "descending colon", "sigmoid colon"
    ],
    "colon": [
        "colonic", "large bowel"  # kept for direct colon matching
    ],
    "rectum": [
        "rectal", "rectal mucosa"
    ],
    "esophagus": [
        "oesophagus", "esophageal", "oesophageal", "barrett oesophagus"
    ],
    "intestine": [
        "gut", "bowel", "enteric", "gastrointestinal", "gi tract"
    ],
    "kidney": [
        "renal", "renal cortex", "cortex of kidney", "medulla", "nephron",
        "proximal tubule", "distal tubule", "collecting duct", "glomerulus", "podocyte"
    ],
    "bladder": [
        "urothelium", "urinary bladder"
    ],
    "prostate": [
        "prostate gland", "prostatic"
    ],
    "ovary": [
        "ovarian", "follicle", "oocyte", "corpus luteum"
    ],
    "uterus": [
        "endometrium", "myometrium", "cervix uteri"  # cervix separately below too
    ],
    "cervix": [
        "cervix", "cervical", "cervical epithelium"
    ],
    "testis": [
        "testes", "testicular", "seminiferous tubule", "leydig", "spermatogonia"
    ],
    "germline": [
        "germ line", "germ cells", "gamete", "spermatocyte", "oocyte"
    ],
    "salivary/oral": [
        "oral cavity", "gingiva", "buccal", "palate", "tongue epithelium"
    ],
    "endocrine": [
        "hormonal", "endocrine tissue", "glandular"
    ],
    "immune": [
        "immune system", "immune", "hematopoietic"
    ],
    "respiratory": [
        "respiratory system", "respiratory tract", "airway"
    ],
}


# Build reverse lookup (token → canonical)
CANON = {}
for canon, syns in ALIASES.items():
    CANON[canon] = canon
    for s in syns:
        CANON[s] = canon

# ---------- Utils ----------
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# ---------- Helpers ----------
def norm(s: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", s.strip().lower()) if s else None

def extract_sample_type_from_filename(filename: str) -> Optional[str]:
    """
    Extract sample_type from filename if present.
    Looks for common sample type keywords: blood, pbmc, plasma, tissue, biopsy, ffpe
    
    Returns the detected sample_type or None if not found.
    """
    filename_lower = re.sub(r"[\s_]+", " ", filename.lower())
    
    # Define sample type keywords (in order of specificity)
    sample_type_keywords = {
        "blood": ["blood"],
        "pbmc": ["pbmc"],
        "plasma": ["plasma"],
        "tissue": ["tissue"],
        "biopsy": ["biopsy"],
        "ffpe": ["ffpe"]
    }
    
    # Check for each sample type (most specific first)
    for stype, keywords in sample_type_keywords.items():
        if any(kw in filename_lower for kw in keywords):
            return stype
    
    return None

def sample_types_compatible(requested: Optional[str], file_sample_type: Optional[str]) -> bool:
    """
    Check if requested sample_type is compatible with the sample_type found in filename.
    
    Compatible cases:
    - Both None/unknown: compatible
    - Requested is None/unknown: compatible (no constraint)
    - File has no sample_type: compatible
    - Both are in same category:
      * Immune-like: blood, pbmc, plasma
      * Tissue-like: tissue, biopsy, ffpe
    - Exact match: compatible
    
    Contradictory cases:
    - Requested is immune-like but file is tissue-like (or vice versa)
    
    Returns True if compatible, False if contradictory.
    """
    if not requested or requested == "unknown":
        return True  # No constraint, always compatible
    
    if not file_sample_type:
        return True  # File doesn't specify, assume compatible
    
    requested_norm = norm(requested)
    file_norm = norm(file_sample_type)
    
    if requested_norm == file_norm:
        return True  # Exact match
    
    # Define categories
    immune_like = {"blood", "pbmc", "plasma"}
    tissue_like = {"tissue", "biopsy", "ffpe"}
    
    requested_cat = None
    if requested_norm in immune_like:
        requested_cat = "immune"
    elif requested_norm in tissue_like:
        requested_cat = "tissue"
    
    file_cat = None
    if file_norm in immune_like:
        file_cat = "immune"
    elif file_norm in tissue_like:
        file_cat = "tissue"
    
    # If both have categories and they're different, it's contradictory
    if requested_cat and file_cat and requested_cat != file_cat:
        return False
    
    # Otherwise compatible (same category or one is unknown)
    return True

def robust_disease_patterns(disease: Optional[str], aliases: List[str]) -> List[str]:
    """
    Generate robust search patterns for disease names that handle:
    - Case variations (crest_syndrome, Crest Syndrome, CREST_SYNDROME)
    - Underscore vs space variations (crest_syndrome vs crest syndrome)
    - Multiple word variations
    
    Returns a list of normalized search patterns (lowercase, space-normalized).
    The search function will handle matching these against filenames with various separators.
    """
    patterns = []
    all_names = [disease] if disease else []
    all_names.extend(aliases or [])
    
    for name in all_names:
        if not name:
            continue
        # Normalize: lowercase, collapse spaces/underscores to single space
        normalized = re.sub(r"[\s_]+", " ", name.strip().lower())
        if normalized:
            patterns.append(normalized)
    
    # Return unique patterns
    return list(dict.fromkeys([p for p in patterns if p]))

# ---------- NEW FLOW: In-house Transcriptomic data (no GEO JSON) ----------
def mode_inhouse_flow(
    count_path: str,
    sample_type: str,
    disease_name: str,
    meta_path: Optional[str] = None,
    sc_base_dir: Optional[str] = None
):
    """
    Runs the decision engine for in-house transcriptomic data.
    
    All required parameters must be provided. No interactive prompting.
    
    Parameters
    ----------
    count_path : str
        Full path to the count/expression matrix file (required).
    sample_type : str
        Sample type: one of "blood", "pbmc", "plasma", "tissue", "biopsy", "ffpe" (required).
    disease_name : str
        Disease name (e.g., "systemic lupus erythematosus", "lupus", "SLE") (required).
    meta_path : Optional[str]
        Full path to the metadata file (optional).
    sc_base_dir : Optional[str]
        Base directory containing SC_Disease_Dataset and SC_Normal_Organ subdirectories.
        If None, uses the default SC_BASE path.
    """
    print("\n[IN-HOUSE MODE] Use your local bulk data .")
    
    # Log received parameters for debugging
    logger.info(
        f"mode_inhouse_flow called with parameters:\n"
        f"  count_path: {count_path}\n"
        f"  sample_type: {repr(sample_type)} (type: {type(sample_type)})\n"
        f"  disease_name: {repr(disease_name)} (type: {type(disease_name)})\n"
        f"  meta_path: {meta_path}\n"
        f"  sc_base_dir: {sc_base_dir}"
    )
    
    # --- validate required inputs ---
    if not count_path:
        logger.error("count_path is missing or empty")
        raise ValueError("count_path is required")
    if not os.path.isfile(count_path):
        logger.error(f"Count file not found: {count_path}")
        raise FileNotFoundError(f"Count file not found: {count_path}")
    
    if not sample_type:
        logger.error(f"sample_type is missing or empty. Received: {repr(sample_type)}")
        raise ValueError("sample_type is required")
    sample_type_in = sample_type.strip().lower()
    if sample_type_in not in {"blood","pbmc","plasma","tissue","biopsy","ffpe"}:
        logger.error(f"Invalid sample_type: {sample_type}. Must be one of: blood, pbmc, plasma, tissue, biopsy, ffpe")
        raise ValueError(f"Invalid sample_type: {sample_type}. Must be one of: blood, pbmc, plasma, tissue, biopsy, ffpe")
    
    if not disease_name:
        logger.error(
            f"disease_name is missing or empty.\n"
            f"  Received value: {repr(disease_name)}\n"
            f"  Type: {type(disease_name)}\n"
            f"  Function parameters: count_path={count_path}, sample_type={sample_type}"
        )
        raise ValueError("disease_name is required")
    disease_name = disease_name.strip()
    logger.info(f"Validated disease_name: {disease_name}")

    # --- validate optional metadata file ---
    if meta_path:
        if not os.path.isfile(meta_path):
            print(f"Warning: Metadata file not found, ignoring: {meta_path}")
            meta_path = None

    # --- minimal meta hint for metadata file ---
    # Trust upstream validation (from deg_file_validator_agent)
    meta_hint: Dict = {"source": "in-house"}
    if meta_path:
        try:
            meta_df = read_table_any(meta_path)
            # Log metadata structure (no rigid validation - upstream already validated)
            meta_hint["meta_status"] = "OK"
            meta_hint["columns"] = list(meta_df.columns)
            meta_hint["num_samples"] = len(meta_df)
            print(f"✅ Metadata loaded: {len(meta_df)} samples, columns: {list(meta_df.columns)}")
        except Exception as e:
            meta_hint["meta_status"] = f"Unreadable ({e})"
    
    # --- normalize disease via LLM/heuristics ---
    llm_norm = llm_normalize_context(
        user_disease_input=disease_name,
        filename_texts=[count_path],
        geo_meta=meta_hint
    )
    llm_norm["sample_type"] = sample_type_in  # honor explicit sample type

    # --- find any h5ad references in vaults (by disease/organ) ---
    sc_refs = locate_sc_reference(
        llm_norm.get("disease_name"),
        llm_norm.get("aliases", []),
        llm_norm.get("organ"),
        sample_type=llm_norm.get("sample_type"),
        sc_base_dir=sc_base_dir
    )

    # --- choose tool & brief ---
    tool, reason, rule = choose_tool([count_path], sc_refs, llm_norm, meta_hint)
    dz_brief = disease_brief_text(llm_norm.get("disease_name"), llm_norm.get("organ"), meta_hint)

    # --- print result (same format) ---
    print("\n================= DRY-RUN RESULT =================")
    print(f"Selected tool : {tool}")
    print(f"Justification : {reason}")
    print("Disease Brief")
    print(dz_brief or "  • (no additional details)")
    print("--------------------------------------------------")
    print("Decision Trace")
    print(f"  • Count file       : {count_path}")
    if meta_path: print(f"  • Metadata file    : {meta_path} ({meta_hint.get('meta_status','-')})")
    print(f"  • Disease (input)  : {disease_name or '(unknown)'}")
    print(f"  • Disease (LLM)    : {llm_norm.get('disease_name') or '(unknown)'}")
    print(f"  • Family (LLM)     : {llm_norm.get('family')}")
    print(f"  • Organ (LLM)      : {llm_norm.get('organ') or '-'}")
    print(f"  • Sample type      : {llm_norm.get('sample_type')}")
    print(f"  • sc refs matched  : {len(sc_refs)}")
    print(f"  • Rule fired       : {rule}")
    print("--------------------------------------------------")
    print("sc refs (.h5ad)")
    for p in sc_refs: print(f"  - {p}")
    print("--------------------------------------------------")
    print("Note: DRY-RUN finished. No external tools were executed.\n")

    return {
        "tool_selected": tool,
        "justification": reason,
        "disease_brief": dz_brief,
        "decision_trace": {
            "count_file": count_path,
            "metadata_file": meta_path,
            "disease_input": disease_name,
            "disease_llm": llm_norm.get("disease_name"),
            "family": llm_norm.get("family"),
            "organ": llm_norm.get("organ"),
            "sample_type": llm_norm.get("sample_type"),
            "sc_refs": sc_refs,
            "rule_fired": rule
        }
    }



def list_from_user_path_input(path_input: str) -> List[str]:
    # Accept folder OR semicolon-separated file list; keep Windows quotes safe
    pin = path_input.strip().strip('"').strip("'")
    if os.path.isdir(pin):
        paths = glob.glob(os.path.join(pin, "*"))
        return [p for p in paths if p.lower().endswith(BULK_EXTS + SCRNA_EXTS)]
    else:
        parts = [x.strip().strip('"').strip("'") for x in pin.split(";") if x.strip()]
        return parts

def read_table_any(path: str) -> pd.DataFrame:
    pl = path.lower()
    if pl.endswith(".csv"):
        return pd.read_csv(path, sep=",")
    if pl.endswith(".tsv") or pl.endswith(".txt"):
        # auto-detect: try tab first; if single column, try comma
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
        return df
    if pl.endswith(".xlsx") or pl.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl" if pl.endswith(".xlsx") else None)
    raise ValueError(f"Unsupported file type: {path}")


def detect_bulk_and_sc(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    bulk = [f for f in files if f.lower().endswith(BULK_EXTS)]
    sc   = [f for f in files if f.lower().endswith(SCRNA_EXTS)]
    meta = [f for f in bulk if re.search(r"(meta|pheno|design)", os.path.basename(f), re.I)]
    return list(dict.fromkeys(bulk)), list(dict.fromkeys(sc)), list(dict.fromkeys(meta))

def try_read_sidecar(any_path: str) -> Dict:
    """
    Look for a JSON in the same folder that smells like GEO metadata.
    We take the first *.json we find and pull a few keys safely.
    """
    folder = os.path.dirname(any_path)
    out = {}
    try:
        for j in glob.glob(os.path.join(folder, "*.json")):
            with open(j, "r", encoding="utf-8") as fh:
                jj = json.load(fh)
            for k in ["disease","filter","dataset_id","source","title","description","overall_design","sample_count"]:
                if k in jj and jj[k]:
                    out[k] = jj[k]
            if out:  # take the first useful json
                break
    except Exception:
        pass
    return out

def search_h5ad(folder: str, needles: List[str]) -> List[str]:
    if not folder or not os.path.isdir(folder): return []
    hits = []
    for nd in needles:
        if not nd: continue
        patt = f"**/*{nd}*.h5ad"
        hits.extend(glob.glob(os.path.join(folder, patt), recursive=True))
    return list(dict.fromkeys(hits))

def search_h5ad_robust(folder: str, disease_patterns: List[str], sample_type: Optional[str] = None) -> List[str]:
    """
    Search for .h5ad files using robust disease name patterns.
    Optionally filter by sample_type (prioritizes matches containing sample_type).
    Excludes files where the filename's sample_type contradicts the requested sample_type.
    
    Handles variations like:
    - Case: "crest_syndrome", "Crest Syndrome", "CREST_SYNDROME"
    - Separators: "crest_syndrome" vs "crest syndrome"
    
    Parameters
    ----------
    folder : str
        Directory to search in
    disease_patterns : List[str]
        List of disease name patterns (normalized, lowercase)
    sample_type : Optional[str]
        Sample type to prioritize (e.g., "blood", "tissue")
        Files with contradictory sample_types will be excluded.
    
    Returns
    -------
    List[str]
        List of matching file paths, prioritized by sample_type match if provided.
        Files with contradictory sample_types are excluded.
    """
    if not folder or not os.path.isdir(folder): 
        return []
    
    all_hits = []
    prioritized_hits = []
    
    # Get all .h5ad files in the directory
    all_files = glob.glob(os.path.join(folder, "**/*.h5ad"), recursive=True)
    
    # Normalize sample_type for matching
    sample_type_norm = norm(sample_type) if sample_type else None
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        filename_lower = filename.lower()
        # Normalize filename: collapse spaces and underscores to single space for comparison
        filename_normalized = re.sub(r"[\s_]+", " ", filename_lower).strip()
        
        # Check if any disease pattern matches
        matched = False
        for pattern in disease_patterns:
            if not pattern:
                continue
            # Normalize pattern the same way
            pattern_normalized = re.sub(r"[\s_]+", " ", pattern.lower()).strip()
            
            # Check if normalized pattern appears in normalized filename
            # This handles: "crest syndrome" matching "crest_syndrome", "Crest Syndrome", etc.
            if pattern_normalized in filename_normalized:
                matched = True
                break
        
        if matched:
            # Extract sample_type from filename
            file_sample_type = extract_sample_type_from_filename(filename)
            
            # Check if file's sample_type contradicts requested sample_type
            if not sample_types_compatible(sample_type, file_sample_type):
                # Skip this file - sample_type contradiction
                continue
            
            # Check if sample_type is present in filename (if provided)
            if sample_type_norm:
                # Create variants for sample_type matching (space, underscore, hyphen)
                sample_variants = [
                    sample_type_norm,
                    sample_type_norm.replace(" ", "_"),
                    sample_type_norm.replace(" ", "-"),
                    sample_type_norm.replace("_", " "),
                    sample_type_norm.replace("-", " ")
                ]
                # Check if any variant appears in the normalized filename
                if any(var in filename_normalized for var in sample_variants if var):
                    prioritized_hits.append(file_path)
                else:
                    all_hits.append(file_path)
            else:
                all_hits.append(file_path)
    
    # Return prioritized matches first, then other matches
    return list(dict.fromkeys(prioritized_hits + all_hits))

def locate_sc_reference(disease: Optional[str], aliases: List[str], organ: Optional[str], sample_type: Optional[str] = None, sc_base_dir: Optional[str] = None) -> List[str]:
    """
    Locate single-cell reference .h5ad files by disease or organ.
    Prioritizes disease-specific matches with sample_type, then disease matches without sample_type,
    then falls back to organ-based matches only if no disease matches found.
    
    If disease_name matches but sample_type contradicts (e.g., requested "blood" but file has "tissue"),
    the file is excluded and the function tries aliases, then organ as fallback.
    
    Parameters
    ----------
    disease : Optional[str]
        Disease name to search for.
    aliases : List[str]
        List of disease aliases to search for.
    organ : Optional[str]
        Organ name to search for (fallback if no disease match).
    sample_type : Optional[str]
        Sample type to prioritize (e.g., "blood", "tissue", "pbmc").
        Files with contradictory sample_types will be excluded.
    sc_base_dir : Optional[str]
        Base directory containing SC_Disease_Dataset and SC_Normal_Organ subdirectories.
        If None, uses the default SC_BASE path.
    
    Returns
    -------
    List[str]
        List of matching .h5ad file paths, prioritized by disease+sample_type match.
    """
    # Use provided base directory or fall back to default
    if sc_base_dir:
        sc_disease_dir = os.path.join(sc_base_dir, "SC_Disease_Dataset")
        sc_organ_dir = os.path.join(sc_base_dir, "SC_Normal_Organ")
    else:
        sc_disease_dir = SC_DISEASE
        sc_organ_dir = SC_ORGAN
    
    h5 = []
    
    # Step 1: Search disease directory with disease name first
    if disease:
        disease_patterns = robust_disease_patterns(disease, [])
        if disease_patterns:
            h5 = search_h5ad_robust(sc_disease_dir, disease_patterns, sample_type=sample_type)
    
    # Step 2: If no matches from disease name (or all contradicted by sample_type), try aliases
    if not h5 and aliases:
        alias_patterns = robust_disease_patterns(None, aliases)
        if alias_patterns:
            h5 = search_h5ad_robust(sc_disease_dir, alias_patterns, sample_type=sample_type)
    
    # Step 3: Only fall back to organ directory if NO disease matches found
    if not h5 and organ:
        organ_normalized = norm(organ)
        if organ_normalized:
            h5 = search_h5ad(sc_organ_dir, [organ_normalized])
    
    return list(dict.fromkeys(h5))

# ---------- LLM-first disease normalization ----------
def llm_normalize_context(
    user_disease_input: Optional[str],
    filename_texts: List[str],
    geo_meta: Dict
) -> Dict:
    """
    Always try LLM normalization first:
      Input:
        - user_disease_input (as typed)
        - filename texts (basenames)
        - geo_meta (title/description/overall_design/filter/disease/source)
      Output:
        disease_name, organ, sample_type, family, aliases, confidence, reasoning
    Heuristic fallback if no OPENAI_API_KEY or client error.
    """
    meta = {
        "title": geo_meta.get("title", ""),
        "description": geo_meta.get("description", ""),
        "overall_design": geo_meta.get("overall_design", ""),
        "filter": geo_meta.get("filter", ""),
        "disease": geo_meta.get("disease", ""),
        "source": geo_meta.get("source", ""),
    }
    filenames = [os.path.splitext(os.path.basename(x))[0].replace("-", " ").replace(".", " ") for x in filename_texts]
    key = os.getenv("OPENAI_API_KEY")
    if OpenAI is not None and key:
        try:
            client = OpenAI(api_key=key)
            prompt = {
                "user_disease_input": user_disease_input or "",
                "filenames": filenames[:30],
                "geo_meta": meta
            }
            sys_msg = (
                "You normalize disease context for transcriptomic deconvolution. "
                "Use medical knowledge. Output STRICT JSON only with keys: "
                "disease_name, organ, sample_type, family, aliases, confidence, reasoning. "
                "sample_type ∈ {blood,pbmc,plasma,tissue,biopsy,ffpe,unknown}; "
                "family ∈ {immune,solid_tumor,fibrosis,neuro,cardio,metabolic,infectious,other}."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":sys_msg},
                    {"role":"user","content":json.dumps(prompt, ensure_ascii=False)}
                ],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                out = json.loads(m.group())
                out["disease_name"] = out.get("disease_name") or (geo_meta.get("disease") or user_disease_input)
                out["organ"] = (out.get("organ") or "").strip()
                st = (out.get("sample_type") or "unknown").lower()
                out["sample_type"] = st if st in {"blood","pbmc","plasma","tissue","biopsy","ffpe","unknown"} else "unknown"
                fam = (out.get("family") or "other").lower()
                out["family"] = fam if fam in {"immune","solid_tumor","fibrosis","neuro","cardio","metabolic","infectious","other"} else "other"
                out["aliases"] = out.get("aliases") or []
                out["confidence"] = out.get("confidence") or "medium"
                out["reasoning"] = out.get("reasoning") or ""
                # Build reverse lookup (token → canonical)
                canon = (out["disease_name"] or "").strip().lower()
                aliases = [a.strip().lower() for a in out["aliases"]]
                CANON = {canon: canon}
                for s in aliases:
                    CANON[s] = canon
                out["_canon_lookup"] = CANON  # keep for trace/debug
                return out
        except Exception:
            pass

    # Heuristic fallback (no API key)
    def _norm(s): return re.sub(r"\s+"," ",s.strip().lower()) if s else ""
    everything = " ".join([
        _norm(user_disease_input),
        _norm(" ".join(filenames)),
        _norm(meta.get("title","")),
        _norm(meta.get("description","")),
        _norm(meta.get("overall_design","")),
        _norm(meta.get("filter","")),
        _norm(meta.get("disease",""))
    ])
    sample_type = "blood" if any(k in everything for k in ["blood","pbmc","plasma","lymphocyte","peripheral"]) else "tissue"
    organ = None
    for org in ["heart","cardiac","myocardium","kidney","pancreas","lung","breast","colon","rectum","liver","brain","skin","cervix","synovium","prostate","ovary"]:
        if org in everything:
            organ = "heart" if org in {"cardiac","myocardium"} else org
            break
    if any(k in everything for k in ["sle","lupus","rheumatoid arthritis","ra","multiple sclerosis","ms","autoimmune"]):
        family = "immune"
    elif "cancer" in everything or "carcinoma" in everything or "sarcoma" in everything or "tumor" in everything or "tumour" in everything:
        family = "solid_tumor"
    elif "fibrosis" in everything or "wound" in everything or "cirrhosis" in everything:
        family = "fibrosis"
    elif any(k in everything for k in ["alzheimer","parkinson","als"]):
        family = "neuro"
    elif any(k in everything for k in ["myocardial infarction","cardiomyopathy","heart failure"]):
        family = "cardio"
    else:
        family = "other"
    disease_name = meta.get("disease") or user_disease_input or ""
    CANON = {}
    canon = (disease_name or "").strip().lower()
    if canon:
        CANON[canon] = canon
    return {
        "disease_name": disease_name,
        "organ": organ or "",
        "sample_type": sample_type,
        "family": family,
        "aliases": [],
        "confidence": "low",
        "reasoning": "Heuristic fallback (no API key).",
        "_canon_lookup": CANON
    }

# ---------- Disease brief (LLM optional) ----------
def disease_brief_text(disease_name: Optional[str],
                       organ: Optional[str],
                       meta_hint: Dict) -> str:
    """
    Returns a short multi-line summary. Tries LLM for a crisp 3-4 bullet abstract; otherwise heuristics.
    """
    title = meta_hint.get("title","").strip()
    desc  = meta_hint.get("description","").strip()
    overall = meta_hint.get("overall_design","").strip()
    src = meta_hint.get("source","").strip() or "local"
    key_ok = (OpenAI is not None) and bool(os.getenv("OPENAI_API_KEY"))

    if key_ok:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"""
Summarize in ≤4 bullets for a deconvolution operator.
INPUT:
- disease: {disease_name}
- likely organ: {organ}
- meta source: {src}
- title: {title[:250]}
- description: {desc[:600]}
- design: {overall[:250]}
Return plain text bullets, no preamble, no JSON.
"""
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a precise scientific writer. Output ≤4 short bullets."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2,
            )
            out = (r.choices[0].message.content or "").strip()
            if out:
                lines = [ln.strip(" -*\t") for ln in out.splitlines() if ln.strip()]
                bullets = "\n".join([f"  • {ln}" for ln in lines[:4]])
                return bullets
        except Exception:
            pass

    bits = []
    if disease_name:
        bits.append(f"{disease_name}")
    if organ:
        bits.append(f"Primary organ/tissue: {organ}")
    if src and (title or desc):
        bits.append(f"Source: {src}; Title: {title[:90] or '-'}")
    return "\n".join([f"  • {b}" for b in bits if b])

# ---------- Decision engine (LLM-driven) ----------
def choose_tool(
    bulk_files: List[str],
    sc_refs: List[str],
    llm_norm: Dict,
    meta_hint: Dict
) -> Tuple[str, str, str]:
    """
    Prioritize biologically appropriate method using LLM-normalized tags:
      family: immune|solid_tumor|fibrosis|neuro|cardio|metabolic|infectious|other
      sample_type: blood|pbmc|plasma|tissue|biopsy|ffpe|unknown
    """
    has_bulk = len(bulk_files) > 0
    has_sc   = len(sc_refs) > 0

    disease_name = (llm_norm.get("disease_name") or "").strip()
    organ        = (llm_norm.get("organ") or "").strip()
    family       = (llm_norm.get("family") or "other").lower()
    stype        = (llm_norm.get("sample_type") or "unknown").lower()

    txt = " ".join(str(meta_hint.get(k,"")) for k in ["filter","title","description","overall_design","source"]).lower()
    immune_like = stype in {"blood","pbmc","plasma"} or any(k in txt for k in ["blood","pbmc","plasma","lymphocyte","peripheral"])

    def _matched_sc() -> bool:
        if not has_sc: return False
        keys = [k for k in [re.sub(r"\s+"," ",disease_name.lower()), organ.lower()] if k]
        for p in sc_refs:
            b = os.path.basename(p).lower()
            if any(k and k in b for k in keys):
                return True
        return False

    matched_sc = _matched_sc()

    # IMMUNE family (blood/PBMC/plasma prioritized to CIBERSORTx)
    if family == "immune" and immune_like:
        if has_bulk and matched_sc:
            return ("BISQUE", "Immune disease with bulk + disease-matched .h5ad → BisqueRNA gives highest fidelity beyond LM22.", "RULE:immune+bulk+sc_matched→BISQUE")
        if has_bulk:
            return ("CIBERSORTx", "Immune/systemic context in blood/PBMC → CIBERSORTx + LM22 is most interpretable & comparable.", "RULE:immune+immune_like+bulk→CIBERSORTx")
        if matched_sc:
            return ("BISQUE (awaiting bulk)", "Matched .h5ad present; add bulk matrix to run BisqueRNA.", "RULE:immune+sc_only_matched→BISQUE_await_bulk")
        return ("CIBERSORTx (awaiting bulk)", "Immune-like context; provide bulk expression for CIBERSORTx.", "RULE:immune+no_bulk→CX_await_bulk")

    # SOLID TUMOR / FIBROSIS or tissue-like
    if family in {"solid_tumor","fibrosis"} or not immune_like:
        if has_bulk and matched_sc:
            return ("BISQUE", "Tissue bulk with disease/organ-matched .h5ad → BisqueRNA models immune + stromal compartments.", "RULE:tissue+bulk+sc_matched→BISQUE")
        if has_bulk:
            return ("xCell", "Tissue bulk without matched .h5ad → xCell (64-type) robustly captures immune + stromal enrichment.", "RULE:tissue+bulk_only→xCell")
        if matched_sc:
            return ("BISQUE (awaiting bulk)", "Matched .h5ad present; add bulk matrix to run BisqueRNA.", "RULE:tissue+sc_only_matched→BISQUE_await_bulk")
        return ("xCell (awaiting bulk)", "Tissue-like context; provide bulk expression to run xCell.", "RULE:tissue_no_bulk→xCell_await_bulk")

    # OTHER families: decide by sample type
    if immune_like:
        if has_bulk and matched_sc:
            return ("BISQUE", "Bulk + matched .h5ad in immune-rich context → BisqueRNA can capture disease-specific subsets.", "RULE:immune_like+bulk+sc_matched→BISQUE")
        if has_bulk:
            return ("CIBERSORTx", "Immune-rich bulk (blood/PBMC/plasma) → CIBERSORTx + LM22.", "RULE:immune_like+bulk→CIBERSORTx")
        if matched_sc:
            return ("BISQUE (awaiting bulk)", "Matched .h5ad present; add bulk matrix to run BisqueRNA.", "RULE:immune_like+sc_only_matched→BISQUE_await_bulk")
        return ("CIBERSORTx (awaiting bulk)", "Immune-like context; provide bulk expression for CIBERSORTx.", "RULE:immune_like_no_bulk→CX_await_bulk")

    # final fallbacks
    if has_bulk and has_sc:
        return ("BISQUE", "Bulk + .h5ad available → BisqueRNA.", "RULE:default_bulk+sc→BISQUE")
    if has_bulk:
        return ("xCell", "Bulk-only → xCell enrichment.", "RULE:default_bulk_only→xCell")
    if has_sc:
        return ("BISQUE (awaiting bulk)", "Provide bulk expression to pair with .h5ad for BisqueRNA.", "RULE:default_sc_only→BISQUE_await_bulk")
    return ("NEED_INPUT", "No bulk or scRNA reference found.", "RULE:default_no_input")



# ---------- Main ----------
def run_inhouse_deconvolution_analysis(
    count_path: str,
    sample_type: str,
    disease_name: str,
    meta_path: Optional[str] = None,
    sc_base_dir: Optional[str] = None
) -> None:
    """
    Run in-house deconvolution analysis workflow.
    
    This function orchestrates the complete deconvolution analysis pipeline for
    in-house transcriptomic data, including disease normalization, tool selection,
    and reference matching.
    
    All required parameters must be provided. No interactive prompting.
    
    Parameters
    ----------
    count_path : str
        Full path to the count/expression matrix file (e.g., "D:/proj/lupus.csv", 
        "C:/data/cervical-cancer_counts.tsv"). Required.
    sample_type : str
        Sample type: one of "blood", "pbmc", "plasma", "tissue", "biopsy", "ffpe". Required.
    disease_name : str
        Disease name (e.g., "systemic lupus erythematosus", "lupus", "SLE", 
        "pancreatic cancer", "PDAC"). Required.
    meta_path : Optional[str]
        Full path to the metadata file (e.g., "D:/proj/study_meta.tsv"). Optional.
    sc_base_dir : Optional[str]
        Base directory containing SC_Disease_Dataset and SC_Normal_Organ subdirectories
        where single-cell reference .h5ad files are stored. If None, uses the default
        SC_BASE path ("C:/Users/shahr/Downloads/Deconv-sc-fallback").
    
    Returns
    -------
    None
        Prints analysis results to stdout. This is a dry-run mode that does not
        execute external tools.
    
    Raises
    ------
    ValueError
        If any required parameter is missing or invalid.
    FileNotFoundError
        If the count file or metadata file (if provided) is not found.
    
    Examples
    --------
    >>> run_inhouse_deconvolution_analysis(
    ...     count_path="D:/proj/lupus.csv",
    ...     sample_type="blood",
    ...     disease_name="systemic lupus erythematosus",
    ...     meta_path="D:/proj/meta.tsv",
    ...     sc_base_dir="D:/data/single_cell_references"
    ... )
    """
    result = mode_inhouse_flow(
        count_path=count_path,
        sample_type=sample_type,
        disease_name=disease_name,
        meta_path=meta_path,
        sc_base_dir=sc_base_dir
    )
    return result

# if __name__ == "__main__":
#     try:
#         ans = run_inhouse_deconvolution_analysis(
#             count_path=r"C:\Ayass Bio Work\Agentic_AI_ABS\deconvolution_pipeline\Lupus.csv",
#             sample_type="blood",
#             disease_name="systemic lupus erythematosus",
#             meta_path=r"C:\Ayass Bio Work\Agentic_AI_ABS\deconvolution_pipeline\Lupus_meta.csv",
#             sc_base_dir=r"C:\Ayass Bio Work\Agentic_AI_ABS\deconvolution_pipeline\SC_BASE"
#         )
#         print(ans)

#     except KeyboardInterrupt:
#         print("\nAborted by user.")
#     except Exception as e:
#         print("Error:", e)
#         traceback.print_exc()
#         sys.exit(1)