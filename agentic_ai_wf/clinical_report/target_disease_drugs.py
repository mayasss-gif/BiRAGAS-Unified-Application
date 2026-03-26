import os
import pandas as pd
import requests
import time
from typing import Any, List, Dict, Optional
from threading import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

# Handle imports for both relative and absolute contexts
try:
    from .llm_client import LLMClient
except ImportError:
    from agentic_ai_wf.clinical_report.llm_client import LLMClient

# 🔒 Thread-safe rate limiting for API calls
_api_rate_limiter = Lock()
_last_api_call_time = 0.0
_min_interval_between_calls = 0.3  # 200 req/min max (safe limit)

# 🚦 Limit concurrent API requests across all threads
_api_semaphore = Semaphore(10)  # Max 10 concurrent API calls


def _rate_limited_api_call(func, *args, **kwargs):
    """
    Thread-safe rate limiting for API calls to prevent hitting limits.
    """
    global _last_api_call_time
    
    with _api_rate_limiter:
        elapsed = time.time() - _last_api_call_time
        if elapsed < _min_interval_between_calls:
            time.sleep(_min_interval_between_calls - elapsed)
        
        result = func(*args, **kwargs)
        _last_api_call_time = time.time()
        return result


def _retry_with_backoff(func, max_retries: int = 3, *args, **kwargs):
    """
    Retry API calls with exponential backoff for transient failures.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
            time.sleep(wait_time)
    return None


def _query_openfda_api(drug_name: str, timeout: int = 5) -> Dict[str, Any]:
    """
    OpenFDA API query with rate limiting and retry logic.
    Thread-safe for concurrent analysis runs.
    """
    def _make_request():
        url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f'openfda.brand_name:"{drug_name}" OR '
                     f'openfda.generic_name:"{drug_name}"',
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    
    try:
        # Limit concurrent requests
        with _api_semaphore:
            # Rate limit + retry
            data = _rate_limited_api_call(
                _retry_with_backoff, _make_request
            )
            
            if data:
                result = data.get("results", [{}])[0] if data else {}
                return result
    except Exception:
        pass
    
    return {}


def _query_pubchem_api(drug_name: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Fallback: PubChem API for drug information.
    More comprehensive than FDA but different data structure.
    """
    def _make_request():
        # First get compound ID
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/JSON"
        response = requests.get(
            url.format(drug_name), 
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if "PC_Compounds" in data and data["PC_Compounds"]:
            cid = data["PC_Compounds"][0].get("id", {}).get("id", {}).get("cid")
            if cid:
                # Get compound details
                detail_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
                detail_resp = requests.get(detail_url, timeout=timeout)
                detail_resp.raise_for_status()
                return detail_resp.json()
        return {}
    
    try:
        with _api_semaphore:
            data = _rate_limited_api_call(
                _retry_with_backoff, _make_request
            )
            if data:
                return data
    except Exception:
        pass
    
    return {}


def _check_disease_indication(
    drug_name: str, 
    disease_name: str
) -> bool:
    """
    Multi-source disease indication check with fallback.
    Tries FDA first, then PubChem for robustness.
    """
    # Try FDA first
    fda_data = _query_openfda_api(drug_name)
    
    if fda_data:
        indications = fda_data.get("indications_and_usage", [])
        if isinstance(indications, list):
            indications_text = " ".join(indications).lower()
        else:
            indications_text = str(indications).lower()
        
        disease_lower = disease_name.lower()
        disease_keywords = disease_lower.split()
        
        if any(kw in indications_text for kw in disease_keywords if len(kw) > 3):
            return True
    
    # Fallback to PubChem
    pubchem_data = _query_pubchem_api(drug_name)
    if pubchem_data:
        # Extract indication info from PubChem structure
        try:
            sections = pubchem_data.get("Record", {}).get("Section", [])
            for section in sections:
                if "Pharmacology and Biochemistry" in section.get("TOCHeading", ""):
                    # Search for indication text
                    info = str(section).lower()
                    disease_lower = disease_name.lower()
                    disease_keywords = disease_lower.split()
                    if any(kw in info for kw in disease_keywords if len(kw) > 3):
                        return True
        except Exception:
            pass
    
    return False

def _score_status(status_str: str) -> int:
    """Assign a score to drug status for prioritization."""
    if not status_str:
        return -1
    
    s = status_str.lower()

    # Highest priority: strictly approved
    if s.strip() == "approved":
        return 5

    # Withdrawn gets lowest
    if "withdrawn" in s:
        return 0

    # Still good if includes approved
    if "approved" in s:
        return 4

    # Investigational / experimental middle tier
    if "investigational" in s or "experimental" in s:
        return 2

    # Vet approved / illicit lowest non-withdrawn
    if "vet" in s or "illicit" in s:
        return 1

    return -1


bad_keywords = [
    "water", "fish oil", "lysine", "threonine", "caffeine", "ouabain",
    "calcium", "phosphate", "menthol", "cholecalciferol", "ergocalciferol",
    "vitamin", "supplement", "nutrient", "cream", "ointment", "gel", 
    "lotion", "patch", "topical"
]

# Non-drug product types to reject
non_drug_types = [
    "supplement", "vitamin", "mineral", "nutrient", "food", "solvent"
]


def _is_valid_drug(name: str, disease: str) -> bool:
    """
    Production-grade drug validation:
    - Multi-source fallback (FDA → PubChem)
    - Rate limiting (200 req/min)
    - Retry logic (3 attempts)
    - Thread-safe API calls
    """
    if not name:
        return False
    
    name_lower = name.lower()
    
    # Quick keyword rejection (microseconds)
    if any(k in name_lower for k in bad_keywords):
        return False
    
    # Multi-source validation with fallback
    is_valid = False
    
    try:
        # Primary: FDA API
        fda_data = _query_openfda_api(name)
        
        if fda_data:
            # Check product type
            product_type = str(fda_data.get("product_type", "")).lower()
            if any(non_type in product_type for non_type in non_drug_types):
                is_valid = False
            else:
                # Check if drug is FDA approved
                has_approval = bool(fda_data.get("openfda", {}))
                
                # Check disease indication (includes PubChem fallback)
                has_indication = _check_disease_indication(name, disease)
                
                is_valid = has_approval and has_indication
        else:
            # If FDA has no data, try PubChem as fallback
            pubchem_data = _query_pubchem_api(name)
            if pubchem_data:
                # Basic validation from PubChem
                has_indication = _check_disease_indication(name, disease)
                is_valid = has_indication
    
    except Exception as e:
        # Graceful degradation
        print(f"⚠️  Drug validation error for {name}: {e}")
        is_valid = False
    
    return is_valid


def batch_validate_drugs(
    drug_names: List[str], 
    disease: str, 
    max_workers: int = 10
) -> Dict[str, bool]:
    """
    Batch validation for hundreds of drugs in parallel.
    Maintains rate limits while maximizing throughput.
    
    Args:
        drug_names: List of drug names to validate
        disease: Disease name for indication check
        max_workers: Max parallel workers (default 10)
    
    Returns:
        Dict mapping drug_name -> is_valid
    """
    results = {}
    
    print(f"🔍 Validating {len(drug_names)} drugs for {disease}")
    
    # Validate all drugs in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_drug = {
            executor.submit(_is_valid_drug, name, disease): name
            for name in drug_names
        }
        
        for future in as_completed(future_to_drug):
            drug_name = future_to_drug[future]
            try:
                results[drug_name] = future.result()
            except Exception as e:
                print(f"❌ Batch validation error for {drug_name}: {e}")
                results[drug_name] = False
    
    return results


def _build_drug_summary_prompt(fields: Dict[str, Any], disease: str) -> str:
    return (
        f"You are generating a concise drug evidence summary for a "
        f"patient diagnosed with {disease}.\n\n"
        f"Drug information:\n"
        f"- Name: {fields.get('name', 'N/A')}\n"
        f"- Status: {fields.get('status', 'N/A')}\n"
        f"- Target Gene: {fields.get('target_gene', 'N/A')} "
        f"({fields.get('target_log2fc_display', 'N/A')})\n"
        f"- Role: {fields.get('role', 'N/A')}\n"
        f"- Sources: {fields.get('sources', 'N/A')}\n"
        f"- Associated Diseases: {fields.get('diseases', 'N/A')}\n\n"
        "Write 1–2 sentences summarizing this drug for the clinical report.\n"
        "Requirements:\n"
        "- Use only the provided information.\n"
        "- Mention the drug name, approval status, target gene, and log2FC.\n"
        "- Include disease association and sources if available.\n"
        "- Do not invent any information.\n"
        "- Exclude vitamins, minerals, dietary supplements, nutrients, "
        "food molecules, or solvents.\n"
        "- Be concise and clinically clear."
    )

def select_best_drugs(
    csv_path: str,
    disease: str,
    llm_client: LLMClient,
    patient_genes_path: str,
    patient_prefix: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    # Check if CSV file exists and is readable
    if not os.path.exists(csv_path):
        print(f"⚠️  Drug CSV not found: {csv_path}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"⚠️  Drug CSV is empty: {csv_path}")
            return []
    except pd.errors.EmptyDataError:
        print(f"⚠️  Drug CSV has no data/columns: {csv_path}")
        return []
    except Exception as e:
        print(f"⚠️  Error reading drug CSV {csv_path}: {e}")
        return []

    # Check patient genes file
    if not os.path.exists(patient_genes_path):
        print(f"⚠️  Patient genes file not found: {patient_genes_path}")
        return []
    
    try:
        patient_df = pd.read_csv(patient_genes_path)
        if patient_df.empty:
            print(f"⚠️  Patient genes file is empty: {patient_genes_path}")
            return []
    except pd.errors.EmptyDataError:
        print(f"⚠️  Patient genes file has no data: {patient_genes_path}")
        return []
    except Exception as e:
        print(f"⚠️  Error reading patient genes {patient_genes_path}: {e}")
        return []

    # Normalize
    df["status"] = df["status"].astype(str)
    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.lower()
    else:
        df["role"] = ""

    patient_df["Gene"] = patient_df["Gene"].astype(str).str.upper()

    # Synonym map
    synonym_map = {}
    if "HGNC_Synonyms" in patient_df.columns:
        for _, row in patient_df.iterrows():
            gene = row["Gene"].upper()
            for s in str(row.get("HGNC_Synonyms", "")).split(";"):
                s = s.strip().upper()
                if s:
                    synonym_map[s] = gene

    # Identify patient log2FC col
    log2fc_cols = [c for c in patient_df.columns if c.startswith(patient_prefix) and c.endswith("_log2FC")]
    log2fc_col = log2fc_cols[0] if log2fc_cols else None

    def get_log2fc(gene_symbol: str) -> Optional[float]:
        gene_symbol = str(gene_symbol).upper()
        if gene_symbol in patient_df["Gene"].values:
            row = patient_df.loc[patient_df["Gene"] == gene_symbol]
        elif gene_symbol in synonym_map:
            row = patient_df.loc[patient_df["Gene"] == synonym_map[gene_symbol]]
        else:
            return None
        if log2fc_col and log2fc_col in row:
            try:
                return float(row.iloc[0][log2fc_col])
            except Exception:
                return None
        return None

    # 🚫 Filter: allow only inhibitors / antagonists / blockers
    allowed_roles = ["inhibitor", "antagonist", "blocker", "negative modulator", "suppressor"]
    df = df[df["role"].apply(lambda r: any(keyword in r for keyword in allowed_roles))]

    if df.empty:
        print("No inhibitor/antagonist drugs found for disease %s", disease)
        return []

    # Add helper metrics
    df["status_score"] = df["status"].apply(_score_status)
    df["disease_match"] = df["diseases"].apply(lambda d: disease.lower() in str(d).lower())
    df["disease_count"] = df["diseases"].apply(lambda d: len(str(d).split(",")) if pd.notna(d) else 0)
    df["target_log2fc"] = df["target_gene"].apply(get_log2fc)

    # Prioritize
    df_sorted = df.sort_values(
        by=["status_score", "disease_match", "target_log2fc", "disease_count"],
        ascending=[False, False, False, False]
    )

    best_drugs = []
    used_genes = set()
    used_names = set()

    for _, row in df_sorted.iterrows():
        fields = row.to_dict()
        name = fields.get("name", "")
        gene = fields.get("target_gene", "")

        # Heuristic + Agent-based validation
        if not _is_valid_drug(name, disease):
            continue

        # Deduplicate by name + target gene
        if name in used_names or gene in used_genes:
            continue

        used_names.add(name)
        used_genes.add(gene)

        # Format log2FC with arrow
        log2fc_val = fields.get("target_log2fc")
        if log2fc_val is not None:
            arrow = "↑" if log2fc_val > 0 else "↓"
            fields["target_log2fc_display"] = f"{arrow} {log2fc_val:.2f}"
        else:
            fields["target_log2fc_display"] = "N/A"

        # Generate summary
        try:
            summary = llm_client.generate(
                prompt=_build_drug_summary_prompt(fields, disease),
                max_words=50
            )
        except Exception:
            summary = (
                f"{fields['name']} is a {fields['status']} drug targeting "
                f"{fields['target_gene']} ({fields['target_log2fc_display']}). "
                f"Associated diseases: {fields['diseases']}. "
                f"Sources: {fields['sources']}."
            )
        fields["summary"] = summary.strip()
        best_drugs.append(fields)

        if len(best_drugs) >= limit:
            break

    return best_drugs