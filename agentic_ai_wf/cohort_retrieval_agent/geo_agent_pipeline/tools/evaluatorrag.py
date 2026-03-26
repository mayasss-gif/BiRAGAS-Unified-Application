"""
Evaluator RAG TOOL for the Cohort Retrieval Agent system.

This tool handles evaluating files from various sources with retry logic,
progress tracking, and concurrent download capabilities.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from urllib.parse import quote
import asyncio
import gzip
import pandas as pd
from io import StringIO

# ==========================
# Utilities
# ==========================
PUBMED_RE = re.compile(r"\b(\d{7,8})\b")
URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)

def _to_lines(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_to_lines(v))
        return out
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            vs = _to_lines(v)
            if not vs:
                continue
            if len(vs) == 1:
                out.append(f"{k}: {vs[0]}")
            else:
                out.append(f"{k}:")
                out.extend([f"- {x}" for x in vs])
        return out
    return []

def extract_geo_support_lines(meta: Dict[str, Any], max_lines: int = 8) -> List[str]:
    fields = [
        "title","summary","overall_design","description","tissue_type",
        "characteristics","sample_characteristics","source_name","disease",
        "phenotype","cell_type","organism","keyword","labels","notes"
    ]
    lines = []
    md = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    for key in fields:
        if key in md:
            lines.extend(_to_lines(md[key]))
    seen, uniq = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line); uniq.append(line)
    return uniq[:max_lines]

def find_pubmed_ids_in_meta(meta: Dict[str, Any]) -> List[str]:
    return sorted(set(PUBMED_RE.findall(json.dumps(meta, ensure_ascii=False))))

def find_urls_in_meta(meta: Dict[str, Any]) -> List[str]:
    return sorted(set(URL_RE.findall(json.dumps(meta, ensure_ascii=False))))

def pmid_to_link(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

def make_geo_series_link(dataset_id: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset_id}" if isinstance(dataset_id, str) and dataset_id.upper().startswith("GSE") else ""

def build_pubmed_query_link(title_or_desc: str, disease: str) -> str:
    # Exact phrase of title/desc + disease keyword search on PubMed
    pattern = r'\s+'
    replacement = '%20'
    q = f"%22{title_or_desc.strip()}%22%20AND%20%28{re.sub(pattern, replacement, disease.strip())}%29"

    return f"https://pubmed.ncbi.nlm.nih.gov/?term={q}"

def build_scholar_query_link(title_or_desc: str, disease: str) -> str:
    
    q = f"\"{title_or_desc.strip()}\" {disease.strip()}"
    return f"https://scholar.google.com/scholar?q={quote(q)}"

def make_literature_support_links(title: str, description: str, disease: str) -> List[str]:
    """Return high-signal search links that verify the dataset’s title/description against the disease."""
    links = []
    if title:
        links.append(build_pubmed_query_link(title, disease))
        links.append(build_scholar_query_link(title, disease))
    if description and description.strip() and description.strip() != title.strip():
        desc_short = description.strip()
        if len(desc_short) > 180:
            desc_short = desc_short[:180]
        links.append(build_pubmed_query_link(desc_short, disease))
        links.append(build_scholar_query_link(desc_short, disease))
    # Dedup but preserve order
    seen, uniq = set(), []
    for u in links:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def strip_code_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)

# ---- Metrics schema helpers ----
PRIMARY_KEYS = [
    "disease_relevance", "tissue_match", "sample_coverage", "suitability_score"
]
COMPOSITE_KEYS = ["overall_geo_score", "biological_quality", "retrieval_quality"]
DETAILED_KEYS = [
    "biological_faithfulness_correct_disease_assignment",
    "biological_faithfulness_correct_tissue_assignment",
    "biological_faithfulness_metadata_consistency",
    "biological_faithfulness_hallucination_check",
    "biological_relevance_relevance",
    "biological_relevance_directness_to_study_question",
    "biological_relevance_sample_diversity",
    "biological_relevance_experimental_design_quality",
    "biological_relevance_completeness",
    "context_precision_precision",
    "context_precision_dataset_description_quality",
    "context_precision_sample_annotation_quality",
    "context_recall_recall",
    "context_recall_gene_coverage",
    "context_recall_sample_coverage",
    "context_recall_tissue_coverage",
    "context_recall_disease_coverage",
    "statistical_robustness_robustness",
    "statistical_robustness_metadata_completeness",
    "statistical_robustness_sample_size_adequacy",
    "statistical_robustness_data_consistency",
    "statistical_robustness_confidence_reporting",
]

def _clamp01(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0: v = 0.0
    if v > 1: v = 1.0
    return v

def _ensure_metrics_block(d: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        out[k] = _clamp01(d.get(k, 0.0)) if isinstance(d, dict) else 0.0
    return out

def enforce_schema(result: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    # Normalize top-level fields
    result["dataset_id"] = result.get("dataset_id") or dataset_id

    # Rename/merge legacy "scores" -> "primary_metrics" if needed
    primary_in = result.get("primary_metrics", result.get("scores", {}))
    composite_in = result.get("composite_scores", {})
    detailed_in = result.get("detailed_metrics", {})

    result["primary_metrics"]   = _ensure_metrics_block(primary_in, PRIMARY_KEYS)
    result["composite_scores"]  = _ensure_metrics_block(composite_in, COMPOSITE_KEYS)
    result["detailed_metrics"]  = _ensure_metrics_block(detailed_in, DETAILED_KEYS)

    # Ensure supporting_references structure
    sr = result.setdefault("supporting_references", {})
    sr.setdefault("geo_support_lines", [])
    sr.setdefault("clinical_db_links", [])
    sr.setdefault("literature_support_links", [])
    
    # Ensure justification exists
    if not isinstance(result.get("justification", ""), str) or not result["justification"]:
        result["justification"] = result.get("notes", "")[:500] if isinstance(result.get("notes", ""), str) else ""

    result.pop("scores", None)
    return result


# ==========================
# LLM Evaluator
# ==========================
class LLMEvaluator:
    def __init__(self, api_key: str = None):
        # NOTE: Prefer environment variable OPENAI_API_KEY in production.
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
    
  
    async def evaluate_dataset(self, dataset: Dict[str, Any],  series_meta : List[Dict[str,Any]], samples : Any,disease: str, query : str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate dataset with context of filtering rules.
        """
        dataset_id = getattr(dataset, "dataset_id", "")
        title = str(getattr(dataset, "title", "") or "")
        description = str(getattr(dataset, "description", "") or "")
        overall_design = getattr(dataset, "overall_design", "")
        tissue_types = getattr(dataset, "tissuecategorization", {}) or {}
        sample_count = getattr(dataset, "sample_count", 0)
        metadata = getattr(dataset, "metadata", {}) or {}
        series_meta = series_meta
        samples = samples
        query= query
        # ---------------------------
        # Build prompt rules
        # ---------------------------

        def build_tissue_rule(user_tissue: str) -> str:
            """Generate clear inclusion/exclusion rules for the given tissue filter."""
            user_tissue = (user_tissue or "").lower().strip()

            if user_tissue == "blood":
                return (
                    "Sample Type: Accept if sample mentions PBMC, peripheral blood, whole blood, "
                    "buffy coat, plasma, or peripheral T lymphocytes. Reject if the dataset not derived from blood."
                )

            elif user_tissue == "tissue":
                return (
                    "Sample Type: Accept if the dataset explicitly mentions organ or primary tissues "
                    "(e.g., liver, lung, heart, tumor, biopsy). "
                )

            else:  # "any"
                return (
                    """Sample Type: Accept both tissue or blood not any from this list ["cell line", "cell-line", "cell culture", "in vitro", "cultured cells",
                                # single-cell
                                "single cell", "single-cell", "scrna", "scrna-seq", "10x genomics", "smart-seq",
                                # common human cancer cell lines
                                "hela", "hek293"]  if experiment is RNA Seq else accept"
                    "any biological material (tissue, cells, or blood) as long as "
                    "the dataset meets the assay and species criteria. Reject only if it is synthetic, "
                    "non-human, or not biological material."""
                
                )


        def build_experiment_rule(experiment_filter: str) -> str:
            """Generate inclusion/exclusion rules for the experiment type."""
            ef = (experiment_filter or "").lower().strip()

            if ef == "rna-seq":
                return (
                    "Assay/Tech: Must clearly indicate bulk RNA-seq or mRNA. "
                    "Accept if it mentions RNA-seq, transcriptome, Illumina HiSeq/NextSeq/NovaSeq, "
                    "featureCounts, or read counts. Reject if it is single-cell RNA-seq or microarray or anyother type of RNA seq except mRNA"
                )
            elif ef == "microarray":
                return (
                    "Assay/Tech: Must clearly indicate gene expression microarray. "
                    "Accept if it mentions microarray, genechip, Affymetrix, Agilent, or Illumina HT array. "
                    "Reject if it is RNA-seq or single-cell sequencing."
                )
            elif ef == "single-cell":
                return (
                    "Assay/Tech: Must clearly indicate single-cell RNA-seq. "
                    "Accept if it mentions scRNA-seq, single-cell sequencing, 10x Genomics, Drop-seq, or Smart-seq. "
                    "Reject if it is bulk RNA-seq or microarray."
                )
            else:
                return (
                    f"Assay/Tech: Must match the provided experiment type '{experiment_filter}'. "
                    "Reject if the dataset uses unrelated methods."
                )
        tissue_filter= filters['tissue_filter']
        experiment_filter = filters['experiment_filter']
        tissue_rule = build_tissue_rule(tissue_filter)
        experiment_rule = build_experiment_rule(experiment_filter)
 
        # --- Add filtering context here ---
        filtering_context = f"""
        You are a biomedical dataset validation expert.
        This dataset has ALREADY PASSED a prior filtering stage with these rules:

        - Control-like group ≥ MIN_CTRL and non-control group ≥ MIN_NONCTRL.
        - Control-like terms: adjacent normal, normal tissue, healthy, untreated, vehicle, sham, wild type, baseline, benign, etc.

        That filtering context means:
        - This dataset should follow the given experiment rule {experiment_rule} and the query {query}.
        - Groups (`control` vs `non_control`) are already defined.
        - Tissue rules: expected tissue = {tissue_rule}.
        """

        prompt = f"""
        {filtering_context}

        Now, refine evaluation and provide STRICT JSON.
        User query : {query}
        Dataset ID: {dataset_id}
        Disease of interest: {disease}
        Expected Tissue: {tissue_filter}
        Expected Experiment : {experiment_filter}
        Dataset tissue types: {tissue_types}
        Sample Count: {sample_count}
        Title: {title}
        Description: {description[:600]}
        Samples Information : {samples}
        Series Meta File : {series_meta}
        GEO metadata lines (verbatim candidates — pick the most relevant 1–5 lines):
        {metadata}

        Output JSON format ONLY:
        {{
          "dataset_id": "<dataset_id>",
          "primary_metrics": {{
              "disease_relevance": 0.0,
              "tissue_match": 0.0,
              "sample_coverage": 0.0,
              "suitability_score": 0.0
          }},
          "composite_scores": {{
              "overall_geo_score": 0.0,
              "biological_quality": 0.0,
              "retrieval_quality": 0.0
          }},
          "detailed_metrics": {{
              "biological_faithfulness_correct_disease_assignment": 0.0,
              "biological_faithfulness_correct_tissue_assignment": 0.0,
              "biological_faithfulness_metadata_consistency": 0.0,
              "biological_faithfulness_hallucination_check": 0.0,
              "biological_relevance_relevance": 0.0,
              "biological_relevance_directness_to_study_question": 0.0,
              "biological_relevance_sample_diversity": 0.0,
              "biological_relevance_experimental_design_quality": 0.0,
              "biological_relevance_completeness": 0.0,
              "context_precision_precision": 0.0,
              "context_precision_dataset_description_quality": 0.0,
              "context_precision_sample_annotation_quality": 0.0,
              "context_recall_recall": 0.0,
              "context_recall_gene_coverage": 0.0,
              "context_recall_sample_coverage": 0.0,
              "context_recall_tissue_coverage": 0.0,
              "context_recall_disease_coverage": 0.0,
              "statistical_robustness_robustness": 0.0,
              "statistical_robustness_metadata_completeness": 0.0,
              "statistical_robustness_sample_size_adequacy": 0.0,
              "statistical_robustness_data_consistency": 0.0,
              "statistical_robustness_confidence_reporting": 0.0
          }},
          "supporting_references": {{
            "geo_support_lines": ["exact lines from GEO above (1-5 items, required)"],
            "clinical_db_links": ["URLs only; include PubMed links and GEO link if present"],
            "literature_support_links": ["PubMed/Scholar search URLs that verify title/description vs disease"]
          }},
          "justification": "brief justification (<= 60 words)"
        }}

        Rules:
        - All metric values MUST be floats in [0,1].
        - "geo_support_lines" MUST be verbatim text, not links.
        - "clinical_db_links" MUST include PubMed links derived from known PMIDs, GEO link (if present), and any other relevant clinical/DB URLs.
        - "literature_support_links" MUST be search URLs constructed from title/description + disease.
        - Output must be VALID JSON ONLY (no markdown).
        """

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": filtering_context},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            content = strip_code_fences(resp.choices[0].message.content)
            result = json.loads(content)

            # --- Enrich with links (deterministic post-processing) ---
            sr = result.setdefault("supporting_references", {})
            sr.setdefault("geo_support_lines", [])
            sr.setdefault("clinical_db_links", [])
            sr.setdefault("literature_support_links", [])

            # 1) PubMed links from any PMIDs in metadata
            # pmid_links = [pmid_to_link(p) for p in (meta_pubmed or [])]

            # 2) Add GEO Series link
            # geo_link = make_geo_series_link(dataset_id)
            # deterministic_links = set(meta_urls) | set(pmid_links)
            # if geo_link:
            #     deterministic_links.add(geo_link)

            # 3) Literature support links from title/description + disease
            lit_links = make_literature_support_links(title, description, disease)

            # Merge with model-returned URLs too
            # clinical_set = set(sr["clinical_db_links"]) | deterministic_links
            # sr["clinical_db_links"] = sorted(clinical_set)
            sr["literature_support_links"] = sorted(set(sr["literature_support_links"]) | set(lit_links))

            # # Fallback geo lines if model omitted
            # if not sr["geo_support_lines"]:
            #     sr["geo_support_lines"] = geo_lines[:3] if geo_lines else []

            # Enforce schema and clamp metrics
            result = enforce_schema(result, dataset_id)
            return result
            

        except Exception as e:
            # Deterministic fallback with links + schema
            geo_link = make_geo_series_link(dataset_id)
            # pmid_links = [pmid_to_link(p) for p in (meta_pubmed or [])]
            # clinical_links = sorted(set(find_urls_in_meta({"u": meta_urls})) | set(pmid_links) | ({geo_link} if geo_link else set()))
            lit_links = make_literature_support_links(title, description, disease)
            result = {
                "dataset_id": dataset_id,
                "primary_metrics": _ensure_metrics_block({}, PRIMARY_KEYS),
                "composite_scores": _ensure_metrics_block({}, COMPOSITE_KEYS),
                "detailed_metrics": _ensure_metrics_block({}, DETAILED_KEYS),
                "supporting_references": {
                    # "geo_support_lines": geo_lines[:3] if geo_lines else [],
                    # "clinical_db_links": clinical_links,
                    "literature_support_links": lit_links
                },
                "justification": f"LLM error: {str(e)[:150]}"
            }
            return enforce_schema(result, dataset_id)

    async def evaluate(self, datasets: List[Dict[str, Any]], series_meta : List[Dict[str,Any]], samples : Any,disease: str, query : str,  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        tasks = await self.evaluate_dataset(datasets,series_meta,samples,query,disease,filters)
        return tasks


# ==========================
# Loader
# ==========================


def load_series_matrix_file(matrix_path: str) -> Dict[str, Any]:
    """
    Load GEO Series Matrix file into metadata + dataframe.
    Supports both gzipped and plain text.
    """
    opener = gzip.open if matrix_path.endswith(".gz") else open

    with opener(matrix_path, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Separate metadata lines starting with '!'
    meta_lines = [l.strip() for l in lines if l.startswith("!")]
    data_lines = [l.strip() for l in lines if not l.startswith("!")]

    df = None
    if data_lines:
        df = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t")

    return {
        "meta_lines": meta_lines,
        "expression_data": df
    }


def load_datasets_from_folder(base_path: str) -> List[Dict[str, Any]]:
    """
    Expects:
      base_path/
        GSEXXXXXX/
          GSEXXXXXX_metadata.json
          GSEXXXXXX_series_matrix.txt[.gz]
    """
    datasets = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        meta_file = os.path.join(folder_path, f"{folder}_metadata.json")
        if not os.path.isfile(meta_file):
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Tissue extraction (safe)
        tissue_vals = []
        tt_raw = meta.get("metadata", {}).get("tissue_type", [])
        if isinstance(tt_raw, list):
            tissue_vals = [str(x).split("tissue:")[-1].strip() for x in tt_raw]
        elif isinstance(tt_raw, str):
            tissue_vals = [tt_raw]

        # Look for series matrix file
        series_matrix = None
        for ext in [".txt", ".txt.gz"]:
            candidate = os.path.join(folder_path, f"{folder}_series_matrix{ext}")
            if os.path.isfile(candidate):
                series_matrix = load_series_matrix_file(candidate)
                break

        datasets.append({
            "dataset_id": meta.get("dataset_id", folder),
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "tissue_types": sorted({t for t in tissue_vals if t}),
            "sample_count": meta.get("sample_count", 0),
            "geo_lines": extract_geo_support_lines(meta, max_lines=8),
            "pubmed_candidates": find_pubmed_ids_in_meta(meta),
            "url_candidates": find_urls_in_meta(meta),
            "series_matrix": series_matrix,
            "raw_meta": meta
        })
    return datasets