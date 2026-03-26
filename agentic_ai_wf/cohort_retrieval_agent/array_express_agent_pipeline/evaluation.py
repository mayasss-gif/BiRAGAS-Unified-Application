"""
ArrayExpress (AE) Evaluator — Cohort Retrieval Agent

AE-only implementation:
- AE/BioStudies aware support-line extraction
- PMID/URL harvesting
- AE + ENA link construction
- LLM-based scoring with strict JSON
- Async batch evaluate() for drop-in use
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import quote
from openai import OpenAI

from   .utils import sanitize_folder_name, summarize_analysis, _norm_experiment_filter
from   .constants import Defaults


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
        out: List[str] = []
        for v in value:
            out.extend(_to_lines(v))
        return out
    if isinstance(value, dict):
        out: List[str] = []
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

def find_pubmed_ids_in_meta(meta: Dict[str, Any]) -> List[str]:
    return sorted(set(PUBMED_RE.findall(json.dumps(meta, ensure_ascii=False))))

def find_urls_in_meta(meta: Dict[str, Any]) -> List[str]:
    return sorted(set(URL_RE.findall(json.dumps(meta, ensure_ascii=False))))

def pmid_to_link(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

def make_arrayexpress_link(accession: str) -> str:
    if not accession:
        return ""
    return f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession.strip()}"

def make_ena_link(run_or_project: str) -> str:
    if not run_or_project:
        return ""
    return f"https://www.ebi.ac.uk/ena/browser/view/{run_or_project.strip()}"

def build_pubmed_query_link(title_or_desc: str, disease: str) -> str:
    q = f"%22{title_or_desc.strip()}%22%20AND%20%28{quote(disease.strip())}%29"
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={q}"

def build_scholar_query_link(title_or_desc: str, disease: str) -> str:
    q = f"\"{title_or_desc.strip()}\" {disease.strip()}"
    return f"https://scholar.google.com/scholar?q={quote(q)}"

def make_literature_support_links(title: str, description: str, disease: str) -> List[str]:
    links: List[str] = []
    if title:
        links.append(build_pubmed_query_link(title, disease))
        links.append(build_scholar_query_link(title, disease))
    if description and description.strip() and (not title or description.strip() != title.strip()):
        desc_short = description.strip()[:180]
        links.append(build_pubmed_query_link(desc_short, disease))
        links.append(build_scholar_query_link(desc_short, disease))
    seen, uniq = set(), []
    for u in links:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def strip_code_fences(s: str) -> str:
    import re
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)

# ---- Metrics schema helpers (AE-only) ----
PRIMARY_KEYS = [
    "disease_relevance", "tissue_match", "sample_coverage", "suitability_score"
]
COMPOSITE_KEYS = ["overall_ae_score", "biological_quality", "retrieval_quality"]
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
    return {k: _clamp01((d or {}).get(k, 0.0)) for k in keys}

def enforce_schema(result: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    result["dataset_id"] = result.get("dataset_id") or dataset_id
    primary_in   = result.get("primary_metrics", result.get("scores", {}))
    composite_in = result.get("composite_scores", {})
    detailed_in  = result.get("detailed_metrics", {})

    result["primary_metrics"]   = _ensure_metrics_block(primary_in, PRIMARY_KEYS)
    result["composite_scores"]  = _ensure_metrics_block(composite_in, COMPOSITE_KEYS)
    result["detailed_metrics"]  = _ensure_metrics_block(detailed_in, DETAILED_KEYS)

    sr = result.setdefault("supporting_references", {})
    sr.setdefault("ae_support_lines", [])
    sr.setdefault("clinical_db_links", [])
    sr.setdefault("literature_support_links", [])

    if not isinstance(result.get("justification", ""), str) or not result["justification"]:
        result["justification"] = (result.get("notes", "") or "")[:500]

    result.pop("scores", None)
    return result

def extract_ae_support_lines(meta: Dict[str, Any], max_lines: int = 8) -> List[str]:
    """
    Pull high-signal lines from AE/BioStudies JSON.
    Tries both your mirrored 'metadata' block and common AE keys.
    """
    candidates: List[str] = []
    md = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    root = meta if isinstance(meta, dict) else {}

    fields = [
        # mirrored-style keys you might store
        "title","description","overall_design","summary","tissue_type","disease",
        "phenotype","cell_type","organism","keyword","labels","notes",
        # AE/BioStudies-ish keys
        "design","factors","protocols","sample_characteristics","characteristics",
        "efo_terms","experimental_design","assay_type"
    ]
    for key in fields:
        if key in md:
            candidates.extend(_to_lines(md[key]))
        elif key in root:
            candidates.extend(_to_lines(root[key]))

    seen, uniq = set(), []
    for line in candidates:
        if line not in seen:
            seen.add(line); uniq.append(line)
    return uniq[:max_lines]

class LLMEvaluatorAE:
    """
    AE-only LLM evaluator. Accepts experiment_filter short codes:
      - 'rna'  -> bulk RNA-seq
      - 'sc'   -> scRNA-seq / snRNA-seq
      - 'st'   -> spatial transcriptomics
      - 'microarray' -> expression arrays
    """
    def __init__(self, api_key: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY", api_key)
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _experiment_rule(experiment_filter: str) -> str:
        ef = (experiment_filter or "").strip().lower()
        if ef in ("rna", "rna-seq", "rnaseq"):
            return ("Assay/Tech: Must indicate bulk RNA-seq/transcriptome sequencing. "
                    "Accept RNA-seq, transcriptome, Illumina HiSeq/NextSeq/NovaSeq, read counts. "
                    "Reject single-cell, cell lines, cells and microarray.")
        if ef in ("sc", "scrna", "scrna-seq", "single-cell"):
            return ("Assay/Tech: Must indicate single-cell RNA-seq (10x/Drop-seq/Smart-seq/single-nucleus). "
                    "Reject bulk RNA-seq and microarray.")
        if ef in ("st", "spatial", "spatial-tx", "spatial transcriptomics"):
            return ("Assay/Tech: Must indicate spatial transcriptomics (Visium/Slide-seq/Stereo-seq). "
                    "Reject non-spatial assays.")
        if ef == "microarray":
            return ("Assay/Tech: Must indicate expression microarray (Affymetrix/Agilent/Illumina array). "
                    "Reject RNA-seq and scRNA-seq.")
        return (f"Assay/Tech: Must match provided experiment type '{experiment_filter}'. "
                "Reject unrelated methods.")

    @staticmethod
    def _tissue_rule(user_tissue: str) -> str:
        t = (user_tissue or "").lower().strip()
        if t == "blood":
            return ("Sample Type: Accept PBMC, peripheral blood, whole blood, buffy coat, plasma, "
                    "peripheral T lymphocytes. Reject non-blood sources.")
        if t == "tissue":
            return ("Sample Type: Accept explicit organ/primary tissues (liver, lung, heart, tumor, biopsy).")
        return ("Sample Type: Accept any human biological material (tissue, cells, or blood) if assay/species match; "
                "reject synthetic/non-human.")

    async def evaluate_dataset(self, dataset: Dict[str, Any], disease: str,
                               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        dataset_id   = dataset.get("dataset_id", "")
        title        = str(dataset.get("title", "") or "")
        description  = str(dataset.get("description", "") or "")
        tissue_types = ", ".join(dataset.get("tissue_types", []))
        sample_count = dataset.get("sample_count", 0)

        ae_lines     = dataset.get("ae_lines", [])
        meta_pubmed  = dataset.get("pubmed_candidates", [])
        meta_urls    = dataset.get("url_candidates", [])

        ae_block = "\n".join(f"- {line}" for line in ae_lines) if ae_lines else "(none)"
        tissue_filter = (filters.get("tissue_filter") or "any").strip().lower() if filters else "any"
        experiment_filter = (filters.get("experiment_filter") or "rna").strip().lower() if filters else "rna"

        tissue_rule = self._tissue_rule(tissue_filter)
        experiment_rule = self._experiment_rule(experiment_filter)

        filtering_context = f"""
        You are a biomedical dataset validation expert for ArrayExpress/BioStudies.

        This dataset has ALREADY PASSED prior filtering with rules such as:
        - Control-like group ≥ MIN_CTRL and non-control ≥ MIN_NONCTRL.
        - Control-like terms include: adjacent normal, healthy, untreated, vehicle, sham, wild type, baseline, benign.
        - Assay rule: {experiment_rule}
        - Tissue rule: {tissue_rule}
        """

        prompt = f"""
        {filtering_context}

        Return STRICT JSON only.

        Dataset ID: {dataset_id}
        Disease of interest: {disease}
        Expected Tissue: {tissue_filter}
        Dataset tissue types: {tissue_types}
        Sample Count: {sample_count}
        Title: {title}
        Description: {description[:600]}

        ArrayExpress/BioStudies metadata lines (verbatim; pick the most relevant 1–5):
        {ae_block}

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
              "overall_ae_score": 0.0,
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
            "ae_support_lines": ["exact lines from AE above (1-5 items, required)"],
            "clinical_db_links": ["URLs only; include PubMed links and AE/ENA links if present"],
            "literature_support_links": ["PubMed/Scholar search URLs that verify title/description vs disease"]
          }},
          "justification": "brief justification (<= 60 words)"
        }}

        Rules:
        - All metric values MUST be floats in [0,1].
        - 'ae_support_lines' MUST be verbatim text, not links.
        - 'clinical_db_links' MUST include PubMed links from known PMIDs and the AE study link if available.
        - 'literature_support_links' MUST be search URLs constructed from title/description + disease.
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

            # Deterministic enrichment
            sr = result.setdefault("supporting_references", {})
            sr.setdefault("ae_support_lines", [])
            sr.setdefault("clinical_db_links", [])
            sr.setdefault("literature_support_links", [])

            pmid_links = [pmid_to_link(p) for p in (meta_pubmed or [])]
            ae_link = make_arrayexpress_link(dataset_id) if dataset_id else ""
            deterministic = set(meta_urls) | set(pmid_links)
            if ae_link:
                deterministic.add(ae_link)

            for ena_id in dataset.get("ena_accessions", []):
                elink = make_ena_link(ena_id)
                if elink:
                    deterministic.add(elink)

            lit_links = make_literature_support_links(title, description, disease)

            sr["clinical_db_links"] = sorted(set(sr["clinical_db_links"]) | deterministic)
            sr["literature_support_links"] = sorted(set(sr["literature_support_links"]) | set(lit_links))

            if not sr["ae_support_lines"]:
                sr["ae_support_lines"] = ae_lines[:3] if ae_lines else []

            result = enforce_schema(result, dataset_id)
            return result

        except Exception as e:
            ae_link = make_arrayexpress_link(dataset_id) if dataset_id else ""
            pmid_links = [pmid_to_link(p) for p in (meta_pubmed or [])]
            clinical_links = sorted(set(find_urls_in_meta({"u": meta_urls})) | set(pmid_links) | ({ae_link} if ae_link else set()))
            lit_links = make_literature_support_links(title, description, disease)
            result = {
                "dataset_id": dataset_id,
                "primary_metrics": _ensure_metrics_block({}, PRIMARY_KEYS),
                "composite_scores": _ensure_metrics_block({}, COMPOSITE_KEYS),
                "detailed_metrics": _ensure_metrics_block({}, DETAILED_KEYS),
                "supporting_references": {
                    "ae_support_lines": ae_lines[:3] if ae_lines else [],
                    "clinical_db_links": clinical_links,
                    "literature_support_links": lit_links
                },
                "justification": f"LLM error: {str(e)[:150]}"
            }
            return enforce_schema(result, dataset_id)

    async def evaluate(self, datasets: List[Dict[str, Any]], disease: str,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        import asyncio
        return await asyncio.gather(*[
            self.evaluate_dataset(ds, disease, filters) for ds in datasets
        ])

# ==========================
# AE Dataset Loader
# ==========================
# def load_ae_datasets_from_files(metadata_path: str,
#                                 filtered_path: Optional[str] = None) -> List[Dict[str, Any]]:
#     """
#     Load AE datasets from your agent outputs.

#     Expects:
#       - metadata_path: JSON (dict or list of dicts) with AE experiment records
#       - filtered_path: optional JSON with filtered experiment IDs
#     """
#     def _load_json_safe(path: Optional[str]) -> Any:
#         if not path:
#             return None
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         except (FileNotFoundError, json.JSONDecodeError):
#             return None

#     meta_obj = _load_json_safe(metadata_path)
#     filt_obj = _load_json_safe(filtered_path)

#     if meta_obj is None:
#         return []

#     meta_list: List[Dict[str, Any]]
#     if isinstance(meta_obj, dict):
#         meta_list = [meta_obj]
#     elif isinstance(meta_obj, list):
#         meta_list = meta_obj
#     else:
#         meta_list = []

#     filtered_ids = set()
#     if isinstance(filt_obj, dict):
#         filtered_ids.update(filt_obj.get("filtered_experiment_ids", []))
#     elif isinstance(filt_obj, list):
#         filtered_ids.update([x for x in filt_obj if isinstance(x, str)])

#     datasets: List[Dict[str, Any]] = []
#     for rec in meta_list:
#         acc = (rec.get("accession") or rec.get("experiment_accession")
#                or rec.get("dataset_id") or rec.get("id") or "")
#         title = rec.get("title", "") or rec.get("name", "")
#         description = rec.get("description", "") or rec.get("summary", "")

#         # Tissue extraction
#         tissue_vals: List[str] = []
#         md = rec.get("metadata", {})
#         tt_raw = md.get("tissue_type", []) if isinstance(md, dict) else []
#         if isinstance(tt_raw, list):
#             tissue_vals.extend([str(x).split("tissue:")[-1].strip() for x in tt_raw])
#         elif isinstance(tt_raw, str):
#             tissue_vals.append(tt_raw)

#         for key in ("characteristics", "sample_characteristics", "factors", "efo_terms", "organism", "cell_type"):
#             if key in rec:
#                 tissue_vals.extend(_to_lines(rec[key]))

#         # Sample count
#         sample_count = (rec.get("sample_count") or rec.get("n_samples") or 0)
#         if not sample_count and acc and acc in filtered_ids:
#             sample_count = rec.get("sample_count", 0)

#         pubmed_candidates = find_pubmed_ids_in_meta(rec)
#         url_candidates = find_urls_in_meta(rec)

#         ena_accessions: List[str] = []
#         for k in ("ena_accessions", "ena", "ena_runs", "ena_projects"):
#             if k in rec:
#                 ena_accessions.extend([s for s in _to_lines(rec[k]) if s])

#         datasets.append({
#             "dataset_id": acc,
#             "title": title,
#             "description": description,
#             "tissue_types": sorted({t for t in tissue_vals if t}),
#             "sample_count": sample_count,
#             "ae_lines": extract_ae_support_lines(rec, max_lines=8),
#             "pubmed_candidates": pubmed_candidates,
#             "url_candidates": url_candidates,
#             "ena_accessions": ena_accessions,
#             "raw_meta": rec
#         })
#     return datasets

def load_ae_datasets_from_files(
    filtered_path: str
) -> List[Dict[str, Any]]:
    """
    Load AE datasets ONLY from filtered_experiments JSON.

    Expects:
      - filtered_path: JSON containing either:
          • {"filtered_experiment_ids": [...], "records": [...]}
          • OR a list of experiment records
          • OR a dict keyed by accession → record
    """

    def _load_json_safe(path: Optional[str]) -> Any:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    filt_obj = _load_json_safe(filtered_path)
    if not filt_obj:
        return []

    # --------------------------------------------------
    # Normalize filtered experiment records
    # --------------------------------------------------
    records: List[Dict[str, Any]] = []

    # Case 1: {"filtered_experiment_ids": [...], "records": [...]}
    if isinstance(filt_obj, dict) and "records" in filt_obj:
        records = filt_obj["records"]

    # Case 2: list of experiment dicts
    elif isinstance(filt_obj, list):
        records = [r for r in filt_obj if isinstance(r, dict)]

    # Case 3: dict keyed by accession
    elif isinstance(filt_obj, dict):
        records = [
            v for v in filt_obj.values()
            if isinstance(v, dict)
        ]

    if not records:
        return []

    # --------------------------------------------------
    # Build datasets
    # --------------------------------------------------
    datasets: List[Dict[str, Any]] = []

    for rec in records:
        acc = (
            rec.get("accession")
            or rec.get("experiment_accession")
            or rec.get("dataset_id")
            or rec.get("id")
            or ""
        )

        title = rec.get("title") or rec.get("name", "")
        description = rec.get("description") or rec.get("summary", "")

        # -------- Tissue extraction --------
        tissue_vals: List[str] = []
        md = rec.get("metadata", {})

        if isinstance(md, dict):
            tt_raw = md.get("tissue_type", [])
            if isinstance(tt_raw, list):
                tissue_vals.extend(
                    str(x).split("tissue:")[-1].strip() for x in tt_raw
                )
            elif isinstance(tt_raw, str):
                tissue_vals.append(tt_raw)

        for key in (
            "characteristics",
            "sample_characteristics",
            "factors",
            "efo_terms",
            "organism",
            "cell_type",
        ):
            if key in rec:
                tissue_vals.extend(_to_lines(rec[key]))

        # -------- Sample count --------
        sample_count = (
            rec.get("sample_count")
            or rec.get("n_samples")
            or 0
        )

        # -------- Supporting evidence --------
        pubmed_candidates = find_pubmed_ids_in_meta(rec)
        url_candidates = find_urls_in_meta(rec)

        ena_accessions: List[str] = []
        for k in ("ena_accessions", "ena", "ena_runs", "ena_projects"):
            if k in rec:
                ena_accessions.extend(
                    s for s in _to_lines(rec[k]) if s
                )

        datasets.append({
            "dataset_id": acc,
            "title": title,
            "description": description,
            "tissue_types": sorted({t for t in tissue_vals if t}),
            "sample_count": sample_count,
            "ae_lines": extract_ae_support_lines(rec, max_lines=8),
            "pubmed_candidates": pubmed_candidates,
            "url_candidates": url_candidates,
            "ena_accessions": ena_accessions,
            "raw_meta": rec,
        })

    return datasets


# def build_ae_datasets_from_summary(analysis_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     Accepts your AnalysisSummary (dict-like) and loads datasets.
#     Expected fields:
#       - metadata_path
#       - filtered_path (optional)
#     """

#     # Helper to read from dict OR attribute
#     def _get(field: str) -> Optional[Any]:
#         if isinstance(analysis_summary, dict):
#             return analysis_summary.get(field)
#         return getattr(analysis_summary, field, None)

#     meta_path = _get("metadata_path") or _get("meta_path")
#     filt_path = _get("filtered_path") or _get("filt_path")

#     if not meta_path:
#         raise KeyError("metadata_path is required but was not found on analysis_summary.")
#     from pathlib import Path

#     # Normalize to strings (load_ae_datasets_from_files likely expects str)
#     meta_path = str(Path(meta_path))
#     filt_path = str(Path(filt_path)) if filt_path else None
#     return load_ae_datasets_from_files(meta_path, filt_path)


#     # Save matching records
#     safe_disease = sanitize_folder_name(disease)
#     tissue_filter =  sanitize_folder_name(tissue_filter)
#     experiment_filter = _norm_experiment_filter(experiment_filter)

#     # Base directory for saving data
#     filepath = Path(Defaults.META_FILEPATH) / f"{safe_disease}_{tissue_filter}_{experiment_filter}"
#     filepath.mkdir(parents=True, exist_ok=True)

def build_ae_datasets_from_summary(analysis_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts your AnalysisSummary (dict-like) and loads datasets ONLY from filtered_path.

    Expected fields (one of these must exist):
      - filtered_path  (preferred)
      - filt_path      (alias)
      - filtered_meta_path (optional alias if you used it elsewhere)
    """

    # Helper to read from dict OR attribute
    def _get(field: str) -> Optional[Any]:
        if isinstance(analysis_summary, dict):
            return analysis_summary.get(field)
        return getattr(analysis_summary, field, None)

    filt_path = (
        _get("filtered_path")
        or _get("filt_path")
        or _get("filtered_meta_path")
    )

    if not filt_path:
        raise KeyError(
            "filtered_path is required but was not found on analysis_summary. "
            "This loader is configured to read ONLY the filtered experiments JSON."
        )

    filt_path = str(Path(filt_path))

    # New filtered-only loader signature
    return load_ae_datasets_from_files(filtered_path=filt_path)

def save_ae_evaluation_results(
    results: List[Dict[str, Any]],
    disease: str,
    tissue_filter: str,
    experiment_filter: str,
    output_dir : str
) -> str:
    """
    Save AE evaluation results into a nested directory based on filters.

    Directory structure:
        outputs/ae_eval/<disease>_<tissue>_<experiment>/
            <disease>_ae_evaluation_<timestamp>.json

    Args:
        results: List of evaluation result dictionaries.
        disease: Disease name (used for naming folder and file).
        tissue_filter: Tissue filter (used for folder name).
        experiment_filter: Experiment type (rna, sc, st, etc.)
        base_dir: Root directory for storing results.

    Returns:
        str: Path to the saved JSON file.
    """
    # Sanitize inputs
    safe_disease = sanitize_folder_name(disease)
    safe_tissue = sanitize_folder_name(tissue_filter)
    safe_experiment = _norm_experiment_filter(experiment_filter)

    # Build subfolder path
    output_path = Path(Defaults.META_FILEPATH)  / Path(output_dir) / safe_disease
    output_path.mkdir(parents=True, exist_ok=True)

    # Build timestamped filename
    filename = f"evaluation_results.json"
    file_path = output_path / filename

    # Write JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[AE Evaluator] Results saved at: {file_path}")
    return str(file_path)

class EvaluationToolAE:
    """Thin wrapper mirroring your EvaluationTool but AE-only."""
    def __init__(self, config, evaluator_ae: LLMEvaluatorAE):
        self.config = config
        self.evaluator = evaluator_ae

    async def evaluate_datasets(self, datasets: List[Dict[str, Any]], disease: str,
                                filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return await self.evaluator.evaluate(datasets, disease, filters)
