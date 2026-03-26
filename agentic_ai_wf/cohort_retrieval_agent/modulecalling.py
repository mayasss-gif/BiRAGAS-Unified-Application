from __future__ import annotations
#!/usr/bin/env python3
"""
Demo script for the Cohort Retrieval Agent system.

This script demonstrates how to use the new agent system for retrieving
biomedical datasets from multiple sources.
"""
import asyncio
import logging
from pathlib import Path
import asyncio
import unicodedata
import os, re, json, unicodedata, datetime, csv
from typing import Any, Iterable, List, Optional, Tuple, Dict
import hashlib

# Project Imports
from .geo_agent_pipeline.agent import CohortRetrievalAgent
from .geo_agent_pipeline.config import CohortRetrievalConfig, DirectoryPathsConfig
from .geo_agent_pipeline.tools.ontology_tool import DiseaseOntologyTool, DiseaseOntologyCriteria
from .array_express_agent_pipeline.utils import _norm_experiment_filter, safe_name

async def cohort_call(disease, tissue_filter, experiment_filter, input_query, no_of_dataset, other_filter, output_dir: str | None = None):
    print("=" * 60)
    print("Cohort Call for GEO")
    print("=" * 60)

    # Create agent with default configuration
    agent = CohortRetrievalAgent()

    # Combine filters into dictionary
    filters = {
        "tissue_filter": tissue_filter,
        "experiment_filter": experiment_filter,
        "other_filter": other_filter
    }

    # if output_dir is not None:
    from pathlib import Path

    if output_dir is not None:
        print("Inside output dir given", )
        out_dir = Path(output_dir) / safe_name(disease) 
        print(output_dir)
    else:
        # Optional: create clean output directory name
        safe_disease = disease.replace(" ", "_").lower()
        experiment_filter = _norm_experiment_filter(experiment_filter)
        output_dir = f"{safe_disease}_{tissue_filter or 'na'}_{experiment_filter or 'na'}"

        file_saving = {
            "disease" : safe_disease,
            "tissue_filter": tissue_filter,
            "experiment_filter": experiment_filter,
            "other_filter": other_filter
        }

        def build_cache_key(filters: dict) -> str:
            # Ensure deterministic ordering
            normalized = json.dumps(filters, sort_keys=True)
            cache_hash = hashlib.sha256(normalized.encode()).hexdigest()

            # Shorten for readability (first 12 chars are enough)
            return f"cache_{cache_hash[:12]}"
        
        output_dir = build_cache_key(file_saving)
        out_dir = Path(output_dir) / safe_name(disease) 

    print(output_dir)
    # print(input_query)
    # Call the async cohort retrieval
    result = await agent.retrieve_cohort(
        disease_name=disease,
        max_datasets_per_source=no_of_dataset,
        filters=filters,     
        query = input_query,             
        output_dir=out_dir
    )

    # ➜ Build the CSV right after retrieval
    csv_path = build_dataset_summary_csv_for_disease(
        disease=disease,
        tissue_filter=tissue_filter,
        experiment_filter=experiment_filter,
        query = input_query,
        output_dir=out_dir,                  # scan this folder recursively
        out_csv=None                            # or give a path string here
    )

    print(f"Success: {result.success}")
    print(f"Datasets found: {result.total_datasets_found}")
    print(f"Datasets downloaded: {result.total_datasets_downloaded}")
    print(f"Files downloaded: {result.total_files_downloaded}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    print(f"Details : {result}")
    print(f"Output Directory : {result.output_directory}")
    print(f"CSV written to: {csv_path}\n")

    return result, disease, filters

# --------- helpers ---------
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().strip().split())

def _build_expansion(ont: Dict) -> Dict[str, List[str]]:
    def _uniqs(xs):
        seen, out = set(), []
        for x in xs:
            n = _norm(x)
            if n and n not in seen:
                seen.add(n); out.append(n)
        return out

    return {
        "synonyms": _uniqs(ont.get("synonyms", [])),
        "children": _uniqs(ont.get("children", [])),
        "siblings": _uniqs(ont.get("siblings", [])),
        "parents": _uniqs(ont.get("parents", [])),
        "primary_tissues":_uniqs(ont.get("primary_tissues", [])),
        "associated_phenotypes": _uniqs(ont.get("associated_phenotypes", [])),
        "cross_disease_drivers":_uniqs(ont.get("cross_disease_drivers", []))
    }

async def _requery_once(term: str, filters: Optional[Dict], input_query, output_dir: str | None = None):
    """
    
    Examples:
      agent = agents["geo"]  # wherever you keep agent instances
      agent_res = await agent.retrieve_cohort(disease_name=term, filters=filters)
      return {"valid_datasets": agent_res.datasets_found, "raw": agent_res}
    """
    agent = CohortRetrievalAgent()
  

    # Optional: create clean output directory name
    safe_disease = term.replace(" ", "_").lower()
    tissue_filter = filters["tissue_filter"]
    experiment_filter = _norm_experiment_filter(filters["experiment_filter"])
    
    file_saving = {
        "disease" : safe_disease,
        "tissue_filter": tissue_filter,
        "experiment_filter": experiment_filter,
        "other_filter": filters["other filter"]
    }

    def build_cache_key(filters: dict) -> str:
        # Ensure deterministic ordering
        normalized = json.dumps(filters, sort_keys=True)
        cache_hash = hashlib.sha256(normalized.encode()).hexdigest()

        # Shorten for readability (first 12 chars are enough)
        return f"cache_{cache_hash[:12]}"
    if not output_dir:
        output_dir = build_cache_key(file_saving)   
    # print(output_dir)
    # print("Before calling retrrieve cohor")
    # print(filters)
    query = f"{term} {filters}"
    
    safe_term = safe_name(term)
    output_dir = Path(output_dir) / safe_term
    result = await agent.retrieve_cohort(
        disease_name=term,
        max_datasets_per_source=filters["no_of_dataset"],
        filters = filters,
        query = query,
        output_dir=output_dir
    )
    
    # ➜ Build the CSV right after retrieval
    csv_path = build_dataset_summary_csv_for_disease(
        disease=term,
        tissue_filter=tissue_filter,
        experiment_filter=experiment_filter,
        query = input_query,
        output_dir=output_dir,                  # scan this folder recursively
        out_csv=None                            # or give a path string here
    )

    print(f"Success: {result.success}")
    print(f"Datasets found: {result.total_datasets_found}")
    print(f"Datasets downloaded: {result.total_datasets_downloaded}")
    print(f"Files downloaded: {result.total_files_downloaded}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    print(f"CSV written to: {csv_path}\n")

    return result

async def geo_ontology_fallback(disease_name: str, filters: Optional[Dict], query : Optional[str], output_dir: str | None = None) -> Optional[Dict]:
    """
    Call ontology tool, then try re-queries in order: synonyms -> children -> siblings.
    Returns the first non-empty retrieval result dict or None.
    """
    print("Ontology Triggered")
    def _slug(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        return re.sub(r"[\s-]+", "_", s) or "unnamed"

    def _disease_dir(base_output: Optional[str], disease_name: str) -> str:
        """
        Create and return: <base>/OnotologyResult/<disease_slug>
        If base_output is None, uses CWD.
        """

        cf = DirectoryPathsConfig()
        base = getattr(cf, "ontology_dir", None)
        ddir = os.path.join(base, "OnotologyResult", _slug(disease_name))
        os.makedirs(ddir, exist_ok=True)
        return ddir

    def _write_attempt(ddir: str, tier: str, term: str, filters: Optional[Dict], result: Optional[Dict]):
        """Write one JSON file per attempt."""
        attempts_dir = os.path.join(ddir, "fallback_attempts")
        os.makedirs(attempts_dir, exist_ok=True)
        fname = f"{tier}__{_slug(term)}.json"
        path = os.path.join(attempts_dir, fname)
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tier": tier,                      # "synonyms" | "children" | "siblings"
            "term": term,
            "filters": filters,
            "result": result,                  # keep full payload; trim here if huge
        }

    cfg = CohortRetrievalConfig()
    onto_tool = DiseaseOntologyTool(cfg)
    crit = DiseaseOntologyCriteria(
        diseases=[disease_name],
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=700,
        use_json_schema=True,
        enable_fewshot=True,
        save_path=None        # read from env
    )
    from dotenv import load_dotenv

    load_dotenv(override=True)
    tool_res = await onto_tool.execute(crit)
    print(tool_res)
    if not tool_res.success or not tool_res.data:
        print("[Ontology] Failed or empty.")
        return None
    
    # --- inside geo_ontology_fallback, after you compute `ont` and `tiers`, and after `disease_out_dir` is set ---
    def _write_json(path: Path, obj: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


    ont = tool_res.data[0].result
    print(ont)
    tiers = _build_expansion(ont)
    # caps (tune as needed)
    tiers["synonyms"] = tiers["synonyms"]
    tiers["siblings"] = tiers["siblings"]
    tiers["primary_tissues"] = tiers["primary_tissues"]
    tiers["parents"] = tiers["parents"]
    tiers["cross_disease_drivers"] = tiers["cross_disease_drivers"]
    tiers["children"] = tiers["children"]
    tiers['associated_phenotypes'] = tiers["associated_phenotypes"]

    disease_out_dir = _disease_dir(getattr(cfg, "output_dir", None), disease_name)

    SAFE = re.compile(r"[^a-z0-9._-]+")
    def _safe_name(s: str) -> str:
        return SAFE.sub("_", (s or "").strip().lower())

    disease_slug = _safe_name(disease_name)
    disease_dir = Path(disease_out_dir) 
    disease_dir.mkdir(parents=True, exist_ok=True)

    # 1) full ontology file at disease root
    _write_json(disease_dir / f"{disease_slug}.json", ont)

    for tier_name in ("synonyms", "siblings","primary_tissues","parents","cross_disease_drivers","children","associated_phenotypes"):
        terms = tiers[tier_name]
        if not terms:
            continue
    
    all_results = []

    for term in terms:
        try:
            # print(term, filters)
            r = await _requery_once(term, filters, query, output_dir)
            # persist every attempt (even successful)
            _write_attempt(disease_out_dir, tier_name, term, filters, r)
            if r.total_datasets_found >0:
                print(f"[Ontology Fallback] Recovered via {tier_name[:-1]} term: '{term}'")
                all_results.append(r)

        except NotImplementedError:
            raise
        except Exception as e:
            # Persist that it errored (result=None)
            _write_attempt(disease_out_dir, tier_name, term, filters, None)
            print(f"[Ontology Fallback] Error querying '{term}': {e}")

    if all_results:
        print(f"[Ontology Fallback] Recovered {len(all_results)} non-empty results.")
        # Option A: return them as a list
        # print(all_results)
        return all_results


    print("[Ontology Fallback] No datasets recovered from synonyms/children/siblings.")
    return "[Ontology Fallback] No datasets recovered from synonyms/children/siblings."

def build_dataset_summary_csv_for_disease(
    disease: str,
    tissue_filter: Optional[str],
    experiment_filter: Optional[str],
    query : Optional[str],
    output_dir: str,
    out_csv: Optional[str] = None,
) -> Path:
    """
    Scan BASE_DIR/output_dir for *metadata.json and export a CSV with columns:

    Input Query | Disease Name | Tissue Requested | Experiment Requested |
    Accession ID | No. of Samples | Tissue Type | Other Characteristics |
    Library Strategy | Library Source | Series Matrix available at GEO |
    Supplementary Files available at GEO | Overall_design
    """

    # -------- inline helpers --------
    def i_contains(hay: Optional[str], needle: Optional[str]) -> bool:
        if not needle: return True
        if hay is None: return False
        return needle.lower() in hay.lower()

    def uniq_join(values: Iterable[str]) -> str:
        seen: List[str] = []
        for v in values:
            v = (v or "").strip()
            if v and v not in seen:
                seen.append(v)
        return "; ".join(seen)

    def as_list(x: Any) -> List[Any]:
        if x is None: return []
        return x if isinstance(x, list) else [x]

    def flatten_deep(seq: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        for v in seq:
            if isinstance(v, list):
                out.extend(flatten_deep(v))
            else:
                out.append(v)
        return out

    def dig_first(d: dict, keys: List[str], default=None):
        for k in keys:
            if k in d and d[k] not in (None, "", []):
                return d[k]
        return default

    def dig_nested(meta: dict, keys: List[str], inner_key: str = "metadata") -> List[str]:
        """Combine values from both top-level and meta['metadata'] (if present)."""
        out: List[str] = []
        inner = meta.get(inner_key, {}) if isinstance(meta.get(inner_key), dict) else {}

        for k in keys:
            out.extend(as_list(meta.get(k)))
            if inner:
                out.extend(as_list(inner.get(k)))

        flat = []
        for v in flatten_deep(out):
            flat.append(v if isinstance(v, str) else (str(v) if v is not None else ""))

        return [x.strip() for x in flat if x and x.strip()]

    def parse_tissue_tokens(vals: Iterable[str]) -> List[str]:
        """Keep 'Blood' from 'tissue: Blood' etc."""
        parsed = []
        for c in vals:
            m = re.search(r"(?:^|\b)(tissue|organ)\s*[:=]\s*(.+)$", c, flags=re.I)
            parsed.append(m.group(2).strip() if m else c.strip())
        return [x for x in parsed if x]

    def extract_other_characteristics(vals: Iterable[str]) -> List[str]:
        """
        From Characteristics entries like ['tissue: Blood', 'cell type: Plasma', ...]
        keep items that are NOT 'tissue:' or 'organ:'.
        """
        out = []
        for c in vals:
            s = c.strip()
            if re.match(r"^(tissue|organ)\s*[:=]\s*", s, flags=re.I):
                continue
            out.append(s)
        return out

    def find_all_json(root: Path) -> List[Path]:
        matches = list(root.rglob("metadata.json"))
        matches += [p for p in root.rglob("*metadata.json") if p.name != "metadata.json"]
        seen, out = set(), []
        for p in matches:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp); out.append(p)
        return out

    def summarize_dataset(meta_path: Path) -> Optional[List[Any]]:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # print(meta)
        except Exception:
            return None

        # IDs & counts
        dataset_id = (
            dig_first(meta, ["dataset_id", "accession", "id", "series_accession", "study_accession"])
            or meta_path.parent.name
        )
        sample_count = (
            meta.get("sample_count")
            or meta.get("num_samples")
            or len(dig_nested(meta, ["sample_id"]))
            or 0
        )

        # Tissue & characteristics
        characteristics_all = dig_nested(meta, ["Characteristics"])
        tissue_vals_raw = (dig_nested(meta, ["tissue_type", "Tissue", "source_name_ch1"]) +
                           [c for c in characteristics_all if re.match(r"^(tissue|organ)\s*[:=]\s*", c, flags=re.I)])
        tissues = parse_tissue_tokens(tissue_vals_raw)
        other_chars = extract_other_characteristics(characteristics_all)

        # Library info
        lib_strategies = dig_nested(meta, ["library_strategy", "LibraryStrategy", "experiment_type"])
        lib_sources   = dig_nested(meta, ["library_source", "LibrarySource", "source"])

        # GEO cached files info
        cached_files = as_list(dig_first(meta.get("metadata", {}), ["cached_files"], default=[]))
        cf_lower = [str(x).lower() for x in cached_files]
        has_series_matrix = any("series_matrix" in x for x in cf_lower)
        # count "supplementary" cached files: everything except series_matrix
        supp_count = sum(1 for x in cf_lower if "series_matrix" not in x)

        # overall design
        overall_design = dig_first(meta, ["overall_design", "OverallDesign"], default="")

        # Build row in requested column order
        input_query = " | ".join([
            (disease or "").strip(),
            (tissue_filter or "").strip(),
            (experiment_filter or "").strip()
        ]).strip(" |")

        def clean_query(text: str) -> str:
            # Remove all special characters except spaces
            cleaned = re.sub(r"[^a-zA-Z0-9 ]+", "", text)
            # Collapse multiple spaces to one
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        cleaned_query = clean_query(query)
        return [
            cleaned_query,                    # Input Query
            disease,                        # Disease Name
            tissue_filter or "",            # Tissue Requested
            experiment_filter or "",        # Experiment Requested
            str(dataset_id),                # Accession ID
            int(sample_count),              # No. of Samples
            uniq_join(tissues),             # Tissue Type
            uniq_join(other_chars),         # Other Characteristics
            uniq_join(lib_strategies),      # Library Strategy
            uniq_join(lib_sources),         # Library Source
            "Yes" if has_series_matrix else "No",  # Series Matrix available at GEO
            supp_count,                     # Supplementary Files available at GEO (count)
            overall_design                  # Overall_design
        ]

    # ---------------------------- main body ----------------------------
    BASE_DIR = Path("./agentic_ai_wf/shared/cohort_data/GEO")
    out_dir = (BASE_DIR / output_dir).resolve()
    out_csv_path = Path(out_csv) if out_csv else (out_dir / "dataset_summary.csv")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_all_json(out_dir)
    rows: List[List[Any]] = []
    # print(files)
    for meta_path in files:
        row = summarize_dataset(meta_path)
        # print(row)
        if row: rows.append(row)

    # Stable sort: by Accession ID (index 4)
    rows.sort(key=lambda r: r[4])

    headers = [
        "Input Query",
        "Disease Name",
        "Tissue Requested",
        "Experiment Requested",
        "Accession ID",
        "No. of Samples",
        "Tissue Type",
        "Other Characteristics",
        "Library Strategy",
        "Library Source",
        "Series Matrix available at GEO",
        "Supplementary Files available at GEO",
        "Overall_design",
    ]

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows: w.writerow(r)

    return out_csv_path

# if __name__ == "__main__":
    import asyncio

    # ---- Set your test parameters here ----
    disease = "Hashimoto thyroiditis"
    tissue_filter = "blood"
    experiment_filter = "rna"

    # Prepare filters dictionary
    filters = {
        "tissue_filter": tissue_filter,
        "experiment_filter": experiment_filter
    }
    filters = {
            "tissue_filter": tissue_filter,
            "experiment_filter": experiment_filter,
            "no_of_dataset" : 2,
            "other filter" : "platform should be Illumina"
        }
    
    query = "Give 2 GEO datasets having blood samples of Hashimoto thyroiditis rna seq and platform should be Illumina"
    # ---- Run ontology fallback ----
    print("=" * 60)
    print(f"Running Ontology Fallback for: {disease}")
    print("=" * 60)
    
    
    asyncio.run(geo_ontology_fallback(disease, filters, query))
