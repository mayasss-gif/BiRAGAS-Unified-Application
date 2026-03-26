"""
DiseaseOntologyTool
- Canonical BaseTool structure (logger, async execute, helpers, validators)
- Batch ontology extraction via OpenAI Chat Completions
- Few-shot prompting (preserved)
- Strict JSON via json_schema with safe fallback to json_object
- jsonschema validation
- Optional JSON output to disk
"""

import os
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from jsonschema import validate, ValidationError
from openai import OpenAI
from pathlib import Path

# Project Imports
from   ..base.base_tool import BaseTool, ToolResult
from   ..base.base_agent import DatasetInfo  
from   ..config import CohortRetrievalConfig
from   ..exceptions import FilterError  
from   ..config import CohortRetrievalConfig
from   ..config import DirectoryPathsConfig

# -------------------------
# Strict output schema
# -------------------------
DISEASE_ONTOLOGY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical": {"type": "string"},
        "synonyms": {"type": "array", "items": {"type": "string"}},
        "children": {"type": "array", "items": {"type": "string"}},
        "parents": {"type": "array", "items": {"type": "string"}},
        "siblings": {"type": "array", "items": {"type": "string"}},
        "primary_tissues": {"type": "array", "items": {"type": "string"}},
        "associated_phenotypes": {"type": "array", "items": {"type": "string"}},
        "cross_disease_drivers": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "canonical",
        "synonyms",
        "children",
        "parents",
        "siblings",
        "primary_tissues",
        "associated_phenotypes",
        "cross_disease_drivers"
    ]
}

# -------------------------
# Criteria & result models
# -------------------------
@dataclass
class DiseaseOntologyCriteria:
    """Configuration for ontology extraction."""
    diseases: List[str]

    # LLM config
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 700
    use_json_schema: bool = True  #

    # Few-shot toggles (kept to let you swap examples later if needed)
    enable_fewshot: bool = True

    # I/O
    save_path: Optional[str] = None  # e.g., "ontology_results.json"

@dataclass
class DiseaseOntologyItem:
    """Single ontology extraction outcome for one input disease."""
    input_disease: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# -------------------------
# Tool implementation
# -------------------------
class DiseaseOntologyTool(BaseTool[List[DiseaseOntologyItem]]):
    """
    Tool for extracting disease ontology JSON via OpenAI, using the canonical tool structure:
    - setup_logger()
    - async execute()
    - _extract_batch() / _extract_one()
    - validate_input() / validate_output()
    """

    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "DiseaseOntologyTool")
        self.setup_logger()
        self.schema = DISEASE_ONTOLOGY_SCHEMA

    # ---- logger ----
    def setup_logger(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.name}.log")

        self.logger = logging.getLogger(f"{self.name}.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.propagate = False

    # ---- validators ----

    def _slugify_filename(self,name: str) -> str:
        s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"[\s-]+", "_", s)
        return s or "unnamed"
    
    def _ensure_ontology_outdir(self,config) -> str:
        """
        Resolve and create the output directory:
        - If config has output_dir, use <output_dir>/OnotologyResult
        - Else use ./OnotologyResult
        """
        base = getattr(config, "ontology_dir", None)
        outdir = os.path.join(base, "OnotologyResult") if base else os.path.join(os.getcwd(), "OnotologyResult")
        os.makedirs(outdir, exist_ok=True)
        return outdir
    
    def validate_input(self, criteria: DiseaseOntologyCriteria) -> bool:
            if not isinstance(criteria, DiseaseOntologyCriteria):
                return False
            if not isinstance(criteria.diseases, list) or not criteria.diseases:
                self.logger.error("`diseases` must be a non-empty list.")
                return False
            if not isinstance(criteria.model, str) or not criteria.model:
                self.logger.error("`model` must be a non-empty string.")
                return False
            if criteria.temperature < 0 or criteria.temperature > 2:
                self.logger.error("`temperature` must be in [0, 2].")
                return False
            if criteria.max_tokens <= 0:
                self.logger.error("`max_tokens` must be > 0.")
                return False
            return True
    
    def validate_output(self, result: List[DiseaseOntologyItem]) -> bool:
            if not isinstance(result, list):
                self.logger.error("Result must be a list")
                return False
            for item in result:
                if not isinstance(item, DiseaseOntologyItem):
                    self.logger.error("All items must be DiseaseOntologyItem")
                    return False
                if item.result is None and item.error is None:
                    self.logger.error("Each item must have either result or error")
                    return False
                if item.result is not None:
                    try:
                        validate(instance=item.result, schema=self.schema)
                    except Exception as e:
                        self.logger.error(f"Output schema validation failed: {e}")
                        return False
            return True
    
    def _extract_batch(self, criteria: DiseaseOntologyCriteria) -> List[DiseaseOntologyItem]:
        """
        Sequentially process diseases (you can parallelize with asyncio.gather later if desired).
        """
        self.logger.debug("Inside extract batch")
        client = self._make_openai_client(criteria)
        messages_fewshot = self._build_fewshot_messages() if criteria.enable_fewshot else None

        # NEW: prepare output directory once
        # outdir = self._ensure_ontology_outdir(self.config)

        config = DirectoryPathsConfig()
        outdir =  getattr(config, "ontology_dir", None)

        items: List[DiseaseOntologyItem] = []
        for disease in criteria.diseases:
            try:
                payload = self._build_messages(disease, messages_fewshot)
                data = self._call_llm_with_schema_fallback(
                    client=client,
                    model=criteria.model,
                    temperature=criteria.temperature,
                    max_tokens=criteria.max_tokens,
                    messages=payload,
                    use_json_schema=criteria.use_json_schema,
                )
                # Validate schema
                validate(instance=data, schema=self.schema)

                # NEW: save per-disease JSON
                slug = self._slugify_filename(disease)

                outdir = Path(outdir) / "OnotologyResult" / slug
                outdir.mkdir(parents=True, exist_ok=True)
                out_path = os.path.join(outdir, f"{slug}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"[Saved] {disease} -> {out_path}")
                self.logger.info(f"[Saved] {disease} -> {out_path}")

                items.append(DiseaseOntologyItem(input_disease=disease, result=data))
                self.logger.info(f"[OK] {disease}")

            except ValidationError as ve:
                msg = f"Schema validation failed: {ve.message}"
                self.logger.warning(f"[SchemaError] {disease}: {msg}")
                items.append(DiseaseOntologyItem(input_disease=disease, error=msg))

                # (Optional) write an error file too:
                err_path = os.path.join(outdir, f"{self._slugify_filename(disease)}__error.json")
                with open(err_path, "w", encoding="utf-8") as f:
                    json.dump({"input_disease": disease, "error": msg}, f, ensure_ascii=False, indent=2)

            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)}"
                self.logger.error(f"[Error] {disease}: {msg}")
                items.append(DiseaseOntologyItem(input_disease=disease, error=msg))

        self.logger.info(f"Processed {len(criteria.diseases)} diseases")
        return items
    
    def _make_openai_client(self, criteria: DiseaseOntologyCriteria) -> OpenAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise FilterError("OPENAI_API_KEY not set. Provide in env or DiseaseOntologyCriteria.openai_api_key.")
            return OpenAI(api_key=api_key)
   
    def _build_fewshot_messages(self) -> List[Dict[str, str]]:
            """Few-shot scaffold identical to your original examples (assistant JSON strictly serialized)."""
            system_msg = (
                "You are a biomedical ontology assistant. "
                "Given a disease name, return ONLY a JSON object that matches the schema. "
                "No extra fields, no prose. Use common biomedical terminology."
            )

            fewshot_user_1 = "Disease: asthma"
            fewshot_assistant_1 = {
                "canonical": "asthma",
                "synonyms": ["bronchial asthma"],
                "children": ["allergic asthma", "nonallergic asthma", "exercise-induced bronchoconstriction"],
                "parents": ["obstructive lung disease", "chronic inflammatory airway disease"],
                "siblings": ["chronic obstructive pulmonary disease", "bronchiectasis"],
                "primary_tissues": ["bronchial epithelium", "airway smooth muscle", "lung parenchyma"],
                "associated_phenotypes": ["wheezing", "airway hyperresponsiveness", "reversible airflow obstruction"],
                "cross_disease_drivers": ["allergy", "air pollution", "respiratory infections"]
            }

            fewshot_user_2 = "Disease: type 2 diabetes mellitus"
            fewshot_assistant_2 = {
                "canonical": "type 2 diabetes mellitus",
                "synonyms": ["T2DM", "adult-onset diabetes", "non–insulin-dependent diabetes mellitus"],
                "children": ["diabetic nephropathy", "diabetic retinopathy", "diabetic neuropathy"],
                "parents": ["diabetes mellitus", "metabolic disease"],
                "siblings": ["type 1 diabetes mellitus", "gestational diabetes"],
                "primary_tissues": ["pancreatic islet", "liver", "skeletal muscle", "adipose tissue"],
                "associated_phenotypes": ["hyperglycemia", "insulin resistance", "impaired glucose tolerance"],
                "cross_disease_drivers": ["obesity", "sedentary lifestyle"]
            }

            return [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": fewshot_user_1},
                {"role": "assistant", "content": json.dumps(fewshot_assistant_1, ensure_ascii=False)},
                {"role": "user", "content": fewshot_user_2},
                {"role": "assistant", "content": json.dumps(fewshot_assistant_2, ensure_ascii=False)},
            ]
    
    def _build_messages(
            self,
            disease_name: str,
            messages_fewshot: Optional[List[Dict[str, str]]] = None
        ) -> List[Dict[str, str]]:
            user_query = f"Disease: {disease_name}"
            if messages_fewshot:
                return [*messages_fewshot, {"role": "user", "content": user_query}]
            else:
                # Minimal non-fewshot system + user
                return [
                    {
                        "role": "system",
                        "content": (
                            "You are a biomedical ontology assistant. "
                            "Given a disease name, return ONLY a JSON object that matches the schema. "
                            "No extra fields, no prose. Use common biomedical terminology."
                        ),
                    },
                    {"role": "user", "content": user_query},
                ]
    
    def _call_llm_with_schema_fallback(
            self,
            client: OpenAI,
            model: str,
            temperature: float,
            max_tokens: int,
            messages: List[Dict[str, str]],
            use_json_schema: bool = True,
        ) -> Dict[str, Any]:
            """
            Try structured outputs with response_format json_schema, then fallback to json_object.
            """
            # Try strict schema
            if use_json_schema:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "DiseaseOntology", "schema": self.schema},
                        },
                        messages=messages,
                    )
                    text = resp.choices[0].message.content
                    data = json.loads(text)
                    return data
                except Exception as e:
                    # If the issue wasn't with response_format support, log and continue
                    self.logger.debug(f"json_schema path failed, falling back to json_object: {e}")

            # Fallback JSON mode
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            text = resp.choices[0].message.content
            data = json.loads(text)
            return data
        
    async def execute(self, criteria: DiseaseOntologyCriteria) -> ToolResult[List[DiseaseOntologyItem]]:
           # ---- internal helpers ----
        if not self.validate_input(criteria):
            self.logger.error("Invalid input parameters")
            return ToolResult(success=False, error="Invalid input parameters", details=None)

        try:
            self.logger.debug("Running with retry...")
            res = self._extract_batch(criteria)
        except Exception as e:
            print(f"Failed{str(e)}")
            self.logger.exception("Batch extraction failed")
            return ToolResult(success=False, error=str(e), details=None)

        # --- unwrap run_with_retry return ---
        retry_count = 0
        items = res
        if isinstance(res, tuple):
            items = res[0]
            if len(res) > 1 and isinstance(res[1], int):
                retry_count = res[1]
        # Optional save
        if criteria.save_path:
            try:
                to_save = []
                for it in items:
                    if getattr(it, "error", None):
                        to_save.append({"input_disease": it.input_disease, "error": it.error})
                    else:
                        to_save.append({"input_disease": it.input_disease, "result": it.result})
                with open(criteria.save_path, "w", encoding="utf-8") as f:
                    json.dump(to_save, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved results to {criteria.save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save results to {criteria.save_path}: {e}")

        # Validate final shape
        if not self.validate_output(items):
            try:
                preview = [getattr(items[0], "__dict__", str(items[0]))] if items else []
            except Exception:
                preview = ["<unavailable>"]
            self.logger.error("Output validation failed. type(items)=%s preview=%s",
                            type(items).__name__, preview)
            return ToolResult(success=False, error="Output validation failed",
                            details={"items_type": type(items).__name__,
                                    "len": len(items) if hasattr(items, "__len__") else None})

        # SUCCESS: note we use data=... (your ToolResult shows data field)
        return ToolResult(success=True, data=items, retry_count=retry_count)

 