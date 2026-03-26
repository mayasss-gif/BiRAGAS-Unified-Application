"""
LLM-assisted wizard to generate:

  - ModelSelection.txt
  - GeneSelection.txt
  - Explanation.txt

for the DepMap downstream pipeline.

Behavior:
- Reads master cell-line table from: CellLines_Master_AllModels.csv/xlsx
- Asks the user <= 5 short questions.
- Uses an LLM to infer:
    * model selection (mode + parameters)
    * gene selection (mode all vs top, top counts)
- Writes commented template .txt files with exactly ONE active mode block.

Defaults:
- If the user does NOT explicitly choose "top genes" and numbers,
  the wizard configures gene selection as "all genes".
- Examples for "top genes" use 20 up and 20 down, so plots can
  naturally focus on ~20 genes in each direction.
"""

import os
import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")


# -------------------------------------------------------------------
# CONFIGURE PATHS HERE
# -------------------------------------------------------------------


# Master metadata table (DepMap-like)
from .constants import MASTER_FILE


# -------------------------------------------------------------------
# LOAD DATA & OPENAI CLIENT
# -------------------------------------------------------------------

def load_openai_client() -> OpenAI:
    """Load OpenAI API key from file and return a client."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


def load_master_table() -> pd.DataFrame:
    """Load the DepMap metadata master table."""
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Cell line master file not found: {MASTER_FILE}")
    if MASTER_FILE.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(MASTER_FILE)
    else:
        df = pd.read_csv(MASTER_FILE)

    required_cols = {
        "ModelID",
        "CellLineName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "has_CRISPR_effect",
        "has_CRISPR_dependency",
        "has_expression_TPMlog1p",
        "has_CNV_WGS",
        "in_all_layers",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Master file is missing required columns: {missing}")
    return df


# -------------------------------------------------------------------
# LLM CALL
# -------------------------------------------------------------------

def ask_llm_for_config(
    client: OpenAI,
    lineages: list[str],
    diseases: list[str],
    cell_lines: list[str],
    user_context: dict,
) -> dict:
    """
    Call the LLM with all available lineages/diseases/cell lines and the
    user's high-level answers. Return a JSON dict with:

      {
        "model_selection": {...},
        "gene_selection": {...},
        "explanation": "text"
      }
    """
    system_prompt = """
You are a bioinformatics assistant helping configure a DepMap-based downstream analysis.

CONTEXT ABOUT THE DATASET
-------------------------
- The dataset is cancer-centric (DepMap-like cell lines).
- Available dimensions:
  - OncotreeLineage: high-level tissue/lineage (e.g. Bladder/Urinary Tract, Myeloid, Lung, ...)
  - OncotreePrimaryDisease: specific tumor type or hematologic malignancy, or "Non-Cancerous".
  - CellLineName: a long list of ~800 cell lines.
- The user will NOT see the full list of diseases/lineages/cell lines; only YOU see them.
- You must infer the best mapping from their free-text answers.

BEHAVIOR RULES
--------------
1. Use ONLY values that appear in the provided lists for:
   - OncotreeLineage
   - OncotreePrimaryDisease
   - CellLineName

2. Mapping non-cancer diseases (very important):
   - If the user asks for a disease that is NOT in the cancer list (e.g. Lupus, Multiple Sclerosis, other autoimmune/neurologic disorders):
     - Clearly (in the explanation) state that the dataset is cancer-focused.
     - Propose the MOST BIOLOGICALLY RELEVANT CANCER(S) to simulate this condition using the available data, for example:
         * Lupus / systemic autoimmunity → B-cell or T-cell malignancies, mature B-cell neoplasms, T-cell lymphomas, etc.
         * Multiple sclerosis / CNS autoimmunity → CNS/Brain tumors and/or immune malignancies with CNS relevance.
         * Other autoimmune conditions → immune cell cancers or tumors from the same affected organ/tissue.
     - Choose one or more existing OncotreePrimaryDisease values that best match this rationale (do NOT invent new disease labels).
     - Provide a SHORT but STRONG biological explanation (1–3 sentences) connecting the surrogate cancer type to the user’s disease of interest (shared immune mechanisms, organ microenvironment, inflammatory context, etc.).

3. Mapping cell lines:
   - If the user mentions a cell line not in the list:
     - DO NOT invent new cell lines.
     - Instead, suggest the most similar existing cell line(s) based on:
         * Name similarity (string resemblance)
         * Tissue / lineage / primary disease match
     - Then use ONLY those existing cell line names in your config.

4. Decide the model selection MODE intelligently:
   - You will receive a hint (model_mode_hint), but it may be null.
   - If the hint is null, infer the best mode based on the user’s disease description.
     * Example: if they just say “I want bladder cancer models” → prefer mode="by_disease" with the matching OncotreePrimaryDisease.
     * If they mention specific cell lines → use mode="by_names".
   - Never choose multiple modes at once; pick ONE.

5. Decide the gene selection MODE intelligently:
   - You will receive a hint (gene_mode_hint = "all" or "top").
   - If gene_mode_hint is null, default to mode="all".
   - If gene_mode_hint="all":
       * Set gene_selection.mode="all"
       * Set top_up=0, top_down=0
   - If gene_mode_hint="top":
       * Set gene_selection.mode="top"
       * Use top_up and top_down from top_up_hint / top_down_hint if provided.
       * If no useful numbers are provided in hints, default to:
           - top_up=20
           - top_down=20
         so that plots naturally focus on ~20 genes in each direction.

6. Multiple plausible matches (VERY IMPORTANT):
   - When there are several reasonable matching diseases or lineages for the user’s description,
     you should include ALL relevant ones in the configuration (not just one).
   - For example, for "leukemia" you might include multiple leukemic primary diseases
     (e.g. acute myeloid leukemia, B-cell acute lymphoblastic leukemia, T-lymphoblastic leukemia/lymphoma, etc.)
     that are present in the provided OncotreePrimaryDisease list.
   - The same applies to lineages and cell lines: if multiple clearly relevant items exist,
     include all of them so the user can later narrow them manually if needed.

EXPLANATION FORMAT (VERY IMPORTANT)
-----------------------------------
In the "explanation" field of the JSON:

1. First give 1–3 sentences of biological reasoning.
2. Then explicitly list the final selected items in a structured way, for example:

   Selected primary diseases:
   1) Acute Myeloid Leukemia
   2) T-Lymphoblastic Leukemia/Lymphoma
   3) ...

   Selected lineages:
   1) Myeloid
   2) Lymphoid

   Selected cell lines (if mode = by_names):
   1) HL60
   2) THP1
   3) U937

- Only include sections that are actually used.
- This list is for the user to see *all* options your config is using.

OUTPUT FORMAT (STRICT JSON)
---------------------------
Return EXACTLY this structure in JSON (no extra keys, no comments):

{
  "model_selection": {
    "mode": "by_disease" | "by_lineage" | "by_ids" | "by_names" | "keyword",
    "diseases": [ ... ],
    "lineages": [ ... ],
    "ids": [ ... ],
    "names": [ ... ],
    "keywords": [ ... ],
    "keyword_mode": "any" | "all"
  },
  "gene_selection": {
    "mode": "all" | "top",
    "top_up": int,
    "top_down": int
  },
  "explanation": "string"
}

CONSTRAINTS
-----------
- "mode" must be ONE of the allowed values.
- If mode="by_disease": fill "diseases" (non-empty), other lists empty.
- If mode="by_lineage": fill "lineages" (non-empty), others empty.
- If mode="by_ids": fill "ids".
- If mode="by_names": fill "names".
- If mode="keyword": fill "keywords" (non-empty) and "keyword_mode".
- For gene_selection:
    * If mode="all": top_up=0 and top_down=0.
    * If mode="top": top_up>0 and top_down>0.
- Do NOT ask the user additional questions; infer everything from the provided context.
"""

    user_message = {
        "available_lineages": lineages,
        "available_diseases": diseases,
        "available_cell_lines": cell_lines,
        "user_context": user_context,
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_message),
            },
        ],
    )

    content = response.choices[0].message.content
    try:
        cfg = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Raw response:\n{content}") from e

    return cfg


# -------------------------------------------------------------------
# BUILD .TXT FILES
# -------------------------------------------------------------------

def build_modelselection_txt(cfg: dict, path: Path) -> None:
    """Build ModelSelection.txt from the LLM configuration."""
    ms = cfg["model_selection"]
    mode = ms["mode"]

    lines: list[str] = []
    lines.append("#######################################################################")
    lines.append("# ModelSelection.txt (AUTO-GENERATED BY LLM)")
    lines.append("# You can edit this file manually if needed.")
    lines.append("#######################################################################")
    lines.append("")

    def active(line: str) -> str:
        return line

    def inactive(line: str) -> str:
        return "# " + line

    # BLOCK A: by_disease
    lines.append("#######################################################################")
    lines.append("# BLOCK A: SELECT MODELS BY DISEASE")
    lines.append("#######################################################################")
    if mode == "by_disease":
        lines.append(active("mode=by_disease"))
        if ms.get("diseases"):
            lines.append(active("diseases=" + ", ".join(ms["diseases"])))
    else:
        lines.append(inactive("mode=by_disease"))
        lines.append(inactive("diseases=Bladder Urothelial Carcinoma"))
    lines.append("")

    # BLOCK B: by_lineage
    lines.append("#######################################################################")
    lines.append("# BLOCK B: SELECT MODELS BY LINEAGE")
    lines.append("#######################################################################")
    if mode == "by_lineage":
        lines.append(active("mode=by_lineage"))
        if ms.get("lineages"):
            lines.append(active("lineages=" + ", ".join(ms["lineages"])))
    else:
        lines.append(inactive("mode=by_lineage"))
        lines.append(inactive("lineages=Bladder, Lung, Breast"))
    lines.append("")

    # BLOCK C: by_ids
    lines.append("#######################################################################")
    lines.append("# BLOCK C: SELECT MODELS BY DepMap MODEL IDs")
    lines.append("#######################################################################")
    if mode == "by_ids":
        lines.append(active("mode=by_ids"))
        if ms.get("ids"):
            lines.append(active("ids=" + ", ".join(ms["ids"])))
    else:
        lines.append(inactive("mode=by_ids"))
        lines.append(inactive("ids=ACH-000017, ACH-000448"))
    lines.append("")

    # BLOCK D: by_names
    lines.append("#######################################################################")
    lines.append("# BLOCK D: SELECT MODELS BY CELL LINE NAMES")
    lines.append("#######################################################################")
    if mode == "by_names":
        lines.append(active("mode=by_names"))
        if ms.get("names"):
            lines.append(active("names=" + ", ".join(ms["names"])))
    else:
        lines.append(inactive("mode=by_names"))
        lines.append(inactive("names=T24, MCF7, HCT116"))
    lines.append("")

    # BLOCK E: keyword
    lines.append("#######################################################################")
    lines.append("# BLOCK E: SELECT MODELS BY KEYWORD SEARCH")
    lines.append("#######################################################################")
    if mode == "keyword":
        lines.append(active("mode=keyword"))
        if ms.get("keywords"):
            lines.append(active("keywords=" + ", ".join(ms["keywords"])))
        kmode = ms.get("keyword_mode", "any")
        lines.append(active(f"keyword_mode={kmode}   # any or all"))
    else:
        lines.append(inactive("mode=keyword"))
        lines.append(inactive("keywords=bladder, carcinoma"))
        lines.append(inactive("keyword_mode=any   # or: all"))
    lines.append("")
    lines.append("#######################################################################")
    lines.append("# Exactly ONE 'mode=...' above should be active (not commented).")
    lines.append("#######################################################################")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_geneselection_txt(cfg: dict, path: Path) -> None:
    """Build GeneSelection.txt from the LLM configuration."""
    gs = cfg["gene_selection"]
    mode = gs["mode"]
    top_up = int(gs.get("top_up", 0) or 0)
    top_down = int(gs.get("top_down", 0) or 0)

    lines: list[str] = []
    lines.append("#######################################################################")
    lines.append("# GeneSelection.txt (AUTO-GENERATED BY LLM)")
    lines.append("# You can edit this file manually if needed.")
    lines.append("#######################################################################")
    lines.append("")

    def active(line: str) -> str:
        return line

    def inactive(line: str) -> str:
        return "# " + line

    # BLOCK A: all genes
    lines.append("#######################################################################")
    lines.append("# BLOCK A: USE ALL GENES")
    lines.append("# This mode passes ALL genes from prepared_DEGs_Simple.csv downstream.")
    lines.append("#######################################################################")
    if mode == "all":
        lines.append(active("mode=all"))
    else:
        lines.append(inactive("mode=all"))
    lines.append("")

    # BLOCK B: top up/down
    lines.append("#######################################################################")
    lines.append("# BLOCK B: USE TOP UP/DOWN GENES BY LOG2FC")
    lines.append("# This mode is useful if you want to limit plots/tables to a subset,")
    lines.append("# e.g. 20 most up-regulated and 20 most down-regulated genes.")
    lines.append("#######################################################################")
    if mode == "top":
        lines.append(active("mode=top"))
        lines.append(active(f"top_up={top_up}"))
        lines.append(active(f"top_down={top_down}"))
    else:
        lines.append(inactive("mode=top"))
        lines.append(inactive("top_up=20"))
        lines.append(inactive("top_down=20"))
    lines.append("")
    lines.append("#######################################################################")
    lines.append("# Exactly ONE 'mode=...' above should be active (not commented).")
    lines.append("# Default behavior: if you do NOT explicitly choose 'top',")
    lines.append("# the pipeline will use ALL genes for downstream analysis.")
    lines.append("#######################################################################")

    path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------
# POST-LLM REFINEMENT: LET USER PICK SUBSET FROM OPTIONS
# -------------------------------------------------------------------

def refine_model_selection_from_user(cfg: dict) -> dict:
    """
    After LLM proposal, let the user refine which diseases/lineages/names/keywords
    to keep, by selecting indices or pressing Enter to keep all.
    """
    ms = cfg.get("model_selection", {})
    mode = ms.get("mode", "")

    key = None
    label = None

    if mode == "by_disease":
        key = "diseases"
        label = "primary diseases"
    elif mode == "by_lineage":
        key = "lineages"
        label = "lineages"
    elif mode == "by_names":
        key = "names"
        label = "cell lines"
    elif mode == "by_ids":
        key = "ids"
        label = "DepMap ModelIDs"
    elif mode == "keyword":
        key = "keywords"
        label = "keywords"

    if not key:
        return cfg

    items = ms.get(key) or []
    if not items:
        return cfg

    print(f"\nProposed {label} from LLM:")
    for i, v in enumerate(items, start=1):
        print(f"  {i}) {v}")

    return cfg


# -------------------------------------------------------------------
# MAIN INTERACTIVE FLOW (max ~5 questions about content)
# -------------------------------------------------------------------

def build_selection_files(
    output_dir: Path,
    disease: str,
    mode_model: str = None,
    genes_selection: str = "all",
    top_up: int = None,
    top_down: int = None,
):

    """
    Builds the ModelSelection.txt and GeneSelection.txt files for the DepMap downstream pipeline.

    Args:
        output_dir (Path): The directory to write the files to.
        disease (str): The disease to model. Example: "Bladder Urothelial Carcinoma", "lupus", "multiple sclerosis".
        mode_model (str): The mode to use for selecting models. Example: "by_disease", "by_lineage", "by_ids", "by_names", "keyword". Default: None.
        genes_selection (str): The mode to use for selecting genes. Example: "all", "top". Default: "all".
        top_up (int): The number of top up genes to select. Example: 20. Default: None.
        top_down (int): The number of top down genes to select. Example: 20. Default: None.
    """
    print("=== LLM-ASSISTED CONFIG WIZARD FOR DepMap ANALYSIS ===")

    client = load_openai_client()
    df = load_master_table()

    lineages = sorted(df["OncotreeLineage"].dropna().unique().tolist())
    diseases = sorted(df["OncotreePrimaryDisease"].dropna().unique().tolist())
    cell_lines = sorted(df["CellLineName"].dropna().unique().tolist())

    print(f"\nLoaded master table from: {MASTER_FILE}")
    print(f"- {len(lineages)} unique OncotreeLineage")
    print(f"- {len(diseases)} unique OncotreePrimaryDisease")
    print(f"- {len(cell_lines)} cell lines\n")


    mode_map = {
        "1": "by_disease",
        "by_disease": "by_disease",
        "disease": "by_disease",

        "2": "by_lineage",
        "by_lineage": "by_lineage",
        "lineage": "by_lineage",

        "3": "by_ids",
        "by_ids": "by_ids",
        "ids": "by_ids",

        "4": "by_names",
        "by_names": "by_names",
        "names": "by_names",
        "cellline": "by_names",
        "celllines": "by_names",

        "5": "keyword",
        "keyword": "keyword",
        "keywords": "keyword",
    }

    model_mode = mode_map.get(mode_model, None)



    if genes_selection == "top" and top_up is None and top_down is None:
        top_up = 20
        top_down = 20


    # Build user_context for LLM
    user_context = {
        "disease_interest": disease,
        "model_mode_hint": model_mode,
        "gene_mode_hint": genes_selection,
        "top_up_hint": top_up,
        "top_down_hint": top_down,
    }

    print("\nCalling LLM to propose model & gene selection based on your inputs...")
    cfg = ask_llm_for_config(
        client=client,
        lineages=lineages,
        diseases=diseases,
        cell_lines=cell_lines,
        user_context=user_context,
    )

    print("\n=== LLM EXPLANATION ===")
    print(cfg.get("explanation", "No explanation provided."))

    # 🔹 NEW: give you a chance to select from the proposed options
    cfg = refine_model_selection_from_user(cfg)


    output_dir.mkdir(parents=True, exist_ok=True)
    model_sel_path = output_dir / "ModelSelection.txt"
    gene_sel_path = output_dir / "GeneSelection.txt"
    explanation_path = output_dir / "Explanation.txt"

    build_modelselection_txt(cfg, model_sel_path)
    build_geneselection_txt(cfg, gene_sel_path)

    # Save explanation + raw JSON config
    explanation_text = cfg.get("explanation", "No explanation provided.")
    expl_lines = [
        "### LLM explanation for DepMap configuration",
        "",
        explanation_text,
        "",
        "----------------------------------------",
        "Raw JSON configuration:",
        "",
        json.dumps(cfg, indent=2),
        "",
    ]
    explanation_path.write_text("\n".join(expl_lines), encoding="utf-8")

    print("\n✅ Configuration files written:")
    print("  -", model_sel_path)
    print("  -", gene_sel_path)
    print("  -", explanation_path)
    print("\nDefault behavior: ALL genes are used unless you explicitly switch")
    print("GeneSelection.txt to 'mode=top'. You can now run your downstream")
    print("DepMap pipeline using these files.")


# if __name__ == "__main__":
#     build_selection_files(
#         output_dir=Path("Output"),
#         disease="Bladder Urothelial Carcinoma",
#         # mode_model="by_disease",
#         # genes_selection="all",
#         # top_up=None,
#         # top_down=None,
#     )
