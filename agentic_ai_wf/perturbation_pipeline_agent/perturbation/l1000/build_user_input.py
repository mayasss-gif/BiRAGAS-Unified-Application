#!/usr/bin/env python
"""
LLM-guided configuration builder for L1000 analysis.

- Reads Metadata.csv to learn available primary_site, subtype, cell lines.
- Asks the user up to ~5 concise questions about the desired indication / context.
- Calls an LLM (OpenAI) using the key in OpenAIkey.txt.
- LLM proposes:
    * Primary site (must map to one of the L1000 primary_site values)
    * Optional cell whitelist (by default: NONE → use all cell lines)
    * Drug / compound (optional)
    * Time points (hours) – by default: ALL time points ("all")
    * Perturbation type (e.g. trt_cp, trt_sh, trt_oe, etc.)
    * Max signatures, max controls per stratum
  plus a biological explanation (especially if the disease is not directly represented).

- Writes:
    src/userinput.txt
    src/Explanation.txt

Example userinput.txt:

    Primary site: breast
    Cell whitelist:
    Drug:
    Times (h): all
    Pert.Type:
    Max sigs: 400000
    Max ctrls/str: 10
    Include Relevance: TRUE
    Include ATE: TRUE
    Export Plots: TRUE
    Auto-augment ≥4 doses/pair: TRUE
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from openai import OpenAI
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")


# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
from .constants import METADATA_CSV


DEFAULT_MAX_SIGS = 400000
DEFAULT_MAX_CTRLS_PER_STR = 10


# ---------------------------------------------------------------------
# OpenAI client loader
# ---------------------------------------------------------------------
def load_openai_client() -> OpenAI:
    """
    Load OpenAI client using the API key stored in the environment variables.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------
def load_metadata() -> pd.DataFrame:
    """
    Load L1000 Metadata.csv which contains at least:
      - cell_id
      - primary_site
      - subtype
    """
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_CSV}\n"
            "Make sure Metadata.csv is present in the project root."
        )

    df = pd.read_csv(METADATA_CSV)
    required_cols = {"cell_id", "primary_site", "subtype"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata.csv is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


def build_metadata_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a compact summary of primary_site / subtype / cells
    to send to the LLM as context.
    """
    primary_sites = sorted(str(s) for s in df["primary_site"].dropna().unique())

    subtype_counts = (
        df["subtype"]
        .fillna("-666")
        .astype(str)
        .value_counts()
    )
    top_subtypes = subtype_counts.head(50).index.tolist()

    site_to_cells = {}
    for site in primary_sites:
        small = df[df["primary_site"] == site]["cell_id"].dropna().astype(str)
        site_to_cells[site] = sorted(small.unique().tolist())[:10]

    return {
        "primary_sites": primary_sites,
        "top_subtypes": top_subtypes,
        "site_to_cells": site_to_cells,
    }


# ---------------------------------------------------------------------
# Simple interactive Q&A (≤ 5 questions)
# ---------------------------------------------------------------------
def ask_user_questions() -> Dict[str, Any]:
    """
    Ask the user a small number of questions about what they want to model.
    Returns a dict of raw answers (strings).
    """
    print("=== L1000 LLM CONFIG WIZARD ===\n")

    disease_desc = input(
        "1) Describe the disease/indication you want to model\n"
        "   (e.g. 'Bladder Urothelial Carcinoma', 'lupus', 'multiple sclerosis'):\n> "
    ).strip()

    primary_hint = input(
        "\n2) Optional: tissue / primary site hint "
        "(e.g. 'breast', 'lung', 'bone'; leave blank to let the model choose):\n> "
    ).strip()

    drug = input(
        "\n3) Optional: specific drug or compound of interest "
        "(e.g. 'trastuzumab', 'doxorubicin'; leave blank if none):\n> "
    ).strip()

    times = input(
        "\n4) Preferred treatment time points in hours "
        "(e.g. '6,24'; leave blank to use ALL available time points):\n> "
    ).strip()

    cell_hint = input(
        "\n5) Optional: specific cell line(s) you like (comma-separated, e.g. "
        "'A375, BT20'; leave blank to use ALL relevant cell lines):\n> "
    ).strip()

    return {
        "disease_desc": disease_desc,
        "primary_hint": primary_hint,
        "drug": drug,
        "times": times,
        "cell_hint": cell_hint,
    }


# ---------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------
def call_llm_for_config(
    client: OpenAI,
    meta_summary: Dict[str, Any],
    user_answers: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Call the LLM with metadata context + user answers,
    ask it to output a strict JSON configuration.

    Expected JSON keys:
        primary_site        (str, MUST be one of primary_sites)
        cell_whitelist      (list[str])
        drug                (str)
        times_h             (list[int] or ['all'])
        pert_type           (str)
        max_sigs            (int)
        max_ctrls_per_str   (int)
        explanation         (str)
    """
    primary_sites = meta_summary["primary_sites"]
    top_subtypes = meta_summary["top_subtypes"]
    site_to_cells = meta_summary["site_to_cells"]

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert in cancer biology and high-throughput L1000 perturbation screens.\n"
            "You will receive:\n"
            "  * A list of available primary_site values from the L1000 metadata.\n"
            "  * Some common subtypes.\n"
            "  * Example cell lines per primary_site.\n"
            "  * A user's free-text disease/indication description and preferences.\n\n"
            "Your job is to choose the best L1000 context to approximate the user's biology "
            "and to propose a small configuration for downstream analysis.\n\n"
            "IMPORTANT:\n"
            "- The L1000 data is largely **cancer-focused**.\n"
            "- If the user asks for a non-cancer disease (e.g. lupus, multiple sclerosis, "
            "thrombocytothemia), you MUST:\n"
            "    * Clearly state that the L1000 data is cancer-oriented.\n"
            "    * Choose the most biologically relevant cancer context (e.g. haematopoietic "
            "or myeloproliferative neoplasms for blood/platelet diseases) and justify it.\n"
            "- The field `primary_site` MUST be one of the provided primary_site options.\n"
            "- `pert_type` can be any valid L1000 perturbation type (e.g. 'trt_cp', 'trt_sh', "
            "'trt_oe', etc.). You are **not restricted** to 'trt_cp'. Choose what is most "
            "appropriate, or use 'trt_cp' when in doubt.\n"
            "- If the user does not give a drug name, leave `drug` as an empty string.\n"
            "- If the user explicitly provides time points, honour them if reasonable.\n"
            "- If the user does NOT specify time points, set `times_h` to ['all'] to indicate "
            "that ALL available time points should be used.\n"
            "- `cell_whitelist` can be empty ([]) or a small list (1–5) of specific cell_ids.\n"
            "- **By default, you should leave `cell_whitelist` empty to allow ALL cell lines** "
            "for the chosen primary_site. Only propose a non-empty whitelist if the user "
            "explicitly requests specific cell lines or a narrow panel.\n"
            "- If user hints at cell lines, try to use them or closest available.\n"
            "- If user hints at tissue (primary site), try to match or use closest.\n"
            "- Use 400000 as a sensible default for `max_sigs` and 10 for `max_ctrls_per_str` "
            "unless there is a strong reason to change.\n"
            "- When the user does **not** specify both drug and time points, include a clear "
            "sentence in the explanation such as:\n"
            "  'No drug or time points were specified, so a default perturbation type and all "
            "available time points are used for a balanced analysis.'\n\n"
            "Output MUST be a single valid JSON object with keys:\n"
            "  primary_site (string)\n"
            "  cell_whitelist (array of strings)\n"
            "  drug (string)\n"
            "  times_h (array of integers OR ['all'])\n"
            "  pert_type (string)\n"
            "  max_sigs (integer)\n"
            "  max_ctrls_per_str (integer)\n"
            "  explanation (string)\n"
            "Do NOT wrap the JSON in backticks or any other text."
        ),
    }

    user_payload = {
        "primary_sites": primary_sites,
        "top_subtypes": top_subtypes,
        "site_to_example_cells": site_to_cells,
        "user_answers": user_answers,
    }

    user_msg = {
        "role": "user",
        "content": json.dumps(user_payload, indent=2),
    }

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[system_msg, user_msg],
        temperature=0.3,
    )

    try:
        text = resp.output[0].content[0].text
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response format: {e}\nRaw: {resp}")

    text = text.strip()
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON. Error: {e}\nReturned text:\n{text}"
        )

    return cfg


# ---------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------
def normalize_config(
    cfg: Dict[str, Any],
    meta_summary: Dict[str, Any],
    user_answers: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure config fields exist and have sensible defaults.
    Enforce that primary_site is among known primary_sites; if not, choose a fallback.
    ALSO: if user did NOT provide any cell_hint, force cell_whitelist = [] (ALL cell lines).
    """
    primary_sites: List[str] = meta_summary["primary_sites"]

    primary_site = str(cfg.get("primary_site", "")).strip()
    if primary_site not in primary_sites:
        lower_map = {p.lower(): p for p in primary_sites}
        if primary_site.lower() in lower_map:
            primary_site = lower_map[primary_site.lower()]
        else:
            if "breast" in primary_sites:
                primary_site = "breast"
            else:
                primary_site = primary_sites[0] if primary_sites else "unknown"

    # Cell whitelist
    cell_whitelist = cfg.get("cell_whitelist", [])
    if not isinstance(cell_whitelist, list):
        cell_whitelist = []

    # If user did NOT provide any cell_hint → do NOT restrict cell lines
    cell_hint_raw = user_answers.get("cell_hint", "") or ""
    if cell_hint_raw.strip() == "":
        cell_whitelist = []

    drug = cfg.get("drug", "")
    if drug is None:
        drug = ""
    drug = str(drug)

    times_h = cfg.get("times_h", [])
    if not isinstance(times_h, list) or not times_h:
        times_h = ["all"]
    else:
        if len(times_h) == 1 and str(times_h[0]).strip().lower() == "all":
            times_h = ["all"]
        else:
            clean_times: List[int] = []
            for t in times_h:
                try:
                    clean_times.append(int(t))
                except Exception:
                    continue
            if clean_times:
                times_h = clean_times
            else:
                times_h = ["all"]

    pert_type = cfg.get("pert_type", "")
    pert_type = str(pert_type).strip()

    try:
        max_sigs = int(cfg.get("max_sigs", DEFAULT_MAX_SIGS))
    except Exception:
        max_sigs = DEFAULT_MAX_SIGS

    try:
        max_ctrls = int(cfg.get("max_ctrls_per_str", DEFAULT_MAX_CTRLS_PER_STR))
    except Exception:
        max_ctrls = DEFAULT_MAX_CTRLS_PER_STR

    explanation = cfg.get("explanation", "")
    explanation = str(explanation)

    return {
        "primary_site": primary_site,
        "cell_whitelist": cell_whitelist,
        "drug": drug,
        "times_h": times_h,
        "pert_type": pert_type,
        "max_sigs": max_sigs,
        "max_ctrls_per_str": max_ctrls,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------
def write_userinput_txt(cfg: Dict[str, Any], output_dir: Path) -> None:
    """
    Write src/userinput.txt in the requested format.
    Other options are always TRUE.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if (
        isinstance(cfg["times_h"], list)
        and len(cfg["times_h"]) == 1
        and str(cfg["times_h"][0]).strip().lower() == "all"
    ):
        times_str = "all"
    else:
        times_str = ", ".join(str(t) for t in cfg["times_h"])

    cell_whitelist_str = ", ".join(cfg["cell_whitelist"])

    lines = [
        f"Primary site: {cfg['primary_site']}",
        f"Cell whitelist: {cell_whitelist_str}",
        f"Drug: {cfg['drug']}",
        f"Times (h): {times_str}",
        f"Pert.Type: {cfg['pert_type']}",
        f"Max sigs: {cfg['max_sigs']}",
        f"Max ctrls/str: {cfg['max_ctrls_per_str']}",
        "Include Relevance: TRUE",
        "Include ATE: TRUE",
        "Export Plots: TRUE",
        "Auto-augment \u22654 doses/pair: TRUE",
        "",
    ]
    USERINPUT_TXT = output_dir / "userinput.txt"
    USERINPUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def write_explanation_txt(explanation: str, output_dir: Path) -> None:
    """
    Save the LLM biological explanation as src/Explanation.txt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    EXPLANATION_TXT = output_dir / "Explanation.txt"
    EXPLANATION_TXT.write_text(explanation.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_user_input(
    output_dir: Path,
    disease: str,
    tissue: str,
    drug: str,
    time_points: str,
    cell_lines: str,
    max_sigs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the user input for the L1000 analysis.

    Parameters:
        output_dir: Path to the output directory
        disease: Disease of interest (e.g. "Bladder Urothelial Carcinoma")
        tissue: Tissue of interest (e.g. "Breast", "Lung", "Bone")
        drug: Drug of interest (e.g. "Trastuzumab", "Doxorubicin")
        time_points: Time points of interest (e.g. "6,24", "all")
        cell_lines: Cell lines of interest (e.g. "A375, BT20", "all")
        max_sigs_override: If provided, force this max_sigs value instead of the LLM output
    Returns:
        The configuration dictionary.
    """
    
    user_answers = {
        "disease_desc": disease,
        "primary_hint": tissue,
        "drug": drug,
        "times": time_points,
        "cell_hint": cell_lines,
    }

    # 1) Load metadata
    df_meta = load_metadata()
    meta_summary = build_metadata_summary(df_meta)

    # 2) Ask user questions
    answers = user_answers

    # 3) Call LLM
    client = load_openai_client()
    cfg_raw = call_llm_for_config(client, meta_summary, answers)

    # 4) Normalize & validate (with access to user_answers)
    cfg = normalize_config(cfg_raw, meta_summary, answers)
    if max_sigs_override is not None:
        try:
            cfg["max_sigs"] = int(max_sigs_override)
        except Exception:
            cfg["max_sigs"] = DEFAULT_MAX_SIGS

    # 5) Show summary & save
    print("\n=== LLM PROPOSED CONFIGURATION ===")
    print(f"Primary site      : {cfg['primary_site']}")
    print(f"Cell whitelist    : {', '.join(cfg['cell_whitelist']) or '(none → ALL cell lines)'}")
    print(f"Drug              : {cfg['drug'] or '(none specified)'}")

    if (
        isinstance(cfg["times_h"], list)
        and len(cfg["times_h"]) == 1
        and str(cfg["times_h"][0]).strip().lower() == "all"
    ):
        times_display = "all"
    else:
        times_display = ", ".join(str(t) for t in cfg["times_h"])

    print(f"Times (h)         : {times_display}")
    print(f"Pert.Type         : {cfg['pert_type'] or '(not specified)'}")
    print(f"Max sigs          : {cfg['max_sigs']}")
    print(f"Max ctrls/str     : {cfg['max_ctrls_per_str']}")
    print("\n=== EXPLANATION (biological rationale) ===")
    print(cfg["explanation"])

    write_userinput_txt(cfg, output_dir)
    write_explanation_txt(cfg["explanation"], output_dir)

    print("\n✅ Configuration files written:")
    print(f"  - {output_dir / 'userinput.txt'}")
    print(f"  - {output_dir / 'Explanation.txt'}")

    return cfg


if __name__ == "__main__":
    build_user_input(
        output_dir=Path("L1000_Output"),
        disease="Bladder Urothelial Carcinoma", 
        tissue = None,
        drug = None,
        time_points = None,
        cell_lines = None,
        max_sigs_override=DEFAULT_MAX_SIGS,
    )

