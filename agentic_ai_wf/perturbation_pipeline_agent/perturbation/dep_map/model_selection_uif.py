# src/model_selection_uif.py

from __future__ import annotations

import re
import difflib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict

import pandas as pd
from logging import Logger


# Try display() if notebook; otherwise fallback to print
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)




@dataclass
class ModelSelectionResult:
    mode: str
    criteria: Dict[str, object]
    models_table: pd.DataFrame
    selected_df: pd.DataFrame
    selected_ids: List[str]
    out_path: str


# ================================================================
# Helpers
# ================================================================
def normalize_list_from_input(s: str) -> List[str]:
    if not s:
        return []
    s = s.strip().strip("[](){}")
    parts = re.split(r"[,\n;]+", s)
    return [p.strip() for p in parts if p.strip()]


def fuzzy_resolve_one(query: str, options: List[str], cutoff: float = 0.55):
    """
    Fuzzy-resolve a single query string to the closest option.
    Returns (best_match, suggestion_list).
    """
    if not query:
        return None, []
    lower_map = {o.lower(): o for o in options}

    matches = difflib.get_close_matches(
        query.lower(),
        list(lower_map.keys()),
        n=10,
        cutoff=cutoff,
    )
    sugg = [lower_map[m] for m in matches]

    if matches:
        best = lower_map[matches[0]]
        return best, sugg

    # Substring fallback
    subs = [o for o in options if query.lower() in o.lower()]
    return (subs[0], subs) if subs else (None, [])


def fuzzy_resolve_list(
    queries: List[str],
    options: List[str],
    label: str = "choice",
) -> List[str]:
    resolved: List[str] = []
    for q in queries:
        best, sugg = fuzzy_resolve_one(q, options)
        if best and best not in resolved:
            resolved.append(best)
            print(f"✔ Resolved {label} '{q}' → '{best}'  (suggestions: {sugg})")
        elif not best:
            print(f"⚠ No match for {label} '{q}'. Skipping.")
    return resolved


def select_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    mode: Literal["any", "all"] = "any",
) -> pd.DataFrame:
    """
    Keyword search in OncotreeLineage + OncotreePrimaryDisease.
    Returns ALL matching rows (no truncation).
    """
    if not keywords:
        return df.iloc[0:0].copy()

    cols = ["OncotreeLineage", "OncotreePrimaryDisease"]
    work = df.copy()
    mask = None

    for kw in keywords:
        kw = kw.strip().lower()
        m_kw = False
        for c in cols:
            if c not in work.columns:
                continue
            m_col = work[c].fillna("").str.lower().str.contains(kw, na=False)
            m_kw = m_kw | m_col
        mask = (
            m_kw
            if mask is None
            else (mask | m_kw) if mode == "any"
            else (mask & m_kw)
        )

    if mask is None:
        return work.iloc[0:0].copy()
    return work[mask.fillna(False)].copy()


# ================================================================
# Building `models` from the master table
# ================================================================
def build_models_table_from_master(master: pd.DataFrame) -> pd.DataFrame:
    """
    Build a `models` table from the master cell-line table
    (with ModelID as index).
    """
    wanted_cols = [
        "ModelID",
        "CellLineName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
    ]
    df = master.reset_index()
    have = [c for c in wanted_cols if c in df.columns]
    if "ModelID" not in have:
        raise RuntimeError("Master table does not contain 'ModelID' column.")

    models = (
        df[have]
        .drop_duplicates(subset=["ModelID"])
        .reset_index(drop=True)
    )
    return models


# ================================================================
# Main selection API (non-interactive)
# ================================================================
def select_models(
    output_dir: Path,
    models: pd.DataFrame,
    mode: Literal["by_ids", "by_lineage", "by_disease", "keyword", "by_names"],
    ids: Optional[List[str]] = None,
    lineages: Optional[List[str]] = None,
    diseases: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    keyword_mode: Literal["any", "all"] = "any",
    names: Optional[List[str]] = None,
    logger: Optional[Logger] = None,
    save: bool = True,
) -> ModelSelectionResult:
    """
    Programmatic model selection.

    Parameters
    ----------
    models : DataFrame
        Table with at least ModelID; ideally also CellLineName, OncotreeLineage, OncotreePrimaryDisease.
    mode : {"by_ids","by_lineage","by_disease","keyword","by_names"}
    ids / lineages / diseases / keywords / names : selection criteria
    """
    MODEL_OUTDIR = output_dir / "DepMap_CellLines"
    MODEL_OUTDIR.mkdir(parents=True, exist_ok=True)

    mode = mode.lower()
    valid_modes = {"by_ids", "by_lineage", "by_disease", "keyword", "by_names"}
    if mode not in valid_modes:
        raise ValueError(f"Unrecognized mode '{mode}'. Valid: {sorted(valid_modes)}")

    selected_df = pd.DataFrame(columns=models.columns)

    # ---------------- by_ids ----------------
    if mode == "by_ids":
        if "ModelID" not in models.columns:
            raise RuntimeError("`models` must contain 'ModelID' for mode='by_ids'.")
        ids = ids or []
        selected_df = models[models["ModelID"].isin(ids)].copy()

    # ---------------- by_lineage ----------------
    elif mode == "by_lineage":
        if "OncotreeLineage" not in models.columns:
            raise RuntimeError("OncotreeLineage column not available in `models`.")
        lineages = lineages or []
        choices = sorted(models["OncotreeLineage"].dropna().unique().tolist())
        resolved = fuzzy_resolve_list(lineages, choices, label="lineage")
        selected_df = models[models["OncotreeLineage"].isin(resolved)].copy()

    # ---------------- by_disease ----------------
    elif mode == "by_disease":
        if "OncotreePrimaryDisease" not in models.columns:
            raise RuntimeError("OncotreePrimaryDisease column not available.")
        diseases = diseases or []
        choices = sorted(models["OncotreePrimaryDisease"].dropna().unique().tolist())
        resolved = fuzzy_resolve_list(diseases, choices, label="disease")
        selected_df = models[models["OncotreePrimaryDisease"].isin(resolved)].copy()

    # ---------------- keyword ----------------
    elif mode == "keyword":
        keywords = keywords or []
        if keyword_mode not in {"any", "all"}:
            print("⚠ Invalid keyword_mode; using 'any'.")
            keyword_mode = "any"
        selected_df = select_by_keywords(models, keywords, mode=keyword_mode)
        print(f"🔎 Keywords {keywords} with mode='{keyword_mode}' matched {len(selected_df)} models.")

    # ---------------- by_names ----------------
    elif mode == "by_names":
        if "CellLineName" not in models.columns:
            raise RuntimeError("CellLineName column not available in `models`.")
        names = names or []
        all_names = sorted(models["CellLineName"].dropna().unique().tolist())
        resolved = fuzzy_resolve_list(names, all_names, label="cell line")
        selected_df = models[models["CellLineName"].isin(resolved)].copy()

    # -----------------------------------------------------------
    # Final checks & save
    # -----------------------------------------------------------
    if selected_df.empty:
        raise RuntimeError("No models selected. Check your criteria.")

    selected_ids = selected_df["ModelID"].tolist()

    out_path = MODEL_OUTDIR / "Selected_Models.csv"
    if save:
        selected_df.to_csv(out_path, index=False)
        print(f"\n✅ Selected {len(selected_ids)} models (mode={mode}).")
        print(f"💾 Saved -> {out_path}")
        display(selected_df)

        if logger:
            logger.info(
                "Selected %d models (mode=%s). Saved to %s",
                len(selected_ids),
                mode,
                out_path,
            )

    criteria = {
        "mode": mode,
        "ids": ids,
        "lineages": lineages,
        "diseases": diseases,
        "keywords": keywords,
        "keyword_mode": keyword_mode,
        "names": names,
    }

    return ModelSelectionResult(
        mode=mode,
        criteria=criteria,
        models_table=models,
        selected_df=selected_df,
        selected_ids=selected_ids,
        out_path=str(out_path),
    )
