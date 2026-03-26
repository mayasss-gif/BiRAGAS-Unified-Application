#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust pathway comparison across up to 30 disease JSON files with enhanced error handling.

This version adds comprehensive error handling, fallbacks, and graceful degradation
to ensure the script completes even with malformed data or missing dependencies.

Outputs (when available):
- <prefix>.xlsx                     : Multi-sheet Excel workbook with all tables
- <prefix>.md                       : Markdown report with summaries
- <prefix>_heatmap.png              : ANY-presence heatmap
- <prefix>_heatmap_direction.png    : Direction heatmap (top variable pathways)
- <prefix>_pairwise_consolidated.csv
- <prefix>_pairwise_wide.csv
- <prefix>_similarity.xlsx          : NEW — all similarity sheets (ordered pairs, all entity types)
"""

import argparse
import csv
import json
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import xlsxwriter  
import openpyxl  

from collections import defaultdict

from similarity_comparison import compute_similarity_block


# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



# Non-interactive backend for headless environments
matplotlib.use('Agg')  

MATPLOTLIB_AVAILABLE = True

# ----------------------------- Logging -----------------------------

def setup_logging(verbosity: int) -> None:
    """Setup logging with error handling."""
    try:
        level = logging.WARNING
        if verbosity == 1:
            level = logging.INFO
        elif verbosity >= 2:
            level = logging.DEBUG
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True
        )
    except Exception as e:
        print(f"Warning: Failed to setup logging: {e}", file=sys.stderr)


# ----------------------------- Safe utils -----------------------------

def normalize_pathway(name: Any) -> str:
    """Normalize pathway names with error handling."""
    try:
        if name is None:
            return ""
        if not isinstance(name, str):
            name = str(name)
        s = re.sub(r"\s+", " ", name.strip())
        s = s.replace("–", "-").replace("—", "-")
        return s
    except Exception as e:
        logging.debug(f"normalize_pathway error for {name!r}: {e}")
        try:
            return str(name) if name is not None else ""
        except Exception:
            return ""


def cap_files(files: List[Path], cap: int = 30) -> List[Path]:
    """Cap file list with error handling."""
    try:
        files = sorted(list(files))
        cap = max(0, int(cap))
        return files[:cap]
    except Exception as e:
        logging.error(f"Error capping files: {e}")
        try:
            return list(files)[:30]  # Fallback to default cap
        except Exception:
            return []


def load_json(fp: Path) -> Optional[dict]:
    """Load JSON with comprehensive error handling."""
    try:
        if not fp.exists():
            logging.error(f"File not found: {fp}")
            return None
        if not fp.is_file():
            logging.error(f"Not a file: {fp}")
            return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {fp.name}: {e}")
    except UnicodeDecodeError as e:
        logging.error(f"Unicode decode error in {fp.name}: {e}")
        try:
            with open(fp, "r", encoding="latin-1") as f:
                return json.load(f)
        except Exception as e2:
            logging.error(f"Fallback encoding also failed for {fp.name}: {e2}")
    except PermissionError as e:
        logging.error(f"Permission denied reading {fp.name}: {e}")
    except OSError as e:
        logging.error(f"OS error reading {fp.name}: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error loading {fp.name}: {e}")
    return None


def load_map_csv(csv_path: Optional[Path]) -> Dict[str, str]:
    """Load CSV mapping with error handling."""
    mapping: Dict[str, str] = {}
    if not csv_path:
        return mapping

    try:
        if not csv_path.exists():
            logging.warning(f"Map CSV not found: {csv_path}")
            return mapping

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logging.warning("Map CSV has no header; expected columns: file,disease")
                return mapping

            cols = {c.lower(): c for c in reader.fieldnames}
            file_col = cols.get("file")
            dis_col = cols.get("disease")

            if not file_col or not dis_col:
                logging.warning("Map CSV missing required columns: file,disease")
                return mapping

            for row_num, row in enumerate(reader, start=2):
                try:
                    basename = (row.get(file_col) or "").strip()
                    disease = (row.get(dis_col) or "").strip()
                    if basename:
                        mapping[basename] = disease
                except Exception as e:
                    logging.debug(f"Error processing CSV row {row_num}: {e}")

    except UnicodeDecodeError as e:
        logging.error(f"Unicode error reading map CSV {csv_path}: {e}")
    except Exception as e:
        logging.error(f"Failed to read map CSV {csv_path}: {e}")

    return mapping


def sanitize_label(s: Any) -> str:
    """Sanitize label strings with error handling."""
    try:
        if s is None:
            return ""
        s = str(s).strip()
        return re.sub(r"\s+", " ", s)
    except Exception as e:
        logging.debug(f"sanitize_label error: {e}")
        try:
            return str(s) if s is not None else ""
        except Exception:
            return ""


def resolve_disease_label(
    fp: Path,
    data: Optional[dict],
    label_source: str,
    label_key: str,
    filename_regex: Optional[str],
    map_csv: Dict[str, str],
) -> str:
    """Resolve disease label with fallback chain."""
    try:
        base = fp.name

        # 1) CSV map (basename) - highest priority
        if base in map_csv:
            label = sanitize_label(map_csv[base])
            if label:
                return label

        # 2) JSON metadata
        if label_source == "metadata" and isinstance(data, dict):
            try:
                md = data.get("metadata")
                if isinstance(md, dict):
                    val = md.get(label_key)
                    if isinstance(val, str) and val.strip():
                        return sanitize_label(val)
                val2 = data.get(label_key)
                if isinstance(val2, str) and val2.strip():
                    return sanitize_label(val2)
            except Exception as e:
                logging.debug(f"Metadata label error in {fp.name}: {e}")

        # 3) Filename regex
        if filename_regex:
            try:
                m = re.search(filename_regex, fp.name)
                if m and "disease" in m.groupdict():
                    disease = m.group("disease")
                    if disease and disease.strip():
                        return sanitize_label(disease)
            except re.error as e:
                logging.warning(f"Bad filename-regex '{filename_regex}': {e}")
            except Exception as e:
                logging.debug(f"Regex match error for {fp.name}: {e}")

        # 4) Fallback to stem
        return sanitize_label(fp.stem)

    except Exception as e:
        logging.error(f"resolve_disease_label failed for {fp.name}: {e}")
        try:
            return sanitize_label(fp.stem)
        except Exception:
            return "unknown"


def _safe_float(x: Any) -> Optional[float]:
    """Safely convert to float."""
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    """Safely convert to int."""
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        return int(x)
    except (ValueError, TypeError):
        return None


def _fmt_overlap_genes(genes: Any) -> Optional[str]:
    """Format overlap genes safely."""
    try:
        if isinstance(genes, list):
            return ", ".join([str(g) for g in genes if g is not None])
        return genes if isinstance(genes, str) else None
    except Exception:
        return None


def _coerce_entity_record(obj: Any) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[int], Optional[int], Optional[int], Optional[int], Optional[str]]:
    """Extract entity record fields with error handling."""
    try:
        if isinstance(obj, dict):
            ent = obj.get("entity")
            ent = "" if ent is None else str(ent)
            return (
                ent,
                _safe_float(obj.get("OR")),
                _safe_float(obj.get("pval")),
                _safe_float(obj.get("qval")),
                _safe_int(obj.get("k")),
                _safe_int(obj.get("a")),
                _safe_int(obj.get("b")),
                _safe_int(obj.get("N")),
                _fmt_overlap_genes(obj.get("overlap_genes")),
            )
        # fallback string
        return (str(obj), None, None, None, None, None, None, None, None)
    except Exception as e:
        logging.debug(f"_coerce_entity_record error: {e}")
        return ("", None, None, None, None, None, None, None, None)


# ----------------------------- Parsing -----------------------------

def collect_from_section(section: Any, disease: str, pathway: str, direction: str, entity_type: str, rows: list) -> None:
    """Collect entity rows with error handling."""
    if not isinstance(section, list):
        return

    for idx, obj in enumerate(section):
        try:
            ent, or_v, pval, qval, k, a, b, N, overlap = _coerce_entity_record(obj)
            jaccard = None
            if obj is not None and isinstance(obj, dict):
                jaccard = _safe_float(obj.get("Jaccard"))

            rows.append({
                "disease": disease,
                "pathway": pathway,
                "direction": direction,
                "entity_type": entity_type,
                "entity": ent,
                "OR": or_v,
                "pval": pval,
                "qval": qval,
                "Jaccard": jaccard,
                "k": k, "a": a, "b": b, "N": N,
                "overlap_genes": overlap
            })
        except Exception as e:
            logging.debug(f"collect_from_section skip item {idx} in {disease}/{pathway}/{entity_type}: {e}")


def parse_one_file(fp: Path, disease_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a single JSON file into presence and entity tables with comprehensive error handling."""
    empty_presence = pd.DataFrame(columns=["disease", "pathway", "UP", "DOWN"])
    empty_entities = pd.DataFrame()

    try:
        raw = load_json(fp)
        if raw is None:
            logging.warning(f"Skipping unreadable JSON: {fp.name}")
            return empty_presence, empty_entities

        if not isinstance(raw, dict):
            logging.warning(f"Top-level JSON not a dict in {fp.name}; skipping.")
            return empty_presence, empty_entities

        entity_rows: List[dict] = []
        presence_rows: List[dict] = []

        for pw, payload in raw.items():
            try:
                if not isinstance(pw, str):
                    continue
                if pw.lower() == "metadata":
                    continue

                pw_norm = normalize_pathway(pw)
                if not pw_norm:
                    continue

                # Detect schema (directional or non-directional)
                is_directional = isinstance(payload, dict) and (("UP" in payload) or ("DOWN" in payload))

                if is_directional:
                    for direction in ("UP", "DOWN"):
                        try:
                            section = payload.get(direction, {}) if isinstance(payload, dict) else {}
                            if not isinstance(section, dict):
                                section = {}

                            present = any(
                                isinstance(section.get(k, []), list) and len(section.get(k, [])) > 0
                                for k in ("metabolites", "epigenetic", "tf")
                            )

                            presence_rows.append({
                                "disease": disease_name,
                                "pathway": pw_norm,
                                "UP": int(direction == "UP" and present),
                                "DOWN": int(direction == "DOWN" and present),
                            })

                            for etype in ("metabolites", "epigenetic", "tf"):
                                collect_from_section(
                                    section.get(etype, []),
                                    disease_name, pw_norm, direction, etype, entity_rows
                                )
                        except Exception as e:
                            logging.debug(f"Directional parse error {fp.name}:{pw_norm}:{direction}: {e}")
                            presence_rows.append({"disease": disease_name, "pathway": pw_norm, "UP": 0, "DOWN": 0})
                else:
                    try:
                        section = payload if isinstance(payload, dict) else {}
                        present = any(
                            isinstance(section.get(k, []), list) and len(section.get(k, [])) > 0
                            for k in ("metabolites", "epigenetic", "tf")
                        )

                        presence_rows.append({
                            "disease": disease_name,
                            "pathway": pw_norm,
                            "UP": int(present),
                            "DOWN": 0,
                        })

                        for etype in ("metabolites", "epigenetic", "tf"):
                            collect_from_section(
                                section.get(etype, []),
                                disease_name, pw_norm, "NA", etype, entity_rows
                            )
                    except Exception as e:
                        logging.debug(f"Non-directional parse error {fp.name}:{pw_norm}: {e}")
                        presence_rows.append({"disease": disease_name, "pathway": pw_norm, "UP": 0, "DOWN": 0})

            except Exception as e:
                logging.debug(f"Pathway parse error {fp.name}:{pw}: {e}")

        presence_df = pd.DataFrame(presence_rows)
        if not presence_df.empty:
            try:
                presence_df = presence_df.groupby(
                    ["disease", "pathway"],
                    as_index=False
                )[["UP", "DOWN"]].max()
            except Exception as e:
                logging.debug(f"Presence groupby error in {fp.name}: {e}")

        entities_df = pd.DataFrame(entity_rows).drop_duplicates() if entity_rows else pd.DataFrame()
        return presence_df, entities_df

    except Exception as e:
        logging.error(f"Fatal error parsing {fp.name}: {e}")
        return empty_presence, empty_entities


# ----------------------------- Analytics -----------------------------

def compute_presence_matrices(presence_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute presence matrices with error handling."""
    empty = pd.DataFrame()

    try:
        if presence_df is None or presence_df.empty:
            return (empty, empty, empty)

        diseases = sorted(presence_df["disease"].dropna().unique().tolist())
        pathways = sorted(presence_df["pathway"].dropna().unique().tolist())

        if not diseases or not pathways:
            logging.warning("No diseases or pathways found in presence data")
            return (empty, empty, empty)

        mat_up = presence_df.pivot_table(
            index="pathway", columns="disease",
            values="UP", aggfunc="max", fill_value=0
        ).reindex(index=pathways, columns=diseases, fill_value=0)

        mat_down = presence_df.pivot_table(
            index="pathway", columns="disease",
            values="DOWN", aggfunc="max", fill_value=0
        ).reindex(index=pathways, columns=diseases, fill_value=0)

        mat_any = ((mat_up.values + mat_down.values) > 0).astype(int)
        mat_any = pd.DataFrame(mat_any, index=mat_up.index, columns=mat_up.columns)

        return mat_up, mat_down, mat_any

    except Exception as e:
        logging.error(f"compute_presence_matrices failed: {e}")
        return (empty, empty, empty)


def summarize_common_unique(mat_any: pd.DataFrame):
    """Summarize common and unique pathways with error handling."""
    try:
        if mat_any is None or mat_any.empty:
            return (pd.Series(dtype=int), pd.Series(dtype=int), {})

        presence_counts = mat_any.sum(axis=1)
        common = presence_counts[presence_counts >= 2].sort_values(ascending=False)

        uniques = {}
        for disease in mat_any.columns:
            try:
                only_here = (mat_any[disease] == 1) & (mat_any.drop(columns=[disease]).sum(axis=1) == 0)
                uniques[disease] = list(mat_any.index[only_here])
            except Exception as e:
                logging.debug(f"Error finding uniques for {disease}: {e}")
                uniques[disease] = []

        return presence_counts, common, uniques

    except Exception as e:
        logging.error(f"summarize_common_unique failed: {e}")
        return (pd.Series(dtype=int), pd.Series(dtype=int), {})


def regulation_agreement(mat_up: pd.DataFrame, mat_down: pd.DataFrame) -> pd.DataFrame:
    """Analyze regulation agreement with error handling."""
    empty = pd.DataFrame(columns=[
        "pathway", "present_in", "up_in", "down_in",
        "present_diseases", "up_diseases", "down_diseases", "direction_status"
    ])

    try:
        if mat_up is None or mat_up.empty:
            return empty

        diseases = mat_up.columns.tolist()
        rows = []

        for pw in mat_up.index:
            try:
                ups = mat_up.loc[pw]
                downs = mat_down.loc[pw] if (mat_down is not None and not mat_down.empty) else pd.Series(0, index=diseases)

                present = (ups.values + downs.values) > 0
                present_diseases = [d for d, p in zip(diseases, present) if p]
                up_diseases = [d for d in diseases if ups.get(d, 0) == 1]
                down_diseases = [d for d in diseases if downs.get(d, 0) == 1]

                if len(up_diseases) > 0 and len(down_diseases) > 0:
                    status = "mixed"
                elif len(up_diseases) > 0:
                    status = "UP"
                elif len(down_diseases) > 0:
                    status = "DOWN"
                else:
                    status = "absent"

                rows.append({
                    "pathway": pw,
                    "present_in": len(present_diseases),
                    "up_in": len(up_diseases),
                    "down_in": len(down_diseases),
                    "present_diseases": ", ".join(present_diseases),
                    "up_diseases": ", ".join(up_diseases),
                    "down_diseases": ", ".join(down_diseases),
                    "direction_status": status
                })
            except Exception as e:
                logging.debug(f"Error processing pathway {pw} in regulation_agreement: {e}")

        if not rows:
            return empty

        return pd.DataFrame(rows).sort_values(
            ["present_in", "direction_status", "up_in", "down_in", "pathway"],
            ascending=[False, True, False, False, True]
        )

    except Exception as e:
        logging.error(f"regulation_agreement failed: {e}")
        return empty


def aggregate_entities(entities_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate entity data with error handling."""
    empty = pd.DataFrame()

    try:
        if entities_df is None or entities_df.empty:
            return empty, empty, empty

        # Per disease aggregation
        try:
            per_disease = entities_df.groupby(
                ["disease", "entity_type"],
                dropna=False
            ).agg(
                n_entities=("entity", "nunique"),
                n_rows=("entity", "size"),
                avg_OR=("OR", "mean"),
                min_q=("qval", "min")
            ).reset_index()
        except Exception as e:
            logging.debug(f"aggregate_entities per_disease error: {e}")
            per_disease = empty

        # Per pathway-type aggregation
        try:
            per_pathway_type = entities_df.groupby(
                ["pathway", "entity_type"],
                dropna=False
            ).agg(
                diseases=("disease", lambda s: ", ".join(sorted(set([d for d in s if isinstance(d, str) and d])))),
                n_diseases=("disease", "nunique"),
                n_entities=("entity", "nunique")
            ).reset_index()
        except Exception as e:
            logging.debug(f"aggregate_entities per_pathway_type error: {e}")
            per_pathway_type = empty

        long_entities = entities_df.copy()
        return per_disease, per_pathway_type, long_entities

    except Exception as e:
        logging.error(f"aggregate_entities failed: {e}")
        return empty, empty, empty


def write_excel(outputs: Dict[str, pd.DataFrame], outpath: Path) -> None:
    """Write Excel file with error handling and fallbacks."""
    try:
        # Try xlsxwriter first
        engine = "xlsxwriter"
        
        csv_dir = outpath.parent / f"{outpath.stem}_csv"
        csv_dir.mkdir(exist_ok=True)
        for name, df in outputs.items():
            if df is not None and not df.empty:
                csv_path = csv_dir / f"{name}.csv"
                df.to_csv(csv_path, index=False)
        logging.info(f"Saved data as CSV files in {csv_dir}")

        with pd.ExcelWriter(outpath, engine=engine) as writer:
            for name, df in outputs.items():
                try:
                    if df is None or df.empty:
                        # Write empty sheet for visibility
                        pd.DataFrame().to_excel(writer, sheet_name=name[:31], index=False)
                    else:
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                except Exception as e:
                    logging.warning(f"Failed to write sheet '{name}': {e}")

        logging.info(f"Excel file written: {outpath}")

    except PermissionError as e:
        logging.error(f"Permission denied writing {outpath}: {e}")
    except Exception as e:
        logging.error(f"Excel write failed for {outpath.name}: {e}")


def make_heatmap(mat_any: pd.DataFrame, out_png: Path) -> None:
    """Generate ANY-presence heatmap with error handling."""
    if not MATPLOTLIB_AVAILABLE:
        logging.info("Matplotlib not available, skipping heatmap")
        return

    if mat_any is None or mat_any.empty:
        logging.info("No data for heatmap")
        return

    try:
        fig_width = max(7.5, len(mat_any.columns) * 0.7)
        fig_height = max(7.5, len(mat_any.index) * 0.28)

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(mat_any.values, aspect='auto', interpolation='nearest', cmap='YlOrRd')
        plt.yticks(range(len(mat_any.index)), mat_any.index, fontsize=8)
        plt.xticks(range(len(mat_any.columns)), mat_any.columns, rotation=90, fontsize=8)
        plt.title("Pathway Presence (ANY of UP/DOWN)")
        plt.xlabel("Disease")
        plt.ylabel("Pathway")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        logging.info(f"Heatmap saved: {out_png}")

    except Exception as e:
        logging.warning(f"Heatmap generation failed: {e}")
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


def build_direction_matrix(presence_df: pd.DataFrame) -> pd.DataFrame:
    """Build direction matrix (+1=UP, 0=absent or mixed, -1=DOWN) with error handling."""
    try:
        if presence_df is None or presence_df.empty:
            return pd.DataFrame()

        up = presence_df.pivot_table(
            index="pathway", columns="disease",
            values="UP", aggfunc="max", fill_value=0
        )
        dn = presence_df.pivot_table(
            index="pathway", columns="disease",
            values="DOWN", aggfunc="max", fill_value=0
        )

        mat = up.copy().astype(int)
        mat = mat * 0 + up.values * 1 + dn.values * (-1)

        # if both UP and DOWN are 1 (shouldn't happen for grouped max), force 0
        both = (up.values == 1) & (dn.values == 1)
        mat[both] = 0

        mat = pd.DataFrame(mat, index=up.index, columns=up.columns).astype(int)
        return mat

    except Exception as e:
        logging.warning(f"Direction matrix failed: {e}")
        return pd.DataFrame()


def build_entity_sets(entities_df: pd.DataFrame) -> dict:
    """Build nested dict pathway->disease->entity_type->set(entity) with error handling."""
    
    ents = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    if entities_df is None or entities_df.empty:
        return ents

    try:
        for _, r in entities_df.iterrows():
            try:
                p = r.get("pathway")
                d = r.get("disease")
                et = r.get("entity_type")
                e = r.get("entity")

                if isinstance(p, str) and isinstance(d, str) and isinstance(et, str) and isinstance(e, str):
                    ents[p][d][et].add(e)
            except Exception as e:
                logging.debug(f"Error adding entity to set: {e}")

    except Exception as e:
        logging.debug(f"build_entity_sets error: {e}")

    return ents


def jaccard(a: set, b: set) -> float:
    """Calculate Jaccard similarity with error handling."""
    try:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / uni if uni else 0.0
    except Exception as e:
        logging.debug(f"jaccard calculation error: {e}")
        return 0.0


def compute_pairwise_diffs(direction_mat: pd.DataFrame, entity_sets: dict, diseases: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute pairwise differences with error handling."""
    cols = ["pathway", "disease_a", "disease_b", "dir_contrast", "diff_metabolites", "diff_epigenetic", "diff_tf"]

    try:
        if direction_mat is None or direction_mat.empty:
            return pd.DataFrame(columns=cols)

        diseases = diseases or list(direction_mat.columns)
        rows = []
        etypes = ["metabolites", "epigenetic", "tf"]

        for pw in direction_mat.index:
            for i in range(len(diseases)):
                for j in range(i + 1, len(diseases)):
                    try:
                        da, db = diseases[i], diseases[j]
                        va, vb = int(direction_mat.loc[pw, da]), int(direction_mat.loc[pw, db])

                        if va == 0 and vb == 0:
                            d_contrast = 0.0
                        elif va * vb == -1:
                            d_contrast = 1.0
                        elif (va == 0) ^ (vb == 0):
                            d_contrast = 0.5
                        else:
                            d_contrast = 0.0

                        diffs = {}
                        for et in etypes:
                            set_a = entity_sets.get(pw, {}).get(da, {}).get(et, set())
                            set_b = entity_sets.get(pw, {}).get(db, {}).get(et, set())
                            js = jaccard(set_a, set_b)
                            diffs[et] = 1.0 - js

                        rows.append({
                            "pathway": pw,
                            "disease_a": da,
                            "disease_b": db,
                            "dir_contrast": d_contrast,
                            "diff_metabolites": diffs["metabolites"],
                            "diff_epigenetic": diffs["epigenetic"],
                            "diff_tf": diffs["tf"],
                        })
                    except Exception as e:
                        logging.debug(f"pairwise diffs error {pw}/{da}/{db}: {e}")

        return pd.DataFrame(rows)

    except Exception as e:
        logging.error(f"compute_pairwise_diffs failed: {e}")
        return pd.DataFrame(columns=cols)


def summarize_pathway_differentiation(pairwise_df: pd.DataFrame, reg_df: pd.DataFrame,
                                      w_dir: float = 0.6, w_ent: float = 0.4) -> pd.DataFrame:
    """Summarize pathway differentiation scores with error handling."""
    cols = ["pathway", "PDS", "mean_dir_contrast", "mean_ent_diff", "present_in", "mixed_flag"]

    try:
        if pairwise_df is None or pairwise_df.empty:
            return pd.DataFrame(columns=cols)

        g = pairwise_df.groupby("pathway", as_index=False).agg(
            mean_dir_contrast=("dir_contrast", "mean"),
            mean_diff_metabolites=("diff_metabolites", "mean"),
            mean_diff_epigenetic=("diff_epigenetic", "mean"),
            mean_diff_tf=("diff_tf", "mean"),
        )

        g["mean_ent_diff"] = g[["mean_diff_metabolites", "mean_diff_epigenetic", "mean_diff_tf"]].mean(axis=1)
        g["PDS"] = w_dir * g["mean_dir_contrast"] + w_ent * g["mean_ent_diff"]

        if reg_df is not None and not reg_df.empty:
            try:
                reg_small = reg_df[["pathway", "present_in", "direction_status"]].copy()
                reg_small["mixed_flag"] = (reg_small["direction_status"] == "mixed").astype(int)
                g = g.merge(reg_small[["pathway", "present_in", "mixed_flag"]], on="pathway", how="left")
            except Exception as e:
                logging.debug(f"Failed to merge regulation data: {e}")
                g["present_in"] = np.nan
                g["mixed_flag"] = 0
        else:
            g["present_in"] = np.nan
            g["mixed_flag"] = 0

        g = g.sort_values(["PDS", "present_in"], ascending=[False, False]).reset_index(drop=True)
        return g

    except Exception as e:
        logging.warning(f"PDS summarization failed: {e}")
        return pd.DataFrame(columns=cols)


def _entity_presence_tables(entities_df: pd.DataFrame, mat_any: pd.DataFrame) -> dict:
    """Create entity presence tables with error handling."""
    out = {"metabolites": pd.DataFrame(), "epigenetic": pd.DataFrame(), "tf": pd.DataFrame()}

    try:
        if entities_df is None or entities_df.empty or mat_any is None or mat_any.empty:
            return out

        diseases_all = list(mat_any.columns)

        for et in ("metabolites", "epigenetic", "tf"):
            try:
                sub = entities_df[entities_df["entity_type"] == et].copy()
                if sub.empty:
                    out[et] = pd.DataFrame()
                    continue

                sub["pw_entity"] = list(zip(sub["pathway"], sub["entity"]))
                pairs = sorted(set(sub["pw_entity"]))
                idx = pd.MultiIndex.from_tuples(pairs, names=["pathway", "entity"])
                mat = pd.DataFrame(0, index=idx, columns=diseases_all, dtype=int)

                for (pw, ent), group in sub.groupby(["pathway", "entity"]):
                    try:
                        ds = [d for d in group["disease"].unique().tolist()
                              if isinstance(d, str) and d in mat.columns]
                        if ds:
                            mat.loc[(pw, ent), ds] = 1
                    except Exception as e:
                        logging.debug(f"Error setting entity presence for {pw}/{ent}: {e}")

                out[et] = mat

            except Exception as e:
                logging.debug(f"_entity_presence_tables error {et}: {e}")
                out[et] = pd.DataFrame()

        return out

    except Exception as e:
        logging.error(f"_entity_presence_tables failed: {e}")
        return out


def compute_entity_differences(entities_df: pd.DataFrame, mat_any: pd.DataFrame):
    """Compute entity-level differences with comprehensive error handling."""
    empty_df = pd.DataFrame()
    empty_mats = {"metabolites": pd.DataFrame(), "epigenetic": pd.DataFrame(), "tf": pd.DataFrame()}

    try:
        result_unique, result_subset, result_core = [], [], []

        if mat_any is None or mat_any.empty:
            return empty_df, empty_df, empty_df, empty_mats

        presence_counts = mat_any.sum(axis=1)
        common_pws = presence_counts[presence_counts >= 2].index.tolist()
        mats = _entity_presence_tables(entities_df, mat_any)

        if entities_df is None or entities_df.empty:
            ent_stats = pd.DataFrame(columns=[
                "disease", "pathway", "entity_type", "entity", "OR", "pval", "qval", "Jaccard", "overlap_genes"
            ])
        else:
            ent_stats = entities_df[[
                "disease", "pathway", "entity_type", "entity", "OR", "pval", "qval", "Jaccard", "overlap_genes"
            ]].drop_duplicates()

        for etype, mat in mats.items():
            if mat is None or mat.empty:
                continue

            for pw in common_pws:
                try:
                    if pw not in mat.index.get_level_values(0):
                        continue

                    active_diseases = mat_any.columns[mat_any.loc[pw] == 1].tolist()
                    if len(active_diseases) < 2:
                        continue

                    pw_rows = mat.loc[pw]
                    if isinstance(pw_rows, pd.Series):
                        pw_rows = pw_rows.to_frame().T
                        pw_rows.index = pd.Index([mat.loc[pw].name], name="entity")
                    else:
                        pw_rows.index.name = "entity"

                    sub = pw_rows[active_diseases].copy()
                    n_present = sub.sum(axis=1).astype(int)
                    n_active = len(active_diseases)

                    # CORE
                    core_mask = (n_present == n_active)
                    for ent in sub.index[core_mask]:
                        result_core.append({
                            "pathway": pw,
                            "entity_type": etype,
                            "entity": ent,
                            "n_active_diseases": n_active
                        })

                    # UNIQUE
                    uniq_mask = (n_present == 1)
                    for ent in sub.index[uniq_mask]:
                        try:
                            disease = sub.columns[sub.loc[ent] == 1][0]
                            stats_row = ent_stats[
                                (ent_stats["disease"] == disease) &
                                (ent_stats["pathway"] == pw) &
                                (ent_stats["entity_type"] == etype) &
                                (ent_stats["entity"] == ent)
                            ]
                            row = {
                                "pathway": pw,
                                "entity_type": etype,
                                "disease": disease,
                                "entity": ent,
                                "OR": None, "pval": None, "qval": None,
                                "Jaccard": None, "overlap_genes": None
                            }
                            if not stats_row.empty:
                                s = stats_row.iloc[0]
                                row.update({
                                    "OR": s.get("OR", None),
                                    "pval": s.get("pval", None),
                                    "qval": s.get("qval", None),
                                    "Jaccard": s.get("Jaccard", None),
                                    "overlap_genes": s.get("overlap_genes", None),
                                })
                            result_unique.append(row)
                        except Exception as e:
                            logging.debug(f"Error processing unique entity {ent}: {e}")

                    # SUBSET-SHARED
                    subset_mask = (n_present >= 2) & (n_present < n_active)
                    for ent in sub.index[subset_mask]:
                        try:
                            ds = sub.columns[sub.loc[ent] == 1].tolist()
                            result_subset.append({
                                "pathway": pw,
                                "entity_type": etype,
                                "entity": ent,
                                "present_in_diseases": ", ".join(ds),
                                "n_present": int(n_present.loc[ent]),
                                "n_active_diseases": n_active
                            })
                        except Exception as e:
                            logging.debug(f"Error processing subset entity {ent}: {e}")

                except Exception as e:
                    logging.debug(f"Error processing pathway {pw} for entity differences: {e}")

        def _sort_df(df: pd.DataFrame, cols: list, ascending: list):
            try:
                if df.empty:
                    return df
                return df.sort_values(cols, ascending=ascending).reset_index(drop=True)
            except Exception as e:
                logging.debug(f"Sort error: {e}")
                return df.reset_index(drop=True) if not df.empty else df

        unique_df = pd.DataFrame(result_unique)
        subset_df = pd.DataFrame(result_subset)
        core_df = pd.DataFrame(result_core)

        unique_df = _sort_df(unique_df, ["pathway", "entity_type", "disease", "entity"], [True, True, True, True])
        subset_df = _sort_df(subset_df, ["pathway", "entity_type", "n_present", "entity"], [True, True, False, True])
        core_df = _sort_df(core_df, ["pathway", "entity_type", "entity"], [True, True, True])

        return unique_df, subset_df, core_df, mats

    except Exception as e:
        logging.error(f"compute_entity_differences failed: {e}")
        return empty_df, empty_df, empty_df, empty_mats


# ----------------------------- Pairwise & Wide -----------------------------

def _dir_label(v: int) -> str:
    """Convert direction value to label."""
    try:
        vi = int(v)
        return {1: "UP", -1: "DOWN", 0: "ABSENT"}.get(vi, "ABSENT")
    except Exception:
        return "ABSENT"


def _sorted_join(items: set, max_show: int = 50) -> str:
    """Join set items as sorted string."""
    try:
        if not items:
            return ""
        arr = sorted([str(x) for x in items])
        if len(arr) > max_show:
            return "; ".join(arr[:max_show]) + f"; (+{len(arr) - max_show} more)"
        return "; ".join(arr)
    except Exception as e:
        logging.debug(f"_sorted_join error: {e}")
        return ""


def build_pairwise_consolidated(direction_mat: pd.DataFrame, entity_sets: dict, mat_any: pd.DataFrame) -> pd.DataFrame:
    """Build consolidated pairwise comparison with error handling."""
    cols = [
        "pathway", "disease_A", "disease_B", "dir_A", "dir_B",
        "metabolites_A_only", "metabolites_B_only", "metabolites_A_only_n", "metabolites_B_only_n",
        "epigenetic_A_only", "epigenetic_B_only", "epigenetic_A_only_n", "epigenetic_B_only_n",
        "tf_A_only", "tf_B_only", "tf_A_only_n", "tf_B_only_n",
        "present_in"
    ]

    try:
        if direction_mat is None or direction_mat.empty:
            return pd.DataFrame(columns=cols)

        diseases = list(direction_mat.columns)
        present_in_series = (mat_any.sum(axis=1) if (mat_any is not None and not mat_any.empty)
                             else pd.Series(0, index=direction_mat.index))
        present_in = present_in_series.reindex(direction_mat.index).fillna(0).astype(int)

        rows = []
        etypes = ["metabolites", "epigenetic", "tf"]

        for pw in direction_mat.index:
            for i in range(len(diseases)):
                for j in range(len(diseases)):
                    if i == j:
                        continue

                    try:
                        da, db = diseases[i], diseases[j]
                        dir_a = _dir_label(direction_mat.loc[pw, da])
                        dir_b = _dir_label(direction_mat.loc[pw, db])

                        sets_a = entity_sets.get(pw, {}).get(da, {})
                        sets_b = entity_sets.get(pw, {}).get(db, {})

                        diffs = {}
                        for et in etypes:
                            set_a = sets_a.get(et, set()) if isinstance(sets_a.get(et, set()), set) else set()
                            set_b = sets_b.get(et, set()) if isinstance(sets_b.get(et, set()), set) else set()
                            a_only = set_a - set_b
                            b_only = set_b - set_a
                            diffs[f"{et}_a_only"] = a_only
                            diffs[f"{et}_b_only"] = b_only

                        rows.append({
                            "pathway": pw,
                            "disease_A": da,
                            "disease_B": db,
                            "dir_A": dir_a,
                            "dir_B": dir_b,
                            "metabolites_A_only": _sorted_join(diffs["metabolites_a_only"]),
                            "metabolites_B_only": _sorted_join(diffs["metabolites_b_only"]),
                            "metabolites_A_only_n": len(diffs["metabolites_a_only"]),
                            "metabolites_B_only_n": len(diffs["metabolites_b_only"]),
                            "epigenetic_A_only": _sorted_join(diffs["epigenetic_a_only"]),
                            "epigenetic_B_only": _sorted_join(diffs["epigenetic_b_only"]),
                            "epigenetic_A_only_n": len(diffs["epigenetic_a_only"]),
                            "epigenetic_B_only_n": len(diffs["epigenetic_b_only"]),
                            "tf_A_only": _sorted_join(diffs["tf_a_only"]),
                            "tf_B_only": _sorted_join(diffs["tf_b_only"]),
                            "tf_A_only_n": len(diffs["tf_a_only"]),
                            "tf_B_only_n": len(diffs["tf_b_only"]),
                            "present_in": int(present_in.loc[pw]) if pw in present_in.index else 0,
                        })
                    except Exception as e:
                        logging.debug(f"pairwise consolidated error {pw}/{da}/{db}: {e}")

        out = pd.DataFrame(rows)
        if not out.empty:
            try:
                out["_diff_score"] = (
                    out["metabolites_A_only_n"] + out["metabolites_B_only_n"] +
                    out["epigenetic_A_only_n"] + out["epigenetic_B_only_n"] +
                    out["tf_A_only_n"] + out["tf_B_only_n"]
                )
                out = out.sort_values(
                    ["present_in", "_diff_score", "pathway", "disease_A", "disease_B"],
                    ascending=[False, False, True, True, True]
                ).drop(columns=["_diff_score"])
            except Exception as e:
                logging.debug(f"Sorting pairwise consolidated failed: {e}")

        return out

    except Exception as e:
        logging.error(f"build_pairwise_consolidated failed: {e}")
        return pd.DataFrame(columns=cols)


def _safe_name(s: str) -> str:
    """Sanitize name for column headers."""
    try:
        s = re.sub(r"[^\w\-\.\s]+", "_", str(s))
        s = re.sub(r"\s+", "_", s.strip())
        return s
    except Exception:
        return "unknown"


def build_pairwise_wide(pairwise_consolidated: pd.DataFrame) -> pd.DataFrame:
    """Build wide pairwise format with error handling."""
    try:
        if pairwise_consolidated is None or pairwise_consolidated.empty:
            return pd.DataFrame()

        frames = []
        for (da, db), sub in pairwise_consolidated.groupby(["disease_A", "disease_B"], sort=True):
            try:
                pair_tag = f"{_safe_name(da)}_vs_{_safe_name(db)}"
                sub2 = sub[[
                    "pathway", "dir_A", "dir_B",
                    "metabolites_A_only", "metabolites_B_only", "metabolites_A_only_n", "metabolites_B_only_n",
                    "epigenetic_A_only", "epigenetic_B_only", "epigenetic_A_only_n", "epigenetic_B_only_n",
                    "tf_A_only", "tf_B_only", "tf_A_only_n", "tf_B_only_n"
                ]].copy()

                ren = {
                    "dir_A": f"{pair_tag}__dir_{_safe_name(da)}",
                    "dir_B": f"{pair_tag}__dir_{_safe_name(db)}",
                    "metabolites_A_only": f"{pair_tag}__metabolites_{_safe_name(da)}_only",
                    "metabolites_B_only": f"{pair_tag}__metabolites_{_safe_name(db)}_only",
                    "metabolites_A_only_n": f"{pair_tag}__metabolites_{_safe_name(da)}_only_n",
                    "metabolites_B_only_n": f"{pair_tag}__metabolites_{_safe_name(db)}_only_n",
                    "epigenetic_A_only": f"{pair_tag}__epigenetic_{_safe_name(da)}_only",
                    "epigenetic_B_only": f"{pair_tag}__epigenetic_{_safe_name(db)}_only",
                    "epigenetic_A_only_n": f"{pair_tag}__epigenetic_{_safe_name(da)}_only_n",
                    "epigenetic_B_only_n": f"{pair_tag}__epigenetic_{_safe_name(db)}_only_n",
                    "tf_A_only": f"{pair_tag}__tf_{_safe_name(da)}_only",
                    "tf_B_only": f"{pair_tag}__tf_{_safe_name(db)}_only",
                    "tf_A_only_n": f"{pair_tag}__tf_{_safe_name(da)}_only_n",
                    "tf_B_only_n": f"{pair_tag}__tf_{_safe_name(db)}_only_n",
                }
                sub2 = sub2.rename(columns=ren)
                frames.append(sub2)
            except Exception as e:
                logging.debug(f"build_pairwise_wide group error for {da}/{db}: {e}")

        if not frames:
            return pd.DataFrame()

        frames_sorted = sorted(frames, key=lambda df: list(df.columns)[1] if len(df.columns) > 1 else "")
        wide = None
        for df in frames_sorted:
            try:
                wide = df if wide is None else wide.merge(df, on="pathway", how="outer")
            except Exception as e:
                logging.debug(f"build_pairwise_wide merge error: {e}")
                continue

        return wide if wide is not None else pd.DataFrame()

    except Exception as e:
        logging.error(f"build_pairwise_wide failed: {e}")
        return pd.DataFrame()


# ----------------------------- Markdown Report -----------------------------

def df_to_markdown(df, index=True):
    """Convert DataFrame to markdown with error handling."""
    try:
        if df is None or df.empty:
            return "_(empty table)_"

        working = df.copy()
        if index:
            working.insert(0, df.index.name or "", df.index)

        headers = [str(h).replace("|", r"\|") for h in list(working.columns)]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = []
        for _, row in working.iterrows():
            cells = [("" if pd.isna(v) else str(v)).replace("|", r"\|") for v in row.tolist()]
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header_line, sep_line] + rows)

    except Exception as e:
        logging.debug(f"df_to_markdown failed: {e}")
        return "_(table render error)_"


def render_markdown_report(common_series, uniques_map, reg_df,
                           per_disease_entities, per_pathway_type,
                           ent_unique_df, ent_subset_df, ent_core_df,
                           pds_df, out_md: Path):
    """Render markdown report with error handling."""
    try:
        lines = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# Pathway Comparison Report\n\nGenerated: {ts}\n")

        # Common pathways
        lines.append("## Common Pathways (present in ≥2 diseases)\n")
        if common_series is None or len(common_series) == 0:
            lines.append("_None yet (add more files to see overlaps)._")
        else:
            lines.append("Top by coverage:\n")
            try:
                for pw, cnt in common_series.sort_values(ascending=False).head(20).items():
                    lines.append(f"- **{pw}**: {int(cnt)} diseases")
            except Exception as e:
                logging.debug(f"Error listing common pathways: {e}")
        lines.append("")

        # Unique pathways
        lines.append("## Unique Pathways per Disease\n")
        try:
            for disease, items in (uniques_map or {}).items():
                lines.append(f"- **{disease}**: {len(items)} unique pathways")
                if items:
                    preview = ", ".join(items[:10])
                    if len(items) > 10:
                        preview += ", ..."
                    lines.append(f"  - e.g., {preview}")
        except Exception as e:
            logging.debug(f"Error listing unique pathways: {e}")
        lines.append("")

        # Regulation
        lines.append("## Direction of Regulation in Common Pathways\n")
        if reg_df is None or reg_df.empty:
            lines.append("_No regulation info yet._\n")
        else:
            try:
                mixed = reg_df[reg_df["direction_status"] == "mixed"].shape[0]
                consensus_up = reg_df[(reg_df["present_in"] >= 2) & (reg_df["direction_status"] == "UP")].shape[0]
                consensus_down = reg_df[(reg_df["present_in"] >= 2) & (reg_df["direction_status"] == "DOWN")].shape[0]
                lines.append(f"- Mixed (contradictory across diseases): **{mixed}** pathways")
                lines.append(f"- Consensus UP: **{consensus_up}** | Consensus DOWN: **{consensus_down}**\n")
                lines.append("Examples (first 15 rows):\n")
                for _, r in reg_df.head(15).iterrows():
                    lines.append(f"- **{r['pathway']}** — present in {int(r['present_in'])}; "
                                 f"UP in [{r['up_diseases']}]; DOWN in [{r['down_diseases']}]")
            except Exception as e:
                logging.debug(f"Error processing regulation section: {e}")

        # Entity associations
        lines.append("\n## Entity Associations\n")
        if per_disease_entities is None or per_disease_entities.empty:
            lines.append("_No entities captured yet._")
        else:
            try:
                lines.append("### Per Disease (counts)\n")
                summary = per_disease_entities.pivot_table(
                    index="disease", columns="entity_type",
                    values="n_entities", fill_value=0
                )
                lines.append(df_to_markdown(summary, index=True))
                lines.append("\n\n### Per Pathway & Entity Type (coverage)\n")
                top_cov = per_pathway_type.sort_values(
                    ["n_diseases", "n_entities"], ascending=[False, False]
                ).head(30).reset_index(drop=True)
                lines.append(df_to_markdown(top_cov, index=False))
            except Exception as e:
                logging.debug(f"Error processing entity associations: {e}")

        # Unique entities
        lines.append("\n## Disease-Unique Entities in Common Pathways\n")
        if ent_unique_df is None or ent_unique_df.empty:
            lines.append("_No disease-unique entities found in common pathways._")
        else:
            try:
                for _, r in ent_unique_df.head(15).iterrows():
                    qtxt = f" (q={r['qval']:.2e})" if pd.notna(r.get("qval")) else ""
                    lines.append(f"- **{r['pathway']}** ({r['entity_type']}): "
                                 f"**{r['entity']}** is unique to **{r['disease']}**{qtxt}")
            except Exception as e:
                logging.debug(f"Error processing unique entities: {e}")

        # Differentiation
        lines.append("\n## Most Differentiated Pathways\n")
        if pds_df is None or pds_df.empty:
            lines.append("_No differentiation scores available._")
        else:
            try:
                topk = pds_df.head(20).copy()
                cols = ["pathway", "PDS", "mean_dir_contrast", "mean_ent_diff", "present_in", "mixed_flag"]
                topk = topk[cols]
                lines.append(df_to_markdown(topk, index=False))
            except Exception as e:
                logging.debug(f"Error processing differentiation section: {e}")

        # Write file
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"Markdown report written: {out_md}")

    except Exception as e:
        logging.error(f"Failed to write markdown report: {e}")
        try:
            # Minimal fallback report
            fallback = f"# Pathway Comparison Report\n\nError generating full report: {e}\n"
            out_md.write_text(fallback, encoding="utf-8")
        except Exception:
            pass


def add_direction_heatmap(direction_mat: pd.DataFrame, out_png: Path, top_n: int = 40) -> Optional[Path]:
    """Add direction heatmap with error handling."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    if direction_mat is None or direction_mat.empty:
        return None

    try:
        variability = direction_mat.std(axis=1).sort_values(ascending=False)
        sel = variability.head(top_n).index
        dm = direction_mat.loc[sel]

        plt.figure(figsize=(max(7.5, len(dm.columns) * 0.7), max(7.5, len(dm.index) * 0.3)))
        plt.imshow(dm.values, aspect='auto', interpolation='nearest', cmap='RdBu_r')
        plt.yticks(range(len(dm.index)), dm.index, fontsize=8)
        plt.xticks(range(len(dm.columns)), dm.columns, rotation=90, fontsize=8)
        plt.title("Direction Heatmap (−1=DOWN, 0=absent, +1=UP) — Top variable pathways")
        plt.xlabel("Disease")
        plt.ylabel("Pathway")
        plt.colorbar(label="Direction")
        plt.tight_layout()

        out_png2 = Path(str(out_png).replace(".png", "_direction.png"))
        plt.savefig(out_png2, dpi=220, bbox_inches="tight")
        plt.close()

        logging.info(f"Direction heatmap saved: {out_png2}")
        return out_png2

    except Exception as e:
        logging.warning(f"Direction heatmap failed: {e}")
        return None
    finally:
        try:
            plt.close('all')
        except Exception:
            pass


# ----------------------------- Main -----------------------------

def main():
    """Main function with comprehensive error handling."""
    try:
        ap = argparse.ArgumentParser(
            description="Compare pathway_entity_overlap.json files (<=30) with robust schema handling.",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        ap.add_argument("--input", required=True, help="Directory containing JSON files.")
        ap.add_argument("--pattern", default="*.json", help="Glob pattern for files (default: *.json).")
        ap.add_argument("--cap", type=int, default=30, help="Max number of files to include (default: 30).")
        ap.add_argument("--outdir", default=".", help="Output directory (default: current dir).")
        ap.add_argument("--prefix", default="pathway_comparison", help="Output filename prefix.")

        # Disease label options
        ap.add_argument("--label-source", choices=["filename", "metadata"], default="filename",
                        help="Where to read disease label from (default: filename).")
        ap.add_argument("--label-key", default="disease",
                        help="Key used when --label-source=metadata (default: disease).")
        ap.add_argument("--filename-regex", default=None,
                        help="Regex with (?P<disease>...) to extract the disease from filename.")
        ap.add_argument("--map-csv", default=None,
                        help="Optional CSV map (file,disease). Overrides other sources if provided.")

        # Logging
        ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

        args = ap.parse_args()
        setup_logging(args.verbose)

        in_dir = Path(args.input)
        out_dir = Path(args.outdir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Cannot create output directory {out_dir}: {e}")
            sys.exit(2)

        files = list(in_dir.glob(args.pattern))
        if len(files) == 0:
            logging.error(f"No files matched {args.pattern} in {in_dir}")
            sys.exit(1)

        files = cap_files(files, args.cap)
        map_csv = load_map_csv(Path(args.map_csv)) if args.map_csv else {}

        all_presence = []
        all_entities = []

        for fp in files:
            raw_for_label = load_json(fp)
            disease_name = resolve_disease_label(
                fp=fp,
                data=raw_for_label,
                label_source=args.label_source,
                label_key=args.label_key,
                filename_regex=args.filename_regex,
                map_csv=map_csv,
            )
            try:
                p_df, e_df = parse_one_file(fp, disease_name)
                if not p_df.empty:
                    all_presence.append(p_df)
                if e_df is not None and not e_df.empty:
                    all_entities.append(e_df)
            except Exception as e:
                logging.warning(f"Failed to parse {fp.name}: {e}")

        presence_df = pd.concat(all_presence, ignore_index=True) if all_presence else pd.DataFrame(
            columns=["disease", "pathway", "UP", "DOWN"]
        )
        entities_df = pd.concat(all_entities, ignore_index=True) if all_entities else pd.DataFrame(
            columns=["disease", "pathway", "direction", "entity_type", "entity",
                     "OR", "pval", "qval", "Jaccard", "k", "a", "b", "N", "overlap_genes"]
        )

        if presence_df.empty:
            logging.error("Parsed presence table is empty. Check your input files and label configuration.")
            sys.exit(1)

        # Presence & regulation
        mat_up, mat_down, mat_any = compute_presence_matrices(presence_df)
        presence_counts, common_pathways, uniques_map = summarize_common_unique(mat_any)
        reg_df = regulation_agreement(mat_up, mat_down)
        per_disease_entities, per_pathway_type, long_entities = aggregate_entities(entities_df)

        # Entity-level differences
        ent_unique_df, ent_subset_df, ent_core_df, ent_mats = compute_entity_differences(entities_df, mat_any)

        # Differentiation & pairwise
        direction_mat = build_direction_matrix(presence_df)  # {+1,0,-1}; for non-directional, {+1,0}
        entity_sets = build_entity_sets(long_entities)
        pairwise_diffs = compute_pairwise_diffs(direction_mat, entity_sets,
                                                diseases=list(direction_mat.columns) if not direction_mat.empty else None)
        pds_df = summarize_pathway_differentiation(pairwise_diffs, reg_df, w_dir=0.6, w_ent=0.4)

        pairwise_consolidated = build_pairwise_consolidated(direction_mat, entity_sets, mat_any)
        pairwise_wide = build_pairwise_wide(pairwise_consolidated)

        # -------------------- Similarity workbook (separate file) --------------------
        sim_sheets = {}
        cluster_png = None
        try:
            # Use positional args for maximum compatibility with your module:
            # (mat_up, mat_down, mat_any, entities_df, entity_sets, out_dir, prefix)
            sim_sheets, cluster_png = compute_similarity_block(
                mat_up, mat_down, mat_any, entities_df, entity_sets, out_dir, args.prefix
            )
        except Exception as e:
            logging.warning(f"Similarity computation failed (skipping similarity workbook): {e}")
            sim_sheets = {}
            cluster_png = None

        if sim_sheets:
            similarity_out = out_dir / f"{args.prefix}_similarity.xlsx"
            try:
                write_excel(sim_sheets, similarity_out)
                logging.info(f"Similarity Excel written: {similarity_out}")
            except Exception as e:
                logging.error(f"Failed writing similarity workbook: {e}")
        # ---------------------------------------------------------------------------

        # Outputs
        excel_out = out_dir / f"{args.prefix}.xlsx"
        md_out = out_dir / f"{args.prefix}.md"
        png_out = out_dir / f"{args.prefix}_heatmap.png"

        make_heatmap(mat_any, png_out)
        _ = add_direction_heatmap(direction_mat, png_out)  # writes *_direction.png next to heatmap

        excel_sheets = {
            "Pathway_Inventory": presence_df.sort_values(["disease", "pathway"]).reset_index(drop=True),
            "Presence_UP": mat_up.reset_index().rename(columns={"index": "pathway"}) if not mat_up.empty else pd.DataFrame(),
            "Presence_DOWN": mat_down.reset_index().rename(columns={"index": "pathway"}) if not mat_down.empty else pd.DataFrame(),
            "Presence_ANY": mat_any.reset_index().rename(columns={"index": "pathway"}) if not mat_any.empty else pd.DataFrame(),
            "Common_Pathways": presence_counts.rename("n_diseases").reset_index().rename(columns={"index": "pathway"}) if not presence_counts.empty else pd.DataFrame(),
            "Regulation_Consensus": reg_df,
            "Per_Disease_Entities": per_disease_entities,
            "Per_Pathway_EntityType": per_pathway_type,
            "Entity_Associations_Long": long_entities,
            "Entity_Unique_Per_Disease": ent_unique_df,
            "Entity_Subset_Shared": ent_subset_df,
            "Entity_Core": ent_core_df,
            "Entity_Presence_Matrix_metabolites": ent_mats.get("metabolites", pd.DataFrame()).reset_index() if isinstance(ent_mats.get("metabolites"), pd.DataFrame) else pd.DataFrame(),
            "Entity_Presence_Matrix_epigenetic": ent_mats.get("epigenetic", pd.DataFrame()).reset_index() if isinstance(ent_mats.get("epigenetic"), pd.DataFrame) else pd.DataFrame(),
            "Entity_Presence_Matrix_tf": ent_mats.get("tf", pd.DataFrame()).reset_index() if isinstance(ent_mats.get("tf"), pd.DataFrame) else pd.DataFrame(),
            "Direction_Matrix": direction_mat.reset_index().rename(columns={"index": "pathway"}) if not direction_mat.empty else pd.DataFrame(),
            "Pairwise_Diffs": pairwise_diffs,
            "Pathway_Diff_Summary": pds_df,
            "Pairwise_Consolidated": pairwise_consolidated,
            "Pairwise_Wide": pairwise_wide,
        }

        write_excel(excel_sheets, excel_out)

        # CSV exports
        try:
            if pairwise_consolidated is not None and not pairwise_consolidated.empty:
                (out_dir / f"{args.prefix}_pairwise_consolidated.csv]").write_text(
                    pairwise_consolidated.to_csv(index=False), encoding="utf-8"
                )
        except Exception:
            # Fix potential bracket typo in path
            try:
                (out_dir / f"{args.prefix}_pairwise_consolidated.csv").write_text(
                    pairwise_consolidated.to_csv(index=False), encoding="utf-8"
                )
            except Exception as e:
                logging.warning(f"CSV export (consolidated) failed: {e}")

        try:
            if pairwise_wide is not None and not pairwise_wide.empty:
                (out_dir / f"{args.prefix}_pairwise_wide.csv").write_text(
                    pairwise_wide.to_csv(index=False), encoding="utf-8"
                )
        except Exception as e:
            logging.warning(f"CSV export (wide) failed: {e}")

        # Report
        render_markdown_report(common_pathways, uniques_map, reg_df,
                               per_disease_entities, per_pathway_type,
                               ent_unique_df, ent_subset_df, ent_core_df,
                               pds_df, md_out)

        # Append cluster plot note to MD if present
        try:
            if cluster_png is not None and Path(cluster_png).exists():
                with md_out.open("a", encoding="utf-8") as f:
                    f.write("\n\n## Disease Clusters\n")
                    f.write(f"Cluster plot saved to: {cluster_png}\n")
        except Exception as _e:
            logging.debug(f"Appending cluster note to markdown failed: {_e}")

        # Console summary
        print("\n=== Comparison summary ===")
        print(f"Files analyzed: {len(files)}")
        print(f"Excel:   {excel_out}")
        print(f"Report:  {md_out}")
        print(f"Heatmap: {png_out}")
        dhm = Path(str(png_out).replace(".png", "_direction.png"))
        if dhm.exists():
            print(f"Heatmap: {dhm}")
        if (out_dir / f"{args.prefix}_pairwise_consolidated.csv").exists():
            print(f"CSV:     {out_dir / f'{args.prefix}_pairwise_consolidated.csv'}")
        if (out_dir / f"{args.prefix}_pairwise_wide.csv").exists():
            print(f"CSV:     {out_dir / f'{args.prefix}_pairwise_wide.csv'}")
        if sim_sheets:
            print(f"Similarity Excel: {out_dir / f'{args.prefix}_similarity.xlsx'}")
        if cluster_png is not None and Path(cluster_png).exists():
            print(f"Clusters: {cluster_png}")

        # Extra terminal clarity for non-directional inputs (no logic/output changes)
        try:
            present_per_disease = (
                mat_any.sum(axis=0).astype(int)
                if (mat_any is not None and not mat_any.empty)
                else pd.Series(dtype=int)
            )
        except Exception:
            present_per_disease = pd.Series(dtype=int)

        print("Pathways present per disease (ANY of UP/DOWN):")
        if present_per_disease is not None and not present_per_disease.empty:
            for d in present_per_disease.index:
                print(f"  - {d}: {int(present_per_disease.loc[d])}")
        else:
            print("  (no presence matrix)")

        print("Unique-only pathways per disease (appear only in that disease):")
        for d, items in (uniques_map or {}).items():
            print(f"  - {d}: {len(items)}")

    except SystemExit:
        # argparse or explicit sys.exit()
        raise
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
