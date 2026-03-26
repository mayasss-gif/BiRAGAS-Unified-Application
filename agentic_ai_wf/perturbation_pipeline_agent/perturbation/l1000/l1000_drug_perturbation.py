# src/l1000_drug_perturbation.py
from logging import Logger
import logging
import json
import math
import gzip
import shutil
import time
import unicodedata
import warnings
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy import stats
from cmapPy.pandasGEXpress import parse as gct_parse


from .constants import CORE_SIG_PATH, CORE_PERT_PATH, CORE_CELL_PATH, CORE_GENE_PATH, CORE_GCTX_PATH

# -------------------------------------------------------------------
# Logging helper (replacement for l1000_cfg.log)
# -------------------------------------------------------------------
def log(msg: str) -> None:
    logging.getLogger("L1000").info(msg)


# -------------------------------------------------------------------
# User input file (backend config)
# -------------------------------------------------------------------
USER_INPUT_FILE = Path(__file__).resolve().parent / "userinput.txt"


def _parse_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v == "":
        return default
    return v in {"true", "t", "1", "yes", "y"}


def _parse_list(val: Optional[str]) -> List[str]:
    if val is None:
        return []
    v = val.strip()
    if not v:
        return []
    # allow comma- or space-separated
    parts = re.split(r"[,\s]+", v)
    return [p.strip() for p in parts if p.strip()]


def _parse_times(val: Optional[str]) -> List[float]:
    """
    Parse 'Times (h)' from userinput.txt.

    Behaviour:
      - If the line is missing or empty -> []  (NO time filter, use all times)
      - If there are values -> parse them as floats (e.g. '24, 48')
    """
    if val is None:
        # No time restriction if user didn't specify anything
        return []
    v = val.strip()
    if not v:
        # Empty string also means: no time restriction
        return []
    times: List[float] = []
    for piece in re.split(r"[,\s]+", v):
        piece = piece.strip()
        if not piece:
            continue
        # allow things like "24" or "24h"
        m = re.match(r"^([-\d\.eE]+)", piece)
        if not m:
            continue
        try:
            times.append(float(m.group(1)))
        except Exception:
            continue
    # If nothing parsed, still treat as "no restriction"
    return times or []


def load_user_config(path: Path) -> Dict[str, Any]:
    """
    Parse src/userinput.txt into a config dict.

    Example lines:
        Primary site: Kidney
        Cell whitelist: HEK293T, MCF7
        Drug: lapatinib, neratinib
        Times (h): 24, 48   # leave blank to use all times
        Pert.Type: trt_cp, ctl_vehicle
        Max sigs: 300
        Max ctrls/str: 10
        Include Relevance: TRUE
        Include ATE: TRUE
        Export Plots: TRUE
        Auto-augment ≥4 doses/pair: TRUE
    """
    if not path.exists():
        raise FileNotFoundError(f"userinput.txt not found at: {path}")

    raw: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            raw[key] = val

    # Required
    primary_site = raw.get("primary site", "").strip()
    if not primary_site:
        raise ValueError("Primary site is required in userinput.txt (line: 'Primary site: ...').")

    # Optional
    cell_whitelist = _parse_list(raw.get("cell whitelist"))
    drugs = _parse_list(raw.get("drug"))
    times = _parse_times(raw.get("times (h)"))

    # 'Pert.Type' line may appear as 'Pert.Type' or 'Pert.Type ' etc.
    pert_types = _parse_list(next((v for k, v in raw.items() if k.startswith("pert.type")), None))

    max_sigs = raw.get("max sigs", "").strip()
    max_sigs_int = int(max_sigs) if max_sigs else 600

    max_ctrls = raw.get("max ctrls/str", "").strip()
    max_ctrls_int = int(max_ctrls) if max_ctrls else 10

    include_rel = _parse_bool(raw.get("include relevance"), True)
    include_ate = _parse_bool(raw.get("include ate"), True)
    export_plots = _parse_bool(raw.get("export plots"), True)

    # Auto-augment line may have unicode ≥ symbol etc.
    auto_key = next((k for k in raw if "auto-augment" in k), None)
    auto_val = raw.get(auto_key) if auto_key else None
    auto_augment = _parse_bool(auto_val, True)

    cfg = dict(
        PRIMARY_SITE_FILTER=primary_site,
        CELL_WHITELIST=cell_whitelist,
        TIME_FILTERS=times,          # [] means "all times"
        drugs=drugs or None,
        pert_types=pert_types or None,
        MAX_SIGS=max_sigs_int,
        MAX_CTLS_PER_STRATUM=max_ctrls_int,
        INCLUDE_RELEVANCE=include_rel,
        INCLUDE_ATE=include_ate,
        EXPORT_PLOTS=export_plots,
        auto_augment_doses=auto_augment,
    )

    log("Loaded user config from userinput.txt:")
    log(json.dumps(cfg, indent=2))
    if not cfg["TIME_FILTERS"]:
        log("No 'Times (h)' specified → using ALL available time points.")
    return cfg


# -------------------------------------------------------------------
# Core helpers (adapted from notebook)
# -------------------------------------------------------------------
def ensure_exists(*paths: Path) -> None:
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file: {p}")



def normalize_unit(u):
    if pd.isna(u):
        return u
    s = str(u)
    s = unicodedata.normalize("NFKC", s).replace("µ", "u").replace("μ", "u").strip()
    return s


def kmeans_labels(X, n_clusters):
    try:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    except TypeError:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(X)


def safe_norm01(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mn, mx = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn + 1e-12)


# Hill model & fit
def hill(dose, ec50, h, emax, b):
    return b + (emax - b) * (dose ** h / (ec50 ** h + dose ** h))


def fit_hill(dose_arr, resp_arr):
    x, y = np.asarray(dose_arr, float), np.asarray(resp_arr, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if np.unique(x).size < 4 or np.isfinite(y).sum() < 4:
        return None
    p0 = [np.nanmedian(x), 1.0, np.nanmax(y), np.nanmin(y)]
    bounds = ([1e-12, 0.1, -np.inf, -np.inf], [np.inf, 10.0, np.inf, np.inf])
    try:
        popt, _ = curve_fit(hill, x, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = hill(x, *popt)
        r2 = r2_score(y, pred)
        ec50, hcoef, emax, base = popt
        si = abs(emax - base) / (ec50 + 1e-8)
        return dict(EC50=ec50, Hill=hcoef, R2=r2, Emax=emax, Baseline=base, SensitivityIndex=si)
    except Exception:
        return None


def _split_genes(s):
    if pd.isna(s):
        return []
    raw = re.split(r"[,\s;]+", str(s))
    return [g.strip().upper() for g in raw if g and g.strip()]


# -------------------------------------------------------------------
# Global state initialised at runtime
# -------------------------------------------------------------------
merged: Optional[pd.DataFrame] = None
cell_df: Optional[pd.DataFrame] = None
LM_IDS: List[str] = []
ID2SYM: Dict[str, str] = {}
DISEASE_COL: Optional[str] = None
deg_summary: Optional[pd.DataFrame] = None
gene2pw: Dict[str, Dict[str, Any]] = {}


def init_l1000_metadata() -> None:
    """
    Load SIG, PERT, CELL, GENE and build merged meta table.
    Also initialise LM_IDS/ID2SYM and DISEASE_COL.
    """
    global merged, cell_df, LM_IDS, ID2SYM, DISEASE_COL

    ensure_exists(CORE_SIG_PATH, CORE_PERT_PATH, CORE_CELL_PATH, CORE_GENE_PATH)

    sig = pd.read_csv(CORE_SIG_PATH, sep="\t", low_memory=False)
    pert = pd.read_csv(CORE_PERT_PATH, sep="\t", low_memory=False)
    cell = pd.read_csv(CORE_CELL_PATH, sep="\t", low_memory=False)
    gene = pd.read_csv(CORE_GENE_PATH, sep="\t", low_memory=False)
    cell_df = cell

    gene["pr_gene_id"] = gene["pr_gene_id"].astype(str)
    gene["pr_gene_symbol"] = gene["pr_gene_symbol"].astype(str)
    is_lm = pd.to_numeric(gene["pr_is_lm"], errors="coerce").fillna(0) > 0
    LM_IDS = gene.loc[is_lm, "pr_gene_id"].astype(str).tolist()
    ID2SYM = gene.set_index("pr_gene_id")["pr_gene_symbol"].to_dict()

    keep_sig = [
        c
        for c in [
            "sig_id",
            "pert_id",
            "pert_type",
            "pert_iname",
            "cell_id",
            "pert_time",
            "pert_dose",
            "pert_dose_unit",
        ]
        if c in sig.columns
    ]
    keep_pert = [c for c in ["pert_id", "pert_iname", "pert_type"] if c in pert.columns]
    keep_cell = [c for c in ["cell_id", "primary_site", "cell_name"] if c in cell.columns]

    merged_meta = (
        sig[keep_sig]
        .merge(pert[keep_pert].drop_duplicates("pert_id"), on=["pert_id", "pert_iname", "pert_type"], how="left")
        .merge(cell[keep_cell].drop_duplicates("cell_id"), on="cell_id", how="left")
    )

    merged_meta["dose_num"] = pd.to_numeric(merged_meta.get("pert_dose"), errors="coerce")
    merged_meta["pert_time_num"] = pd.to_numeric(merged_meta.get("pert_time"), errors="coerce")
    if "pert_dose_unit" in merged_meta.columns:
        merged_meta["pert_dose_unit"] = merged_meta["pert_dose_unit"].apply(normalize_unit)

    # disease column auto-detect
    _DISEASE_CANDS = ["disease", "disease_name", "subtype", "primary_disease", "histology", "diagnosis"]
    DISEASE_COL_LOCAL = next((c for c in _DISEASE_CANDS if c in merged_meta.columns), None)

    merged = merged_meta
    DISEASE_COL = DISEASE_COL_LOCAL

    log(
        f"L1000 metadata loaded: sig={sig.shape}, pert={pert.shape}, "
        f"cell={cell.shape}, gene={gene.shape}; merged={merged.shape}"
    )


def init_degs_and_pathways(output_dir: Path) -> None:
    """
    Load prepared DEGs and Pathways from output_dir for this run.

    Requires that you have already run:
        run_l1000_prepare_files.py
    for the same L1000_RUN_STAMP.
    """
    global deg_summary, gene2pw

    deg_path = output_dir / "tables" / "degs_prepared.csv"
    pw_path = output_dir / "tables" / "pathways_prepared.csv"
    ensure_exists(deg_path, pw_path)

    deg_df = pd.read_csv(deg_path).rename(columns=lambda c: c.strip())
    deg_df["Gene"] = deg_df["Gene"].astype(str).str.upper().str.strip()
    deg_df["Log2FC"] = pd.to_numeric(deg_df.get("Log2FC"), errors="coerce")
    if "Patient_LFC_Trend" in deg_df.columns:
        deg_df["Patient_LFC_Trend"] = (
            deg_df["Patient_LFC_Trend"].astype(str).str.upper().str.strip()
        )
    else:
        deg_df["Patient_LFC_Trend"] = np.where(
            deg_df["Log2FC"] > 0,
            "UP",
            np.where(deg_df["Log2FC"] < 0, "DOWN", "FLAT"),
        )
    deg_summary = deg_df[["Gene", "Log2FC", "Patient_LFC_Trend"]].drop_duplicates("Gene")

    pw = pd.read_csv(pw_path).rename(columns=lambda c: c.strip())
    for col in [
        "Pathway",
        "Main_Class",
        "Sub_Class",
        "Regulation",
        "Pathway associated genes",
        "p_value",
        "fdr",
    ]:
        if col not in pw.columns:
            pw[col] = np.nan

    pw_expl = []
    for _, r in pw.iterrows():
        for g in _split_genes(r.get("Pathway associated genes")):
            pw_expl.append(
                {
                    "Gene": g,
                    "Pathway": r.get("Pathway"),
                    "Main_Class": r.get("Main_Class"),
                    "Sub_Class": r.get("Sub_Class"),
                    "Regulation": r.get("Regulation"),
                    "p_value": r.get("p_value"),
                    "fdr": r.get("fdr"),
                }
            )
    pw_expl = pd.DataFrame(pw_expl)
    if not pw_expl.empty:
        pw_expl["fdr"] = pd.to_numeric(pw_expl["fdr"], errors="coerce")
        pw_expl["p_value"] = pd.to_numeric(pw_expl["p_value"], errors="coerce")
        best_pw = pw_expl.sort_values(["Gene", "fdr", "p_value"]).drop_duplicates("Gene")
    else:
        best_pw = pd.DataFrame(columns=["Gene", "Pathway", "Main_Class", "Sub_Class", "Regulation"])

    gene2pw = best_pw.set_index("Gene")[
        ["Pathway", "Main_Class", "Sub_Class", "Regulation"]
    ].to_dict(orient="index")

    log(
        f"Prepared DEGs and Pathways loaded: DEGs={deg_df.shape}, "
        f"Pathways={pw.shape}, pw_expl={pw_expl.shape}"
    )


def tag_gene_info(gene_symbol: str) -> Tuple[str, str, str, str, str]:
    global deg_summary, gene2pw
    g = str(gene_symbol).upper()
    info = gene2pw.get(g, {})
    if deg_summary is None or deg_summary.empty:
        trend = "NA"
    else:
        trend = (
            deg_summary.set_index("Gene")
            .get("Patient_LFC_Trend", pd.Series())
            .get(g, "NA")
        )
    return (
        info.get("Main_Class", "Unknown"),
        info.get("Sub_Class", "Unknown"),
        trend,
        info.get("Pathway", "Unknown"),
        info.get("Regulation", "Unknown"),
    )


def attach_gene_classes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    data = [tag_gene_info(g) for g in df["Gene"].astype(str)]
    df = df.copy()
    df["Gene_Main_Class"], df["Gene_Sub_Class"], df["Gene_DEG_Trend"], df[
        "Gene_Pathway"
    ], df["Gene_Regulation"] = zip(*data)
    return df



def select_signatures(
    meta_df: pd.DataFrame,
    primary_site=None,
    cell_whitelist=None,
    time_filters=None,
    max_sigs=600,
    disease=None,
    drugs=None,
    pert_types=None,
):
    global DISEASE_COL

    def _filter(_primary, _wl, _times, _disease, _drugs, _ptypes):
        m = meta_df.copy()
        if _primary and "primary_site" in m.columns:
            m = m[m["primary_site"].astype(str).str.lower() == str(_primary).lower()]
        if DISEASE_COL and _disease:
            m = m[m[DISEASE_COL].astype(str).str.lower() == str(_disease).lower()]
        if _wl:
            present = set(m["cell_id"].dropna().unique())
            keep = [c for c in _wl if c in present]
            m = m[m["cell_id"].isin(keep)] if keep else m.iloc[0:0]
        if _ptypes:
            m = m[m["pert_type"].astype(str).isin(list(_ptypes))]
        if _drugs:
            m = m[m["pert_iname"].astype(str).isin(list(_drugs))]
        if _times and "pert_time" in m.columns:
            m = m[pd.to_numeric(m["pert_time"], errors="coerce").isin([float(t) for t in _times])]

        # default keep compound treatments, plus any user-selected extra pert_types
        if not _ptypes or any(t.startswith("trt_") for t in _ptypes):
            m = m[
                m["pert_type"].astype(str).str.startswith("trt_cp")
                | m["pert_type"].astype(str).isin(list(_ptypes or []))
            ]

        m["pert_dose_unit"] = m["pert_dose_unit"].apply(normalize_unit)
        m["dose_num"] = pd.to_numeric(m["pert_dose"], errors="coerce")
        m = m.dropna(subset=["sig_id", "pert_iname", "dose_num", "pert_dose_unit"])
        return m

    candidates = [
        ("strict", primary_site, cell_whitelist or [], time_filters or [], disease, drugs, pert_types),
        ("no_whitelist", primary_site, [], time_filters or [], disease, drugs, pert_types),
        ("no_time", primary_site, [], [], disease, drugs, pert_types),
        ("no_primary_site", None, [], [], None, drugs, pert_types),
    ]
    for tag, p, wl, ts, dis, drs, pts in candidates:
        m = _filter(p, wl, ts, dis, drs, pts)
        if not m.empty:
            dose_div = (
                m.groupby(["pert_iname", "pert_dose_unit"])["dose_num"]
                .nunique()
                .reset_index(name="n_doses")
                .sort_values("n_doses", ascending=False)
            )
            top_drugs = dose_div["pert_iname"].unique().tolist()
            sel = []
            per_drug_cap = max(4, math.ceil(max_sigs / max(1, len(top_drugs))) + 2)
            for dn in top_drugs:
                block = m[m["pert_iname"] == dn].sort_values(["pert_dose_unit", "dose_num"])
                sel.append(block.groupby("pert_dose_unit", group_keys=False).head(per_drug_cap))
            sm = pd.concat(sel, ignore_index=True).drop_duplicates("sig_id").head(max_sigs)
            sm["pert_dose_unit"] = sm["pert_dose_unit"].apply(normalize_unit)
            scope = dict(
                mode=tag,
                primary_site=p,
                whitelist=[*wl],
                disease=dis,
                times=[*ts],
                pert_types=[*(pts or [])],
                drugs=[*(drs or [])],
                n_signatures=len(sm),
                cells=sorted(sm["cell_id"].dropna().unique().tolist()),
                times_present=sorted(
                    pd.to_numeric(sm["pert_time"], errors="coerce").dropna().unique().tolist()
                ),
            )
            return sm, scope
    return pd.DataFrame(), dict(mode="none")


def augment_selection_for_doses(selection_df, meta_df, min_doses=4, max_add_per_pair=16):
    if selection_df.empty:
        return selection_df
    sel = selection_df.copy()
    sel["pert_dose_unit"] = sel["pert_dose_unit"].apply(normalize_unit)
    pair_counts = (
        sel.groupby(["pert_iname", "pert_dose_unit"])["dose_num"]
        .nunique()
        .reset_index(name="n_doses")
    )
    need_pairs = pair_counts[pair_counts["n_doses"] < min_doses][
        ["pert_iname", "pert_dose_unit"]
    ].values.tolist()
    if not need_pairs:
        return sel

    meta = meta_df.copy()
    meta = meta[meta["pert_type"].astype(str).str.startswith("trt_cp")]
    meta["pert_dose_unit"] = meta["pert_dose_unit"].apply(normalize_unit)
    meta["dose_num"] = pd.to_numeric(meta["pert_dose"], errors="coerce")
    meta = meta.dropna(subset=["dose_num", "pert_dose_unit"])

    added = []
    for drug, unit in need_pairs:
        cur = sel[(sel["pert_iname"] == drug) & (sel["pert_dose_unit"] == unit)]
        have_doses = set(np.round(cur["dose_num"].dropna().values, 12))
        pool = meta[(meta["pert_iname"] == drug) & (meta["pert_dose_unit"] == unit)].copy()
        if pool.empty:
            continue
        pool = pool.sort_values("dose_num")
        picks = []
        for _, row in pool.iterrows():
            d = round(float(row["dose_num"]), 12)
            if d not in have_doses:
                picks.append(row)
                have_doses.add(d)
            if len(have_doses) >= min_doses or len(picks) >= max_add_per_pair:
                break
        if picks:
            added.append(pd.DataFrame(picks))
    if added:
        aug = pd.concat([sel] + added, ignore_index=True).drop_duplicates("sig_id")
        log(
            f"Augmented selection with {len(aug) - len(sel)} extra signatures "
            f"to reach ≥{min_doses} doses per pair (where possible)."
        )
        return aug
    return sel


# -------------------------------------------------------------------
# Main pipeline (identical logic, but uses globals + our paths)
# -------------------------------------------------------------------
def run_pipeline(
    output_dir: Path,
    logger: Logger,
    INCLUDE_RELEVANCE=True,
    INCLUDE_ATE=True,
    EXPORT_PLOTS=True,
    TOP_K_PER_GENE=None,
    PRIMARY_SITE_FILTER=None,
    CELL_WHITELIST=None,
    TIME_FILTERS=None,
    MAX_SIGS=600,
    MAX_CTLS_PER_STRATUM=10,
    force_override=False,
    auto_augment_doses=True,
    disease=None,
    drugs=None,
    pert_types=None,
):
    global merged, cell_df, LM_IDS, ID2SYM

    assert merged is not None, "L1000 metadata not initialised."
    assert deg_summary is not None, "DEG summary not initialised."

    CELL_WHITELIST = CELL_WHITELIST or []
    TIME_FILTERS = TIME_FILTERS or []

    # -------------------------------------------------------------------
    # Selection helpers (unchanged logic, using global merged/DISEASE_COL)
    # -------------------------------------------------------------------
    SELECTION_META_CSV = output_dir / "selected_signatures_meta.csv"
    SELECTION_CFG_JSON = output_dir / "user_selection.json"
    # ----- Selection -----
    use_saved = SELECTION_META_CSV.exists() and not force_override
    if use_saved:
        log("Using existing selected_signatures_meta.csv")
        selected_meta = pd.read_csv(SELECTION_META_CSV)
        if "dose_num" not in selected_meta.columns and "pert_dose" in selected_meta.columns:
            selected_meta["dose_num"] = pd.to_numeric(selected_meta["pert_dose"], errors="coerce")
        if "pert_dose_unit" in selected_meta.columns:
            selected_meta["pert_dose_unit"] = selected_meta["pert_dose_unit"].apply(normalize_unit)
        if "primary_site" not in selected_meta.columns and "cell_id" in selected_meta.columns:
            assert cell_df is not None
            selected_meta = selected_meta.merge(
                cell_df[["cell_id", "primary_site"]].drop_duplicates("cell_id"),
                on="cell_id",
                how="left",
            )
        req_cols = [
            "sig_id",
            "pert_id",
            "pert_type",
            "pert_iname",
            "cell_id",
            "pert_time",
            "pert_dose",
            "pert_dose_unit",
            "dose_num",
            "primary_site",
        ]
        for c in req_cols:
            if c not in selected_meta.columns:
                selected_meta[c] = np.nan
        small_meta = selected_meta.dropna(subset=["sig_id", "pert_iname"]).copy()
        scope = {"mode": "user_saved"}
    else:
        log("Building selection from user inputs …")
        small_meta, scope = select_signatures(
            merged,
            PRIMARY_SITE_FILTER,
            CELL_WHITELIST,
            TIME_FILTERS,
            MAX_SIGS,
            disease=disease,
            drugs=drugs,
            pert_types=pert_types,
        )
        if small_meta.empty:
            raise ValueError(
                "No signatures found with current filters. "
                "Loosen tissue/cell/time filters and include 'trt_cp'."
            )
        small_meta.to_csv(SELECTION_META_CSV, index=False)
        with open(SELECTION_CFG_JSON, "w") as f:
            json.dump(
                dict(
                    PRIMARY_SITE_FILTER=PRIMARY_SITE_FILTER,
                    CELL_WHITELIST=CELL_WHITELIST,
                    TIME_FILTERS=TIME_FILTERS,
                    MAX_SIGS=int(MAX_SIGS),
                    disease=disease,
                    drugs=drugs,
                    pert_types=pert_types,
                ),
                f,
                indent=2,
            )
        log(f"Wrote selection CSV ({len(small_meta)})")

    # Optional auto-augment
    if auto_augment_doses:
        small_meta = augment_selection_for_doses(small_meta, merged, min_doses=4, max_add_per_pair=24)

    # ----- Controls (ATE) -----
    def collect_matched_controls(meta_df, treat_df, max_per_stratum=10):
        ctl = meta_df[meta_df["pert_type"].astype(str).str.startswith("ctl_")].copy()
        if ctl.empty:
            return ctl.iloc[0:0]
        strata = treat_df[["cell_id", "pert_time"]].dropna().drop_duplicates()
        ctl = ctl.merge(strata, on=["cell_id", "pert_time"], how="inner")
        ctl["pert_dose_unit"] = ctl["pert_dose_unit"].apply(normalize_unit)
        ctl["dose_num"] = 0.0
        ctl = (
            ctl.sort_values("sig_id")
            .groupby(["cell_id", "pert_time"], group_keys=False)
            .head(int(max_per_stratum))
        )
        return ctl

    control_meta = (
        collect_matched_controls(merged, small_meta, MAX_CTLS_PER_STRATUM)
        if INCLUDE_ATE
        else pd.DataFrame()
    )
    if INCLUDE_ATE and control_meta.empty:
        log("No matched controls found → disabling ATE for this run.")
        INCLUDE_ATE = False

    # ----- Expression slice -----
    expr_meta = (
        pd.concat([small_meta, control_meta], ignore_index=True).drop_duplicates("sig_id")
        if INCLUDE_ATE
        else small_meta
    )
    cid_list = expr_meta["sig_id"].astype(str).tolist()
    if not cid_list:
        raise ValueError("No signatures to fetch after selection.")
    gctoo = gct_parse.parse(str(CORE_GCTX_PATH), rid=LM_IDS, cid=cid_list)
    expr = gctoo.data_df.copy()
    expr.index = expr.index.astype(str)
    expr_sym = expr.copy()
    expr_sym.index = [ID2SYM.get(i, i) for i in expr.index]

    # ----- Genes present/missing -----
    CUSTOM_GENES = (
        deg_summary["Gene"].dropna().astype(str).str.upper().unique().tolist()
        if deg_summary is not None
        else []
    )
    GENES_PRESENT = [g for g in CUSTOM_GENES if g in expr_sym.index]
    MISSING = sorted(set(CUSTOM_GENES) - set(GENES_PRESENT))
    log(
        f"Genes requested: {len(CUSTOM_GENES)} | present: {len(GENES_PRESENT)} | missing: {len(MISSING)}"
    )
    pd.Series(MISSING, name="missing_genes").to_csv(output_dir / "missing_genes.txt", index=False)

    # ----- Fits -----
    fits = []
    if GENES_PRESENT and not small_meta.empty:
        for (drug, unit), dfD in small_meta.groupby(["pert_iname", "pert_dose_unit"]):
            d = dfD[["sig_id", "dose_num"]].dropna().sort_values("dose_num")
            if d["dose_num"].nunique() < 4:
                continue
            x = d["dose_num"].astype(float).values
            for g in GENES_PRESENT:
                if g not in expr_sym.index:
                    continue
                y = (
                    pd.to_numeric(expr_sym.loc[g, d["sig_id"]], errors="coerce")
                    .astype(float)
                    .values
                )
                res = fit_hill(x, y)
                if res is None:
                    continue
                fits.append(
                    {
                        "Gene": g,
                        "Drug": drug,
                        "DoseUnit": normalize_unit(unit),
                        "EC50 (µM)": res["EC50"],
                        "Hill Slope": res["Hill"],
                        "R²": res["R2"],
                        "Emax": res["Emax"],
                        "Baseline": res["Baseline"],
                        "Sensitivity Index": res["SensitivityIndex"],
                    }
                )

    causal_df = pd.DataFrame(
        fits,
        columns=[
            "Gene",
            "Drug",
            "DoseUnit",
            "EC50 (µM)",
            "Hill Slope",
            "R²",
            "Emax",
            "Baseline",
            "Sensitivity Index",
        ],
    )
    if causal_df.empty:
        log("No dose–response fits found (need ≥4 distinct doses per (drug, unit)).")
    else:
        causal_df["Direction"] = np.where(
            (
                pd.to_numeric(causal_df["Emax"], errors="coerce")
                - pd.to_numeric(causal_df["Baseline"], errors="coerce")
            )
            >= 0,
            "activation_like",
            "repression_like",
        )
        causal_df = attach_gene_classes(causal_df)

    # optional relevance
    if not causal_df.empty:
        pert_type_map = (
            small_meta.drop_duplicates("pert_iname")
            .set_index("pert_iname")["pert_type"]
            .to_dict()
        )
        causal_df["Perturbation_Type"] = causal_df["Drug"].map(pert_type_map)

        def infer_rel(row):
            trend = str(row.get("Gene_DEG_Trend", "NA")).upper()
            direction = str(row.get("Direction", "NA")).lower()
            if trend == "UP" and direction == "repression_like":
                return "Reversal"
            if trend == "DOWN" and direction == "activation_like":
                return "Reversal"
            if trend in {"UP", "DOWN"} and direction in {
                "activation_like",
                "repression_like",
            }:
                return "Aggravating"
            return "Ambiguous"

        if INCLUDE_RELEVANCE:
            causal_df["Therapeutic_Relevance"] = causal_df.apply(infer_rel, axis=1)

    out_raw = output_dir / "causal_link_table_with_relevance.csv"
    causal_df.to_csv(out_raw, index=False)
    log(f"Saved {out_raw.name} ({len(causal_df)})")

    # QC + tiers
    Z_MIN, Z_MAX = -10.0, 10.0
    if not causal_df.empty:
        c = causal_df.copy()
        c["Emax_clamped"] = pd.to_numeric(c["Emax"], errors="coerce").clip(Z_MIN, Z_MAX)
        c["Baseline_clamped"] = (
            pd.to_numeric(c["Baseline"], errors="coerce").clip(Z_MIN, Z_MAX)
        )
        eps = 1e-8
        c["amp_abs"] = (c["Emax_clamped"] - c["Baseline_clamped"]).abs()
        c["SI_clamped"] = c["amp_abs"] / (pd.to_numeric(c["EC50 (µM)"], errors="coerce") + eps)

        def _dose_range(df_meta, drug, unit):
            d = df_meta[
                (df_meta["pert_iname"] == drug) & (df_meta["pert_dose_unit"] == unit)
            ]["dose_num"].dropna()
            return (float(d.min()) if len(d) else np.nan, float(d.max()) if len(d) else np.nan)

        qc_flags = []
        for _, r in c.iterrows():
            r2_ok = pd.to_numeric(r["R²"], errors="coerce") >= 0.80
            ec50 = pd.to_numeric(r["EC50 (µM)"], errors="coerce")
            lo, hi = _dose_range(small_meta, r["Drug"], r["DoseUnit"])
            in_range = (
                np.isfinite(ec50)
                and np.isfinite(lo)
                and np.isfinite(hi)
                and lo > 0
                and lo * 0.5 <= ec50 <= hi * 2.0
            )
            qc_flags.append(bool(r2_ok and in_range))
        causal_qc = c.loc[qc_flags].copy()
        if not causal_qc.empty:
            p33 = np.nanpercentile(causal_qc["SI_clamped"], 33)
            p67 = np.nanpercentile(causal_qc["SI_clamped"], 67)
            causal_qc["Sensitivity Tier (QC)"] = causal_qc["SI_clamped"].apply(
                lambda s: "High" if s >= p67 else ("Moderate" if s >= p33 else "Low")
            )
            qc_name = output_dir / "causal_link_table_qc_with_relevance.csv"
            causal_qc.to_csv(qc_name, index=False)
            log(f"Saved {qc_name.name} ({len(causal_qc)})")
    else:
        causal_qc = pd.DataFrame()

    # ---- ATE (optional; only if enabled and controls exist) ----
    if INCLUDE_ATE and not causal_df.empty and not control_meta.empty:

        def compute_stratified_ate(gene, drug, unit, treat_meta, ctl_meta, expr_sym_local):
            block_t = treat_meta[
                (treat_meta["pert_iname"] == drug)
                & (treat_meta["pert_dose_unit"] == unit)
            ]
            if block_t.empty or gene not in expr_sym_local.index:
                return np.nan, np.nan, 0, 0, 0
            strata = block_t[["cell_id", "pert_time"]].dropna().drop_duplicates()
            diffs = []
            vars_ = []
            wts = []
            n_t_total = n_c_total = n_s = 0
            for _, srow in strata.iterrows():
                cid, t = srow["cell_id"], srow["pert_time"]
                t_ids = block_t[
                    (block_t["cell_id"] == cid) & (block_t["pert_time"] == t)
                ]["sig_id"].astype(str).tolist()
                c_ids = ctl_meta[
                    (ctl_meta["cell_id"] == cid) & (ctl_meta["pert_time"] == t)
                ]["sig_id"].astype(str).tolist()
                if not t_ids or not c_ids:
                    continue
                y_t = (
                    pd.to_numeric(expr_sym_local.loc[gene, t_ids], errors="coerce")
                    .astype(float)
                    .values
                )
                y_c = (
                    pd.to_numeric(expr_sym_local.loc[gene, c_ids], errors="coerce")
                    .astype(float)
                    .values
                )
                y_t, y_c = y_t[np.isfinite(y_t)], y_c[np.isfinite(y_c)]
                nt, nc = len(y_t), len(y_c)
                if nt < 1 or nc < 1:
                    continue
                mu_t, mu_c = np.mean(y_t), np.mean(y_c)
                var_t = np.var(y_t, ddof=1) if nt > 1 else 0.0
                var_c = np.var(y_c, ddof=1) if nc > 1 else 0.0
                diffs.append(mu_t - mu_c)
                vars_.append((var_t / nt) + (var_c / nc))
                wts.append(nt)
                n_t_total += nt
                n_c_total += nc
                n_s += 1
            if n_s == 0:
                return np.nan, np.nan, 0, 0, 0
            W = np.sum(wts)
            wts = np.asarray(wts, float) / (W if W > 0 else 1.0)
            diffs = np.asarray(diffs, float)
            vars_ = np.asarray(vars_, float)
            ate = float(np.sum(wts * diffs))
            se = float(np.sqrt(np.sum((wts ** 2) * vars_)))
            return ate, se, n_t_total, n_c_total, n_s

        ate_rows = []
        for _, r in causal_df[["Gene", "Drug", "DoseUnit"]].drop_duplicates().iterrows():
            g, d, u = str(r["Gene"]), str(r["Drug"]), str(r["DoseUnit"])
            ate, se, n_t, n_c, n_s = compute_stratified_ate(
                g, d, u, small_meta, control_meta, expr_sym
            )
            if np.isfinite(ate):
                z = ate / (se + 1e-12) if np.isfinite(se) and se > 0 else np.nan
                p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
                ate_rows.append(
                    {
                        "Gene": g,
                        "Drug": d,
                        "DoseUnit": u,
                        "ATE": ate,
                        "ATE_se": se,
                        "ATE_p": p,
                        "ATE_n_treat": n_t,
                        "ATE_n_ctrl": n_c,
                        "ATE_n_strata": n_s,
                    }
                )
        ate_df = pd.DataFrame(ate_rows)
        if not ate_df.empty:
            ate_out = output_dir / "control_ate_table.csv"
            ate_df.to_csv(ate_out, index=False)
            log(f"Saved {ate_out.name} ({len(ate_df)})")
            causal_df = causal_df.merge(ate_df, on=["Gene", "Drug", "DoseUnit"], how="left")
            if isinstance(causal_qc, pd.DataFrame) and not causal_qc.empty:
                causal_qc = causal_qc.merge(
                    ate_df, on=["Gene", "Drug", "DoseUnit"], how="left"
                )
    else:
        ate_df = pd.DataFrame()

    # ---- CIS & CIS_causal ranks ----
    def _ensure_direction(df):
        if "Direction" not in df.columns:
            df["Direction"] = np.where(
                pd.to_numeric(df["Emax"], errors="coerce")
                > pd.to_numeric(df["Baseline"], errors="coerce"),
                "activation_like",
                "repression_like",
            )
        return df

    def _norm_by_gene(df, col):
        def _mm(x):
            x = pd.to_numeric(x, errors="coerce")
            rng = x.max() - x.min()
            if not np.isfinite(rng) or rng == 0:
                return pd.Series(np.zeros(len(x)), index=x.index)
            return (x - x.min()) / (rng + 1e-12)

        return df.groupby("Gene")[col].transform(_mm)

    src_for_cis = (
        causal_qc
        if (isinstance(causal_qc, pd.DataFrame) and not causal_qc.empty)
        else causal_df
    )
    if src_for_cis is None or src_for_cis.empty:
        log("[CIS] No rows for CIS. Skipping ranking/plots.")
        ranks = pd.DataFrame()
    else:
        t = _ensure_direction(src_for_cis).copy()
        emax = pd.to_numeric(t.get("Emax_clamped", t.get("Emax")), errors="coerce")
        base = pd.to_numeric(t.get("Baseline_clamped", t.get("Baseline")), errors="coerce")
        r2v = pd.to_numeric(t.get("R²"), errors="coerce")
        sival = pd.to_numeric(
            t.get("SI_clamped", t.get("Sensitivity Index")), errors="coerce"
        )
        t["amp_abs"] = (emax - base).abs()
        t["n_amp"] = _norm_by_gene(t, "amp_abs")
        t["n_r2"] = _norm_by_gene(t.assign(_R2=r2v), "_R2")
        t["si_for_cis"] = sival
        t["n_si"] = _norm_by_gene(t, "si_for_cis")
        t["CIS"] = 0.4 * t["n_amp"] + 0.2 * t["n_r2"] + 0.4 * t["n_si"]

        if "ATE" in t.columns:
            def robust_norm(vals, genes):
                out = pd.Series(index=vals.index, dtype=float)
                for g, idx in genes.groupby(genes).groups.items():
                    v = pd.to_numeric(vals.loc[idx], errors="coerce").astype(float)
                    if v.notna().sum() == 0:
                        out.loc[idx] = 0.0
                        continue
                    lo, hi = np.nanpercentile(v, [5, 95])
                    if not np.isfinite(hi - lo) or (hi - lo) < 1e-9:
                        rnk = v.rank(method="average", na_option="keep")
                        out.loc[idx] = (rnk - rnk.min()) / (rnk.max() - rnk.min() + 1e-12)
                    else:
                        out.loc[idx] = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
                return out

            abs_ate = t["ATE"].abs()
            pvals = pd.to_numeric(t.get("ATE_p"), errors="coerce")
            p_weight = (1.0 - np.minimum(1.0, pvals / 0.05)).fillna(0.0)
            t["CIS_causal"] = 0.7 * robust_norm(abs_ate, t["Gene"]) + 0.3 * p_weight
        else:
            t["CIS_causal"] = np.nan

        keep = [
            "Gene",
            "Gene_Pathway",
            "Gene_Main_Class",
            "Gene_Sub_Class",
            "Gene_Regulation",
            "Gene_DEG_Trend",
            "Drug",
            "DoseUnit",
            "Direction",
            "CIS",
            "EC50 (µM)",
            "Hill Slope",
            "R²",
            "Sensitivity Index",
            "Emax",
            "Baseline",
            "Perturbation_Type",
            "Therapeutic_Relevance",
        ]
        if "ATE" in t.columns:
            keep += [
                "CIS_causal",
                "ATE",
                "ATE_se",
                "ATE_p",
                "ATE_n_treat",
                "ATE_n_ctrl",
                "ATE_n_strata",
            ]

        t = attach_gene_classes(t)
        t = t[[c for c in keep if c in t.columns]]

        # Save causal_link_table_plus_ATE.csv with CIS included
        out_plus_ate = output_dir / "causal_link_table_plus_ATE.csv"
        t.to_csv(out_plus_ate, index=False)
        log(f"Saved {out_plus_ate.name} with CIS ({len(t)} rows)")

        if "CIS_causal" in t.columns and t["CIS_causal"].notna().any():
            ranks = (
                t.sort_values(["Gene", "CIS_causal", "CIS"], ascending=[True, False, False])
                .groupby("Gene", as_index=False)
                .head(3)
                .sort_values(["CIS_causal", "CIS"], ascending=[False, False])
            )
            out_name = "therapeutic_targets_ATE.csv"
        else:
            ranks = (
                t.sort_values(["Gene", "CIS"], ascending=[True, False])
                .groupby("Gene", as_index=False)
                .head(3)
                .sort_values("CIS", ascending=False)
            )
            out_name = "therapeutic_targets.csv"
        ranks.to_csv(output_dir / out_name, index=False)
        log(f"Saved {out_name} ({len(ranks)})")

        # ---- Plots ----
        if EXPORT_PLOTS and len(t) > 0:
            from matplotlib.backends.backend_pdf import PdfPages

            curve_dir = output_dir / "figures" / "dose_curves"
            curve_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = output_dir / "All_Dose_Response_Curves.pdf"

            plot_df = t.copy()
            plot_df["R2"] = pd.to_numeric(plot_df.get("R²", np.nan), errors="coerce")
            plot_df["EC50"] = pd.to_numeric(plot_df.get("EC50 (µM)", np.nan), errors="coerce")
            if "SI_clamped" not in plot_df.columns:
                eps = 1e-8
                emax2 = pd.to_numeric(plot_df.get("Emax"), errors="coerce")
                base2 = pd.to_numeric(plot_df.get("Baseline"), errors="coerce")
                plot_df["amp_abs"] = (emax2 - base2).abs()
                plot_df["SI_clamped"] = plot_df["amp_abs"] / (plot_df["EC50"] + eps)

            if TOP_K_PER_GENE is not None:
                plot_df = (
                    plot_df.sort_values(["Gene", "SI_clamped"], ascending=[True, False])
                    .groupby("Gene", as_index=False)
                    .head(int(TOP_K_PER_GENE))
                )

            n_saved = 0
            with PdfPages(pdf_path) as pdf:
                for _, r in plot_df.iterrows():
                    gene_name = str(r["Gene"])
                    drug_name = str(r["Drug"])
                    unit = str(r.get("DoseUnit", "uM")).replace("µ", "u").replace("μ", "u")
                    r2_val = r.get("R2")
                    ec50_val = r.get("EC50")

                    dsub = (
                        small_meta[
                            (small_meta["pert_iname"] == drug_name)
                            & (small_meta["pert_dose_unit"] == unit)
                        ]
                        .dropna(subset=["dose_num"])
                        .sort_values("dose_num")
                    )
                    if dsub.empty or gene_name not in expr_sym.index:
                        continue

                    x = dsub["dose_num"].astype(float).values
                    y = (
                        pd.to_numeric(expr_sym.loc[gene_name, dsub["sig_id"]], errors="coerce")
                        .astype(float)
                        .values
                    )

                    fres = None
                    try:
                        fres = fit_hill(x, y)
                    except Exception:
                        pass

                    plt.figure(figsize=(6.5, 4.2))
                    plt.scatter(x, y, s=40, label="observed")
                    if fres is not None and np.isfinite(fres.get("EC50", np.nan)):
                        xgrid = np.logspace(
                            np.log10(max(min(x), 1e-6)), np.log10(max(x) * 1.2), 200
                        )
                        ygrid = hill(
                            xgrid,
                            fres["EC50"],
                            fres["Hill"],
                            fres["Emax"],
                            fres["Baseline"],
                        )
                        plt.plot(xgrid, ygrid, label=f"Hill fit (R²={fres['R2']:.2f})")
                    elif pd.notna(r2_val):
                        plt.plot([], [], label=f"Hill fit (R²={r2_val:.2f})")

                    ann = []
                    if pd.notna(ec50_val):
                        ann.append(f"EC50≈{float(ec50_val):.3g} {unit}")
                    if pd.notna(r2_val):
                        ann.append(f"R²={float(r2_val):.2f}")
                    if "SI_clamped" in r and pd.notna(r["SI_clamped"]):
                        ann.append(f"SI={float(r['SI_clamped']):.2f}")
                    tier = r.get("Sensitivity Tier (QC)")
                    if isinstance(tier, str) and tier:
                        ann.append(f"Tier: {tier}")
                    if "ATE" in r and pd.notna(r.get("ATE")):
                        ann.append(f"ATE={float(r['ATE']):.2f}")
                        if pd.notna(r.get("ATE_p")):
                            ann.append(f"p={float(r['ATE_p']):.2g}")

                    plt.xscale("log")
                    plt.xlabel(f"Dose ({unit})")
                    plt.ylabel(f"{gene_name} z-score")
                    plt.title(f"{gene_name} response to {drug_name}\n" + " ; ".join(ann))
                    plt.legend()
                    plt.tight_layout()

                    safe_gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene_name)
                    safe_drug = re.sub(r"[^A-Za-z0-9_.-]+", "_", drug_name)
                    from ..plotting_utils import safe_matplotlib_savefig
                    fig = plt.gcf()
                    png_path = curve_dir / f"{safe_gene}__{safe_drug}.png"
                    if safe_matplotlib_savefig(fig, png_path, dpi=150):
                        pdf.savefig(fig)
                        n_saved += 1
                    else:
                        log(f"⚠️ Failed to save PNG for {safe_gene}__{safe_drug}, skipping PDF page")

            log(f"Saved {n_saved} PNGs in {curve_dir}")
            log(f"Saved multi-page PDF: {pdf_path}")
        else:
            log("EXPORT_PLOTS is off or there are no rows to plot — skipping plots.")

    log("Pipeline finished.")
    return True


# -------------------------------------------------------------------
# Public entry point: read userinput.txt and run everything
# -------------------------------------------------------------------
def run_drug_perturbation_from_userinput(output_dir: Path, logger: Logger):
    """
    High-level entry:
      - ensure deps & logger
      - load L1000 metadata
      - load prepared DEGs/pathways
      - parse userinput.txt
      - run pipeline
    """

    log("=== STEP 2: Drug perturbation analysis (Hill, ATE, CIS) ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR = output_dir / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = output_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    init_l1000_metadata()
    init_degs_and_pathways(output_dir)
    userinput_path = output_dir / "userinput.txt"
    cfg = load_user_config(userinput_path)

    run_pipeline(
        output_dir=output_dir,
        logger=logger,
        INCLUDE_RELEVANCE=cfg["INCLUDE_RELEVANCE"],
        INCLUDE_ATE=cfg["INCLUDE_ATE"],
        EXPORT_PLOTS=cfg["EXPORT_PLOTS"],
        TOP_K_PER_GENE=None,
        PRIMARY_SITE_FILTER=cfg["PRIMARY_SITE_FILTER"],
        CELL_WHITELIST=cfg["CELL_WHITELIST"],
        TIME_FILTERS=cfg["TIME_FILTERS"],
        MAX_SIGS=cfg["MAX_SIGS"],
        MAX_CTLS_PER_STRATUM=cfg["MAX_CTLS_PER_STRATUM"],
        force_override=True,  # always rebuild selection from config file
        auto_augment_doses=cfg["auto_augment_doses"],
        disease=None,
        drugs=cfg["drugs"],
        pert_types=cfg["pert_types"],
    )
