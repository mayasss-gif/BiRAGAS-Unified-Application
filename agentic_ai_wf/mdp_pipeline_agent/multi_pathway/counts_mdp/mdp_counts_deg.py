# mdp_counts_deg.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri as p2
from scipy.stats import ttest_ind

from .mdp_logging import info, warn, err, trace
from .mdp_io import _load_any_table
from .mdp_config import CONFIG

_GENE_HINTS = [
    "gene", "genes", "symbol", "gene_symbol",
    "hgnc_symbol", "ensembl", "ensembl_id", "id",
]

def _pick_gene_col(df: pd.DataFrame) -> str:
    low = {c.lower(): c for c in df.columns}
    for k in _GENE_HINTS:
        if k in low:
            return low[k]
    cand = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return cand[0] if cand else df.columns[0]

def _standardize_gene_ids(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.\d+$", "", regex=True)
    is_ens = x.str.match(r"ENSG\d+", na=False)
    x = np.where(is_ens, x, x.str.upper())
    return pd.Series(x, index=s.index)

def _coerce_counts_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
    return out

def _load_counts_file_as_matrix(p: Path) -> pd.DataFrame:
    df = _load_any_table(p)
    if df.empty:
        return pd.DataFrame()
    try:
        gcol = _pick_gene_col(df)
        df[gcol] = _standardize_gene_ids(df[gcol])
        file_tag = p.stem.replace(" ", "_")
        num_cols = [c for c in df.columns if c != gcol and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            mat = (
                df[[gcol] + num_cols]
                .dropna(subset=[gcol])
                .drop_duplicates(subset=[gcol])
                .set_index(gcol)
            )
            mat.columns = [f"{file_tag}__{str(c)}" for c in mat.columns]
            return _coerce_counts_numeric(mat)
        non_gene_cols = [c for c in df.columns if c != gcol]
        if not non_gene_cols:
            sample_name = f"{file_tag}__sample"
            mat = df[[gcol]].copy().assign(**{sample_name: 1.0})
            return mat.set_index(gcol)
        best, best_sd = None, -1
        for c in non_gene_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            sd = v.std(skipna=True)
            if sd > best_sd:
                best, best_sd = c, sd
        sample_name = f"{file_tag}__{best}"
        mat = (
            df[[gcol, best]]
            .rename(columns={best: sample_name})
            .dropna(subset=[gcol])
            .drop_duplicates(subset=[gcol])
            .set_index(gcol)
        )
        return _coerce_counts_numeric(mat)
    except Exception as e:
        warn(f"_load_counts_file_as_matrix skip {p.name}: {trace(e)}")
        return pd.DataFrame()

def _merge_counts_matrices(mats: List[pd.DataFrame]) -> pd.DataFrame:
    out = None
    for m in mats:
        if m is None or m.empty:
            continue
        out = m if out is None else out.join(m, how="outer")
    if out is None:
        return pd.DataFrame()
    out = out.fillna(0.0)
    nz = out.sum(axis=1) > 0
    return out.loc[nz]

def _base_sample_id(col: str) -> str:
    c = str(col)
    return c.split("__", 1)[-1] if "__" in c else c

def _collapse_technical_replicates(counts: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
    if counts is None or counts.empty:
        return counts
    try:
        groups: Dict[str, List[str]] = {}
        for col in counts.columns:
            base = _base_sample_id(col)
            groups.setdefault(base, []).append(col)
        pieces = []
        for base, cols in groups.items():
            sub = counts[cols]
            if len(cols) == 1:
                agg = sub.rename(columns={cols[0]: base})
            else:
                if how == "sum":
                    agg = sub.sum(axis=1).to_frame(name=base)
                elif how == "mean":
                    agg = sub.mean(axis=1).to_frame(name=base)
                else:
                    raise ValueError(f"Unknown TECH_REP_METHOD: {how}")
            pieces.append(agg)
        out = pd.concat(pieces, axis=1)
        out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return out
    except Exception as e:
        warn(f"_collapse_technical_replicates failed: {trace(e)}")
        return counts

def _drop_identical_count_columns(counts: pd.DataFrame) -> pd.DataFrame:
    if counts is None or counts.empty or counts.shape[1] <= 1:
        return counts
    try:
        col_hash = counts.apply(
            lambda s: pd.util.hash_pandas_object(s, index=True).values.sum(),
            axis=0,
        )
        keep_mask = ~col_hash.duplicated()
        kept = counts.loc[:, keep_mask.values]
        dropped = counts.shape[1] - kept.shape[1]
        if dropped > 0:
            info(f"[counts] dropped {dropped} duplicate columns (identical content).")
        return kept
    except Exception as e:
        warn(f"_drop_identical_count_columns failed: {trace(e)}")
        return counts

_META_REGEX_CTRL = re.compile(r"(control|normal|healthy)", re.IGNORECASE)

def _detect_groups_from_names(samples: List[str]) -> pd.Series:
    lab = []
    for s in samples:
        lab.append("control" if _META_REGEX_CTRL.search(s) else "case")
    return pd.Series(lab, index=samples, dtype="object")

def _load_metadata_if_present(counts_dir: Path, coh: dict) -> Optional[pd.DataFrame]:
    meta_name = coh.get("metadata", None)
    try:
        if meta_name:
            mp = Path(meta_name) if Path(meta_name).is_absolute() else counts_dir / meta_name
            if mp.exists():
                return _load_any_table(mp)
        for fn in [
            "metadata.csv", "metadata.tsv", "metadata.txt",
            "samples.csv", "samples.tsv", "samples.xlsx",
        ]:
            mp = counts_dir / fn
            if mp.exists():
                return _load_any_table(mp)
        return None
    except Exception as e:
        warn(f"_load_metadata_if_present failed: {trace(e)}")
        return None

def _groups_from_metadata(meta: pd.DataFrame, samples: List[str], coh: dict) -> Optional[pd.Series]:
    try:
        sc = coh.get("sample_col") or "sample"
        gc = coh.get("group_col") or "group"
        low = {c.lower(): c for c in meta.columns}
        sc = low.get(sc.lower(), next(iter(low.values())))
        gc = low.get(gc.lower(), next(iter(low.values())))
        sub = meta[[sc, gc]].copy()
        sub[sc] = sub[sc].astype(str)
        sub[gc] = sub[gc].astype(str)
        sub = sub.dropna().drop_duplicates(subset=[sc])
        m = dict(zip(sub[sc], sub[gc]))
        out = pd.Series([m.get(s, None) for s in samples], index=samples, dtype="object").str.lower()
        case_label = (coh.get("case_label") or "case").lower()
        ctrl_label = (coh.get("control_label") or "control").lower()
        uniq = set(out.dropna().unique().tolist())
        if case_label in uniq and ctrl_label in uniq:
            out = out.map({case_label: "case", ctrl_label: "control"}).astype("object")
        return out
    except Exception as e:
        warn(f"_groups_from_metadata failed: {trace(e)}")
        return None

def _fallback_balance_group(samples: List[str]) -> pd.Series:
    n = len(samples)
    half = n // 2
    return pd.Series(["case"] * half + ["control"] * (n - half), index=samples, dtype="object")

# -----------------------------
# NEW: robust per-file meta pairing (only affects grouping)
# -----------------------------

_SAMPLE_SANITIZE_RE = re.compile(r"[^\w\-\.]+")

def _sanitize_sample_name(x: str) -> str:
    return _SAMPLE_SANITIZE_RE.sub("_", str(x))

_ALLOWED_EXTS = {".csv", ".tsv", ".txt", ".xlsx"}

def _find_counts_files_with_suffix(counts_dir: Path) -> List[Path]:
    """Return files matching '*_counts.<ext>' anywhere under counts_dir."""
    hits: List[Path] = []
    for p in counts_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _ALLOWED_EXTS and p.stem.lower().endswith("_counts"):
            hits.append(p)
    return sorted(hits)

def _find_any_count_like_files_current(counts_dir: Path) -> List[Path]:
    """Current behavior: all supported table files anywhere under counts_dir."""
    return sorted([p for p in counts_dir.rglob("*") if p.suffix.lower() in _ALLOWED_EXTS])

def _pick_paired_meta_file(counts_file: Path) -> Optional[Path]:
    """
    For '<id>_counts.ext', look for '<id>_meta.<ext>' in the SAME folder as counts_file.
    """
    stem = counts_file.stem
    if not stem.lower().endswith("_counts"):
        return None
    base_id = stem[:-len("_counts")]
    parent = counts_file.parent
    candidates: List[Path] = []
    for ext in _ALLOWED_EXTS:
        mp = parent / f"{base_id}_meta{ext}"
        if mp.exists() and mp.is_file():
            candidates.append(mp)
    if not candidates:
        return None
    # deterministic choice if multiple exist
    candidates = sorted(candidates, key=lambda x: x.suffix.lower())
    if len(candidates) > 1:
        warn(f"[groups] multiple meta files found for {counts_file.name}; using {candidates[0].name}")
    return candidates[0]

def _groups_from_perfile_meta(meta: pd.DataFrame, sample_names: List[str]) -> Optional[pd.Series]:
    """
    meta must have columns: sample, condition (case-insensitive).
    condition is mapped using the SAME control/normal/healthy regex rule:
      - matches -> 'control'
      - else -> 'case'
    sample values are matched robustly against:
      - exact sample column
      - base sample id (strip prefix before '__')
      after sanitization (same as counts.columns sanitization).
    """
    try:
        if meta is None or meta.empty:
            return None
        low = {c.lower(): c for c in meta.columns}
        if "sample" not in low or "condition" not in low:
            return None

        sc = low["sample"]
        cc = low["condition"]

        sub = meta[[sc, cc]].copy()
        sub[sc] = sub[sc].astype(str).map(_sanitize_sample_name)
        sub[cc] = sub[cc].astype(str)

        sub = sub.dropna().drop_duplicates(subset=[sc])

        # Build mapping: sample -> case/control using SAME rule as before
        cond_to_group: Dict[str, str] = {}
        for s, cond in zip(sub[sc].tolist(), sub[cc].tolist()):
            grp = "control" if _META_REGEX_CTRL.search(str(cond)) else "case"
            cond_to_group[s] = grp

        # Now map to provided sample_names
        out_vals: List[Optional[str]] = []
        for col in sample_names:
            col_s = _sanitize_sample_name(col)
            base = _sanitize_sample_name(_base_sample_id(col))
            grp = cond_to_group.get(col_s)
            if grp is None:
                grp = cond_to_group.get(base)
            out_vals.append(grp)

        return pd.Series(out_vals, index=sample_names, dtype="object")
    except Exception as e:
        warn(f"_groups_from_perfile_meta failed: {trace(e)}")
        return None

def _run_deseq2_with_rpy2(counts: pd.DataFrame, groups: pd.Series) -> Optional[pd.DataFrame]:
    
    try:
        C = counts.copy()
        C = C.applymap(lambda x: max(0, int(round(float(x)))))
        cond = groups.loc[C.columns].astype(str).tolist()
        r = ro.r
        with localconverter(pandas2ri.converter):
            c_r = ro.conversion.py2rpy(C)
        r.assign("count_df", c_r)
        r.assign("conds", ro.StrVector(cond))
        r("suppressMessages(library(DESeq2))")
        r("coldata <- data.frame(row.names=colnames(count_df), condition = factor(conds))")
        r("dds <- DESeqDataSetFromMatrix(countData = count_df, colData = coldata, design = ~ condition)")
        r("dds$condition <- relevel(dds$condition, ref='control')")
        r("dds <- DESeq(dds)")
        r("rn <- resultsNames(dds)")
        r('coef_name <- rn[grep("^condition.*case.*vs.*control$", rn)][1]')
        r(
            "if (is.na(coef_name) || coef_name=='') { "
            "cand <- rn[grep('condition', rn)]; "
            "coef_name <- ifelse(length(cand)>0, cand[length(cand)], rn[length(rn)]) "
            "}"
        )
        r(
            "res <- tryCatch({ suppressMessages(library(apeglm)); "
            "lfcShrink(dds, coef=coef_name, type='apeglm') }, "
            "error=function(e) NULL)"
        )
        r("if (is.null(res)) { res <- lfcShrink(dds, coef=coef_name, type='normal') }")
        res = r("as.data.frame(res)")
        
        with localconverter(p2.converter):
            df = ro.conversion.rpy2py(res)
        df = df.reset_index().rename(columns={"index": "Gene"})
        low = {c.lower(): c for c in df.columns}
        out = df.rename(
            columns={
                low.get("log2foldchange", "log2FoldChange"): "log2FoldChange",
                low.get("padj", "padj"): "padj",
                low.get("pvalue", "pvalue"): "pvalue",
                low.get("stat", "stat"): "stat",
            }
        )
        out["Gene"] = _standardize_gene_ids(out["Gene"])
        return out
    except Exception as e:
        warn(f"DESeq2 pipeline failed: {trace(e)}")
        return None

def _run_simple_ttest_fallback(counts: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    try:
        lib = counts.sum(axis=0)
        X = counts.div(lib, axis=1) * 1e6
        X = np.log2(X + 1.0)
        case_cols = groups[groups == "case"].index
        ctrl_cols = groups[groups == "control"].index
        if len(case_cols) == 0 or len(ctrl_cols) == 0:
            raise RuntimeError("Welch fallback needs at least one case and one control.")
        A = X[case_cols].to_numpy(dtype=float)
        B = X[ctrl_cols].to_numpy(dtype=float)
        tstat, pval = ttest_ind(A, B, axis=1, equal_var=False, nan_policy="omit")
        lfc = np.nanmean(A, axis=1) - np.nanmean(B, axis=1)
        out = pd.DataFrame(
            {"Gene": X.index.to_numpy(),
             "log2FoldChange": lfc,
             "stat": tstat,
             "pvalue": pval}
        )
        
        mask = np.isfinite(out["pvalue"].to_numpy())
        padj = np.full(out.shape[0], np.nan)
        if mask.sum() > 0:
            padj[mask] = multipletests(out.loc[mask, "pvalue"], method="fdr_bh")[1]
        out["padj"] = padj
        return out
    except Exception as e:
        err(f"Welch fallback failed: {trace(e)}")
        return pd.DataFrame(columns=["Gene", "log2FoldChange", "stat", "pvalue", "padj"])

def build_degs_from_counts_dir(
    coh: dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Path, Path, pd.Series]:
    counts_dir = Path(coh.get("counts_dir"))
    assert counts_dir.exists(), f"counts_dir not found: {counts_dir}"

    # NEW: if any *_counts.<ext> exists, use ONLY those files for this run;
    # otherwise keep the current behavior (all supported table files).
    counts_suffix_files = _find_counts_files_with_suffix(counts_dir)
    if counts_suffix_files:
        files = counts_suffix_files
        info(f"[counts] detected {len(files)} '*_counts.*' files; using only these for counts input.")
    else:
        files = _find_any_count_like_files_current(counts_dir)

    if not files:
        raise FileNotFoundError(f"No count-like files found in {counts_dir}")

    mats: List[pd.DataFrame] = []

    # NEW: collect per-file meta-derived group hints (sample/base -> case/control)
    perfile_group_map: Dict[str, str] = {}

    for p in files:
        try:
            m = _load_counts_file_as_matrix(p)
            if m.empty:
                continue

            # Per-file meta pairing only applies when *_counts.* naming convention is used
            if counts_suffix_files and p.stem.lower().endswith("_counts"):
                mp = _pick_paired_meta_file(p)
                if mp is not None:
                    meta = _load_any_table(mp)
                    g = _groups_from_perfile_meta(meta, m.columns.tolist())
                    if g is None:
                        warn(f"[groups] {mp.name} present but invalid (needs columns: sample, condition).")
                    else:
                        # Store mapping by BOTH full col and base id (sanitized) for robustness
                        for col, grp in g.items():
                            if grp is None or (isinstance(grp, float) and np.isnan(grp)):
                                continue
                            col_s = _sanitize_sample_name(col)
                            base_s = _sanitize_sample_name(_base_sample_id(col))
                            # if conflict, keep first and warn (deterministic)
                            if col_s in perfile_group_map and perfile_group_map[col_s] != grp:
                                warn(f"[groups] conflict for sample '{col_s}' between meta files; keeping first.")
                            else:
                                perfile_group_map[col_s] = grp
                            if base_s in perfile_group_map and perfile_group_map[base_s] != grp:
                                warn(f"[groups] conflict for base sample '{base_s}' between meta files; keeping first.")
                            else:
                                perfile_group_map[base_s] = grp
                # if no per-file meta exists, we do nothing here; fallback remains unchanged later

            mats.append(m)

        except Exception as e:
            warn(f"counts skip {p.name}: {trace(e)}")

    counts = _merge_counts_matrices(mats)
    if counts.empty:
        raise RuntimeError("Merged counts matrix is empty.")
    counts.columns = [re.sub(r"[^\w\-\.]+", "_", str(c)) for c in counts.columns]

    before_cols = counts.shape[1]
    if CONFIG.get("COLLAPSE_TECH_REPS", True):
        counts = _collapse_technical_replicates(counts, how=CONFIG.get("TECH_REP_METHOD", "sum"))
    if CONFIG.get("DROP_IDENTICAL_COLUMNS", True):
        counts = _drop_identical_count_columns(counts)
    after_cols = counts.shape[1]
    if after_cols < before_cols:
        info(f"[counts] columns reduced {before_cols} → {after_cols} (collapse + dedup).")
    info(f"[counts] samples after cleanup: n={counts.shape[1]}")

    # -----------------------------
    # GROUP SPLITTING (ONLY SECTION CHANGED)
    # -----------------------------
    samples = counts.columns.tolist()

    if counts_suffix_files:
        # Start with NA groups, fill from per-file meta map where possible
        groups = pd.Series([None] * len(samples), index=samples, dtype="object")
        for s in samples:
            s_san = _sanitize_sample_name(s)
            base_san = _sanitize_sample_name(_base_sample_id(s))
            grp = perfile_group_map.get(s_san)
            if grp is None:
                grp = perfile_group_map.get(base_san)
            if grp is not None:
                groups.loc[s] = grp

        # For anything still unresolved, use the ORIGINAL workflow (global metadata if present else name heuristic),
        # but ONLY to fill missing values (no overrides).
        unresolved = groups[groups.isna()].index.tolist()
        if unresolved:
            meta = _load_metadata_if_present(counts_dir, coh)
            if meta is not None and not meta.empty:
                g2 = _groups_from_metadata(meta, unresolved, coh)
                if g2 is not None and not g2.empty:
                    groups.loc[unresolved] = g2
            # if still NA, fall back to name heuristic for those
            unresolved2 = groups[groups.isna()].index.tolist()
            if unresolved2:
                g3 = _detect_groups_from_names(unresolved2)
                groups.loc[unresolved2] = g3

    else:
        # EXACT ORIGINAL behavior when no *_counts.* files exist
        meta = _load_metadata_if_present(counts_dir, coh)
        if meta is not None and not meta.empty:
            groups = _groups_from_metadata(meta, samples, coh)
        else:
            groups = _detect_groups_from_names(samples)

    if groups.isna().any():
        unresolved = groups[groups.isna()].index.tolist()
        if unresolved:
            warn(f"[DEG] Could not infer groups for samples {unresolved}; balanced split.")
            groups.loc[unresolved] = _fallback_balance_group(unresolved)

    if (groups == "case").sum() < 1 or (groups == "control").sum() < 1:
        warn("[DEG] minimal group sizes; DE may be unstable.")

    degs = _run_deseq2_with_rpy2(counts, groups)
    if degs is None or degs.empty:
        info("[DEG] Falling back to simple Welch t-test pipeline.")
        degs = _run_simple_ttest_fallback(counts, groups)

    disease_dir = Path(CONFIG["OUT_ROOT"]) / (coh.get("name") or "disease")
    disease_dir.mkdir(parents=True, exist_ok=True)
    combined_counts_path = disease_dir / "combined_counts.csv"
    degs_out_path = disease_dir / "degs_from_counts.csv"
    try:
        counts.reset_index().rename(columns={"index": "Gene"}).to_csv(combined_counts_path, index=False)
        degs.to_csv(degs_out_path, index=False)
        info(f"[DEG] wrote merged counts: {combined_counts_path}")
        info(f"[DEG] wrote DEGs: {degs_out_path}")
    except Exception as e:
        warn(f"Failed writing counts/DEGs files: {trace(e)}")
    return degs, counts, combined_counts_path, degs_out_path, groups
