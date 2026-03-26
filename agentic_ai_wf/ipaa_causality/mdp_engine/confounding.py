from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, List

import pandas as pd

from mdp_engine.activity import ipaa_activity
from mdp_engine.exceptions import DataError, ValidationError
from mdp_engine.logging_utils import get_logger

log = get_logger("mdp_engine.engines.confounding")


# -----------------------------
# Minimal default cell programs
# (dependency-free, stable)
# -----------------------------
DEFAULT_CELL_PROGRAMS: Dict[str, Set[str]] = {
    "Immune": {"PTPRC", "LST1", "TYROBP", "SPI1", "FCER1G", "LYZ", "HLA-DRA", "HLA-DRB1"},
    "T_cell": {"CD3D", "CD3E", "TRAC", "TRBC1", "IL7R", "LTB", "MALAT1"},
    "B_cell": {"MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA", "CD37", "BANK1"},
    "Myeloid": {"LYZ", "S100A8", "S100A9", "FCN1", "CTSS", "LST1", "TYROBP"},
    "NK": {"NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "KLRD1", "FCGR3A"},
    "Fibroblast": {"COL1A1", "COL1A2", "DCN", "LUM", "COL3A1", "TAGLN"},
    "Endothelial": {"PECAM1", "VWF", "KDR", "RAMP2", "EMCN", "ESAM"},
    "Epithelial": {"EPCAM", "KRT8", "KRT18", "KRT19", "MSLN"},
    "CellCycle": {"MKI67", "TOP2A", "HMGB2", "CENPF", "BUB1", "TYMS"},
    "Interferon": {"ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1"},
}


@dataclass(frozen=True)
class ConfoundingPaths:
    out_dir: Path
    cell_type_scores: Path
    confounding_report: Path
    manifest: Path
    skipped_flag: Path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_tsv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=index)
    tmp.replace(path)


def _cohort_dir(out_root: Path, disease: str) -> Path:
    d1 = out_root / disease
    d2 = out_root / "cohorts" / disease
    if d1.exists():
        return d1
    if d2.exists():
        return d2
    raise DataError(f"Cannot find cohort folder for '{disease}' in {out_root} (tried {d1} and {d2}).")


def _find_expression_used(cohort: Path) -> Optional[Path]:
    for cand in ("expression_used.tsv", "expression_used.csv", "expression_used.txt"):
        p = cohort / cand
        if p.exists():
            return p
    return None


def _read_expression(expr_path: Path) -> pd.DataFrame:
    sep = "\t" if expr_path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(expr_path, sep=sep, index_col=0)
    if df.empty:
        raise DataError(f"Empty expression table: {expr_path}")

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index = df.index.astype(str).str.strip()

    # IMPORTANT: normalize gene symbols to uppercase for marker matching
    df.columns = df.columns.astype(str).str.strip().str.upper()

    return df


def _dedup_gene_columns(expr: pd.DataFrame) -> pd.DataFrame:
    """
    If duplicate gene columns exist, collapse by mean.
    This is conservative + stable.
    """
    if not expr.columns.duplicated().any():
        return expr

    log.warning("Duplicate gene columns detected; collapsing duplicates by mean.")
    df = expr.copy()
    df = df.groupby(df.columns, axis=1).mean()
    df.columns = df.columns.astype(str)
    return df


def _find_engine1_feature_matrix(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "feature_matrix.tsv"
    return p if p.exists() else None


def run_confounding_engine_for_disease(
    out_root: Path,
    disease: str,
    *,
    cell_programs: Optional[Dict[str, Set[str]]] = None,
    corr_method: str = "spearman",
    corr_flag_threshold: float = 0.40,
    min_markers: int = 5,
    feature_matrix_path: Optional[Path] = None,
    strict: bool = False,
) -> ConfoundingPaths:
    """
    Engine 2: Cell composition / program scoring + correlation penalty channel.

    Outputs:
      OUT_ROOT/engines/confounding/<Disease>/
        cell_type_scores.tsv          (samples x CELL:program)
        confounding_report.tsv        (feature penalties)
        ENGINE_MANIFEST.json
        SKIPPED.txt (if skipped)

    Behavior:
      - If expression_used missing:
          strict=False -> SKIPPED (no crash)
          strict=True  -> raise
      - If feature_matrix missing: still writes cell_type_scores; report is empty.
    """
    out_root = Path(out_root).resolve()
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    if corr_method not in {"spearman", "pearson"}:
        raise ValidationError("corr_method must be 'spearman' or 'pearson'")

    cohort = _cohort_dir(out_root, disease)

    engine_dir = out_root / "engines" / "confounding" / disease
    engine_dir.mkdir(parents=True, exist_ok=True)

    paths = ConfoundingPaths(
        out_dir=engine_dir,
        cell_type_scores=engine_dir / "cell_type_scores.tsv",
        confounding_report=engine_dir / "confounding_report.tsv",
        manifest=engine_dir / "ENGINE_MANIFEST.json",
        skipped_flag=engine_dir / "SKIPPED.txt",
    )

    expr_path = _find_expression_used(cohort)
    if expr_path is None:
        msg = f"Engine2 skipped: No expression_used.* found in {cohort}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_tsv(pd.DataFrame(), paths.cell_type_scores, index=True)
        _atomic_write_tsv(pd.DataFrame(columns=[
            "feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"
        ]), paths.confounding_report, index=False)
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "confounding",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "inputs": {"cohort_dir": str(cohort)},
            "outputs": {
                "cell_type_scores": str(paths.cell_type_scores),
                "confounding_report": str(paths.confounding_report),
                "skipped_flag": str(paths.skipped_flag),
            },
        }, indent=2))
        return paths

    expr = _dedup_gene_columns(_read_expression(expr_path))

    programs = cell_programs or DEFAULT_CELL_PROGRAMS
    prog_upper = {k: {str(g).strip().upper() for g in v} for k, v in programs.items()}

    # Scores: samples x programs
    try:
        scores = ipaa_activity(
            expression=expr,
            pathways=prog_upper,
            method="mean",
            standardize_pathways=True,
            min_size=int(min_markers),
        )
    except Exception as e:
        msg = f"Engine2 skipped: failed scoring cell programs ({type(e).__name__}): {e}"
        if strict:
            raise
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_tsv(pd.DataFrame(), paths.cell_type_scores, index=True)
        _atomic_write_tsv(pd.DataFrame(columns=[
            "feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"
        ]), paths.confounding_report, index=False)
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "confounding",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "inputs": {"expression_used": str(expr_path)},
            "outputs": {"skipped_flag": str(paths.skipped_flag)},
        }, indent=2))
        return paths

    scores.columns = [f"CELL:{c}" for c in scores.columns.astype(str)]
    _atomic_write_tsv(scores, paths.cell_type_scores, index=True)

    if feature_matrix_path is None:
        feature_matrix_path = _find_engine1_feature_matrix(out_root, disease)

    # If no feature matrix, write empty report (still valid)
    if feature_matrix_path is None:
        rep = pd.DataFrame(columns=[
            "feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"
        ])
        _atomic_write_tsv(rep, paths.confounding_report, index=False)
    else:
        feat = pd.read_csv(feature_matrix_path, sep="\t", index_col=0)
        feat.index = feat.index.astype(str).str.strip()
        feat = feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        common = scores.index.intersection(feat.index)
        if len(common) < 5:
            rep = pd.DataFrame(columns=[
                "feature", "feature_type", "max_abs_corr", "top_cell_program", "top_corr", "penalty", "flag_high"
            ])
            _atomic_write_tsv(rep, paths.confounding_report, index=False)
        else:
            S = scores.loc[common]
            F = feat.loc[common]

            corr = pd.concat([S, F], axis=1).corr(method=corr_method).loc[S.columns, F.columns]

            rows: List[Dict[str, object]] = []
            for f in corr.columns:
                cvec = corr[f].astype(float)
                if cvec.empty or cvec.isna().all():
                    continue
                top_cell = str(cvec.abs().idxmax())
                top_corr = float(cvec.loc[top_cell])
                max_abs = float(abs(top_corr))
                ftype = "TF" if str(f).startswith("TF:") else ("PW" if str(f).startswith("PW:") else "OTHER")

                rows.append({
                    "feature": str(f),
                    "feature_type": ftype,
                    "max_abs_corr": max_abs,
                    "top_cell_program": top_cell,
                    "top_corr": top_corr,
                    "penalty": max_abs,
                    "flag_high": int(max_abs >= float(corr_flag_threshold)),
                })

            rep = pd.DataFrame(rows).sort_values(["flag_high", "max_abs_corr"], ascending=[False, False]).reset_index(drop=True)
            _atomic_write_tsv(rep, paths.confounding_report, index=False)

    manifest = {
        "engine": "confounding",
        "version": "1.0.0",
        "status": "ok",
        "inputs": {
            "out_root": str(out_root),
            "disease": disease,
            "cohort_dir": str(cohort),
            "expression_used": str(expr_path),
            "feature_matrix": str(feature_matrix_path) if feature_matrix_path else None,
        },
        "params": {
            "corr_method": corr_method,
            "corr_flag_threshold": float(corr_flag_threshold),
            "min_markers": int(min_markers),
            "strict": bool(strict),
        },
        "outputs": {
            "cell_type_scores": str(paths.cell_type_scores),
            "confounding_report": str(paths.confounding_report),
        },
    }
    _atomic_write_text(paths.manifest, json.dumps(manifest, indent=2))

    if paths.skipped_flag.exists():
        try:
            paths.skipped_flag.unlink()
        except Exception:
            pass

    return paths
