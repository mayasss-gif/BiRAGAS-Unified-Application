#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import gseapy as gp  # type: ignore
# ----------------------------
# Logging
# ----------------------------
def setup_logging(verbosity: int = 0) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


log = logging.getLogger("mdp_engine0_evidence_bundle")


# ----------------------------
# Manifest / Summary
# ----------------------------
@dataclass
class EngineManifest:
    engine: str
    version: str
    generated_at: str
    out_root: str
    disease: str
    disease_dir_used: str
    inputs_used: Dict[str, str]
    outputs_written: Dict[str, str]
    warnings: List[str]


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf in {".csv"}:
        return pd.read_csv(path)
    # fallback: try tab, then csv
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def find_first_existing(d: Path, candidates: List[str]) -> Optional[Path]:
    """
    Case-insensitive search for any candidate file within a directory.
    """
    if not d.exists() or not d.is_dir():
        return None
    name_map = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
    for c in candidates:
        p = name_map.get(c.lower())
        if p is not None and p.exists():
            return p
    return None


def find_any_matching(d: Path, contains_any: List[str]) -> Optional[Path]:
    """
    Return first file whose lowercase name contains ANY of the substrings.
    """
    if not d.exists() or not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        n = p.name.lower()
        if any(s.lower() in n for s in contains_any):
            return p
    return None


# ----------------------------
# Normalization helpers
# ----------------------------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in cols:
            return cols[o.lower()]
    return None


def normalize_gene_evidence(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Output columns (best-effort):
      gene, log2fc, stat, p_value, fdr, score
    Keeps extra columns too, but ensures at least 'gene'.
    """
    warnings: List[str] = []
    df = _clean_cols(df)

    gene_col = _pick_col(df, ["gene", "genes", "symbol", "hgnc", "gene_symbol", "gene name"])
    if gene_col is None:
        # sometimes index-like column exists
        # try first column if it looks like gene symbols
        gene_col = df.columns[0]
        warnings.append(f"Could not find gene column by name; using first column: {gene_col}")

    out = df.copy()
    out.rename(columns={gene_col: "gene"}, inplace=True)
    out["gene"] = out["gene"].astype(str).str.strip()
    out = out[out["gene"] != ""].copy()

    # Normalize common numeric fields
    log2fc_col = _pick_col(out, ["log2fc", "logfc", "lfc", "log_fold_change", "log2foldchange"])
    stat_col = _pick_col(out, ["stat", "t", "t_stat", "t_like", "score", "wald_stat", "test_statistic"])
    p_col = _pick_col(out, ["p", "pvalue", "p_value", "p-value"])
    fdr_col = _pick_col(out, ["fdr", "padj", "q", "qvalue", "adj_p", "adjusted_p_value", "bh_fdr"])

    if log2fc_col and log2fc_col != "log2fc":
        out.rename(columns={log2fc_col: "log2fc"}, inplace=True)
    if stat_col and stat_col not in {"stat", "score"}:
        out.rename(columns={stat_col: "stat"}, inplace=True)
    elif stat_col == "score":
        out.rename(columns={stat_col: "stat"}, inplace=True)

    if p_col and p_col != "p_value":
        out.rename(columns={p_col: "p_value"}, inplace=True)
    if fdr_col and fdr_col != "fdr":
        out.rename(columns={fdr_col: "fdr"}, inplace=True)

    # Coerce numeric columns if present
    for c in ["log2fc", "stat", "p_value", "fdr"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Add a generic "score" if missing, derived from stat or log2fc
    if "score" not in out.columns:
        if "stat" in out.columns:
            out["score"] = out["stat"]
        elif "log2fc" in out.columns:
            out["score"] = out["log2fc"]
        else:
            out["score"] = pd.NA
            warnings.append("No stat/log2fc found; score column set to NA.")

    # Drop duplicate genes keeping best (largest |score|)
    out["__abs_score__"] = pd.to_numeric(out["score"], errors="coerce").abs()
    out.sort_values("__abs_score__", ascending=False, inplace=True, na_position="last")
    out = out.drop_duplicates(subset=["gene"], keep="first").drop(columns=["__abs_score__"])

    return out.reset_index(drop=True), warnings


def load_prerank(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Accepts 2-column prerank: gene \t score
    """
    warnings: List[str] = []
    df = read_table(path)
    df = _clean_cols(df)

    # allow "gene, score" or first two columns
    gene_col = _pick_col(df, ["gene", "genes", "symbol"]) or df.columns[0]
    score_col = _pick_col(df, ["score", "stat", "t", "rank", "ranking", "log2fc"]) or (df.columns[1] if len(df.columns) > 1 else None)

    if score_col is None:
        raise ValueError(f"Prerank file {path} doesn't have a usable score column.")

    out = df[[gene_col, score_col]].copy()
    out.rename(columns={gene_col: "gene", score_col: "score"}, inplace=True)
    out["gene"] = out["gene"].astype(str).str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["gene", "score"])
    out = out[out["gene"] != ""].copy()

    # Drop duplicates by |score|
    out["__abs__"] = out["score"].abs()
    out.sort_values("__abs__", ascending=False, inplace=True)
    out = out.drop_duplicates(subset=["gene"], keep="first").drop(columns=["__abs__"])

    # optional: derive sign columns
    out["direction"] = out["score"].apply(lambda x: "UP" if x > 0 else ("DOWN" if x < 0 else "ZERO"))

    return out.reset_index(drop=True), warnings


def normalize_pathway_evidence(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = _clean_cols(df)

    pw_col = _pick_col(df, ["pathway", "term", "name", "geneset", "set", "id"])
    if pw_col is None:
        pw_col = df.columns[0]
        warnings.append(f"Could not find pathway column by name; using first column: {pw_col}")

    out = df.copy()
    out.rename(columns={pw_col: "pathway"}, inplace=True)
    out["pathway"] = out["pathway"].astype(str).str.strip()
    out = out[out["pathway"] != ""].copy()

    # normalize common stats
    p_col = _pick_col(out, ["p", "pvalue", "p_value", "p-value"])
    fdr_col = _pick_col(out, ["fdr", "padj", "q", "qvalue", "adj_p", "adjusted_p_value", "bh_fdr"])
    stat_col = _pick_col(out, ["t_like", "stat", "score", "nes", "es", "z", "delta_mean"])

    if p_col and p_col != "p_value":
        out.rename(columns={p_col: "p_value"}, inplace=True)
    if fdr_col and fdr_col != "fdr":
        out.rename(columns={fdr_col: "fdr"}, inplace=True)
    if stat_col and stat_col != "stat":
        out.rename(columns={stat_col: "stat"}, inplace=True)

    for c in ["p_value", "fdr", "stat"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop duplicates
    out = out.drop_duplicates(subset=["pathway"], keep="first")

    return out.reset_index(drop=True), warnings


def load_overlap_entities(overlap_json: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Best-effort parsing for "regulators_evidence.tsv" from overlap JSON.
    We output columns:
      entity_type, entity, n_pathways, mean_score
    If schema doesn't match, we return an empty df with warnings.
    """
    warnings: List[str] = []
    try:
        obj = json.loads(overlap_json.read_text(encoding="utf-8"))
    except Exception as e:
        warnings.append(f"Failed to parse overlap JSON: {overlap_json} ({e})")
        return pd.DataFrame(columns=["entity_type", "entity", "n_pathways", "mean_score"]), warnings

    # Many possible schemas. Try common patterns.
    # Pattern A: {"pathways": { "PW1": {"entities": {"tf":[{"name":..,"score":..},..], ...}}}}
    # Pattern B: {"pathway_entity": [{"pathway":..,"entity":..,"entity_type":..,"score":..}, ...]}
    rows: List[Dict[str, object]] = []

    if isinstance(obj, dict):
        # Pattern B
        pe = obj.get("pathway_entity") or obj.get("edges") or obj.get("links")
        if isinstance(pe, list):
            for it in pe:
                if not isinstance(it, dict):
                    continue
                et = it.get("entity_type") or it.get("type") or it.get("category") or "unknown"
                en = it.get("entity") or it.get("name") or it.get("target")
                sc = it.get("score") or it.get("weight") or None
                pw = it.get("pathway") or it.get("source") or None
                if en:
                    rows.append({"entity_type": str(et), "entity": str(en), "pathway": str(pw) if pw else None, "score": sc})

        # Pattern A
        pws = obj.get("pathways") or obj.get("Pathways")
        if isinstance(pws, dict):
            for pw_name, pw_obj in pws.items():
                if not isinstance(pw_obj, dict):
                    continue
                ents = pw_obj.get("entities") or pw_obj.get("Entities")
                if not isinstance(ents, dict):
                    continue
                for et, et_list in ents.items():
                    if isinstance(et_list, list):
                        for eobj in et_list:
                            if isinstance(eobj, dict):
                                en = eobj.get("name") or eobj.get("entity") or eobj.get("id")
                                sc = eobj.get("score") or eobj.get("weight")
                                if en:
                                    rows.append({"entity_type": str(et), "entity": str(en), "pathway": str(pw_name), "score": sc})
                            elif isinstance(eobj, str):
                                rows.append({"entity_type": str(et), "entity": eobj, "pathway": str(pw_name), "score": None})

    if not rows:
        warnings.append("Could not extract entities from overlap JSON (unknown schema).")
        return pd.DataFrame(columns=["entity_type", "entity", "n_pathways", "mean_score"]), warnings

    df = pd.DataFrame(rows)
    df["entity_type"] = df["entity_type"].astype(str)
    df["entity"] = df["entity"].astype(str)

    # Aggregate to per-entity evidence summary
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce")
    agg = (
        df.groupby(["entity_type", "entity"], dropna=False)
        .agg(
            n_pathways=("pathway", lambda x: len(set([p for p in x if p and str(p) != "None"]))),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )
    agg["mean_score"] = pd.to_numeric(agg["mean_score"], errors="coerce")

    return agg, warnings


# ----------------------------
# Optional gseapy ORA (Enrichr)
# ----------------------------
def try_gseapy_enrichr(
    genes: List[str],
    enrichr_library: str,
    organism: str = "Human",
) -> Optional[pd.DataFrame]:
    

    genes = [g for g in genes if g and str(g).strip()]
    genes = sorted(set(genes))
    if not genes:
        return None

    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=enrichr_library,
            organism=organism,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
    except Exception as e:
        log.debug("gseapy.enrichr failed: %s", e)
        return None

    res = getattr(enr, "results", None)
    if res is None or not isinstance(res, pd.DataFrame) or res.empty:
        return None
    return res.copy()


# ----------------------------
# Disease discovery
# ----------------------------
def discover_disease_dirs(out_root: Path) -> Dict[str, Path]:
    """
    IPAA writes primary per-disease at:
      out_root/<Disease>/
    and also legacy at:
      out_root/cohorts/<Disease>/
    We'll discover both, prefer primary if it contains key files.
    """
    candidates: Dict[str, List[Path]] = {}

    def add_dir(d: Path) -> None:
        if not d.is_dir():
            return
        name = d.name.strip()
        if not name:
            return
        # ignore engine folders
        if name.lower() in {"engines", "cohorts", "data", "classification", "report", "results", "logs"}:
            return
        candidates.setdefault(name, []).append(d)

    # direct children
    for p in out_root.iterdir():
        if p.is_dir():
            add_dir(p)

    # cohorts/<Disease>
    cohorts_dir = out_root / "cohorts"
    if cohorts_dir.exists() and cohorts_dir.is_dir():
        for p in cohorts_dir.iterdir():
            if p.is_dir():
                candidates.setdefault(p.name.strip(), []).append(p)

    # choose best directory for each disease
    chosen: Dict[str, Path] = {}
    key_files = ["pathway_stats_with_baseline.tsv", "pathway_stats.tsv", "de_gene_stats.tsv", "prerank_all.tsv"]

    for disease, dirs in candidates.items():
        # score dirs: count key files present
        best = None
        best_score = -1
        for d in dirs:
            score = 0
            for k in key_files:
                if (d / k).exists():
                    score += 1
            # also accept any file containing pathway_stats
            if find_any_matching(d, ["pathway_stats"]) is not None:
                score += 1
            if score > best_score:
                best_score = score
                best = d
        if best is not None and best_score > 0:
            chosen[disease] = best

    return chosen


# ----------------------------
# Engine 0 builder
# ----------------------------
def build_evidence_bundle_for_disease(
    out_root: Path,
    disease: str,
    disease_dir: Path,
    enrichr_library: Optional[str],
    enrichr_top_n: int,
    organism: str,
) -> EngineManifest:
    warnings: List[str] = []
    inputs_used: Dict[str, str] = {}
    outputs_written: Dict[str, str] = {}

    engine_dir = out_root / "engines" / "evidence_bundle" / disease
    ensure_dir(engine_dir)

    # ---- genes evidence ----
    gene_stats = find_first_existing(
        disease_dir,
        ["de_gene_stats.tsv", "de_gene_stats.csv", "gene_stats.tsv", "gene_stats.csv"],
    )
    prerank = find_first_existing(
        disease_dir,
        ["prerank_all.tsv", "prerank.tsv", "prerank_all.csv", "prerank.csv", "prerank.rnk"],
    )

    genes_df: Optional[pd.DataFrame] = None

    if gene_stats and gene_stats.exists():
        raw = read_table(gene_stats)
        genes_df, w = normalize_gene_evidence(raw)
        warnings.extend(w)
        inputs_used["de_gene_stats"] = str(gene_stats)
    elif prerank and prerank.exists():
        genes_df, w = load_prerank(prerank)
        warnings.extend(w)
        inputs_used["prerank"] = str(prerank)
    else:
        warnings.append("No de_gene_stats or prerank found; genes_evidence.tsv will be empty.")
        genes_df = pd.DataFrame(columns=["gene", "score"])

    genes_out = engine_dir / "genes_evidence.tsv"
    write_tsv(genes_df, genes_out)
    outputs_written["genes_evidence"] = str(genes_out)

    # ---- pathways evidence ----
    pw_stats = find_first_existing(
        disease_dir,
        ["pathway_stats_with_baseline.tsv", "pathway_stats.tsv", "pathway_stats_with_baseline.csv", "pathway_stats.csv"],
    ) or find_any_matching(disease_dir, ["pathway_stats"])

    pw_df: Optional[pd.DataFrame] = None
    if pw_stats and pw_stats.exists():
        raw = read_table(pw_stats)
        pw_df, w = normalize_pathway_evidence(raw)
        warnings.extend(w)
        inputs_used["pathway_stats"] = str(pw_stats)
    else:
        warnings.append("No pathway_stats found; pathways_evidence.tsv will be empty.")
        pw_df = pd.DataFrame(columns=["pathway"])

    # If IPAA already produced ORA/GSEA result files, include them (best-effort)
    gsea_file = find_any_matching(disease_dir, ["gsea", "prerank"])
    ora_file = find_any_matching(disease_dir, ["ora", "enrich", "overrep"])

    if gsea_file and gsea_file.exists():
        try:
            gsea_raw = read_table(gsea_file)
            gsea_norm, _w = normalize_pathway_evidence(gsea_raw)
            gsea_norm = gsea_norm.add_prefix("gsea_")
            gsea_norm.rename(columns={"gsea_pathway": "pathway"}, inplace=True)
            pw_df = pw_df.merge(gsea_norm, on="pathway", how="left")
            inputs_used["gsea_results"] = str(gsea_file)
        except Exception as e:
            warnings.append(f"Failed to merge GSEA results from {gsea_file.name}: {e}")

    if ora_file and ora_file.exists():
        try:
            ora_raw = read_table(ora_file)
            ora_norm, _w = normalize_pathway_evidence(ora_raw)
            ora_norm = ora_norm.add_prefix("ora_")
            ora_norm.rename(columns={"ora_pathway": "pathway"}, inplace=True)
            pw_df = pw_df.merge(ora_norm, on="pathway", how="left")
            inputs_used["ora_results"] = str(ora_file)
        except Exception as e:
            warnings.append(f"Failed to merge ORA results from {ora_file.name}: {e}")

    # Optional: enrichr ORA (gseapy) if requested
    if enrichr_library:
        top_genes = genes_df.copy()
        if "score" in top_genes.columns:
            top_genes["__abs__"] = pd.to_numeric(top_genes["score"], errors="coerce").abs()
            top_genes = top_genes.sort_values("__abs__", ascending=False).dropna(subset=["__abs__"])
        gene_list = top_genes["gene"].astype(str).head(int(enrichr_top_n)).tolist()

        enr = try_gseapy_enrichr(gene_list, enrichr_library=enrichr_library, organism=organism)
        if enr is not None and not enr.empty:
            # normalize enrichr columns into something joinable if possible
            enr2 = enr.copy()
            if "Term" in enr2.columns:
                enr2.rename(columns={"Term": "pathway"}, inplace=True)
            else:
                # if no Term column, just dump separately
                enr_path = engine_dir / "enrichr_results.tsv"
                write_tsv(enr2, enr_path)
                outputs_written["enrichr_results"] = str(enr_path)
                enr2 = None

            if enr2 is not None:
                enr2 = _clean_cols(enr2)
                # keep minimal set + prefix
                keep_cols = [c for c in enr2.columns if c != "pathway"]
                enr2 = enr2[["pathway"] + keep_cols].copy()
                enr2 = enr2.add_prefix("enrichr_")
                enr2.rename(columns={"enrichr_pathway": "pathway"}, inplace=True)
                pw_df = pw_df.merge(enr2, on="pathway", how="left")
                inputs_used["enrichr_library"] = enrichr_library
        else:
            warnings.append("gseapy.enrichr requested but returned no results (or failed).")

    pw_out = engine_dir / "pathways_evidence.tsv"
    write_tsv(pw_df, pw_out)
    outputs_written["pathways_evidence"] = str(pw_out)

    # ---- regulators evidence ----
    # Prefer tf_activity.tsv if IPAA wrote it (Engine 1 later), else overlap JSON
    tf_act = find_any_matching(disease_dir, ["tf_activity", "viper", "ulm", "collectri"])
    reg_df: Optional[pd.DataFrame] = None

    if tf_act and tf_act.exists():
        try:
            reg_df = read_table(tf_act)
            reg_df = _clean_cols(reg_df)
            inputs_used["tf_activity_like"] = str(tf_act)
        except Exception as e:
            warnings.append(f"Failed to read TF activity file {tf_act.name}: {e}")

    if reg_df is None:
        # Try overlap JSON
        overlap_json = (disease_dir / "overlap" / "pathway_entity_overlap.json")
        if not overlap_json.exists():
            # also check common alternative locations
            overlap_json = find_any_matching(disease_dir, ["pathway_entity_overlap.json"]) or overlap_json

        if overlap_json and overlap_json.exists():
            reg_df, w = load_overlap_entities(overlap_json)
            warnings.extend(w)
            inputs_used["overlap_json"] = str(overlap_json)
        else:
            reg_df = pd.DataFrame(columns=["entity_type", "entity", "n_pathways", "mean_score"])
            warnings.append("No TF activity or overlap JSON found; regulators_evidence.tsv will be empty.")

    reg_out = engine_dir / "regulators_evidence.tsv"
    write_tsv(reg_df, reg_out)
    outputs_written["regulators_evidence"] = str(reg_out)

    # ---- mechanism summary ----
    summary = {
        "disease": disease,
        "inputs_used": inputs_used,
        "counts": {
            "genes": int(len(genes_df)) if genes_df is not None else 0,
            "pathways": int(len(pw_df)) if pw_df is not None else 0,
            "regulators": int(len(reg_df)) if reg_df is not None else 0,
        },
        "notes": warnings[:],
    }

    summary_path = engine_dir / "mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs_written["mechanism_summary"] = str(summary_path)

    # ---- manifest ----
    manifest = EngineManifest(
        engine="evidence_bundle",
        version="1.0.0",
        generated_at=datetime.utcnow().isoformat() + "Z",
        out_root=str(out_root),
        disease=disease,
        disease_dir_used=str(disease_dir),
        inputs_used=inputs_used,
        outputs_written=outputs_written,
        warnings=warnings,
    )
    manifest_path = engine_dir / "ENGINE_MANIFEST.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    outputs_written["engine_manifest"] = str(manifest_path)

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MDP Engine 0: Build evidence bundles from IPAA OUT_ROOT outputs."
    )
    ap.add_argument("--out-root", required=True, help="Path to IPAA OUT_ROOT")
    ap.add_argument("--enrichr-library", default=None, help="Optional: Enrichr library name (gseapy.enrichr).")
    ap.add_argument("--enrichr-top-n", type=int, default=250, help="Top N genes for enrichr (by |score|).")
    ap.add_argument("--organism", default="Human", help="Organism for gseapy.enrichr (default Human).")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    args = ap.parse_args()

    setup_logging(args.verbose)

    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists() or not out_root.is_dir():
        log.error("OUT_ROOT not found or not a directory: %s", out_root)
        return 2

    diseases = discover_disease_dirs(out_root)
    if not diseases:
        log.error("No disease/cohort folders found under OUT_ROOT: %s", out_root)
        log.error("Expected folders like OUT_ROOT/<Disease>/ or OUT_ROOT/cohorts/<Disease>/ with pathway_stats.tsv etc.")
        return 3

    log.info("Found %d disease folders.", len(diseases))

    manifests: List[EngineManifest] = []
    for disease, ddir in sorted(diseases.items(), key=lambda x: x[0].lower()):
        log.info("Building evidence bundle for: %s (dir=%s)", disease, ddir)
        try:
            m = build_evidence_bundle_for_disease(
                out_root=out_root,
                disease=disease,
                disease_dir=ddir,
                enrichr_library=args.enrichr_library,
                enrichr_top_n=args.enrichr_top_n,
                organism=args.organism,
            )
            manifests.append(m)
        except Exception as e:
            log.exception("FAILED for disease=%s: %s", disease, e)

    # Write top-level manifest index
    index_path = out_root / "engines" / "evidence_bundle" / "ENGINE_INDEX.json"
    ensure_dir(index_path.parent)
    index_obj = {
        "engine": "evidence_bundle",
        "version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "out_root": str(out_root),
        "n_diseases": len(manifests),
        "diseases": [
            {
                "disease": m.disease,
                "disease_dir_used": m.disease_dir_used,
                "engine_dir": str((out_root / "engines" / "evidence_bundle" / m.disease).resolve()),
                "warnings": m.warnings,
            }
            for m in manifests
        ],
    }
    index_path.write_text(json.dumps(index_obj, indent=2), encoding="utf-8")
    log.info("Wrote ENGINE_INDEX.json: %s", index_path)

    log.info("Done. Evidence bundles at: %s", out_root / "engines" / "evidence_bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
