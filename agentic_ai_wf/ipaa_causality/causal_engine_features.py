# File: causal_engine_features.py  (drop-in replacement for your Engine 1 script)
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import decoupler as dc  # type: ignore


from .mdp_engine.engine.evidence_bundle import build_evidence_bundle_for_disease

def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")


def _read_table(path: Path, index_col: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, index_col=index_col)
    if df is None or df.empty:
        raise ValueError(f"Empty table: {path}")
    if index_col is not None:
        df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _find_first_existing(base_dir: Path, names: Tuple[str, ...]) -> Optional[Path]:
    for n in names:
        p = base_dir / n
        if p.exists():
            return p
    return None


def resolve_ipaa_disease_dir(out_root: Path, disease: str) -> Path:
    out_root = Path(out_root)
    disease = str(disease)

    primary = out_root / disease
    alt = out_root / "cohorts" / disease

    expr_names = ("expression_used.tsv", "expression_used.csv", "expression_used.txt")
    if _find_first_existing(primary, expr_names):
        return primary
    if _find_first_existing(alt, expr_names):
        return alt

    raise FileNotFoundError(
        f"Could not find expression_used.* under {primary} or {alt}. "
        f"Expected one of: {', '.join(expr_names)}"
    )


def _looks_like_gene(x: str) -> bool:
    s = str(x).strip()
    if not s or len(s) > 30:
        return False
    if " " in s or "|" in s or "/" in s:
        return False
    alnum = sum(ch.isalnum() for ch in s)
    if alnum == 0:
        return False
    up = sum(ch.isupper() for ch in s)
    return (up / alnum) >= 0.4


def ensure_samples_by_genes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.apply(pd.to_numeric, errors="coerce")
    r, c = d.shape

    if r > 2000 and c < 2000:
        logging.info("Detected genes x samples (rows=%d, cols=%d). Transposing.", r, c)
        d = d.T
    elif c > 2000 and r < 2000:
        logging.info("Detected samples x genes (rows=%d, cols=%d).", r, c)
    else:
        idx = d.index.astype(str)
        col = d.columns.astype(str)
        n = min(len(idx), 200)
        m = min(len(col), 200)
        gene_like_idx = float(np.mean([_looks_like_gene(x) for x in idx[:n]])) if n else 0.0
        gene_like_col = float(np.mean([_looks_like_gene(x) for x in col[:m]])) if m else 0.0
        if gene_like_idx > gene_like_col:
            logging.info("Ambiguous; index looks more gene-like -> transposing.")
            d = d.T
        else:
            logging.info("Ambiguous; columns look more gene-like -> keeping as-is.")

    d.index = d.index.astype(str).str.strip()
    d.columns = d.columns.astype(str).str.strip()
    d = d.fillna(0.0)
    return d


def dedup_gene_columns(expr: pd.DataFrame) -> pd.DataFrame:
    df = expr.copy()
    df.columns = df.columns.astype(str).str.upper().str.strip()

    if not df.columns.duplicated().any():
        return df

    keep_idxs = []
    col_series = pd.Series(df.columns)
    for _, idxs in col_series.groupby(col_series).groups.items():
        idx_list = list(idxs)
        if len(idx_list) == 1:
            keep_idxs.append(idx_list[0])
        else:
            sub = df.iloc[:, idx_list]
            v = sub.var(axis=0).to_numpy(dtype=float)
            keep_idxs.append(idx_list[int(np.nanargmax(v))])

    out = df.iloc[:, keep_idxs].copy()
    out.columns = out.columns.astype(str)
    return out


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        cur = min(prev, ranked[i] * n / rank)
        q[i] = cur
        prev = cur

    out = np.empty_like(q)
    out[order] = q
    out = np.clip(out, 0.0, 1.0)
    return out


def _try_scipy_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        

        res = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        t = float(res.statistic) if res.statistic is not None and not math.isnan(float(res.statistic)) else 0.0
        p = float(res.pvalue) if res.pvalue is not None and not math.isnan(float(res.pvalue)) else 1.0
        return t, p
    except Exception:
        mx, my = float(np.nanmean(x)), float(np.nanmean(y))
        vx, vy = float(np.nanvar(x, ddof=1)), float(np.nanvar(y, ddof=1))
        nx, ny = int(np.sum(~np.isnan(x))), int(np.sum(~np.isnan(y)))
        denom = math.sqrt((vx / max(nx, 1)) + (vy / max(ny, 1))) if (nx > 1 and ny > 1) else 0.0
        t = (mx - my) / denom if denom > 0 else (mx - my)
        return float(t), 1.0


def load_labels_used(ipaa_disease_dir: Path) -> Optional[pd.Series]:
    names = ("labels_used.tsv", "labels_used.csv", "labels_used.txt")
    p = _find_first_existing(ipaa_disease_dir, names)
    if p is None:
        return None

    df = _read_table(p, index_col=0 if p.suffix.lower() != ".csv" else 0)
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        s.index = df.index.astype(str)
        return s

    low = {c.lower(): c for c in df.columns}
    for key in ("label", "group", "condition", "phenotype", "class"):
        if key in low:
            s = df[low[key]].astype(str)
            s.index = df.index.astype(str)
            return s

    s = df.iloc[:, 0].astype(str)
    s.index = df.index.astype(str)
    return s


# -----------------------------
# Overlap JSON -> TF -> n_pathways
# -----------------------------
def _walk_objs(obj: Any):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_objs(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_objs(it)


def _normalize_entity_type(t: Any) -> str:
    s = str(t or "").strip().lower()
    if s in {"tf", "tfs", "transcription_factor", "transcription factors"}:
        return "TF"
    if not s:
        return "unknown"
    return str(t).strip()


def parse_overlap_tf_counts(overlap_json: Path) -> pd.DataFrame:
    if overlap_json is None or not overlap_json.exists():
        return pd.DataFrame(columns=["entity", "n_pathways"])

    try:
        obj = json.loads(overlap_json.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=["entity", "n_pathways"])

    rows: List[Dict[str, Any]] = []

    # main common pattern
    if isinstance(obj, dict):
        pws = obj.get("pathways") or obj.get("Pathways") or obj.get("pathway_to_entities")
        if isinstance(pws, dict):
            for pw, pw_obj in pws.items():
                if not isinstance(pw_obj, dict):
                    continue
                ents = pw_obj.get("entities") or pw_obj.get("Entities") or pw_obj.get("drivers") or pw_obj.get("regulators")
                if not isinstance(ents, dict):
                    continue
                for et, et_val in ents.items():
                    if _normalize_entity_type(et) != "TF":
                        continue
                    if isinstance(et_val, dict):
                        for en in et_val.keys():
                            rows.append({"pathway": str(pw), "entity": str(en)})
                    elif isinstance(et_val, list):
                        for it in et_val:
                            if isinstance(it, dict):
                                en = it.get("name") or it.get("entity") or it.get("id") or it.get("target")
                                if en:
                                    rows.append({"pathway": str(pw), "entity": str(en)})
                            elif isinstance(it, str):
                                rows.append({"pathway": str(pw), "entity": it})

    # brute scan edges
    for node in _walk_objs(obj):
        if not isinstance(node, dict):
            continue
        pw = node.get("pathway") or node.get("Pathway") or node.get("Term") or node.get("source")
        en = node.get("entity") or node.get("Entity") or node.get("target") or node.get("name")
        et = node.get("entity_type") or node.get("type") or node.get("category") or node.get("layer")
        if pw and en and _normalize_entity_type(et) == "TF":
            rows.append({"pathway": str(pw), "entity": str(en)})

    if not rows:
        return pd.DataFrame(columns=["entity", "n_pathways"])

    df = pd.DataFrame(rows)
    df["entity"] = df["entity"].astype(str).str.replace("^TF:", "", regex=True).str.upper().str.strip()
    df["pathway"] = df["pathway"].astype(str).str.strip()

    agg = df.groupby("entity", dropna=False).agg(n_pathways=("pathway", "nunique")).reset_index()
    agg["n_pathways"] = pd.to_numeric(agg["n_pathways"], errors="coerce").fillna(0).astype(int)
    return agg


def summarize_tf_to_regulators_evidence(tf_activity: pd.DataFrame, labels: pd.Series, tf_counts: pd.DataFrame) -> pd.DataFrame:
    if tf_activity is None or tf_activity.empty:
        return pd.DataFrame(columns=["entity_type", "entity", "n_pathways", "mean_score"])

    y = labels.reindex(tf_activity.index.astype(str))
    y = y.dropna()
    if y.nunique() < 2:
        return pd.DataFrame(columns=["entity_type", "entity", "n_pathways", "mean_score"])

    counts = y.value_counts()
    gA, gB = counts.index[0], counts.index[1]
    idxA = y.index[y == gA]
    idxB = y.index[y == gB]

    X = tf_activity.loc[y.index].copy()
    colmap = {c: str(c).replace("TF:", "", 1).upper().strip() for c in X.columns}
    X = X.rename(columns=colmap)

    deltas, tvals, pvals, meansA, meansB = [], [], [], [], []
    for tf in X.columns:
        a = pd.to_numeric(X.loc[idxA, tf], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(X.loc[idxB, tf], errors="coerce").to_numpy(dtype=float)
        mA = float(np.nanmean(a)) if a.size else 0.0
        mB = float(np.nanmean(b)) if b.size else 0.0
        t, p = _try_scipy_ttest(a, b)
        meansA.append(mA)
        meansB.append(mB)
        deltas.append(mA - mB)
        tvals.append(t)
        pvals.append(p)

    p_arr = np.asarray(pvals, dtype=float)
    q_arr = bh_fdr(p_arr)

    out = pd.DataFrame(
        {
            "entity_type": "TF",
            "entity": list(X.columns.astype(str)),
            "mean_score": deltas,
            "mean_A": meansA,
            "mean_B": meansB,
            "t": tvals,
            "p_value": p_arr,
            "fdr": q_arr,
            "group_A": str(gA),
            "group_B": str(gB),
            "n_A": int(len(idxA)),
            "n_B": int(len(idxB)),
        }
    )

    # add n_pathways from overlap counts
    if tf_counts is not None and not tf_counts.empty:
        out = out.merge(tf_counts, on="entity", how="left")
        out["n_pathways"] = pd.to_numeric(out["n_pathways"], errors="coerce").fillna(0).astype(int)
    else:
        out["n_pathways"] = 0

    out = out.sort_values(["fdr", "p_value"], ascending=[True, True]).reset_index(drop=True)
    return out


# -----------------------------
# decoupler wrappers (same as yours)
# -----------------------------
def _try_import_decoupler() -> Optional[Any]:
    try:
        return dc
    except Exception as e:
        logging.warning("decoupler not installed or import failed: %s", e)
        return None


def _normalize_net(net: pd.DataFrame) -> pd.DataFrame:
    net = net.copy()
    net.columns = [str(c).strip() for c in net.columns]
    if "mor" in net.columns and "weight" not in net.columns:
        net["weight"] = net["mor"]
    if "weight" not in net.columns:
        net["weight"] = 1.0
    net["source"] = net["source"].astype(str).str.strip()
    net["target"] = net["target"].astype(str).str.upper().str.strip()
    net["weight"] = pd.to_numeric(net["weight"], errors="coerce").fillna(1.0)
    return net[["source", "target", "weight"]]


def get_regulon(dc: Any, license_mode: str = "academic") -> Optional[pd.DataFrame]:
    getters = []
    if hasattr(dc, "op") and hasattr(dc.op, "dorothea"):
        getters.append(lambda: dc.op.dorothea(organism="human", license=license_mode))
    if hasattr(dc, "get_resource"):
        getters.append(lambda: dc.get_resource("dorothea_hs", split_complexes=False))
    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        getters.append(lambda: dc.op.collectri(organism="human", license=license_mode))

    for g in getters:
        try:
            net = g()
            if isinstance(net, pd.DataFrame) and {"source", "target"}.issubset(net.columns):
                return _normalize_net(net)
        except Exception as e:
            logging.warning("Regulon fetch failed: %s", e)

    logging.warning("No regulon available via decoupler.")
    return None


def get_progeny(dc: Any) -> Optional[pd.DataFrame]:
    getters = []
    if hasattr(dc, "op") and hasattr(dc.op, "progeny"):
        getters.append(lambda: dc.op.progeny(organism="human"))
    if hasattr(dc, "get_resource"):
        getters.append(lambda: dc.get_resource("progeny"))

    for g in getters:
        try:
            net = g()
            if isinstance(net, pd.DataFrame) and {"source", "target"}.issubset(net.columns):
                return _normalize_net(net)
        except Exception as e:
            logging.warning("PROGENy fetch failed: %s", e)

    logging.warning("No PROGENy resource available via decoupler.")
    return None


def run_ulm(dc: Any, data: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    targets = set(net["target"].astype(str).str.upper())
    cols = [c for c in data.columns if c in targets]
    if len(cols) < 20:
        logging.warning("ULM: few overlapping genes (%d). Returning empty.", len(cols))
        return pd.DataFrame(index=data.index)

    X = data.loc[:, cols]

    if hasattr(dc, "mt") and hasattr(dc.mt, "ulm"):
        res = dc.mt.ulm(data=X, net=net, verbose=False)  # type: ignore
        scores = res[0] if isinstance(res, tuple) else res
        if not isinstance(scores, pd.DataFrame):
            raise RuntimeError("decoupler.mt.ulm returned unexpected type")
        scores.index = X.index
        return scores

    if hasattr(dc, "run_ulm"):
        long = dc.run_ulm(mat=X, net=net, source="source", target="target", weight="weight")  # type: ignore
        if isinstance(long, pd.DataFrame) and {"sample", "source", "es"}.issubset(long.columns):
            mat = long.pivot(index="sample", columns="source", values="es").fillna(0.0)
            mat.index = X.index.astype(str)
            return mat

    raise RuntimeError("No compatible ULM function found in installed decoupler.")


def run_viper(dc: Any, data: pd.DataFrame, net: pd.DataFrame, tmin: int = 5) -> pd.DataFrame:
    targets = set(net["target"].astype(str).str.upper())
    cols = [c for c in data.columns if c in targets]
    if len(cols) < 50:
        logging.warning("VIPER: few overlapping genes (%d). Falling back to ULM.", len(cols))
        return run_ulm(dc, data, net)

    X = data.loc[:, cols]

    if hasattr(dc, "mt") and hasattr(dc.mt, "viper"):
        res = dc.mt.viper(data=X, net=net, tmin=int(tmin), pleiotropy=False, verbose=False)  # type: ignore
        scores = res[0] if isinstance(res, tuple) else res
        if not isinstance(scores, pd.DataFrame):
            raise RuntimeError("decoupler.mt.viper returned unexpected type")
        scores.index = X.index
        return scores

    if hasattr(dc, "run_viper"):
        long = dc.run_viper(  # type: ignore
            mat=X,
            net=net,
            source="source",
            target="target",
            weight="weight",
            min_n=int(tmin),
        )
        if isinstance(long, pd.DataFrame) and {"sample", "source", "es"}.issubset(long.columns):
            mat = long.pivot(index="sample", columns="source", values="es").fillna(0.0)
            mat.index = X.index.astype(str)
            return mat

    logging.warning("No compatible VIPER; falling back to ULM.")
    return run_ulm(dc, data, net)


def run_engine1_causal_features(
    *,
    out_root: Path,
    disease: str,
    license_mode: str = "academic",
    tf_method: str = "viper",
    tmin: int = 5,
    overwrite: bool = True,
    also_write_regulators_evidence: bool = True,
    rebuild_evidence_bundle: bool = True,
) -> Dict[str, str]:
    out_root = Path(out_root)
    disease = str(disease)

    ipaa_dir = resolve_ipaa_disease_dir(out_root, disease)

    expr_path = _find_first_existing(ipaa_dir, ("expression_used.tsv", "expression_used.csv", "expression_used.txt"))
    if expr_path is None:
        raise FileNotFoundError(f"expression_used.* not found in {ipaa_dir}")

    expr_raw = _read_table(expr_path, index_col=0)
    expr = ensure_samples_by_genes(expr_raw)
    expr = dedup_gene_columns(expr)
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr.columns = expr.columns.astype(str).str.upper().str.strip()

    dc = _try_import_decoupler()

    tf_act = pd.DataFrame(index=expr.index)
    pw_act = pd.DataFrame(index=expr.index)

    regulon = None
    progeny = None

    if dc is not None:
        regulon = get_regulon(dc, license_mode=license_mode)
        progeny = get_progeny(dc)

        if regulon is not None and not regulon.empty:
            if tf_method.lower().strip() == "viper":
                tf_act = run_viper(dc, expr, regulon, tmin=int(tmin))
            else:
                tf_act = run_ulm(dc, expr, regulon)

        if progeny is not None and not progeny.empty:
            pw_act = run_ulm(dc, expr, progeny)

    tf_act = tf_act.copy()
    tf_act.index = expr.index.astype(str)
    tf_act.columns = [f"TF:{c}" for c in tf_act.columns.astype(str)]

    pw_act = pw_act.copy()
    pw_act.index = expr.index.astype(str)
    pw_act.columns = [f"PW:{c}" for c in pw_act.columns.astype(str)]

    feat = pd.concat([pw_act, tf_act], axis=1)
    feat.index.name = "sample"

    out_dir = out_root / "engines" / "causal_pathway_features" / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_path = out_dir / "tf_activity.tsv"
    pw_path = out_dir / "pathway_footprint_activity.tsv"
    fm_path = out_dir / "feature_matrix.tsv"

    if overwrite or not tf_path.exists():
        tf_act.to_csv(tf_path, sep="\t")
    if overwrite or not pw_path.exists():
        pw_act.to_csv(pw_path, sep="\t")
    if overwrite or not fm_path.exists():
        feat.to_csv(fm_path, sep="\t")

    prov = {
        "engine": "causal_pathway_features",
        "disease": disease,
        "inputs": {"ipaa_dir": str(ipaa_dir), "expression_used": str(expr_path)},
        "params": {"license_mode": license_mode, "tf_method": tf_method, "tmin": int(tmin)},
        "env": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "decoupler": getattr(dc, "__version__", None) if dc is not None else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "shapes": {
            "expression_samples_x_genes": [int(expr.shape[0]), int(expr.shape[1])],
            "tf_activity_samples_x_tfs": [int(tf_act.shape[0]), int(tf_act.shape[1])],
            "progeny_samples_x_pathways": [int(pw_act.shape[0]), int(pw_act.shape[1])],
            "feature_matrix_samples_x_features": [int(feat.shape[0]), int(feat.shape[1])],
        },
    }
    prov_path = out_dir / "feature_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    manifest = {
        "engine_id": "engine1_causal_pathway_features",
        "version": "2.0.0",
        "disease": disease,
        "ipaa_dir": str(ipaa_dir),
        "output_dir": str(out_dir),
        "artifacts": {
            "tf_activity": str(tf_path),
            "pathway_footprint_activity": str(pw_path),
            "feature_matrix": str(fm_path),
            "feature_provenance": str(prov_path),
        },
    }
    man_path = out_dir / "ENGINE_MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    artifacts = {
        "engine1_dir": str(out_dir),
        "tf_activity": str(tf_path),
        "pathway_footprint_activity": str(pw_path),
        "feature_matrix": str(fm_path),
        "feature_provenance": str(prov_path),
        "engine_manifest": str(man_path),
    }

    if also_write_regulators_evidence:
        labels = load_labels_used(ipaa_dir)
        if labels is None:
            logging.info("labels_used.* not found; skipping regulators_evidence.tsv summary.")
        else:
            # overlap json for TF->pathway mapping
            overlap_json = ipaa_dir / "overlap" / "pathway_entity_overlap.json"
            if not overlap_json.exists():
                # try disease dir search
                for cand in ipaa_dir.rglob("pathway_entity_overlap.json"):
                    overlap_json = cand
                    break

            tf_counts = parse_overlap_tf_counts(overlap_json) if overlap_json.exists() else pd.DataFrame(columns=["entity", "n_pathways"])
            reg_df = summarize_tf_to_regulators_evidence(tf_act, labels, tf_counts)

            ev_dir = out_root / "engines" / "evidence_bundle" / disease
            ev_dir.mkdir(parents=True, exist_ok=True)
            reg_path = ev_dir / "regulators_evidence.tsv"
            reg_df.to_csv(reg_path, sep="\t", index=False)
            reg_df.to_csv(out_dir / "regulators_evidence.tsv", sep="\t", index=False)
            artifacts["regulators_evidence"] = str(reg_path)

    # Optional: rebuild Engine0 evidence bundle so combined evidence appears immediately
    if rebuild_evidence_bundle:
        try:
            build_evidence_bundle_for_disease(out_root=out_root, disease=disease, q_cutoff=0.25)
            artifacts["evidence_bundle_rebuilt"] = "true"
        except Exception as e:
            logging.warning("Could not rebuild evidence_bundle automatically (%s). You can run it manually.", e)

    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="MDP Engine 1: causal pathway features (VIPER/ULM + PROGENy)")
    ap.add_argument("--out-root", required=True, help="IPAA OUT_ROOT (contains <Disease>/ or cohorts/<Disease>/)")
    ap.add_argument("--disease", required=True, help="Disease/cohort folder name")
    ap.add_argument("--license-mode", default="academic", help="decoupler license mode (academic|commercial)")
    ap.add_argument("--tf-method", default="viper", choices=["viper", "ulm"], help="TF activity method")
    ap.add_argument("--tmin", default=5, type=int, help="VIPER min targets threshold")
    ap.add_argument("--no-regulators-evidence", action="store_true", help="Do not write regulators_evidence.tsv")
    ap.add_argument("--no-rebuild-evidence-bundle", action="store_true", help="Do not rebuild Engine0 evidence bundle")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    args = ap.parse_args()

    configure_logging(args.log_level)

    artifacts = run_engine1_causal_features(
        out_root=Path(args.out_root),
        disease=str(args.disease),
        license_mode=str(args.license_mode),
        tf_method=str(args.tf_method),
        tmin=int(args.tmin),
        overwrite=True,
        also_write_regulators_evidence=not bool(args.no_regulators_evidence),
        rebuild_evidence_bundle=not bool(args.no_rebuild_evidence_bundle),
    )

    logging.info("Engine1 complete. Artifacts:")
    for k, v in artifacts.items():
        logging.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
