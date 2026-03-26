# mdp_viper_ulm.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import decoupler as dc

from .mdp_logging import info, warn, trace
from .mdp_config import CONFIG, LICENSE_MODE
from .mdp_deg_gsea import _resolve_deg_columns


def _load_regulon_from_config() -> Optional[pd.DataFrame]:
    try:
        path = (CONFIG.get("REGULON_CSV") or "").strip()
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            warn(f"[VIPER] REGULON_CSV not found: {path}")
            return None
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        need = {"source", "target"}
        if not need.issubset(set(cols)):
            raise ValueError(
                "REGULON_CSV must have columns at least: source, target [weight optional]."
            )
        if "weight" not in cols:
            df["weight"] = 1.0
            cols["weight"] = "weight"
        df = df.rename(
            columns={
                cols["source"]: "source",
                cols["target"]: "target",
                cols["weight"]: "weight",
            }
        )
        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
        return df[["source", "target", "weight"]]
    except Exception as e:
        warn(f"_load_regulon_from_config failed: {trace(e)}")
        return None

def _fetch_regulon_from_packages() -> Optional[pd.DataFrame]:
    for getter in (
        lambda: getattr(dc, "get_resource", None) and dc.get_resource("dorothea_hs", split_complexes=False),
        lambda: hasattr(dc, "op") and hasattr(dc.op, "dorothea") and dc.op.dorothea(organism="human", license=LICENSE_MODE),
        lambda: hasattr(dc, "op") and hasattr(dc.op, "collectri") and dc.op.collectri(organism="human", license=LICENSE_MODE),
    ):
        try:
            net = getter()
            if isinstance(net, pd.DataFrame) and {"source", "target"}.issubset(net.columns):
                if "weight" not in net.columns:
                    w = net["mor"] if "mor" in net.columns else 1.0
                    net = net.assign(weight=pd.to_numeric(w, errors="coerce").fillna(1.0))
                return net[["source", "target", "weight"]]
        except Exception as e:
            warn(f"regulon fetch attempt failed: {trace(e)}")
    return None

def _get_regulon() -> Optional[pd.DataFrame]:
    reg = _load_regulon_from_config()
    if reg is not None and not reg.empty:
        info("[VIPER] Using REGULON_CSV.")
        return reg
    reg = _fetch_regulon_from_packages()
    if reg is not None and not reg.empty:
        info("[VIPER] Using regulon from installed packages.")
        return reg
    warn("[VIPER] No regulon source available.")
    return None

def dedup_genes(expr: pd.DataFrame, method: str = "varmax") -> pd.DataFrame:
    expr = expr.copy()
    expr.index = expr.index.astype(str).str.strip()
    if not expr.index.duplicated().any():
        return expr
    try:
        if method == "varmax":
            tmp = expr.copy()
            tmp["_var"] = tmp.var(axis=1).values
            tmp = tmp.assign(gene=tmp.index.astype(str)).reset_index(drop=True)
            tmp = tmp.sort_values(["gene", "_var"], ascending=[True, False])
            tmp = tmp.drop_duplicates(subset="gene", keep="first")
            out = tmp.drop(columns=["_var"]).set_index("gene")
        else:
            out = expr.groupby(level=0).median()
        out = out[~out.index.duplicated(keep="first")]
        return out
    except Exception as e:
        warn(f"dedup_genes failed: {trace(e)}")
        return expr

def viper_from_counts(
    counts: pd.DataFrame,
    license_mode: str = "academic",
    tmin: int = 5,
    pleiotropy: bool = False,
) -> pd.DataFrame:
    try:
        Xg = counts.copy()
        Xg.index = Xg.index.astype(str).str.upper().str.strip()
        Xg = dedup_genes(Xg, method="varmax")
        if Xg.empty:
            return pd.DataFrame()
        net = _get_regulon()
        if net is None or net.empty:
            return pd.DataFrame()
        net = net.rename(columns={"mor": "weight"}) if "mor" in net.columns else net
        if "weight" not in net.columns:
            net["weight"] = 1.0
        X = Xg.T
        info(f"[VIPER] Running mt.viper on {X.shape} (samples x genes)")
        res = dc.mt.viper(
            data=X,
            net=net[["source", "target", "weight"]],
            tmin=tmin,
            pleiotropy=pleiotropy,
            verbose=False,
        )
        es = res[0] if isinstance(res, tuple) else res
        es.index.name = "sample"
        es.columns = [f"TF:{c}" for c in es.columns]
        return es
    except Exception as e:
        warn(f"[VIPER] compute failed: {trace(e)}")
        return pd.DataFrame()

def viper_from_degs_fallback(rnk: pd.Series) -> pd.DataFrame:
    try:
        x = pd.DataFrame({"score": rnk}).T
        x.index = ["DEG"]
        x.columns = x.columns.astype(str).str.upper().str.strip()
        net = _get_regulon()
        if net is None or net.empty:
            warn("[VIPER-DEG] No regulon available. Skipping.")
            return pd.DataFrame()
        net = net.rename(columns={"mor": "weight"}) if "mor" in net.columns else net
        if "weight" not in net.columns:
            net["weight"] = 1.0
        ulm = dc.run_ulm(mat=x, net=net, source="source", target="target", weight="weight")
        es = ulm.pivot(index="sample", columns="source", values="es").fillna(0.0)
        es.index.name = "sample"
        es.columns = [f"TF:{c}" for c in es.columns]
        return es
    except Exception as e:
        warn(f"[VIPER-DEG] fallback failed: {trace(e)}")
        return pd.DataFrame()

def _as_ulm_input_from_degs(
    df_degs: Optional[pd.DataFrame],
    lfc_col_hint: str,
) -> Optional[pd.DataFrame]:
    if df_degs is None or df_degs.empty:
        return None
    try:
        d = df_degs.copy()
        for c in d.columns:
            if d[c].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
                d[c] = d[c].astype(str)
        gcol, _, _ = _resolve_deg_columns(d, "Gene", lfc_col_hint, None)
        stat_col = None
        low = {c.lower(): c for c in d.columns}
        for k in ["stat", "t", "t_stat", "t.value", "wald", "score"]:
            if k in low:
                stat_col = low[k]
                break
        if stat_col is None:
            stat_col = _resolve_deg_columns(d, "Gene", lfc_col_hint, None)[1]
        if gcol not in d.columns or stat_col not in d.columns:
            return None
        d[gcol] = d[gcol].astype(str).str.upper().str.strip()
        d = d.dropna(subset=[gcol, stat_col])
        ser = pd.to_numeric(d[stat_col], errors="coerce")
        mat = pd.DataFrame({g: float(v) for g, v in zip(d[gcol], ser) if pd.notna(v)}, index=[0])
        mat.index = ["treatment.vs.control"]
        return mat
    except Exception as e:
        warn(f"_as_ulm_input_from_degs failed: {trace(e)}")
        return None

def run_ulm_panels(
    outdir: Path,
    df_degs: Optional[pd.DataFrame],
    lfc_col_hint: str,
    license_mode: str,
) -> None:
    
    mat = _as_ulm_input_from_degs(df_degs, lfc_col_hint)
    if mat is None or mat.shape[1] < 10:
        warn("[ULM] Not enough DE statistics to run ULM panels; skipping.")
        return
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    if CONFIG.get("RUN_ULM_TF_COLLECTRI", True):
        try:
            collectri = None
            if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
                collectri = dc.op.collectri(organism="human", license=license_mode)
            if isinstance(collectri, pd.DataFrame) and {"source", "target"}.issubset(collectri.columns):
                tf_acts, _ = dc.mt.ulm(data=mat, net=collectri)
                tf_acts.to_csv(outdir / "ulm_collectri_tf_scores.tsv", sep="\t")
                info("[ULM] CollecTRI TF panel done.")
            else:
                warn("[ULM] CollecTRI unavailable; skipping TF panel.")
        except Exception as e:
            warn(f"[ULM] CollecTRI failed: {trace(e)}")
    if CONFIG.get("RUN_ULM_PROGENY", True):
        try:
            progeny = None
            if hasattr(dc, "op") and hasattr(dc.op, "progeny"):
                progeny = dc.op.progeny(organism="human")
            if isinstance(progeny, pd.DataFrame) and {"source", "target", "weight"}.issubset(progeny.columns):
                pw_acts, _ = dc.mt.ulm(data=mat, net=progeny)
                pw_acts.to_csv(outdir / "ulm_progeny_pathway_scores.tsv", sep="\t")
                info("[ULM] PROGENy panel done.")
            else:
                warn("[ULM] PROGENy unavailable; skipping.")
        except Exception as e:
            warn(f"[ULM] PROGENy failed: {trace(e)}")
    if CONFIG.get("RUN_ULM_HALLMARK", True):
        try:
            hallmark = None
            if hasattr(dc, "op") and hasattr(dc.op, "hallmark"):
                hallmark = dc.op.hallmark(organism="human")
            if isinstance(hallmark, pd.DataFrame) and {"source", "target"}.issubset(hallmark.columns):
                hm_acts, _ = dc.mt.ulm(data=mat, net=hallmark)
                hm_acts.to_csv(outdir / "ulm_hallmark_scores.tsv", sep="\t")
                info("[ULM] Hallmark panel done.")
            else:
                warn("[ULM] Hallmark unavailable; skipping.")
        except Exception as e:
            warn(f"[ULM] Hallmark failed: {trace(e)}")
