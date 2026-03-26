# mdp_enrichr.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re
import time
import numpy as np
import pandas as pd
import gseapy as gp

from .mdp_config import CONFIG
from .mdp_logging import info, warn, err, trace
from .mdp_io import save_barh
from .mdp_config import CONFIG

_HUMAN_EXCLUDE_RX = re.compile(
    r"mouse|rat|zebrafish|fly|worm|arabidopsis|plant|yeast|ecoli|bacteria|dmel|mmus|rnor|drosophila|saccharomyces",
    re.IGNORECASE,
)

def enrichr_lib_names() -> list[str]:
    try:
        import gseapy as gp
        names = gp.get_library_name()
        return list(names) if names else []
    except Exception as e:
        warn(f"Could not list Enrichr libraries (maybe offline): {trace(e)}")
        return []

def _human_only_filter(names: list[str]) -> list[str]:
    if not CONFIG.get("HUMAN_ONLY_ORA", True):
        return names
    return [n for n in names if not _HUMAN_EXCLUDE_RX.search(n)]

def filter_libs_for_ORA(libs: list[str]) -> list[str]:
    avail = set(enrichr_lib_names())
    if not avail:
        warn("No Enrichr library list. Proceeding with human-filtered requested libs.")
        return _human_only_filter(libs)
    libs = _human_only_filter(libs)
    keep = [x for x in libs if x in avail]
    miss = [x for x in libs if x not in avail]
    if miss:
        warn(f"Skipping unavailable libs: {miss}")
    return keep

def _latest_helper(names: list[str], patterns: list[str], max_libs: int) -> list[str]:
    def year_from_name(n: str) -> int:
        ys = [int(x) for x in re.findall(r"(20\d{2}|19\d{2})", n)]
        return max(ys) if ys else -1
    chosen, seen = [], set()
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        sub = [n for n in names if rx.search(n)]
        sub.sort(key=lambda n: (year_from_name(n), n), reverse=True)
        for n in sub:
            if n not in seen:
                chosen.append(n)
                seen.add(n)
            if len(chosen) >= max_libs:
                break
        if len(chosen) >= max_libs:
            break
    return chosen

def latest_libs_by_patterns_any(patterns: list[str], max_libs: int = 6) -> list[str]:
    names = enrichr_lib_names()
    if not names:
        return []
    return _latest_helper(names, patterns, max_libs)

def latest_libs_by_patterns_human(patterns: list[str], max_libs: int = 6) -> list[str]:
    names = _human_only_filter(enrichr_lib_names())
    if not names:
        return []
    return _latest_helper(names, patterns, max_libs)

def enrichr_ora_human(gene_list: list[str], libs: list[str]) -> pd.DataFrame:
    try:
        libs = filter_libs_for_ORA(libs)
        if not libs or not gene_list:
            return pd.DataFrame(
                columns=["library", "term", "pval", "qval",
                         "odds_ratio", "combined_score", "genes"]
            )
        rows = []
        for lib in libs:
            t0 = time.time()
            local_sets = None
            mode = "api"
            try:
                # Try to use cached GMT (local mode avoids Enrichr API calls / timeouts)
                local_sets = gp.get_library(name=lib)
                if isinstance(local_sets, dict) and local_sets:
                    mode = "local"
                else:
                    local_sets = None
            except Exception as e:
                warn(f"get_library failed for {lib}: {trace(e)}; falling back to API")

            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=local_sets if local_sets is not None else [lib],
                    organism="Human",
                    outdir=None,
                    cutoff=1.0,
                    no_plot=True,
                    verbose=False,
                )
                if enr is None or enr.results is None or enr.results.empty:
                    info(f"[ORA] {lib}: no results (mode={mode}, {time.time()-t0:.1f}s)")
                    continue
                df = enr.results.rename(
                    columns={c: c.replace(" ", "_") for c in enr.results.columns}
                )
                for _, r in df.iterrows():
                    rows.append([
                        lib,
                        r.get("Term"),
                        r.get("P-value"),
                        r.get("Adjusted_P-value"),
                        r.get("Odds_Ratio", np.nan),
                        r.get("Combined_Score", np.nan),
                        r.get("Genes", ""),
                    ])
                info(f"[ORA] {lib}: rows={len(df)} (mode={mode}, {time.time()-t0:.1f}s)")
            except Exception as e:
                warn(f"enrichr failed for {lib}: {trace(e)}")
        out = pd.DataFrame(
            rows,
            columns=["library", "term", "pval", "qval",
                     "odds_ratio", "combined_score", "genes"],
        )
        if not out.empty:
            out = out.sort_values(
                ["library", "qval", "pval"],
                na_position="last"
            ).reset_index(drop=True)
        return out
    except Exception as e:
        err(f"enrichr_ora_human: {trace(e)}")
        return pd.DataFrame(
            columns=["library", "term", "pval", "qval",
                     "odds_ratio", "combined_score", "genes"]
        )

def fetch_gene_sets_dict_any(libs: list[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lib in libs:
        try:
            gsd = gp.get_library(name=lib)
            for term, genes in gsd.items():
                out[f"{lib}:{term}"] = [
                    str(g).upper().strip()
                    for g in genes if isinstance(g, str)
                ]
        except Exception as e:
            warn(f"get_library failed for {lib}: {trace(e)}")
    return out

def gsea_prerank(
    rnk: pd.Series,
    libs: list[str],
    outdir: Path,
    figdpi: int,
) -> tuple[pd.DataFrame, str]:
    

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        if rnk is None or rnk.dropna().shape[0] < 20:
            warn("[GSEA] Ranked list too small (<20). Falling back to ORA.")
            genes = [g for g in (rnk.index.tolist() if rnk is not None else [])]
            df = enrichr_ora_human(genes, libs)
            tsv = outdir / "gsea_ORA_fallback.tsv"
            df.to_csv(tsv, sep="\t", index=False)
            return df, str(tsv)

        gsets = fetch_gene_sets_dict_any(libs)
        if not gsets:
            warn("[GSEA] Could not fetch gene sets. Falling back to ORA.")
            df = enrichr_ora_human(rnk.index.tolist(), libs)
            tsv = outdir / "gsea_ORA_fallback.tsv"
            df.to_csv(tsv, sep="\t", index=False)
            return df, str(tsv)

        rnk_df = rnk.dropna().reset_index()
        rnk_df.columns = ["gene", "score"]

        procs = int(CONFIG.get("GSEA_PROCESSES", 1))
        perms = int(CONFIG.get("GSEA_PERMUTATIONS", 100))
        info(f"[GSEA] prerank: genes={rnk_df.shape[0]} | procs={procs} | perms={perms}")
        prer = gp.prerank(
            rnk=rnk_df,
            gene_sets=gsets,
            min_size=10,
            max_size=2000,
            processes=procs,
            permutation_num=perms,
            seed=7,
            outdir=None,
            verbose=False,
        )
        res = prer.res2d.copy()

        if "ledge_genes" in res.columns and "Term" in res.columns:
            res = res.rename(columns={"Term": "term"})
            res["lead_genes"] = res["ledge_genes"].fillna("").astype(str)
        elif "term" not in res.columns and "Term" in res.columns:
            res = res.rename(columns={"Term": "term"})
        if "fdr" in res.columns and "FDR" not in res.columns:
            res["FDR"] = res["fdr"]
        if "nes" in res.columns and "NES" not in res.columns:
            res["NES"] = res["nes"]

        tsv = outdir / "gsea_prerank.tsv"
        res.to_csv(tsv, sep="\t", index=False)
        if "FDR" in res.columns and "term" in res.columns:
            save_barh(
                res,
                "term",
                "FDR",
                "GSEA — Top pathways",
                outdir / "plots/top_pathways.png",
                top_n=CONFIG["TOP_N_PATHWAYS"],
                dpi=figdpi,
            )
        info("[GSEA] prerank done.")
        return res, str(tsv)
    except Exception as e:
        warn(f"[GSEA] prerank failed: {trace(e)}. Falling back to ORA.")
        df = enrichr_ora_human(rnk.index.tolist(), libs)
        tsv = outdir / "gsea_ORA_fallback.tsv"
        try:
            df.to_csv(tsv, sep="\t", index=False)
        except Exception:
            pass
        return df, str(tsv)

def leading_edge_from_gsea(res: pd.DataFrame) -> dict[str, list[str]]:
    le: dict[str, list[str]] = {}
    try:
        if res is None or res.empty:
            return le
        for _, r in res.iterrows():
            term = r.get("term") or r.get("Term")
            if not term:
                continue
            lg = r.get("lead_genes") or r.get("ledge_genes") or r.get("lead_genes_list")
            if pd.isna(lg) or not isinstance(lg, str):
                continue
            genes = [g.strip().upper() for g in re.split(r"[;,]", lg) if g.strip()]
            if genes:
                le[term] = genes
        return le
    except Exception as e:
        warn(f"leading_edge_from_gsea failed: {trace(e)}")
        return {}
