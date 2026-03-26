#!/usr/bin/env python3
"""
enzyme_and_signaling.py

OmniPath PTM + Intercell integration (CLI version, with global cache).

What this script does for EACH cohort folder:
- Reads a DE table from the cohort directory:
    degs_from_counts.csv  OR  DEGs.csv  OR  DEGs.tsv
- Computes a rank_score per gene:
    rank_score = sign(log2FC) * -log10(padj or pvalue)

PTM block (enzyme activity):
- Fetches/caches OmniPath PTM (enzyme→substrate relationships).
- Builds enzyme→substrate sets.
- Computes KSEA-style NES for enzymes based on DE ranks.
- Writes:
    <cohort_dir>/Enzyme_and_Signaling/<cohort_name>_PTM_kinase_activity.csv

Intercell block (ligand/receptor/etc. roles):
- Fetches/caches OmniPath intercell annotations (global cache shared by all cohorts).
- Matches intercell roles to DE genes.
- Aggregates roles per gene.
- Writes:
    <cohort_dir>/Enzyme_and_Signaling/<cohort_name>_Intercell_roles.csv

Global caching
--------------

- PTM and intercell both use ONE shared cache directory per run tree:
    BATCH mode:
        global cache = <parent_input>/OmniPath_cache
    SINGLE mode:
        global cache = <parent_of_cohort>/OmniPath_cache

- Once PTM / intercell are downloaded once, all cohorts in that tree reuse:
    <OmniPath_cache>/omnipath_ptm.csv
    <OmniPath_cache>/omnipath_intercell.csv

CLI usage
---------

1) Single cohort folder:

    python enzyme_and_signaling.py \
        --input /mnt/d/Counts_Out_Single_Samples/DiseaseA

2) Batch mode on a parent folder (many diseases):

    python enzyme_and_signaling.py \
        --input /mnt/d/Counts_Out_Single_Samples \
        --batch

In batch mode, it will process **each subfolder** of --input
*if and only if* that subfolder contains a DE table.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

try:
    import requests  # HTTP fetch for OmniPath
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# ========================= Config Dataclasses ===========================


@dataclass
class PTMConfig:
    """
    Configuration for PTM-based enzyme activity scoring.
    """
    cache_dir: Path
    out_dir: Path
    cohort_name: str
    # DE table columns
    id_col: str = "Gene"
    lfc_col: str = "log2FoldChange"
    p_col: str = "pvalue"
    padj_col: str = "padj"
    # scoring parameters
    min_substrates: int = 5
    n_perm: int = 200
    random_seed: int = 17


@dataclass
class IntercellConfig:
    """
    Configuration for OmniPath intercell role annotation.
    """
    cache_dir: Path
    out_dir: Path
    cohort_name: str
    # DE table columns
    id_col: str = "Gene"
    lfc_col: str = "log2FoldChange"
    p_col: str = "pvalue"
    padj_col: str = "padj"
    # Networking
    allow_http: bool = True
    http_timeout: int = 45


# ============================== IO Helpers ==============================


def _ensure_dir(p: Path) -> None:
    """Ensure that directory `p` exists."""
    p.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV/TSV file into a DataFrame.

    - If file doesn't exist or reading fails, returns None.
    - Separator is auto-detected by pandas.
    """
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            logging.warning(f"[io] failed reading {p}: {e}")
    return None


def _safe_read_table(p: Path) -> Optional[pd.DataFrame]:
    """
    Safely read a cached table (CSV or TSV), based on file suffix.
    """
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in {".tsv", ".txt"}:
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
        return df
    except Exception as e:
        logging.warning(f"[io] failed reading cached table {p}: {e}")
        return None


# ====================== Shared DE Rank Computation ======================


def _compute_rank_from_de(
    de: pd.DataFrame,
    id_col: str,
    lfc_col: str,
    p_col: str,
    padj_col: str,
    tag: str = "rank",
) -> pd.Series:
    """
    Compute a simple signed rank score from a DE table:

        rank_score = sign(log2FC) * -log10(padj or pvalue)

    - Uses padj if available, else pvalue if available.
    - If neither is present, falls back to log2FC alone.
    - Returned Series is indexed by gene IDs (from id_col).
    """
    df = de.copy()
    for col in [id_col, lfc_col]:
        if col not in df.columns:
            raise ValueError(f"[{tag}] DE table missing required column: {col}")

    use_p = padj_col if padj_col in df.columns else p_col if p_col in df.columns else None

    if use_p is None:
        logging.warning(f"[{tag}] no p/padj column; using log2FC only for ranking")
        score = df[lfc_col].fillna(0.0)
    else:
        import numpy as np
        pvals = df[use_p].clip(lower=1e-300)
        score = (-np.log10(pvals)) * df[lfc_col].apply(lambda x: 1.0 if x >= 0 else -1.0)

    score.index = df[id_col].astype(str).values
    return score


# ============================ PTM Fetching ==============================


def _fetch_ptm_via_client() -> Optional[pd.DataFrame]:
    """Try to fetch PTM data via the OmniPath Python client."""
    try:
        from omnipath import interactions  # type: ignore
        df = interactions.get_ptms()
        return df
    except Exception as e:
        logging.info(f"[ptm] OmniPath client fetch failed, will try HTTP: {e}")
        return None


def _fetch_ptm_via_http() -> Optional[pd.DataFrame]:
    """
    Fetch PTM data directly via HTTP from OmniPath:

        https://omnipathdb.org/ptms?format=tsv / csv
    """
    if requests is None:
        logging.warning("[ptm] requests is not available; cannot use HTTP fallback")
        return None

    urls = [
        "https://omnipathdb.org/ptms?format=tsv",
        "https://omnipathdb.org/ptms?format=csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            from io import StringIO as _SIO

            buf = _SIO(r.text)
            if url.endswith("tsv"):
                df = pd.read_csv(buf, sep="\t")
            else:
                df = pd.read_csv(buf)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                logging.info(f"[ptm] fetched PTM via HTTP: {url}")
                return df
        except Exception as e:
            logging.debug(f"[ptm] fetch failed {url}: {e}")
    logging.warning("[ptm] all HTTP PTM fetch attempts failed")
    return None


def load_omnipath_ptm(cache_dir: Path) -> pd.DataFrame:
    """
    Load OmniPath PTM table, with caching.

    - Looks for <cache_dir>/omnipath_ptm.csv.
    - If present and non-empty, uses that.
    - Otherwise, tries client, then HTTP.
    - On success, writes omnipath_ptm.csv to cache_dir.
    """
    _ensure_dir(cache_dir)
    cache_fp = cache_dir / "omnipath_ptm.csv"

    if cache_fp.exists():
        df = _safe_read_table(cache_fp)
        if df is not None and len(df) > 0:
            logging.info(f"[ptm] using cached PTM: {cache_fp}")
            return df
        logging.warning(f"[ptm] cached PTM empty/unreadable: {cache_fp}")

    df = _fetch_ptm_via_client()
    if df is not None and len(df) > 0:
        df.to_csv(cache_fp, index=False)
        logging.info(f"[ptm] cached PTM via client: {cache_fp}")
        return df

    df = _fetch_ptm_via_http()
    if df is not None and len(df) > 0:
        df.to_csv(cache_fp, index=False)
        logging.info(f"[ptm] cached PTM via HTTP: {cache_fp}")
        return df

    logging.warning("[ptm] could not fetch PTM; returning empty DataFrame")
    return pd.DataFrame()


# =================== Build Enzyme→Substrate Sets (PTM) ==================


def _standardize_ptm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize PTM DataFrame to have columns: enzyme, substrate, residue."""
    cols = {c.lower(): c for c in df.columns}
    enzyme = (
        cols.get("enzyme")
        or cols.get("enzyme_gene")
        or cols.get("enzyme_genesymbol")
        or "enzyme"
    )
    substrate = (
        cols.get("substrate")
        or cols.get("substrate_gene")
        or cols.get("substrate_genesymbol")
        or "substrate"
    )
    residue = cols.get("residue") if "residue" in cols else None

    out = pd.DataFrame()
    out["enzyme"] = df[enzyme].astype(str)
    out["substrate"] = df[substrate].astype(str)
    if residue and residue in df.columns:
        out["residue"] = df[residue].astype(str)
    else:
        out["residue"] = ""
    return out


def build_enzyme_sets(ptm_df: pd.DataFrame,
                      site_level: bool = False) -> Dict[str, Set[str]]:
    """
    Build a mapping: enzyme -> set of substrates.
    """
    if ptm_df.empty:
        return {}

    pdf = _standardize_ptm_columns(ptm_df).dropna(subset=["enzyme", "substrate"])
    subs: Dict[str, Set[str]] = {}
    for _, row in pdf.iterrows():
        enz = row["enzyme"].strip()
        sub = row["substrate"].strip()
        tok = f"{sub}@{row['residue']}" if (site_level and row.get("residue")) else sub
        if not enz or not sub:
            continue
        subs.setdefault(enz, set()).add(tok)
    return subs


# ========================= KSEA-like NES Scoring ========================


def _mean_on_set(series: pd.Series, items: Set[str]) -> float:
    """Compute mean of `series` restricted to `items`."""
    s = series.reindex(list(items)).dropna()
    if s.empty:
        return 0.0
    return float(s.mean())


def _permute_nes(series: pd.Series,
                 items: Set[str],
                 n_perm: int,
                 rng: random.Random) -> Tuple[float, float, float]:
    """
    Build a null distribution for the mean of a set via permutation.
    """
    raw = _mean_on_set(series, items)
    n = len(items)
    if n == 0 or series.empty:
        return raw, 0.0, 1.0

    values = series.dropna().values
    k = min(n, len(values))
    rand_means: List[float] = []
    for _ in range(n_perm):
        idx = rng.sample(range(len(values)), k)
        rand_means.append(float(values[idx].mean()))

    import numpy as np
    mu = float(np.mean(rand_means)) if rand_means else 0.0
    sd = float(np.std(rand_means)) if rand_means else 1.0
    sd = sd if sd > 1e-8 else 1.0
    return raw, mu, sd


def score_enzyme_activity(rank: pd.Series,
                          enz_sets: Dict[str, Set[str]],
                          min_substrates: int = 5,
                          n_perm: int = 200,
                          seed: int = 17) -> pd.DataFrame:
    """
    Score enzyme activity using a KSEA-like normalized enrichment score (NES).
    """
    rng = random.Random(seed)
    rows = []

    for enz, subs in enz_sets.items():
        if len(subs) < min_substrates:
            continue
        raw, mu, sd = _permute_nes(rank, subs, n_perm=n_perm, rng=rng)
        nes = (raw - mu) / sd if sd > 1e-8 else 0.0
        rows.append({
            "enzyme": enz,
            "substrates": len(subs),
            "raw_mean": raw,
            "mu_null": mu,
            "sd_null": sd,
            "NES": nes,
        })

    if not rows:
        logging.warning("[ptm] no enzymes passed min_substrates filter")
        return pd.DataFrame(columns=["enzyme", "substrates",
                                     "raw_mean", "mu_null",
                                     "sd_null", "NES"])

    df = pd.DataFrame(rows).sort_values("NES", ascending=False).reset_index(drop=True)
    return df


def run_ptm_activity(cohort_dir: Path,
                     cfg: PTMConfig,
                     de: pd.DataFrame,
                     rank: pd.Series) -> None:
    """Run PTM activity block for a single cohort."""
    _ensure_dir(cfg.out_dir)

    ptm_df = load_omnipath_ptm(cfg.cache_dir)
    enz_sets = build_enzyme_sets(ptm_df, site_level=False)
    if not enz_sets:
        logging.info(f"[ptm] no enzyme sets available for cohort={cfg.cohort_name}; skipping")
        return

    act = score_enzyme_activity(
        rank,
        enz_sets,
        min_substrates=cfg.min_substrates,
        n_perm=cfg.n_perm,
        seed=cfg.random_seed,
    )

    if act.empty:
        logging.info(f"[ptm] activity table empty for cohort={cfg.cohort_name}")
        return

    out_csv = cfg.out_dir / f"{cfg.cohort_name}_PTM_kinase_activity.csv"
    act.to_csv(out_csv, index=False)
    logging.info(f"[ptm] wrote PTM activity: {out_csv}")


# ========================= Intercell Fetching ===========================


def _fetch_intercell_via_http(timeout: int) -> Optional[pd.DataFrame]:
    """Fetch intercell data directly via HTTP from OmniPath."""
    if requests is None:
        logging.warning("[intercell] requests not available; cannot use HTTP")
        return None

    url = "https://omnipathdb.org/intercell?format=tsv"
    try:
        logging.info(f"[intercell] HTTP download: {url}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        buf = StringIO(r.text)
        df = pd.read_csv(buf, sep="\t")
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            logging.info("[intercell] HTTP download successful")
            return df
        logging.warning("[intercell] HTTP download returned empty DataFrame")
    except Exception as e:
        logging.warning(f"[intercell] HTTP download failed: {e}")
    return None


def load_omnipath_intercell(cfg: IntercellConfig) -> pd.DataFrame:
    """
    Load OmniPath intercell table, with caching.

    Order (global cache dir):
    1) cached CSV:  <cache_dir>/omnipath_intercell.csv
    2) cached TSV:  <cache_dir>/omnipath_intercell.tsv
    3) HTTP (if allow_http)
    4) else: empty DataFrame
    """
    cache_dir = cfg.cache_dir
    _ensure_dir(cache_dir)
    cache_csv = cache_dir / "omnipath_intercell.csv"
    cache_tsv = cache_dir / "omnipath_intercell.tsv"

    if cache_csv.exists():
        df = _safe_read_table(cache_csv)
        if df is not None and len(df) > 0:
            logging.info(f"[intercell] using cached intercell (csv): {cache_csv}")
            return df
        logging.warning(f"[intercell] cached intercell csv empty/unreadable: {cache_csv}")

    if cache_tsv.exists():
        df = _safe_read_table(cache_tsv)
        if df is not None and len(df) > 0:
            logging.info(f"[intercell] using cached intercell (tsv): {cache_tsv}")
            return df
        logging.warning(f"[intercell] cached intercell tsv empty/unreadable: {cache_tsv}")

    if cfg.allow_http:
        logging.info("[intercell] no local cache, trying HTTP download...")
        df = _fetch_intercell_via_http(timeout=cfg.http_timeout)
        if df is not None and len(df) > 0:
            df.to_csv(cache_csv, index=False)
            logging.info(f"[intercell] cached intercell via HTTP: {cache_csv}")
            return df

    logging.warning(
        "[intercell] WARNING: no intercell data available "
        "(no local cache and HTTP failed)."
    )
    return pd.DataFrame()


def _standardize_intercell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize intercell DataFrame to a simpler schema:

        Gene, category, parent, generic_cat, secreted, pm_transmem, pm_peripheral
    """
    if df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}

    gene_col = (
        cols.get("genesymbol")
        or cols.get("gene_symbol")
        or cols.get("target_genesymbol")
        or cols.get("source_genesymbol")
        or cols.get("uniprot")
    )
    if gene_col is None:
        raise ValueError("[intercell] could not find a gene symbol column in intercell DF")

    category_col = cols.get("category")
    parent_col = cols.get("parent")
    generic_col = cols.get("generic_category") or cols.get("generic_cat")

    secreted_col = cols.get("secreted")
    pm_tm_col = cols.get("plasma_membrane_transmembrane")
    pm_per_col = cols.get("plasma_membrane_peripheral")

    out = pd.DataFrame()
    out["Gene"] = df[gene_col].astype(str)

    def _safe_get(col_name: Optional[str]) -> pd.Series:
        if col_name and col_name in df.columns:
            return df[col_name].astype(str)
        return pd.Series([""] * len(df), index=df.index)

    out["category"] = _safe_get(category_col)
    out["parent"] = _safe_get(parent_col)
    out["generic_cat"] = _safe_get(generic_col)
    out["secreted"] = _safe_get(secreted_col)
    out["pm_transmem"] = _safe_get(pm_tm_col)
    out["pm_peripheral"] = _safe_get(pm_per_col)

    return out


def run_intercell_roles(cohort_dir: Path,
                        cfg: IntercellConfig,
                        de: pd.DataFrame,
                        rank: pd.Series) -> None:
    """
    Intercell runner, assuming DE table and rank are already computed.

    If intercell data is unavailable:
    - Writes a DE-only CSV with rank_score (no intercell columns).
    """
    _ensure_dir(cfg.out_dir)

    intercell_df = load_omnipath_intercell(cfg)

    if intercell_df.empty:
        logging.warning(
            f"[intercell] intercell data unavailable for cohort={cfg.cohort_name}; "
            "writing DE-only file."
        )
        out = de.copy()
        out["rank_score"] = out[cfg.id_col].astype(str).map(rank.to_dict())
        out_csv = cfg.out_dir / f"{cfg.cohort_name}_Intercell_roles.csv"
        out.to_csv(out_csv, index=False)
        logging.info(f"[intercell] wrote DE-only intercell file: {out_csv}")
        return

    ic_std = _standardize_intercell(intercell_df)
    gene_universe: Set[str] = set(rank.index)
    ic_std = ic_std[ic_std["Gene"].isin(gene_universe)].copy()
    if ic_std.empty:
        logging.warning(
            f"[intercell] no DE genes with intercell roles for cohort={cfg.cohort_name}; "
            "writing DE-only file."
        )
        out = de.copy()
        out["rank_score"] = out[cfg.id_col].astype(str).map(rank.to_dict())
        out_csv = cfg.out_dir / f"{cfg.cohort_name}_Intercell_roles.csv"
        out.to_csv(out_csv, index=False)
        logging.info(f"[intercell] wrote DE-only intercell file: {out_csv}")
        return

    def _agg_unique(series: pd.Series) -> str:
        vals = sorted({v for v in series.astype(str).tolist() if v and v != "nan"})
        return ";".join(vals)

    grouped = ic_std.groupby("Gene").agg({
        "category": _agg_unique,
        "parent": _agg_unique,
        "generic_cat": _agg_unique,
        "secreted": _agg_unique,
        "pm_transmem": _agg_unique,
        "pm_peripheral": _agg_unique,
    }).reset_index()

    grouped["rank_score"] = grouped["Gene"].map(rank.to_dict())
    if cfg.lfc_col in de.columns:
        grouped["log2FoldChange"] = grouped["Gene"].map(
            de.set_index(cfg.id_col)[cfg.lfc_col].to_dict()
        )
    if cfg.padj_col in de.columns:
        grouped["padj"] = grouped["Gene"].map(
            de.set_index(cfg.id_col)[cfg.padj_col].to_dict()
        )

    out_csv = cfg.out_dir / f"{cfg.cohort_name}_Intercell_roles.csv"
    grouped.to_csv(out_csv, index=False)
    logging.info(f"[intercell] wrote intercell roles: {out_csv}")


# ========================== Cohort Processing ===========================


def _find_de_table(cohort_dir: Path) -> Optional[Path]:
    """Find a DE file in the cohort directory."""
    candidates = [
        cohort_dir / "degs_from_counts.csv",
        cohort_dir / "DEGs.csv",
        cohort_dir / "DEGs.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def process_one_cohort(cohort_dir: Path, cache_dir: Path) -> None:
    """
    Run PTM + intercell for a single cohort directory.

    Expects:
        cohort_dir/
            degs_from_counts.csv or DEGs.csv or DEGs.tsv
    Outputs:
        cohort_dir/Enzyme_and_Signaling/...
    """
    cohort_dir = cohort_dir.resolve()
    cohort_name = cohort_dir.name

    logging.info(f"[main] processing cohort: {cohort_name} ({cohort_dir})")

    de_path = _find_de_table(cohort_dir)
    if de_path is None:
        logging.warning(f"[main] no DE table found in {cohort_dir}; skipping.")
        return

    de = _safe_read_csv(de_path)
    if de is None or de.empty:
        logging.warning(f"[main] DE table at {de_path} is empty/unreadable; skipping.")
        return

    logging.info(f"[main] using DE file: {de_path}")

    out_dir = cohort_dir / "Enzyme_and_Signaling"
    _ensure_dir(out_dir)

    # Rank once, reuse for PTM + intercell
    rank = _compute_rank_from_de(
        de,
        id_col="Gene",
        lfc_col="log2FoldChange",
        p_col="pvalue",
        padj_col="padj",
        tag="main",
    )

    ptm_cfg = PTMConfig(
        cache_dir=cache_dir,
        out_dir=out_dir,
        cohort_name=cohort_name,
        id_col="Gene",
        lfc_col="log2FoldChange",
        p_col="pvalue",
        padj_col="padj",
        min_substrates=5,
        n_perm=200,
        random_seed=17,
    )

    inter_cfg = IntercellConfig(
        cache_dir=cache_dir,
        out_dir=out_dir,
        cohort_name=cohort_name,
        id_col="Gene",
        lfc_col="log2FoldChange",
        p_col="pvalue",
        padj_col="padj",
        allow_http=True,
        http_timeout=45,
    )

    # PTM block
    try:
        run_ptm_activity(cohort_dir, ptm_cfg, de, rank)
    except Exception as e:
        logging.error(f"[main] PTM activity failed for {cohort_name}: {e}", exc_info=True)

    # Intercell block
    try:
        run_intercell_roles(cohort_dir, inter_cfg, de, rank)
    except Exception as e:
        logging.error(f"[main] Intercell roles failed for {cohort_name}: {e}", exc_info=True)

    logging.info(f"[main] finished cohort: {cohort_name}")


# ================================ CLI ==================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniPath PTM + intercell analysis on one or many cohorts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to a cohort folder (single mode) or to a parent folder "
            "containing many cohort subfolders (batch mode)."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If set, treat --input as parent directory and process ALL subfolders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logging.error(f"[main] input path does not exist: {input_path}")
        return

    if args.batch:
        if not input_path.is_dir():
            logging.error("[main] --batch requires that --input is a directory.")
            return

        # One global cache for all cohorts under this parent
        global_cache_dir = input_path / "OmniPath_cache"
        _ensure_dir(global_cache_dir)

        logging.info(f"[main] batch mode on parent folder: {input_path}")
        subdirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
        if not subdirs:
            logging.warning("[main] no subdirectories found in input; nothing to do.")
            return

        for sub in subdirs:
            try:
                process_one_cohort(sub, global_cache_dir)
            except Exception as e:
                logging.error(f"[main] ERROR processing cohort folder {sub}: {e}", exc_info=True)

        logging.info("[main] batch processing finished.")
    else:
        # Single mode: --input is one cohort folder (or possibly a DE file)
        if input_path.is_file():
            cohort_dir = input_path.parent
        else:
            cohort_dir = input_path

        if not cohort_dir.is_dir():
            logging.error(f"[main] input is not a directory: {cohort_dir}")
            return

        # Global cache just above this cohort (reused across runs if same parent)
        global_cache_dir = cohort_dir.parent / "OmniPath_cache"
        _ensure_dir(global_cache_dir)

        process_one_cohort(cohort_dir, global_cache_dir)

    logging.info("[main] all done.")


if __name__ == "__main__":
    main()
