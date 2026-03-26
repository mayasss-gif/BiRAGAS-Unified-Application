#!/usr/bin/env python3
"""
auto_tissue_baseline.py (IPAA)

What this does (IPAA-native):
1) Detect input kind: COUNTS vs DEGS (light heuristics).
2) Resolve tissue per disease: exact -> fuzzy -> (optional) LLM pick constrained to consensus tissues.
3) Build IPAA consensus baseline expectations for the resolved tissue (HPA/GTEx/FANTOM consensus Z).
4) Build background gene universe (only where applicable; defaults to COUNTS only).
5) Writes a cohort preparation manifest so downstream IPAA steps can consume baseline/universe/tissue.

No mdp_* imports. No counts_mdp imports.

ENV / CONFIG:
- OPENAI_API_KEY (only if you allow LLM tissue fallback)
- IPAA_TISSUE_MODEL (optional; default gpt-4.1-mini)

Baseline data:
- IPAA_BASELINE_DATA_DIR: directory holding baseline inputs
- IPAA_HPA_FILE, IPAA_GTEX_FILE, IPAA_FANTOM_FILE: filenames inside that dir
- IPAA_BASELINE_PATHWAY_LIBS: comma-separated Enrichr library names (e.g. "KEGG_2021_Human,Reactome_2022")

Baseline stack code:
This module expects the IPAA paper baseline utilities to exist as:
- baseline_expectations.py
- utils_io.py
in one of these locations:
- IPAA_BASELINE_STACK_DIR (env)
- ./IPAA_2/
- ./ (repo root)
"""

from __future__ import annotations

import csv
import difflib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
from openai import OpenAI


# =============================================================================
# Errors
# =============================================================================

class IPAAError(RuntimeError):
    pass

class TissueResolutionError(IPAAError):
    pass

class BaselineBuildError(IPAAError):
    pass

class UniverseBuildError(IPAAError):
    pass


# =============================================================================
# Logging (simple + robust)
# =============================================================================

def info(msg: str) -> None:
    print(msg)

def warn(msg: str) -> None:
    print(msg, file=sys.stderr)

def trace(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# =============================================================================
# Consensus tissues
# =============================================================================

CONSENSUS_TISSUES: List[str] = sorted({
    "adipose tissue", "adrenal gland", "amygdala", "appendix", "basal ganglia", "bone marrow", "breast",
    "cerebellum", "cerebral cortex", "cervix", "choroid plexus", "colon", "corpus callosum", "ductus deferens",
    "duodenum", "endometrium", "epididymis", "esophagus", "fallopian tube", "gallbladder", "heart muscle",
    "hippocampal formation", "hypothalamus", "kidney", "liver", "lung", "lymph node", "midbrain", "olfactory bulb",
    "ovary", "pancreas", "parathyroid gland", "pituitary gland", "placenta", "pons", "pons and medulla", "prostate",
    "rectum", "retina", "salivary gland", "seminal vesicle", "skeletal muscle", "skin", "small intestine",
    "smooth muscle", "spinal cord", "spleen", "stomach", "testis", "thalamus", "thymus", "thyroid gland",
    "tongue", "tonsil", "urinary bladder", "vagina",
})

FUZZY_MIN_SCORE = 0.86


def _norm_tissue(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _exact_or_none(tissue_in: str) -> Optional[str]:
    if not tissue_in:
        return None
    tnorm = _norm_tissue(tissue_in)
    norm_map = {_norm_tissue(t): t for t in CONSENSUS_TISSUES}
    return norm_map.get(tnorm)

def _best_fuzzy_tissue(query: str) -> Tuple[Optional[str], float]:
    if not query:
        return None, 0.0
    q = _norm_tissue(query)
    best_t = None
    best_s = 0.0
    for c in CONSENSUS_TISSUES:
        s = difflib.SequenceMatcher(None, q, _norm_tissue(c)).ratio()
        if s > best_s:
            best_t, best_s = c, s
    return best_t, best_s


def _llm_pick_tissue(disease: str, user_tissue: str, hint: str = "") -> str:
    """
    LLM fallback: MUST output exactly one tissue from CONSENSUS_TISSUES.
    """
   

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise TissueResolutionError("LLM tissue fallback requested but OPENAI_API_KEY is not set.")

    model = (os.environ.get("IPAA_TISSUE_MODEL") or "gpt-4.1-mini").strip()
    client = OpenAI(api_key=api_key)

    allowed = "\n".join(f"- {t}" for t in CONSENSUS_TISSUES)
    prompt = f"""
Select the single best-matching tissue label for an IPAA disease cohort.

Disease: {disease}
User tissue (may be empty/invalid): {user_tissue}
Hint (may include folder/file name): {hint}

Output EXACTLY ONE tissue from this list, verbatim, with NO extra text:
{allowed}
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    out = (resp.output_text or "").strip()
    if out not in CONSENSUS_TISSUES:
        raise TissueResolutionError(f"LLM returned invalid tissue '{out}'. Must be one of consensus tissues.")
    return out


def resolve_tissue_or_die(
    disease: str,
    tissue_in: str,
    hint: str = "",
    allow_llm: bool = True,
) -> str:
    tissue_in = (tissue_in or "").strip()

    exact = _exact_or_none(tissue_in)
    if exact:
        return exact

    if tissue_in:
        best, score = _best_fuzzy_tissue(tissue_in)
        if best and score >= FUZZY_MIN_SCORE:
            info(f"[tissue] '{tissue_in}' not exact; using fuzzy match '{best}' (score={score:.3f})")
            return best
        warn(f"[tissue] '{tissue_in}' not in consensus tissues; best fuzzy score={score:.3f} < {FUZZY_MIN_SCORE:.2f}")

    if not allow_llm:
        raise TissueResolutionError(
            "Tissue unresolved and LLM fallback disabled. Provide a valid tissue from consensus tissues."
        )

    picked = _llm_pick_tissue(disease=disease, user_tissue=tissue_in, hint=hint)
    info(f"[tissue] using LLM-picked tissue for '{disease}': '{picked}'")
    return picked


# =============================================================================
# Table loading (auto)
# =============================================================================

def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        return dialect.delimiter
    except Exception:
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def load_table_auto(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()

    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suf in {".csv", ".tsv", ".txt"}:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        delim = _sniff_delimiter(sample)
        return pd.read_csv(path, sep=delim, engine="python")

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        # try common shapes
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame.from_dict(obj, orient="index").reset_index().rename(columns={"index": "key"})
        return pd.DataFrame()

    # fallback
    return pd.read_csv(path, engine="python")


# =============================================================================
# Input kind detection (COUNTS vs DEGS)
# =============================================================================

_DEG_LFC_CAND = ["log2fc", "log2foldchange", "logfc", "lfc", "log2_fold_change", "logfoldchange", "log_fold_change"]
_DEG_Q_CAND = ["padj", "p_adj", "adj_p", "fdr", "q", "qval", "qvalue", "pvalue_adj", "p_value_adj", "adjp", "fdrq"]

def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", "", h)
    return h.replace("-", "").replace("_", "")

def _header_has_any(cands: Iterable[str], header: List[str]) -> bool:
    norm = [_normalize_header(h) for h in header]
    c_norm = {_normalize_header(c) for c in cands}
    return any(h in c_norm for h in norm)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "nan", "null", "none"}:
            return None
        return float(s)
    except Exception:
        return None

def detect_input_kind(path: Path) -> str:
    """
    Returns: "counts" or "degs" (default "counts" if unsure for matrices).
    """
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        # pick a likely file
        candidates: List[Path] = []
        for pat in ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"):
            candidates.extend(sorted(p.glob(pat)))
        if not candidates:
            return "counts"
        # prefer csv/tsv/txt
        candidates = sorted(candidates, key=lambda x: (x.suffix.lower() not in {".csv", ".tsv", ".txt"}, x.name.lower()))
        return detect_input_kind(candidates[0])

    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        # name hint
        if "deg" in p.name.lower():
            return "degs"
        return "counts"

    if suf not in {".csv", ".tsv", ".txt"}:
        return "counts"

    # inspect header lightly
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(8192)
            delim = _sniff_delimiter(sample)
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delim)
            header = next(reader, [])
            rows = []
            for _ in range(120):
                try:
                    rows.append(next(reader))
                except StopIteration:
                    break
    except Exception:
        return "counts"

    header = header or []
    has_lfc = _header_has_any(_DEG_LFC_CAND, header)
    has_q = _header_has_any(_DEG_Q_CAND, header)
    if has_lfc or (has_q and len(header) <= 12):
        return "degs"

    # counts heuristic: many numeric-ish cols, mostly non-negative, integer-like
    if not rows or len(header) < 4:
        return "counts"

    n_cols = max(len(header), max(len(r) for r in rows))
    num_cols = [0] * n_cols
    neg = 0
    tot = 0
    int_like = 0

    for r in rows:
        r = r + [""] * (n_cols - len(r))
        for j in range(1, n_cols):  # skip first col (often gene)
            fv = _safe_float(r[j])
            if fv is None:
                continue
            num_cols[j] += 1
            tot += 1
            if fv < 0:
                neg += 1
            if abs(fv - round(fv)) < 1e-8 and fv >= 0:
                int_like += 1

    numeric_cols = sum(1 for j in range(n_cols) if num_cols[j] >= max(5, len(rows)//5))
    if numeric_cols >= 4:
        neg_rate = neg / max(1, tot)
        int_rate = int_like / max(1, tot)
        if neg_rate <= 0.01 and int_rate >= 0.50:
            return "counts"

    return "counts"


# =============================================================================
# Enrichr gene set fetch (IPAA baseline)
# =============================================================================

def fetch_gene_sets_dict_any(libraries: Sequence[str], timeout_s: int = 30) -> Dict[str, List[str]]:
    """
    Returns { pathway_id: [GENE_SYMBOL, ...] } merged across libraries.
    Uses Enrichr 'geneSetLibrary' endpoint.
    Robust: best-effort per library.
    """
    out: Dict[str, List[str]] = {}
    
    
    for lib in libraries:
        lib = str(lib).strip()
        if not lib:
            continue
        url = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
        params = {"mode": "json", "libraryName": lib}
        try:
            r = requests.get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            obj = r.json()
            # shape: {"terms":[{"term_name":..., "genes":[...]}...]} or dict mapping
            if isinstance(obj, dict) and "terms" in obj and isinstance(obj["terms"], list):
                for t in obj["terms"]:
                    name = str(t.get("term_name", "")).strip()
                    genes = t.get("genes", [])
                    if name and isinstance(genes, list) and genes:
                        out[name] = list({str(g).strip() for g in genes if str(g).strip()})
            elif isinstance(obj, dict):
                # sometimes {term:[genes]}
                for k, v in obj.items():
                    if isinstance(v, list) and k:
                        out[str(k)] = list({str(g).strip() for g in v if str(g).strip()})
        except Exception as e:
            warn(f"[baseline][warn] Enrichr fetch failed for '{lib}': {trace(e)}")
            continue

    return out


# =============================================================================
# Baseline stack importer (baseline_expectations.py + utils_io.py)
# =============================================================================

def _load_module_at(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _import_ipaa_baseline_stack() -> Tuple[Optional[object], Optional[object]]:
    candidates: List[Path] = []

    env_dir = (os.environ.get("IPAA_BASELINE_STACK_DIR") or "").strip()
    if env_dir:
        candidates.append(Path(env_dir))

    # repo-local candidates
    candidates += [
        Path.cwd() / "IPAA_2",
        Path.cwd(),
        Path(__file__).resolve().parent / "IPAA_2",
        Path(__file__).resolve().parent,
    ]

    for base in candidates:
        utils_p = base / "utils_io.py"
        base_p = base / "baseline_expectations.py"
        if utils_p.exists() and base_p.exists():
            try:
                if str(base) not in sys.path:
                    sys.path.insert(0, str(base))
                utils_mod = _load_module_at("utils_io", utils_p)
                sys.modules["utils_io"] = utils_mod  # baseline_expectations may import it
                info(f"[baseline] loaded utils_io from {utils_p}")
                base_mod = _load_module_at("baseline_expectations", base_p)
                info(f"[baseline] loaded baseline_expectations from {base_p}")
                return base_mod, utils_mod
            except Exception as e:
                warn(f"[baseline] import failed at {base_p}: {trace(e)}")

    warn("[baseline] baseline_expectations.py or utils_io.py not found; skipping baseline.")
    return None, None


# =============================================================================
# SYMBOL -> ENSEMBL mapping helpers (as in your snippet)
# =============================================================================

def symbol_to_ensembl_map(std_frames: List[pd.DataFrame]) -> Dict[str, str]:
    try:
        cat = pd.concat(std_frames, ignore_index=True)
        if not {"ensembl_id", "gene_symbol"}.issubset(cat.columns):
            return {}
        cat = cat.dropna(subset=["ensembl_id", "gene_symbol"])
        cat["gene_symbol"] = cat["gene_symbol"].astype(str).str.upper().str.strip()
        cat["ensembl_id"] = cat["ensembl_id"].astype(str).str.strip()
        m = (
            cat.groupby(["gene_symbol", "ensembl_id"])
            .size()
            .reset_index(name="n")
            .sort_values(["gene_symbol", "n"], ascending=[True, False])
            .drop_duplicates("gene_symbol")
        )
        return dict(zip(m["gene_symbol"].values, m["ensembl_id"].values))
    except Exception as e:
        warn(f"symbol_to_ensembl_map failed: {trace(e)}")
        return {}

def gene_sets_SYMBOL_to_ENSEMBL(
    gs_sym: Mapping[str, Iterable[str]],
    sym2ens: Mapping[str, str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        for pid, genes in gs_sym.items():
            ens = [sym2ens.get(str(g).upper(), None) for g in genes]
            ens = [e for e in ens if isinstance(e, str) and len(e) > 0]
            if ens:
                out[str(pid)] = ens
        return out
    except Exception as e:
        warn(f"gene_sets_SYMBOL_to_ENSEMBL failed: {trace(e)}")
        return {}


# =============================================================================
# Build baseline expectations (IPAA)
# =============================================================================

@dataclass(frozen=True)
class BaselineConfig:
    data_dir: Path
    hpa_file: str
    gtex_file: str
    fantom_file: str
    pathway_libs: List[str]

def _baseline_config_from_env() -> Optional[BaselineConfig]:
    data_dir = (os.environ.get("IPAA_BASELINE_DATA_DIR") or "").strip()
    hpa = (os.environ.get("IPAA_HPA_FILE") or "").strip()
    gtex = (os.environ.get("IPAA_GTEX_FILE") or "").strip()
    fantom = (os.environ.get("IPAA_FANTOM_FILE") or "").strip()
    libs_raw = (os.environ.get("IPAA_BASELINE_PATHWAY_LIBS") or "").strip()

    if not all([data_dir, hpa, gtex, fantom, libs_raw]):
        warn("[baseline] Missing env baseline config; skipping baseline. "
             "Set IPAA_BASELINE_DATA_DIR, IPAA_HPA_FILE, IPAA_GTEX_FILE, IPAA_FANTOM_FILE, IPAA_BASELINE_PATHWAY_LIBS.")
        return None

    p = Path(data_dir).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        warn(f"[baseline] IPAA_BASELINE_DATA_DIR not found: {p}")
        return None

    libs = [x.strip() for x in libs_raw.split(",") if x.strip()]
    if not libs:
        warn("[baseline] IPAA_BASELINE_PATHWAY_LIBS empty; skipping baseline.")
        return None

    return BaselineConfig(data_dir=p, hpa_file=hpa, gtex_file=gtex, fantom_file=fantom, pathway_libs=libs)

def expected_baseline_path(out_root: Path, tissue: str) -> Path:
    return Path(out_root) / "baseline_consensus" / tissue / "baseline.pathway_expectations.consensus.tsv"

def ensure_consensus_baseline_for_tissue(out_root: Path, tissue: str, force: bool = False) -> Optional[Path]:
    out_root = Path(out_root).expanduser().resolve()
    tissue = (tissue or "").strip()
    if not tissue:
        return None

    cached = expected_baseline_path(out_root, tissue)
    if cached.exists() and cached.is_file() and not force:
        info(f"[baseline] cached: {cached}")
        return cached

    cfg = _baseline_config_from_env()
    if cfg is None:
        return None

    base_mod, utils_io = _import_ipaa_baseline_stack()
    if base_mod is None or utils_io is None:
        return None

    try:
        hpa_raw = load_table_auto(cfg.data_dir / cfg.hpa_file)
        gtex_raw = load_table_auto(cfg.data_dir / cfg.gtex_file)
        fantom_raw = load_table_auto(cfg.data_dir / cfg.fantom_file)
        if any(x is None or x.empty for x in [hpa_raw, gtex_raw, fantom_raw]):
            warn("[baseline] one or more baseline inputs empty; skipping baseline.")
            return None

        std_hpa = utils_io.standardize_long_any(hpa_raw)
        std_gtex = utils_io.standardize_long_any(gtex_raw)
        std_fantom = utils_io.standardize_long_any(fantom_raw)

        _, Z_HPA = base_mod.build_z_from_long(std_hpa)
        _, Z_GTEX = base_mod.build_z_from_long(std_gtex)
        _, Z_FANTOM = base_mod.build_z_from_long(std_fantom)

        Z_cons, _ = base_mod.build_consensus_Z({"HPA": Z_HPA, "GTEx": Z_GTEX, "FANTOM": Z_FANTOM})

        gs_sym = fetch_gene_sets_dict_any(cfg.pathway_libs)
        sym2ens = symbol_to_ensembl_map([std_hpa, std_gtex, std_fantom])
        gs_ens = gene_sets_SYMBOL_to_ENSEMBL(gs_sym, sym2ens)
        if not gs_ens:
            warn("[baseline] no gene sets after SYMBOL→ENSEMBL; skipping baseline.")
            return None

        all_expect = base_mod.compute_pathway_expectations_for_all_tissues(
            Z_cons=Z_cons,
            gene_sets_ENSEMBL=gs_ens,
            min_genes=5,
            source_used="CONSENSUS",
        )

        out_dir = (out_root / "baseline_consensus" / tissue)
        out_dir.mkdir(parents=True, exist_ok=True)

        # try canonicalization if present
        tkey = tissue
        try:
            canon = getattr(utils_io, "canon_tissue", None)
            if callable(canon):
                tkey2 = canon(tissue)
                if tkey2:
                    tkey = tkey2
        except Exception:
            pass

        if tkey not in all_expect:
            # fallback case-insensitive
            keys = [k for k in all_expect.keys() if str(k).lower() == tissue.lower()]
            if not keys:
                warn(f"[baseline] tissue '{tissue}' not found in consensus expectations; skipping.")
                return None
            tkey = keys[0]

        exp = all_expect[tkey]
        exp_path = out_dir / "baseline.pathway_expectations.consensus.tsv"
        exp.to_csv(exp_path, sep="\t", index=False)
        info(f"[baseline] {tissue} -> {exp_path}")
        return exp_path

    except Exception as e:
        warn(f"[baseline] build failed for tissue='{tissue}': {trace(e)}")
        return None


# =============================================================================
# Background universe (only where applicable)
# =============================================================================

_GENE_COL_CAND = ["gene", "genes", "symbol", "gene_symbol", "hgnc_symbol", "feature", "id", "ensembl_id", "ensembl"]

def _looks_like_gene(token: str) -> bool:
    s = (token or "").strip()
    if not s or len(s) > 40:
        return False
    if re.match(r"^ENS[A-Z]*G\d+", s):
        return True
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9\-\.]{0,39}$", s))

def _pick_gene_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in _GENE_COL_CAND:
        if cand in cols_map:
            return df[cols_map[cand]]
    # index fallback
    try:
        idx = df.index.astype(str)
        if len(idx) >= 10:
            sample = idx[: min(200, len(idx))]
            gene_like = sum(1 for v in sample if _looks_like_gene(v))
            if gene_like / max(1, len(sample)) >= 0.7:
                return pd.Series(idx, name="index_gene")
    except Exception:
        pass
    return None

def _safe_slug(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def universe_paths(out_root: Path, disease: str) -> Tuple[Path, Path]:
    out_root = Path(out_root).expanduser().resolve()
    d = _safe_slug(disease)
    udir = out_root / "background_universe"
    return (udir / f"{d}.txt", udir / f"{d}.meta.json")

def ensure_background_universe(
    out_root: Path,
    disease: str,
    input_path: Path,
    kind: str,
    force: bool = False,
) -> Optional[Path]:
    """
    Writes:
      OUT_ROOT/background_universe/<Disease>.txt  (one gene per line)
      OUT_ROOT/background_universe/<Disease>.meta.json
    Intended use:
      - COUNTS: YES (applicable)
      - DEGS: only if you truly want it (usually you lack full universe)
    """
    kind = (kind or "").strip().lower()
    if kind not in {"counts", "degs"}:
        return None

    out_root = Path(out_root).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()
    u_txt, u_meta = universe_paths(out_root, disease)
    u_txt.parent.mkdir(parents=True, exist_ok=True)

    if u_txt.exists() and u_txt.is_file() and not force:
        info(f"[universe] cached: {u_txt}")
        return u_txt

    # choose a file if a dir was provided
    table_path = input_path
    if input_path.is_dir():
        candidates: List[Path] = []
        for pat in ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"):
            candidates.extend(sorted(input_path.glob(pat)))
        if not candidates:
            warn(f"[universe] no table files in {input_path}")
            return None
        candidates = sorted(candidates, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        table_path = candidates[0]

    try:
        df = load_table_auto(table_path)
        ser = _pick_gene_series(df)
        if ser is None:
            warn(f"[universe] could not infer gene column/index for '{disease}' from {table_path}")
            return None

        genes = ser.dropna().astype(str).map(lambda x: x.strip())
        genes = genes[genes != ""].tolist()

        # stable unique
        seen = set()
        uniq: List[str] = []
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            uniq.append(g)

        # sanity filter if large enough
        if len(uniq) >= 50:
            sample = uniq[: min(300, len(uniq))]
            gene_like = sum(1 for v in sample if _looks_like_gene(v))
            if gene_like / max(1, len(sample)) >= 0.5:
                uniq = [g for g in uniq if _looks_like_gene(g)]

        if not uniq:
            warn(f"[universe] empty universe for '{disease}' from {table_path}")
            return None

        u_txt.write_text("\n".join(uniq) + "\n", encoding="utf-8")
        u_meta.write_text(json.dumps({
            "disease": disease,
            "kind": kind,
            "source_table": str(table_path),
            "n_genes": len(uniq),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        info(f"[universe] wrote: {u_txt} (n={len(uniq)})")
        return u_txt

    except Exception as e:
        warn(f"[universe] failed for '{disease}': {trace(e)}")
        return None


# =============================================================================
# Multi-cohort preparation for IPAA pipeline
# =============================================================================

@dataclass(frozen=True)
class PreparedCohort:
    disease: str
    input_path: str
    kind: str                 # counts|degs
    tissue: str               # resolved
    baseline_path: Optional[str]
    universe_path: Optional[str]


def prepare_cohort_for_ipaa(
    *,
    disease: str,
    input_path: Path,
    tissue_in: str = "",
    out_root: Path,
    allow_llm_tissue: bool,
    build_universe_for_degs: bool = False,
    force_baseline: bool = False,
    force_universe: bool = False,
) -> PreparedCohort:
    disease = (disease or "").strip() or "UnnamedDisease"
    input_path = Path(input_path).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()

    kind = detect_input_kind(input_path)
    hint = f"input={input_path.name}"

    tissue = resolve_tissue_or_die(
        disease=disease,
        tissue_in=(tissue_in or ""),
        hint=hint,
        allow_llm=allow_llm_tissue,
    )

    baseline_path: Optional[str] = None
    universe_path: Optional[str] = None

    # "only where applicable":
    # baseline is IPAA paper core for COUNTS-mode cohorts (tissue-adjusted expectations).
    if kind == "counts":
        b = ensure_consensus_baseline_for_tissue(out_root=out_root, tissue=tissue, force=force_baseline)
        baseline_path = str(b) if b else None

        u = ensure_background_universe(out_root=out_root, disease=disease, input_path=input_path,
                                       kind="counts", force=force_universe)
        universe_path = str(u) if u else None

    elif kind == "degs" and build_universe_for_degs:
        u = ensure_background_universe(out_root=out_root, disease=disease, input_path=input_path,
                                       kind="degs", force=force_universe)
        universe_path = str(u) if u else None

    return PreparedCohort(
        disease=disease,
        input_path=str(input_path),
        kind=kind,
        tissue=tissue,
        baseline_path=baseline_path,
        universe_path=universe_path,
    )


def write_prepared_manifest(out_root: Path, cohorts: Sequence[PreparedCohort]) -> Path:
    out_root = Path(out_root).expanduser().resolve()
    p = out_root / "IPAA_prepared_cohorts.json"
    payload = {
        "out_root": str(out_root),
        "cohorts": [
            {
                "disease": c.disease,
                "input_path": c.input_path,
                "kind": c.kind,
                "tissue": c.tissue,
                "baseline_path": c.baseline_path,
                "universe_path": c.universe_path,
            }
            for c in cohorts
        ],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"[ipaa] wrote manifest: {p}")
    return p
