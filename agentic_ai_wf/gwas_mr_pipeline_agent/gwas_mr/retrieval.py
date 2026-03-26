"""GWAS and eQTL data retrieval.

Refactored from gwas_eqtl_data_retrieval_pipeline.py into a callable
function ``retrieve_gwas_data()`` that accepts all parameters explicitly
(no interactive ``input()`` calls, no module-level side-effects).
"""

import gzip
import json
import os
import re
import shutil
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

# ── Constants ──────────────────────────────────────────────────────────────

CHUNK_SIZE = 1024 * 1024
HTTP_TIMEOUT = 30
HTTP_RETRIES = 5
RETRY_SLEEP = 3
MIN_HARMONISED_FILE_TYPES = 2

EURO_POS = ["european", "eur", "white british", "british", "irish", "ceu", "nfe"]
NON_EURO_NEG = ["asian", "african", "hispanic", "latino", "admixed"]

# ── Logging ────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(msg, flush=True)


# ── String helpers ─────────────────────────────────────────────────────────


def safe_name(s: str) -> str:
    """Normalise a string into a safe filesystem-friendly key."""
    return re.sub(r"[^\w]+", "_", str(s).lower())[:150].strip("_")


# ── HTTP helpers ───────────────────────────────────────────────────────────


def safe_request_get(
    url: str,
    stream: bool = False,
    timeout: int = HTTP_TIMEOUT,
    retries: int = HTTP_RETRIES,
) -> requests.Response:
    """GET with retries and simple back-off."""
    last_err: Optional[Exception] = None
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            log(f"[WARN] GET failed ({i}/{retries}) for {url} -> {e}")
            time.sleep(RETRY_SLEEP * i)
    raise RuntimeError(
        f"Failed GET after {retries} retries: {url} | Last error: {last_err}"
    )


# ── Sample-size extraction ─────────────────────────────────────────────────


def extract_n_better(text: Any) -> Optional[int]:
    """
    Conservative N extraction from free-text sample descriptions.
    Prefers ``total N=xxxxx``; falls back to max number found.
    """
    if not isinstance(text, str):
        return None
    t = text.lower()

    m = re.search(r"(total\s*)?n\s*=\s*([\d,]+)", t)
    if m:
        return int(m.group(2).replace(",", ""))

    nums = re.findall(r"\d[\d,]*", text)
    vals = [int(n.replace(",", "")) for n in nums] if nums else []
    return max(vals) if vals else None


# ── Ancestry helpers ───────────────────────────────────────────────────────


def clean_discovery_ancestry(val: Any) -> str:
    if not isinstance(val, str):
        return ""
    return val.split(",")[0].strip()


def is_eur(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(x in t for x in EURO_POS) and not any(x in t for x in NON_EURO_NEG)


# ── GTEx biosample + sample counts ────────────────────────────────────────


def load_gtex_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_biosample_fuzzy(user_biosample: str, gtex_biosamples_set: Set[str]) -> Optional[str]:
    """Exact / substring / fuzzy biosample matching against GTEx biosample list."""
    if not user_biosample:
        return None
    user_norm = user_biosample.lower().strip()

    for t in gtex_biosamples_set:
        if user_norm == t:
            return t
        if user_norm in t or t in user_norm:
            return t

    matches = get_close_matches(user_norm, list(gtex_biosamples_set), n=1, cutoff=0.4)
    return matches[0] if matches else None


def gtex_sample_counts(
    gtex_df: pd.DataFrame, biosample_lower: str
) -> tuple:
    """
    Pull RNA-seq and genotype sample counts from the GTEx biosample CSV.
    Returns ``(rnaseq_n, genotype_n)`` or ``(None, None)``.
    """
    if "Tissue" not in gtex_df.columns:
        return (None, None)

    sub = gtex_df[gtex_df["Tissue"].astype(str).str.lower().str.strip() == biosample_lower]
    if sub.empty:
        sub = gtex_df[
            gtex_df["Tissue"]
            .astype(str)
            .str.lower()
            .str.contains(re.escape(biosample_lower), na=False)
        ]
        if sub.empty:
            return (None, None)
        sub = sub.head(1)

    row = sub.iloc[0].to_dict()

    rnaseq_col: Optional[str] = None
    geno_col: Optional[str] = None
    for c in gtex_df.columns:
        cl = c.lower()
        if rnaseq_col is None and (
            ("rna" in cl and ("seq" in cl or "sequencing" in cl)) or ("rnaseq" in cl)
        ):
            rnaseq_col = c
        if geno_col is None and ("geno" in cl or "genotype" in cl):
            geno_col = c

    def _to_int(v: Any) -> Optional[int]:
        try:
            if pd.isna(v):
                return None
            if isinstance(v, str):
                v = v.replace(",", "").strip()
            return int(float(v))
        except Exception:
            return None

    rnaseq_n = _to_int(row.get(rnaseq_col)) if rnaseq_col else None
    genotype_n = _to_int(row.get(geno_col)) if geno_col else None
    return rnaseq_n, genotype_n


# ── eQTL file finding ─────────────────────────────────────────────────────


def find_eqtl_file(eqtl_root: str, biosample: str) -> Optional[Path]:
    key = safe_name(biosample)
    for p in Path(eqtl_root).rglob("*"):
        if p.is_file() and key in safe_name(p.name):
            return p
    return None


# ── LLM disease-name normalisation ────────────────────────────────────────


def _normalize_disease_llm(user_query: str) -> tuple:
    from decouple import config, UndefinedValueError

    try:
        api_key = config("OPENAI_API_KEY")
    except UndefinedValueError:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it in a .env file in your "
            "working directory, as a system environment variable, or in "
            "a settings.ini file."
        )

    from openai import OpenAI  # deferred import – package is optional

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a biomedical disease-name normalizer.\n\n"
        "Rules:\n"
        "- Correct spelling errors\n"
        "- If already valid, return unchanged\n"
        "- Output standard clinical disease name\n"
        "- If unknown, return UNKNOWN\n\n"
        "Return JSON:\n"
        '{"normalized_name": "...", "synonyms": ["...", "..."]}\n\n'
        f"Input:\n{user_query}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("normalized_name", user_query), data.get("synonyms", [])


def normalize_disease(user_query: str, use_llm: bool = True) -> tuple:
    if not use_llm:
        return user_query, []
    try:
        return _normalize_disease_llm(user_query)
    except Exception as e:
        log(f"[WARN] LLM normalization failed, fallback to raw input. Reason: {e}")
        return user_query, []


# ── Option A+ exact phenotype matching ─────────────────────────────────────


def filter_exact_with_fallback(
    hits_df: pd.DataFrame,
    primary_name: str,
    synonyms: List[str],
    wait_seconds: int = 5,
) -> pd.DataFrame:
    """
    Exact-match filter with synonym fallback.
    Prevents accidental inclusion of unrelated phenotype subtypes.
    """
    if hits_df.empty:
        return hits_df

    df = hits_df.copy()
    df["efo_clean"] = df["efoTraits"].astype(str).str.lower().str.strip()

    target = primary_name.lower().strip()
    exact = df[df["efo_clean"] == target]
    if not exact.empty:
        log(f"[OK] Exact GWAS trait match found for '{primary_name}'")
        return exact

    if synonyms:
        log(
            f"[INFO] No exact GWAS for '{primary_name}'. "
            f"Trying synonyms after {wait_seconds}s..."
        )
        time.sleep(wait_seconds)
        for syn in synonyms:
            syn_clean = str(syn).lower().strip()
            if not syn_clean:
                continue
            syn_hits = df[df["efo_clean"] == syn_clean]
            if not syn_hits.empty:
                log(f"[OK] Exact GWAS trait match found using synonym '{syn}'")
                return syn_hits

    log("[STOP] No exact GWAS trait match found for disease or its synonyms.")
    try:
        sample_traits = sorted(set(df["efoTraits"].astype(str)))[:10]
        log("Nearby available traits (sample):")
        for t in sample_traits:
            log(f"  - {t}")
    except Exception:
        pass
    return df.iloc[0:0]


# ── Download + decompress ──────────────────────────────────────────────────


def download_resume(url: str, out: str, max_retries: int = 5) -> None:
    """Resume-safe downloader for large files."""
    out = Path(out)
    retries = 0

    while retries < max_retries:
        try:
            headers: Dict[str, str] = {}
            if out.exists():
                existing_size = out.stat().st_size
                headers["Range"] = f"bytes={existing_size}-"
                log(f"Resuming download at byte {existing_size:,}")
            else:
                existing_size = 0
                log("Starting new download")

            with requests.get(url, stream=True, headers=headers, timeout=HTTP_TIMEOUT) as r:
                if r.status_code == 416:
                    log("Server rejected range request. Restarting download.")
                    out.unlink(missing_ok=True)
                    continue

                r.raise_for_status()
                mode = "ab" if existing_size > 0 else "wb"
                with open(out, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            log(f"Download completed: {out}")
            return

        except (ChunkedEncodingError, ConnectionError, Timeout) as e:
            retries += 1
            log(f"[WARN] Download interrupted ({retries}/{max_retries}). Retrying... {e}")
            time.sleep(RETRY_SLEEP)

        except Exception as e:
            retries += 1
            log(f"[WARN] Unexpected download error ({retries}/{max_retries}): {e}")
            time.sleep(RETRY_SLEEP)

    raise RuntimeError(f"Failed to download after {max_retries} retries: {url}")


def gunzip_file(gz_path: Path, out_path: Path) -> bool:
    """Unzip .gz to *out_path* (overwrite-safe).

    Removes the partial output file on failure so downstream code
    never picks up a corrupt/incomplete TSV.
    """
    try:
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        log(f"[WARN] Failed to unzip {gz_path} -> {out_path}: {e}")
        Path(out_path).unlink(missing_ok=True)
        return False


# ── Harmonised folder mirroring ────────────────────────────────────────────


def list_harmonised_files(harmonised_url: str) -> List[str]:
    """Parse directory listing HTML for href links."""
    try:
        html = safe_request_get(harmonised_url, stream=False).text
        files = re.findall(r'href="([^"]+)"', html)
        files = [f for f in files if f and not f.startswith("?") and not f.startswith("/")]
        return sorted(set(files))
    except Exception as e:
        log(f"[WARN] Could not list harmonised files: {harmonised_url} | {e}")
        return []


def mirror_harmonised_folder(harmonised_url: str, out_dir: Path) -> Dict[str, Any]:
    """
    Download all files from harmonised folder into ``out_dir/harmonised/``.
    Unzips the first ``.h.tsv.gz`` encountered.
    """
    harm_dir = out_dir / "harmonised"
    harm_dir.mkdir(parents=True, exist_ok=True)

    files = list_harmonised_files(harmonised_url)
    if not files:
        return {"harmonised_dir": str(harm_dir), "files": [], "gwas_gz": None, "gwas_tsv": None}

    downloaded: List[str] = []
    gwas_gz: Optional[Path] = None
    gwas_tsv: Optional[Path] = None

    for f in files:
        if f.endswith("/"):
            continue
        url = harmonised_url.rstrip("/") + "/" + f
        out_path = harm_dir / f

        try:
            log(f"Downloading harmonised file: {f}")
            download_resume(url, str(out_path), max_retries=HTTP_RETRIES)
            downloaded.append(str(out_path))

            if f.endswith(".h.tsv.gz") and gwas_gz is None:
                gwas_gz = out_path
                tsv_out = harm_dir / f.replace(".gz", "")
                if gunzip_file(gwas_gz, tsv_out):
                    gwas_tsv = tsv_out
                    log(f"Unzipped -> {tsv_out}")
        except Exception as e:
            log(f"[WARN] Failed download for {url}: {e}")
            continue

    return {
        "harmonised_dir": str(harm_dir),
        "files": downloaded,
        "gwas_gz": str(gwas_gz) if gwas_gz else None,
        "gwas_tsv": str(gwas_tsv) if gwas_tsv else None,
    }


# ── Reporting ──────────────────────────────────────────────────────────────


def write_dataset_excel(out_dir: Path, dataset_name: str, row_dict: dict) -> Path:
    """Write an Excel report inside each dataset folder."""
    xlsx_path = out_dir / f"{dataset_name}.xlsx"
    df = pd.DataFrame([row_dict])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dataset")
    return xlsx_path


# ── Cache detection ────────────────────────────────────────────────────────


def find_existing_local_dataset(out_root: Path, normalized_trait: str) -> Optional[Path]:
    """Detect cached GWAS data (layout-agnostic)."""
    trait_key = safe_name(normalized_trait)
    out_root = Path(out_root)

    if not out_root.exists():
        return None

    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        if safe_name(d.name) != trait_key:
            continue

        harm = d / "harmonised"
        if harm.exists() and any(harm.iterdir()):
            return d

        files = list(d.glob("*"))
        if any(
            f.is_file()
            and (
                f.name.lower().endswith(".meta.yaml")
                or f.name.lower().endswith(".yaml")
                or f.name.lower().endswith(".tsv")
                or f.name.lower().endswith(".tsv.gz")
                or f.name.lower().endswith(".gz")
            )
            for f in files
        ):
            return d

    return None


def detect_cached_gwas_files(existing_dataset_dir: Path) -> Dict[str, Any]:
    """
    Search for GWAS + meta files in both the dataset directory
    and its ``harmonised/`` subfolder.
    """
    existing_dataset_dir = Path(existing_dataset_dir)
    search_dirs = [existing_dataset_dir, existing_dataset_dir / "harmonised"]

    gwas_tsv: Optional[Path] = None
    gwas_gz: Optional[Path] = None
    meta: Optional[Path] = None

    for sd in search_dirs:
        if not sd.exists():
            continue
        if gwas_tsv is None:
            gwas_tsv = next(sd.glob("*.h.tsv"), None)
        if gwas_gz is None:
            gwas_gz = next(sd.glob("*.h.tsv.gz"), None)
        if meta is None:
            meta = next(sd.glob("*.meta.yaml"), None)

    if (existing_dataset_dir / "harmonised").exists() and any(
        (existing_dataset_dir / "harmonised").iterdir()
    ):
        harmonised_dir_used = existing_dataset_dir / "harmonised"
        layout = "newstyle_harmonised_subfolder"
    else:
        harmonised_dir_used = existing_dataset_dir
        layout = "legacy_or_root_files"

    if gwas_tsv:
        status = f"exists_unzipped_{layout}"
    elif gwas_gz:
        status = f"exists_gz_{layout}"
    elif meta:
        status = f"exists_meta_only_{layout}"
    else:
        status = f"exists_partial_only_{layout}"

    return {
        "gwas_tsv": str(gwas_tsv) if gwas_tsv else None,
        "gwas_gz": str(gwas_gz) if gwas_gz else None,
        "meta": str(meta) if meta else None,
        "harmonised_dir_used": str(harmonised_dir_used),
        "status": status,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main retrieval function
# ═══════════════════════════════════════════════════════════════════════════


def retrieve_gwas_data(
    disease_name: str,
    biosample_type: str,
    out_root: str,
    *,
    tsv_path: Optional[str] = None,
    eqtl_root: Optional[str] = None,
    gtex_biosample_csv: Optional[str] = None,
    use_llm: bool = True,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Retrieve GWAS + eQTL data for a disease / biosample type combination.

    This is the programmatic equivalent of running
    ``gwas_eqtl_data_retrieval_pipeline.py`` interactively.

    Parameters
    ----------
    disease_name : str
        Disease or phenotype name (e.g. ``"Breast Cancer"``).
    biosample_type : str
        GTEx biosample type (e.g. ``"Whole Blood"``, ``"Pancreas"``, ``"PBMC"``).
    out_root : str
        Output directory for downloaded GWAS data.
    tsv_path : str, optional
        Path to the GWAS catalog TSV.  Defaults to
        ``gwas_mr_reference/GWAS-DATABASE-FTP.tsv``.
    eqtl_root : str, optional
        Path to the GTEx eQTL biosample expression directory.  Defaults to
        ``gwas_mr_reference/GTEx_eQTL_TISSUE_EXPRESSION/``.
    gtex_biosample_csv : str, optional
        Path to the GTEx biosample sample-count CSV.  Defaults to
        ``gwas_mr_reference/GTEx_EQTL_SAMPLE_COUNTS/EQTL-SAMPLE-COUNT-GTEx Portal.csv``.
    use_llm : bool
        Use OpenAI LLM for disease-name normalisation (default ``True``).
        The API key is read via ``python-decouple`` (checks ``.env``,
        environment variables, and ``settings.ini`` automatically).
    top_k : int
        Number of top GWAS rows per trait by sample size (default ``1``).

    Returns
    -------
    list[dict]
        One dict per processed trait containing paths, sample sizes, and
        metadata needed to feed the downstream MR pipeline.
    """
    from .defaults import (
        DEFAULT_TSV_PATH,
        DEFAULT_EQTL_ROOT,
        DEFAULT_GTEX_BIOSAMPLE_CSV,
    )

    tsv_path = tsv_path or DEFAULT_TSV_PATH
    eqtl_root = eqtl_root or DEFAULT_EQTL_ROOT
    gtex_biosample_csv = gtex_biosample_csv or DEFAULT_GTEX_BIOSAMPLE_CSV
    # ── Disease normalisation ──────────────────────────────────────────────
    normalized_name, synonyms = normalize_disease(disease_name, use_llm=use_llm)
    log(f"Normalized disease: {normalized_name}")
    if synonyms:
        log(f"Synonyms: {', '.join(synonyms[:10])}")

    # ── Load GWAS table ────────────────────────────────────────────────────
    log("\nLoading GWAS FTP table (this can take time)...")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, on_bad_lines="skip")

    df["discoverySampleAncestry"] = df["discoverySampleAncestry"].apply(
        clean_discovery_ancestry
    )
    df["N"] = df["initialSampleDescription"].apply(extract_n_better)
    df["is_EUR"] = df["initialSampleDescription"].apply(is_eur) | df[
        "discoverySampleAncestry"
    ].apply(is_eur)

    def _search_terms(terms: List[str]) -> pd.DataFrame:
        mask = (
            df["efoTraits"]
            .fillna("")
            .str.lower()
            .apply(
                lambda x: any(
                    t.lower() in x for t in terms if isinstance(t, str) and t.strip()
                )
            )
        )
        return df[mask & df["summaryStatistics"].notna() & df["is_EUR"]]

    hits = _search_terms([normalized_name])
    if hits.empty and synonyms:
        hits = _search_terms(synonyms)

    if hits.empty:
        log("No GWAS dataset found (EUR + summaryStats required).")
        return []

    log(f"Found {len(hits)} GWAS rows (broad match).")

    # ── Option A+ narrowing ────────────────────────────────────────────────
    hits = filter_exact_with_fallback(
        hits_df=hits,
        primary_name=normalized_name,
        synonyms=synonyms,
        wait_seconds=5,
    )
    if hits.empty:
        log("\nPipeline stopped safely (no clean GWAS phenotype).")
        return []

    log(f"Proceeding with exact phenotype: '{hits.iloc[0]['efoTraits']}'")

    # ── Biosample resolution ─────────────────────────────────────────────
    biosample_lower = biosample_type.strip().lower()
    gtex_df = load_gtex_table(gtex_biosample_csv)
    gtex_biosamples = set(gtex_df["Tissue"].astype(str).str.lower().str.strip())

    resolved_biosample = resolve_biosample_fuzzy(biosample_lower, gtex_biosamples)
    if not resolved_biosample:
        log("[WARN] Could not confidently match biosample type -> fallback to Whole Blood")
        resolved_biosample = "whole blood"
    elif resolved_biosample != biosample_lower:
        log(f"Biosample '{biosample_lower}' matched to GTEx entry: '{resolved_biosample}'")

    user_biosample = resolved_biosample
    eqtl_file = find_eqtl_file(eqtl_root, user_biosample)
    rnaseq_n, genotype_n = gtex_sample_counts(gtex_df, user_biosample)

    log(f"Selected biosample type: {user_biosample}")
    log(f"Selected eQTL file: {eqtl_file}")
    log(f"GTEx RNA-seq N: {rnaseq_n} | Genotype N: {genotype_n}")

    # ── Create output root ─────────────────────────────────────────────────
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    master_rows: List[Dict[str, Any]] = []

    log("\nStarting GWAS selection & download...")

    # ── Trait loop (offline-first + single pass) ───────────────────────────
    for trait, sub in hits.groupby("efoTraits"):
        log("\n==================================================")
        log(f"Processing trait: {trait}")

        harmonised_url: Optional[str] = None
        existing_dataset_dir = find_existing_local_dataset(out_root_path, trait)

        if existing_dataset_dir:
            out_dir = Path(existing_dataset_dir)
            data_status = "cached"
            log(f"[CACHE] Found existing local dataset folder: {out_dir}")

            cached_info = detect_cached_gwas_files(out_dir)
            log(f"[CACHE] Layout status: {cached_info['status']}")

            gwas_tsv = cached_info.get("gwas_tsv")
            gwas_gz = cached_info.get("gwas_gz")
            harm_dir_used = Path(cached_info["harmonised_dir_used"])

            if gwas_tsv is None and gwas_gz:
                gwas_tsv_path = Path(gwas_gz).with_suffix("")
                if not gwas_tsv_path.exists():
                    if gunzip_file(Path(gwas_gz), gwas_tsv_path):
                        log(f"[CACHE] Unzipped cached GWAS -> {gwas_tsv_path}")
                    else:
                        log(
                            f"[WARN] Corrupt .gz detected. Deleting "
                            f"{gwas_gz} so next run will re-download."
                        )
                        Path(gwas_gz).unlink(missing_ok=True)
                        gwas_gz = None
                gwas_tsv = str(gwas_tsv_path) if gwas_tsv_path.exists() else None

            if not gwas_tsv and not gwas_gz:
                log(
                    "[WARN] Cached dataset found but no GWAS TSV/GZ present "
                    "(metadata-only / partial). Delete the cache folder and "
                    "re-run to trigger a fresh download."
                )

            mirror_info: Dict[str, Any] = {
                "harmonised_dir": str(harm_dir_used),
                "files": [],
                "gwas_gz": gwas_gz,
                "gwas_tsv": gwas_tsv,
            }

            best = sub.iloc[0]
            try:
                harmonised_url = (
                    str(best["summaryStatistics"]).rstrip("/") + "/harmonised/"
                )
            except Exception:
                pass

        else:
            out_dir = out_root_path / safe_name(trait)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "harmonised").mkdir(parents=True, exist_ok=True)

            data_status = "downloaded"
            sub = sub.copy()
            sub["N_num"] = pd.to_numeric(sub["N"], errors="coerce")
            sub = sub.sort_values("N_num", ascending=False)

            if len(sub) > 1:
                topn = sub.iloc[0]["N_num"]
                nextn = sub.iloc[1]["N_num"]
                if pd.notna(topn) and pd.notna(nextn):
                    log(
                        f"[INFO] Largest sample exists: selected top "
                        f"N={int(topn)} over next N={int(nextn)}"
                    )

            sub = sub.head(top_k)
            best = None
            best_score = -1

            for _, r in sub.iterrows():
                harm_url = str(r["summaryStatistics"]).rstrip("/") + "/harmonised/"
                log(f"Checking harmonised folder: {harm_url}")

                try:
                    html = safe_request_get(harm_url, stream=False).text
                except Exception as e:
                    log(f"[WARN] Cannot open harmonised URL: {e}")
                    continue

                score = sum(
                    [
                        ".h.tsv.gz" in html,
                        "meta.yaml" in html,
                        "README" in html,
                    ]
                )
                if score >= MIN_HARMONISED_FILE_TYPES and score > best_score:
                    best = r
                    best_score = score

            if best is None:
                log("[WARN] No harmonised GWAS found online (score threshold not met).")
                continue

            harmonised_url = (
                str(best["summaryStatistics"]).rstrip("/") + "/harmonised/"
            )
            log(f"[DOWNLOAD] Downloading GWAS from {harmonised_url}")
            mirror_info = mirror_harmonised_folder(harmonised_url, out_dir)

        # ── Build result row ───────────────────────────────────────────────
        n_eqtl_for_mr = genotype_n if genotype_n is not None else rnaseq_n
        gwas_for_cmd = mirror_info.get("gwas_tsv") or mirror_info.get("gwas_gz") or ""

        n_gwas_val = best.get("N") if best is not None else None
        try:
            n_gwas_val = int(n_gwas_val) if n_gwas_val is not None else None
        except Exception:
            pass

        dataset_row: Dict[str, Any] = {
            "trait": trait,
            "accession": best.get("accessionId") if best is not None else None,
            "N_gwas": n_gwas_val,
            "ancestry": (
                best.get("discoverySampleAncestry") if best is not None else None
            ),
            "eqtl_biosample_type": user_biosample,
            "eqtl_path": str(eqtl_file) if eqtl_file else None,
            "n_eqtl_rnaseq": rnaseq_n,
            "n_eqtl_genotype": genotype_n,
            "n_eqtl_for_mr": n_eqtl_for_mr,
            "harmonised_url": harmonised_url,
            "harmonised_dir": mirror_info.get("harmonised_dir"),
            "gwas_path_gz": mirror_info.get("gwas_gz"),
            "gwas_path_tsv": mirror_info.get("gwas_tsv"),
            "gwas_path": str(gwas_for_cmd) if gwas_for_cmd else None,
            "dataset_folder": str(out_dir),
            "dataset_report_xlsx": None,
            "data_status": data_status,
        }

        log(f"[MASTER] Dataset recorded ({data_status}) -> {out_dir.name}")

        try:
            dataset_name = out_dir.name
            xlsx_path = write_dataset_excel(out_dir, dataset_name, dataset_row)
            dataset_row["dataset_report_xlsx"] = str(xlsx_path)
            log(f"[OK] Dataset report saved: {xlsx_path}")
        except Exception as e:
            log(f"[WARN] Failed to write dataset Excel: {e}")

        master_rows.append(dataset_row)

    # ── Write master report ────────────────────────────────────────────────
    if master_rows:
        out_master_tsv = out_root_path / "MASTER_REPORT.tsv"
        out_master_xlsx = out_root_path / "MASTER_REPORT.xlsx"
        try:
            mdf = pd.DataFrame(master_rows)
            mdf.to_csv(out_master_tsv, sep="\t", index=False)
            with pd.ExcelWriter(out_master_xlsx, engine="openpyxl") as writer:
                mdf.to_excel(writer, index=False, sheet_name="master")
            log(f"\nMASTER_REPORT saved: {out_master_tsv}")
        except Exception as e:
            log(f"[ERROR] Failed to write master reports: {e}")
    else:
        log("\n[INFO] No datasets were recorded.")

    return master_rows
