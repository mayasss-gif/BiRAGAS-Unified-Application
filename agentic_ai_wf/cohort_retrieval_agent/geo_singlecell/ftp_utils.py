import os
import re
import tarfile
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from .constants import GEO_FTP_HTTP_BASE
# shutil was in your original imports but unused; we keep it if you want:
import shutil  # noqa: F401


def gse_to_ftp_dir(gse: str) -> str:
    gse = gse.strip()
    if not gse.upper().startswith("GSE"):
        raise ValueError(f"Not a GSE accession: {gse}")
    numeric = gse[3:]
    if len(numeric) <= 3:
        range_part = "nnn"
    else:
        range_part = numeric[:-3] + "nnn"
    return f"/geo/series/GSE{range_part}/{gse}"


def list_suppl_files_for_gse(gse: str, logger) -> List[str]:
    series_dir = gse_to_ftp_dir(gse)
    url = f"{GEO_FTP_HTTP_BASE}{series_dir}/suppl/"

    logger.info("Listing supplementary files for %s at %s", gse, url)
    r = requests.get(url)
    if r.status_code == 404:
        logger.warning("No 'suppl' directory for %s at %s", gse, url)
        return []
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or href in (".", ".."):
            continue
        files.append(href)
    return files


def analyze_suppl_files(gse: str, logger) -> Dict[str, Any]:
    """
    Look at supplementary files and classify:
      - matrix/barcodes/features
      - RAW.tar
      - annotation-like files
      - fastq-like (FASTQ/SRA/BAM/CRAM)
    Decide if likely 10x single-cell dataset.
    """
    files = list_suppl_files_for_gse(gse, logger)

    matrix_files, barcodes_files, features_files = [], [], []
    raw_tar_files, anno_files, fastq_like, others = [], [], [], []

    mtx_pattern = re.compile(r"\.mtx(\.gz)?$", re.IGNORECASE)
    barcodes_pattern = re.compile(r"barcodes\.(tsv|txt)(\.gz)?$", re.IGNORECASE)
    features_pattern = re.compile(r"(features|genes)\.(tsv|txt)(\.gz)?$", re.IGNORECASE)
    raw_tar_pattern = re.compile(r"RAW\.tar$", re.IGNORECASE)
    fastq_pattern = re.compile(r"\.(fastq|fq)(\.gz)?$", re.IGNORECASE)
    sra_pattern = re.compile(r"\.(sra)$", re.IGNORECASE)
    bam_pattern = re.compile(r"\.(bam|cram)$", re.IGNORECASE)

    anno_keywords = [
        "annot", "annotation", "celltype", "cell_type", "cell-type",
        "metadata", "meta", "coldata", "cluster", "seurat", "labels",
        "celllabels", "cell_labels", "bulk"
    ]
    anno_exts = [".rds", ".rda", ".rdata", ".h5ad", ".loom", ".tsv", ".csv", ".txt"]

    for fname in files:
        lower = fname.lower()

        if raw_tar_pattern.search(fname):
            raw_tar_files.append(fname)
            continue
        if mtx_pattern.search(fname):
            matrix_files.append(fname)
            continue
        if barcodes_pattern.search(fname):
            barcodes_files.append(fname)
            continue
        if features_pattern.search(fname):
            features_files.append(fname)
            continue

        if fastq_pattern.search(fname) or sra_pattern.search(fname) or bam_pattern.search(fname):
            fastq_like.append(fname)
            continue

        if any(k in lower for k in anno_keywords) or any(lower.endswith(ext) for ext in anno_exts):
            anno_files.append(fname)
            continue

        others.append(fname)

    is_10x = False
    reason: Optional[str] = None

    if matrix_files and barcodes_files and features_files:
        is_10x = True
        reason = "matrix+barcodes+features present in suppl"
    elif raw_tar_files:
        is_10x = True
        reason = "RAW.tar present (will inspect for 10x files)"

    if not is_10x:
        for fname in files:
            low = fname.lower()
            if "filtered_feature_bc_matrix" in low or "raw_feature_bc_matrix" in low:
                is_10x = True
                reason = "filtered/raw_feature_bc_matrix mentioned"
                break

    result = {
        "all_files": files,
        "matrix_files": matrix_files,
        "barcodes_files": barcodes_files,
        "features_files": features_files,
        "raw_tar_files": raw_tar_files,
        "anno_files": anno_files,
        "fastq_like": fastq_like,
        "others": others,
        "is_10x_singlecell": is_10x,
        "is_10x_reason": reason,
    }

    logger.info(
        "GSE %s supplementary summary: %d files | matrix=%d, barcodes=%d, features=%d, RAW.tar=%d, anno=%d, fastq-like=%d",
        gse,
        len(files),
        len(matrix_files),
        len(barcodes_files),
        len(features_files),
        len(raw_tar_files),
        len(anno_files),
        len(fastq_like),
    )
    if fastq_like:
        logger.info("GSE %s: FASTQ/SRA/BAM/CRAM present but will NOT be downloaded.", gse)
    if is_10x:
        logger.info("GSE %s: looks like 10x/single-cell (%s).", gse, reason)
    else:
        logger.info("GSE %s: does NOT clearly look like 10x single-cell.", gse)

    return result


def download_file(url: str, dest_path: str, logger,
                  chunk_size: int = 8192) -> None:
    logger.info("Downloading %s -> %s", url, dest_path)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)


def extract_10x_from_raw_tar(gse: str, gse_dir: str,
                             raw_tar_files: List[str],
                             logger) -> List[str]:
    """
    From RAW.tar, pull out GSM..._matrix.mtx.gz / barcodes.tsv.gz / features.tsv.gz,
    and organize them into per-sample folders:
      outdir/GSEXXXXXX/samples/<sample_id>/*.gz
    Returns the sample_id list.
    """
    sample_ids: List[str] = []
    if not raw_tar_files:
        return sample_ids

    samples_root = os.path.join(gse_dir, "samples")
    os.makedirs(samples_root, exist_ok=True)

    for raw_name in raw_tar_files:
        tar_path = os.path.join(gse_dir, raw_name)
        if not os.path.exists(tar_path):
            logger.warning("RAW.tar path not found for %s: %s", gse, tar_path)
            continue

        logger.info("Extracting 10x-style files from RAW tar %s for %s", raw_name, gse)
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    lower = base.lower()

                    if not (
                        lower.endswith("matrix.mtx.gz")
                        or lower.endswith("barcodes.tsv.gz")
                        or lower.endswith("features.tsv.gz")
                        or lower.endswith("genes.tsv.gz")
                    ):
                        continue

                    sample_id = base
                    for suf in [
                        "_matrix.mtx.gz",
                        "_barcodes.tsv.gz",
                        "_features.tsv.gz",
                        "_genes.tsv.gz",
                    ]:
                        if sample_id.endswith(suf):
                            sample_id = sample_id[: -len(suf)]
                            break

                    if sample_id not in sample_ids:
                        sample_ids.append(sample_id)

                    sample_dir = os.path.join(samples_root, sample_id)
                    os.makedirs(sample_dir, exist_ok=True)

                    dest_path = os.path.join(sample_dir, base)
                    if os.path.exists(dest_path):
                        logger.info("Skipping existing extracted file: %s", dest_path)
                        continue

                    logger.info("  Extracting %s -> %s", base, dest_path)
                    f = tar.extractfile(member)
                    if f is None:
                        logger.warning("Could not read member %s from tar.", member.name)
                        continue
                    with open(dest_path, "wb") as out_f:
                        while True:
                            chunk = f.read(8192)
                            if not chunk:
                                break
                            out_f.write(chunk)
        except Exception as e:
            logger.error("Error extracting RAW.tar for %s (%s): %s", gse, raw_name, e)

    return sample_ids


def download_gse_singlecell_bundle(record: Dict[str, Any],
                                   out_root: str,
                                   logger) -> Optional[Dict[str, Any]]:
    """
    Same logic as your original download_gse_singlecell_bundle.
    """
    gse = record["accession"]
    logger.info("==== Processing GSE %s ====", gse)

    # 1) Inspect supplementary files
    analysis = analyze_suppl_files(gse, logger)
    if not analysis["all_files"]:
        logger.warning("GSE %s: no supplementary files, skipping.", gse)
        return None

    # 2) If it is NOT 10x/single-cell, record it in a separate folder and exit
    if not analysis["is_10x_singlecell"]:
        non10x_root = os.path.join(out_root, "No_Feature_barcode_matrix")
        non10x_dir = os.path.join(non10x_root, gse)
        os.makedirs(non10x_dir, exist_ok=True)

        meta = {
            "accession": record.get("accession"),
            "title": record.get("title"),
            "summary": record.get("summary"),
            "taxon": record.get("taxon"),
            "gdstype": record.get("gdstype"),
            "entrytype": record.get("entrytype"),
            "geo_link": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}",
            "is_10x_singlecell": False,
            "has_feature_barcode_matrix": False,
            "reason": "No 10x-style feature/barcode/matrix files detected in suppl.",
            "esummary_raw": record.get("esummary", {}),
        }
        meta_path = os.path.join(non10x_dir, f"{gse}_metadata.json")
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            logger.info("Recorded NON-10x GSE %s metadata -> %s", gse, meta_path)
        except Exception as e:
            logger.error("Failed to write NON-10x metadata for %s: %s", gse, e)

        logger.info("GSE %s: skipping download (not classified as 10x/single-cell).", gse)
        return None

    # 3) It IS 10x / single-cell candidate → use special folder name
    gse_dir = os.path.join(out_root, f"10_Feature_barcode_matrix_{gse}")
    os.makedirs(gse_dir, exist_ok=True)

    # metadata JSON for this 10x GSE
    meta = {
        "accession": record.get("accession"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "taxon": record.get("taxon"),
        "gdstype": record.get("gdstype"),
        "entrytype": record.get("entrytype"),
        "geo_link": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}",
        "is_10x_singlecell": True,  # from analyze_suppl_files
        "esummary_raw": record.get("esummary", {}),
    }
    meta_path = os.path.join(gse_dir, f"{gse}_metadata.json")
    try:
        import json
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved metadata JSON for 10x GSE %s -> %s", gse, meta_path)
    except Exception as e:
        logger.error("Failed to write metadata for %s: %s", gse, e)

    # 4) Decide which files to download
    series_dir = gse_to_ftp_dir(gse)
    base_url = f"{GEO_FTP_HTTP_BASE}{series_dir}/suppl/"

    to_download: List[str] = []
    to_download.extend(analysis["matrix_files"])
    to_download.extend(analysis["barcodes_files"])
    to_download.extend(analysis["features_files"])
    to_download.extend(analysis["raw_tar_files"])
    to_download.extend(analysis["anno_files"])

    if not to_download:
        logger.warning("GSE %s: no matrix/RAW/annotation files to download.", gse)
        return None

    logger.info(
        "GSE %s: will download %d files (matrix/barcodes/features/RAW.tar/annotations).",
        gse, len(to_download)
    )

    downloaded_files: List[str] = []
    for fname in to_download:
        file_url = base_url + fname
        dest = os.path.join(gse_dir, fname)
        if os.path.exists(dest):
            logger.info("Skipping existing file: %s", dest)
            downloaded_files.append(fname)
            continue
        try:
            download_file(file_url, dest, logger)
            downloaded_files.append(fname)
        except Exception as e:
            logger.error("Failed to download %s for %s: %s", fname, gse, e)

    # 5) Extract per-sample 10x content from RAW.tar if present
    sample_ids: List[str] = []
    if analysis["raw_tar_files"]:
        sample_ids = extract_10x_from_raw_tar(gse, gse_dir,
                                              analysis["raw_tar_files"], logger)

    # --- infer sample IDs from matrix files if they exist directly in 'suppl' ---
    inferred_sample_ids: List[str] = []
    for fname in analysis["matrix_files"]:
        base = os.path.basename(fname)
        sample_id = re.sub(r"\.mtx(\.gz)?$", "", base, flags=re.IGNORECASE)
        sample_id = re.sub(r"_matrix$", "", sample_id, flags=re.IGNORECASE)
        if sample_id not in inferred_sample_ids:
            inferred_sample_ids.append(sample_id)

    for sid in inferred_sample_ids:
        if sid not in sample_ids:
            sample_ids.append(sid)

    has_feature_barcode_matrix = bool(analysis["matrix_files"]) or bool(sample_ids)

    meta["has_feature_barcode_matrix"] = has_feature_barcode_matrix
    try:
        import json
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Updated metadata JSON with has_feature_barcode_matrix for %s", gse)
    except Exception as e:
        logger.error("Failed to update metadata for %s: %s", gse, e)

    summary = {
        "accession": gse,
        "title": record.get("title"),
        "summary": record.get("summary"),
        "taxon": record.get("taxon"),
        "gdstype": record.get("gdstype"),
        "entrytype": record.get("entrytype"),
        "geo_link": meta["geo_link"],
        "gse_dir": gse_dir,
        "num_suppl_files": len(analysis["all_files"]),
        "downloaded_files": downloaded_files,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "has_feature_barcode_matrix": has_feature_barcode_matrix,
    }
    return summary
