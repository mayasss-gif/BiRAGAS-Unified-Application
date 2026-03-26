#!/usr/bin/env python3
# strict_10x_postclean_with_summary.py
#
# RULE YOU ASKED:
# - If a sample has ANY of these present (anywhere in its folder):
#     matrix.mtx(.gz) OR barcodes.tsv/txt(.gz) OR features.tsv/txt(.gz) OR genes.tsv/txt(.gz)
#   => DO NOT MOVE IT to No_Feature_barcode_matrix.
#
# - Only move samples that have NONE of those three types.
# - Root-level GSM groups: same rule (if any trio file exists, do not move).
#
# Creates: 10x_Validation.json + 10x_Validation.tsv in --outdir

import os
import re
import shutil
import tarfile
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# --- detect any of the 10x trio files (presence-based, not "complete bundle") ---
MATRIX_ANY_RE   = re.compile(r"matrix\.mtx(\.gz)?$", re.IGNORECASE)
BARCODES_ANY_RE = re.compile(r"barcodes\.(tsv|txt)(\.gz)?$", re.IGNORECASE)
FEATURES_ANY_RE = re.compile(r"(features|genes)\.(tsv|txt)(\.gz)?$", re.IGNORECASE)

# 10x folder hints
TENX_DIR_HINTS = [
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix",
    "feature_bc_matrix",
    "filtered_gene_bc_matrices",
    "raw_gene_bc_matrices",
]

ARCHIVE_RE = re.compile(r".*\.(tar|tar\.gz|tgz)$", re.IGNORECASE)
GSM_PREFIX_RE = re.compile(r"^(GSM\d+)", re.IGNORECASE)

# optional reporting only
COUNTS_LIKE_RE = re.compile(r"(counts|count|umi|expression).*?\.(txt|tsv|csv)(\.gz)?$", re.IGNORECASE)


def has_any_10x_component(file_names: List[str]) -> Tuple[bool, str]:
    """
    YOUR NEW LOGIC:
      keep if ANY of {matrix, barcodes, features/genes} exists
      OR any 10x directory hint exists
    """
    lower = [f.lower() for f in file_names]

    if any(any(h in x for h in TENX_DIR_HINTS) for x in lower):
        return True, "10x directory hint present"

    has_m = any(MATRIX_ANY_RE.search(os.path.basename(f)) for f in file_names)
    has_b = any(BARCODES_ANY_RE.search(os.path.basename(f)) for f in file_names)
    has_f = any(FEATURES_ANY_RE.search(os.path.basename(f)) for f in file_names)

    if has_m or has_b or has_f:
        parts = []
        if has_m: parts.append("matrix")
        if has_b: parts.append("barcodes")
        if has_f: parts.append("features/genes")
        return True, f"10x component present: {', '.join(parts)}"

    if any(COUNTS_LIKE_RE.search(os.path.basename(f)) for f in file_names):
        return False, "counts-like file(s) only; no 10x trio components"

    return False, "no matrix/barcodes/features (or hints) found"


# ----------------------------
# TAR review (optional, reporting)
# ----------------------------
def list_tar_members(tar_path: str) -> List[str]:
    with tarfile.open(tar_path, "r:*") as tar:
        return [m.name for m in tar.getmembers() if m.isfile() or m.isdir()]


def group_tar_by_sample(members: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}

    for name in members:
        base = os.path.basename(name)

        m = re.match(r"^(GSM\d+_[^_]+)", base, re.IGNORECASE)
        if m:
            sid = m.group(1)
            groups.setdefault(sid, []).append(base)
            continue

        m2 = re.search(r"(GSM\d+)", name, re.IGNORECASE)
        if m2:
            sid = m2.group(1)
            groups.setdefault(sid, []).append(base)
            continue

        if any(h in name.lower() for h in TENX_DIR_HINTS):
            parts = name.split("/")
            hint_idx = next((i for i, p in enumerate(parts) if p.lower() in TENX_DIR_HINTS), None)
            sid = parts[hint_idx - 1] if hint_idx is not None and hint_idx > 0 else "TENX_BUNDLE"
            groups.setdefault(sid, []).append(base)
            continue

        groups.setdefault("UNASSIGNED", []).append(base)

    return groups


def validate_tar_samples(tar_path: str) -> Dict[str, Tuple[bool, str, List[str]]]:
    members = list_tar_members(tar_path)
    grouped = group_tar_by_sample(members)
    out = {}
    for sid, files in grouped.items():
        ok, reason = has_any_10x_component(files)
        out[sid] = (ok, reason, files)
    return out


# ----------------------------
# Disk helpers
# ----------------------------
def list_files_recursive_rel(sample_dir: str) -> List[str]:
    files = []
    for root, _, fnames in os.walk(sample_dir):
        for f in fnames:
            rel = os.path.relpath(os.path.join(root, f), sample_dir)
            files.append(rel)
    return files


def safe_move(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    final = dst
    if os.path.exists(final):
        i = 2
        while os.path.exists(final + f"__dup{i}"):
            i += 1
        final = final + f"__dup{i}"
    shutil.move(src, final)
    return final


def group_root_files_by_gsm(gse_dir: str) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for fname in os.listdir(gse_dir):
        fpath = os.path.join(gse_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.upper().startswith("GSM"):
            continue
        m = GSM_PREFIX_RE.match(fname)
        sid = m.group(1) if m else "UNASSIGNED"
        groups[sid].append(fpath)
    return groups


def get_sample_dirs(gse_dir: str) -> List[str]:
    samples_root = os.path.join(gse_dir, "samples")
    if not os.path.isdir(samples_root):
        return []
    return sorted([d for d in os.listdir(samples_root) if os.path.isdir(os.path.join(samples_root, d))])


# ----------------------------
# Summary writer
# ----------------------------
def write_summary(outdir: str, summary: Dict) -> None:
    json_path = os.path.join(outdir, "10x_Validation.json")
    tsv_path = os.path.join(outdir, "10x_Validation.tsv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join([
            "GSE",
            "status",
            "kept_samples_count",
            "kept_samples",
            "moved_samples_count",
            "moved_samples",
            "moved_root_gsm_groups_count",
            "moved_root_gsm_groups",
            "moved_root_gsm_files_count",
            "moved_gse_folder",
        ]) + "\n")

        for gse, rec in summary["per_gse"].items():
            fh.write("\t".join([
                gse,
                rec.get("status", ""),
                str(rec.get("kept_samples_count", 0)),
                ",".join(rec.get("kept_samples", [])),
                str(rec.get("moved_samples_count", 0)),
                ",".join(rec.get("moved_samples", [])),
                str(rec.get("moved_root_gsm_groups_count", 0)),
                ",".join(rec.get("moved_root_gsm_groups", [])),
                str(rec.get("moved_root_gsm_files_count", 0)),
                rec.get("moved_gse_folder", ""),
            ]) + "\n")


# ----------------------------
# Main walker
# ----------------------------
def process_outdir(outdir: str) -> None:
    no_feature_root = os.path.join(outdir, "No_Feature_barcode_matrix")
    os.makedirs(no_feature_root, exist_ok=True)

    overall = {
        "outdir": outdir,
        "rule": "KEEP if ANY of {matrix.mtx(.gz), barcodes.tsv/txt(.gz), features/genes.tsv/txt(.gz)} exists OR 10x dir-hint exists. MOVE only if none exist.",
        "total_gse_seen": 0,
        "total_gse_moved_whole": 0,
        "total_kept_samples": 0,
        "total_moved_samples": 0,
        "total_moved_root_gsm_files": 0,
        "per_gse": {}
    }

    for name in os.listdir(outdir):
        if not name.startswith("10_Feature_barcode_matrix_GSE"):
            continue

        overall["total_gse_seen"] += 1
        gse_dir = os.path.join(outdir, name)
        gse = name.replace("10_Feature_barcode_matrix_", "")

        moved_samples = []
        moved_root_gsm_groups = []
        moved_root_gsm_files_count = 0
        moved_gse_folder = ""

        print(f"\n=== Checking {gse} ===")

        # (A) TAR review (optional, reporting only)
        tar_files = [f for f in os.listdir(gse_dir) if ARCHIVE_RE.match(f)]
        for tf in tar_files:
            tar_path = os.path.join(gse_dir, tf)
            print(f"  [TAR REVIEW] {tf}")
            try:
                res = validate_tar_samples(tar_path)
                for sid, (ok, reason, _) in res.items():
                    if not ok:
                        print(f"    - TAR group '{sid}' would be MOVED: {reason}")
            except Exception as e:
                print(f"    ! Failed to inspect tar {tf}: {e}")

        # (B) samples/ : move only if sample has NONE of matrix/barcodes/features
        samples_root = os.path.join(gse_dir, "samples")
        if os.path.isdir(samples_root):
            for sample_id in list(os.listdir(samples_root)):
                sample_dir = os.path.join(samples_root, sample_id)
                if not os.path.isdir(sample_dir):
                    continue

                files = list_files_recursive_rel(sample_dir)
                keep, reason = has_any_10x_component(files)
                if keep:
                    print(f"  [KEEP] samples/{sample_id}: {reason}")
                else:
                    dest = os.path.join(no_feature_root, gse, "samples", sample_id)
                    safe_move(sample_dir, dest)
                    moved_samples.append(sample_id)
                    overall["total_moved_samples"] += 1
                    print(f"  [MOVE] samples/{sample_id}: {reason}")

        # (C) root GSM groups : move only if group has NONE of matrix/barcodes/features
        root_groups = group_root_files_by_gsm(gse_dir)
        for gsm_id, files in root_groups.items():
            basenames = [os.path.basename(x) for x in files]
            keep, reason = has_any_10x_component(basenames)
            if keep:
                print(f"  [KEEP] root GSM {gsm_id}: {reason}")
            else:
                for fpath in files:
                    dest = os.path.join(no_feature_root, gse, "root_gsm", gsm_id, os.path.basename(fpath))
                    safe_move(fpath, dest)
                    moved_root_gsm_files_count += 1
                    overall["total_moved_root_gsm_files"] += 1
                moved_root_gsm_groups.append(gsm_id)
                print(f"  [MOVE] root GSM {gsm_id}: {reason} -> moved {len(files)} files")

        # (D) Move whole GSE folder only if NO samples/ remain AND no root GSM files remain
        remaining_samples = get_sample_dirs(gse_dir)
        remaining_root_gsm = group_root_files_by_gsm(gse_dir)

        if len(remaining_samples) == 0 and len(remaining_root_gsm) == 0:
            dest_gse = os.path.join(no_feature_root, gse, "GSE_folder_backup", os.path.basename(gse_dir))
            moved_gse_folder = safe_move(gse_dir, dest_gse)
            overall["total_gse_moved_whole"] += 1
            status = "MOVED_WHOLE_GSE"
            print(f"  [MOVE-GSE] Nothing left after cleaning. Moved whole GSE folder -> {moved_gse_folder}")
            kept_samples = []
        else:
            status = "KEPT_GSE"
            kept_samples = remaining_samples
            overall["total_kept_samples"] += len(kept_samples)
            print(f"  [FINAL] Kept sample folders in 10x: {', '.join(kept_samples) if kept_samples else '(none)'}")

        overall["per_gse"][gse] = {
            "status": status,
            "kept_samples_count": len(kept_samples),
            "kept_samples": kept_samples,
            "moved_samples_count": len(moved_samples),
            "moved_samples": moved_samples,
            "moved_root_gsm_groups_count": len(moved_root_gsm_groups),
            "moved_root_gsm_groups": moved_root_gsm_groups,
            "moved_root_gsm_files_count": moved_root_gsm_files_count,
            "moved_gse_folder": moved_gse_folder,
        }

    write_summary(outdir, overall)
    print(f"\n[SUMMARY] Wrote: {os.path.join(outdir, '10x_Validation.json')}")
    print(f"[SUMMARY] Wrote: {os.path.join(outdir, '10x_Validation.tsv')}")


# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--outdir", required=True)
#     args = ap.parse_args()
#     process_outdir(args.outdir)
#python -m geo_singlecell.cli --query "LUPUS" --retmax 2000 --max-gses 30 --outdir LUPUS
#python 10x_validation.py --outdir "D:\AyassBio_Workspace_Downloads\cohort_single_cell\10x_sc_geo_cohort_module\sle_sc-p1"
