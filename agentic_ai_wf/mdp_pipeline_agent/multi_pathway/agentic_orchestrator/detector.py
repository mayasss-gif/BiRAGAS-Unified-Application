
# detector.py
"""Lightweight input type detection for mdp_pipeline_3."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal, Optional, Tuple, List
import pandas as pd

DataType = Literal["COUNTS", "DEGS", "GL", "GC", "JSONS"]

class DataTypeDetector:
    GENE_COL_CANDIDATES = {"gene","genes","symbol","hgnc","hgnc_symbol","ensembl","ensembl_id","Gene","SYMBOL","HGNC","ENSEMBL"}
    LOGFC_CANDS = {"logfc","log2fc","lfc"}
    PVALUE_CANDS = {"pvalue","p_value","pval","p","padj","fdr","qvalue","q_value","q"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def detect(self) -> Tuple[Optional[DataType], str]:
        p = self.path
        if not p.exists():
            return None, f"Path not found: {p}"
        if p.is_dir():
            if p.name == "jsons_all_folder" or any((p / f).is_file() and f.endswith(".json") for f in self._ls(p)):
                if list(p.glob("*.json")):
                    return "JSONS", f"Detected JSONS in folder {p}"
            counts = degs = gl = gc = 0
            for f in self._iter_candidate_files(p):
                dtype, _ = self._detect_file(f)
                if dtype == "COUNTS": counts += 1
                elif dtype == "DEGS": degs += 1
                elif dtype == "GL": gl += 1
                elif dtype == "GC": gc += 1
            ranking = [("JSONS",0),("COUNTS",counts),("DEGS",degs),("GL",gl),("GC",gc)]
            ranking.sort(key=lambda x: x[1], reverse=True)
            top, n = ranking[0]
            if n>0:
                return top, f"Detected dominant type {top} in {p} (counts={counts}, degs={degs}, gl={gl}, gc={gc})"
            nested = list(p.rglob("jsons_all_folder"))
            if nested:
                return "JSONS", f"Found nested jsons_all_folder under {nested[0]}"
            return None, f"Could not classify directory {p}"
        if p.is_file():
            return self._detect_file(p)
        return None, f"Unknown path type: {p}"

    def _detect_file(self, f: Path) -> Tuple[Optional[DataType], str]:
        suf = f.suffix.lower()
        if suf == ".json":
            try:
                js = json.loads(f.read_text()[:200000])
                if isinstance(js, dict) and "data" in js and isinstance(js["data"], dict):
                    keys = set(map(str.lower, js["data"].keys()))
                    if "symbol" in keys or "symbols" in keys:
                        return "GC", f"GeneCards-like JSON: {f.name}"
            except Exception:
                pass
        if suf in (".csv",".tsv",".txt",".tab",".xlsx",".xls"):
            try:
                df = self._load_table_quick(f)
                if df is None or df.empty:
                    return None, f"Empty/invalid table: {f.name}"
                cols_lower = [c.lower() for c in df.columns]
                has_gene = any(c.lower() in self.GENE_COL_CANDIDATES for c in df.columns)
                has_logfc = any(c in cols_lower for c in self.LOGFC_CANDS)
                has_p = any(c in cols_lower for c in self.PVALUE_CANDS)
                if has_gene and not has_logfc and not has_p and df.shape[1] >= 4:
                    numeric_cols = df.select_dtypes("number").shape[1]
                    if numeric_cols >= 3:
                        return "COUNTS", f"Counts-like matrix: {f.name}"
                if has_gene and has_logfc and has_p:
                    return "DEGS", f"DEG-like table: {f.name}"
                if has_gene:
                    numeric_cols = df.select_dtypes("number").shape[1]
                    if numeric_cols <= 1 or df.shape[1] <= 2:
                        return "GL", f"Gene list / scored list: {f.name}"
            except Exception as e:
                return None, f"Failed parse {f.name}: {e}"
        return None, f"Unrecognized file pattern: {f.name}"

    def _load_table_quick(self, f: Path):
        suf = f.suffix.lower()
        if suf in (".xlsx",".xls"):
            return pd.read_excel(f, nrows=50)
        sep = "," if suf == ".csv" else "\t"
        try:
            return pd.read_csv(f, sep=sep, nrows=50)
        except Exception:
            return pd.read_csv(f, sep=None, engine="python", nrows=50)

    def _iter_candidate_files(self, folder: Path):
        exts = (".json",".csv",".tsv",".txt",".tab",".xlsx",".xls")
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                yield f

    def _ls(self, folder: Path) -> List[str]:
        try:
            return [x.name for x in folder.iterdir()]
        except Exception:
            return []
