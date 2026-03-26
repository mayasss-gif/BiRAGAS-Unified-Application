# mdp_io.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .mdp_logging import info, warn, err, trace

def _sniff_delim_bytes(b: bytes) -> str | None:
    try:
        d = csv.Sniffer().sniff(
            b.decode("utf-8", errors="ignore"),
            delimiters=[",", "\t", ";", "|", " "],
        )
        return d.delimiter
    except Exception:
        return None

def _dedup_columns(cols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: Dict[str, int] = {}
    for c in cols:
        base = str(c)
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            k = seen[base]
            new = f"{base}.{k}"
            while new in seen:
                k += 1
                new = f"{base}.{k}"
            seen[base] = k + 1
            seen[new] = 1
            out.append(new)
    return out

def load_table_auto(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = p.suffix.lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(p, header=0)
        else:
            with open(p, "rb") as fh:
                sample = fh.read(65536)
            sep = _sniff_delim_bytes(sample) or ","
            try:
                df = pd.read_csv(p, sep=sep, header=0)
            except Exception:
                df = pd.read_csv(p, sep=sep, header=None)
                df.columns = [f"col{i}" for i in range(df.shape[1])]
            if any(str(c).startswith("Unnamed") for c in df.columns) and df.shape[1] > 2:
                first_row_header = df.iloc[0].astype(str).tolist()
                if sum(c != "nan" for c in first_row_header) > df.shape[1] // 2:
                    df = pd.read_csv(p, sep=sep, header=None)
                    df.columns = df.iloc[0].astype(str).tolist()
                    df = df.drop(index=0).reset_index(drop=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "__".join([str(x) for x in tup if str(x) != "nan"]).strip()
                for tup in df.columns.values
            ]
        else:
            df.columns = [str(c).strip() for c in df.columns]
        df.columns = _dedup_columns(df.columns)
        return df
    except Exception as e:
        err(f"load_table_auto failed for {p.name}: {trace(e)}")
        return pd.DataFrame()

def save_barh(
    df: pd.DataFrame,
    term_col: str,
    fdr_col: str,
    title: str,
    out_png: Path,
    top_n: int,
    dpi: int,
) -> None:
    try:
        if df is None or df.empty or term_col not in df.columns:
            return
        fdr = fdr_col if fdr_col in df.columns else ("qval" if "qval" in df.columns else None)
        if fdr is None:
            return
        dd = df.copy()
        dd[fdr] = pd.to_numeric(dd[fdr], errors="coerce").fillna(1.0).clip(lower=1e-300)
        dd["neglog10FDR"] = -np.log10(dd[fdr])
        top = dd.sort_values([fdr, term_col]).head(top_n)
        if top.empty:
            return
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, max(4, top.shape[0] * 0.35)))
        plt.barh(top[term_col], top["neglog10FDR"])
        plt.title(title)
        plt.xlabel("–log10(FDR)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi)
        plt.close()
    except Exception as e:
        warn(f"plot failed: {title} -> {trace(e)}")

def _load_any_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    try:
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        with open(path, "rb") as fh:
            sample = fh.read(65536)
        sep = _sniff_delim_bytes(sample) or ","
        return pd.read_csv(path, sep=sep)
    except Exception as e:
        warn(f"_load_any_table failed ({path.name}): {trace(e)}")
        return pd.DataFrame()
