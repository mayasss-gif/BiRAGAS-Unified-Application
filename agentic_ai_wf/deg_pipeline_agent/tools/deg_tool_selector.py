"""
DEG Tool Selection Logic
"""
import numpy as np
import pandas as pd


def is_integer_like(df: pd.DataFrame, atol: float = 1e-6) -> bool:
    """Check whether the matrix is raw integer-like (required for DESeq2/edgeR)."""
    arr = df.values.astype(float)
    return np.allclose(arr, np.round(arr), atol=atol)


def looks_normalized_like(df: pd.DataFrame) -> bool:
    """Detect CPM/TMM/logCPM/log2/TPM/harmonized-like data."""
    vals = df.values.astype(float)
    if vals.size == 0:
        return False

    frac_decimals = (np.abs(vals - np.round(vals)) > 1e-6).mean()
    small_vals = (vals < 1).mean()
    lib_sizes = vals.sum(axis=0)
    cv_lib = np.std(lib_sizes) / (np.mean(lib_sizes) + 1e-12)

    many_decimals = frac_decimals > 0.10
    many_small = small_vals > 0.20
    flat_libsizes = cv_lib < 0.05

    return (many_decimals and (many_small or flat_libsizes)) or flat_libsizes


def determine_replicates(meta_df: pd.DataFrame) -> int:
    """Return minimum replicates per condition."""
    if "condition" not in meta_df.columns:
        return 0
    return int(meta_df["condition"].value_counts().min())


def choose_deg_tool(counts_df: pd.DataFrame,
                    meta_df: pd.DataFrame,
                    harmonized: bool = False) -> str:
    """
    Decide which DEG tool to use.
    
    Logic:
    - If harmonized/log-normalized → limma
    - If integer raw counts:
          → DESeq2 if good replicates + decent library variability
          → else edgeR (more flexible)
    - If normalized (CPM/TMM/log2/TPM) → limma
    """
    # Rule A: If harmonized → limma
    if harmonized:
        return "limma-voom"

    # Detect type
    int_like = is_integer_like(counts_df)
    norm_like = looks_normalized_like(counts_df)
    min_reps = determine_replicates(meta_df)

    # Rule C: Normalized (CPM/TMM/log2) → limma only
    if norm_like and not int_like:
        return "limma-voom"

    # Rule B: Raw integer counts
    if int_like:
        # Library size CV
        lib_sizes = counts_df.sum(axis=0)
        cv_lib = np.std(lib_sizes) / (np.mean(lib_sizes) + 1e-12)

        if min_reps >= 3 and cv_lib < 1.0:
            return "deseq2"
        else:
            return "edger"

    # Fallback — safest option
    return "limma-voom"


def normalize_deg(df: pd.DataFrame, tool_name: str) -> pd.DataFrame:
    """
    Produce a fully standardized DEG table for downstream processing.
    Supports: DESeq2, edgeR, limma-voom/trend.
    """
    tool = tool_name.lower()
    cols = [
        "Gene", "Original_ID", "baseMean",
        "log2FoldChange", "lfcSE", "stat",
        "pvalue", "padj", "Comparison", "tool"
    ]

    # DESeq2 already standardized (your DESeq2 tool produces correct columns)
    if tool == "deseq2":
        # Ensure Gene column exists
        if "Gene" not in df.columns:
            if df.index.name == "Gene" or df.index.name is None:
                df = df.reset_index()
                if "Gene" not in df.columns and len(df.columns) > 0:
                    df.rename(columns={df.columns[0]: "Gene"}, inplace=True)
        
        # Ensure Original_ID exists
        if "Original_ID" not in df.columns:
            df["Original_ID"] = df.get("Gene", df.index if hasattr(df, 'index') else None)
        
        # Ensure tool column exists
        if "tool" not in df.columns:
            df["tool"] = "deseq2"
        
        # Map column names to standard format
        column_mapping = {}
        if "pvalue" not in df.columns and "PValue" in df.columns:
            column_mapping["PValue"] = "pvalue"
        if "padj" not in df.columns and "padj" not in df.columns:
            if "FDR" in df.columns:
                column_mapping["FDR"] = "padj"
        if column_mapping:
            df.rename(columns=column_mapping, inplace=True)
        
        # Ensure all required columns exist (fill missing with NA)
        for col in cols:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Return with standard columns
        return df[cols] if all(c in df.columns for c in cols) else df

    norm = pd.DataFrame(index=df.index, columns=cols)

    # edgeR
    if tool == "edger":
        # Ensure Gene is a column, not index
        if "Gene" not in df.columns:
            if df.index.name == "Gene" or (df.index.name is None and len(df) > 0):
                df = df.reset_index()
                if "Gene" not in df.columns and len(df.columns) > 0:
                    df.rename(columns={df.columns[0]: "Gene"}, inplace=True)
        
        norm = pd.DataFrame(index=range(len(df)), columns=cols)
        norm["Gene"] = df.get("Gene", pd.NA)
        norm["Original_ID"] = norm["Gene"]
        norm["baseMean"] = pd.NA
        norm["log2FoldChange"] = df.get("logFC", pd.NA)
        norm["lfcSE"] = pd.NA
        norm["stat"] = df.get("LR", pd.NA)
        norm["pvalue"] = df.get("PValue", pd.NA)
        norm["padj"] = df.get("FDR", pd.NA)
        norm["Comparison"] = df.get("Comparison", pd.NA)
        norm["tool"] = "edger"
        return norm

    # limma (voom/trend unified)
    if tool == "limma-voom" or tool == "limma-trend":
        # Ensure Gene is a column, not index
        if "Gene" not in df.columns:
            if df.index.name == "Gene" or (df.index.name is None and len(df) > 0):
                df = df.reset_index()
                if "Gene" not in df.columns and len(df.columns) > 0:
                    df.rename(columns={df.columns[0]: "Gene"}, inplace=True)
        
        norm = pd.DataFrame(index=range(len(df)), columns=cols)
        norm["Gene"] = df.get("Gene", pd.NA)
        norm["Original_ID"] = norm["Gene"]
        norm["baseMean"] = df.get("AveExpr", pd.NA)
        norm["log2FoldChange"] = df.get("logFC", pd.NA)
        norm["lfcSE"] = pd.NA
        norm["stat"] = df.get("t", pd.NA)
        norm["pvalue"] = df.get("P.Value", pd.NA)
        norm["padj"] = df.get("adj.P.Val", pd.NA)
        norm["Comparison"] = df.get("Comparison", pd.NA)
        norm["tool"] = tool
        return norm

    # Fallback
    df["tool"] = tool
    return df

