import pandas as pd
from typing import Dict, List, Tuple, Optional
from .config import GO_IDS, UPREG_TARGET, MIN_ABS_LFC, DEFAULT_PRIORITY
import io
import base64
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------------------------------------------------
# DEG‑LEVEL STATISTICS                                                  
# ----------------------------------------------------------------------

def _infer_columns_deg(df: pd.DataFrame, patient_prefix: str = "patient") -> Tuple[str, str, str]:
    """Return (Gene, log2FC, adj_p) auto‑detected col names."""
    try:
        gene_col = next(c for c in df.columns if c.lower() == "gene")
    except StopIteration:
        raise ValueError(f"No 'Gene' column found in DEG dataframe. Available columns: {list(df.columns)}")
    
    try:
        fc_col = next(c for c in df.columns if c.lower().startswith(patient_prefix) and "log2fc" in c.lower())
    except StopIteration:
        raise ValueError(f"No log2FC column found with prefix '{patient_prefix}'. Available columns: {list(df.columns)}")
    
    try:
        # Try multiple p-value patterns
        p_value_patterns = ["p-value", "p_value", "pvalue", "adj_p", "padj"]
        p_value_col = None
        for pattern in p_value_patterns:
            try:
                p_value_col = next(c for c in df.columns if c.lower().startswith(patient_prefix) and pattern in c.lower())
                break
            except StopIteration:
                continue
        
        if p_value_col is None:
            raise ValueError(f"No p-value column found with prefix '{patient_prefix}'. Available columns: {list(df.columns)}")
            
    except Exception as e:
        raise ValueError(f"Error finding p-value column: {str(e)}")
    
    return gene_col, fc_col, p_value_col

# -------------------------- input validation ------------------------------ #

def _validate_inputs(
    deg_df: pd.DataFrame,
    sig_pathway_df: pd.DataFrame,
    disease_name: str,
) -> None:
    """Validate required inputs for signature analysis."""
    if deg_df is None or deg_df.empty:
        raise ValueError(
            "DEG dataframe cannot be empty for molecular signature analysis"
        )
    if sig_pathway_df is None or sig_pathway_df.empty:
        raise ValueError(
            "Pathway dataframe cannot be empty for molecular signature analysis"
        )
    if not disease_name or not disease_name.strip():
        raise ValueError("Disease name is required for clinical context")


# --------------------------- data preparation ----------------------------- #

def _rename_deg_columns(
    deg_df: pd.DataFrame,
    patient_prefix: str,
) -> pd.DataFrame:
    """Standardize DEG column names to Gene/log2FC/p_value."""
    gene_col, fc_col, p_col = _infer_columns_deg(deg_df, patient_prefix)
    return deg_df.rename(
        columns={gene_col: "Gene", fc_col: "log2FC", p_col: "p_value"}
    )


def _filter_significant_pathways(sig_pathway_df: pd.DataFrame) -> pd.DataFrame:
    """Filter significant pathways and prefer High/Medium confidence.

    Falls back to all significant pathways if fewer than 10 remain.
    """
    sig = sig_pathway_df[sig_pathway_df["FDR"] < 0.05]
    high_med = sig[
        sig["Confidence_Level"].isin(["High", "Medium"])
    ]
    return high_med if len(high_med) >= UPREG_TARGET else sig


def _build_log2fc_map(deg_df: pd.DataFrame) -> Dict[str, float]:
    """Create {GENE: log2FC} with cleaned gene names."""
    clean = (
        deg_df.assign(
            Gene=lambda d: (
                d["Gene"].astype(str)
                .str.strip()
                .str.split(".").str[0]
                .str.upper()
            )
        )
        .drop_duplicates(subset="Gene", keep="first")
        .set_index("Gene")["log2FC"]
        .astype(float)
    )
    return clean.to_dict()


# -------------------------- classification utils -------------------------- #

def _signature_type(db_id: str) -> str:
    """Return human label for signature type from DB_ID."""
    if db_id == "GO_CC":
        return "Cellular Compartments"
    if db_id == "GO_BP":
        return "Biological Processes"
    if db_id == "GO_MF":
        return "Molecular Functions"
    return "Pathway"


def _avg_fc_and_top_genes(
    genes: List[str],
    log2fc_map: Dict[str, float],
    k: int = 3,
) -> Tuple[Optional[float], List[Dict[str, float]]]:
    """Compute average FC and top-|k| genes by |log2FC| from a list of genes."""
    entries: List[Dict[str, float]] = []
    total = 0.0
    n = 0

    for raw in genes:
        gene = raw.strip().split(".")[0].upper()
        if gene in log2fc_map:
            val = float(log2fc_map[gene])
            if abs(val) >= MIN_ABS_LFC:
                entries.append({"gene": gene, "log2fc": val})
                total += val
                n += 1

    if n == 0:
        return None, []

    avg = total / n
    top = sorted(entries, key=lambda x: abs(x["log2fc"]), reverse=True)[:k]
    return avg, top


def _sort_key_base(pathway: Dict) -> Tuple[int, int]:
    """Sort by pathway type (Pathway before GO) then by priority rank."""
    is_go = pathway.get("db_id") in GO_IDS
    pathway_type_priority = 1 if is_go else 0
    priority_rank = int(pathway.get("priority_rank", DEFAULT_PRIORITY))
    return pathway_type_priority, priority_rank


def _sort_key_final(pathway: Dict) -> Tuple[int, int, int]:
    """Sort by: pathogenic first, Pathway before GO, then priority rank."""
    pathogenic_priority = 0 if pathway.get("is_pathogenic") else 1
    type_priority, priority_rank = _sort_key_base(pathway)
    return pathogenic_priority, type_priority, priority_rank


# --------------------------- clinical validation -------------------------- #

def _safe_series_get(series: pd.Series, key: str, default="N/A"):
    """Read-safe accessor for pandas Series values."""
    try:
        if key in series.index:
            val = series[key]
            return default if pd.isna(val) or val is None else val
        return default
    except Exception:
        return default


def _build_pathway_context(row: pd.Series) -> Dict:
    """Collect pathway fields for validator context."""
    fields = [
        "P_Value", "FDR", "LLM_Score", "Confidence_Level", "Score_Justification",
        "Clinical_Relevance", "Functional_Relevance", "Main_Class", "Sub_Class",
        "Disease_Category", "Disease_Subcategory", "Number_of_Genes", "Input_Genes",
        "Pathway_Associated_Genes", "Priority_Rank", "Pathway_Source", "Pathway_ID",
    ]
    return {f: _safe_series_get(row, f, "N/A") for f in fields}


def cap_confidence(confidence: float) -> float:
    """
    Cap confidence scores at 85% maximum to maintain scientific uncertainty.
    
    Args:
        confidence: Raw confidence score (0.0 to 1.0)
        
    Returns:
        Capped confidence score (0.0 to 0.85)
    """
    return min(max(confidence, 0.0), 0.85)

# ────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS                                                        
# ────────────────────────────────────────────────────────────────────────

def _fig_to_base64() -> str:
    """Serialize current Matplotlib figure → base64 PNG string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ────────────────────────────────────────────────────────────────────────
# PUBLIC API                                                              
# ────────────────────────────────────────────────────────────────────────

def load_inputs(deg_csv: str | Path, path_csv: str | Path, drug_csv: str | Path) -> Dict[str, pd.DataFrame]:
    """Read the three CSVs and return them as DataFrames."""
    return {
        "deg_df": pd.read_csv(deg_csv),
        "path_df": pd.read_csv(path_csv),
        "drug_df": pd.read_csv(drug_csv) if drug_csv else None,
    }

# ------------------------- composition utilities -------------------------- #

def _fill_to_ten_upregulated(pathways: List[Dict]) -> List[Dict]:
    """Pick up to 10 upregulated with pathogenic-first policy."""
    pathogenic = [p for p in pathways if p.get("is_pathogenic")]
    non_pathogenic = [p for p in pathways if not p.get("is_pathogenic")]

    if len(pathogenic) >= UPREG_TARGET:
        top = pathogenic[:UPREG_TARGET]
    else:
        remaining = UPREG_TARGET - len(pathogenic)
        top = pathogenic + non_pathogenic[:remaining]

    top.sort(key=_sort_key_final)
    return top


def _enrich_downregulated(
    downregulated: List[Dict]
) -> List[Dict]:
    """Finalize downregulated list (sort and annotate already computed fields)."""
    downregulated.sort(key=_sort_key_final)
    return downregulated[:UPREG_TARGET]


def _signature_block(
    name: str,
    regulation: str,
    pathways: List[Dict],
) -> Dict:
    """Construct a signature block with composition stats."""
    go_cc = sum(1 for p in pathways if p.get("db_id") == "GO_CC")
    go_bp = sum(1 for p in pathways if p.get("db_id") == "GO_BP")
    go_mf = sum(1 for p in pathways if p.get("db_id") == "GO_MF")
    non_go = sum(1 for p in pathways if p.get("db_id") not in GO_IDS)

    return {
        "signature_name": f"{name} ({len(pathways)})",
        "regulation_type": regulation,
        "pathways": pathways,
        "signature_composition": {
            "pathway_count": non_go,
            "go_cc_count": go_cc,
            "go_bp_count": go_bp,
            "go_mf_count": go_mf,
        },
    }


def _top_and_bottom_gene(deg_df: pd.DataFrame) -> Tuple[str, float, str, float]:
    """Return highest and lowest log2FC gene tuples for reporting."""
    if deg_df.empty:
        return "No genes available", 0.0, "No genes available", 0.0

    top_row = deg_df.nlargest(1, "log2FC")
    bot_row = deg_df.nsmallest(1, "log2FC")

    top_gene = top_row["Gene"].iloc[0] if not top_row.empty else "NA"
    top_lfc = float(top_row["log2FC"].iloc[0]) if not top_row.empty else 0.0
    low_gene = bot_row["Gene"].iloc[0] if not bot_row.empty else "NA"
    low_lfc = float(bot_row["log2FC"].iloc[0]) if not bot_row.empty else 0.0
    return top_gene, top_lfc, low_gene, low_lfc