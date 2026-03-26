# ----------------------------------------------------------------------
# GENERIC PANEL / PATHWAY SIGNATURE                                     
# ----------------------------------------------------------------------
import pandas as pd
from typing import List, Dict
from .utils import _infer_columns_deg

def compute_custom_signature(
    df: pd.DataFrame,
    gene_list: List[str],
    *,
    adj_p_thr: float = 0.05,
    logfc_thr: float = 1.0,
    patient_prefix: str = "patient"
) -> Dict:
    """Analyse any list of genes for up/down significant hits."""
    gene_col, fc_col, p_value_col = _infer_columns_deg(df, patient_prefix)
    df = df.rename(columns={gene_col: "Gene", fc_col: "log2FC", p_value_col: "p_value"})

    subset = df[df.Gene.isin(gene_list)].copy()
    subset["is_sig"] = (subset.p_value < adj_p_thr) & (subset.log2FC.abs() >= logfc_thr)
    up_sig   = subset[(subset.is_sig) & (subset.log2FC > 0)]
    down_sig = subset[(subset.is_sig) & (subset.log2FC < 0)]

    return {
        "panel_size": len(gene_list),
        "detected": len(subset),
        "sig_up": len(up_sig),
        "sig_down": len(down_sig),
        "details": subset.sort_values("p_value").to_dict(orient="records"),
    }
