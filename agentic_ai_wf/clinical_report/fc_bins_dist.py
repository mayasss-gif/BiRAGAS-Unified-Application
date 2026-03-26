# -----------------------------------------------------------------------
# FOLD‑CHANGE DISTRIBUTION                                                
# -----------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from .utils import _infer_columns_deg, _fig_to_base64


def compute_fc_bins(df:pd.DataFrame,*,adj_thr=0.05, patient_prefix: str = "patient"):
    gene,fc,p_value_col=_infer_columns_deg(df, patient_prefix)
    df=df.rename(columns={gene:"Gene",fc:"log2FC",p_value_col:"p_value"})
    sig=df[df.p_value<adj_thr]
    
    # Separate up and down regulated genes
    up_sig = sig[sig.log2FC > 0]
    down_sig = sig[sig.log2FC < 0]
    
    # Up-regulated bins
    up_mild = up_sig[(up_sig.log2FC >= 1) & (up_sig.log2FC < 2)]
    up_mod = up_sig[(up_sig.log2FC >= 2) & (up_sig.log2FC < 4)]
    up_ext = up_sig[up_sig.log2FC >= 4]
    
    # Down-regulated bins
    down_mild = down_sig[(down_sig.log2FC <= -1) & (down_sig.log2FC > -2)]
    down_mod = down_sig[(down_sig.log2FC <= -2) & (down_sig.log2FC > -4)]
    down_ext = down_sig[down_sig.log2FC <= -4]
    
    # Create grouped bar plot
    plt.figure(figsize=(4,2.8))
    categories = ["Mild\n(2-4x)", "Moderate\n(4-16x)", "Extreme\n(>16x)"]
    up_counts = [len(up_mild), len(up_mod), len(up_ext)]
    down_counts = [len(down_mild), len(down_mod), len(down_ext)]
    
    x = range(len(categories))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], up_counts, width, label='Up-regulated', color='red', alpha=0.7)
    plt.bar([i + width/2 for i in x], down_counts, width, label='Down-regulated', color='blue', alpha=0.7)
    
    plt.xlabel('Fold-change bins')
    plt.ylabel('Number of genes')
    plt.title('Fold-change distribution by regulation')
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    
    b64=_fig_to_base64()
    
    return {
        "up_mild": len(up_mild),
        "up_moderate": len(up_mod), 
        "up_extreme": len(up_ext),
        "down_mild": len(down_mild),
        "down_moderate": len(down_mod),
        "down_extreme": len(down_ext),
        "plot_b64": b64
    }