# ----------------------------------------------------------------------
# PATHWAY‑LEVEL SUMMARY                                                 
# ----------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from .config import PATHWAY_CATEGORIES
from .utils import _fig_to_base64
from matplotlib.cm import ScalarMappable

def summarize_pathways(
    path_df: pd.DataFrame,
    *,
    fdr_thr: float = 0.05,
    top_n: int = 10,
    main_col: str = "Main_Class",
    sub_col: str = "Sub_Class"
):
    """Summarise enriched pathways.

    If the uploaded `Pathways.csv` already contains high‑level columns
    `Main_Class` and `Sub_Class`, we rely on them **instead of** keyword
    heuristics.  Otherwise we gracefully fall back to the internal
    `PATHWAY_CATEGORIES` mapping.
    """
    # Apply DB_ID and Confidence_Level filter
    # ENHANCED: Include all databases (no DB_ID restrictions)
    filtered_df = path_df[((path_df["Confidence_Level"] == "High") | (path_df["Confidence_Level"] == "Medium"))]
    # If less than 10 rows after filter, remove DB_ID filter
    if len(filtered_df) < 10:
        filtered_df = path_df[(path_df["Confidence_Level"] == "High") | (path_df["Confidence_Level"] == "Medium")]
    path_df = filtered_df

    # Normalise primary stats columns first
    path_df = path_df.rename(
        columns={
            "FDR": "FDR",
            "Pathway_Name": "Pathway",
            "Regulation": "Reg",
        }
    )

    sig = path_df[path_df.FDR < fdr_thr].copy().sort_values("FDR")
    # ── Category assignment ────────────────────────────────────────────
    if main_col in sig.columns:
        sig["Category"] = sig[main_col].fillna("Other")
    else:
        # Fallback to keyword heuristics used previously
        def _by_kw(name: str) -> str:
            low = name.lower()
            for cat, kws in PATHWAY_CATEGORIES.items():
                if any(kw in low for kw in kws):
                    return cat
            return "Other"

        sig["Category"] = sig.Pathway.astype(str).map(_by_kw)

    if sub_col in sig.columns:
        sig["SubCategory"] = sig[sub_col].fillna("Unspecified")
    else:
        sig["SubCategory"] = "Unspecified"
    
    # significant pathways in dict
    sig_dict = sig[["Pathway","Reg","Category","SubCategory","FDR"]].to_dict(orient="records")
    
    main_counts = sig[sig["Confidence_Level"].isin(["High", "Medium"])].Category.value_counts().to_dict()
    
    # donut chart for Main_Class
    plt.figure(figsize=(8, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(main_counts)))
    
    # Calculate percentages for labels
    total = sum(main_counts.values())
    percentages = [f'{count/total*100:.1f}%' for count in main_counts.values()]
    
    plt.pie(main_counts.values(), labels=percentages, pctdistance=0.85, colors=colors)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')  # Increased radius from 0.50 to 0.70
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    # Create legend with colors mapped to labels
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i]) for i in range(len(main_counts))]
    plt.legend(legend_elements, list(main_counts.keys()), loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              fontsize=14, frameon=True, fancybox=True, shadow=True, title="Pathway Classes", title_fontsize=14)
    plt.tight_layout()
    
    pie_b64 = _fig_to_base64()

    # pathway activity bar plot (horizontal)
    # Separate up and down regulated pathways
    up = sig[sig['Reg'].str.lower() == 'up']
    down = sig[sig['Reg'].str.lower() == 'down']
    
    # Prepare data for plotting
    up_paths = up['Pathway'].tolist()
    up_vals = [1]*len(up_paths)
    down_paths = down['Pathway'].tolist()
    down_vals = [-1]*len(down_paths)
    
    # Combine for plotting (down first, then up)
    pathways = down_paths[::-1] + up_paths
    values = down_vals[::-1] + up_vals
    colors = ['#ff7f0e']*len(down_paths) + ['#1f77b4']*len(up_paths)  # orange for down, blue for up
    
    plt.figure(figsize=(10, 7))
    plt.barh(pathways, values, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Pathway Activity Direction')
    plt.title('Extended Transcriptomic Pathway Activity')
    plt.yticks(fontsize=12)
    plt.xticks([-1, 0, 1], ['Downregulated', '', 'Upregulated'])
    plt.legend([
        plt.Rectangle((0,0),1,1,color='#1f77b4'),
        plt.Rectangle((0,0),1,1,color='#ff7f0e')
    ], ['Upregulated', 'Downregulated'], loc='upper center')
    plt.tight_layout()
    pathway_activity_plot_b64 = _fig_to_base64()


    # up/down regulated pathways bar plot --------------------------------
    up   = path_df[path_df['Reg']=='Up'].nlargest(top_n, 'Number_of_Genes').copy()
    down = path_df[path_df['Reg']=='Down'].nlargest(top_n, 'Number_of_Genes').copy()
    if up.empty and down.empty:
        return {"error": "No Up or Down pathways in your DataFrame."}

    up['sig']   = -np.log10(up['FDR'])
    down['sig'] = -np.log10(down['FDR'])
    up['sig_signed']   = up['sig']
    down['sig_signed'] = -down['sig']

    all_sig = pd.concat([up['sig_signed'], down['sig_signed']])
    vmax   = all_sig.abs().max()
    norm   = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap   = plt.cm.bwr
    colors = cmap(norm(pd.concat([up['sig_signed'], down['sig_signed']])))

    # Prepare data for plotting
    up_labels = list(up['Pathway'])
    down_labels = list(down['Pathway'])
    up_values = list(up['Number_of_Genes'])
    down_values = list(-down['Number_of_Genes'])

    labels = up_labels + down_labels
    values = up_values + down_values

    # Make the plot much bigger to accommodate long labels
    fig, ax = plt.subplots(figsize=(20, max(18, 0.7*len(labels))))  # width=20, height scales with number of bars

    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor='black', height=0.6)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_yticks(y)
    # Estimate the maximum number of characters that fit in the axes width
    fig.canvas.draw()  # Needed to get renderer and axes size
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    # Use a fraction of the axes width for label area (e.g., 25% for left, 25% for right)
    label_area_frac = 0.23  # 23% of axes width for label area
    label_area_px = ax_bbox.width * label_area_frac

    # Use a test string to estimate average character width in pixels
    test_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    test_text = ax.text(0, 0, test_str, fontsize=16, fontweight='bold', visible=False)
    test_bbox = test_text.get_window_extent(renderer=renderer)
    avg_char_width = test_bbox.width / len(test_str)
    test_text.remove()

    # Compute max chars per line that fit in label area
    max_chars_per_line = max(10, int(label_area_px // avg_char_width))

    # Wrap labels accordingly
    wrapped_labels = []
    for label in labels:
        wrapped = "\n".join(textwrap.wrap(str(label), width=max_chars_per_line))
        wrapped_labels.append(wrapped)

    # Remove y-axis ticks and labels
    ax.set_yticks([])  # No ticks at y axis
    ax.set_yticklabels([])  # No tick labels at y axis

    ax.invert_yaxis()
    ax.set_xlabel('Number of Genes', fontsize=20)
    ax.set_title(f'Top {top_n} Up vs Down-Regulated Pathways\n(Bar color ∝ signed –log₁₀(FDR))',
                 fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)

    # Add pathway labels at left (for upregulated) and right (for downregulated), wrapped to fit
    for i, (v, wrapped_label) in enumerate(zip(values, wrapped_labels)):
        if v > 0:
            # Upregulated: label on left side of bar
            ax.text(-0.5, i, wrapped_label, va='center', ha='right', fontsize=16, fontweight='bold', color='#155724', wrap=True, clip_on=True)
            # Value at end of bar
            ax.text(v + 0.5, i, str(abs(v)), va='center', ha='left', fontweight='bold', fontsize=14)
        else:
            # Downregulated: label on right side of bar
            ax.text(0.5, i, wrapped_label, va='center', ha='left', fontsize=16, fontweight='bold', color='#7f1d1d', wrap=True, clip_on=True)
            # Value at end of bar
            ax.text(v - 0.5, i, str(abs(v)), va='center', ha='right', fontweight='bold', fontsize=14)

    # Make sure everything fits
    fig.subplots_adjust(left=0.35, right=0.98, top=0.92, bottom=0.05)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('signed –log₁₀(FDR)\n(positive=Up, negative=Down)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick label size

    #plt.tight_layout()

    up_down_pathways_plot_b64 = _fig_to_base64()
    
    return {
        "significant": sig_dict,                # filtered & annotated DataFrame
        "significant_df": sig[sig["Confidence_Level"].isin(["High", "Medium"])],
        "category_counts": main_counts,   # e.g., {'Immune': 7, 'Metabolism': 3}
        "pie_b64": pie_b64,
        "pathway_activity_plot_b64": pathway_activity_plot_b64,
        "up_down_pathways_plot_b64": up_down_pathways_plot_b64,
        "high_priority_pathways": len(sig)
    }