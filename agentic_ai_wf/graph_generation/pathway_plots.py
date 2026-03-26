import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np


def safe_float(value):
    """Convert value to float, replacing NaN/inf with 0.0 for JSON safety"""
    # Handle non-numeric types (strings, etc.)
    if not isinstance(value, (int, float, np.number)):
        return value  # Return as-is for strings
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def safe_float_list(values):
    """Convert list of values to safe floats"""
    return [safe_float(v) for v in values]





def plot_pathway_class_distribution(pathway_path: str):
    """
    Read CSV, tally counts in 'Main_Class', draw a donut chart,
    place class+percent labels to the right as a legend, encode to Base64,
    and return {'image', 'data', 'meta_data'}.
    """
    # 1) load & validate
    df = pd.read_csv(pathway_path)
    if 'Main_Class' not in df.columns:
        return {"error": f"'Main_Class' column not found in {pathway_path}"}

    # 2) tally up
    counts = df['Main_Class'].value_counts()
    classes = counts.index.tolist()
    values  = counts.values.tolist()
    total   = sum(values)
    # compute percents
    percents = [v/total*100 for v in values]

    # 3) plot donut without labels on slices
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        values,
        radius=1.0,
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='white')
    )
    # add white center circle
    centre = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre)

    ax.set_title('Top Dysregulated Pathway Classes', fontsize=16, fontweight='bold')
    ax.axis('equal')  # keep it circular

    # 4) build legend entries "Class (xx.x%)"
    legend_labels = [
        f"{cls} ({pct:.1f}%)"
        for cls, pct in zip(classes, percents)
    ]
    # place legend to the right
    ax.legend(
        wedges,
        legend_labels,
        title="Pathway Classes",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=10,
        title_fontsize=12
    )

    plt.tight_layout()

    # 5) encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 6) return
    return {
        'image': img_b64,
        'data': {'classes': classes, 'counts': values, 'percents': safe_float_list(percents)},
        'meta_data': {'title': 'Pathway Class Distribution'}
    }



def plot_up_down_pathways(pathway_path: str, top_n: int = 5):
    """
    Plot the top N Up- and Down-regulated pathways by gene count,
    colored by signed –log10(FDR), and return a dict with:
      - 'image': base64 PNG
      - 'data': labels, values, and signed sig
      - 'meta_data': title
    """
    df = pd.read_csv(pathway_path)
    # 1) select top N up/down
    up   = df[df['Regulation']=='Up'].nlargest(top_n, 'Number_of_Genes').copy()
    down = df[df['Regulation']=='Down'].nlargest(top_n, 'Number_of_Genes').copy()
    if up.empty and down.empty:
        return {"error": "No Up or Down pathways in your DataFrame."}

    # 2) compute signed significance with safe log transformation
    up['FDR'] = up['FDR'].clip(lower=1e-10)  # Avoid log(0)
    down['FDR'] = down['FDR'].clip(lower=1e-10)  # Avoid log(0)
    
    up['sig']   = -np.log10(up['FDR'])
    down['sig'] = -np.log10(down['FDR'])
    
    # Handle infinite values
    up['sig'] = up['sig'].replace([np.inf, -np.inf], 0).fillna(0)
    down['sig'] = down['sig'].replace([np.inf, -np.inf], 0).fillna(0)
    
    up['sig_signed']   = up['sig']
    down['sig_signed'] = -down['sig']

    # 3) normalize & colormap
    all_sig = pd.concat([up['sig_signed'], down['sig_signed']])
    vmax   = all_sig.abs().max()
    norm   = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap   = plt.cm.bwr
    colors = cmap(norm(pd.concat([up['sig_signed'], down['sig_signed']])))

    # 4) labels & values
    labels = list(up['Pathway_Name']) + list(down['Pathway_Name'])
    values = list(up['Number_of_Genes']) + list(-down['Number_of_Genes'])

    # 5) build the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor='black', height=0.6)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Genes')
    ax.set_title(f'Top {top_n} Up vs Down-Regulated Pathways\n(Bar color ∝ signed –log₁₀(FDR))',
                 fontsize=14, fontweight='bold')

    # 6) annotate counts
    for i, v in enumerate(values):
        ax.text(v + (1 if v>0 else -1), i, str(abs(v)),
                va='center', ha='left' if v>0 else 'right', fontweight='bold')

    # 7) colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('signed –log₁₀(FDR)\n(positive=Up, negative=Down)')

    #plt.tight_layout()

    # 8) encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {
        "image": img_b64,
        "data": {
            "labels": labels,
            "values": safe_float_list(values),
            "sig_signed": safe_float_list(list(up['sig_signed']) + list(down['sig_signed']))
        },
        "meta_data": {"title": f"Up vs Down Pathways (top {top_n})"}
    }



def plot_pathway_activity(pathway_path:str):
    """
    Plot the 10 pathways with the highest LLM_Score as horizontal bars—
    colored by severity—and return a Base64‐encoded PNG plus data/meta.
    """
    df = pd.read_csv(pathway_path)
    # 1) compute activity & severity
    df = df.copy()
    df['activity'] = df['Number_of_Genes'] / df['Number_of_Genes_in_Background']
    def classify_severity(r):
        if r > 0.25:   return 'Severe'
        elif r > 0.15: return 'Active'
        else:          return 'Moderate'
    df['severity'] = df['activity'].apply(classify_severity)

    # 2) color map
    color_map = {
        'Severe':   '#e63946',
        'Active':   '#f4a261',
        'Moderate': '#2a9d8f',
    }

    # 3) top 10 by highest LLM_Score
    top10 = df.nlargest(10, 'LLM_Score').reset_index(drop=True)

    # 4) build the figure
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(top10))
    # grey background track
    ax.barh(y, [1]*len(top10), color='#ececec', height=0.6)
    # overlay colored activity bars
    for i, row in top10.iterrows():
        ax.barh(i, row['activity'],
                color=color_map[row['severity']],
                height=0.6)
    # severity labels
    for i, sev in enumerate(top10['severity']):
        ax.text(1.02, i, sev, va='center', ha='left',
                fontsize=9, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(top10['Pathway_Name'])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Gene Ratio')
    ax.invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Pathway Activity (Top 10 LLM_Score)', loc='left',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 5) encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 6) return image + data + meta with safe JSON serialization
    return {
        'image': img_b64,
        'data': {
            'pathways': top10['Pathway_Name'].tolist(),
            'activity': safe_float_list(top10['activity'].tolist()),
            'severity': safe_float_list(top10['severity'].tolist())
        },
        'meta_data': {'title': 'Pathway Activity Plot'}
    }



# pathway_path = r"C:\Ayass Bio Work\final_pathways.csv"


# result = plot_pathway_class_distribution(pathway_path)

# png_bytes = base64.b64decode(result['image'])
# buf = io.BytesIO(png_bytes)
# # read & show
# img = plt.imread(buf, format='png')
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.axis('off')
# plt.show()