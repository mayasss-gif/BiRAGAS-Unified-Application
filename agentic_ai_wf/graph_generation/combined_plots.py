import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np
from itertools import chain


def safe_float(value):
    """Convert value to float, replacing NaN/inf with 0.0 for JSON safety"""
    # Handle non-numeric types (strings, etc.)
    if not isinstance(value, (int, float, np.number)):
        return value  # Return as-is for strings
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def safe_float_dict(series_or_dict):
    """Convert pandas series or dict to dict with NaN-safe floats"""
    if hasattr(series_or_dict, 'items'):
        return {k: safe_float(v) for k, v in series_or_dict.items()}
    else:
        return {k: safe_float(v) for k, v in series_or_dict.to_dict().items()}


def safe_records(df):
    """Convert DataFrame to records with NaN-safe values"""
    records = df.to_dict(orient='records')
    safe_records = []
    for record in records:
        safe_record = {}
        for key, value in record.items():
            if isinstance(value, (int, float)):
                safe_record[key] = safe_float(value)
            else:
                safe_record[key] = value
        safe_records.append(safe_record)
    return safe_records



def get_pathway_names(pathway_path: str):
    pathways_df = pd.read_csv(pathway_path)
    pathways_df = pathways_df[pathways_df['Ontology_Source']=='KEGG']
    pathways_df = pathways_df[pathways_df['LLM_Score']>=70]
    pathways = pathways_df['Pathway_Name'].tolist()

    return pathways



def plot_pathway_genes(
    pathway_name: str,
    degs_csv: str,
    pathways_csv: str,
    lfc_col: str = 'Patient_LFC_mean',
    tier_col: str = 'Tier',
    ppi_col: str = 'PPI_Degree',
    llm_cutoff: float = 70.0
) -> dict:
    """
    Read DEGs and pathways CSVs, filter for a given KEGG pathway with LLM ≥ cutoff,
    plot Patient mean log₂FC bars colored by Tier, overlay PPI‐degree circles,
    encode the figure to Base64, and return a dict:
      {
        "image": <base64‐png string>,
        "data": [ {Gene:…, Patient_LFC_mean:…, Tier:…, PPI_Degree:…}, … ],
        "meta_data": {"pathway": pathway_name, "title": ...}
      }
    """
    # 1) load data
    degs_df     = pd.read_csv(degs_csv)
    pathways_df = pd.read_csv(pathways_csv)

    # 2) filter KEGG & LLM
    kegg = pathways_df[
        (pathways_df['Ontology_Source']=='KEGG') &
        (pathways_df['LLM_Score']>=llm_cutoff)
    ]

    # 2a) get top 5 pathways by Priority_Rank
    top5 = kegg.nsmallest(5, 'Priority_Rank')['Pathway_Name'].tolist()
    # print(f"Top 5 KEGG pathways (LLM ≥ {llm_cutoff}): {top5}")
    # 3) pick the best (lowest rank) pathway
    if pathway_name is None:
        if not top5:
            return {"error": "No pathways meet the LLM cutoff."}
        pathway_name = top5[0]
    pw = kegg[kegg['Pathway_Name']==pathway_name]
    if pw.empty:
        return {"error": f"Pathway '{pathway_name}' not found or below LLM cutoff."}
    
    # 4) get genes & subset
    genes = [g.strip() for g in pw.iloc[0]['Pathway_Associated_Genes'].split(',')]
    sub = degs_df[degs_df['Gene'].isin(genes)].dropna(subset=[lfc_col]).copy()
    if sub.empty:
        return {"error": f"No DEGs found for pathway '{pathway_name}'."}

    # 5) coloring by tier
    sub['Tier_num'] = sub[tier_col].str.extract(r'(\d+)').astype(int)
    max_tier = sub['Tier_num'].max()
    palette = sns.color_palette("Blues_r", n_colors=max_tier)
    sub['color'] = sub['Tier_num'].apply(lambda t: palette[t-1])

    # 6) sizing by PPI degree
    ppis = sub[ppi_col].fillna(0)
    norm = (ppis - ppis.min())/(ppis.max()-ppis.min()+1e-6)
    sub['ppi_size'] = 50 + norm * 250

    # 7) plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(sub)*0.4)))
    ax.barh(sub['Gene'], sub[lfc_col], color=sub['color'], edgecolor='k')
    ax.scatter(sub[lfc_col], sub['Gene'], s=sub['ppi_size'],
               facecolors='none', edgecolors='gray', linewidths=1.2)
    ax.axvline(0, color='k', linestyle='--', lw=0.8)

    # add value labels
    for _, row in sub.iterrows():
        ax.text(
            row[lfc_col] + 0.02*abs(row[lfc_col]),
            row['Gene'],
            f"{row[lfc_col]:.2f}",
            va='center', fontsize=9
        )

    rank = int(pw.iloc[0]['Priority_Rank'])
    ax.set_title(f"{pathway_name} (Rank {rank})", fontsize=12, fontweight='bold')
    ax.set_xlabel('Patient Mean log₂FC')
    sns.despine(ax=ax, left=True, bottom=True)

    # legend for tiers
    for t in sorted(sub['Tier_num'].unique()):
        ax.bar(0,0, color=palette[t-1], label=f"Tier {t}")
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()

    # 8) encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # 9) return with safe JSON serialization
    return {
        "image": img_b64,
        "data": safe_records(sub[['Gene', lfc_col, tier_col, ppi_col]]),
        "meta_data": {"top5": top5, "pathway": pathway_name, "title": f"{pathway_name} Pathway Plot"}
    }



def plot_cross_talk_enhanced(
    degs_csv: str,
    pathways_csv: str,
    llm_cutoff: float = 70.0,
    top_n: int = 5,
    lfc_col: str = "Patient_LFC_mean",
    ppi_col: str = "PPI_Degree",
    tier_col: str = "Tier",
):
    """
    Read DEGs and pathways CSVs, pick the top_n KEGG pathways with LLM ≥ llm_cutoff,
    build a cross‐talk dot‐matrix, encode the figure to Base64, and return:
      {
        "image": <base64‐png string>,
        "data": {
          "genes": [...],
          "pathways": [...],
          "sizes": {gene: size, ...},
          "colors": {gene: color, ...},
          "lfc": {gene: lfc_value, ...}
        },
        "meta_data": {"title": "Cross-Talk Between Top Pathways: Key Genes"}
      }
    """
    # 1) Load tables
    degs_df     = pd.read_csv(degs_csv)
    pathways_df = pd.read_csv(pathways_csv)

    # 2) Top KEGG pathways by Priority_Rank
    kegg = (
        pathways_df[(pathways_df["Ontology_Source"] == "KEGG") &
                    (pathways_df["LLM_Score"] >= llm_cutoff)]
        .nsmallest(top_n, "Priority_Rank")
    )
    paths = kegg["Pathway_Name"].tolist()

    # 3) Build membership
    path_genes = {
        pw: [g.strip() for g in genes.split(",")]
        for pw, genes in zip(kegg["Pathway_Name"], kegg["Pathway_Associated_Genes"])
    }
    all_genes = sorted(set(chain.from_iterable(path_genes.values())))
    membership = pd.DataFrame(0, index=all_genes, columns=paths)
    for pw, genes in path_genes.items():
        membership.loc[genes, pw] = 1

    # 4) Filter cross‐talk genes (in ≥2 pathways)
    cross_counts = membership.sum(axis=1)
    cross_genes  = cross_counts[cross_counts >= 2].index.tolist()
    if not cross_genes:
        return {"error": "No genes shared across multiple pathways."}
    sub_mem = membership.loc[cross_genes]

    # 5) Gather metadata for those genes
    meta = (
        degs_df.set_index("Gene")[[lfc_col, ppi_col, tier_col]]
        .loc[cross_genes]
        .fillna({ppi_col: 0})
    )
    meta["Tier_num"] = meta[tier_col].str.extract(r"(\d+)").astype(int)

    # 6) Compute dot sizes (sqrt scaling) with NaN safety
    raw_ppi = meta[ppi_col].astype(float).fillna(0)  # Handle NaN
    raw_ppi = raw_ppi.replace([np.inf, -np.inf], 0)  # Handle infinite values
    sqrt_ppi = np.sqrt(raw_ppi.clip(lower=0))  # Avoid sqrt of negative
    min_s, max_s = 150, 1000
    
    # Normalize with safety checks
    min_val = sqrt_ppi.min()
    max_val = sqrt_ppi.max()
    
    # Additional safety for infinite/NaN values after sqrt
    if pd.isna(min_val) or pd.isna(max_val) or np.isinf(min_val) or np.isinf(max_val) or max_val == min_val:
        norm_ppi = pd.Series([0.5] * len(sqrt_ppi), index=sqrt_ppi.index)
    else:
        norm_ppi = (sqrt_ppi - min_val) / (max_val - min_val + 1e-6)
        # Ensure normalization result is safe
        norm_ppi = norm_ppi.fillna(0.5).replace([np.inf, -np.inf], 0.5)
    
    sizes = safe_float_dict(min_s + norm_ppi * (max_s - min_s))

    # 7) Compute dot colors by tier
    tier_colors = {1: "red", 2: "orange", 3: "yellow"}
    colors      = meta["Tier_num"].map(lambda t: tier_colors.get(t, "blue")).to_dict()

    # 8) Plot in‐memory
    fig, ax = plt.subplots(
        figsize=(1 + len(paths)*1.8, max(8, len(cross_genes)*0.5))
    )
    for i, gene in enumerate(cross_genes):
        for j, pw in enumerate(paths):
            if sub_mem.at[gene, pw] == 1:
                ax.scatter(
                    j, i,
                    s=sizes[gene],
                    color=colors[gene],
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.7,
                )

    ax.set_xticks(range(len(paths)))
    ax.set_xticklabels(paths, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(cross_genes)))
    ax.set_yticklabels(cross_genes, fontsize=9)
    ax.set_xlim(-0.5, len(paths)-0.5)
    ax.set_ylim(-0.5, len(cross_genes)-0.5)
    ax.set_xlabel("Top KEGG Pathways (LLM ≥ 70)", fontsize=12)
    ax.set_ylabel("Cross-Talk Genes (in ≥2 pathways)", fontsize=12)
    ax.set_title("Cross-Talk Between Top Pathways: Key Genes",
                 fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    # Legend outside
    handles, labels = [], []
    for t, col in tier_colors.items():
        h = ax.scatter([], [], s=300, color=col,
                       edgecolors="black", linewidth=0.7)
        handles.append(h)
        labels.append(f"Tier {t}")
    ax.legend(handles, labels,
              title="Gene Tier",
              loc="upper left",
              bbox_to_anchor=(1.02, 1),
              frameon=False,
              fontsize=10,
              title_fontsize=12)

    plt.subplots_adjust(right=0.75, top=0.9)
    plt.tight_layout()

    # 9) Encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # 10) Return everything with safe JSON serialization
    return {
        "image": img_b64,
        "data": {
            "genes": cross_genes,
            "pathways": paths,
            "sizes": sizes,
            "colors": colors,
            "lfc": safe_float_dict(meta[lfc_col])
        },
        "meta_data": {"title": "Cross-Talk Between Top Pathways: Key Genes"}
    }



# degs_path = r"C:\Ayass Bio Work\DEGs.csv"
# pathway_path = r"C:\Ayass Bio Work\final_pathways.csv"

# degs_path = r"C:\Ayass Bio Work\Agentic_AI_ABS\graph_generation\agentic_ai_abs\Lupus_DEGs_prioritized.csv"
# pathway_path = r"C:\Ayass Bio Work\Agentic_AI_ABS\graph_generation\agentic_ai_abs\Lupus_Pathways_Consolidated.csv"


# result = plot_cross_talk_enhanced(
#     degs_csv=degs_path,
#     pathways_csv=pathway_path,
#     llm_cutoff=70.0,
#     top_n=5,
# )

# png_bytes = base64.b64decode(result['image'])
# buf = io.BytesIO(png_bytes)
# # read & show
# img = plt.imread(buf, format='png')
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.axis('off')
# plt.show()