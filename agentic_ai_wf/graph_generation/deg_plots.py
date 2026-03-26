import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np


def safe_float(value):
    """Convert value to float, replacing NaN/inf with 0.0 for JSON safety"""
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def safe_float_dict(series_or_dict):
    """Convert pandas series or dict to dict with NaN-safe floats"""
    if hasattr(series_or_dict, 'items'):
        return {k: safe_float(v) for k, v in series_or_dict.items()}
    else:
        return {k: safe_float(v) for k, v in series_or_dict.to_dict().items()}


def plot_top20_variance_heatmap(degs_path: str) -> dict:
    """
    If Cohort_LFC_mean exists:
      • Compute Δ = Patient_LFC_mean – Cohort_LFC_mean, pick top20 by |Δ|, heatmap both means.
    Else:
      • Compute variance across all Patient_*_log2FC cols, pick top20 by variance,
        heatmap raw per-sample LFCs.
    Returns:
      {
        "image": <base64-png>,
        "data": { ... },
        "meta_data": {"title": ...}
      }
    """
    df = pd.read_csv(degs_path)
 
    # Detect cohort means
    has_cohort = 'Cohort_LFC_mean' in df.columns and not df['Cohort_LFC_mean'].isna().all()
 
    if has_cohort:
        # Δ-based ranking
        df['delta'] = df['Patient_LFC_mean'] - df['Cohort_LFC_mean']
        df['abs_delta'] = df['delta'].abs()                # <— new column
        subset = (
            df.nlargest(20, 'abs_delta')                   # rank by abs_delta
              .set_index('Gene')
        )
        matrix = subset[['Patient_LFC_mean', 'Cohort_LFC_mean']]
        title = "Top 20 Genes by |Patient–Cohort|"
        data_payload = {
            gene: {
                'Patient_LFC_mean': safe_float(row['Patient_LFC_mean']),
                'Cohort_LFC_mean': safe_float(row['Cohort_LFC_mean']),
                'delta': safe_float(row['delta'])
            }
            for gene, row in subset.iterrows()
        }
 
    else:
        # variance-based fallback
        lfc_cols = [
            "Patient_LFC_mean"
        ]
        if not lfc_cols:
            return {"error": "No Patient _log2FC columns found."}
        df['variance'] = df[lfc_cols].var(axis=1)
        subset = (
            df.nlargest(20, 'variance')
              .set_index('Gene')
        )
        matrix = subset[lfc_cols]
        title = "Top 20 Variable Genes by LFC Variance"
        data_payload = {
            gene: {
                **{c: safe_float(row[c]) for c in lfc_cols},
                'variance': safe_float(row['variance'])
            }
            for gene, row in subset.iterrows()
        }
 
    # draw heatmap
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "log2 FC"},
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel("Gene")
    plt.tight_layout()
 
    # encode to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
 
    return {
        "image": img_b64,
        "data": data_payload,
        "meta_data": {"title": title}
    }



def plot_log2fc_by_origin(degs_path: str, patient_prefix: str = 'Patient', cohort_prefix: str = 'GSE') -> dict:
    """
    Plot log2FC distributions for Patient vs Cohort groups:
      - If both Patient & Cohort present: overlaid histograms, side-by-side boxplots, two stat tables.
      - If only Patient present: single histogram, single boxplot, one stat table.
    Returns:
      {
        'image': base64-PNG,
        'stats': {'Patient': {...}, 'Cohort': {...}?},
        'meta': {'title': 'log2FC by Origin'}
      }
    """
    df = pd.read_csv(degs_path)
    if df.empty:
        return {"error": "No data provided. Please supply a non-empty DataFrame."}

    # identify log2FC columns
    log2fc_cols = [c for c in df.columns if c.lower().endswith('_log2fc')]
    if not log2fc_cols:
        return {"error": "No 'log2FC' columns found in DataFrame."}

    # split into Patient vs Cohort
    patient_cols = [c for c in log2fc_cols if c.lower().startswith(patient_prefix.lower())]
    cohort_cols  = [c for c in log2fc_cols if c.lower().startswith(cohort_prefix.lower())]

    if not patient_cols:
        return {"error": "No Patient log2FC columns found."}

    def extract_series(cols):
        return pd.to_numeric(df[cols].stack(), errors='coerce').dropna()

    patient_series = extract_series(patient_cols)
    cohort_series  = extract_series(cohort_cols) if cohort_cols else pd.Series(dtype=float)

    def compute_stats(s: pd.Series) -> dict:
        Q1, Q3 = s.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = (s < (Q1 - 1.5*IQR)) | (s > (Q3 + 1.5*IQR))
        return {
            'count': int(s.count()),
            'mean': safe_float(s.mean()),
            'median': safe_float(s.median()),
            'std': safe_float(s.std()),
            'min': safe_float(s.min()),
            'max': safe_float(s.max()),
            'skewness': safe_float(s.skew()),
            'Q1': safe_float(Q1),
            'Q3': safe_float(Q3),
            'outlier_count': int(outliers.sum()),
            'outlier_pct': safe_float(outliers.mean()*100)
        }

    stats = {'Patient': compute_stats(patient_series)}
    has_cohort = not cohort_series.empty
    if has_cohort:
        stats['Cohort'] = compute_stats(cohort_series)

    # create subplots with appropriate layout
    if has_cohort:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax_hist = axs[0, 0]
        ax_box  = axs[0, 1]
        ax_stat_patient = axs[1, 0]
        ax_stat_cohort  = axs[1, 1]
    else:
        fig, axs = plt.subplots(3, 1, figsize=(6, 15))
        ax_hist = axs[0]
        ax_box  = axs[1]
        ax_stat_patient = axs[2]

    # plot histogram(s)
    bins = 30
    ax_hist.hist(patient_series, bins=bins, alpha=0.6, label='Patient')
    if has_cohort:
        ax_hist.hist(cohort_series, bins=bins, alpha=0.6, label='Cohort')
    ax_hist.set_title('log2FC Distribution by Origin')
    ax_hist.set_xlabel('log2FC')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # plot boxplot(s)
    data = [patient_series]
    labels = ['Patient']
    colors = ['#79C']
    if has_cohort:
        data.append(cohort_series)
        labels.append('Cohort')
        colors.append('#C79')
    bplot = ax_box.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax_box.set_title('Box Plot by Origin')
    ax_box.set_ylabel('log2FC')
    ax_box.grid(alpha=0.3)

    # plot stats table(s)
    def draw_table(ax, stats_dict, title):
        ax.axis('off')
        rows = [[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in stats_dict.items()]
        tbl = ax.table(rows, colLabels=['Stat', 'Value'], cellLoc='center', loc='center')
        ax.set_title(title)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

    draw_table(ax_stat_patient, stats['Patient'], 'Patient Stats')
    if has_cohort:
        draw_table(ax_stat_cohort, stats['Cohort'], 'Cohort Stats')

    plt.tight_layout()

    # encode figure to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {
        'image': img_b64,
        'data': stats,
        'meta': {'title': 'log2FC by Origin'}
    }


def plot_deg_dotplot(degs_path: str, patient_prefix: str = 'Patient'):
    """
    Create a dot‐plot of top 10 DEGs (size ∝ –log10(p), color ∝ LFC),
    encode the figure to Base64, and return image + data + meta_data.
    """
    # Compute sizes and color normalization
    df = pd.read_csv(degs_path)
    Patient_pvalues_cols = [col for col in df.columns if patient_prefix.lower() in col.lower() and "_p-value" in col.lower()]
    df["Patient_P-value_Mean"] = df[Patient_pvalues_cols].mean(axis=1)
    df = df.copy()
    
    # Safe log transformation - handle zero/negative p-values
    df["Patient_P-value_Mean"] = df["Patient_P-value_Mean"].clip(lower=1e-10)  # Avoid log(0)
    df["size"] = -np.log10(df["Patient_P-value_Mean"]) * 100
    df["size"] = df["size"].replace([np.inf, -np.inf], 0).fillna(0)
    norm = plt.Normalize(vmin=df["Patient_LFC_mean"].min(),
                        vmax=df["Patient_LFC_mean"].max())
    cmap = plt.get_cmap("bwr")

    # Select top 10 by LFC
    df_sorted = df.sort_values("Patient_LFC_mean", ascending=False).head(10)

    # Build the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df_sorted["Patient_LFC_mean"],
        df_sorted["Gene"],
        s=df_sorted["size"],
        c=df_sorted["Patient_LFC_mean"],
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.5
    )
    plt.colorbar(sc, ax=ax, label="log2 Fold Change")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("Gene")
    ax.set_title("Dot‐Plot of DEGs\n(Size ∝ significance, Color ∝ log2FC)")
    ax.grid(axis="x", linestyle=":", alpha=0.7)
    plt.tight_layout()

    # Encode to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Prepare summary data
    data = {
        "genes": df_sorted["Gene"].tolist(),
        "lfc": [round(x, 3) for x in df_sorted["Patient_LFC_mean"].tolist()],
        "sizes": [round(x, 2) for x in df_sorted["size"].tolist()]
    }
    meta_data = {"title": "Dot‐Plot of DEGs"}

    return {"image": img_b64, "data": data, "meta_data": meta_data}



def plot_mini_volcano(degs_path: str, top_n: int = 10, patient_prefix: str = 'Patient'):
    """
    Create a mini‐volcano plot of all DEGs, label the top N by LFC, and
    return a Base64‐encoded PNG plus data and meta_data.

    :param df: DataFrame with columns 'Patient_LFC_mean', 'Patient_P-value_Mean', 'Gene'
    :param top_n: how many top genes (by LFC) to label
    :return: dict with 'image', 'data', 'meta_data' or {'error': msg}
    
    """
    df = pd.read_csv(degs_path)
    Patient_pvalues_cols = [
    col
    for col in df.columns
    if patient_prefix.lower() in col.lower() and "_p-value" in col.lower()
]
    df["Patient_P-value_Mean"] = df[Patient_pvalues_cols].mean(axis=1)
    df = df.copy()

    # validation
    required = ["Patient_LFC_mean", "Patient_P-value_Mean", "Gene"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        return {"error": f"Missing columns: {missing}"}

    # prepare data with safe log transformation
    df = df.copy()
    df["Patient_P-value_Mean"] = df["Patient_P-value_Mean"].clip(lower=1e-10)  # Avoid log(0)
    df["neg_log_p"] = -np.log10(df["Patient_P-value_Mean"])
    df["neg_log_p"] = df["neg_log_p"].replace([np.inf, -np.inf], 0).fillna(0)
    norm = plt.Normalize(df["Patient_LFC_mean"].min(), df["Patient_LFC_mean"].max())
    cmap = plt.get_cmap("bwr")

    # select top N by abs(LFC)
    top_idx = df["Patient_LFC_mean"].abs().nlargest(top_n).index
    top_df = df.loc[top_idx]

    # build figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["Patient_LFC_mean"],
        df["neg_log_p"],
        c=df["Patient_LFC_mean"],
        cmap=cmap,
        norm=norm,
        s=100,
        edgecolor="black",
        linewidth=0.5
    )

    # label top N
    for _, row in top_df.iterrows():
        xi, yi, gene = row["Patient_LFC_mean"], row["neg_log_p"], row["Gene"]
        ax.text(
            xi, yi, gene,
            fontsize=8,
            ha="right" if xi < 0 else "left",
            va="bottom"
        )

    # embellish
    plt.colorbar(sc, ax=ax, label="log2 Fold Change")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", label="p = 0.05")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Mini‐Volcano Plot of DEGs")
    ax.legend(frameon=False)
    plt.tight_layout()

    # encode to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # prepare summary data
    data = {
        "top_genes": df["Gene"].tolist(),
        "top_lfc": [safe_float(x) for x in df["Patient_LFC_mean"].tolist()],
        "top_neg_log_p": [safe_float(x) for x in df["neg_log_p"].tolist()]
    }
    meta_data = {"title": "Mini‐Volcano Plot"}

    return {"image": img_b64, "data": data, "meta_data": meta_data}



def plot_mean_lfc(degs_path: str):
    """
    Read the CSV at `path`, compute mean ± SEM for Patient vs. Cohort log₂FC,
    draw a styled bar chart, encode it to Base64, and return a dict:
      {
        "image": <base64‐png string>,
        "data": {"means": {...}, "sems": {...}},
        "meta_data": {"title": ...}
      }
    """
    # 1) Load and validate
    df = pd.read_csv(degs_path)
    required = ["Patient_LFC_mean", "Cohort_LFC_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    # 2) Compute statistics with NaN handling
    means = df[required].mean()
    sems  = df[required].sem()
    
    # Handle NaN values - replace with 0 for safe JSON serialization
    means = means.fillna(0)
    sems = sems.fillna(0)
    
    # Additional validation
    if means.isna().any() or sems.isna().any():
        return {"error": "Unable to compute valid statistics - insufficient data"}

    # 3) Build the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868"]  # blue / green
    bars = ax.bar(
        means.index,
        means.values,
        yerr=sems.values,
        capsize=8,
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )

    # Labels & title
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(["Patient", "Cohort"], fontsize=12)
    ax.set_ylabel("Mean log₂ Fold Change", fontsize=12)
    ax.set_title("Mean log₂FC: Patient vs. Cohort", fontsize=14, fontweight="bold")

    # Annotate above the error bars
    ax.bar_label(
        bars,
        labels=[f"{m:.2f}" for m in means.values],
        padding=6,
        fontsize=11,
        fontweight="semibold"
    )

    # Polish axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", color="gray", alpha=0.7)
    ax.set_axisbelow(True)


    # 4) Encode to Base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # 5) Return the payload with safe JSON serialization
    return {
        "image": img_b64,
        "data": {
            "means": safe_float_dict(means),
            "sems": safe_float_dict(sems)
        },
        "meta_data": {
            "title": "Mean log₂FC: Patient vs. Cohort"
        }
    }



# degs_path = r"C:\Ayass Bio Work\DEGs.csv"


# result = plot_mean_lfc(degs_path)
# # print(result['data'])

# png_bytes = base64.b64decode(result['image'])
# buf = io.BytesIO(png_bytes)
# # read & show
# img = plt.imread(buf, format='png')
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.axis('off')
# plt.show()