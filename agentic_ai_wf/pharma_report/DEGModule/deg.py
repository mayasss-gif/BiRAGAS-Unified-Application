import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import seaborn as sns
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def plot_and_get_base64(plot_func, **kwargs):
    """Render a plot and return its base64-encoded PNG image."""
    fig = plot_func(**kwargs)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def plot_deg_distribution_by_fold_change(csv_path):
    """
    Creates and returns a matplotlib Figure showing the distribution of DEGs by fold-change.
    Assumes the input CSV contains columns: 'Gene', 'Patient_LFC_mean', and 'Patient_LFC_Trend'.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """

    # Load the data
    df = pd.read_csv(csv_path)

    # Check for required columns
    required_columns = {'Gene', 'Patient_LFC_mean', 'Patient_LFC_Trend'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain the columns: {required_columns}")

    # Categorize based on log2 fold change
    def categorize_fc(fc):
        abs_fc = abs(fc)
        if 1 <= abs_fc < 2:
            return 'Mild (2-4x)'
        elif 2 <= abs_fc <= 4:
            return 'Moderate (4-16x)'
        elif abs_fc > 4:
            return 'Extreme (>16x)'
        else:
            return None

    df['FC_Category'] = df['Patient_LFC_mean'].apply(categorize_fc)
    df = df.dropna(subset=['FC_Category'])

    # Count by category and trend
    summary = df.groupby(['FC_Category', 'Patient_LFC_Trend']
                         ).size().unstack(fill_value=0)

    # Reorder categories
    category_order = ['Mild (2-4x)', 'Moderate (4-16x)', 'Extreme (>16x)']
    summary = summary.reindex(category_order, fill_value=0)

    # Create the plot
    fig, ax = plt.subplots()
    summary.plot(kind='bar', stacked=False, color=[
                 'red', 'cornflowerblue'], ax=ax)

    ax.set_title('Distribution of DEGs by Fold-Change')
    ax.set_ylabel('Number of Genes')
    ax.set_xlabel('')
    ax.legend(title='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()

    return fig


def plot_mini_volcano(csv_path: str, patient_id, top_n: int = 10):
    """
    Create a mini-volcano plot of DEGs and return the figure.

    :param csv_path: Path to the CSV containing DEG data
    :param top_n: Number of top genes (by |LFC|) to label
    :return: matplotlib.figure.Figure
    """
    df = pd.read_csv(csv_path)

    # Compute mean p-value
    pval_cols = [col for col in df.columns if patient_id in col.lower()
                 and "_p-value" in col.lower()]
    df["Patient_P-value_Mean"] = df[pval_cols].mean(axis=1)

    # Validation
    required = ["Patient_LFC_mean", "Patient_P-value_Mean", "Gene"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"Missing columns: {missing}")

    df["neg_log_p"] = -np.log10(df["Patient_P-value_Mean"])
    norm = plt.Normalize(df["Patient_LFC_mean"].min(),
                         df["Patient_LFC_mean"].max())
    cmap = plt.get_cmap("bwr")

    # Select top N genes by abs LFC
    top_idx = df["Patient_LFC_mean"].abs().nlargest(top_n).index
    top_df = df.loc[top_idx]

    # Create plot
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

    # Label top N genes
    for _, row in top_df.iterrows():
        xi, yi, gene = row["Patient_LFC_mean"], row["neg_log_p"], row["Gene"]
        ax.text(
            xi, yi, gene,
            fontsize=8,
            ha="right" if xi < 0 else "left",
            va="bottom"
        )

    plt.colorbar(sc, ax=ax, label="log₂ Fold Change")
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", label="p = 0.05")
    ax.set_xlabel(r'$\log_{2}\mathrm{FC}$')
    ax.set_ylabel('Log -10 value')
    ax.set_title("Mini‐Volcano Plot of DEGs")
    ax.legend(frameon=False)
    plt.tight_layout()

    return fig


def plot_top20_variance_heatmap(csv_path: str, patient_id: str) -> plt.Figure:
    """
    Reads a CSV and selects the top 20 genes based on available data:
    - If both Patient_LFC_mean and Cohort_LFC_mean columns exist: rank by |Patient_LFC_mean - Cohort_LFC_mean|.
    - Elif Cohort_LFC_mean exists and exactly one Patient_*_log2FC sample column: rank by |sample - Cohort_LFC_mean|.
    - Elif no Cohort_LFC_mean and multiple Patient_*_log2FC columns: rank by variance across samples.
    - Elif no Cohort_LFC_mean and one Patient_*_log2FC column: rank by absolute fold-change in that sample.
    Returns a Matplotlib Figure object (to be passed into `plot_and_get_base64`).
    """
    import re
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if 'Gene' not in df.columns:
        raise ValueError("No 'Gene' column found in CSV.")

    # Detect relevant columns using regex
    cohort_regex = re.compile(r'cohort.*lfc.*mean', re.IGNORECASE)
    patient_mean_regex = re.compile(r'patient.*lfc.*mean', re.IGNORECASE)
    patient_sample_regex = re.compile(r'patient.*_log2fc$', re.IGNORECASE)

    cohort_cols = [c for c in df.columns if cohort_regex.match(c)]
    patient_mean_cols = [c for c in df.columns if patient_mean_regex.match(c)]
    patient_sample_cols = [
        c for c in df.columns if patient_sample_regex.match(c)]

    cohort_col = cohort_cols[0] if cohort_cols else None
    patient_mean_col = patient_mean_cols[0] if patient_mean_cols else None

    if cohort_col and patient_mean_col:
        df['delta'] = df[patient_mean_col] - df[cohort_col]
        df['abs_delta'] = df['delta'].abs()
        subset = df.nlargest(20, 'abs_delta').set_index('Gene')
        matrix = subset[[patient_mean_col, cohort_col]]
        title = "Top 20 Genes by |Patient Mean – Cohort Mean|"
        data_payload = {
            gene: {
                patient_mean_col: float(row[patient_mean_col]),
                cohort_col: float(row[cohort_col]),
                'delta': float(row['delta'])
            }
            for gene, row in subset.iterrows()
        }

    elif cohort_col and len(patient_sample_cols) == 1:
        sample = patient_sample_cols[0]
        df['delta'] = df[sample] - df[cohort_col]
        df['abs_delta'] = df['delta'].abs()
        subset = df.nlargest(20, 'abs_delta').set_index('Gene')
        matrix = subset[[sample, cohort_col]]
        title = f"Top 20 Genes by Patient vs Cohort"
        data_payload = {
            gene: {
                sample: float(row[sample]),
                cohort_col: float(row[cohort_col]),
                'delta': float(row['delta'])
            }
            for gene, row in subset.iterrows()
        }

    elif not cohort_col and len(patient_sample_cols) >= 2:
        df['variance'] = df[patient_sample_cols].var(axis=1)
        subset = df.nlargest(20, 'variance').set_index('Gene')
        matrix = subset[patient_sample_cols]
        title = "Top 20 Genes by LFC Variance Across Patients"
        data_payload = {
            gene: {
                **{c: float(row[c]) for c in patient_sample_cols},
                'variance': float(row['variance'])
            }
            for gene, row in subset.iterrows()
        }

    elif not cohort_col and len(patient_sample_cols) == 1:
        sample = patient_sample_cols[0]
        df['abs_fc'] = df[sample].abs()
        subset = df.nlargest(20, 'abs_fc').set_index('Gene')
        matrix = subset[[sample]]
        title = f"Top 20 Genes by Patient LFC"
        data_payload = {gene: float(row[sample])
                        for gene, row in subset.iterrows()}

    else:
        raise ValueError(
            "Insufficient data: need cohort mean or patient sample columns.")

    # Generate plot
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, max(6, len(matrix) * 0.4)))
    if matrix.shape[1] == 1:
        col = matrix.columns[0]
        sorted_vals = matrix[col].sort_values(ascending=False)
        ax.barh(sorted_vals.index, sorted_vals.values)
        ax.set_xlabel(r"$\log_{2}\mathrm{FC}$ Comparison")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.invert_yaxis()
    else:
        sns.heatmap(
            matrix,
            cmap="vlag",
            center=0,
            linewidths=0.5,
            linecolor="grey",
            cbar_kws={"label": "log₂ FC"},
            ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Gene")
        ax.set_xlabel("Sample / Mean")

    plt.tight_layout()

    # Attach metadata to fig
    fig.data_payload = data_payload
    fig.meta_data = {"title": title}

    return fig


def generate_chart_explanation(base64_img):

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical data analyst tasked with summarizing visual cohort analysis charts "
                    "in a domain-specific, research-ready format. Use precise technical language."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please generate a summary for this chart."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64_img
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": (
                    "This plot displays the diversity of biological material and disease annotations within the assembled cohort. "
                    "CD206+ and CD206− macrophages are prominently represented among cell types, while the clinical status breakdown "
                    "shows a predominance of non-diabetic controls. This stratification supports downstream differential expression, "
                    "biomarker identification, and subgroup-specific analyses."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now generate a similar explanation for this chart."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64_img
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


def disease_description(csv_path: str, disease_name: str, model="gpt-4o"):
    """
    Generates a clinical-style disease activity explanation using GPT,
    based on gene expression values from a CSV and a given disease name.

    Args:
        csv_path (str): Path to the CSV file.
        disease_name (str): Name of the disease (e.g., 'Systemic Lupus Erythematosus').
        model (str): OpenAI model (default: gpt-4o).
        max_genes (int): Max number of genes to include in prompt.

    Returns:
        str: GPT-generated explanation.
    """

    # Load and filter
    df = pd.read_csv(csv_path)
    required_cols = {"Gene", "Patient_LFC_mean", "Patient_LFC_Trend"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the columns: {required_cols}")

    df_small = df.loc[:, ["Gene", "Patient_LFC_mean",
                          "Patient_LFC_Trend"]].dropna()
    gene_table = df_small.to_csv(index=False)

    # GPT messages with disease context
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a biomedical data analyst summarizing transcriptomic findings related to {disease_name}. "
                "You receive tables of differentially expressed genes (DEGs) from patient samples. Your job is to generate a "
                "short clinical-style narrative on disease activity, immune dysregulation, and systemic involvement. "
                "Focus on interpreting fold changes, up/down regulation balance, and any transcriptomic signatures relevant to disease severity."
            )
        },
        {
            "role": "user",
            "content": (
                f"Generate a 3–5 sentence clinical summary of transcriptomic activity in the context of {disease_name}, "
                "based on the following gene expression data (log2 fold changes and regulation direction):\n\n"
                f"{gene_table}"
            )
        }
    ]

    # GPT Call
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content.strip()


def deg_profile(csv_path: str, patient_id, model="gpt-4o"):
    """
    Generates a summary of the differential expression profile based on DEGs
    in the uploaded CSV using GPT in a clinical/research tone.

    Args:
        csv_path (str): Path to the CSV file.
        model (str): OpenAI model to use (default: gpt-4o).

    Returns:
        str: GPT-generated explanation paragraph.
    """

    # Load and filter
    df = pd.read_csv(csv_path)
    tr1 = [col for col in df.columns if col.lower().startswith(patient_id)]
    print(tr1)
    adj_pvalue_cols = [col for col in df.columns if col.lower().startswith(
        patient_id.lower()) and "_adj-p-value" in col.lower()]
    if not adj_pvalue_cols:
        raise ValueError(
            f"No adjusted p-value column found for patient ID: {patient_id}")
    adj_pvalue = adj_pvalue_cols[0]
    required_cols = {"Gene", "Patient_LFC_mean",
                     "Patient_LFC_Trend", adj_pvalue}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the columns: {required_cols}")

    # Filter for significant DEGs (adj p < 0.05)
    sig_df = df[df[adj_pvalue] < 0.05].copy()
    sig_df = sig_df.dropna(subset=["Patient_LFC_mean", "Patient_LFC_Trend"])

    if sig_df.empty:
        return "No significant DEGs found (adjusted p < 0.05)."

    total_deg = len(sig_df)
    up_df = sig_df[sig_df["Patient_LFC_Trend"].str.lower() == "up"]
    down_df = sig_df[sig_df["Patient_LFC_Trend"].str.lower() == "down"]

    up_count = len(up_df)
    down_count = len(down_df)
    up_pct = (up_count / total_deg) * 100
    down_pct = (down_count / total_deg) * 100

    up_min, up_max = up_df["Patient_LFC_mean"].min(
    ), up_df["Patient_LFC_mean"].max()
    down_min, down_max = down_df["Patient_LFC_mean"].min(
    ), down_df["Patient_LFC_mean"].max()

    # Build prompt for GPT
    deg_stats_text = (
        f"A total of {total_deg:,} differentially expressed genes (DEGs) were identified, all meeting strict significance "
        f"criteria (adjusted p < 0.05). Of these, {up_count:,} genes are upregulated ({up_pct:.1f}%) and "
        f"{down_count:,} are downregulated ({down_pct:.1f}%).\n\n"
        f"Fold-change ranges:\n"
        f" - Upregulated: {up_min:+.2f} to {up_max:+.2f}\n"
        f" - Downregulated: {down_min:+.2f} to {down_max:+.2f}\n\n"
        "Please use this to generate a 3–5 sentence clinical/research-style explanation focusing on overall expression "
        "profile, inflammatory or immune relevance, and dynamic expression behavior."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical data analyst generating technical summaries of differential gene expression profiles "
                "in patient transcriptomic data. The style should be suitable for use in clinical or translational research."
            )
        },
        {
            "role": "user",
            "content": deg_stats_text
        }
    ]

    # Call GPT
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content.strip()
