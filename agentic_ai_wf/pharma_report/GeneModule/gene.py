import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import base64
import numpy as np
import openai
import os
import re
openai.api_key = os.getenv("OPENAI_API_KEY")


def plot_lfc_grouped_bar(csv_path: str, patient_id: str) -> plt.Figure:
    """
    Generates a grouped or single bar plot comparing Patient vs Cohort log₂FC for top 20 genes.
    Returns a Matplotlib Figure object with `data` and `meta_data` attributes attached.
    """
    df = pd.read_csv(csv_path)
    if 'Gene' not in df.columns:
        raise ValueError("No 'Gene' column found in CSV.")

    pid = patient_id.lower()
    cohort_col = next((c for c in df.columns if re.match(
        r'cohort.*lfc.*mean', c, re.IGNORECASE)), None)
    samp_cols = [c for c in df.columns if re.match(
        rf'{re.escape(pid)}.*log2fc', c, re.IGNORECASE)]
    mean_col = next((c for c in df.columns if re.match(
        rf'{re.escape(pid)}.*lfc.*mean', c, re.IGNORECASE)), None)

    # Determine data matrix and title
    if cohort_col and mean_col:
        df['delta'] = df[mean_col] - df[cohort_col]
        df['abs_delta'] = df['delta'].abs()
        sub = df.nlargest(20, 'abs_delta').set_index('Gene')
        matrix = sub[[mean_col, cohort_col]].rename(
            columns={mean_col: 'Patient', cohort_col: 'Cohort'})
        title = "Top 20 Genes by |Patient – Cohort|"
        plot_type = 'grouped'

    elif cohort_col and len(samp_cols) == 1:
        col = samp_cols[0]
        df['delta'] = df[col] - df[cohort_col]
        df['abs_delta'] = df['delta'].abs()
        sub = df.nlargest(20, 'abs_delta').set_index('Gene')
        matrix = sub[[col, cohort_col]].rename(
            columns={col: 'Patient', cohort_col: 'Cohort'})
        title = "Top 20 Genes by |Patient – Cohort|"
        plot_type = 'grouped'

    elif samp_cols:
        if len(samp_cols) > 1:
            df['variance'] = df[samp_cols].var(axis=1)
            sub = df.nlargest(20, 'variance').set_index('Gene')
            matrix = sub[samp_cols].rename(columns=lambda _: 'Patient')
            title = "Top 20 Genes by LFC Variance Across Samples"
            plot_type = 'grouped'
        else:
            col = samp_cols[0]
            df['abs_fc'] = df[col].abs()
            sub = df.nlargest(20, 'abs_fc').set_index('Gene')
            matrix = sub[[col]].rename(columns={col: 'Patient'})
            title = "Top 20 Genes by |Patient|"
            plot_type = 'single'
    else:
        raise ValueError("Need cohort mean or patient sample columns.")

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(matrix) * 0.4), 6))

    if plot_type == 'single':
        df_bar = matrix.reset_index()
        sns.barplot(data=df_bar, x='Gene', y='Patient', ax=ax)
        ax.set_xlabel('Gene')
        ax.set_ylabel(r'$\log_{2}\mathrm{FC}$ ')
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
    else:
        dfm = matrix.reset_index().melt(
            id_vars='Gene', var_name='Group', value_name='log2FC')
        sns.barplot(data=dfm, x='Gene', y='log2FC', hue='Group', ax=ax)
        ax.set_xlabel('Gene')
        ax.set_ylabel(r'$\log_{2}\mathrm{FC}$ Fold Change')
        ax.set_title(title, fontsize=14)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Group')

    plt.tight_layout()

    # Attach metadata and data for optional use
    fig.meta_data = {"title": title}
    fig.data_payload = matrix.to_dict()

    return fig


def plot_and_get_base64(plot_func, **kwargs):
    """Render a plot and return its base64-encoded PNG image."""
    fig = plot_func(**kwargs)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def plot_bubble_chart(csv_path, patient_id, mode='UP'):
    """
    Returns a matplotlib figure of a bubble chart for top 20 up/downregulated DEGs.

    Args:
        csv_path (str): Path to the CSV file.
        patient_id (str): UUID-style patient ID that column names start with.
        mode (str): 'UP' or 'DOWN' to indicate which bubble chart to plot.

    Returns:
        matplotlib.figure.Figure
    """
    # Load data
    df = pd.read_csv(csv_path)
    tr1 = [col for col in df.columns if col.lower().startswith(patient_id)]
    print(tr1)
    # Find log2FC and p-value columns that start with patient ID
    log2fc_cols = [col for col in df.columns if col.lower(
    ).startswith(patient_id) and "log2fc" in col.lower()]
    pval_cols = [col for col in df.columns if col.lower().startswith(
        patient_id) and "_p-value" in col.lower()]

    if not log2fc_cols or not pval_cols:
        raise ValueError(
            f"Could not find both log2FC and adjusted p-value columns starting with patient ID: {patient_id}")

    log2fc_col = log2fc_cols[0]
    pval_col = pval_cols[0]

    print(f"log2FC column: {log2fc_col}")
    print(f"Adjusted p-value column: {pval_col}")

    # Subset and rename
    df = df[['Gene', log2fc_col, pval_col]].dropna()
    df = df.rename(columns={log2fc_col: 'log2FC', pval_col: 'adj_pval'})
    df['neg_log10_pval'] = -np.log10(df['adj_pval'])

    # Filter top DEGs
    if mode.upper() == 'UP':
        df_plot = df[df['log2FC'] > 0].sort_values('adj_pval').head(20)
        title = "Top 20 Upregulated DEGs"
        cmap = "Reds"
    elif mode.upper() == 'DOWN':
        df_plot = df[df['log2FC'] < 0].sort_values('adj_pval').head(20)
        title = "Top 20 Downregulated DEGs"
        cmap = "Blues_r"
    else:
        raise ValueError("Invalid mode. Use 'UP' or 'DOWN'.")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        df_plot['log2FC'],
        df_plot['Gene'],
        s=df_plot['neg_log10_pval'] * 80,
        c=df_plot['log2FC'],
        cmap=cmap,
        edgecolors='black',
        alpha=0.8
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"$\log_{2}\mathrm{FC}$")
    ax.set_ylabel("Gene")
    ax.axvline(0, linestyle='--', color='gray')
    plt.colorbar(sc, ax=ax, label='log₂ Fold Change')
    plt.tight_layout()

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


def get_gene_clinical_summary_by_tier(csv_path, patient_id):

    # Load JSON as DataFrame
    df = pd.read_csv(csv_path)
    log2fc_col = [col for col in df.columns if col.lower().startswith(
        patient_id.lower()) and col.lower().endswith("log2fc")][0]

    # Step 1: Filter top 10 up- and down-regulated genes
    upregulated = df[df['Patient_LFC_Trend'] == 'UP'].sort_values(
        by=log2fc_col, ascending=False).head(10)
    downregulated = df[df['Patient_LFC_Trend'] == 'DOWN'].sort_values(
        by=log2fc_col, ascending=True).head(10)
    top_genes = pd.concat([upregulated, downregulated], ignore_index=True)

    # Step 2: Keep only required columns
    top_genes = top_genes[['Gene', log2fc_col,'Tier',
                           'Confidence', 'Patient_LFC_Trend']].copy()

    # Step 3: Get clinical relevance from GPT
    def get_clinical_relevance(gene_name):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Provide a concise clinical relevance summary of the human gene '{gene_name}'. Limit it to only 2-3 lines."
                }],
                temperature=0.4,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    top_genes['Clinical_Relevance'] = top_genes['Gene'].apply(
        get_clinical_relevance)

    # Step 4: Sort by Tier and Trend
    top_genes = top_genes.sort_values(by=['Tier', 'Confidence','Patient_LFC_Trend'])
    print(top_genes)
    # Step 5: Build structured dictionary
    structured = {}
    for tier in sorted(top_genes['Tier'].unique()):
        print(tier)
        tier_label = f"{tier}"
        structured[tier_label] = {
            "Upregulated Gene": [], "Downregulated Gene": []}

        for trend in ['UP', 'DOWN']:
            sub_df = top_genes[(top_genes['Tier'] == tier) & (
                top_genes['Patient_LFC_Trend'] == trend)]
            for _, row in sub_df.iterrows():
                structured[tier_label]["Upregulated Gene" if trend == 'UP' else "Downregulated Gene"].append({
                    "Gene": row["Gene"],
                    "Log2FC": row[log2fc_col],
                    "Clinical_Relevance": row["Clinical_Relevance"]
                })
    return structured


def summarize_gene_counts(csv_path):
    try:

        # Load CSV
        df = pd.read_csv(csv_path)

        # Drop rows without gene names
        df = df.dropna(subset=['Gene'])

        # Total unique genes
        total_genes = df['Gene'].nunique()

        # Tier-wise counts
        tier_counts = df['Confidence'].value_counts().to_dict()
        print(tier_counts)

        # Build final summary with default 0 if tier not found
        summary = {
            "total_genes": total_genes,
            "tier1": tier_counts.get('High', 0),
            "tier2": tier_counts.get('Medium', 0),
            "tier3": tier_counts.get('Low', 0)
        }

        return summary
    except Exception as e:
        print(f"Error in summarize_gene_counts: {e}")
        df = pd.read_csv(csv_path)

        # Drop rows without gene names
        df = df.dropna(subset=['Gene'])

        # Total unique genes
        total_genes = df['Gene'].nunique()

        summary = {
            "total_genes": total_genes,
            "tier1": 0,
            "tier2": 0,
            "tier3": 0
        }

        return summary
