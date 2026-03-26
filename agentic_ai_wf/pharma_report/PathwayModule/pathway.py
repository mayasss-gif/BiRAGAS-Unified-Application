
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import io
import base64
from matplotlib.cm import ScalarMappable
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


def plot_pathway_class_distribution(csv_path: str):
    """
    Read CSV, tally counts in 'Main_Class', draw a donut chart,
    and return a matplotlib Figure object.
    """
    df = pd.read_csv(csv_path)
    if 'Main_Class' not in df.columns:
        raise ValueError(f"'Main_Class' column not found in {csv_path}")

    counts = df['Main_Class'].value_counts()
    classes = counts.index.tolist()
    values = counts.values.tolist()
    total = sum(values)
    percents = [v/total*100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        values,
        radius=1.0,
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='white')
    )

    centre = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre)
    ax.axis('equal')

    legend_labels = [f"{cls} ({pct:.1f}%)" for cls,
                     pct in zip(classes, percents)]
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
    return fig


def get_pathway_regulation_summary(csv_path: str):
    """
    Load pathway CSV and return total, upregulated, and downregulated counts.
    """
    df = pd.read_csv(csv_path)

    total_pathways = len(df)
    regulation_counts = df['Regulation'].str.lower().value_counts()
    up_count = regulation_counts.get('up', 0)
    down_count = regulation_counts.get('down', 0)

    return {
        "total_pathways": total_pathways,
        "up_count": up_count,
        "down_count": down_count
    }


def extract_top_pathway_narratives(csv_path: str):
    """
    Extract top 10 upregulated and downregulated pathways by Priority_Rank
    and return selected fields as lists of dictionaries.
    """
    df = pd.read_csv(csv_path)

    # Normalize 'Regulation' column to lowercase
    df['Regulation'] = df['Regulation'].str.lower()

    # Sort by Priority_Rank (ascending is highest priority)
    df_sorted = df.sort_values("Priority_Rank", ascending=True)

    # Extract top 10 for each group
    upregulated_df = df_sorted[df_sorted['Regulation'] == 'up'].head(10)
    downregulated_df = df_sorted[df_sorted['Regulation'] == 'down'].head(10)

    # Columns to extract
    fields = [
        'Pathway_Name',
        'Regulation',
        'LLM_Score',
        'Confidence_Level',
        'Score Justification',
        'Priority_Rank'
    ]

    # Convert DataFrame rows to dictionaries
    def to_dict_list(sub_df):
        return [
            {
                "Pathway_Name": row.get("Pathway_Name", "N/A"),
                "Regulation": row.get("Regulation", "N/A"),
                "LLM_Score": row.get("LLM_Score", "N/A"),
                "Confidence_Level": row.get("Confidence_Level", "N/A"),
                "Score Justification": row.get("Score_Justification", "N/A"),
                "Priority_Rank": row.get("Priority_Rank", "N/A")
            }
            for _, row in sub_df.iterrows()
        ]

    upregulated_data = to_dict_list(upregulated_df)
    downregulated_data = to_dict_list(downregulated_df)

    return upregulated_data, downregulated_data


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
