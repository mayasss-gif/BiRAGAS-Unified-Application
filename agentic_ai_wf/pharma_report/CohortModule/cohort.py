import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import pandas as pd
from fpdf import FPDF
from PIL import Image
from collections import defaultdict
import GEOparse
import os
import openai

from decouple import config
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = config("OPENAI_API_KEY", default="", cast=str)


def analyze_cohort_data(json_path):
    """Analyze a GEO cohort JSON file and return summary statistics."""
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Early return if no results
    if not results:
        return []

    # Total dataset size
    total_datasets = len(results)

    # Total cohort/sample count
    total_samples = sum(ds.get("Valid Sample Count", 0) for ds in results)

    # Normalize and count tissue types
    tissue_counts = defaultdict(int)

    def normalize_tissue(tissue_name):
        """Normalize tissue type strings for consistency."""
        tissue = tissue_name.lower().replace("tissue: ", "").strip()
        return tissue.title()

    for ds in results:
        tissue_cat = ds.get("Tissue Categorization", {})
        for raw_tissue, info in tissue_cat.items():
            normalized = normalize_tissue(raw_tissue)
            tissue_counts[normalized] += info.get("count", 0)

    # Final unique tissue types
    unique_tissue_types = len(tissue_counts)

    # Print output
    print(f"Total Dataset Size: {total_datasets}")
    print(f"Total Cohort Size (Samples): {total_samples}")
    print(f"Unique Tissue Type Count: {unique_tissue_types}")
    print("\nBreakdown of Tissue Types:")
    for tissue, count in tissue_counts.items():
        print(f" - {tissue}: {count}")

    # Return summary if needed programmatically
    return {
        "total_datasets": total_datasets,
        "total_samples": total_samples,
        "unique_tissue_types": unique_tissue_types,
        "tissue_counts": dict(tissue_counts)
    }


# ------------------------------ Plot Function -------------------------------


def extract_sample_metadata_from_json(json_path, verbose=True):
    """
    Load a JSON GEO dataset file, extract valid sample IDs, and fetch their metadata.

    Args:
        json_path (str): Path to the GEO results JSON file.
        verbose (bool): Whether to print logs for progress and errors.

    Returns:
        pd.DataFrame or list: DataFrame with metadata if results exist, otherwise an empty list.
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Return early if results is empty
    if not results:
        if verbose:
            print("⚠️ No results found in the JSON file.")
        return []

    # Total dataset size & cohort sample count
    total_datasets = len(results)
    total_samples = sum(ds.get("Valid Sample Count", 0) for ds in results)

    if verbose:
        print(f"✅ Total Datasets: {total_datasets}")
        print(f"✅ Total Valid Samples: {total_samples}")

    # Extract all sample IDs
    all_sample_ids = []
    for ds in results:
        all_sample_ids.extend(ds.get("Valid Sample IDs", []))

    if verbose:
        print(f"✅ Total Sample IDs Extracted: {len(all_sample_ids)}")

    # Fetch metadata
    sample_metadata = []
    for gsm_id in all_sample_ids:
        if verbose:
            print(f"🔎 Fetching {gsm_id}...")
        try:
            gsm = GEOparse.get_GEO(
                geo=gsm_id, destdir="temporary", how="brief")
            metadata = {
                "Sample ID": gsm_id,
                "Country": gsm.metadata.get("contact_country", ["Unknown"])[0],
                "Instrument Model": gsm.metadata.get("instrument_model", ["Unknown"])[0],
                "Characteristics": gsm.metadata.get("characteristics_ch1", ["None"])
            }
            sample_metadata.append(metadata)
        except Exception as e:
            if verbose:
                print(f"❌ Error fetching {gsm_id}: {e}")

    # Return as DataFrame
    return pd.DataFrame(sample_metadata)


def tissue_data(json_path):

    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    # 3. Normalize and count tissue types
    tissue_counts = defaultdict(int)

    def normalize_tissue(tissue_name):
        """Normalize tissue type strings for consistency."""
        tissue = tissue_name.lower().replace("tissue: ", "").strip()
        return tissue.title()

    for ds in results:
        tissue_cat = ds.get("Tissue Categorization", {})
        for raw_tissue, info in tissue_cat.items():
            normalized = normalize_tissue(raw_tissue)
            tissue_counts[normalized] += info.get("count", 0)

    tissue_df = pd.DataFrame(list(tissue_counts.items()), columns=[
                             "Tissue Type", "Count"])
    tissue_df = tissue_df.sort_values(by="Count", ascending=False)

    # Final unique tissue types
    return tissue_df


def prepare_country_instrument_df(df):
    """Combine Country and Instrument Model into a single long-format dataframe."""
    country_df = df["Country"].value_counts().reset_index()
    country_df.columns = ["Value", "Count"]
    country_df["Category"] = "Country"

    instrument_df = df["Instrument Model"].value_counts().reset_index()
    instrument_df.columns = ["Value", "Count"]
    instrument_df["Category"] = "Instrument Model"

    combined_df = pd.concat([country_df, instrument_df], ignore_index=True)
    return combined_df


sns.set(style="whitegrid", context="notebook", palette="pastel")


def add_data_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=9, padding=3)


def plot_and_get_base64(plot_func, **kwargs):
    """Render a plot and return its base64-encoded PNG image."""
    fig = plot_func(**kwargs)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def plot_tissue_distribution(tissue_df):
    tissue_counts = defaultdict(int)
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=tissue_df,
        y="Tissue Type",
        x="Count",
        hue="Tissue Type",
        palette="pastel",
        dodge=False,
        legend=False
    )
    ax.set_title("Distribution of Tissue Types", fontsize=14)
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("Tissue Type")
    add_data_labels(ax)
    return fig


def plot_country_distribution(df):
    print(df.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="Country",
        order=df["Country"].value_counts().index,
        hue="Country",
        palette="pastel",
        legend=False,
        ax=ax
    )
    ax.set_title("Sample Count by Country", fontsize=14)
    ax.set_xlabel("Count")
    ax.set_ylabel("Country")
    add_data_labels(ax)
    return fig


def plot_country_instrument_combined(df):
    """Plot a unified bar chart for Country and Instrument Model counts."""
    combined_df = prepare_country_instrument_df(df)

    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=combined_df,
        y="Value",
        x="Count",
        hue="Category",
        dodge=True,
        palette="pastel"
    )
    ax.set_title(
        "Sample Distribution by Country and Instrument Model", fontsize=14)
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("")

    add_data_labels(ax)
    plt.tight_layout()
    return fig


def plot_instrument_model_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        y="Instrument Model",
        order=df["Instrument Model"].value_counts().index,
        hue="Instrument Model",
        palette="pastel",
        legend=False,
        ax=ax
    )
    ax.set_title("Sample Count by Instrument Model", fontsize=14)
    ax.set_xlabel("Count")
    ax.set_ylabel("Instrument Model")
    add_data_labels(ax)
    return fig


def plot_characteristics_distribution(df_exploded, keys=("status", "cell type")):
    # Split 'Characteristics' into Key and Value
    char_df = df_exploded["Characteristics"].str.split(":", n=1, expand=True)
    char_df.columns = ["Key", "Value"]
    char_df["Key"] = char_df["Key"].str.strip()
    char_df["Value"] = char_df["Value"].str.strip()

    # Combine with original index to preserve grouping
    char_df["SampleID"] = df_exploded.index

    # Filter only the keys that are actually present
    present_keys = set(char_df["Key"]).intersection(keys)

    if not present_keys:
        return []

    # Filter relevant keys
    char_df = char_df[char_df["Key"].isin(present_keys)]

    # Pivot data
    pivot_df = char_df.pivot(index="SampleID", columns="Key", values="Value")

    # Drop rows with any missing values for the selected keys
    pivot_df = pivot_df.dropna(subset=present_keys)

    # Plot only if there are at least 2 unique rows
    if pivot_df.empty or len(pivot_df) < 2:
        return []

    # Plotting
    plt.figure(figsize=(10, 6))

    # Determine what keys to use for y and hue
    y_key = keys[1] if keys[1] in present_keys else list(present_keys)[0]
    hue_key = keys[0] if keys[0] in present_keys and keys[0] != y_key else None

    ax = sns.countplot(
        data=pivot_df,
        y=y_key,
        hue=hue_key,
        palette="pastel"
    )

    title = f"Distribution of {y_key}"
    if hue_key:
        title += f" by {hue_key}"

    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(y_key)
    plt.tight_layout()
    return plt.gcf()


# --------------------------- Graph Descripton --------------------------

def generate_chart_explanation(base64_img):
    openai.api_key = OPENAI_API_KEY

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

# Usage
