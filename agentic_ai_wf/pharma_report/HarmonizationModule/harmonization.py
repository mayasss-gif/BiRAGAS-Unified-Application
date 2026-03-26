import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import openai
import os
# --------------- Graph Descripton --------------------------

# Use your actual key here
openai.api_key = os.getenv("OPENAI_API_KEY")


def plot_and_get_base64(plot_func, **kwargs):
    """Render a plot and return its base64-encoded PNG image."""
    try:
        fig = plot_func(**kwargs)
    except Exception as e:
        # If plot function fails, create a default "Error" figure
        fig = create_default_figure()
    
    if isinstance(fig, plt.Figure):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        return img_base64
    else:
        # If fig is not a plt.Figure (dict, None, etc.), create a default figure
        if isinstance(fig, dict) and 'error' in fig:
            message = fig['error']
        else:
            message = "No Data Available"
        
        default_fig = create_default_figure()
        buffer = io.BytesIO()
        default_fig.savefig(buffer, format='png', bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(default_fig)
        return img_base64

def create_default_figure(message="No Data Available"):
    """Create a default figure with a message when data is not available."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, 
            horizontalalignment='center', 
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Data Visualization", fontsize=18, pad=20)
    plt.tight_layout()
    return fig

def plot_log2fc_by_origin(csv_path: str, patient_id) -> plt.Figure:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"error": "No data provided. Please supply a non-empty DataFrame."}

    # identify log2FC columns
    log2fc_cols = [c for c in df.columns if c.lower().endswith('_log2fc')]
    if not log2fc_cols:
        return {"error": "No 'log2FC' columns found in DataFrame."}

    # split into Patient vs Cohort
    patient_cols = [c for c in log2fc_cols if c.lower().startswith(patient_id)]
    cohort_cols = [c for c in log2fc_cols if c.lower().startswith('gse')]

    if not patient_cols:
        return {"error": "No Patient log2FC columns found."}

    def extract_series(cols):
        return pd.to_numeric(df[cols].stack(), errors='coerce').dropna()

    patient_series = extract_series(patient_cols)
    cohort_series = extract_series(
        cohort_cols) if cohort_cols else pd.Series(dtype=float)

    def compute_stats(s: pd.Series) -> dict:
        Q1, Q3 = s.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = (s < (Q1 - 1.5*IQR)) | (s > (Q3 + 1.5*IQR))
        return {
            'count': int(s.count()),
            'mean': float(s.mean()),
            'median': float(s.median()),
            'std': float(s.std()),
            'min': float(s.min()),
            'max': float(s.max()),
            'skewness': float(s.skew()),
            'Q1': float(Q1),
            'Q3': float(Q3),
            'outlier_count': int(outliers.sum()),
            'outlier_pct': float(outliers.mean()*100)
        }

    stats = {'Patient': compute_stats(patient_series)}
    has_cohort = not cohort_series.empty
    if has_cohort:
        stats['Cohort'] = compute_stats(cohort_series)

    # create subplots with appropriate layout
    if has_cohort:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax_hist = axs[0, 0]
        ax_box = axs[0, 1]
        ax_stat_patient = axs[1, 0]
        ax_stat_cohort = axs[1, 1]
    else:
        fig, axs = plt.subplots(3, 1, figsize=(6, 15))
        ax_hist = axs[0]
        ax_box = axs[1]
        ax_stat_patient = axs[2]

    # plot histogram(s)
    bins = 30
    ax_hist.hist(patient_series, bins=bins, alpha=0.6, label='Patient')
    if has_cohort:
        ax_hist.hist(cohort_series, bins=bins, alpha=0.6, label='Cohort')
    ax_hist.set_title(r'$\log_{2}\mathrm{FC}$ Distribution by Origin')
    ax_hist.set_xlabel(r'$\log_{2}\mathrm{FC}$')
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
    bplot = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax_box.set_title('Box Plot by Origin')
    ax_box.set_ylabel(r'$\log_{2}\mathrm{FC}$')
    ax_box.grid(alpha=0.3)

    # plot stats table(s)
    def draw_table(ax, stats_dict, title):
        ax.axis('off')
        rows = [[k, f"{v:.3f}" if isinstance(
            v, float) else v] for k, v in stats_dict.items()]
        tbl = ax.table(rows, colLabels=[
                       'Stat', 'Value'], cellLoc='center', loc='center')
        ax.set_title(title)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

    draw_table(ax_stat_patient, stats['Patient'], 'Patient Stats')
    if has_cohort:
        draw_table(ax_stat_cohort, stats['Cohort'], 'Cohort Stats')

    plt.tight_layout()
    return fig


def plot_log2fc_histogram(path, advanced_stats=True):
    """
    Reads 'log2FC' from a CSV and returns a matplotlib Figure with histogram,
    boxplot, and stats table (optional).
    """
    df = pd.read_csv(path)

    # Validation
    if df.empty or "log2FC" not in df.columns:
        raise ValueError("Missing or empty 'log2FC' column.")

    series = pd.to_numeric(df["log2FC"], errors="coerce").dropna()
    if series.empty:
        raise ValueError("No valid numeric values in 'log2FC' column.")

    # Compute statistics
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = (series < lower) | (series > upper)

    stats = {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "range": float(series.max() - series.min()),
        "skewness": float(series.skew()),
        "Q1": float(Q1),
        "Q3": float(Q3),
        "outlier_count": int(outliers.sum()),
        "outlier_percentage": float(outliers.mean() * 100),
    }

    # Plotting
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax1.hist(series, bins=30, alpha=0.7)
    ax1.set_title(r'$\log_{2}\mathrm{FC}$ Distribution')
    ax1.set_xlabel(r'$\log_{2}\mathrm{FC}$')
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(stats["mean"], color="red", linestyle="--",
                label=f"Mean: {stats['mean']:.3f}")
    ax1.axvline(stats["median"], color="orange", linestyle="--",
                label=f"Median: {stats['median']:.3f}")
    ax1.legend()

    if advanced_stats:
        # Boxplot
        ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        bplot = ax2.boxplot(series, vert=True, patch_artist=True)
        bplot["boxes"][0].set_facecolor("#79C")
        ax2.set_title("Box Plot")
        ax2.set_ylabel(r'$\log_{2}\mathrm{FC}$')
        ax2.grid(True, alpha=0.3)

        # Stats Table
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax3.axis("off")
        table_data = [
            ["Count", f"{stats['count']:,}"],
            ["Mean ± SD", f"{stats['mean']:.3f} ± {stats['std']:.3f}"],
            ["Median [Q1,Q3]",
                f"{stats['median']:.3f} [{stats['Q1']:.3f}, {stats['Q3']:.3f}]"],
            ["Range", f"{stats['min']:.3f} to {stats['max']:.3f}"],
            ["Skewness", f"{stats['skewness']:.3f}"],
            ["Outliers",
                f"{stats['outlier_count']} ({stats['outlier_percentage']:.1f}%)"],
        ]
        tbl = ax3.table(
            cellText=table_data,
            colLabels=["Statistic", "Value"],
            cellLoc="center",
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)

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


def get_stats_sections_from_txt(file_path):
    """
    Reads a log2FC statistics .txt file and extracts key sections into a dictionary.

    Parameters:
        file_path (str): Full path to the .txt file.

    Returns:
        dict: {
            "summary": str,
            "descriptive_statistics": str,
            "distribution_shape": str,
            "outlier_analysis": str
        } or {'error': str} if something went wrong.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return {"error": f"File not found at '{file_path}'"}
    except Exception as e:
        return {"error": f"Error reading file: {e}"}

    # Initialize result
    sections = {
        "summary": "",
        "descriptive_statistics": "",
        "distribution_shape": "",
        "outlier_analysis": ""
    }

    patterns = {
        "descriptive_statistics": r"Descriptive Statistics:",
        "distribution_shape": r"Distribution Shape:",
        "outlier_analysis": r"Outlier Analysis:"
    }

    current_key = "summary"
    buffer = []
    start_collecting = False

    for line in lines:
        stripped = line.strip()
        if not start_collecting:
            # Skip until we find the first meaningful line (Data Points)
            if stripped.startswith("Data Points:"):
                start_collecting = True
            else:
                continue  # Skip header and decorative lines
        matched = False
        for key, pattern in patterns.items():
            if re.match(pattern, stripped):
                sections[current_key] = "\n".join(buffer).strip()
                buffer = []
                current_key = key
                matched = True
                break
        if not matched:
            buffer.append(stripped)

    # Capture last section
    sections[current_key] = "\n".join(buffer).strip()

    return sections
