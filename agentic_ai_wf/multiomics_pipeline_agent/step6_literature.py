import json
import logging
import os
import textwrap
from datetime import datetime
from time import sleep
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Entrez, Medline
from urllib.error import HTTPError

from .utils import ensure_dir

# ---------------- Optional: reportlab for PDF ----------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader

    HAS_REPORTLAB = True
except Exception:  # ImportError and others
    HAS_REPORTLAB = False
    A4 = canvas = inch = ImageReader = None

# ---------------- Optional: OpenAI for GPT summaries ----------------
try:
    from openai import OpenAI

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    if OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        HAS_OPENAI = True
    else:
        _openai_client = None
        HAS_OPENAI = False
except Exception:
    _openai_client = None
    HAS_OPENAI = False


def _load_json_safe(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def run_step6(
    base_dir: str,
    query_term: Optional[str],
    email: Optional[str],
    disease_term: Optional[str] = None,
    top_n: int = 20,
) -> Dict[str, str]:
    """
    STEP 6 — Refined literature validation + (optional) PDF report.

    - Uses Step 5 extended biomarker evidence table.
    - For top biomarkers, queries PubMed (biomarker + disease_term).
    - Builds PubMed hit summary, keyword evidence matrix, JSON summaries.
    - Generates barplot + heatmap.
    - If reportlab is installed, also generates a multi-page PDF report.

    Robust to:
      - missing email / query_term (step is skipped)
      - network / SSL errors (step degrades gracefully, pipeline continues)
    """
    logger = logging.getLogger("step6_literature")

    # --------- Basic checks / graceful skip ----------
    if not query_term or not email:
        logger.warning("query_term or email missing; skipping literature step.")
        return {"skipped": "no_query_or_email"}

    # Use disease_term for the biology header; fall back to query_term
    disease_term = disease_term or query_term

    base_dir = os.path.abspath(base_dir)
    step1_dir = os.path.join(base_dir, "step_1_ingestion")
    step2_dir = os.path.join(base_dir, "step_2_preprocessing")
    step3_dir = os.path.join(base_dir, "step_3_integration")
    step4_dir = os.path.join(base_dir, "step_4_ml_biomarkers")
    step5_dir = os.path.join(base_dir, "step_5_crossomics")
    step6_dir = ensure_dir(os.path.join(base_dir, "step_6_literature"))

    # --------- Load upstream summaries (best-effort) ----------
    ingest_manifest = _load_json_safe(os.path.join(step1_dir, "manifest.json"), {})
    norm_summary = _load_json_safe(os.path.join(step2_dir, "normalization_summary.json"), {})
    integration_sum = _load_json_safe(os.path.join(step3_dir, "integration_summary.json"), {})
    ml_summary = _load_json_safe(os.path.join(step4_dir, "ml_summary.json"), {})
    crossomics_sum = _load_json_safe(
        os.path.join(step5_dir, "extended_crossomics_summary.json"), {}
    )

    # --------- Load extended biomarker evidence (Step 5) ----------
    evidence_table_path = os.path.join(step5_dir, "extended_biomarker_evidence.csv")
    if not os.path.exists(evidence_table_path):
        logger.error(
            "Extended biomarker evidence table not found at %s. "
            "Make sure Step 5 completed successfully.",
            evidence_table_path,
        )
        return {
            "step_dir": step6_dir,
            "results": "",
            "error": "missing_step5_evidence",
        }

    extended_df = pd.read_csv(evidence_table_path)
    extended_df = extended_df.sort_values(
        "extended_biology_score", ascending=False
    ).reset_index(drop=True)
    extended_df["biomarker"] = extended_df["primary_original_feature"].astype(str)
    extended_df["layer"] = extended_df["primary_layer"].astype(str)

    logger.info("Loaded extended evidence: %d biomarkers", extended_df.shape[0])

    # --------- Configure Entrez (NCBI) ----------
    Entrez.email = email
    Entrez.api_key = os.getenv("NCBI_API_KEY", None)

    # Top N biomarkers to send to PubMed
    TOP_N_LIT = min(top_n, extended_df.shape[0])
    TOP_N_PLOT = min(top_n, extended_df.shape[0])
    TOP_N_IN_REPORT = min(15, TOP_N_LIT)
    MAX_ABSTRACTS_PER_BM = 3

    keyword_list = ["diagnostic", "prognostic", "predictive", "mechanistic"]

    # Flag to avoid spamming SSL errors: once we detect SSL failure,
    # we skip all remaining PubMed queries in this run.
    ssl_broken = False

    # ===============================
    # 1. OMICS LAYER SUMMARY
    # ===============================
    layer_rows = []
    if "layers" in ingest_manifest:
        core_layers = integration_sum.get("core_layers", [])
        external_layers = integration_sum.get("external_layers", [])
        for lname, info in ingest_manifest["layers"].items():
            n_samples = info.get("n_samples", None)
            n_features = info.get("n_features", None)
            if lname in core_layers:
                role = "core"
            elif lname in external_layers:
                role = "external"
            else:
                role = "unused"
            layer_rows.append(
                {
                    "layer": lname,
                    "role": role,
                    "n_samples": n_samples,
                    "n_features": n_features,
                }
            )

    layer_summary_df = pd.DataFrame(layer_rows).sort_values(["role", "layer"])
    layer_summary_path = os.path.join(step6_dir, "layer_summary.csv")
    layer_summary_df.to_csv(layer_summary_path, index=False)
    logger.info("Saved layer summary → %s", layer_summary_path)

    # ===============================
    # 2. PubMed search + abstracts
    # ===============================

    def pubmed_search(query: str, max_results: int = 200):
        """
        Robust PubMed search.

        - Any network/SSL error is caught and returns (0, []).
        - On first CERTIFICATE_VERIFY_FAILED, we log once, mark ssl_broken,
          and skip all remaining queries in this run.
        """
        nonlocal ssl_broken

        # If we already know SSL is broken, don't even try.
        if ssl_broken:
            logger.warning(
                "Skipping PubMed call for %r because SSL is already known to be broken in this run.",
                query,
            )
            return 0, []

        try:
            handle = Entrez.esearch(
                db="pubmed", term=query, retmax=max_results, sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            count = int(record.get("Count", 0))
            ids = record.get("IdList", [])
            return count, ids

        except HTTPError as e:
            msg = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in msg:
                ssl_broken = True
                logger.error(
                    "PubMed SSL/certificate problem detected; "
                    "skipping remaining literature queries this run. Details: %s",
                    e,
                )
            else:
                logger.error("HTTPError for query %r: %s", query, e)
            return 0, []

        except Exception as e:
            msg = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in msg:
                ssl_broken = True
                logger.error(
                    "PubMed SSL/certificate problem detected; "
                    "skipping remaining literature queries this run. Details: %s",
                    e,
                )
            else:
                logger.error(
                    "Error for query %r (likely network/SSL/DNS issue): %s",
                    query,
                    e,
                )
            return 0, []

    def pubmed_fetch_medline(id_list, max_fetch=3):
        """Fetch Medline records for a small list of PMIDs. Robust to network errors."""
        id_list = id_list[:max_fetch]
        if not id_list:
            return []
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(id_list),
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()
            out = []
            for rec in records:
                pmid = rec.get("PMID", "")
                title = rec.get("TI", "")
                abstr = rec.get("AB", "")
                journal = rec.get("JT", "")
                year = rec.get("DP", "").split(" ")[0]
                authors = rec.get("AU", [])
                out.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstr,
                        "journal": journal,
                        "year": year,
                        "authors": authors,
                    }
                )
            return out
        except Exception as e:
            logger.error("Error fetching Medline for IDs %s: %s", id_list, e)
            return []

    def summarize_with_gpt(biomarker: str, disease_term: str, articles):
        """Optional GPT-based summarization; falls back to snippet if unavailable."""
        abstracts = [a.get("abstract", "") for a in articles or []]
        abstracts = [a.strip() for a in abstracts if a and a.strip()]
        if not abstracts:
            return (
                f"No non-empty PubMed abstracts were retrieved for {biomarker} in "
                f"{disease_term}."
            )

        joined = "\n\n".join(abstracts[:3])

        if HAS_OPENAI and _openai_client is not None:
            prompt = (
                "You are a cancer biology expert. Based on the abstracts below, "
                f"summarize the role of the biomarker {biomarker} in {disease_term}. "
                "Focus on:\n"
                " - Whether it behaves as an oncogene, tumor suppressor, or context-dependent regulator.\n"
                " - The main pathways or processes it is involved in.\n"
                " - Any evidence for diagnostic, prognostic, or predictive value.\n"
                "Be concise (max ~200 words) and avoid speculation.\n\n"
                f"ABSTRACTS:\n{joined}\n\n"
                "Now provide a concise summary."
            )
            try:
                resp = _openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in multi-omics cancer biomarker "
                                "interpretation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error("OpenAI summarization failed for %s: %s", biomarker, e)

        # Fallback: snippet of first abstract
        snip = abstracts[0]
        snip = " ".join(snip.split()[:80]) + "..."
        return (
            f"Representative abstract snippet for {biomarker} in {disease_term}:\n\n"
            f"{snip}"
        )

    pubmed_records = []
    kw_matrix = []
    bm_articles = {}
    bm_summaries = {}

    top_for_lit = extended_df.head(TOP_N_LIT).copy()

    for i, row in top_for_lit.iterrows():
        biomarker = row["biomarker"]
        layer = row["layer"]
        logger.info(
            "[%d/%d] PubMed queries for biomarker %s",
            i + 1,
            len(top_for_lit),
            biomarker,
        )

        # Main disease-specific query
        main_q = f'{biomarker} AND ("{disease_term}"[Title/Abstract])'
        total_hits, id_list = pubmed_search(main_q, max_results=200)
        logger.info("  main hits: %d", total_hits)
        sleep(0.34)

        # Keyword-specific evidence
        kw_counts = {}
        for kw in keyword_list:
            qkw = (
                f'{biomarker} AND ("{disease_term}"[Title/Abstract]) '
                f'AND ({kw}[Title/Abstract])'
            )
            ckw, _ = pubmed_search(qkw, max_results=50)
            kw_counts[kw] = ckw
            logger.info("    - %s: %d", kw, ckw)
            sleep(0.34)

        # Fetch abstracts
        articles = pubmed_fetch_medline(id_list, max_fetch=MAX_ABSTRACTS_PER_BM)
        bm_articles[biomarker] = articles

        # Summarize
        summary_text = summarize_with_gpt(biomarker, disease_term, articles)
        bm_summaries[biomarker] = summary_text

        rec = {
            "biomarker": biomarker,
            "layer": layer,
            "extended_biology_score": float(row["extended_biology_score"]),
            "ml_importance_raw": float(row["ml_importance_raw"])
            if not np.isnan(row.get("ml_importance_raw", np.nan))
            else None,
            "pubmed_hits": int(total_hits),
        }
        pubmed_records.append(rec)

        kw_row = {"biomarker": biomarker}
        kw_row.update(kw_counts)
        kw_matrix.append(kw_row)

    pubmed_df = pd.DataFrame(pubmed_records).sort_values(
        "pubmed_hits", ascending=False
    ).reset_index(drop=True)
    if kw_matrix:
        kw_df = (
            pd.DataFrame(kw_matrix)
            .set_index("biomarker")
            .fillna(0)
            .astype(int)
        )
    else:
        kw_df = pd.DataFrame(columns=keyword_list)

    pubmed_df_path = os.path.join(step6_dir, "pubmed_hits_summary.csv")
    kw_df_path = os.path.join(step6_dir, "keyword_evidence_matrix.csv")
    bm_summ_path = os.path.join(step6_dir, "biomarker_literature_summaries.json")
    bm_articles_path = os.path.join(step6_dir, "biomarker_articles.json")

    pubmed_df.to_csv(pubmed_df_path, index=False)
    kw_df.to_csv(kw_df_path)
    with open(bm_summ_path, "w") as f:
        json.dump(bm_summaries, f, indent=2)
    with open(bm_articles_path, "w") as f:
        json.dump(bm_articles, f, indent=2)

    logger.info("Saved PubMed summary   → %s", pubmed_df_path)
    logger.info("Saved keyword matrix   → %s", kw_df_path)
    logger.info("Saved biomarker summaries → %s", bm_summ_path)
    logger.info("Saved biomarker articles   → %s", bm_articles_path)

    # ===============================
    # 3. PLOTS (barplot + heatmap)
    # ===============================
    sns.set(style="white", font_scale=0.9)

    TOP_PLOT = min(TOP_N_PLOT, len(pubmed_df))
    if TOP_PLOT > 0:
        plot_df = pubmed_df.head(TOP_PLOT).copy()

        palette_layer = {
            "genomics": "#2c3e50",
            "transcriptomics": "#34495e",
            "epigenomics": "#7f8c8d",
            "proteomics": "#16a085",
            "metabolomics": "#2980b9",
        }

        plot_df["layer"] = plot_df["layer"].astype(str)

        # Barplot of PubMed hits
        plt.figure(figsize=(6, 6))
        ax = sns.barplot(
            data=plot_df,
            y="biomarker",
            x="pubmed_hits",
            hue="layer",
            dodge=False,
            palette=palette_layer,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xlabel("PubMed hit count")
        plt.ylabel("Biomarker")
        plt.title(f"Top {TOP_PLOT} biomarkers by PubMed hits")
        plt.legend(
            title="Layer",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
        plt.tight_layout()
        barplot_path = os.path.join(step6_dir, "Step6_PubMed_hits_barplot.png")
        plt.savefig(barplot_path, dpi=300)
        plt.close()
        logger.info("Saved barplot → %s", barplot_path)

        # Keyword heatmap (if kw_df has rows/cols)
        if not kw_df.empty:
            kw_for_plot = kw_df.loc[plot_df["biomarker"]].copy()
            kw_norm = kw_for_plot.astype(float)
            row_sums = kw_norm.sum(axis=1).replace(0, 1.0)
            kw_norm = (kw_norm.T / row_sums).T

            cg = sns.clustermap(
                kw_norm,
                cmap="viridis",
                row_cluster=True,
                col_cluster=False,
                figsize=(6, 7),
                cbar_kws={"label": "Relative keyword evidence"},
            )
            heatmap_path = os.path.join(
                step6_dir, "Step6_keyword_evidence_heatmap.png"
            )
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            logger.info("Saved keyword heatmap → %s", heatmap_path)
        else:
            heatmap_path = ""
            logger.warning("Keyword matrix is empty; skipping keyword heatmap.")
    else:
        barplot_path = ""
        heatmap_path = ""
        logger.warning("No PubMed records to plot; skipping plots.")

    # ===============================
    # Collect plots from earlier steps
    # ===============================
    def find_layer_plot(layer, directory, hint_substring=None):
        if not os.path.exists(directory):
            return None
        files = sorted(
            f for f in os.listdir(directory) if f.lower().endswith(".png")
        )
        cand = None
        for fn in files:
            low = fn.lower()
            if layer.lower() in low:
                if hint_substring is None or hint_substring.lower() in low:
                    cand = os.path.join(directory, fn)
                    break
        return cand

    layer_panels = {}
    for row in layer_summary_df.itertuples():
        lname = row.layer
        qc_dir = os.path.join(step1_dir, "plots")
        norm_dir = os.path.join(step2_dir, "plots_norm_compare")
        qc_plot = find_layer_plot(lname, qc_dir, None)
        norm_plot = find_layer_plot(lname, norm_dir, "normalization")
        layer_panels[lname] = {"qc": qc_plot, "norm": norm_plot}

    def find_first_png(directory, max_n=None):
        if not os.path.exists(directory):
            return []
        files = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".png")
        )
        return files if max_n is None else files[:max_n]

    plots_step4 = find_first_png(os.path.join(step4_dir, "plots"), max_n=3)

    plots_step5 = []
    if isinstance(crossomics_sum, dict) and "plots" in crossomics_sum:
        pinfo = crossomics_sum["plots"]
        for key in ["evidence_heatmap", "correlation_heatmap", "network_graph"]:
            p = pinfo.get(key, None)
            if p:
                full = os.path.join(step5_dir, "plots", p)
                if os.path.exists(full):
                    plots_step5.append(full)

    # ===============================
    # 5. Nature-style PDF report (if reportlab available)
    # ===============================
    if HAS_REPORTLAB:
        REPORT_PATH = os.path.join(step6_dir, "Multiomics_Literature_Report.pdf")
        c = canvas.Canvas(REPORT_PATH, pagesize=A4)
        width, height = A4

        MARGIN_X = 40
        MARGIN_TOP = height - 50
        CONTENT_WIDTH = width - 2 * MARGIN_X

        def draw_wrapped(cvs, text, x, y, max_width, leading=11, font="Helvetica", size=9):
            cvs.setFont(font, size)
            est_chars = int(max_width / (size * 0.55))
            est_chars = max(est_chars, 30)
            wrapper = textwrap.TextWrapper(width=est_chars)
            lines = []
            for para in text.split("\n"):
                if not para.strip():
                    lines.append("")
                else:
                    lines.extend(wrapper.wrap(para))
            for line in lines:
                if not line:
                    y -= leading
                else:
                    cvs.drawString(x, y, line)
                    y -= leading
            return y

        def add_header_footer(cvs, title):
            cvs.setFont("Helvetica", 8)
            cvs.drawString(
                MARGIN_X, height - 20, "Multi-Omics Biomarker Discovery — Cohort"
            )
            cvs.drawRightString(width - MARGIN_X, height - 20, title)
            page_num = cvs.getPageNumber()
            cvs.setFont("Helvetica", 8)
            cvs.drawRightString(width - MARGIN_X, 20, f"Page {page_num}")

        # Cover
        add_header_footer(c, "Cover")
        c.setFont("Helvetica-Bold", 18)
        c.drawString(MARGIN_X, MARGIN_TOP, "Multi-Omics Biomarker Discovery Report")

        c.setFont("Helvetica", 11)
        c.drawString(
            MARGIN_X, MARGIN_TOP - 30, f"Cohort disease term: {disease_term}"
        )
        c.drawString(
            MARGIN_X,
            MARGIN_TOP - 46,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        mode = integration_sum.get("integration_mode", "N/A")
        core_layers = integration_sum.get("core_layers", [])
        external_layers = integration_sum.get("external_layers", [])
        n_core_samples = integration_sum.get("n_core_samples", "N/A")

        y = MARGIN_TOP - 80
        cov_text = (
            f"Integration mode: {mode}\n"
            f"Core layers   : {', '.join(core_layers) if core_layers else 'N/A'}\n"
            f"External layers: {', '.join(external_layers) if external_layers else 'None'}\n"
            f"Core samples used for ML: {n_core_samples}\n\n"
            "This report presents a summary of: data quality, normalization, integration, "
            "machine learning–based biomarker ranking, and PubMed-supported biological "
            "interpretation."
        )
        y = draw_wrapped(c, cov_text, MARGIN_X, y, CONTENT_WIDTH, leading=12, font="Helvetica", size=10)
        c.showPage()

        # Omics overview
        add_header_footer(c, "Omics overview")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, MARGIN_TOP, "Omics Layer Overview")

        y = MARGIN_TOP - 25
        txt = (
            "Summary of all omics layers used in the pipeline, including sample counts, "
            "feature counts, and role in the integration (core vs external)."
        )
        y = draw_wrapped(c, txt, MARGIN_X, y, CONTENT_WIDTH, leading=11)

        y -= 5
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN_X, y, "Layer")
        c.drawString(MARGIN_X + 120, y, "Role")
        c.drawString(MARGIN_X + 200, y, "Samples")
        c.drawString(MARGIN_X + 280, y, "Features")
        y -= 12
        c.setFont("Helvetica", 9)

        for row in layer_summary_df.itertuples():
            if y < 80:
                c.showPage()
                add_header_footer(c, "Omics overview (cont.)")
                y = MARGIN_TOP - 20
                c.setFont("Helvetica-Bold", 9)
                c.drawString(MARGIN_X, y, "Layer")
                c.drawString(MARGIN_X + 120, y, "Role")
                c.drawString(MARGIN_X + 200, y, "Samples")
                c.drawString(MARGIN_X + 280, y, "Features")
                y -= 12
                c.setFont("Helvetica", 9)

            c.drawString(MARGIN_X, y, str(row.layer))
            c.drawString(MARGIN_X + 120, y, str(row.role))
            c.drawString(MARGIN_X + 200, y, str(row.n_samples))
            c.drawString(MARGIN_X + 280, y, str(row.n_features))
            y -= 12

        c.showPage()

        # Per-layer QC & normalization pages
        for row in layer_summary_df.itertuples():
            lname = row.layer
            panels = layer_panels.get(lname, {})
            qc_plot = panels.get("qc", None)
            norm_plot = panels.get("norm", None)

            if not qc_plot and not norm_plot:
                continue

            add_header_footer(c, f"{lname} QC & normalization")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(MARGIN_X, MARGIN_TOP, f"{lname.capitalize()} — QC & normalization")

            y_top = MARGIN_TOP - 30
            c.setFont("Helvetica", 9)
            y_top = draw_wrapped(
                c,
                "QC and normalization plots for this omics layer.",
                MARGIN_X,
                y_top,
                CONTENT_WIDTH,
                leading=11,
            )

            img_h = 3.3 * inch
            img_w = (CONTENT_WIDTH - 10) / 2
            y_img = y_top - 10

            if qc_plot and os.path.exists(qc_plot):
                try:
                    img = ImageReader(qc_plot)
                    c.drawImage(
                        img,
                        MARGIN_X,
                        y_img - img_h,
                        width=img_w,
                        height=img_h,
                        preserveAspectRatio=True,
                    )
                except Exception as e:
                    c.drawString(MARGIN_X, y_img - 12, f"Could not render QC plot: {e}")

            if norm_plot and os.path.exists(norm_plot):
                try:
                    img = ImageReader(norm_plot)
                    c.drawImage(
                        img,
                        MARGIN_X + img_w + 10,
                        y_img - img_h,
                        width=img_w,
                        height=img_h,
                        preserveAspectRatio=True,
                    )
                except Exception as e:
                    c.drawString(
                        MARGIN_X + img_w + 10,
                        y_img - 12,
                        f"Could not render normalization plot: {e}",
                    )

            c.showPage()

        # ML & cross-omics evidence plots
        def place_plots_in_grid(cvs, img_paths, title):
            idx = 0
            while idx < len(img_paths):
                add_header_footer(cvs, title)
                y_top = MARGIN_TOP - 20
                img_h = 3.3 * inch
                img_w = CONTENT_WIDTH
                for _ in range(2):
                    if idx >= len(img_paths):
                        break
                    p = img_paths[idx]
                    if os.path.exists(p):
                        try:
                            img = ImageReader(p)
                            cvs.drawImage(
                                img,
                                MARGIN_X,
                                y_top - img_h,
                                width=img_w,
                                height=img_h,
                                preserveAspectRatio=True,
                            )
                            y_top -= img_h + 10
                        except Exception as e:
                            cvs.setFont("Helvetica", 9)
                            cvs.drawString(
                                MARGIN_X,
                                y_top - 12,
                                f"Could not render {os.path.basename(p)}: {e}",
                            )
                            y_top -= 20
                    idx += 1
                cvs.showPage()

        place_plots_in_grid(
            c, plots_step4 + plots_step5, "ML & cross-omics evidence (cont.)"
        )

        # PubMed barplot & heatmap
        add_header_footer(c, "Literature evidence")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, MARGIN_TOP, "Literature evidence (PubMed)")

        y = MARGIN_TOP - 25
        txt = (
            f"Top {TOP_PLOT} biomarkers were queried in PubMed with the disease term "
            f'"{disease_term}". Hit counts and keyword-targeted evidence '
            "(diagnostic, prognostic, predictive, mechanistic) are shown below."
        )
        y = draw_wrapped(c, txt, MARGIN_X, y, CONTENT_WIDTH, leading=11)

        img_h = 3.3 * inch
        img_w = (CONTENT_WIDTH - 10) / 2
        y_img = y - 10

        if barplot_path and os.path.exists(barplot_path):
            try:
                img = ImageReader(barplot_path)
                c.drawImage(
                    img,
                    MARGIN_X,
                    y_img - img_h,
                    width=img_w,
                    height=img_h,
                    preserveAspectRatio=True,
                )
            except Exception as e:
                c.drawString(MARGIN_X, y_img - 12, f"Could not render barplot: {e}")

        if heatmap_path and os.path.exists(heatmap_path):
            try:
                img = ImageReader(heatmap_path)
                c.drawImage(
                    img,
                    MARGIN_X + img_w + 10,
                    y_img - img_h,
                    width=img_w,
                    height=img_h,
                    preserveAspectRatio=True,
                )
            except Exception as e:
                c.drawString(
                    MARGIN_X + img_w + 10,
                    y_img - 12,
                    f"Could not render heatmap: {e}",
                )

        c.showPage()

        # Top biomarker summary table
        TOP_TABLE = min(TOP_N_IN_REPORT, len(pubmed_df))
        add_header_footer(c, "Top biomarkers summary")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, MARGIN_TOP, "Top biomarkers (extended score + PubMed)")

        y = MARGIN_TOP - 25
        txt = (
            f"The table below lists the top {TOP_TABLE} biomarkers by extended biology "
            "score and summarizes their PubMed evidence."
        )
        y = draw_wrapped(c, txt, MARGIN_X, y, CONTENT_WIDTH, leading=11)

        c.setFont("Helvetica-Bold", 9)
        y -= 5
        c.drawString(MARGIN_X, y, "#")
        c.drawString(MARGIN_X + 20, y, "Biomarker")
        c.drawString(MARGIN_X + 150, y, "Layer")
        c.drawString(MARGIN_X + 220, y, "Ext. score")
        c.drawString(MARGIN_X + 310, y, "PubMed hits")
        y -= 12
        c.setFont("Helvetica", 9)

        for i in range(TOP_TABLE):
            row = pubmed_df.iloc[i]
            if y < 80:
                c.showPage()
                add_header_footer(c, "Top biomarkers summary (cont.)")
                y = MARGIN_TOP - 25
                c.setFont("Helvetica-Bold", 9)
                c.drawString(MARGIN_X, y, "#")
                c.drawString(MARGIN_X + 20, y, "Biomarker")
                c.drawString(MARGIN_X + 150, y, "Layer")
                c.drawString(MARGIN_X + 220, y, "Ext. score")
                c.drawString(MARGIN_X + 310, y, "PubMed hits")
                y -= 12
                c.setFont("Helvetica", 9)

            biomarker = str(row["biomarker"])
            layer = str(row["layer"])
            score = float(row["extended_biology_score"])
            hits = int(row["pubmed_hits"])

            c.drawString(MARGIN_X, y, str(i + 1))
            c.drawString(MARGIN_X + 20, y, biomarker[:25])
            c.drawString(MARGIN_X + 150, y, layer)
            c.drawString(MARGIN_X + 220, y, f"{score:.2f}")
            c.drawString(MARGIN_X + 310, y, str(hits))
            y -= 12

        c.showPage()

        # Per-biomarker literature summaries
        add_header_footer(c, "Biomarker literature summaries")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, MARGIN_TOP, "Representative literature for top biomarkers")

        y = MARGIN_TOP - 25
        txt = (
            "For each biomarker, representative PubMed abstracts were retrieved and, "
            "when possible, summarized using a GPT-style model. Below are concise "
            "biological summaries followed by top article references."
        )
        y = draw_wrapped(c, txt, MARGIN_X, y, CONTENT_WIDTH, leading=11)

        for i in range(TOP_TABLE):
            row = pubmed_df.iloc[i]
            biomarker = str(row["biomarker"])
            layer = str(row["layer"])
            hits = int(row["pubmed_hits"])
            summary = bm_summaries.get(biomarker, "No summary available.")
            articles = bm_articles.get(biomarker, [])

            if y < 140:
                c.showPage()
                add_header_footer(c, "Biomarker literature summaries (cont.)")
                y = MARGIN_TOP - 20

            c.setFont("Helvetica-Bold", 11)
            c.drawString(
                MARGIN_X,
                y,
                f"{i+1}. {biomarker} (Layer: {layer}, PubMed hits: {hits})",
            )
            y -= 14
            c.setFont("Helvetica", 9)
            y = draw_wrapped(c, summary, MARGIN_X, y, CONTENT_WIDTH, leading=11)
            y -= 4

            if articles:
                c.setFont("Helvetica-Bold", 9)
                c.drawString(MARGIN_X, y, "Top references:")
                y -= 12
                c.setFont("Helvetica", 8)
                for j, art in enumerate(articles[:2]):
                    title = art.get("title", "")
                    year = art.get("year", "")
                    journal = art.get("journal", "")
                    pmid = art.get("pmid", "")
                    ref_text = f"[{j+1}] {title} ({journal}, {year}). PMID: {pmid}"
                    y = draw_wrapped(
                        c,
                        ref_text,
                        MARGIN_X + 10,
                        y,
                        CONTENT_WIDTH - 10,
                        leading=10,
                        size=8,
                    )
                    y -= 4

            y -= 6

        c.showPage()

        # Final notes
        add_header_footer(c, "Notes")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, MARGIN_TOP, "Notes & caveats")

        y = MARGIN_TOP - 25
        txt = (
            "This report is derived from multi-omics data integrated via an adaptive "
            "pipeline. Biomarker ranking is based on extended scores combining ML "
            "importance, cross-omics support, and network centrality.\n\n"
            "PubMed-based evidence and GPT-style summaries provide a structured starting "
            "point for interpretation, but final biological conclusions should be "
            "validated by domain experts and, where possible, supported by experimental "
            "data."
        )
        y = draw_wrapped(c, txt, MARGIN_X, y, CONTENT_WIDTH, leading=11)

        c.showPage()
        c.save()
        logger.info("STEP 6 PDF report saved at: %s", REPORT_PATH)
    else:
        REPORT_PATH = ""
        logger.warning(
            "reportlab not installed; skipping PDF report generation in Step 6."
        )

    # ------------- Return a compact summary dict -------------
    return {
        "step_dir": step6_dir,
        "pubmed_summary": pubmed_df_path,
        "keyword_matrix": kw_df_path,
        "literature_summaries": bm_summ_path,
        "articles": bm_articles_path,
        "barplot": barplot_path,
        "heatmap": heatmap_path,
        "report_pdf": REPORT_PATH,
    }
