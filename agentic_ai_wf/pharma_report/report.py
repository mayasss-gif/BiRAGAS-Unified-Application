import logging
import os
import io
import pandas as pd
from pathlib import Path
from .CohortModule.cohort import (
    analyze_cohort_data, extract_sample_metadata_from_json, tissue_data, 
    plot_country_instrument_combined, plot_characteristics_distribution, 
    plot_tissue_distribution, prepare_country_instrument_df, add_data_labels
)
from .GeneModule.gene import (
    plot_lfc_grouped_bar, plot_bubble_chart, get_gene_clinical_summary_by_tier, 
    summarize_gene_counts
)
from .PathwayModule.pathway import (
    plot_pathway_class_distribution, get_pathway_regulation_summary, 
    extract_top_pathway_narratives
)
from .DrugModule.drug import (
    get_drug_summary_counts, get_top_5_approved_drugs, get_top_5_not_approved_drugs, 
    generate_table_summary
)
from .DEGModule.deg import (
    plot_mini_volcano, plot_deg_distribution_by_fold_change, plot_top20_variance_heatmap, 
    disease_description, deg_profile
)
from .HarmonizationModule.harmonization import (
    get_stats_sections_from_txt, plot_log2fc_by_origin
)
from .utils import plot_and_get_base64, generate_chart_explanation, ensure_valid_base64_for_html


logger = logging.getLogger(__name__)


def pharmareport(path, output_html_path, patient_prefix):
    logger.info("Starting pharmaceutical report generation.")
    try:
        # Validate input parameters
        if not path or not isinstance(path, dict):
            raise ValueError("Invalid path parameter: must be a dictionary")
        if not output_html_path:
            raise ValueError("output_html_path cannot be empty")
        if not patient_prefix:
            raise ValueError("patient_prefix cannot be empty")
            
        # Log input paths for debugging
        logger.info(f"Input paths: {path}")
        logger.info(f"Output HTML path: {output_html_path}")
        logger.info(f"Patient prefix: {patient_prefix}")
        
        # Validate required path components
        required_keys = ['Cohort', 'Harmonization', 'DEG', 'Gene', 'Pathway', 'Drug']
        for key in required_keys:
            if key not in path:
                logger.warning(f"Missing required path key: {key}")
        
        # Setup output directory for images
        output_path = Path(output_html_path)
        images_dir = output_path.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        images_dir_str = str(images_dir)
        logger.info(f"Images will be saved to: {images_dir}")
        
        # -------------------------Cohort Module-----------------------------
        logger.info("Processing Cohort Module...")
        try:
            json_path = path['Cohort']
            if not json_path or not Path(json_path).exists():
                logger.warning(f"Cohort JSON file not found: {json_path}. Cohort module will be skipped.")
                df = None
            else:
                df = extract_sample_metadata_from_json(json_path)
                logger.info(f"Successfully extracted sample metadata from JSON DataFrame df : {df}")
        except Exception as cohort_error:
            logger.error(f"Error processing cohort module: {cohort_error}")
            df = None
        # Initialize all base64 variables to prevent undefined variables
        tissue_base64 = char_base64 = combined_base64 = ""
        tissue_explanation = char_explanation = combined_explanation = ""
        histogram_base64 = minivol_base64 = dfc_base64 = heatmap_b64 = ""
        lfc_base64 = up_b64 = down_b64 = fc_base64 = ""
        histogram_explanation = minivol_explanation = dfc_explanation = heatmap_explanation = ""
        lfc_explanation = up_explanation = down_explanation = fc_explanation = ""
        summary_cohort = ""

        if df is not None and not isinstance(df, list) and not df.empty:
            # Call the function
            summary_cohort = analyze_cohort_data(json_path)
            logger.info("Successfully analyzed cohort data.")
            tissue_df = tissue_data(json_path)
            logger.info("Successfully extracted tissue data")
            tissue_base64 = plot_and_get_base64(
                plot_tissue_distribution, output_dir=images_dir_str, 
                image_name=f"{patient_prefix}_tissue_distribution", tissue_df=tissue_df)
            logger.info("Successfully plotted tissue distribution.")
            char_base64 = plot_and_get_base64(
                plot_characteristics_distribution,
                output_dir=images_dir_str,
                image_name=f"{patient_prefix}_characteristics_distribution",
                df_exploded=df.explode("Characteristics")
            )
            if not char_base64:
                logger.warning("Characteristics distribution plot returned empty/invalid; continuing without it.")
            else:
                logger.info("Successfully plotted characteristics distribution.")
            combined_base64 = plot_and_get_base64(
                plot_country_instrument_combined, output_dir=images_dir_str,
                image_name=f"{patient_prefix}_country_instrument_combined", df=df)
            logger.info(
                "Successfully plotted country/instrument distribution.")
            tissue_explanation = generate_chart_explanation(tissue_base64) if tissue_base64 else ""
            logger.info("Successfully generated tissue explanation.")
            char_explanation = generate_chart_explanation(char_base64) if char_base64 else ""
            logger.info("Successfully generated characteristics explanation.")
            combined_explanation = generate_chart_explanation(combined_base64) if combined_base64 else ""
            logger.info("Successfully generated combined explanation.")

        else:
            print("No valid data in df; skipping plots.")
            logger.info("Successfully generated combined explanation.")

        logger.info("Cohort Module processing complete.")

        # ---------------------------Harmonization Module --------------------------
        logger.info("Processing Harmonization Module...")
        txt_path = path["Harmonization"][0]
        sections = get_stats_sections_from_txt(txt_path)
        logger.info("Successfully extracted stats sections from text.")
        csv_path = path["Harmonization"][1]

        try:
            histogram_base64 = plot_and_get_base64(
                plot_log2fc_by_origin, output_dir=images_dir_str,
                image_name=f"{patient_prefix}_log2fc_histogram", csv_path=csv_path, patient_id=patient_prefix)
            logger.info("Successfully plotted log2fc histogram.")

            histogram_explanation = generate_chart_explanation(
                histogram_base64)
            logger.info("Successfully generated histogram explanation.")
            logger.info("Harmonization Module processing complete.")
        except Exception as e:
            logger.error(f"Error in Harmonization Module: {e}")
            histogram_base64 = None
            histogram_explanation = None
            logger.info("Harmonization Module processing complete.")

        txt_path = path["Harmonization"][0]
        sections = get_stats_sections_from_txt(txt_path)
        print(sections)
        csv_path = path["Harmonization"][1]

        # --------------------- DEG Module --------------------------------------
        logger.info("Processing DEG Module...")
        csv_path = path["DEG"][0]

        minivol_base64 = plot_and_get_base64(plot_mini_volcano,
                                             output_dir=images_dir_str, image_name=f"{patient_prefix}_mini_volcano",
                                             csv_path=csv_path, patient_id=patient_prefix)
        logger.info("Successfully plotted mini volcano plot.")
        dfc_base64 = plot_and_get_base64(
            plot_deg_distribution_by_fold_change, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_deg_distribution", csv_path=csv_path)
        logger.info("Successfully plotted DEG distribution.")
        heatmap_b64 = plot_and_get_base64(
            plot_top20_variance_heatmap, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_top20_variance_heatmap", csv_path=csv_path, patient_id=patient_prefix)

        logger.info("Successfully plotted top 20 variance heatmap.")

        minivol_explanation = generate_chart_explanation(minivol_base64)
        logger.info("Successfully generated mini volcano explanation.")
        dfc_explanation = generate_chart_explanation(dfc_base64)
        logger.info("Successfully generated DEG distribution explanation.")
        heatmap_explanation = generate_chart_explanation(heatmap_b64)
        logger.info("Successfully generated heatmap explanation.")

        disease = disease_description(csv_path, path["DEG"][1])
        logger.info("Successfully generated disease description.")
        deg = deg_profile(csv_path, patient_prefix)
        logger.info("Successfully generated DEG profile.")
        logger.info("DEG Module processing complete.")

        # ---------------------- Gene Module ----------------------------------
        logger.info("Processing Gene Module...")
        csv_path = path["Gene"]

        summary_gene = summarize_gene_counts(csv_path)
        logger.info("Successfully summarized gene counts.")

        lfc_base64 = plot_and_get_base64(
            plot_lfc_grouped_bar, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_lfc_grouped_bar", csv_path=csv_path, patient_id=patient_prefix)
        logger.info("Successfully plotted LFC grouped bar chart.")
        up_b64 = plot_and_get_base64(
            plot_bubble_chart, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_upregulated_bubble", patient_id=patient_prefix, csv_path=csv_path, mode="UP")

        logger.info("Successfully plotted UP-regulated bubble chart.")
        down_b64 = plot_and_get_base64(
            plot_bubble_chart, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_downregulated_bubble", patient_id=patient_prefix, csv_path=csv_path, mode="DOWN")
        logger.info("Successfully plotted DOWN-regulated bubble chart.")

        lfc_explanation = generate_chart_explanation(lfc_base64)
        logger.info("Successfully generated LFC explanation.")
        up_explanation = generate_chart_explanation(up_b64)
        logger.info("Successfully generated UP-regulated explanation.")
        down_explanation = generate_chart_explanation(down_b64)
        logger.info("Successfully generated DOWN-regulated explanation.")

        gene_data = get_gene_clinical_summary_by_tier(
            csv_path, patient_id=patient_prefix)
        logger.info("Successfully retrieved gene clinical summary.")
        logger.info("Gene Module processing complete.")

        # --------------------- Pathway Module ---------------------------------
        logger.info("Processing Pathway Module...")
        csv_path = path["Pathway"]

        summary_pathway = get_pathway_regulation_summary(csv_path)
        logger.info("Successfully summarized pathway regulation.")
        fc_base64 = plot_and_get_base64(
            plot_pathway_class_distribution, output_dir=images_dir_str,
            image_name=f"{patient_prefix}_pathway_class_distribution", csv_path=csv_path)
        logger.info("Successfully plotted pathway class distribution.")
        fc_explanation = generate_chart_explanation(fc_base64)
        logger.info("Successfully generated pathway class explanation.")
        up_data, down_data = extract_top_pathway_narratives(csv_path)
        logger.info("Successfully extracted top pathway narratives.")
        logger.info("Pathway Module processing complete.")

        # ----------------------Drug Module -----------------------------
        logger.info("Processing Drug Module...")
        csv_path = path["Drug"]

        if csv_path is None:
            logger.warning("Using the sample drugs data.")
            csv_path = Path(
                "./agentic_ai_wf/pharma_report/data/sample_drugs.csv").resolve()

        logging.info("Reading data from CSV...")
        data = pd.read_csv(csv_path)

        summary_drug = get_drug_summary_counts(csv_path)
        
        # Ensure summary_drug is never None to prevent template errors
        if summary_drug is None:
            logger.warning("Drug summary is None, creating default summary")
            summary_drug = {
                "total_drugs": 0,
                "approved_drugs": 0,
                "not_approved_drugs": 0,
                "error_drugs": 0,
                "unclassified_drugs": 0,
                "total_pathways": 0,
                "approved_percentage": 0.0,
                "not_found_percentage": 0.0,
                "top_pathways": [],
                "data_quality": {
                    "total_rows": 0,
                    "unique_drugs": 0,
                    "coverage": 0.0
                }
            }
        
        logger.info("Successfully summarized drug counts.")
        top_5_approved_df = get_top_5_approved_drugs(csv_path)
        logger.info("Successfully retrieved top 5 approved drugs.")
        top_5_not_approved_df = get_top_5_not_approved_drugs(csv_path)
        logger.info("Successfully retrieved top 5 not-approved drugs.")

        if top_5_approved_df is not None:
            approved_df_text = generate_table_summary(top_5_approved_df)
            logger.info("Successfully generated summary for approved drugs.")

        else:

            approved_df_text = "No Drug Found"
        if top_5_not_approved_df is not None:
            nonapproved_df_text = generate_table_summary(top_5_not_approved_df)
            logger.info(
                "Successfully generated summary for non-approved drugs.")

        else:
            nonapproved_df_text = "No Drug Found"

        logger.info("Drug Module processing complete.")
        logger.info("All modules processed. Generating HTML report.")
        
        # Debug: Check base64 variables before HTML generation
        base64_vars = {
            'tissue_base64': len(tissue_base64) if tissue_base64 else 0,
            'char_base64': len(char_base64) if char_base64 else 0,
            'combined_base64': len(combined_base64) if combined_base64 else 0,
            'histogram_base64': len(histogram_base64) if histogram_base64 else 0,
            'minivol_base64': len(minivol_base64) if minivol_base64 else 0,
            'dfc_base64': len(dfc_base64) if dfc_base64 else 0,
            'heatmap_b64': len(heatmap_b64) if heatmap_b64 else 0,
            'lfc_base64': len(lfc_base64) if lfc_base64 else 0,
            'up_b64': len(up_b64) if up_b64 else 0,
            'down_b64': len(down_b64) if down_b64 else 0,
            'fc_base64': len(fc_base64) if fc_base64 else 0,
        }
        logger.info(f"Base64 variables lengths: {base64_vars}")
        
        # Validate and ensure all base64 strings are valid for HTML embedding
        tissue_base64 = ensure_valid_base64_for_html(tissue_base64, "tissue_distribution") if tissue_base64 else ""
        char_base64 = ensure_valid_base64_for_html(char_base64, "characteristics_distribution") if char_base64 else ""
        combined_base64 = ensure_valid_base64_for_html(combined_base64, "country_instrument_combined") if combined_base64 else ""
        histogram_base64 = ensure_valid_base64_for_html(histogram_base64, "log2fc_histogram") if histogram_base64 else ""
        minivol_base64 = ensure_valid_base64_for_html(minivol_base64, "mini_volcano") if minivol_base64 else ""
        dfc_base64 = ensure_valid_base64_for_html(dfc_base64, "deg_distribution") if dfc_base64 else ""
        heatmap_b64 = ensure_valid_base64_for_html(heatmap_b64, "top20_variance_heatmap") if heatmap_b64 else ""
        lfc_base64 = ensure_valid_base64_for_html(lfc_base64, "lfc_grouped_bar") if lfc_base64 else ""
        up_b64 = ensure_valid_base64_for_html(up_b64, "upregulated_bubble") if up_b64 else ""
        down_b64 = ensure_valid_base64_for_html(down_b64, "downregulated_bubble") if down_b64 else ""
        fc_base64 = ensure_valid_base64_for_html(fc_base64, "pathway_class_distribution") if fc_base64 else ""
        
        logger.info("Successfully validated all base64 strings for HTML embedding.")
        logger.info("Successfully generated HTML content.")

        # ---------------------------------Cohort HTML ------------------------------------
        html_cohort = ""

        if summary_cohort and summary_cohort != {}:
            html_cohort += f"""
            <div class="content-header">
                <h2><span class="tab-icon">📊</span> Cohort Data Retrieval Module</h2>
                <p>Comprehensive public dataset mining and cohort characterization</p>
            </div>

            <div class="content-section">
                <div class="section-header">
                    <span>Cohort Module</span>
                </div>
                <div class="section-content">
                    <div class="subsection">
                        <div class="metric-grid">
                            <div class="metric-card">
                                <div class="metric-value">{summary_cohort['total_datasets']}</div>
                                <div class="metric-label">Total Datasets Size</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{summary_cohort['total_samples']}</div>
                                <div class="metric-label">Total Cohort Size</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{summary_cohort['unique_tissue_types']}</div>
                                <div class="metric-label">Tissue Types</div>
                            </div>
                        </div>
                    </div>
                    <div class="subsection">
                        <h4>Tissue Distribution Across Retrieved Cohorts Bar plot</h4>
                        <p>{tissue_explanation}</p>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{tissue_base64}" alt="Dataset Relevance Chart" width="1000" height="500">
                        </div>
                    </div>
                    <div class="subsection">
                        <h4>Distribution of Samples by Country and Instrument Model</h4>
                        <p>{combined_explanation}</p>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{combined_base64}" alt="Dataset Relevance Chart" width="1000" height="500">
                        </div>
                    </div>
                    <div class="subsection">
                        <h4>Distribution of Samples by Cell Type and Clinical Status</h4>
                        <p>{char_explanation}</p>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{char_base64}" alt="Dataset Relevance Chart" width="1000" height="500">
                        </div>
                    </div>
                </div>
            </div>
            """
        else:
            html_cohort += """
            <div class="content-header">
                <h2><span class="tab-icon">📊</span> Cohort Data Retrieval Module</h2>
                <p>No relevant RNA Seq datasets found.</p>
            </div>
            """

        # --------------------------------- Gene HTML -------------------------------------
        html_content = ""
        html_content += '<div class="subsection">\n'
        for tier, data in gene_data.items():
            print(tier, data)
            html_content += f'<h4>{tier}</h4>\n'

            for direction in ['Upregulated Gene', 'Downregulated Gene']:
                html_content += '<div class="subsection">\n'
                html_content += f'<h4 style="text-align:center; text-transform:uppercase;">{direction}</h4>\n'

                if data[direction]:
                    for idx, gene_info in enumerate(data[direction], 1):
                        gene = gene_info["Gene"]
                        log2fc = gene_info["Log2FC"]
                        relevance = gene_info["Clinical_Relevance"]

                        html_content += f'<h5>{idx}. {gene}</h5>\n'
                        html_content += f'<p><strong>Log2FC:</strong> {log2fc}</p>\n'
                        html_content += f'<p style="margin-bottom: 20px;"><strong>Clinical Relevance:</strong> {relevance}</p>\n'

                else:
                    html_content += f'<p>No {direction.lower()} genes in this tier.</p>\n'
                html_content += '</div>'  # Note: use += instead of =

        html_content += '</div>'  # Note: use += instead of =

        # ---------------------- Pathway HTML -------------------------------------------------------
        # Pathway narrative HTML
        html_pathways = ""
        for label, data in [("Upregulated Pathways", up_data), ("Downregulated Pathways", down_data)]:
            html_pathways += '<div class="subsection">\n'
            html_pathways += f'<h4 style="text-align:center; text-transform:uppercase;">{label}</h4>\n'

            if data:
                for idx, entry in enumerate(data, 1):
                    html_pathways += '<div class="pathway-block">\n'
                    html_pathways += f'<h5>{idx}. {entry["Pathway_Name"]}</h5>\n'
                    html_pathways += f'<p><strong>Regulation:</strong> {entry["Regulation"]}</p>\n'
                    html_pathways += f'<p><strong>LLM Score:</strong> {entry["LLM_Score"]}</p>\n'
                    html_pathways += f'<p><strong>Confidence Level:</strong> {entry["Confidence_Level"]}</p>\n'
                    html_pathways += f'<p><strong>Priority Rank:</strong> {entry["Priority_Rank"]}</p>\n'
                    html_pathways += f'<p style="margin-bottom: 20px;"><strong>Score Justification:</strong> {entry["Score Justification"]}</p>\n'
                    html_pathways += '</div>\n'
            else:
                html_pathways += f'<p>No {label.lower()} available.</p>\n'

            html_pathways += '</div>\n'

        # ---------------------------------------Drug HTML ---------------------------------------------------------------------------------------
        def generate_drug_html_table(title, explanation, df):
            try:
                df = df.sort_values(by=['molecular_evidence_score', 'llm_score'], ascending=[False, False])
            except Exception as e:
                print(f"Error in sorting df: {e}")
                df = df

            html = '<div class="subsection">\n'
            html += f'<h4 >{title}</h4>\n'
            html += f'<p style="margin-bottom: 20px;">{explanation}</p>\n'

            if df is None or df.empty:
                pass
            else:
                html += '<table border="1" cellspacing="0" cellpadding="6" style="width:100%; border-collapse:collapse;">\n'
                html += (
                    "<thead><tr>"
                    "<th>Drug</th>"
                    "<th>Pathway</th>"
                    "<th>Molecular Evidence Score</th>"
                    "<th>Evidence Summary</th>"
                    "<th>Target Mechanism</th>"
                    "<th>FDA Approved Status</th>"
                    "</tr></thead>\n"
                )
                html += "<tbody>\n"
                for _, row in df.iterrows():
                    html += "<tr>"
                    html += f"<td>{row.get('drug_name', 'N/A')}</td>"
                    html += f"<td>{row.get('pathway_name', 'N/A')}</td>"
                    html += f"<td>{row.get('molecular_evidence_score', 'N/A')}</td>"
                    html += f"<td>{row.get('justification', 'N/A')}</td>"
                    html += f"<td>{row.get('target_mechanism', 'N/A')}</td>"
                    html += f"<td>{row.get('fda_approved_status', 'N/A')}</td>"
                    html += "</tr>\n"
                html += "</tbody>\n</table>\n"
            html += "</div>\n"
            return html

        html_drug = ""
        html_drug += generate_drug_html_table(
            "Approved Drugs", approved_df_text, top_5_approved_df)
        html_drug += generate_drug_html_table(
            "Additional Therapeutic Insights", nonapproved_df_text, top_5_not_approved_df)

        # Final HTML layout
        final_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agentic AI Framework Portal</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.18.2/plotly.min.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}

                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}

                .portal-header {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    padding: 20px 0;
                    text-align: center;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                }}

                .portal-title {{
                    font-size: 2.5em;
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 5px;
                    font-weight: bold;
                }}

                        .portal-subtitle {{
                    color: #666;
                    font-size: 1.1em;
                }}

                .top-tabs {{
                    background: rgba(255, 255, 255, 0.9);
                    backdrop-filter: blur(10px);
                    display: flex;
                    justify-content: center;
                    flex-wrap: wrap;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    overflow-x: auto;
                    padding: 0 10px;
                }}

                .tab-button {{
                    padding: 15px 20px;
                    background: transparent;
                    border: none;
                    cursor: pointer;
                    font-size: 0.9em;
                    font-weight: 500;
                    color: #666;
                    transition: all 0.3s ease;
                    border-bottom: 3px solid transparent;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    min-width: 180px;
                    justify-content: center;
                    white-space: nowrap;
                    flex-shrink: 0;
                }}

                .tab-button:hover {{
                    background: rgba(102, 126, 234, 0.1);
                    color: #667eea;
                }}

                .tab-button.active {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-bottom-color: #4facfe;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                }}

                .tab-icon {{
                    font-size: 1.2em;
                }}

                .main-content {{
                    padding: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}

                .tab-content {{
                    display: none;
                    animation: fadeIn 0.5s ease;
                }}

                .tab-content.active {{
                    display: block;
                }}

                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                .content-header {{
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    border-radius: 20px;
                    margin-bottom: 25px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    backdrop-filter: blur(10px);
                    text-align: center;
                }}

                .content-header h2 {{
                    color: #4facfe;
                    margin-bottom: 10px;
                    font-size: 2.2em;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 15px;
                }}

                .content-header p {{
                    color: #666;
                    font-size: 1.1em;
                }}

                .content-section {{
                    background: rgba(255, 255, 255, 0.95);
                    margin-bottom: 25px;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                    backdrop-filter: blur(10px);
                    transition: all 0.3s ease;
                }}

                .content-section:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
                }}


                .content-section p {{
                    color: #666;
                    font-size: 1.1em;
                }}

                .section-header {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px 25px;
                    cursor: pointer;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 1.1em;
                    font-weight: 600;
                }}

                .section-content {{
                    padding: 25px;
                    display: block;
                }}

                .section-content.active {{
                    display: block;
                    animation: slideDown 0.3s ease;
                }}

                @keyframes slideDown {{
                    from {{ opacity: 0; transform: translateY(-10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                .subsection {{
                    margin-bottom: 25px;
                    border-left: 4px solid #4facfe;
                    padding-left: 20px;
                }}

                .subsection p {{
                    color: #666;
                    font-size: 1.1em;
                }}

                .subsection h4 {{
                    color: #667eea;
                    margin-bottom: 15px;
                    font-size: 1.3em;
                }}

                .subsection h5 {{
                    color: #5a67d8; /* Slightly darker than h4 for distinction */
                    margin-bottom: 10px;
                    font-size: 1.1em;
                    font-weight: 800;
                }}

                .subsection h6 {{
                    color: #4c51bf; /* Even darker shade */
                    margin-bottom: 8px;
                    font-size: 1em;
                    font-weight: 800;
                }}


                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 20px;
                    margin: 25px 0;
                }}

                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                }}

                .metric-card:hover {{
                    transform: scale(1.05) translateY(-5px);
                }}

                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 8px;
                }}

                .metric-label {{
                    font-size: 1em;
                    opacity: 0.9;
                }}

                .chart-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin: 25px 0;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                }}

                .chart-title {{
                    text-align: center;
                    margin-bottom: 25px;
                    color: #667eea;
                    font-size: 1.4em;
                    font-weight: 600;
                }}

                .download-btn {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 30px;
                    cursor: pointer;
                    margin: 8px;
                    transition: all 0.3s ease;
                    font-size: 1em;
                    font-weight: 500;
                }}

                .download-btn:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                }}

                .gene-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
                    gap: 12px;
                    margin: 20px 0;
                }}

                .gene-tag {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 12px 16px;
                    border-radius: 25px;
                    text-align: center;
                    font-size: 0.9em;
                    font-weight: 600;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }}

                .gene-tag:hover {{
                    transform: scale(1.05);
                }}

                .pathway-item {{
                    background: rgba(102, 126, 234, 0.1);
                    border-left: 4px solid #667eea;
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 8px;
                    transition: all 0.3s ease;
                }}

                .pathway-item:hover {{
                    background: rgba(102, 126, 234, 0.15);
                    transform: translateX(5px);
                }}

                .pathway-name {{
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 8px;
                    font-size: 1.1em;
                }}

                .pathway-stats {{
                    font-size: 0.95em;
                    color: #666;
                }}

                .drug-card {{
                    background: white;
                    border: 2px solid #4facfe;
                    border-radius: 15px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease;
                }}

                .drug-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                }}

                .drug-name {{
                    font-weight: bold;
                    color: #667eea;
                    font-size: 1.2em;
                    margin-bottom: 8px;
                }}

                .drug-targets {{
                    color: #666;
                    font-size: 1em;
                    line-height: 1.5;
                }}

                .toggle-arrow {{
                    transition: transform 0.3s ease;
                    font-size: 1.2em;
                }}

                .toggle-arrow.active {{
                    transform: rotate(180deg);
                }}

                .tier-badge {{
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin-left: 10px;
                }}

                .tier-1 {{ background: #ff6b6b; color: white; }}
                .tier-2 {{ background: #ffa726; color: white; }}
                .tier-3 {{ background: #66bb6a; color: white; }}

                
                .score-badge {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: bold;
                    display: inline-block;
                    margin-top: 10px;
                }}

                @media (max-width: 768px) {{
                    .portal-title {{
                        font-size: 2em;
                    }}
                    
                    .top-tabs {{
                        justify-content: flex-start;
                    }}
                    
                    .tab-button {{
                        min-width: 150px;
                        font-size: 0.85em;
                        padding: 12px 15px;
                    }}
                    
                    .metric-grid {{
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 15px;
                    }}

                    .content-header h2 {{
                        font-size: 1.8em;
                        flex-direction: column;
                        gap: 10px;
                    }}
                }}

                p {{
                    font-size: 20px;  
                }}

                .table-container {{
                    overflow-x: auto;
                    margin-top: 20px;
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    border-radius: 12px;
                    overflow: hidden;
                    font-size: 0.95em;
                    background: white;
                }}

                thead {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-weight: bold;
                }}

                th, td {{
                    padding: 14px 16px;
                    text-align: left;
                    border-bottom: 1px solid #e6e6e6;
                }}

                tr:hover {{
                    background-color: #eef2ff; /* light lavender-blue to match your theme */
                    color: #333; /* ensures text remains visible */
                    transition: background-color 0.2s ease;
                }}

                th:first-child, td:first-child {{
                    border-left: none;
                }}

                th:last-child, td:last-child {{
                    border-right: none;
                }}

                tbody tr:last-child td {{
                    border-bottom: none;
                }}
            </style>
        </head>
        <body>
            <div class="portal-header">
                <h1 class="portal-title">🧬 Agentic AI Framework</h1>
                <p class="portal-subtitle">Transcriptomic Analysis & Therapeutic Targeting Portal</p>
            </div>


            <div class="main-content">
                <!-- Cohort Data Retrieval Module -->
                <div id="cohort-content">
                    {html_cohort}
                </div>

                <!-- Harmonization Mapping Module -->
                <div id="harmonization-content"">
                    <div class="content-header">
                        <h2><span class="tab-icon">🧬</span> Harmonization Mapping Module </h2>
                        <p>Cross-platform normalization and batch effect correction</p>
                    </div>

                <div class="content-section">
                    <div class="section-header">
                        <span>Harmonization Mapping Module</span>
                    </div>
                    
                    <div class="section-content">
                        <div class="subsection">
                            <h4>Log2FC Statistics Summary​</h4>
                            <p>{sections["summary"]}​​</p>
                        </div>

                        <div class="subsection">
                            <h4>Descriptive Statistics</h4>
                            <p>{sections["descriptive_statistics"]}​​</p>
                        </div>

                        <div class="subsection">
                            <h4>Distribution Shape</h4>
                            <p>{sections["distribution_shape"]}​​</p>
                        </div>

                        <div class="subsection">
                            <h4>Outlier Analysis</h4>
                            <p>{sections["outlier_analysis"]}​​</p>
                        </div>
                        
                        <div class="subsection">
                            <h4>Distribution of log₂ Fold-Change (log₂FC) </h4>
                            <p>{histogram_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{histogram_base64}" alt="Dataset Relevance Chart" width="800" height="700">
                            </div>
                        </div>

                    </div>
                </div>
                </div>
                <!-- DEG Mapping Module -->
                <div id="deg-content">
                <div class="content-header">
                    <h2><span class="tab-icon">🧬</span> Patient Transcriptome Profiling</h2>
                    <p>Comprehensive RNA-seq analysis and differential expression</p>
                </div>

                <div class="content-section">
                    <div class="section-header">
                        <span>Patient Transcriptome Profiling</span>
                    </div>
                    
                    <div class="section-content">

                        <div class="subsection">
                            <h4>1. Disease Activity Assessment​</h4>
                            <p>{disease}​​</p>
                        </div>

                        <div class="subsection">
                            <h4>Distribution of Differential Gene Expression</h4>
                            <p>{dfc_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{dfc_base64}" alt="Dataset Relevance Chart" width="800" height="700">
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>2. Differential Expression Profile​</h4>
                            <p>{deg}</p>
                        </div>
                        <div class="subsection">
                            <h4>Differential Gene Expression Analysis</h4>
                            <p>{minivol_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{minivol_base64}" alt="Dataset Relevance Chart" width="800" height="700">
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>Heatmap of Genes</h4>
                            <p>{heatmap_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{heatmap_b64}" alt="Dataset Relevance Chart" width="800" height="700">
                            </div>
                        </div>
                    </div>
                </div>
                </div>

                <!-- Gene Prioritization Module -->
                <div id="prioritization-content">
                    <div class="content-header">
                        <h2><span class="tab-icon">🎯</span> Gene Prioritization Module</h2>
                        <p>Machine learning-based ranking of candidate genes</p>
                    </div>

                <div class="content-section">
                    <div class="section-header">
                        <span>Gene Module</span>
                    </div>
                    <div class="section-content">
                        <div class="subsection">
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{summary_gene['total_genes']}</div>
                                    <div class="metric-label">Total No. of Gene</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_gene['tier1']}</div>
                                    <div class="metric-label">Tier1</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_gene['tier2']}</div>
                                    <div class="metric-label">Tier2</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_gene['tier3']}</div>
                                    <div class="metric-label">Tier3</div>
                                </div>
                            </div>

                            </div>
                        <div class="subsection">
                            <h4>Patient vs Cohort log₂FC</h4>
                            <p>{lfc_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{lfc_base64}" alt="Dataset Relevance Chart" width="1000" height="500">
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>Dot-Plot of Top Upregulated DEGs​</h4>
                            <p>{up_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{up_b64}" alt="Dataset Relevance Chart" width="1000" height="500">
                            </div>
                        </div>
                        <div class="subsection">
                            <h4>Dot-Plot of Top Downregulated DEGs​</h4>
                            <p>{down_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{down_b64}" alt="Dataset Relevance Chart" width="1000" height="500">
                            </div>
                        </div>
                    {html_content}
                    </div>
                </div>
                </div>
                <!-- Pathway Enrichment Module -->
                <div id="pathway-content">
                    <div class="content-header">
                        <h2><span class="tab-icon">🔗</span> Pathway Enrichment Module</h2>
                        <p>Functional enrichment and pathway set analysis</p>
                    </div>

                    <div class="content-section">
                    <div class="section-header">
                        <span>Pathway Module</span>
                    </div>
                    <div class="section-content">
                        <div class="subsection">
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{summary_pathway['total_pathways']}</div>
                                    <div class="metric-label">Total Pathways</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_pathway['up_count']}</div>
                                    <div class="metric-label">Upregulated Pathways</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_pathway['down_count']}</div>
                                    <div class="metric-label">Downregulated Pathways</div>
                                </div>
                            </div>

                             </div>
                        <div class="subsection">
                            <h4>Pathway Disease Activity Analysis​</h4>
                            <h4>Top Dysregulated Pathway Classes </h4>
                            <p>{fc_explanation}</p>
                            <div class="chart-container">
                                <img src="data:image/png;base64,{fc_base64}" alt="Dataset Relevance Chart" width="800" height="700">
                            </div>
                        </div>
                    {html_pathways}
                </div>
                <!-- Drug Mapping Module -->
                <div id="drug-content">
                    <div class="content-header">
                        <h2><span class="tab-icon">💊</span> Drug Mapping Module</h2>
                        <p>Target identification and drug repurposing analysis</p>
                    </div>

                    <div class="content-section">
                        <div class="section-header">
                            <span>Drug Module</span>
                        </div>
                        <div class="section-content">
                        <div class="subsection">
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{summary_drug['total_drugs']}</div>
                                    <div class="metric-label">Total Drugs</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_drug['approved_drugs']}</div>
                                    <div class="metric-label">Approved Drugs</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{summary_drug['not_approved_drugs']}</div>
                                    <div class="metric-label">Non Approved Drugs</div>
                                </div>
                            </div>

                        {html_drug}
                    </div>
                </div>
                        
            </div>
        </body>
        </html>
        """

        filepath = Path(output_html_path)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write to a standalone HTML file
        with open(filepath, "w", encoding="utf-8") as out_file:
            out_file.write(final_html)

        logger.info(f"Successfully wrote report to {output_html_path}.")
        logger.info(f"Chart images saved to: {images_dir}")

        logger.info(
            f"Pharmaceutical report generation complete. Saved to {output_html_path}")
        return filepath
    except Exception as e:
        logger.error(
            f"An error occurred during report generation: {e}", exc_info=True)
        raise
