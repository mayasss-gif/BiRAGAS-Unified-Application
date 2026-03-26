import pandas as pd
import os

from agentic_ai_wf.drug_agent_report.data_processing import data_processing
from agentic_ai_wf.drug_agent_report.data_analysis import data_analysis

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def generate_report(main_df, disease_name):
    main_df.rename({
        "name": "Drug",
        "pathway": "Pathway",
        "target-mechanism": "Target-Mechanism",
        "target only": "Targets",
        "deg_match_status": "Target Match Status",
        "approved": "Approved",
    }, axis=1, inplace=True)

    selected_columns = ["Drug", "Pathway", "Target-Mechanism", "Approved", "Notes"]
    styles = getSampleStyleSheet()
    para_style = styles["Normal"]
    para_style.fontSize = 6
    para_style.leading = 8

    def make_wrapped_table(df_slice, title, highlight_blue_rows=None):
        elements = []
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading4"]))
        elements.append(Spacer(1, 6))

        if df_slice.empty:
            elements.append(Paragraph("No data extracted", para_style))
            elements.append(Spacer(1, 12))
            return elements

        data = [selected_columns]
        blue_indices = []

        for idx, row in enumerate(df_slice[selected_columns].values.tolist()):
            wrapped_row = [Paragraph(str(cell), para_style) for cell in row]
            data.append(wrapped_row)
            if highlight_blue_rows is not None and highlight_blue_rows[idx]:
                blue_indices.append(idx + 1)  # +1 to account for header row

        table = Table(data, colWidths=[80, 100, 120, 40, 100])

        style = TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 6),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ('BOX', (0, 0), (-1, -1), 0, colors.white),
            ('INNERGRID', (0, 0), (-1, -1), 0, colors.white)
        ])

        for row_idx in blue_indices:
            style.add('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.blue)

        table.setStyle(style)
        elements.append(table)
        elements.append(Spacer(1, 12))
        return elements

    # Partition the data
    pan_df = main_df[main_df["pan_status"] == "Yes"].copy()
    remaining_df = main_df[main_df["pan_status"] != "Yes"]

    # Sort and prepare tables
    pan_df = pan_df.sort_values(by="Approved", ascending=False)
    pan_approved = pan_df[pan_df["Approved"] == "Yes"]
    if len(pan_approved) > 5:
        table1_df = pd.concat([pan_approved, pan_df[pan_df["Approved"] != "Yes"]])
    else:
        table1_df = pan_df.head(5)

    highlight_blue_rows = (table1_df["Approved"] == "Yes").tolist()

    table2_df = remaining_df[remaining_df["Approved"] == "Yes"].head(5)
    table3_df = remaining_df[remaining_df["Approved"] != "Yes"].head(5)

    # Build report elements
    os.makedirs("processed_data", exist_ok=True)
    
    main_df.to_csv(f"processed_data/{disease_name}_drugs_data.csv", index=False)
    
    doc = SimpleDocTemplate(f"processed_data/{disease_name}_drug_table_wrapped.pdf", pagesize=A4)
    elements = []
    elements += make_wrapped_table(table1_df, "Suggested Therapies", highlight_blue_rows)
    elements += make_wrapped_table(table2_df, "Relevant Approved Drugs")
    elements += make_wrapped_table(table3_df, "Other Unapproved Drugs")

    doc.build(elements)
    return elements, main_df

def data_reporting(process_pathways_dir, disease_name):
    try: 
        #df = pd.read_csv("processed_data/updated_main_df_pan.csv")
        df_procc = data_processing(process_pathways_dir)
        print("data processing completed.")
        
        print("Analyzing the data for report...")
        df_analyzed= data_analysis(df_procc, disease_name)
        print("data analysis ocmpleted.")

        df_analyzed.to_csv(f"processed_data/{disease_name}_drugs_data(whole).csv", index=False)

        #df_analyzed = pd.read_csv('processed_data/analyzed_data.csv')
        df = df_analyzed[df_analyzed["harmful_status"]=='No']

        return generate_report(df, disease_name)
    except Exception as exp:
       print("Exception Occurs - data reporting: ", str(exp)) 