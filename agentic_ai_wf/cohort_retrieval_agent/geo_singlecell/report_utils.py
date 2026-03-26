import os
import json
from typing import List, Dict, Any


def write_global_reports(processed: List[Dict[str, Any]],
                         outdir: str,
                         logger) -> None:
    if not processed:
        return

    json_path = os.path.join(outdir, "geo_singlecell_summary.json")
    tsv_path = os.path.join(outdir, "geo_singlecell_summary.tsv")
    pdf_path = os.path.join(outdir, "geo_singlecell_summary.pdf")

    # JSON
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(processed, fh, indent=2)
        logger.info("Wrote JSON summary: %s", json_path)
    except Exception as e:
        logger.error("Failed to write JSON summary: %s", e)

    # TSV (with has_feature_barcode_matrix)
    try:
        header = [
            "accession", "title", "taxon", "gdstype", "entrytype",
            "geo_link", "gse_dir",
            "num_suppl_files", "num_samples", "sample_ids",
            "has_feature_barcode_matrix",
        ]
        with open(tsv_path, "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for rec in processed:
                row = [
                    rec.get("accession", ""),
                    (rec.get("title") or "").replace("\t", " "),
                    (rec.get("taxon") or ""),
                    (rec.get("gdstype") or ""),
                    (rec.get("entrytype") or ""),
                    rec.get("geo_link", ""),
                    rec.get("gse_dir", ""),
                    str(rec.get("num_suppl_files", 0)),
                    str(rec.get("num_samples", 0)),
                    ",".join(rec.get("sample_ids", [])),
                    str(rec.get("has_feature_barcode_matrix", "")),
                ]
                fh.write("\t".join(row) + "\n")
        logger.info("Wrote TSV summary: %s", tsv_path)
    except Exception as e:
        logger.error("Failed to write TSV summary: %s", e)

    # PDF
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed; skipping PDF summary. Install with 'pip install fpdf2'.")
        return

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for rec in processed:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 8, f"{rec['accession']} – {rec.get('title', '')}")
            pdf.ln(2)

            pdf.set_font("Arial", "", 11)
            summary_txt = rec.get("summary") or ""
            if summary_txt:
                pdf.multi_cell(0, 6, f"Summary:\n{summary_txt}")
                pdf.ln(2)

            pdf.multi_cell(0, 6, f"GEO link: {rec['geo_link']}")
            pdf.multi_cell(0, 6, f"Local folder: {rec['gse_dir']}")
            pdf.multi_cell(0, 6, f"Suppl files: {rec['num_suppl_files']}")
            pdf.multi_cell(0, 6, f"Number of 10x sample folders: {rec['num_samples']}")

            hf = rec.get("has_feature_barcode_matrix", "")
            pdf.multi_cell(0, 6, f"Has feature/barcode/matrix files: {hf}")

            if rec["sample_ids"]:
                pdf.ln(2)
                pdf.set_font("Arial", "B", 11)
                pdf.multi_cell(0, 6, "Sample IDs:")
                pdf.set_font("Arial", "", 10)
                for sid in rec["sample_ids"]:
                    pdf.multi_cell(0, 5, f"• {sid}")
            pdf.ln(4)

        pdf.output(pdf_path)
        logger.info("Wrote PDF summary: %s", pdf_path)
    except Exception as e:
        logger.error("Failed to write PDF summary: %s", e)
