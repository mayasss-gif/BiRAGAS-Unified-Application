import os
import sys
import time
import argparse

from dotenv import load_dotenv

from .logging_utils import setup_logging
from .llm_utils import (
    init_openai_client,
    validate_biomedical_query,
    build_entrez_query_with_llm,
    normalize_biomedical_query,
    rank_gse_records_with_llm,
)
from .entrez_utils import (
    esearch_gds,
    collect_all_gse_records,
    filter_gse_records,
    filter_single_cell_records,
)
from .ftp_utils import (
    analyze_suppl_files,
    download_gse_singlecell_bundle,
)
from .report_utils import write_global_reports

from .validation_10x import process_outdir

load_dotenv()



def run_geo_sc_pipeline(
    disease : str,
    gse: list[str] = None,
    outdir: str = "geo_sc_data",
    retmax: int = 2000,
    max_gses: int = 10,
    no_rerank: bool = False,
    search_only: bool = False,
    log_file: str = "geo_singlecell.log",
) -> list[dict]:

    """
    Run the pipeline to search GEO for single-cell RNA-seq datasets.
    Args:
        disease: str, # Disease to search for.
        gse: list[str] = None, # List of GSE IDs to process.
        outdir: str = "geo_sc_data", # Output directory to store the downloaded datasets.
        retmax: int = 2000, # Maximum number of GEO records to retrieve.
        max_gses: int = 10, # Maximum number of GSEs to download or show after ranking/filtering.
        no_rerank: bool = False, # Disable LLM-based ranking and keep GEO ESearch order.
        search_only: bool = False, # Only search/analyze GEO datasets and show single-cell candidates; do not download.
        log_file: str = "geo_singlecell.log", # Path to log file.
    Returns:
        list[dict]: A list of dictionaries containing the summary of the downloaded datasets.
    Raises:
        ValueError: If you must provide either a disease or a list of GSE IDs.
        ValueError: If the output directory already exists.
        ValueError: If the log file already exists.
        ValueError: If the disease is not a valid biomedical/single-cell disease.
        ValueError: If the GSE IDs are not valid GSE IDs.
    """

    
    if not disease and not gse:
        raise ValueError("You must provide either a disease or a list of GSE IDs.")

    os.makedirs(outdir, exist_ok=True)
    logger = setup_logging(outdir + "/" + log_file)

    # -------------------------------------------------------
    # 1) DIRECT GSE MODE
    # -------------------------------------------------------
    if gse:
        logger.info("Direct GSE mode: processing explicit GSE IDs: %s", ", ".join(gse))
        processed_summaries = []

        for gse_id in gse:
            gse_id = gse_id.strip()
            if not gse_id.upper().startswith("GSE"):
                logger.warning("Skipping invalid GSE accession (does not start with GSE): %s", gse_id)
                continue

            try:
                # Mini search by accession to get metadata
                search_term = f"{gse_id}[Accession]"
                esearch_data = esearch_gds(search_term, retmax=1, logger=logger)
                es_result = esearch_data.get("esearchresult", {})
                idlist = es_result.get("idlist", [])
                webenv = es_result.get("webenv")
                query_key = es_result.get("querykey")

                if not idlist or not webenv or not query_key:
                    logger.warning("No GDS record found for %s; skipping.", gse_id)
                    continue

                recs = collect_all_gse_records(
                    webenv,
                    query_key,
                    total=len(idlist),
                    logger=logger,
                    batch_size=1,
                )
                if not recs:
                    logger.warning("No ESummary record resolved for %s; skipping.", gse_id)
                    continue

                record = recs[0]
                logger.info("Fetched metadata for %s | %s", record["accession"], record.get("title", ""))

                if search_only:
                    # analyze supplementary files but do NOT download
                    logger.info("Search-only mode: summarizing supplementary files for %s", gse_id)
                    analysis = analyze_suppl_files(gse_id, logger)
                    logger.info(
                        "GSE %s: is_10x_singlecell=%s (%s) | matrix=%d | barcodes=%d | features=%d | RAW.tar=%d | "
                        "anno_files=%d | fastq_like=%d",
                        gse_id,
                        analysis["is_10x_singlecell"],
                        analysis["is_10x_reason"],
                        len(analysis["matrix_files"]),
                        len(analysis["barcodes_files"]),
                        len(analysis["features_files"]),
                        len(analysis["raw_tar_files"]),
                        len(analysis["anno_files"]),
                        len(analysis["fastq_like"]),
                    )
                else:
                    summary = download_gse_singlecell_bundle(record, outdir, logger)
                    if summary:
                        processed_summaries.append(summary)

            except Exception as e:
                logger.error("Error while processing GSE %s in direct mode: %s", gse_id, e)

            time.sleep(0.5)

        if processed_summaries:
            write_global_reports(processed_summaries, outdir, logger)
            logger.info("=== GEO single-cell direct-GSE mode finished ===")
        else:
            print("No single-cell datasets could be downloaded for the provided GSE IDs.")
            logger.info("=== GEO single-cell direct-GSE mode finished (nothing downloaded) ===")
        return processed_summaries

    # -------------------------------------------------------
    # 2) NORMAL QUERY MODE (LLM + Entrez search)
    # -------------------------------------------------------
    user_query = disease
    logger.info("=== GEO single-cell search started ===")
    logger.info("User query: %s", user_query)
    logger.info("Output directory: %s", outdir)
    logger.info("retmax (ESearch): %d | max_gses (post-filter): %d", retmax, max_gses)

    client = init_openai_client(logger)

    # Guardrail: reject non-biomedical / non-single-cell queries
    if not validate_biomedical_query(client, user_query, logger):
        logger.error("Input query is not a valid biomedical/single-cell query. Aborting.")
        print(
            "This tool only supports biomedical / omics / single-cell queries.\n"
            "Examples:\n"
            "  - 'EGFR-mutant lung adenocarcinoma single-cell'\n"
            "  - 'breast carcinoma T cells scRNA-seq'\n"
            "  - 'NSCLC tumor microenvironment single-cell'\n"
            "  - 'PBMC lupus scRNA-seq'\n"
        )
        return []

    # Optional normalization / spell correction
    cleaned_query = normalize_biomedical_query(client, user_query, logger)
    if cleaned_query != user_query:
        logger.info("User query normalized from %r to %r", user_query, cleaned_query)
        print(f"\nNormalized query (fixed spelling/typos):\n  '{cleaned_query}'")
        print("Waiting 5 seconds before searching GEO with the corrected query...")
        time.sleep(5)
        user_query = cleaned_query

    # Build Entrez query via LLM
    entrez_query = build_entrez_query_with_llm(client, user_query, logger)

    # 2) ESearch
    esearch_data = esearch_gds(entrez_query, retmax=retmax, logger=logger)
    es_result = esearch_data.get("esearchresult", {})
    idlist = es_result.get("idlist", [])
    webenv = es_result.get("webenv")
    query_key = es_result.get("querykey")

    if not idlist:
        logger.warning("No GEO records found for this query.")
        print("No GEO datasets found for this query. Try different disease keywords or increase --retmax.")
        logger.info("=== GEO single-cell search finished (no results) ===")
        return []

    total_found = int(es_result.get("count", len(idlist)))
    logger.info("ESearch found %d records; retrieving details for %d (retmax).",
                total_found, len(idlist))

    # 3) ESummary in batches
    records_all = collect_all_gse_records(
        webenv,
        query_key,
        total=len(idlist),
        logger=logger,
        batch_size=100,
    )
    logger.info(
        "Retrieved %d GSE-like records from ESummary (before filtering).",
        len(records_all),
    )

    # 4) Filter to human HTS
    records_filtered = filter_gse_records(records_all)
    logger.info("After Homo sapiens + HTS filter: %d GSEs.", len(records_filtered))

    if not records_filtered:
        print("No human expression-by-HTS GEO datasets found for this query.")
        logger.info("=== GEO single-cell search finished (no human HTS results) ===")
        return

    # 5) Filter to single-cell keyword hits
    records_sc = filter_single_cell_records(records_filtered)
    logger.info("After single-cell keyword filter: %d GSEs.", len(records_sc))

    if not records_sc:
        print("Human RNA-seq datasets exist for this query, but no single-cell / 10x datasets were detected.")
        logger.info("=== GEO single-cell search finished (no single-cell candidates) ===")
        return

    # 6) Rank with LLM (optional)
    if not no_rerank and len(records_sc) > 1:
        records_ranked = rank_gse_records_with_llm(client, user_query, records_sc, logger)
    else:
        records_ranked = records_sc

    selected = records_ranked[:max_gses]
    logger.info("Top %d GEO single-cell candidates after ranking/filtering:", len(selected))
    for r in selected:
        logger.info("  %s | %s", r["accession"], r["title"])

    # SEARCH-ONLY MODE
    if search_only:
        logger.info("Search-only mode enabled: analyzing supplementary files but NOT downloading.")
        for r in selected:
            gse = r["accession"]
            logger.info("---- Search-only summary for %s ----", gse)
            analysis = analyze_suppl_files(gse, logger)
            logger.info(
                "GSE %s: is_10x_singlecell=%s (%s) | matrix=%d | barcodes=%d | features=%d | RAW.tar=%d | anno_files=%d | fastq_like=%d",
                gse,
                analysis["is_10x_singlecell"],
                analysis["is_10x_reason"],
                len(analysis["matrix_files"]),
                len(analysis["barcodes_files"]),
                len(analysis["features_files"]),
                len(analysis["raw_tar_files"]),
                len(analysis["anno_files"]),
                len(analysis["fastq_like"]),
            )
        logger.info("=== GEO single-cell search finished (search-only mode) ===")
        return

    # DOWNLOAD MODE
    logger.info("Download mode: starting download for up to %d GSEs.", len(selected))
    processed_summaries = []
    for r in selected:
        try:
            summary = download_gse_singlecell_bundle(r, outdir, logger)
            if summary:
                processed_summaries.append(summary)
        except Exception as e:
            logger.error("Error while processing %s: %s", r.get("accession"), e)
        time.sleep(0.5)

    if not processed_summaries:
        print("No single-cell datasets could be downloaded for this query (after file inspection).")
        logger.info("=== GEO single-cell search finished (nothing downloaded) ===")
        return

    write_global_reports(processed_summaries, outdir, logger)
    logger.info("=== GEO single-cell search + download finished ===")

    process_outdir(outdir)
    logger.info("=== 10x validation finished ===")
    return processed_summaries

