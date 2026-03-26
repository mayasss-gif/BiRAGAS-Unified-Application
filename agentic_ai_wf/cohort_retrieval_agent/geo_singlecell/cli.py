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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Search GEO (db=gds) for human single-cell RNA-seq datasets by disease/description "
            "and (optionally) download 10x-style matrix/barcodes/features/annotation files, "
            "including content packed in *_RAW.tar, OR process explicit GSE IDs directly."
        )
    )
    parser.add_argument(
        "--query",
        "-q",
        required=False,
        help="Free-text disease/experiment description (e.g. 'EGFR-mutant lung adenocarcinoma TKI resistance').",
    )
    parser.add_argument(
        "--gse",
        nargs="+",
        required=False,
        help=(
            "One or more GEO Series accessions (e.g. GSE137029). "
            "If provided, the tool skips text search and processes only these GSE IDs."
        ),
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="geo_sc_data",
        help="Output directory to store downloaded GSE folders (default: geo_sc_data).",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=500,
        help="Maximum GEO records to retrieve from ESearch (default: 500).",
    )
    parser.add_argument(
        "--max-gses",
        type=int,
        default=10,
        help="Maximum number of GSEs to download or show after ranking/filtering (default: 10).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM-based ranking and keep GEO ESearch order.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only search/analyze GEO datasets and show single-cell candidates; do not download.",
    )
    parser.add_argument(
        "--log-file",
        default="geo_singlecell.log",
        help="Path to log file (default: geo_singlecell.log).",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_file)

    # You must provide either a query or at least one GSE ID
    if not args.query and not args.gse:
        parser.error("You must provide either --query/-q or --gse.")

    outdir = args.outdir
    retmax = args.retmax
    max_gses = args.max_gses
    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------------
    # 1) DIRECT GSE MODE
    # -------------------------------------------------------
    if args.gse:
        logger.info("Direct GSE mode: processing explicit GSE IDs: %s", ", ".join(args.gse))
        processed_summaries = []

        for gse_id in args.gse:
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

                if args.search_only:
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
        sys.exit(0)

    # -------------------------------------------------------
    # 2) NORMAL QUERY MODE (LLM + Entrez search)
    # -------------------------------------------------------
    user_query = args.query
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
        sys.exit(1)

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
        sys.exit(0)

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
        sys.exit(0)

    # 5) Filter to single-cell keyword hits
    records_sc = filter_single_cell_records(records_filtered)
    logger.info("After single-cell keyword filter: %d GSEs.", len(records_sc))

    if not records_sc:
        print("Human RNA-seq datasets exist for this query, but no single-cell / 10x datasets were detected.")
        logger.info("=== GEO single-cell search finished (no single-cell candidates) ===")
        sys.exit(0)

    # 6) Rank with LLM (optional)
    if not args.no_rerank and len(records_sc) > 1:
        records_ranked = rank_gse_records_with_llm(client, user_query, records_sc, logger)
    else:
        records_ranked = records_sc

    selected = records_ranked[:max_gses]
    logger.info("Top %d GEO single-cell candidates after ranking/filtering:", len(selected))
    for r in selected:
        logger.info("  %s | %s", r["accession"], r["title"])

    # SEARCH-ONLY MODE
    if args.search_only:
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
        sys.exit(0)

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
        sys.exit(0)

    write_global_reports(processed_summaries, outdir, logger)
    logger.info("=== GEO single-cell search + download finished ===")
if __name__ == "__main__":
    main()


# python -m geo_singlecell.cli --query "EGFR lung cancer" --retmax 2000 --max-gses 10 --outdir efgr_2 