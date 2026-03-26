import logging
from agents import function_tool

import httpx
import xml.etree.ElementTree as ET
import logging
from ..models.drugs_models import PubMedSearchResult, PubMedArticle

logger = logging.getLogger()


@function_tool
async def pubmed_search_drug(query: str) -> PubMedSearchResult:
    print("------------------------------------------")
    print(" --- Pubmed search tool for drug....")
    print("query: ", query)

    """
    Search PubMed for scientific literature relevant to a given biological or biomedical query.

    Input:
        query (str):
            A keyword-based search string related to drug discovery, pathways, targets, or biomedical topics.
            Examples:
                - "drugs associated with hsa05206"
                - "drug treatment pathways for colorectal cancer"

    Behavior:
        - Performs a two-step PubMed API search:
            1. `esearch` to obtain up to 5 matching PMIDs.
            2. `efetch` to retrieve metadata (title and abstract) for those PMIDs.
        - Results are parsed from XML into a structured list of articles.

    Output:
        PubMedSearchResult:
            An object containing a list of up to 5 articles. Each article includes:
            - pmid (str): PubMed ID
            - title (str): Article title
            - abstract (str): Concatenated abstract text (may be empty if not available)

    Notes:
        - If no results are found, the function returns an empty list.
        - This tool is typically used as a fallback when KEGG drug data is unavailable.
        - The agent is expected to summarize or extract relevant drug/pathway insights from the abstracts.
        - No local caching is performed; all requests are made in real-time using NCBI E-utilities.

    Exceptions:
        - Raises RuntimeError if the PubMed API fails or response parsing encounters an error.
        - This signals the agent to fall back to `WebSearchTool`.

    Example:
        input: "EGFR pathway inhibitors"
        output: PubMedSearchResult(articles=[PubMedArticle(pmid='12345678', title='...', abstract='...'), ...])
    """
    try:
        logger.info("Performing PubMed search...")

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        async with httpx.AsyncClient() as client:
            # Step 1: Use esearch to get list of PMIDs for the query
            esearch_url = f"{base_url}esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 3  # limit to top 5 articles
            }
            esearch_resp = await client.get(esearch_url, params=params)
            esearch_resp.raise_for_status()
            id_list = esearch_resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.info("No PubMed results found — returning empty list.")
                return PubMedSearchResult(articles=[])

            # Step 2: Use efetch to get article details for these PMIDs
            efetch_url = f"{base_url}efetch.fcgi"
            efetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }
            efetch_resp = await client.get(efetch_url, params=efetch_params)
            efetch_resp.raise_for_status()
            xml_data = efetch_resp.text

            # Parse XML to extract titles and abstract
            root = ET.fromstring(xml_data)

            articles = []
            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle")
                abstract_texts = article.findall(".//AbstractText")
                abstract = " ".join([a.text or "" for a in abstract_texts])

                articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract))
            logger.info("Returning PubMed results.")
            return PubMedSearchResult(articles=articles)

    except Exception as exp:
        logger.error("Exception during PubMed search", exc_info=True)
        #print("falling back to websearch")
        raise RuntimeError("no relevant data on pubmed — use web search tool as fallback.")

