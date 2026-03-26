from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from Bio import Entrez

from ..helpers import logger 



class PubMedArticle(BaseModel):
    pmid: str = Field(description="The PubMed ID of the article (e.g. '34567890')")
    title: str = Field(description="The title of the article (e.g. 'Association of TP53 with Cervical Cancer')")
    abstract: str = Field(description="The abstract of the article (e.g. 'This study investigates the association between TP53 and cervical cancer.')")
    url: str = Field(description="The URL of the article (e.g. 'https://www.ncbi.nlm.nih.gov/pubmed/34567890')")
    dated: date = Field(description="The date of the article (e.g. '2023-01-01')")
    journal: str = Field(description="The journal of the article (e.g. 'Breast Cancer Research', 'Cancer Research', 'Journal of Clinical Oncology')")
    article_types: list[str] = Field(description="The types of the article (e.g. 'Clinical Trial', 'Randomized Controlled Trial', 'Meta-Analysis', 'Review')")
  



Entrez.email = "f420testing@ayassbioscience.com"  # Replace with your actual email


def pubmed_search(query: str) -> Optional[list[PubMedArticle]]:
    """
    Searches PubMed for the given query and returns the most relevant article.

    Args:
        query (str): A free-text query 
        (e.g."TP53 association in cervical cancer with signaling pathway.", 
        “Explores TP53 gene association in the cell signaling pathway within the context of cervical cancer.”,
        “Analyzes BRCA1 involvement in the homologous recombination pathway in the setting of ovarian cancer.”
        ).

    Returns:
        Optional[Dict]: A dictionary with PMID, title, abstract, and PubMed URL, or None if not found.
    """
    # print(f"Searching PubMed for: {query}")
    try:
        # Step 1: Search PubMed for relevant article IDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=3,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            # print(f"No articles found for the query")
            return []
        
        articles = []
        for id in id_list:
        
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=id,
                rettype="medline",
                retmode="xml"
            )
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()

            article = fetch_results["PubmedArticle"][0]
            citation = article["MedlineCitation"]
            article_info = citation["Article"]

            pmid = citation["PMID"]
            title = article_info["ArticleTitle"]
            abstract_parts = article_info.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."
            journal = article_info["Journal"].get("Title","Unknown")
            pub_types = article_info.get("PublicationTypeList", [])
            article_types = [str(pt) for pt in pub_types]


            raw_dated = citation["DateCompleted"] if citation["DateCompleted"] else citation["DateRevised"]
            dated = date(int(raw_dated['Year']), int(raw_dated['Month']), int(raw_dated['Day']))
            
        
            articles.append(PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                dated=dated,
                journal=journal,
                article_types=article_types,
                # similarity_score=round(float(final_score), 4)
            ))
        
        articles.sort(key=lambda x: x.dated, reverse=True)
        # print(f"Found {len(articles)} articles")
        return articles
        

    except Exception as e:
        logger.error(f"Error fetching article for query: {query} {str(e)}")
        return []