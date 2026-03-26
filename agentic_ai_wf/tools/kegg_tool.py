import logging
#from cache.cache_layer import TTLCache
from ..models.drugs_models import RichDrug, RichDrugInput
from agents import function_tool  # or wherever this is defined
from .kegg_api_tool_enhanced import find_drug_info
from ..cache.json_cache import JsonTTLCache
from ..cache.cache_manager import kegg_cache
logger = logging.getLogger()
#kegg_cache = TTLCache()
#kegg_cache = JsonTTLCache("cache/kegg_cache.json")

#tool function
@function_tool
def extract_kegg(pathway: str) -> RichDrugInput:
    """
    Retrieve rich drug metadata associated with a biological pathway using the KEGG database.

    Input:
        pathway (str):
            A pathway identifier string. Examples include:
            - Human KEGG format: 'hsa05206' (e.g., 'Pathways in cancer')
            - Reactome/alternate formats if supported by the backend function.

    Behavior:
        - Queries the KEGG database for drugs associated with the specified pathway.
        - Uses a helper function `find_drug_info()` to extract drug data, including:
            - Drug ID, name, associated pathway ID and name, drug class, molecular target,
              efficacy, BRITE hierarchy, and approval status.
        - Structures the results into a list of `RichDrug` objects.

    Output:
        RichDrugInput:
            A container with a list of drugs and their metadata in the following fields:
            - drug_id: KEGG drug identifier (e.g., Dxxxx)
            - name: Drug name (generic or common name)
            - pathway_id: KEGG pathway ID
            - pathway_name: Descriptive name of the pathway
            - drug_class: Drug category or classification
            - target: Main molecular target(s) of the drug
            - efficacy: Brief description of drug efficacy
            - brite: KEGG BRITE hierarchy category
            - approved: Boolean indicating whether the drug is approved for use

    Notes:
        - This tool is typically the first step in the drug discovery pipeline.
        - If the pathway is invalid or no drugs are found, the function raises an exception
          to trigger fallback tools such as literature or web search.
        - It is expected that this function uses a live or cached KEGG API wrapper.

    Exceptions:
        - Raises RuntimeError if KEGG returns no results or the helper function fails.
        - This exception is used to signal fallback to alternate data sources (e.g., PubMed, WebSearch).

    Example:
        input: 'hsa05206'
        output: RichDrugInput(drugs=[RichDrug(...), RichDrug(...), ...])
    """
    print("kegg tool executing...")
    try:
        cached = kegg_cache.get(pathway)
        key = pathway
        if cached:
            logger.info(f"[CACHE HIT] KEGG for pathway: {pathway}")
            # return cached
            return dict(list(cached.items())[:5])
        
        print("no cache hit, hiting new req...")
        print("pathway: ", pathway)
        drug_info =  find_drug_info(pathway) 
        
        logger.info("KEGG query successful — shape: %s", drug_info.shape)
        logger.info("Drugs found in KEGG.")
        drugs = [
            RichDrug(
                drug_id=drug_info["drug_id"][i],
                name=drug_info["name"][i],
                pathway_id=drug_info["pathway_id"][i],
                pathway_name=drug_info["pathway_name"][i],
                drug_class=drug_info["drug_class"][i],
                target=drug_info["target"][i],
                efficacy=drug_info["efficacy"][i],
                brite=drug_info["brite"][i],
                approved=drug_info["approved"][i],
            )
            for i in range(len(drug_info["drug_id"]))
        ]
        drugs = drugs[:5]  # Limit to first 5 drugs for performance
        print("drugs kegg check...")
        #drugs = []
        if not drugs:
            raise RuntimeError(
                f"[KEGG_TOOL:NoDrugsFound] No drugs found for pathway '{pathway}'. "
                "Recommend switching to pubmed searc or web search fallback."
            )
        print("drugs are not empty...")
        output = RichDrugInput(drugs = drugs)
        #kegg_cache.set(pathway, output)
        kegg_cache.set(key, output.model_dump())
        return RichDrugInput(drugs=drugs)

    except Exception as exp:
        print("Exception occrus (kegg main tool): ", exp)
        #logger.warning("No data found in KEGG. Triggering literature fallback...")
        
        raise RuntimeError(
            f"[KEGG_TOOL:UnhandledError] KEGG tool failed on pathway '{pathway}'. "
            "Original exception: " + str(exp)
            ) from exp