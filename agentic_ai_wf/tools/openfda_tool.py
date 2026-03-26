import logging
from agents import function_tool
#from cache.cache_layer import TTLCache
from .openfda_api_tool import find_drug_info_openfda
from ..models.drugs_models import RichDrugInput, RichDrug_openfda, RichDrugInput_openfda

from ..cache.json_cache import JsonTTLCache
from ..cache.cache_manager import openfda_cache

logger = logging.getLogger()
# openfda_cache = TTLCache()

#tool function
@function_tool
def extract_openfda(drugs_kegg: RichDrugInput) -> RichDrugInput_openfda:
    """
    Retrieve additional metadata for a given drug using the OpenFDA API.

    Input:
        drug (str):
            The name of the drug for which additional information is to be retrieved.
            This drug name is typically obtained from KEGG via the `extract_kegg` tool.

    Behavior:
        - Queries the OpenFDA drug label endpoint for adverse reactions and routes of administration.
        - If the drug is found (status = 1):
            - Extracts and returns structured information, including:
                - Adverse Reactions
                - Route of Administration
        - If the drug is not found or partial data is missing (status ≠ 1):
            - Returns fallback values:
                - 'not found in openfda' for both fields.
            - The tool still returns one entry per drug, preserving alignment with KEGG output.

    Output:
        RichDrugInput_openfda:
            A container object containing a list of `RichDrug_openfda` items with the following fields:
            - name: Name of the queried drug
            - adv_reactions: Description of adverse reactions (if found)
            - route: Route of administration (if found)

    Notes:
        - Matching in OpenFDA is based on the plain-text drug name.
        - The tool is tolerant to missing or incomplete OpenFDA data.
        - It is expected to be used in conjunction with `extract_kegg` and `enriched_drugs`.

    Exceptions:
        - Raises RuntimeError if OpenFDA fails or unexpected errors occur.
        - This is intended to signal fallback to literature or web search tools.
    """
    
    try:
        key = "|".join(sorted(drug.name for drug in drugs_kegg.drugs))
        cached = openfda_cache.get(key)
        if cached:
            logger.info(f"[CACHE HIT] OpenFDA for {key}")
            return RichDrugInput_openfda(**cached)


        drugs_names = [drug.name for drug in drugs_kegg.drugs]
        drug_info_openfda, status=  find_drug_info_openfda(drugs_names) 
        logger.info("OpenFDA query response: %s", drug_info_openfda)
        
        if status == 1:
            print("- openFDA Data Extraction Completed.")
            drugs = [
                RichDrug_openfda(
                    name=drug_info_openfda["Drug Queried"][i],
                    adv_reactions = drug_info_openfda["Adverse Reactions"][i],
                    route = drug_info_openfda["Route of Administration"][i]
                )
                for i in range(len(drug_info_openfda))
            ]
        else: 
            print(" - opendFDA Data Extraction Failed.")
            drugs = [
            RichDrug_openfda(
                name=drug,
                adv_reactions = 'not found in openfda',
                route = 'not found in openfda'
            )
            for i in range(len(drug_info_openfda))
             ]
        
        result = RichDrugInput_openfda(drugs=drugs)     
        openfda_cache.set(key, result.model_dump())
        return result #RichDrugInput_openfda(drugs=drugs)

    except Exception as exp:
        logger.error("OpenFDA query failed", exc_info=True)
        print("Exception Occurs(openFDA) as: ", exp)
        raise RuntimeError("not data found in openfda — use web search tool as fallback.") from exp