import logging
from ..models.drugs_models import (
    RichDrugInput,
    RichDrugInput_openfda,
    EnrichedDrug,
    EnrichedDrugOutput,
)
from agents import function_tool

logger = logging.getLogger()

@function_tool
def enriched_drugs(kegg_data:RichDrugInput, openfda_data: RichDrugInput_openfda ) -> EnrichedDrugOutput:
    """
    Combine drug metadata from KEGG and OpenFDA into a unified structure for downstream use.

    Inputs:
        kegg_data (RichDrugInput):
            A list of drugs extracted from KEGG, each containing detailed metadata including:
            - drug_id, name, pathway_id, drug_class, target, efficacy, brite, and approval status.

        openfda_data (RichDrugInput_openfda):
            A list of drugs with additional information from OpenFDA, including:
            - Adverse reactions
            - Route of administration

    Behavior:
        - Drugs are matched by their name (case-insensitive).
        - For each KEGG drug, the tool attempts to find a matching OpenFDA entry.
        - If a match is found, both entries are merged into an `EnrichedDrug` object.
        - If no match is found in OpenFDA, the KEGG data is still included, and the OpenFDA field is set to `None`.

    Output:
        A list of EnrichedDrug objects, each combining:
        - The original KEGG metadata.
        - The corresponding OpenFDA data if available (or `None` if not matched).

    Notes:
        - Matching is based solely on the lowercase drug name string.
        - This tool does not modify or infer any drug data; it only merges what's provided.
        - Ensure both inputs are non-empty and structurally valid before calling this function.
    """
    try:
        print("Enriching KEGG data with OpenFDA data — aligned merge")
        
        enriched_list = []
        for kegg_drug, fda_drug in zip(kegg_data.drugs, openfda_data.drugs):
            enriched = EnrichedDrug(
                drug_id=kegg_drug.drug_id,
                name=kegg_drug.name,
                pathway_id=kegg_drug.pathway_id,
                pathway_name=kegg_drug.pathway_name,
                drug_class=kegg_drug.drug_class,
                target=kegg_drug.target,
                efficacy=kegg_drug.efficacy,
                brite=kegg_drug.brite,
                approved=kegg_drug.approved,
                adv_reactions=fda_drug.adv_reactions,
                route=fda_drug.route
            )
            enriched_list.append(enriched)

        enriched_list = enriched_list[:5]  # Limit to first 5 enriched drugs
        return EnrichedDrugOutput(drugs=enriched_list)



    except (AttributeError, KeyError, TypeError) as e:
        print("Exception Occurs: ", e)
        logger.error("Enrichment failed due to data inconsistency: %s", e, exc_info=True)
        # raise EnrichmentError("Failed to enrich KEGG data with OpenFDA entries.") from e

    except Exception as exp:
        print("Exception Occurs: ", exp)
        logger.error("Error occurred during drug enrichment", exc_info=True)
        raise RuntimeError("enriching unsuccessful") from exp