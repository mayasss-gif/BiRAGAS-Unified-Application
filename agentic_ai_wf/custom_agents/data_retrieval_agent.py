import logging
from agents import Agent, handoff, WebSearchTool
from ..tools.kegg_tool import extract_kegg
from ..tools.openfda_tool import extract_openfda
from ..tools.pubmed_tool import pubmed_search_drug
from ..processing.enrichment_tool import enriched_drugs
from ..processing.prioritization_tool import prioritize_drugs
from ..config.config import DEFAULT_MODEL
from ..models.drugs_models import EnrichedDrugInput
from agents import Agent, input_guardrail, Runner, GuardrailFunctionOutput, RunContextWrapper
from .guardrail import intent_guardrail
from .drug_prioritizer_agent import drug_prioritizer_agent

WebSearchTool.__doc__ = "Web search tool that retrieves online sources. Use this when KEGG fails or returns no results. Mention the sources found in your final output."


logger = logging.getLogger()


# on_handoff
def on_handoff(ctx: RunContextWrapper[None], input_data: EnrichedDrugInput):
    # print("input_data: ", input_data)
    logger.info("Data extracted successfully.")
    logger.info("Handing off data to Drug Prioritizer Agent.")


agent = Agent(
    name="Drug Data Retrieval Agent",
    instructions=(
        "Your goal is to extract detailed drug-related information based on a biological pathway identifier provided by the user.\n\n"
        "Use the tools provided to gather accurate, up-to-date information from trusted biomedical databases or literature.\n\n "
        "unless asked, Do not rely on your internal knowledge — always use tools to obtain real data.\n\n"
        "if primary sources (kegg, openfda) does not returned data (empty or not-found), fallback to pubmed serch.\n\n"
        "if pubmed fails to retrieve data(empty), use websearch, if that fails as well use your own knowledge and format the data accordingly.\n\n"

        "You should:\n"
        "- Identify and retrieve drugs associated with the pathway.\n"
        "- Augment these drugs data with additional metadata (e.g., adverse reactions, administration routes) using the tools provided.\n"
        "- If primary sources return no results, try alternate tools (e.g., pubmed search , use querry like `'drug' and 'pathway_name'`.\n"
        "- If no data is retrieved from PubMed on the first attempt, retry up to 2 more times using alternative queries. Stop retrying immediately if any response contains non-empty data. \n"
        "- find the relevant drugs in the data extracted through pubmed_search and use openfda to retrieve other attribtues of those drugs.\n"
        "- if no data found for attributes in openfda use the pubmed search again for find the drug attributes, use querry like `drug_name` and 'route, adverse reactions`.\n"
        "- If no data is retrieved from PubMed as well for the attributes on the first attempt, retry up to 2 more times using alternative queries. Stop retrying immediately if any response contains non-empty data. \n"
        "- if still no attributes data is found for a drug, use your own knowledge to populate the attributes. \n"
        "Do not ask the user for any follow-up or questions.\n"

        "- Once you have enriched data, hand it off to the Drug Prioritizer Agent.\n\n"

    ),
    model=DEFAULT_MODEL,  # constat variable that we can put in config. file
    tools=[extract_kegg, extract_openfda, enriched_drugs,
           pubmed_search_drug, WebSearchTool()],
    input_guardrails=[intent_guardrail],
    handoffs=[
        handoff(
            agent=drug_prioritizer_agent,
            input_type=EnrichedDrugInput,
            on_handoff=on_handoff
        )
    ]
)
