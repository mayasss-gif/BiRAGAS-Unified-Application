######################## Imports ################
#Standard Library
import os
import asyncio
import xml.etree.ElementTree as ET

import json
from datetime import datetime

#Third-Party Libraries
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

#Internal / SDK Imports
from agents import (
    Agent,
    Runner,
    function_tool,
    input_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
    handoff,
    WebSearchTool,
    HandoffInputData,
)

from agents.extensions import handoff_filters
################# key load ################

############### Guardrail function #################
# input data validation - input guardrail.
intent_agent = Agent(
    name='irrelevant input check',
    instructions=(
        "You're a filter. Determine if the user's input is a valid request to extract drug data "
        "based on a biological pathway. "
        "Return is_triggered: true if the query is *not* related to drug extraction through pathway."
        "Return is_triggered: false if it clearly asks for extracting drug info using a pathway."
        "Return is_triggered: false if it asks about hi, hello thanks"
        "Return is_triggered: false if it asks about the assistance you can provide, specifically related to drugs data"
        "Return is_triggered: false if it asks about the previous responses."
    ),
    output_type=GuardrailOutput
)

#input guardrail
@input_guardrail
async def intent_guardrail(
    ctx: RunContextWrapper[None], 
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:
    response = await Runner.run(starting_agent = intent_agent, input=input)

    return GuardrailFunctionOutput(
        output_info = response.final_output,
        tripwire_triggered = response.final_output.is_triggered,
    )
############### tools ###################
#tool function
@function_tool
def extract_kegg(pathway: str) -> RichDrugInput:
    
    try:
        cached = kegg_cache.get(pathway)
        if cached:
            logger.info(f"[CACHE HIT] KEGG for pathway: {pathway}")
            return cached

        drug_info =  find_drug_info(pathway) #helper function 
        
############### starting agent 

agent = Agent(
    name = "Reactome Assistant",
    instructions = ( "according to agent "
    ),
    model = 'gpt 4o - rought', # constat variable that we can put in config. file
    tools = [extract_reactom],
    input_guardrails = [pathway_guardrail],

)
#################### main 
async def main(input_query, first_prompt):
    result = await Runner.run(
            starting_agent=agent,
            input=messages
        )
#xcept InputGuardrailTripwireTriggered:
#        logger.warning("Input blocked by guardrail: not a valid drug-pathway query.")


################# prompt - input querry
prompt = ""
while True:
    input_query = input("Prompt: ")
    if input_query.lower().strip() == "exit":
        logger.info("👋 Exiting the agent.")
        break
        asyncio.run(main(input_query))


