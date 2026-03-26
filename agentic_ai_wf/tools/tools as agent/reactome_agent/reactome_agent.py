import asyncio
import os
from config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
)
from agents import (
    Agent,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    InputGuardrailTripwireTriggered
)
from reactome_tool import extract_reactome
from reactome_models import GuardrailOutput

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables.")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


# Define a guardrail agent .........................
guardrail_agent = Agent(
    name="Reactome Format Checker",
    instructions=(
        "Return is_triggered: true if the input is NOT a valid Reactome pathway ID "
        "(must start with 'R-HSA-' and followed by digits)."
        "Return is_triggered: false if input looks valid."
    ),
    output_type=GuardrailOutput
)

# Guardrail function
@input_guardrail
async def validate_pathway_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(starting_agent=guardrail_agent, input=input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_triggered,
    )

# Reactome agent using tool + guardrail ...............................
reactome_agent = Agent(
    name="Reactome Drug Extractor",
     instructions=(
        "You are an expert biomedical assistant designed to extract drug information from the Reactome pathway database.\n\n"
        
        "Your primary goal is to identify and return a list of drugs that are associated with a given Reactome pathway ID.\n"
        "These pathway IDs follow the format 'R-HSA-XXXXXX'. You must validate the input format using guardrails and reject invalid queries.\n\n"
        
        "Once a valid pathway ID is received.\n"
        "Use the tool to recursively explore subpathways, nested entity sets, and retrieve physical entities of type Drug, ProteinDrug, ChemicalDrug, or RNADrug.\n"
        
        "Ensure that the final output is a clean, deduplicated list of drug names relevant to the input pathway.\n\n"
        
        "You must:\n"
        "- Validate the input format using guardrails.\n"
        "- Use only the provided tool to obtain results — do not guess or rely on internal knowledge.\n"
        "- Return informative, concise, and structured output.\n"
        "- If no drugs are found, state this clearly.\n"
        
        "You are part of a broader pipeline and your response may be used downstream, so accuracy and consistency are critical."
    ),
    tools=[extract_reactome],
    model=DEFAULT_MODEL,
    input_guardrails=[validate_pathway_guardrail]
)

# Entry point
async def main():
    from logger import get_logger
    logger = get_logger("reactome.agent")
    print("🔬 Reactome Drug Finder Agent 🔬")
    while True:
        user_input = input("Enter Reactome Pathway ID (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting.")
            break
        try:
            result = await Runner.run(
                starting_agent=reactome_agent,
                input=user_input
            )
            output = result.final_output
            print("✅ Drugs Found:")
            if isinstance(output, dict) and "drugs" in output:
                for drug in output["drugs"]:
                    print(f" - {drug}")
            else:
                print("Response:", output)
        except InputGuardrailTripwireTriggered:
         logger.warning("Input blocked by guardrail: not a valid drug-pathway query.")
if __name__ == "__main__":
    asyncio.run(main())
