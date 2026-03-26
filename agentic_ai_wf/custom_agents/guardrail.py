from agents import Agent, input_guardrail, Runner, GuardrailFunctionOutput, RunContextWrapper
from ..models.drugs_models import GuardrailOutput

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