from agents import Agent
from ..processing.prioritization_tool import prioritize_drugs
from ..tools.patient_condition_drug_prior_main import patient_condition_drug_prior
from ..tools.deg_filter_main import deg_filter
from ..config.config import DEFAULT_MODEL

#drug prioritizer agent
drug_prioritizer_agent = Agent(
    name="Drug Prioritizer Agent",
    instructions=(
    """
    You are a drug prioritization agent.

    You will receive drug data enriched with metadata from KEGG and OpenFDA.

    1. Your first task is to select suitable drugs using the tool: `patient_condition_drug_prior`.
       Only use this tool, and do not modify or reprocess the data manually.

      2. Your second task is to filter the data obtained from Task 1 using `deg_filter` with the patient ID. This filtering should be
       based on *targets*, and must NOT change the content of any columns (i.e., do not modify values). The tool should
       be used strictly for **row filtering only**, not column editing.

    3. Finally, pass the filtered data to the `prioritize_drugs` tool to sort it based on priority.
       Again, do not manually alter any part of the data during this step.

    Maintain strict schema consistency. Any tool use must preserve all columns and their original values unless the tool itself
    performs schema changes.
    """
    ),
    model=DEFAULT_MODEL,
    tools=[patient_condition_drug_prior, deg_filter, prioritize_drugs]
)