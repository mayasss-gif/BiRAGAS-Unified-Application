from agents import Agent
from ..models.drugs_models import TaskPlan
from ..config.config import DEFAULT_MODEL

planner_agent = Agent(
    name="PlannerAgent",
    instructions=(
        "You're a planning agent. A user may give you a request involving multiple biological pathways "
        "or drug-related subgoals. Your job is to break the query into a list of subtasks. "
        "Each subtask should be a clear, self-contained sentence describing what needs to be done, "
        "like: 'Extract drug data for pathway hsa05206' or 'Compare drug results for hsa05206 and hsa05210'. "
        "Return your output as a list of subtasks."
    ),
    model=DEFAULT_MODEL,
    output_type=TaskPlan
)