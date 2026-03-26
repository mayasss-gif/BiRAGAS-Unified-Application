import asyncio
from .orchestrator_agent import orchestrator_agent

if __name__ == "__main__":
    user_query = "Perform gene prioritization for pancreatic cancer using my uploaded data."
    asyncio.run(orchestrator_agent(user_query))
