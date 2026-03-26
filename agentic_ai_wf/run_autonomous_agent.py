import asyncio
from .meta_agent import run_meta_agent

async def demo():
    result = await run_meta_agent(
        user_query="Perform transcriptome analysis for pancreatic cancer on only provided data, no cohort.",
        disease_name="Pancreatic Cancer",
        patient_id="40fd7dbb-14d7-4459-9d3f-c503076c32ea",
        patient_name="Test Patient",
        analysis_transcriptome_dir="media/uploads/analysis/c8fc3428-ba9f-4a46-8eb2-95fb984a3750",
        user_id="9b5bb964-b1c7-4f02-972a-48c7c10a4810",
        max_attempts=3,
        analysis_id="c8fc3428-ba9f-4a46-8eb2-95fb984a3750"
    )
    print(result["success"], result.get("reports"))
asyncio.run(demo())
