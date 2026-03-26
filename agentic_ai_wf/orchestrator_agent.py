import asyncio
import json
from uuid import uuid4
from typing import Optional, Dict, Any
from decouple import config
from pydantic import BaseModel, Field, field_validator
from agents import Agent, ModelSettings, Runner, RunConfig, TResponseInputItem
from openai.types.shared.reasoning import Reasoning
from .meta_agent import run_meta_agent

# =======================================================
#  CONFIGURATION MODEL
# =======================================================

class WorkflowConfig(BaseModel):
    """Structured configuration for transcriptome analysis workflow."""
    user_query: str = Field(..., description="Natural language query describing the analysis request")
    disease_name: str = Field(..., description="Name of the disease being analyzed", min_length=1)
    patient_id: str = Field(default="unknown_patient", description="Patient identifier (UUID or placeholder)")
    patient_name: str = Field(default="unknown_patient", description="Patient name or identifier")
    analysis_transcriptome_dir: str = Field(default="unknown_directory", description="Path to transcriptome data directory")
    user_id: str = Field(default="unknown_user", description="User identifier (UUID or placeholder)")
    max_attempts: int = Field(default=3, description="Maximum retry attempts for workflow", ge=1, le=10)
    analysis_id: str = Field(default_factory=lambda: str(uuid4()), description="Analysis identifier (UUID or placeholder)")

    @field_validator("disease_name")
    @classmethod
    def validate_disease(cls, v: str):
        if not v.strip():
            raise ValueError("disease_name cannot be empty")
        return v.strip().title()

    @field_validator("max_attempts")
    @classmethod
    def validate_attempts(cls, v: int):
        if v < 1 or v > 10:
            raise ValueError("max_attempts must be between 1 and 10")
        return v

    def has_placeholders(self) -> bool:
        """Detect placeholder values."""
        placeholders = ["unknown", "placeholder"]
        return any(p in str(val).lower() for val in [
            self.patient_id, self.patient_name, self.analysis_transcriptome_dir, self.user_id, self.analysis_id
        ] for p in placeholders)

    def get_placeholder_fields(self) -> list[str]:
        placeholders = ["unknown", "placeholder"]
        return [
            name for name, val in self.model_dump().items()
            if any(p in str(val).lower() for p in placeholders)
        ]

# =======================================================
#  DEFINE THE AGENT (OpenAI Agents SDK)
# =======================================================

SYSTEM_INSTRUCTIONS = """
You are an orchestration AI agent responsible for preparing inputs for a transcriptome analysis workflow.

Your job:
- Interpret user requests (e.g., “Perform gene prioritization for pancreatic cancer using uploaded data”)
- Generate a JSON configuration object for calling `run_meta_agent`.
- Fill in missing fields intelligently, using placeholders (like "unknown_patient") if not provided.

### Required JSON fields:
{
  "user_query": "...",
  "disease_name": "...",
  "patient_id": "...",
  "patient_name": "...",
  "analysis_transcriptome_dir": "...",
  "user_id": "...",
  "max_attempts": 3,
  "analysis_id": "..."
}

### Rules:
1. Only include valid JSON (no markdown).
2. Guess missing values sensibly and mark them for user confirmation.
3. The goal is to provide a fully structured configuration object ready for `run_meta_agent`.
"""

orchestrator_agent = Agent(
    name="Transcriptome Orchestrator Agent",
    instructions=SYSTEM_INSTRUCTIONS,
    model="gpt-5",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium"),
        store=True
    )
)

# =======================================================
#  INPUT MODEL
# =======================================================

class OrchestratorInput(BaseModel):
    user_query: str
    user_context: Optional[Dict[str, Any]] = None


# =======================================================
#  RUNNER FUNCTION
# =======================================================

async def run_orchestrator(workflow_input: OrchestratorInput):
    """Autonomous orchestrator that builds config and runs meta-agent plan."""
    conversation_history: list[TResponseInputItem] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"User query: {workflow_input.user_query}"},
                {"type": "input_text", "text": f"Known context: {json.dumps(workflow_input.user_context or {}, indent=2)}"},
            ],
        }
    ]

    # Run agent reasoning
    result = await Runner.run(
        orchestrator_agent,
        input=[*conversation_history],
        run_config=RunConfig(
            trace_metadata={
                "__trace_source__": "agent-orchestrator",
                "workflow_id": f"wf_{uuid4().hex}",
            }
        ),
    )

    # Extract model output
    structured_output = result.final_output_as(str)
    print("\n🧠 Agent structured output (raw):\n", structured_output)

    # Parse config
    try:
        config_dict = json.loads(structured_output)
        config = WorkflowConfig(**config_dict)
    except Exception as e:
        print(f"❌ Failed to parse configuration: {e}")
        return None

    print("\n📋 Proposed run configuration:")
    for k, v in config.model_dump().items():
        print(f"   {k}: {v}")

    if config.has_placeholders():
        print(f"\n⚠️ Warning: Placeholder fields detected: {config.get_placeholder_fields()}")

    confirm = input("\n✅ Run with this configuration? (y/n): ").strip().lower()
    if confirm != "y":
        print("🚫 Execution cancelled by user.")
        return None

    print("\n🚀 Running autonomous workflow with run_meta_agent...\n")
    result = await run_meta_agent(**config.model_dump())

    print(f"\n✅ Success: {result['success']}")
    print(f"🧾 Reports: {result.get('reports')}")
    return result


# =======================================================
#  LOCAL ENTRYPOINT
# =======================================================

if __name__ == "__main__":
    async def main():
        user_query = "Perform gene prioritization for pancreatic cancer using uploaded data"
        orchestrator_input = OrchestratorInput(user_query=user_query)
        await run_orchestrator(orchestrator_input)

    asyncio.run(main())
