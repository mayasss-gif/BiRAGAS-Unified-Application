from pydantic import BaseModel
from typing import List

class ReactomeInput(BaseModel):
    pathway_id: str

class ReactomeOutput(BaseModel):
    drugs: List[str]

class GuardrailOutput(BaseModel):
    is_triggered: bool
