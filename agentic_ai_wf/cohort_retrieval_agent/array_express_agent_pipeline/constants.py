from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------- Pydantic Schema ----------------
class BioStudyMetadata(BaseModel):
    accession: str
    title: str
    organism: str
    study_type: str
    assay_by_molecule: str
    description: str
    files : Optional[List[str]] = field(default_factory=list)

class BioStudyKeys:
    TITLE = "Title"
    ORGANISM = "Organism"
    STUDY_TYPE = "Study type"
    DESCRIPTION = "Description"
    ASSAY_BY_MOLECULE = "assay by molecule"


class Defaults:
    NA = "N/A"
    META_FILEPATH = "./agentic_ai_wf/shared/cohort_data/arrayexpress"
    INVALID_FILEPATH = "./agentic_ai_wf/shared/cohort_data/invalid/arrayexpress"
    ONTOLOGY_FILEPATH = "./agentic_ai_wf/shared/cohort_data/arrayexpress/ontology"


class LLMFilterConstants:
    CHUNK_SIZE = 5



