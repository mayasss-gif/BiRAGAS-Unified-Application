from pydantic import BaseModel
from typing import List, Optional

class GuardrailOutput(BaseModel):
    is_triggered: bool
    reasoning: str

class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str

class PubMedSearchResult(BaseModel):
    articles: List[PubMedArticle]

class RichDrug(BaseModel):
    drug_id: str
    name: str
    pathway_id: str
    pathway_name: str
    drug_class: str
    target: str
    efficacy: str
    brite: str
    approved: bool

class RichDrugInput(BaseModel):
    drugs: List[RichDrug] 

class RichDrug_openfda(BaseModel):
    name: str
    adv_reactions: str
    route: str

class RichDrugInput_openfda(BaseModel):
    drugs: List[RichDrug_openfda] 

class EnrichedDrug(BaseModel):
    drug_id: str
    name: str
    pathway_id: str
    pathway_name: str
    drug_class: str
    target: str
    efficacy: str
    brite: str
    approved: bool
    adv_reactions: Optional[str] = None
    route: Optional[str] = None 

class EnrichedDrugOutput(BaseModel):
    drugs: List[EnrichedDrug]

class EnrichedDrugInput(BaseModel):
    drugs: List[EnrichedDrug]

# for patient condition matching
class EnrichedDrugMatch(BaseModel):
    drug_id: str
    name: str
    pathway_id: str
    pathway_name: str
    drug_class: str
    target: str
    efficacy: str
    brite: str
    approved: bool
    adv_reactions: Optional[str] = None
    route: Optional[str] = None 
    matching_status: bool  
    LLM_Match_Reason:str 

class EnrichedDrugMatchOutput(BaseModel):
    drugs: List[EnrichedDrugMatch]

class EnrichedDrugMatchInput(BaseModel):
    drugs: List[EnrichedDrugMatch]
#.....................................................
# for deg mathcing
class EnrichedDrugDeg(BaseModel):
    drug_id: str
    name: str
    pathway_id: str
    pathway_name: str
    drug_class: str
    target: str
    efficacy: str
    brite: str
    approved: bool
    adv_reactions: Optional[str] = None
    route: Optional[str] = None 
    matching_status: bool
    LLM_Match_Reason:str
    deg_match_status:bool

class EnrichedDrugDegOutput(BaseModel):
    drugs: List[EnrichedDrugDeg]

class EnrichedDrugDegInput(BaseModel):
    drugs: List[EnrichedDrugDeg]
#.................................................................
#for prioritization 
class EnrichedDrugPriority(BaseModel):
    drug_id: str
    name: str
    pathway_id: str
    pathway_name: str
    drug_class: str
    target: str
    efficacy: str
    brite: str
    approved: bool
    adv_reactions: Optional[str] = None
    route: Optional[str] = None 
    matching_status: bool
    LLM_Match_Reason:str
    deg_match_status: bool
    priority_status: int

class EnrichedDrugPriorityOutput(BaseModel):
    drugs: List[EnrichedDrugPriority]

class Subtask(BaseModel):
    task_description: str

class TaskPlan(BaseModel):
    subtasks: List[Subtask]
