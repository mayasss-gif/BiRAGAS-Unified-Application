from datetime import datetime
from copy import deepcopy

BASE_LOG = {
    "timestamp": None,          
    "agent_id": "agent-123",
    "agent_name": "DataProcessorAgent",
    "workflow_id": "workflow-456",
    "workflow_name": "Data Ingestion Pipeline",
    "run_id": "run-789",
    "step": "fetch_data",
    "step_index": 2,
    "log_level": "INFO",
    "status": "running",
    "log_message": None,        
    "elapsed_time_ms": 0,
    "error": None,
}
def build_log(log_message: str, **overrides) -> dict:
    """
    Return a populated log dict.
    
    Args:
        log_message: The message you want to send.
        **overrides: Any other fields you’d like to change ad-hoc
                     (e.g. step='parse_csv', status='success', etc.).
    """
    log = deepcopy(BASE_LOG)        
    log["timestamp"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    log["log_message"] = log_message
    log.update(overrides)           
    return log
