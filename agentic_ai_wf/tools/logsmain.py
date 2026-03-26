import json
from datetime import datetime
import os

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def log_step_to_json(task_description, agent_output, tools_used=None, log_file='execution_log.json'):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_description": task_description,
        "agent_output": agent_output,
        "tools_used": tools_used if tools_used else [],
    }
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.error("Failed to write log to JSON: %s", e)