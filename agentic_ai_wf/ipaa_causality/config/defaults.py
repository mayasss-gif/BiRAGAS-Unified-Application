"""Default values and env overrides for IPAA config."""
from __future__ import annotations

import os


def from_env() -> dict:
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "omnipath_timeout": int(os.getenv("IPAA_OMNIPATH_TIMEOUT", "180")),
        "celery_task_time_limit": int(os.getenv("IPAA_TASK_TIME_LIMIT", "7200")),
    }
