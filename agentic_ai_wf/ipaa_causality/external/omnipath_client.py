"""
Robust OmniPath HTTP client: retry, timeout, connection pooling.
Replaces raw requests.get in mdp_engine/omnipath_layers.
"""
from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_omnipath_session(
    connect: int = 30,
    read: int = 180,
    retries: int = 3,
    backoff: float = 2.0,
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.timeout = (connect, read)
    return s


_OMNIPATH_SESSION: Optional[requests.Session] = None


def get_omnipath_session() -> requests.Session:
    global _OMNIPATH_SESSION
    if _OMNIPATH_SESSION is None:
        timeout = int(os.getenv("IPAA_OMNIPATH_TIMEOUT", "180"))
        _OMNIPATH_SESSION = build_omnipath_session(read=timeout)
    return _OMNIPATH_SESSION
