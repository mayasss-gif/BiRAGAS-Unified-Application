"""
Shared Neo4j Connection Pool for utils modules

Provides a singleton connection pool to avoid connection churn across
all Neo4j utility functions (genecards_scorer, disease_pathways, etc.)

IMPORTANT: This is a per-process singleton. With multiple workers (Gunicorn/Uvicorn),
each worker process will have its own connection pool instance. This is expected
and correct behavior for production deployments.
"""

import threading
import logging
import os
from agentic_ai_wf.neo4j_integration.connection import Neo4jConnection

logger = logging.getLogger(__name__)

# Global connection pool (singleton per process)
_connection_pool = None
_connection_lock = threading.Lock()
_connection_created = False
_process_id = os.getpid()

def get_shared_connection():
    """
    Get or create a shared Neo4j connection (singleton pattern).
    
    This connection is reused across all requests within the same worker process
    to avoid connection churn. Thread-safe and lazy-initialized.
    
    Note: With multiple workers, each worker will have its own connection instance.
    This is correct behavior - connections are not shared across processes.
    
    Returns:
        Neo4jConnection: Shared connection instance (same instance reused)
    """
    global _connection_pool, _connection_created
    
    # Double-check pattern for thread safety
    if _connection_pool is None:
        with _connection_lock:
            # Check again inside lock (another thread might have created it)
            if _connection_pool is None:
                _connection_pool = Neo4jConnection(lazy=True)
                _connection_created = True
                conn_id = id(_connection_pool)
                logger.debug(f"Created shared Neo4j connection pool (process {_process_id}, connection_id={conn_id})")
            else:
                # Another thread created it while we waited
                conn_id = id(_connection_pool)
                logger.debug(f"Reusing existing Neo4j connection pool (process {_process_id}, connection_id={conn_id})")
    else:
        # Connection exists - verify it's still valid
        conn_id = id(_connection_pool)
        if not _connection_pool._connected or not _connection_pool.driver:
            logger.warning(f"Connection pool invalid, recreating (process {_process_id}, connection_id={conn_id})")
            with _connection_lock:
                if _connection_pool is None or not _connection_pool._connected:
                    _connection_pool = Neo4jConnection(lazy=True)
                    _connection_created = True
                    conn_id = id(_connection_pool)
                    logger.info(f"Recreated Neo4j connection pool (process {_process_id}, connection_id={conn_id})")
    
    return _connection_pool
