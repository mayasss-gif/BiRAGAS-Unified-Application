# mdp_engine/logging_utils.py
from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> None:
    """
    Configure root logging safely (idempotent). If already configured, only sets level.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a named logger. Optionally set its level.
    """
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    return log
