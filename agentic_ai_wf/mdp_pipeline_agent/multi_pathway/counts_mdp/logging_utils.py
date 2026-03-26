# mdp_pipeline/logging_utils.py
from __future__ import annotations
import logging
import traceback

LOGGER = logging.getLogger("mdp_pipeline")

def setup_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)

def log(msg: str) -> None:
    LOGGER.info(msg)

def warn(msg: str) -> None:
    LOGGER.warning(msg)

def err(msg: str) -> None:
    LOGGER.error(msg)

def trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
