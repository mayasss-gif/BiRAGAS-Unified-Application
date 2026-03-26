# mdp_logging.py
from __future__ import annotations
import traceback

def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)

def info(msg: str) -> None:  _log("INFO", msg)
def warn(msg: str) -> None:  _log("WARN", msg)
def err(msg: str) -> None:   _log("ERROR", msg)
def debug(msg: str) -> None: _log("DEBUG", msg)

def trace(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()
