from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


class TeeLogger:
    """Mirror stdout/stderr to both terminal and a log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:  # pragma: no cover - passthrough
        self.terminal.flush()
        self.log.flush()


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Configure structured logging to both console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

