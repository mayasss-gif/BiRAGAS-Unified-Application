# src/logging_utils.py
import logging
import os
from pathlib import Path

def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    """
    Creates a timestamped log file inside:
        RUN_OUTPUT_DIR/logs/DepMap_<RUN_STAMP>.log

    RUN_STAMP is shared by all steps in this run (from paths.RUN_STAMP).

    With mode='a', all scripts in the same RUN_STAMP append to a single log file.

    NEW:
    ----
    If the environment variable DEPMAP_COMPACT_LOG is set (to anything),
    we reduce log verbosity to WARNING and above. This keeps the log file
    much smaller, while still recording all important warnings/errors.

        Normal run (default):
            log level = INFO  (same behaviour as before)

        Compact run:
            DEPMAP_COMPACT_LOG=1 python run_full_pipeline.py
            -> log level = WARNING
    """

    
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}_logs.log"

# Decide log level based on environment
    compact_flag = os.getenv("DEPMAP_COMPACT_LOG", "").strip()
    if compact_flag:
        level = logging.WARNING
    else:
        level = logging.INFO  # original behaviour

# Clean old handlers (avoid duplicate logging when re-running in the same session)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

# Configure ROOT logger with file + console handlers
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

# Our project logger
    logger = logging.getLogger(name)
    
# Let it propagate to root so handlers are used
    logger.setLevel(level)
    logger.propagate = True

# Quiet down some very chatty third-party loggers a bit
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(level)} (DEPMAP_COMPACT_LOG={compact_flag})")
    return logger
