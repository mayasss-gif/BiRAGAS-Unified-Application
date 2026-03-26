import logging
import os
from datetime import datetime

def setup_logger(agent_name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Setup a logger for an workflow instance.
    param: agent_name: str - The name of the agent.
    param: log_dir: str - The directory to save the log file.
    param: level: logging.Level - The level of the logger.
    returns: logging.Logger - The logger.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{agent_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(agent_name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs if root logger is also logging

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger("Gene Prioritization Logger")
