import sys
import logging


def setup_logging(log_file: str) -> logging.Logger:
    """
    Set up a logger that logs to both stdout and a file.
    """
    logger = logging.getLogger("geo_singlecell")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid adding handlers twice if setup_logging is called multiple times
    if logger.handlers:
        return logger

    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
