import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: str) -> str:
    """Create directory if missing, return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(base_dir: str) -> None:
    """Basic file + console logger."""
    ensure_dir(base_dir)
    log_path = os.path.join(base_dir, "multiomics.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def read_feature_table(path: Optional[str], layer_name: str) -> Optional[pd.DataFrame]:
    """Read a feature × sample matrix from CSV.

    Expects:
        - A column named 'feature' (used as index).
        - Remaining columns = sample IDs.

    Returns None if path is None.
    """
    if path is None:
        logging.info(f"[{layer_name}] No path provided, skipping.")
        return None

    if not os.path.exists(path):
        logging.warning(f"[{layer_name}] File not found: {path}. Skipping layer.")
        return None

    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"[{layer_name}] Expected a 'feature' column in {path}")

    df = df.set_index("feature")
    logging.info(f"[{layer_name}] Loaded matrix with shape {df.shape} from {path}")
    return df


def read_metadata(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read metadata table with samples as rows."""
    if path is None:
        logging.info("No metadata path provided, skipping metadata.")
        return None

    if not os.path.exists(path):
        logging.warning(f"Metadata file not found: {path}. Skipping metadata.")
        return None

    df = pd.read_csv(path)
    logging.info(f"Loaded metadata with shape {df.shape} from {path}")
    return df
