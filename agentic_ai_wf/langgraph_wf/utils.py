"""Utility functions for file detection and disease name normalization."""

import re
from pathlib import Path
from typing import Optional


def normalize_disease_name(name: Optional[str]) -> str:
    """Sanitize a name for filesystem usage."""
    if name is None:
        return "unnamed"
    sanitized = name.lower().replace(" ", "_")
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "unnamed"
    return sanitized


def find_counts_file(transcriptome_dir: Path, file_type: str = "counts") -> Optional[Path]:
    """Robustly find counts/expression or metadata files with flexible naming."""
    if not transcriptome_dir.exists():
        return None

    if file_type == "counts":
        naming_patterns = [
            "*_counts_data.*", "*counts_data.*",
            "*_counts.*", "*counts.*", "*_count.*", "*count.*",
            "*_raw_counts.*", "*raw_counts.*", "*_raw_count.*",
            "*raw_count.*", "*_raw.*", "*raw.*",
        ]
        exclude_keywords = ["metadata", "meta", "sample", "barcode", "feature"]
    elif file_type == "metadata":
        naming_patterns = ["*_metadata.*", "*metadata.*", "*_meta.*", "*meta.*"]
        exclude_keywords = []
    else:
        return None

    extensions = [".csv", ".tsv", ".xlsx", ".xls"]
    found_files = []

    for pattern_base in naming_patterns:
        for ext in extensions:
            pattern = pattern_base.replace(".*", ext)
            matches = list(transcriptome_dir.glob(pattern))
            for match in matches:
                if match.is_file():
                    if file_type == "counts":
                        name_lower = match.name.lower()
                        if any(kw in name_lower for kw in exclude_keywords):
                            continue
                    priority = naming_patterns.index(pattern_base) * 10 + extensions.index(ext)
                    found_files.append((priority, match))

    if not found_files and file_type == "counts":
        all_files = list(transcriptome_dir.glob("*"))
        for ext in extensions:
            for file_path in all_files:
                if file_path.is_file() and file_path.suffix.lower() == ext:
                    name_lower = file_path.name.lower()
                    if any(kw in name_lower for kw in exclude_keywords):
                        continue
                    found_files.append((999, file_path))
                    break

    if not found_files:
        return None
    found_files.sort(key=lambda x: x[0])
    return found_files[0][1]


def find_metadata_file(transcriptome_dir: Path) -> Optional[Path]:
    """Robustly find metadata files."""
    return find_counts_file(transcriptome_dir, file_type="metadata")
