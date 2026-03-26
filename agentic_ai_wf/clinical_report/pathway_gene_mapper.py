#!/usr/bin/env python3
"""Utilities for enriching grouped pathways with patient gene data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)

SignatureGroups = Dict[str, List[Dict[str, Any]]]

GeneDataInput = Union[str, Path, pd.DataFrame]


def load_patient_genes(patient_genes_file: GeneDataInput) -> pd.DataFrame:
    """Load patient gene data from a CSV file or return a copy of a DataFrame.

    Args:
        patient_genes_file: Path to a CSV file or an existing DataFrame.

    Returns:
        A DataFrame containing patient gene data.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        TypeError: If the input type is unsupported.
        ValueError: If the resulting DataFrame is empty.
    """

    if isinstance(patient_genes_file, pd.DataFrame):
        patient_genes_df = patient_genes_file.copy()
    elif isinstance(patient_genes_file, (str, Path)):
        path = Path(patient_genes_file)
        if not path.exists():
            raise FileNotFoundError(f"Patient genes file not found: {path}")
        patient_genes_df = pd.read_csv(path)
    else:
        raise TypeError(
            "patient_genes_file must be a path or pandas DataFrame, "
            f"got {type(patient_genes_file)}"
        )

    if patient_genes_df.empty:
        raise ValueError("Patient genes data is empty")

    return patient_genes_df


def map_genes_to_pathways(
    grouped_signatures: SignatureGroups,
    patient_genes_df: pd.DataFrame,
    patient_prefix: str = "patient",
) -> SignatureGroups:
    """Attach patient gene data to pathways grouped by main class.

    Args:
        grouped_signatures: Output of :func:`group_signatures`.
        patient_genes_df: DataFrame containing patient gene measurements.
        patient_prefix: Prefix for patient gene columns.
    Returns:
        A new grouped signatures dictionary with ``top_3_genes_data`` per
        pathway.

    Raises:
        ValueError: If a gene identifier column cannot be determined.
    """

    if not grouped_signatures:
        return {}

    if patient_genes_df.empty:
        logger.warning("Patient genes DataFrame is empty; skipping enrichment")
        return {
            main_class: [
                {**pathway, "top_3_genes_data": []}
                for pathway in pathways
            ]
            for main_class, pathways in grouped_signatures.items()
        }

    working_df = patient_genes_df.copy()
    gene_column = _detect_gene_column(working_df)

    working_df[gene_column] = (
        working_df[gene_column]
        .astype(str)
        .str.strip()
    )
    working_df["__gene_key__"] = working_df[gene_column].str.upper()

    log2fc_cols = _collect_log2fc_columns(working_df.columns, patient_prefix)
    if log2fc_cols:
        numeric_fc = working_df[log2fc_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        working_df["__log2fc_score__"] = (
            numeric_fc.mean(axis=1, skipna=True).fillna(0.0)
        )
    else:
        working_df["__log2fc_score__"] = 0.0

    index_columns = ["__gene_key__", "__log2fc_score__"]
    working_df = working_df.drop_duplicates(subset=["__gene_key__"], keep="first")
    gene_lookup = working_df.set_index("__gene_key__", drop=False)

    enriched_groups: SignatureGroups = {}

    for main_class, pathways in grouped_signatures.items():
        enriched_pathways: List[Dict[str, Any]] = []

        for pathway in pathways:
            pathway_copy = pathway.copy()
            associated = _extract_gene_keys(pathway)

            if not associated:
                pathway_copy["top_3_genes_data"] = []
                enriched_pathways.append(pathway_copy)
                continue

            subset = _subset_genes(gene_lookup, associated)

            if subset.empty:
                logger.debug(
                    "No patient gene data found for pathway '%s' (genes: %s)",
                    pathway.get("Pathway_Name", "Unknown"),
                    ", ".join(sorted(associated)),
                )
                pathway_copy["top_3_genes_data"] = []
                enriched_pathways.append(pathway_copy)
                continue

            regulation = str(
                pathway.get("Regulation") or pathway.get("regulation") or ""
            ).strip().lower()
            ascending = regulation == "down"

            sorted_subset = subset.sort_values(
                by="__log2fc_score__", ascending=ascending
            )
            top_records = (
                sorted_subset.head(3)
                .drop(columns=index_columns, errors="ignore")
                .to_dict(orient="records")
            )

            summary = _build_gene_summary(
                top_records,
                gene_column,
                log2fc_cols,
                patient_prefix,
            )

            pathway_copy["gene_display_column"] = gene_column
            pathway_copy["log2fc_columns"] = log2fc_cols
            pathway_copy["patient_prefix"] = patient_prefix
            pathway_copy["top_3_genes_data"] = top_records
            pathway_copy["top_3_genes_summary"] = summary
            pathway_copy["top_3_genes"] = summary

            missing = associated - set(subset.index)
            if missing:
                logger.debug(
                    "Missing patient gene entries for pathway '%s': %s",
                    pathway.get("Pathway_Name", "Unknown"),
                    ", ".join(sorted(missing)),
                )

            enriched_pathways.append(pathway_copy)

        enriched_groups[main_class] = enriched_pathways

    return enriched_groups


def _detect_gene_column(df: pd.DataFrame) -> str:
    """Identify the column that contains gene identifiers."""

    preferred = (
        "Gene",
        "gene",
        "Gene_Symbol",
        "gene_symbol",
        "symbol",
        "Symbol",
    )

    for column in preferred:
        if column in df.columns:
            return column

    for column in df.columns:
        lowered = column.lower()
        if (
            lowered == "gene"
            or lowered.startswith("gene_")
            or lowered.endswith("_gene")
            or lowered.endswith("gene_symbol")
        ):
            return column

    raise ValueError(
        "Unable to determine gene column in patient genes DataFrame"
    )


def _collect_log2fc_columns(
    columns: Iterable[str], patient_prefix: str
) -> List[str]:
    """Return columns that contain log2 fold change values."""

    prefix_lower = patient_prefix.lower()

    return [
        column
        for column in columns
        if column.lower().startswith(prefix_lower)
        and column.lower().endswith("_log2fc")
    ]


def _extract_gene_keys(pathway: Dict[str, Any]) -> Optional[set[str]]:
    """Extract gene identifiers from pathway metadata."""

    genes_field = pathway.get("Pathway_Associated_Genes") or pathway.get(
        "Input_Genes"
    )

    if not genes_field:
        return None

    if isinstance(genes_field, str):
        gene_keys = {
            gene.strip().upper()
            for gene in genes_field.split(",")
            if gene and gene.strip()
        }
    elif isinstance(genes_field, Iterable):
        gene_keys = {
            str(gene).strip().upper()
            for gene in genes_field
            if gene and str(gene).strip()
        }
    else:
        return None

    return gene_keys or None


def _subset_genes(
    gene_lookup: pd.DataFrame,
    gene_keys: set[str],
) -> pd.DataFrame:
    """Retrieve patient gene rows matching provided gene keys."""

    try:
        subset = gene_lookup.loc[list(gene_keys)]
    except KeyError:
        available = gene_lookup.index.intersection(list(gene_keys))
        subset = gene_lookup.loc[available]

    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T

    return subset


def _build_gene_summary(
    records: List[Dict[str, Any]],
    gene_column: str,
    log2fc_cols: List[str],
    patient_prefix: str,
) -> List[Dict[str, Any]]:
    """Create a condensed summary for top pathway genes."""

    summary: List[Dict[str, Any]] = []

    for record in records:
        gene_symbol = record.get(gene_column)
        if gene_symbol is None:
            gene_symbol = record.get(gene_column.lower(), "")
        gene_symbol = str(gene_symbol) if gene_symbol is not None else ""

        log2fc_values: List[Dict[str, Any]] = []
        primary_log2fc: Optional[float] = None

        for column in log2fc_cols:
            numeric_value = _to_float(record.get(column))
            if numeric_value is None:
                continue
            log2fc_values.append(
                {
                    "name": column,
                    "label": _format_metric_label(column, patient_prefix),
                    "value": numeric_value,
                }
            )
            if primary_log2fc is None:
                primary_log2fc = numeric_value

        metadata = _extract_metadata(record, gene_column, log2fc_cols)

        direction = "flat"
        if primary_log2fc is not None:
            if primary_log2fc > 0:
                direction = "up"
            elif primary_log2fc < 0:
                direction = "down"

        summary.append(
            {
                "gene": gene_symbol,
                "log2fc": primary_log2fc,
                "direction": direction,
                "log2fc_values": log2fc_values,
                "metadata": metadata,
            }
        )

    return summary


def _extract_metadata(
    record: Dict[str, Any],
    gene_column: str,
    log2fc_cols: List[str],
) -> List[Dict[str, Any]]:
    """Collect supplemental gene metadata for display."""

    metadata: List[Dict[str, Any]] = []

    for key, value in record.items():
        if key in log2fc_cols or key == gene_column or key.startswith("__"):
            continue
        if pd.isna(value):
            continue
        if isinstance(value, (list, dict)):
            continue

        metadata.append({
            "name": key,
            "value": str(value),
        })

        if len(metadata) >= 4:
            break

    return metadata


def _format_metric_label(column: str, patient_prefix: str) -> str:
    """Generate a human-readable label for a log2FC column."""

    lower_column = column.lower()
    lower_prefix = patient_prefix.lower()
    label = column

    if lower_prefix and lower_column.startswith(lower_prefix):
        label = column[len(patient_prefix):]
        if label.startswith("_"):
            label = label[1:]

    label = label.replace("_log2FC", "")
    label = label.replace("_log2fc", "")
    label = label.replace("_", " ").strip()

    return label.title() if label else column


def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float if possible."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric_value):
        return None

    return numeric_value


__all__ = [
    "load_patient_genes",
    "map_genes_to_pathways",
]


