"""
Tar file processor for extracting count files from TAR archives.
"""

import tarfile
import io
import gzip
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def process_tar_counts(
    tar_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_name: str = "prep_counts.csv",
    verbose: bool = True,
) -> Optional[Path]:
    """
    Parse per-sample count files inside a TAR/TGZ archive
    and write a merged genes×samples matrix to output_dir/output_name.

    Args:
        tar_path: Path to tar/tar.gz/tgz file
        output_dir: Directory to save the merged counts CSV
        output_name: Name of the output file (default: prep_counts.csv)
        verbose: Whether to log progress

    Returns:
        Path to the merged CSV file, or None if processing failed

    Behavior:
      - Accepts .tar, .tar.gz, .tgz archives.
      - Reads member files that end with .gz/.txt/.tsv/.csv.
      - Decompresses member bytes if gzipped.
      - Heuristically detects gene and count columns.
      - Drops HTSeq summary rows (those starting with "__").
      - Outer-joins per-sample vectors by gene ID (fill NAs with 0).
      - Casts to int only if values are integer-like; otherwise keeps floats.
    """
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / output_name

    if not tar_path.exists():
        logger.error(f"Tar file does not exist: {tar_path}")
        return None

    def _guess_sample_name(p: str) -> str:
        """Extract sample name from file path."""
        base = Path(p).name
        for ext in (".txt.gz", ".tsv.gz", ".csv.gz", ".counts.gz", ".count.gz",
                    ".txt", ".tsv", ".csv", ".counts", ".count", ".gz"):
            if base.endswith(ext):
                return base[:-len(ext)]
        return base

    def _read_member_as_df(tar: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
        """Read a tar member file and return as DataFrame with sample name as column."""
        sample = _guess_sample_name(member.name)
        fh = tar.extractfile(member)
        if fh is None:
            raise IOError(f"Cannot read {member.name}")
        raw = fh.read()
        
        # Try to decompress if gzipped
        try:
            data = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        except OSError:
            data = raw
        
        buf = io.BytesIO(data)

        # Try different separators and header options
        for sep in ("\t", ","):
            for header in (0, None):
                buf.seek(0)
                try:
                    df = pd.read_csv(buf, sep=sep, header=header, engine="python")
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    # Find gene column
                    gene_col = next((c for c in df.columns if str(c).lower() in
                                     ["gene", "gene_id", "geneid", "target_id", "feature_id",
                                      "ensembl", "ensembl_id", "id"]), None)
                    
                    # Find count column (first numeric column that's not the gene column)
                    count_col = None
                    for c in df.columns:
                        if c != gene_col and pd.api.types.is_numeric_dtype(df[c]):
                            count_col = c
                            break
                    
                    if gene_col is None:
                        # Fallback: use first two columns
                        if df.shape[1] >= 2:
                            gene_col, count_col = df.columns[0], df.columns[1]
                        else:
                            continue
                    
                    # Extract gene and count columns
                    sub = df[[gene_col, count_col]].copy()
                    # Drop HTSeq summary rows
                    sub = sub[~sub[gene_col].astype(str).str.startswith("__")]
                    # Convert counts to numeric
                    sub[count_col] = pd.to_numeric(sub[count_col], errors="coerce").fillna(0)
                    # Group by gene and sum (in case of duplicates)
                    sub = sub.groupby(gene_col, as_index=True)[count_col].sum().to_frame(name=sample)
                    return sub
                except Exception:
                    continue
        raise ValueError(f"Could not parse {member.name}")

    try:
        matrices = []
        with tarfile.open(tar_path, mode="r:*") as tar:
            members = [m for m in tar.getmembers()
                      if m.isfile() and m.name.lower().endswith((".gz", ".txt", ".tsv", ".csv"))]
            
            if not members:
                logger.warning(f"No count-like files found inside {tar_path.name}")
                return None
            
            for m in members:
                try:
                    df = _read_member_as_df(tar, m)
                    matrices.append(df)
                    if verbose:
                        logger.info(f"✓ parsed {m.name} → {df.shape[0]} genes")
                except Exception as e:
                    if verbose:
                        logger.warning(f"✗ skipped {m.name}: {e}")

        if not matrices:
            logger.error(f"No valid count tables parsed from {tar_path.name}")
            return None

        # Merge all sample matrices
        merged = pd.concat(matrices, axis=1, join="outer").fillna(0)
        
        # Convert to int if all values are integer-like
        vals = merged.values.astype(float)
        if np.allclose(vals, np.round(vals), atol=1e-8):
            merged = np.round(merged).astype(int)
        else:
            if verbose:
                logger.info("⚠️ Detected non-integer values; leaving as floats")

        # Sort by gene ID and sample names
        merged = merged.sort_index().reindex(sorted(merged.columns), axis=1)
        
        # Ensure Gene column name for consistency with pipeline
        merged.index.name = "Gene"
        
        # Save to CSV
        merged.to_csv(out_csv)
        
        if verbose:
            logger.info(f"✅ Saved merged counts matrix: {out_csv}  [{merged.shape[0]} genes × {merged.shape[1]} samples]")
        
        return out_csv
        
    except Exception as e:
        logger.error(f"Error processing tar file {tar_path}: {e}")
        return None

