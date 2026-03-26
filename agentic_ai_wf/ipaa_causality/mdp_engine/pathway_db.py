# mdp_engine/pathway_db.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

from .exceptions import DataError, ValidationError


@dataclass(frozen=True)
class Pathway:
    pid: str
    name: str
    genes: Set[str]
    source: str = "custom"


def load_gmt(gmt_path: Path, source: str = "gmt") -> Dict[str, Pathway]:
    """
    Load GMT:
      pathway_name <tab> description <tab> gene1 <tab> gene2 ...
    If pathway_name contains a '|', split into pid|name.
    """
    if gmt_path is None:
        raise ValidationError("gmt_path is None")
    gmt_path = Path(gmt_path)
    if not gmt_path.exists():
        raise DataError(f"GMT file not found: {gmt_path}")

    out: Dict[str, Pathway] = {}
    with gmt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            name = parts[0].strip()
            genes = {g.strip() for g in parts[2:] if g.strip()}
            if not name or not genes:
                continue

            if "|" in name:
                pid, pname = name.split("|", 1)
                pid = pid.strip() or name
                pname = pname.strip() or name
            else:
                pid, pname = name, name

            if pid in out:
                out[pid] = Pathway(pid=pid, name=out[pid].name or pname, genes=out[pid].genes | genes, source=source)
            else:
                out[pid] = Pathway(pid=pid, name=pname, genes=genes, source=source)

    if not out:
        raise DataError(f"No pathways parsed from GMT: {gmt_path}")
    return out


def merge_pathway_dicts(*dicts: Dict[str, Pathway]) -> Dict[str, Pathway]:
    merged: Dict[str, Pathway] = {}
    for d in dicts:
        if not d:
            continue
        for pid, pw in d.items():
            if pid in merged:
                merged[pid] = Pathway(
                    pid=pid,
                    name=merged[pid].name or pw.name,
                    genes=set(merged[pid].genes) | set(pw.genes),
                    source=f"{merged[pid].source}+{pw.source}",
                )
            else:
                merged[pid] = pw
    return merged
