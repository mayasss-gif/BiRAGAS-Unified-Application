from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def scan_fastq_samples(folder: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Discover FASTQ files and group them by sample, tracking R1/R2 pairs."""
    files = list(folder.glob("*.fastq*"))
    sample_map: Dict[str, Dict[str, Optional[str]]] = defaultdict(
        lambda: {"R1": None, "R2": None}
    )
    for f in files:
        match = re.match(
            r"(.+?)(?:_R|_)([12])\.f(ast)?q(\.gz)?$", f.name, re.IGNORECASE
        )
        if match:
            base, pair = match.group(1), match.group(2)
            sample_map[base][f"R{pair}"] = str(f)
        else:
            base = re.sub(r"\.f(ast)?q(\.gz)?$", "", f.name, flags=re.IGNORECASE)
            sample_map[base]["R1"] = str(f)
    return {k: v for k, v in sample_map.items() if v["R1"]}


def print_samples(samples: Dict[str, Dict[str, Optional[str]]]) -> None:
    """Log a concise sample summary."""
    print(f"Found {len(samples)} samples:")
    for name, files in samples.items():
        paired = "paired-end" if files["R2"] else "single-end"
        print(f" - {name}: {paired}")


def make_jobs(samples: Dict[str, Dict[str, Optional[str]]], results_dir: Path) -> list:
    """Transform discovered samples into jobs expected by the agent."""
    jobs = []
    for name, files in samples.items():
        jobs.append(
            {
                "sample_name": name,
                "fastq_1": files["R1"],
                "fastq_2": files["R2"],
                "work_dir": str(results_dir / name),
            }
        )
    return jobs

