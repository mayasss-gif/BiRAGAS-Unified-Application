from __future__ import annotations
from pathlib import Path
import sys

def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

