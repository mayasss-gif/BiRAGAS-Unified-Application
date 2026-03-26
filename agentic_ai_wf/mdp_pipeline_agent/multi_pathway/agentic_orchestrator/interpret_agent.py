
# interpret_agent.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def _iter_jsons(json_root: str | Path):
    p = Path(json_root)
    for f in sorted(p.glob("*.json")):
        yield f

def _extract_regulators(js: Dict) -> Dict[str, List[str]]:
    out = {"tf": [], "epigenetic": [], "metabolites": []}
    if "layers" in js:
        layers = js.get("layers", {})
        for key in out.keys():
            terms = layers.get(key, {}).get("terms", [])
            if isinstance(terms, list):
                out[key].extend([t if isinstance(t, str) else str(t) for t in terms])
    else:
        for pw, d in js.items():
            if not isinstance(d, dict): continue
            for key in out.keys():
                vals = d.get(key, [])
                if isinstance(vals, list):
                    out[key].extend([v if isinstance(v, str) else str(v) for v in vals])
    for k in out: out[k] = sorted(set(out[k]))
    return out

class InterpretationAgent:
    def regulators_shared_vs_specific(self, json_root: str | Path, min_shared: int = 2):
        json_root = Path(json_root)
        disease_reg: Dict[str, Dict[str, List[str]]] = {}
        for f in _iter_jsons(json_root):
            try:
                js = json.loads(f.read_text())
            except Exception:
                continue
            disease = f.stem.replace("_pathway_entity_overlap","")
            disease_reg[disease] = _extract_regulators(js)
        out = {}
        for reg_type in ["tf","epigenetic","metabolites"]:
            rows: List[Tuple[str, int, List[str], str]] = []
            all_regs = sorted(set(r for d in disease_reg.values() for r in d.get(reg_type, [])))
            for r in all_regs:
                present = [d for d, regs in disease_reg.items() if r in regs.get(reg_type, [])]
                n = len(present)
                cat = "shared" if n >= min_shared else ("specific" if n == 1 else "intermediate")
                rows.append((r, n, present, cat))
            df = pd.DataFrame(rows, columns=["regulator","present_in","diseases","category"])
            df.sort_values(["category","present_in","regulator"], ascending=[True,False,True], inplace=True)
            out[reg_type] = df.reset_index(drop=True)
        return out
