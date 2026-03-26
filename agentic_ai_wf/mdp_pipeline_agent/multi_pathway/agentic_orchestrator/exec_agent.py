
# exec_agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import json, subprocess

from detector import DataTypeDetector
from capabilities import CAPABILITY_MAP

def run_cmd(cmd: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr

@dataclass
class ExecutionPlan:
    data_type: str
    actions: List[Dict] = field(default_factory=list)
    expected_features: Set[str] = field(default_factory=set)
    out_root: Optional[str] = None
    jsons_root: Optional[str] = None

class ExecutionAgent:
    def __init__(self, project_root: str | Path = "."):
        self.project_root = Path(project_root).resolve()

    def detect_inputs(self, input_path: str | Path):
        return DataTypeDetector(input_path).detect()

    def plan(self, input_path: str | Path, goal: str, out_root: Optional[str] = None) -> ExecutionPlan:
        p = Path(input_path)
        data_type, reason = self.detect_inputs(p)
        if data_type is None:
            raise ValueError(f"Cannot detect input type for {input_path}: {reason}")
        plan = ExecutionPlan(data_type=data_type)
        if out_root is None:
            out_root = str((self.project_root / "agentic_out").resolve())
        plan.out_root = out_root
        if data_type == "JSONS":
            plan.jsons_root = str(Path(input_path).resolve())
            plan.expected_features = CAPABILITY_MAP["JSONS"]
            return plan
        scripts = {
            "COUNTS": ["python","run_counts_from_dict.py","--spec",str(Path(input_path).resolve())],
            "DEGS":   ["python","-m","genelist_mdp.GL_enrich","--input",str(Path(input_path).resolve()),"--out-root",out_root],
            "GL":     ["python","-m","genelist_mdp.GL_enrich","--input",str(Path(input_path).resolve()),"--out-root",out_root],
            "GC":     ["python","-m","genelist_mdp.GC_enrich","--input",str(Path(input_path).resolve()),"--out-root",out_root],
        }
        if data_type not in scripts:
            raise ValueError(f"No script mapping for data type {data_type}")
        plan.actions.append({"name": f"run_{data_type.lower()}_pipeline","cmd": scripts[data_type],"expect": "OVERLAP_JSONS"})
        plan.actions.append({"name": "discover_jsons","cmd": ["python","-c","import sys;print(sys.argv[1])", str(Path(out_root)/"results"/"all_jsons")],"expect":"OVERLAP_JSONS"})
        plan.jsons_root = str(Path(out_root)/"results"/"all_jsons")
        plan.expected_features = CAPABILITY_MAP[data_type]
        return plan

    def execute(self, plan: ExecutionPlan) -> Dict:
        log = []
        for step in plan.actions:
            code, out = run_cmd(step["cmd"], cwd=self.project_root)
            log.append({"step": step["name"], "cmd": step["cmd"], "exit_code": code, "output": out})
            if code != 0:
                break
        jsons = Path(plan.jsons_root) if plan.jsons_root else None
        available = {"OVERLAP_JSONS": (jsons and jsons.exists())}
        result = {"data_type": plan.data_type,"out_root": plan.out_root,"jsons_root": plan.jsons_root,
                  "expected_features": sorted(list(plan.expected_features)),"available": available,"log": log}
        prov_dir = Path(plan.out_root)/"agentic"
        prov_dir.mkdir(parents=True, exist_ok=True)
        (prov_dir/"execution_result.json").write_text(json.dumps(result, indent=2))
        return result
