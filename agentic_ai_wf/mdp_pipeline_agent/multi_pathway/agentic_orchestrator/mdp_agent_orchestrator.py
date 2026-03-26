
# mdp_agent_orchestrator.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from exec_agent import ExecutionAgent
from interpret_agent import InterpretationAgent

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan")
    sp.add_argument("--input", required=True)
    sp.add_argument("--goal", required=True)
    sp.add_argument("--out-root", default=None)
    sr = sub.add_parser("run")
    sr.add_argument("--input", required=True)
    sr.add_argument("--goal", required=True)
    sr.add_argument("--out-root", default=None)
    si = sub.add_parser("interpret")
    si.add_argument("--jsons-root", required=True)
    si.add_argument("--task", choices=["regulators"], required=True)
    si.add_argument("--min-shared", type=int, default=2)
    args = ap.parse_args()
    if args.cmd in {"plan","run"}:
        agent = ExecutionAgent(project_root=Path(".").resolve())
        plan = agent.plan(args.input, args.goal, out_root=args.out_root)
        if args.cmd == "plan":
            print(json.dumps({"data_type": plan.data_type,"out_root": plan.out_root,"jsons_root": plan.jsons_root,
                              "actions": plan.actions,"expected_features": sorted(list(plan.expected_features))}, indent=2))
            return 0
        else:
            result = agent.execute(plan)
            print(json.dumps(result, indent=2)); return 0
    if args.cmd == "interpret":
        interp = InterpretationAgent()
        if args.task == "regulators":
            tables = interp.regulators_shared_vs_specific(args.jsons_root, min_shared=args.min_shared)
            out_dir = Path(args.jsons_root).parent / "agentic" / "interpretation"
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, df in tables.items():
                df.to_csv(out_dir / f"regulators_{k}_shared_vs_specific.csv", index=False)
            print(json.dumps({"saved": [str((out_dir / f'regulators_{k}_shared_vs_specific.csv')) for k in tables.keys()]}, indent=2))
            return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
