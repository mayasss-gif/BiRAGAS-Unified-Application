from pathlib import Path
import argparse, json, sys, os

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .pipeline_agent import harmonize_from_local, harmonize_single_paths
except Exception as e:
    print("Could not import pipeline code. Make sure pipeline_agent.py and harmonizer_core.py are in the same folder as this cli.py.")
    print(f"Import error: {e}")
    raise


def pretty(obj):
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


def cmd_local(args):
    result = harmonize_from_local(
        data_root=args.data_root,
        combine=(not args.no_combine),
        out_root=args.out_root
    )
    print("\n=== Run summary (local) ===")
    print(pretty(result))

    outputs = []
    if result.get("mode") == "single":
        outputs.append(result["result"]["outdir"])
        outputs.append(result["result"]["figdir"])
        outputs.append(result["result"]["zip"])
    elif result.get("mode") == "multi":
        combined = result.get("combined") or {}
        for name, run in (result.get("runs") or {}).items():
            outputs.extend([run.get("outdir"), run.get("figdir"), run.get("zip")])
        if combined:
            outputs.extend([combined.get("outdir"), combined.get("figdir"), combined.get("zip")])

    outputs = [o for o in outputs if o]
    summary_path = HERE / "last_run_summary.local.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"result": result, "outputs": outputs}, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    for o in outputs:
        print(f" - {o}")


def cmd_single(args):
    res = harmonize_single_paths(
        counts_path=args.counts,
        meta_path=args.meta,
        out_mode=args.out_mode,
        out_root=args.out_root,
    )
    print("\n=== Run summary (single) ===")
    print(pretty(res))

    outputs = [res.get("outdir"), res.get("figdir"), res.get("zip")]
    outputs = [o for o in outputs if o]
    summary_path = HERE / "last_run_summary.single.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"result": res, "outputs": outputs}, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    for o in outputs:
        print(f" - {o}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="harmonizer",
        description="Local harmonizer CLI: auto-detect 'prep' folders (counts + meta) and run the pipeline."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # local mode
    pl = sub.add_parser("local", help="Crawl a root folder, auto-discover 'prep' folders and (counts, meta) pairs")
    pl.add_argument("--data-root", required=True, help="Root directory to crawl. Any subfolder whose path contains 'prep' will be considered.")
    pl.add_argument("--no-combine", action="store_true", help="If set, skip cross-dataset combine step.")
    pl.add_argument("--out-root", default=None, help="Base directory for multi-dataset outputs (defaults to current working directory).")
    pl.set_defaults(func=cmd_local)

    # single mode
    ps = sub.add_parser("single", help="Run on explicit counts/meta file paths")
    ps.add_argument("--counts", required=True, help="Path to the counts/expression table (csv/tsv/xlsx).")
    ps.add_argument("--meta", required=True, help="Path to the metadata table (csv/tsv/xlsx).")
    ps.add_argument("--out-mode", choices=["co_locate", "default"], default="co_locate", help="Where to put outputs for a single dataset.")
    ps.add_argument("--out-root", default=None, help="Output base when using out-mode=default.")
    ps.set_defaults(func=cmd_single)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
