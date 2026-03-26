import argparse
import sys

from .wizard import run_wizard
from .run import run_from_config


def main():
    parser = argparse.ArgumentParser(
        prog="scrna-pipe",
        description="Interactive end-to-end scRNA / Perturb-seq pipeline"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------
    # wizard command
    # -----------------------
    wizard_p = subparsers.add_parser(
        "wizard",
        help="Interactive wizard to generate run_config.yaml"
    )
    wizard_p.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing dataset folders (e.g. input_data/)"
    )
    wizard_p.add_argument(
        "--out-root",
        required=True,
        help="Root directory for results output"
    )
    wizard_p.add_argument(
        "--config-out",
        default="run_config.yaml",
        help="Path to write generated config (default: run_config.yaml)"
    )

    # -----------------------
    # run command
    # -----------------------
    run_p = subparsers.add_parser(
        "run",
        help="Run pipeline from an existing config YAML"
    )
    run_p.add_argument(
        "--config",
        required=True,
        help="Path to run_config.yaml"
    )

    args = parser.parse_args()

    if args.command == "wizard":
        run_wizard(
            input_root=args.input_root,
            out_root=args.out_root,
            config_out=args.config_out,
        )

    elif args.command == "run":
        run_from_config(args.config)

    else:
        parser.print_help()
        sys.exit(1)
