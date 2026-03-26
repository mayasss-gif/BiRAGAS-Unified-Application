from __future__ import annotations
from pathlib import Path
import re
import yaml
import pandas as pd
import questionary

from .utils import die

GSM_RE = re.compile(r"^(GSM\d+)(?:_.*)?$")

def _list_dataset_dirs(input_root: str) -> list[Path]:
    root = Path(input_root)
    if not root.exists():
        die(f"Input root not found: {root}")
    ds = [p for p in root.iterdir() if p.is_dir()]
    if not ds:
        die(f"No dataset folders found inside: {root}")
    return sorted(ds)

def _discover_gsms(dataset_dir: Path) -> list[str]:
    # handles filenames like GSM2406675_10X001_barcodes.tsv.gz, GSM2406675_... etc
    gsms = set()
    for f in dataset_dir.iterdir():
        m = GSM_RE.match(f.name.split("_")[0])
        if m:
            gsms.add(m.group(1))
    return sorted(gsms)

def _load_metadata(dataset_dir: Path) -> pd.DataFrame | None:
    # expects metadata.csv with columns GSM,Group (Group can be any)
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        return None
    df = pd.read_csv(meta_path)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    if "gsm" not in cols or "group" not in cols:
        # allow flexible names
        return df
    df = df.rename(columns={cols["gsm"]: "GSM", cols["group"]: "Group"})
    df["GSM"] = df["GSM"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.strip()
    return df

def run_wizard(input_root: str, out_root: str, config_out: str):
    ds_dirs = _list_dataset_dirs(input_root)

    ds_choice = questionary.select(
        "Select dataset folder (inside input_root):",
        choices=[d.name for d in ds_dirs],
    ).ask()

    dataset_dir = next(d for d in ds_dirs if d.name == ds_choice)

    gsms = _discover_gsms(dataset_dir)
    if not gsms:
        die(f"No GSM-like prefixes detected in {dataset_dir}")

    gsm_mode = questionary.select(
        "Which GSMs to use?",
        choices=["All GSMs", "Select subset"],
    ).ask()

    if gsm_mode == "All GSMs":
        gsm_selected = gsms
    else:
        gsm_selected = questionary.checkbox(
            "Select GSM(s):",
            choices=gsms,
            validate=lambda x: True if len(x) > 0 else "Select at least one GSM",
        ).ask()

    run_mode = questionary.select(
        "Run mode:",
        choices=["Single-sample", "Multi-sample"],
    ).ask()

    # If user picked single-sample but selected multiple, ask which one
    if run_mode == "Single-sample" and len(gsm_selected) > 1:
        gsm_selected = [
            questionary.select("Pick one GSM for single-sample:", choices=gsm_selected).ask()
        ]

    meta_df = _load_metadata(dataset_dir)
    use_conditions = questionary.confirm(
        "Do you want condition labeling (control vs disease) and Disease vs Control DEGs?",
        default=True
    ).ask()

    control_groups, disease_groups = [], []
    if use_conditions:
        if meta_df is None:
            die(
                "metadata.csv not found. Put metadata.csv in the dataset folder with columns like:\n"
                "GSM,Group\nGSM2406675,Pilot_TF\n..."
            )

        # Use "Group" column if present, else ask user to name the column
        if "Group" not in meta_df.columns:
            col = questionary.select(
                "metadata.csv found. Which column contains sample group names?",
                choices=list(meta_df.columns),
            ).ask()
            meta_df = meta_df.rename(columns={col: "Group"})

        if "GSM" not in meta_df.columns:
            col = questionary.select(
                "Which column contains GSM IDs?",
                choices=list(meta_df.columns),
            ).ask()
            meta_df = meta_df.rename(columns={col: "GSM"})

        # Filter to selected GSMs
        meta_df = meta_df[meta_df["GSM"].isin(gsm_selected)].copy()
        if meta_df.empty:
            die("metadata.csv has no rows matching your selected GSMs.")

        groups = sorted(meta_df["Group"].unique().tolist())

        control_groups = questionary.checkbox(
            "Select CONTROL group(s) from metadata Group values:",
            choices=groups,
            validate=lambda x: True if len(x) > 0 else "Pick at least 1 control group",
        ).ask()

        remaining = [g for g in groups if g not in control_groups]
        disease_groups = questionary.checkbox(
            "Select DISEASE group(s) (remaining groups):",
            choices=remaining,
            validate=lambda x: True if len(x) > 0 else "Pick at least 1 disease group",
        ).ask()

    # Integration preference: you asked to run all
    run_all_integrations = questionary.confirm(
        "For multi-sample, run ALL integration methods (none, harmony, bbknn, scvi) if available?",
        default=True
    ).ask()

    # Annotation engines
    annot_engines = questionary.checkbox(
        "Select annotation engines to run:",
        choices=["CellTypist", "CellO", "scVI/scANVI (if available)"],
        validate=lambda x: True if len(x) > 0 else "Pick at least one engine",
    ).ask()

    traj_mode = questionary.select(
        "Trajectory options:",
        choices=[
            "PAGA + DPT (always)",
            "PAGA + DPT + RNA velocity if spliced/unspliced exists (recommended)"
        ],
    ).ask()

    cfg = {
        "paths": {
            "input_root": str(Path(input_root).resolve()),
            "dataset_dir": str(dataset_dir.resolve()),
            "out_root": str(Path(out_root).resolve()),
        },
        "dataset_name": dataset_dir.name,
        "gsm_selected": gsm_selected,
        "run_mode": "single" if run_mode == "Single-sample" else "multi",
        "conditions": {
            "enabled": bool(use_conditions),
            "metadata_csv": str((dataset_dir / "metadata.csv").resolve()) if use_conditions else None,
            "control_groups": control_groups,
            "disease_groups": disease_groups,
            "deg_direction": "disease_vs_control"
        },
        "integration": {
            "run_all": bool(run_all_integrations),
            "methods": ["none", "harmony", "bbknn", "scvi"],
        },
        "annotation": {
            "engines": annot_engines,
            "celltypist_model": "Immune_All_Low.pkl",  # user can change after
            "final_label_strategy": "majority_vote",
            "allow_weak_labels_for_scanvi": True,
        },
        "trajectory": {
            "mode": "paga_dpt_velocity_if_possible" if "velocity" in traj_mode.lower() else "paga_dpt"
        },
        "qc": {
            "min_genes": 200,
            "min_cells": 3,
            "max_mt_pct": 20.0,
            "max_genes": 6000,
            "doublets": {"enabled": True, "expected_doublet_rate": 0.06},
        },
        "preprocess": {
            "hvg_n": 3000,
            "n_pcs": 50,
            "neighbors_k": 15,
            "leiden_resolution": 0.8,
        },
    }

    with open(config_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\nWrote config: {Path(config_out).resolve()}\nNext:\n  scrna-pipe run --config {config_out}\n")

