import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MIN_SAMPLES_FOR_JOINT = 50  # same as notebook


def _load_norm_manifest(step2_dir: str) -> Dict:
    path = os.path.join(step2_dir, "normalization_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalization summary not found at {path}. "
            "Run Step 2 (preprocessing) before Step 3."
        )
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print("✅ Loaded normalization summary from Step 2.")
    print(json.dumps(manifest, indent=2))
    return manifest


def _load_metadata(step1_dir: str) -> pd.Series:
    """Load metadata.csv from Step 1 and return a label Series."""
    meta_path = os.path.join(step1_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found at {meta_path}. "
            "Step 1 should have created metadata.csv."
        )

    meta = pd.read_csv(meta_path, index_col=0)
    meta.index = meta.index.astype(str)

    if "label" not in meta.columns:
        meta["label"] = "Diseased"

    labels = meta["label"].astype(str)

    print("\n✅ Loaded metadata with labels.")
    print("Label distribution:")
    print(labels.value_counts())
    return labels


def _load_normalized_layers(step2_dir: str, norm_manifest: Dict) -> Dict[str, pd.DataFrame]:
    """Load all normalized layers from Step 2."""
    layers_norm: Dict[str, pd.DataFrame] = {}
    for lname, info in norm_manifest["layers"].items():
        fpath = os.path.join(step2_dir, info["file"])
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
            layers_norm[lname] = df
            print(f"[{lname}] loaded: {df.shape[0]} samples × {df.shape[1]} features")
        else:
            print(f"⚠️ Normalized file for layer '{lname}' not found at {fpath}, skipping.")

    available_layers = list(layers_norm.keys())
    print("\nAvailable layers:", available_layers)

    if len(available_layers) == 0:
        raise RuntimeError("❌ No normalized layers available. Cannot proceed.")

    return layers_norm


def _compute_overlaps(layers_norm: Dict[str, pd.DataFrame]):
    sample_sets = {lname: set(df.index) for lname, df in layers_norm.items()}

    if len(sample_sets) > 1:
        all_intersection = set.intersection(*sample_sets.values())
    else:
        all_intersection = next(iter(sample_sets.values()))

    pairwise_overlaps: Dict[str, int] = {}
    layer_list = list(layers_norm.keys())
    for i, li in enumerate(layer_list):
        for lj in layer_list[i + 1 :]:
            inter = sample_sets[li].intersection(sample_sets[lj])
            pairwise_overlaps[f"{li}|{lj}"] = len(inter)

    print("\n🔍 Sample overlap:")
    print("Global intersection across all layers:", len(all_intersection))
    print("Pairwise overlaps:")
    for k, v in pairwise_overlaps.items():
        print(f"  {k}: {v}")

    return sample_sets, all_intersection, pairwise_overlaps


def choose_core_layers(
    sample_sets: Dict[str, set], min_samples: int = MIN_SAMPLES_FOR_JOINT
) -> Tuple[List[str], set]:
    """
    Greedy choice of a maximal set of layers with sufficient overlap.
    Returns (core_layers, core_samples_set).
    """
    layers_sorted = sorted(sample_sets.keys(), key=lambda k: len(sample_sets[k]), reverse=True)
    core_layers: List[str] = []
    core_samples: set = set()

    for lname in layers_sorted:
        if not core_layers:
            core_layers = [lname]
            core_samples = set(sample_sets[lname])
        else:
            new_samples = core_samples.intersection(sample_sets[lname])
            if len(new_samples) >= min_samples:
                core_layers.append(lname)
                core_samples = new_samples

    if len(core_layers) < 2:
        return [], set()
    return core_layers, core_samples


def prefix_columns(df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    return df.rename(columns={c: f"{layer_name}__{c}" for c in df.columns})


def run_step3(base_dir: str) -> Dict[str, str]:
    """
    STEP 3 — Integration Controller & ML Matrix

    Generalized version of your notebook Step 3.

    Parameters
    ----------
    base_dir : str
        Same base directory used in Steps 1 & 2 (where step_1_ingestion and step_2_preprocessing live).

    Returns
    -------
    dict with:
      - step_dir: path to step_3_integration
      - integration_summary: path to integration_summary.json
      - ml_matrix: path to integrated_matrix_for_ml.parquet
      - labels: path to labels_for_ml.csv
      - feature_map: path to feature_map.csv
    """
    step1_dir = os.path.join(base_dir, "step_1_ingestion")
    step2_dir = os.path.join(base_dir, "step_2_preprocessing")
    step3_dir = os.path.join(base_dir, "step_3_integration")
    os.makedirs(step3_dir, exist_ok=True)

    # 1) Load inputs
    norm_manifest = _load_norm_manifest(step2_dir)
    labels_full = _load_metadata(step1_dir)
    layers_norm = _load_normalized_layers(step2_dir, norm_manifest)

    available_layers = list(layers_norm.keys())
    sample_sets, all_intersection, pairwise_overlaps = _compute_overlaps(layers_norm)

    # 2) Decide integration mode (same logic as notebook)
    integration_mode = None
    core_layers: List[str] = []
    core_samples: set = set()
    external_layers: List[str] = []

    if len(available_layers) == 1:
        integration_mode = "single_omics"
        core_layers = available_layers.copy()
        core_samples = sample_sets[core_layers[0]]
        external_layers = []
    else:
        if len(all_intersection) >= MIN_SAMPLES_FOR_JOINT:
            integration_mode = "full_joint"
            core_layers = available_layers.copy()
            core_samples = all_intersection
            external_layers = []
        else:
            core_layers_candidate, core_samples_candidate = choose_core_layers(sample_sets)
            if len(core_layers_candidate) >= 2 and len(core_samples_candidate) >= MIN_SAMPLES_FOR_JOINT:
                integration_mode = "hybrid_core"
                core_layers = core_layers_candidate
                core_samples = core_samples_candidate
                external_layers = [l for l in available_layers if l not in core_layers]
            else:
                integration_mode = "multi_cohort"
                core_layers = []
                core_samples = set()
                external_layers = available_layers.copy()

    print("\n🧠 Integration decision:")
    print("  mode           :", integration_mode)
    print("  core_layers    :", core_layers)
    print("  core_samples   :", len(core_samples))
    print("  external_layers:", external_layers)

    # 3) Build ML matrix (always)
    ML_MATRIX_PATH = os.path.join(step3_dir, "integrated_matrix_for_ml.parquet")
    FEATURE_MAP_PATH = os.path.join(step3_dir, "feature_map.csv")
    LABELS_PATH = os.path.join(step3_dir, "labels_for_ml.csv")

    feature_rows = []

    # Determine ML base strategy
    if integration_mode in ["full_joint", "hybrid_core"]:
        if len(core_samples) == 0:
            raise RuntimeError("Core samples empty in joint mode. This should not happen.")
        core_samples_sorted = sorted(core_samples)
        X_list = []

        for lname in core_layers:
            df = layers_norm[lname].loc[core_samples_sorted].copy()
            df_pref = prefix_columns(df, lname)
            X_list.append(df_pref)

            feature_rows.extend(
                [
                    {"feature": col, "layer": lname, "original_feature": orig}
                    for orig, col in zip(df.columns, df_pref.columns)
                ]
            )

        X_ml = pd.concat(X_list, axis=1)
        y_ml = labels_full.reindex(core_samples_sorted)
        ml_base_mode = "core_block"

    elif integration_mode == "single_omics":
        lname = available_layers[0]
        df = layers_norm[lname].copy()
        df_pref = prefix_columns(df, lname)

        X_ml = df_pref
        y_ml = labels_full.reindex(df.index)

        feature_rows = [
            {"feature": col, "layer": lname, "original_feature": orig}
            for orig, col in zip(df.columns, df_pref.columns)
        ]

        ml_base_mode = f"single_layer_{lname}"

    elif integration_mode == "multi_cohort":
        best_layer = max(available_layers, key=lambda l: layers_norm[l].shape[0])
        df = layers_norm[best_layer].copy()
        df_pref = prefix_columns(df, best_layer)

        X_ml = df_pref
        y_ml = labels_full.reindex(df.index)

        feature_rows = [
            {"feature": col, "layer": best_layer, "original_feature": orig}
            for orig, col in zip(df.columns, df_pref.columns)
        ]

        ml_base_mode = f"multi_cohort_single_base_{best_layer}"

    else:
        raise RuntimeError(f"Unknown integration_mode: {integration_mode}")

    # Clean labels just like in notebook
    if y_ml.isna().all():
        y_ml = pd.Series("Diseased", index=X_ml.index, name="label")
    else:
        y_ml = y_ml.fillna("Diseased")
        y_ml.name = "label"

    # Drop samples with all-NaN features (safety)
    valid_rows = X_ml.notna().any(axis=1)
    X_ml = X_ml.loc[valid_rows]
    y_ml = y_ml.loc[valid_rows]

    print(f"\n📐 Final ML matrix shape: {X_ml.shape[0]} samples × {X_ml.shape[1]} features")
    print("Label distribution in ML set:")
    print(y_ml.value_counts())

    # Save ML artifacts
    X_ml.to_parquet(ML_MATRIX_PATH)
    pd.DataFrame(feature_rows).to_csv(FEATURE_MAP_PATH, index=False)
    y_ml.to_frame().to_csv(LABELS_PATH)

    print("\n💾 Saved ML-ready matrix →", ML_MATRIX_PATH)
    print("💾 Saved feature map      →", FEATURE_MAP_PATH)
    print("💾 Saved ML labels        →", LABELS_PATH)

    # 4) Save integration summary
    integration_summary = {
        "base_dir": step3_dir,
        "integration_mode": integration_mode,
        "ml_base_mode": ml_base_mode,
        "available_layers": available_layers,
        "core_layers": core_layers,
        "external_layers": external_layers,
        "n_core_samples": int(len(core_samples)),
        "pairwise_overlaps": pairwise_overlaps,
        "ml_matrix_file": os.path.basename(ML_MATRIX_PATH),
        "labels_file": os.path.basename(LABELS_PATH),
        "feature_map_file": os.path.basename(FEATURE_MAP_PATH),
    }

    summary_path = os.path.join(step3_dir, "integration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(integration_summary, f, indent=2)

    print("\n🧾 Integration summary written to:", summary_path)
    print(json.dumps(integration_summary, indent=2))

    print("\n✅ STEP 3 (Integration controller + ML matrix) complete.")

    return {
        "step_dir": step3_dir,
        "integration_summary": summary_path,
        "ml_matrix": ML_MATRIX_PATH,
        "labels": LABELS_PATH,
        "feature_map": FEATURE_MAP_PATH,
    }
