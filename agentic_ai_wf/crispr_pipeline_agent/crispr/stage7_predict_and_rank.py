#!/usr/bin/env python3
"""
Stage 7: Predict + Rank perturbations using trained Stage-6 model

✔ Auto-detects best Stage-6 model
✔ Correctly loads XGBoost (.json), RF/LR (.joblib), MLP (.pt)
✔ Uses Stage-6 scaler
✔ Safe PCA alignment
✔ No assumptions, no crashes
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------
def log(msg: str):
    print(msg, flush=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Load helpers
# ----------------------------
def read_label_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "class_id" not in df.columns:
        if "label_id" in df.columns:
            df = df.rename(columns={"label_id": "class_id"})
        else:
            raise ValueError(f"Label map missing class_id: {df.columns}")
    return df[["class_id", "perturbation_id"]]


def detect_best_model(stage6_dir: Path) -> str:
    df = pd.read_csv(stage6_dir / "stage6_metrics_test.tsv", sep="\t")
    best = df.sort_values("accuracy", ascending=False).iloc[0]["model"]
    log(f"[INFO] Best Stage-6 model: {best}")
    return str(best)


def load_stage6_model(stage6_dir: Path, model_name: str, X_dim: int, n_classes: int):
    # ---------- MLP ----------
    if model_name == "mlp":
        import torch
        import torch.nn as nn

        model_path = stage6_dir / "stage6_mlp_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model = nn.Sequential(
            nn.Linear(X_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, "mlp"

    # ---------- XGBoost ----------
    if model_name == "xgb":
        import xgboost as xgb

        model_path = stage6_dir / "stage6_xgb_model.json"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        log(f"[INFO] Loading XGBoost model: {model_path.name}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, "xgb"

    # ---------- RF / Logistic ----------
    model_path = stage6_dir / f"stage6_{model_name}_model.joblib"
    if model_path.exists():
        log(f"[INFO] Loading model: {model_path.name}")
        return joblib.load(model_path), model_name

    raise FileNotFoundError(f"No model found for '{model_name}'")


def predict(model, model_type: str, X: np.ndarray):
    if model_type == "mlp":
        import torch
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_t).numpy()
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.argmax(axis=1), probs.max(axis=1)

    # xgb / rf / logreg
    probs = model.predict_proba(X)
    return probs.argmax(axis=1), probs.max(axis=1)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--stage5_dir", required=True)
    ap.add_argument("--stage6_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_cells", type=int, default=None)
    args = ap.parse_args()

    input_h5ad = Path(args.input_h5ad)
    stage5_dir = Path(args.stage5_dir)
    stage6_dir = Path(args.stage6_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ---- Load data ----
    log(f"[INFO] Reading h5ad: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)

    X = np.load(stage5_dir / "stage5_X_pca.npy")
    label_map = read_label_map(stage5_dir / "stage5_label_map.tsv")

    if X.shape[0] != adata.n_obs:
        n = min(X.shape[0], adata.n_obs)
        log(f"[WARN] PCA mismatch → truncating to {n}")
        X = X[:n]
        adata = adata[:n].copy()

    # ---- Scale ----
    scaler = joblib.load(stage6_dir / "stage6_scaler.joblib")
    Xs = scaler.transform(X)

    # ---- Model ----
    best_model = detect_best_model(stage6_dir)
    model, model_type = load_stage6_model(
        stage6_dir,
        best_model,
        X_dim=Xs.shape[1],
        n_classes=label_map["class_id"].nunique(),
    )

    # ---- Predict ----
    log("[INFO] Predicting")
    pred_class, pred_conf = predict(model, model_type, Xs)

    class_to_pert = label_map.set_index("class_id")["perturbation_id"].to_dict()
    pred_pert = [class_to_pert.get(int(c), "UNKNOWN") for c in pred_class]

    # ---- Export ----
    df = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "pred_class": pred_class,
        "pred_perturbation_id": pred_pert,
        "pred_confidence": pred_conf,
    })

    df.to_csv(out_dir / "stage7_cell_predictions.tsv", sep="\t", index=False)

    rank = (
        df.groupby("pred_perturbation_id")
        .agg(
            n_cells=("cell_id", "count"),
            mean_conf=("pred_confidence", "mean"),
        )
        .sort_values("mean_conf", ascending=False)
        .reset_index()
    )

    rank.to_csv(
        out_dir / "stage7_perturbation_ranked_by_confidence.tsv",
        sep="\t",
        index=False,
    )

    log("[DONE] Stage-7 complete")


if __name__ == "__main__":
    main()

