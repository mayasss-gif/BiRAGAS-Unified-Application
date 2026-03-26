#!/usr/bin/env python3
"""
Stage 6: Train ML models on Stage-5 PCA embeddings.

Inputs (from Stage-5):
- stage5_X_pca.npy          shape (n_cells, n_pcs)
- stage5_y_perturbation.npy shape (n_cells,)
- stage5_label_map.tsv      maps integer labels -> perturbation_id

Outputs (to out_dir):
- stage6_split_summary.tsv
- stage6_metrics_val.tsv
- stage6_metrics_test.tsv
- stage6_per_class_report_test.tsv
- stage6_confusion_topk.png
- stage6_prf_barplot_test.png
- stage6_topk_barplot_test.png
- stage6_lr_model.joblib
- stage6_rf_model.joblib
- stage6_xgb_model.json
- stage6_mlp_model.pt
- stage6_mlp_training_log.tsv
- stage6_scaler.joblib
- stage6_test_predictions.tsv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost
import xgboost as xgb

# Optional parallel
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Utils
# ----------------------------

def log(msg: str):
    print(msg, flush=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_tsv(df: pd.DataFrame, outpath: Path):
    df.to_csv(outpath, sep="\t", index=False)


def read_label_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    if "perturbation_id" not in df.columns:
        raise ValueError(f"Label map missing perturbation_id: {df.columns}")

    if "class_id" in df.columns:
        pass
    elif "label_id" in df.columns:
        df = df.rename(columns={"label_id": "class_id"})
    elif "label" in df.columns:
        df = df.rename(columns={"label": "class_id"})
    else:
        raise ValueError(f"Label map missing class_id/label_id/label: {df.columns}")

    df["class_id"] = df["class_id"].astype(int)
    df["perturbation_id"] = df["perturbation_id"].astype(str)
    return df[["class_id", "perturbation_id"]]


def stratified_split(X, y, test_size, val_size, seed):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    val_fraction_of_trainval = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize_split(y_train, y_val, y_test, label_map: pd.DataFrame) -> pd.DataFrame:
    df_map = label_map.set_index("class_id")["perturbation_id"].to_dict()

    def counts(y):
        s = pd.Series(y).value_counts().sort_index()
        out = pd.DataFrame({"class_id": s.index.astype(int), "n_cells": s.values})
        out["perturbation_id"] = out["class_id"].map(df_map)
        return out

    c_train = counts(y_train).rename(columns={"n_cells": "train_n"})
    c_val = counts(y_val).rename(columns={"n_cells": "val_n"})
    c_test = counts(y_test).rename(columns={"n_cells": "test_n"})

    merged = c_train.merge(c_val, on=["class_id", "perturbation_id"], how="outer")
    merged = merged.merge(c_test, on=["class_id", "perturbation_id"], how="outer")
    merged = merged.fillna(0)

    for c in ["train_n", "val_n", "test_n"]:
        merged[c] = merged[c].astype(int)

    merged["total_n"] = merged["train_n"] + merged["val_n"] + merged["test_n"]
    return merged.sort_values("total_n", ascending=False)


def eval_model(name, clf, X, y, topk=5, proba=None):
    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    f1w = f1_score(y, y_pred, average="weighted")

    topk_acc = np.nan
    if topk and topk > 1:
        if proba is None and hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X)
            except Exception:
                proba = None
        if proba is not None:
            try:
                topk_acc = top_k_accuracy_score(y, proba, k=topk, labels=np.unique(y))
            except Exception:
                topk_acc = np.nan

    return {
        "model": name,
        "n_cells": int(len(y)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        f"top{topk}_acc": float(topk_acc) if not np.isnan(topk_acc) else np.nan,
    }, y_pred


def plot_confusion_topk(y_true, y_pred, label_map: pd.DataFrame, outpath: Path, topk=20):
    df_map = label_map.set_index("class_id")["perturbation_id"].to_dict()
    vc = pd.Series(y_true).value_counts()
    top_classes = vc.index[:topk].tolist()

    mask = np.isin(y_true, top_classes)
    yt = y_true[mask]
    yp = y_pred[mask]

    cm = confusion_matrix(yt, yp, labels=top_classes)
    names = [df_map.get(int(c), str(c)) for c in top_classes]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix (Top {topk} classes)")
    plt.colorbar()
    tick_marks = np.arange(len(top_classes))
    plt.xticks(tick_marks, names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, names, fontsize=7)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_prf_bar(df_metrics_test: pd.DataFrame, outpath: Path):
    """
    Barplot accuracy + macro-F1 + weighted-F1 + balanced-acc
    """
    use = df_metrics_test.copy()
    cols = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    use = use.set_index("model")[cols]

    plt.figure(figsize=(10, 5))
    use.plot(kind="bar")
    plt.title("Test Metrics per Model")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_topk_bar(df_metrics_test: pd.DataFrame, topk: int, outpath: Path):
    col = f"top{topk}_acc"
    if col not in df_metrics_test.columns:
        return

    use = df_metrics_test[["model", col]].dropna().copy()
    if use.empty:
        return

    use = use.set_index("model")

    plt.figure(figsize=(8, 4))
    use.plot(kind="bar", legend=False)
    plt.title(f"Top-{topk} Accuracy (Test)")
    plt.ylabel("Top-k accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def per_class_report(y_true, y_pred, label_map: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class precision/recall/f1/support in a TSV.
    """
    labels = np.sort(np.unique(y_true))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    df_map = label_map.set_index("class_id")["perturbation_id"].to_dict()

    out = pd.DataFrame({
        "class_id": labels.astype(int),
        "perturbation_id": [df_map.get(int(x), str(x)) for x in labels],
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": s.astype(int),
    }).sort_values(["f1", "support"], ascending=[False, False])

    return out


# ----------------------------
# PyTorch MLP
# ----------------------------

def train_mlp_pytorch(
    X_train, y_train,
    X_val, y_val,
    n_classes: int,
    out_dir: Path,
    epochs=30,
    batch_size=1024,
    lr=1e-3,
    seed=42,
    cpu_threads: int = 4,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    # force CPU threads for EC2
    torch.set_num_threads(max(1, int(cpu_threads)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] MLP device: {device}, CPU threads={cpu_threads}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    in_dim = X_train.shape[1]

    model = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, n_classes),
    ).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log_rows = []
    best_val_acc = -1.0
    best_path = out_dir / "stage6_mlp_model.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * len(yb)
            tr_n += len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()

        tr_loss /= max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_correct = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)

                val_loss += loss.item() * len(yb)
                val_n += len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()

        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        log_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        log(f"[INFO] MLP epoch {ep:03d}/{epochs}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    pd.DataFrame(log_rows).to_csv(out_dir / "stage6_mlp_training_log.tsv", sep="\t", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return model, device


def mlp_predict(model, device, X):
    import torch

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    return y_pred, probs


# ----------------------------
# XGBoost train/predict
# ----------------------------

def train_xgb(X_train_s, y_train, X_val_s, y_val, n_classes: int, seed: int, n_threads: int):
    """
    CPU multi-class softprob XGBoost.
    """
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        max_depth=8,
        learning_rate=0.15,
        n_estimators=350,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=n_threads,
        random_state=seed,
    )
    clf.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )
    return clf


# ----------------------------
# Model runner (for optional parallel)
# ----------------------------

def train_single_model(model_name: str, payload: dict):
    """
    Runs one model and returns trained object + predictions + metrics.
    Payload contains data/params, making it ProcessPool-safe.
    """
    X_train_s = payload["X_train_s"]
    X_val_s = payload["X_val_s"]
    X_test_s = payload["X_test_s"]

    X_train = payload["X_train"]
    X_val = payload["X_val"]
    X_test = payload["X_test"]

    y_train = payload["y_train"]
    y_val = payload["y_val"]
    y_test = payload["y_test"]

    n_classes = payload["n_classes"]
    seed = payload["seed"]
    topk = payload["topk"]

    # threads
    lr_threads = payload["lr_threads"]
    rf_threads = payload["rf_threads"]
    xgb_threads = payload["xgb_threads"]
    mlp_threads = payload["mlp_threads"]

    out_dir = Path(payload["out_dir"])

    if model_name == "logreg":
        log("[INFO] Training Logistic Regression")
        lr = LogisticRegression(
            max_iter=payload["lr_max_iter"],
            solver=payload["lr_solver"],
            C=payload["lr_C"],
        )
        lr.fit(X_train_s, y_train)

        m_val, _ = eval_model("logreg", lr, X_val_s, y_val, topk=topk)
        m_test, y_pred_test = eval_model("logreg", lr, X_test_s, y_test, topk=topk)
        proba_test = lr.predict_proba(X_test_s)

        return ("logreg", lr, m_val, m_test, y_pred_test, proba_test)

    if model_name == "rf":
        log("[INFO] Training RandomForest")
        rf = RandomForestClassifier(
            n_estimators=payload["rf_n_estimators"],
            max_depth=payload["rf_max_depth"],
            n_jobs=rf_threads,
            random_state=seed
        )
        rf.fit(X_train, y_train)

        m_val, _ = eval_model("rf", rf, X_val, y_val, topk=topk)
        m_test, y_pred_test = eval_model("rf", rf, X_test, y_test, topk=topk)
        proba_test = rf.predict_proba(X_test)

        return ("rf", rf, m_val, m_test, y_pred_test, proba_test)

    if model_name == "xgb":
        log("[INFO] Training XGBoost")
        xgbm = train_xgb(X_train_s, y_train, X_val_s, y_val, n_classes=n_classes, seed=seed, n_threads=xgb_threads)

        m_val, _ = eval_model("xgb", xgbm, X_val_s, y_val, topk=topk)
        m_test, y_pred_test = eval_model("xgb", xgbm, X_test_s, y_test, topk=topk)
        proba_test = xgbm.predict_proba(X_test_s)

        return ("xgb", xgbm, m_val, m_test, y_pred_test, proba_test)

    if model_name == "mlp":
        log("[INFO] Training MLP (PyTorch)")
        mlp_model, device = train_mlp_pytorch(
            X_train_s, y_train,
            X_val_s, y_val,
            n_classes=n_classes,
            out_dir=out_dir,
            epochs=payload["mlp_epochs"],
            seed=seed,
            cpu_threads=mlp_threads,
        )

        val_pred, val_proba = mlp_predict(mlp_model, device, X_val_s)
        test_pred, test_proba = mlp_predict(mlp_model, device, X_test_s)

        m_val = {
            "model": "mlp",
            "n_cells": int(len(y_val)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
            "f1_macro": float(f1_score(y_val, val_pred, average="macro")),
            "f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
            f"top{topk}_acc": float(top_k_accuracy_score(y_val, val_proba, k=topk, labels=np.unique(y_val))),
        }

        m_test = {
            "model": "mlp",
            "n_cells": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
            "f1_macro": float(f1_score(y_test, test_pred, average="macro")),
            "f1_weighted": float(f1_score(y_test, test_pred, average="weighted")),
            f"top{topk}_acc": float(top_k_accuracy_score(y_test, test_proba, k=topk, labels=np.unique(y_test))),
        }

        return ("mlp", ("pytorch_mlp", device), m_val, m_test, test_pred, test_proba)

    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-6: train ML models on PCA embeddings")

    parser.add_argument("--stage5_dir", required=True, help="Directory with stage5_X_pca.npy etc.")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    parser.add_argument("--models", default="all",
                        help="Models to train: all OR comma list: xgb,rf,logreg,mlp")
    parser.add_argument("--parallel", action="store_true",
                        help="Train selected models in parallel processes")

    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk_confmat", type=int, default=20)
    parser.add_argument("--topk_acc", type=int, default=5)
    parser.add_argument("--mlp_epochs", type=int, default=30)

    # RF
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=None)

    # Logistic regression
    parser.add_argument("--lr_solver", type=str, default="saga",
                        choices=["lbfgs", "saga", "liblinear", "newton-cg", "sag"])
    parser.add_argument("--lr_C", type=float, default=1.0)
    parser.add_argument("--lr_max_iter", type=int, default=4000)

    # Threads for EC2 tuning
    parser.add_argument("--xgb_threads", type=int, default=12)
    parser.add_argument("--rf_threads", type=int, default=10)
    parser.add_argument("--mlp_threads", type=int, default=6)
    parser.add_argument("--lr_threads", type=int, default=4)  # placeholder (LR doesn't really use it)

    args = parser.parse_args()

    stage5_dir = Path(args.stage5_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # load stage5
    log("[INFO] Loading Stage-5 outputs")
    X_path = stage5_dir / "stage5_X_pca.npy"
    y_path = stage5_dir / "stage5_y_perturbation.npy"
    map_path = stage5_dir / "stage5_label_map.tsv"

    if not X_path.exists() or not y_path.exists() or not map_path.exists():
        raise RuntimeError(f"Missing Stage5 files in {stage5_dir}")

    X = np.load(X_path)
    y = np.load(y_path)
    label_map = read_label_map(map_path)

    n_classes = int(len(np.unique(y)))
    log(f"[INFO] X={X.shape} y={y.shape} classes={n_classes}")

    # split
    log("[INFO] Splitting train/val/test stratified")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, args.test_size, args.val_size, args.seed
    )

    split_df = summarize_split(y_train, y_val, y_test, label_map)
    save_tsv(split_df, out_dir / "stage6_split_summary.tsv")
    log(f"[INFO] [OK] split summary -> {out_dir / 'stage6_split_summary.tsv'}")

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, out_dir / "stage6_scaler.joblib")

    # model selection
    if args.models.strip().lower() == "all":
        selected = ["logreg", "rf", "mlp", "xgb"]
    else:
        selected = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        # allow aliases
        alias = {"lr": "logreg"}
        selected = [alias.get(m, m) for m in selected]

    # filter valid
    valid = {"logreg", "rf", "mlp", "xgb"}
    selected = [m for m in selected if m in valid]
    if not selected:
        raise RuntimeError("No valid models selected. Use: all OR xgb,rf,logreg,mlp")

    log(f"[INFO] Models selected: {selected}")
    log(f"[INFO] Parallel mode: {args.parallel}")

    payload = dict(
        X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        n_classes=n_classes,
        seed=args.seed,
        topk=args.topk_acc,
        out_dir=str(out_dir),
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        lr_solver=args.lr_solver,
        lr_C=args.lr_C,
        lr_max_iter=args.lr_max_iter,
        mlp_epochs=args.mlp_epochs,
        xgb_threads=args.xgb_threads,
        rf_threads=args.rf_threads,
        mlp_threads=args.mlp_threads,
        lr_threads=args.lr_threads,
    )

    results = []

    if args.parallel and len(selected) > 1:
        # parallel across models
        log("[INFO] Running model training in parallel processes")
        with ProcessPoolExecutor(max_workers=len(selected)) as ex:
            futs = {ex.submit(train_single_model, m, payload): m for m in selected}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for m in selected:
            results.append(train_single_model(m, payload))

    # gather metrics
    metrics_val = []
    metrics_test = []

    # choose best
    best_model_name = None
    best_acc = -1
    best_pred = None
    best_true = y_test
    best_proba = None

    for model_name, model_obj, m_val, m_test, y_pred_test, proba_test in results:
        metrics_val.append(m_val)
        metrics_test.append(m_test)

        if m_test["accuracy"] > best_acc:
            best_acc = m_test["accuracy"]
            best_model_name = model_name
            best_pred = y_pred_test
            best_proba = proba_test

        # save models
        if model_name == "logreg":
            joblib.dump(model_obj, out_dir / "stage6_lr_model.joblib")
        elif model_name == "rf":
            joblib.dump(model_obj, out_dir / "stage6_rf_model.joblib")
        elif model_name == "xgb":
            model_obj.save_model(out_dir / "stage6_xgb_model.json")
        elif model_name == "mlp":
            # stage6_mlp_model.pt already saved inside training loop
            pass

    df_val = pd.DataFrame(metrics_val).sort_values("accuracy", ascending=False)
    df_test = pd.DataFrame(metrics_test).sort_values("accuracy", ascending=False)

    save_tsv(df_val, out_dir / "stage6_metrics_val.tsv")
    save_tsv(df_test, out_dir / "stage6_metrics_test.tsv")

    log(f"[INFO] [OK] val metrics -> {out_dir / 'stage6_metrics_val.tsv'}")
    log(f"[INFO] [OK] test metrics -> {out_dir / 'stage6_metrics_test.tsv'}")

    log(f"[INFO] Best model on test: {best_model_name} (acc={best_acc:.4f})")

    # confusion
    plot_confusion_topk(
        y_true=best_true,
        y_pred=best_pred,
        label_map=label_map,
        outpath=out_dir / "stage6_confusion_topk.png",
        topk=args.topk_confmat
    )
    log(f"[INFO] [OK] confusion plot -> {out_dir / 'stage6_confusion_topk.png'}")

    # PRF plot
    plot_prf_bar(df_test, out_dir / "stage6_prf_barplot_test.png")
    log(f"[INFO] [OK] PR/F1 plot -> {out_dir / 'stage6_prf_barplot_test.png'}")

    # topK plot
    plot_topk_bar(df_test, args.topk_acc, out_dir / "stage6_topk_barplot_test.png")
    log(f"[INFO] [OK] TopK plot -> {out_dir / 'stage6_topk_barplot_test.png'}")

    # per-class report for BEST model
    df_report = per_class_report(best_true, best_pred, label_map)
    save_tsv(df_report, out_dir / "stage6_per_class_report_test.tsv")
    log(f"[INFO] [OK] per-class report -> {out_dir / 'stage6_per_class_report_test.tsv'}")

    # predictions table
    pred_df = pd.DataFrame({"y_true": best_true, "y_pred": best_pred})
    save_tsv(pred_df, out_dir / "stage6_test_predictions.tsv")
    log(f"[INFO] [OK] predictions -> {out_dir / 'stage6_test_predictions.tsv'}")

    log("[INFO] [DONE] Stage-6 complete")


if __name__ == "__main__":
    main()

