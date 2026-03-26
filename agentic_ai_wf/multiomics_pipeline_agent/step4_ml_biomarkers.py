import os
import json
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

# Optional: XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Optional: SciPy for t-tests (needed for p-values/FDR + boxplots)
try:
    from scipy.stats import ttest_ind  # type: ignore

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def split_feature_name(f: str):
    """
    Utility: split layer & original feature from colname.
    feature names are like "genomics__TP53", "transcriptomics__BRCA1", etc.
    """
    if "__" in f:
        layer, orig = f.split("__", 1)
    else:
        layer, orig = "unknown", f
    return layer, orig


def run_step4(base_dir: str) -> Dict[str, str]:
    """
    STEP 4 — Adaptive ML-based Biomarker Discovery
    (works with and without labels)

    Generalized version of your notebook Step 4.

    Parameters
    ----------
    base_dir : str
        Same base directory used in previous steps.

    Returns
    -------
    dict with:
      - step_dir: path to step_4_ml_biomarkers
      - ml_summary: path to ml_summary.json
    """
    # ---------------- Paths ----------------
    step3_dir = os.path.join(base_dir, "step_3_integration")
    step4_dir = os.path.join(base_dir, "step_4_ml_biomarkers")
    plots_dir = os.path.join(step4_dir, "plots")
    boxplots_dir = os.path.join(plots_dir, "boxplots")

    os.makedirs(step4_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(boxplots_dir, exist_ok=True)

    # ---------------- Load integration summary and data ----------------
    summary_path = os.path.join(step3_dir, "integration_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"integration_summary.json not found at {summary_path}. "
            "Make sure Step 3 completed successfully."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        integration_summary = json.load(f)

    print("✅ Loaded integration summary:")
    print(json.dumps(integration_summary, indent=2))

    X_path = os.path.join(step3_dir, integration_summary["ml_matrix_file"])
    y_path = os.path.join(step3_dir, integration_summary["labels_file"])
    featmap_path = os.path.join(step3_dir, integration_summary["feature_map_file"])

    X = pd.read_parquet(X_path)
    y = pd.read_csv(y_path, index_col=0)["label"].astype(str)
    feature_map = pd.read_csv(featmap_path)

    # Align X and y (just in case)
    X = X.loc[y.index]
    print(f"\n📐 ML matrix loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print("Label distribution:")
    print(y.value_counts())

    # ---------------- Decide supervised vs unsupervised ----------------
    MIN_SAMPLES_PER_CLASS = 10

    label_counts = y.value_counts(dropna=False)
    unique_labels = label_counts.index.tolist()
    n_classes = len(unique_labels)

    print("\n🔍 Label inspection:")
    print("Unique labels:", unique_labels)
    print("Counts per label:")
    print(label_counts)

    use_supervised = False
    reason_unsupervised = None

    if n_classes < 2:
        use_supervised = False
        reason_unsupervised = "Only one label present (no class contrast)."
    elif label_counts.min() < MIN_SAMPLES_PER_CLASS:
        use_supervised = False
        reason_unsupervised = (
            f"At least one class has < {MIN_SAMPLES_PER_CLASS} samples "
            f"(min={label_counts.min()})."
        )
    else:
        use_supervised = True

    print("\n🧠 Mode decision:")
    if use_supervised:
        print("  → Using SUPERVISED ML for biomarker discovery.")
    else:
        print("  → Using UNSUPERVISED feature ranking.")
        print("    Reason:", reason_unsupervised or "No usable labels.")

    # ============================================================
    # UNSUPERVISED BRANCH
    # ============================================================
    if not use_supervised:
        print("\n🚀 Running unsupervised feature ranking...")

        # Simple variance + MAD-based score
        variances = X.var(axis=0)
        mad = (X - X.median(axis=0)).abs().median(axis=0)

        var_norm = (variances - variances.min()) / (
            variances.max() - variances.min() + 1e-9
        )
        mad_norm = (mad - mad.min()) / (mad.max() - mad.min() + 1e-9)
        combined_score = 0.6 * var_norm + 0.4 * mad_norm

        scores_df = pd.DataFrame(
            {
                "feature": X.columns,
                "variance": variances.values,
                "mad": mad.values,
                "score": combined_score.values,
            }
        )

        layers = []
        orig_feats = []
        for f in scores_df["feature"]:
            L, o = split_feature_name(f)
            layers.append(L)
            orig_feats.append(o)
        scores_df["layer"] = layers
        scores_df["original_feature"] = orig_feats

        scores_df = scores_df.sort_values("score", ascending=False).reset_index(
            drop=True
        )

        # Save unsupervised ranking
        top_biomarkers_path = os.path.join(step4_dir, "top_biomarkers_unsupervised.csv")
        scores_df.to_csv(top_biomarkers_path, index=False)
        print(f"💾 Saved unsupervised biomarker ranking → {top_biomarkers_path}")

        # Plot top 20
        topN = 20
        top_df = scores_df.head(topN)

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=top_df,
            x="score",
            y="feature",
            hue="layer",
            dodge=False,
        )
        plt.title(f"Top {topN} features (unsupervised score)")
        plt.tight_layout()
        out_plot = os.path.join(plots_dir, "Top_features_unsupervised.png")
        plt.savefig(out_plot, dpi=300)
        plt.close()
        print("📊 Saved unsupervised ranking plot →", out_plot)

        summary = {
            "mode": "unsupervised",
            "reason": reason_unsupervised,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "top_biomarkers_file": os.path.basename(top_biomarkers_path),
        }
        with open(os.path.join(step4_dir, "ml_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n✅ STEP 4 complete (UNSUPERVISED branch).")

        return {
            "step_dir": step4_dir,
            "ml_summary": os.path.join(step4_dir, "ml_summary.json"),
            "mode": "unsupervised",
        }

    # ============================================================
    # SUPERVISED BRANCH
    # ============================================================
    print("\n🚀 Running supervised ML-based biomarker discovery...")

    # 1) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 2) Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns,
    )

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_enc,
        test_size=0.25,
        stratify=y_enc,
        random_state=42,
    )

    # 4) Define models + grids (with class balancing)
    scoring_auc = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_specs = {}

    # Logistic Regression (L2)
    model_specs["LogReg_L2"] = {
        "estimator": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
    }

    # ElasticNet-penalty Logistic Regression
    model_specs["ElasticNet"] = {
        "estimator": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
    }

    # Random Forest
    model_specs["RandomForest"] = {
        "estimator": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "param_grid": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5],
        },
    }

    # XGBoost (optional)
    if HAS_XGB:
        if n_classes == 2:
            pos_ratio = (y_train == 1).sum()
            neg_ratio = (y_train == 0).sum()
            spw = neg_ratio / max(pos_ratio, 1)
        else:
            spw = 1.0

        model_specs["XGBoost"] = {
            "estimator": XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                tree_method="hist",
                random_state=42,
                scale_pos_weight=spw,
            ),
            "param_grid": {
                "max_depth": [3, 5],
                "min_child_weight": [1, 5],
            },
        }

    # 5) Train models with GridSearchCV
    results = {}
    best_models = {}
    feat_importances = {}
    roc_data = {}

    for name, spec in model_specs.items():
        print(f"\n🔧 Grid search for {name}...")

        grid = GridSearchCV(
            spec["estimator"],
            spec["param_grid"],
            cv=kf,
            scoring=scoring_auc,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_clf = grid.best_estimator_
        best_models[name] = best_clf

        print(f"  Best params for {name}: {grid.best_params_}")
        print(f"  Best CV {scoring_auc}: {grid.best_score_:.3f}")

        prob_test = best_clf.predict_proba(X_test)
        if n_classes > 2:
            auc_test = roc_auc_score(y_test, prob_test, multi_class="ovr")
        else:
            auc_test = roc_auc_score(y_test, prob_test[:, 1])

        y_pred = best_clf.predict(X_test)
        average = "binary" if n_classes == 2 else "macro"

        acc_test = accuracy_score(y_test, y_pred)
        prec_test = precision_score(
            y_test, y_pred, average=average, zero_division=0
        )
        rec_test = recall_score(
            y_test, y_pred, average=average, zero_division=0
        )
        f1_test = f1_score(y_test, y_pred, average=average, zero_division=0)

        results[name] = {
            "cv_best_auc": float(grid.best_score_),
            "test_auc": float(auc_test),
            "test_accuracy": float(acc_test),
            "test_precision": float(prec_test),
            "test_recall": float(rec_test),
            "test_f1": float(f1_test),
        }
        print(
            f"  {name}: Test AUC={auc_test:.3f}, "
            f"Acc={acc_test:.3f}, Prec={prec_test:.3f}, "
            f"Rec={rec_test:.3f}, F1={f1_test:.3f}"
        )

        # ROC curve data (binary only)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, prob_test[:, 1])
            roc_data[name] = (fpr, tpr, auc_test)

        # Feature importance
        if isinstance(best_clf, LogisticRegression):
            coefs = np.abs(best_clf.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)
            feat_importances[name] = pd.Series(coefs, index=X.columns)
        elif hasattr(best_clf, "feature_importances_"):
            feat_importances[name] = pd.Series(
                best_clf.feature_importances_, index=X.columns
            )
        else:
            print(f"  ⚠️ {name}: no feature_importances_ available.")
            feat_importances[name] = pd.Series(
                np.zeros(X.shape[1]), index=X.columns
            )

    # 6) Save performance metrics
    perf_rows = []
    for name, stats in results.items():
        row = {"model": name}
        row.update(stats)
        perf_rows.append(row)
    perf_df = pd.DataFrame(perf_rows)
    perf_path = os.path.join(step4_dir, "model_performance_metrics.csv")
    perf_df.to_csv(perf_path, index=False)
    print(f"\n💾 Saved model performance metrics → {perf_path}")

    # 7) ROC curves plot (binary only)
    if n_classes == 2 and len(roc_data) > 0:
        plt.figure(figsize=(6, 5))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (held-out test set)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_plot_path = os.path.join(plots_dir, "ROC_curves.png")
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        print(f"📊 Saved ROC curves plot → {roc_plot_path}")
    else:
        print("⚠️ Skipping ROC curves (requires binary labels).")

    # 8) Top 10 biomarkers per model
    per_model_rows = []
    for name, imp_series in feat_importances.items():
        imp_sorted = imp_series.sort_values(ascending=False).head(10)
        for feat, importance in imp_sorted.items():
            layer, orig = split_feature_name(feat)
            per_model_rows.append(
                {
                    "model": name,
                    "feature": feat,
                    "layer": layer,
                    "original_feature": orig,
                    "importance": float(importance),
                }
            )

    top10_per_model_df = pd.DataFrame(per_model_rows)
    top10_path = os.path.join(step4_dir, "top10_biomarkers_per_model.csv")
    top10_per_model_df.to_csv(top10_path, index=False)
    print(f"💾 Saved top 10 biomarkers per model → {top10_path}")

    # 9) Build importance table across models
    imp_df = pd.DataFrame(feat_importances)
    imp_df["mean_importance"] = imp_df.mean(axis=1)

    best_model_name = max(results, key=lambda m: results[m]["test_auc"])
    print(f"\n🏆 Best model by test AUC: {best_model_name}")

    # 9a) BEST-MODEL-based biomarker ranking
    biomarker_best_rows = []
    for feat_name, row in imp_df.iterrows():
        layer, orig = split_feature_name(feat_name)
        entry = {
            "feature": feat_name,
            "layer": layer,
            "original_feature": orig,
            "importance_best_model": float(row[best_model_name]),
            "mean_importance_consensus": float(row["mean_importance"]),
        }
        for m in feat_importances.keys():
            entry[f"importance_{m}"] = float(row[m])
        biomarker_best_rows.append(entry)

    biomarker_best_df = pd.DataFrame(biomarker_best_rows)
    biomarker_best_df = biomarker_best_df.sort_values(
        "importance_best_model", ascending=False
    ).reset_index(drop=True)

    top_biomarkers_best_path = os.path.join(step4_dir, "top_biomarkers_supervised.csv")
    biomarker_best_df.to_csv(top_biomarkers_best_path, index=False)
    print(f"💾 Saved BEST-MODEL supervised biomarker ranking → {top_biomarkers_best_path}")

    # 9b) CONSENSUS biomarker ranking
    biomarker_consensus_rows = []
    for feat_name, row in imp_df.iterrows():
        layer, orig = split_feature_name(feat_name)
        entry = {
            "feature": feat_name,
            "layer": layer,
            "original_feature": orig,
            "mean_importance": float(row["mean_importance"]),
        }
        for m in feat_importances.keys():
            entry[f"importance_{m}"] = float(row[m])
        biomarker_consensus_rows.append(entry)

    biomarker_consensus_df = pd.DataFrame(biomarker_consensus_rows)
    biomarker_consensus_df = biomarker_consensus_df.sort_values(
        "mean_importance", ascending=False
    ).reset_index(drop=True)

    top_biomarkers_consensus_path = os.path.join(
        step4_dir, "top_biomarkers_consensus.csv"
    )
    biomarker_consensus_df.to_csv(top_biomarkers_consensus_path, index=False)
    print(f"💾 Saved CONSENSUS biomarker ranking → {top_biomarkers_consensus_path}")

    # 10) Class-specific stats + boxplots (BEST model)
    if n_classes == 2 and HAS_SCIPY:
        print("\n📦 Computing class-specific stats and boxplots for top biomarkers (BEST model)...")

        label_a, label_b = unique_labels[0], unique_labels[1]

        features = X.columns
        means_a = []
        means_b = []
        pvals = []

        for feat in features:
            vals_a = X.loc[y == label_a, feat]
            vals_b = X.loc[y == label_b, feat]
            means_a.append(vals_a.mean())
            means_b.append(vals_b.mean())

            _, p = ttest_ind(
                vals_a,
                vals_b,
                equal_var=False,
                nan_policy="omit",
            )
            pvals.append(p)

        means_a = pd.Series(means_a, index=features)
        means_b = pd.Series(means_b, index=features)
        pvals = pd.Series(pvals, index=features)

        # Benjamini–Hochberg FDR
        m = len(pvals)
        p_sorted = pvals.sort_values()
        ranks = np.arange(1, m + 1)
        bh = p_sorted * m / ranks
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        fdr_series = pd.Series(bh, index=p_sorted.index)

        # Which class has higher mean?
        higher_in_series = pd.Series(
            np.where(means_a > means_b, label_a, label_b),
            index=features,
        )

        stats_df = pd.DataFrame(
            {
                f"mean_{label_a}": means_a,
                f"mean_{label_b}": means_b,
                "p_value": pvals,
                "fdr_bh": fdr_series,
                "higher_in": higher_in_series,
            }
        ).reset_index().rename(columns={"index": "feature"})

        biomarker_best_df = biomarker_best_df.merge(
            stats_df,
            on="feature",
            how="left",
        )

        stats_path = os.path.join(step4_dir, "top_biomarkers_with_stats.csv")
        biomarker_best_df.to_csv(stats_path, index=False)
        print(f"💾 Saved BEST-model biomarker stats with p-values/FDR → {stats_path}")

        # Boxplots for top K biomarkers
        topK_box = 10
        top_feats = biomarker_best_df.head(topK_box)["feature"].tolist()

        for feat in top_feats:
            layer, orig = split_feature_name(feat)
            row = biomarker_best_df.loc[
                biomarker_best_df["feature"] == feat
            ].iloc[0]
            fdr_val = row["fdr_bh"]
            higher = row["higher_in"]

            df_plot = pd.DataFrame(
                {
                    "value": X[feat],
                    "label": y,
                }
            )

            plt.figure(figsize=(4, 5))
            palette = ["#077A7D", "#5E936C"]
            sns.boxplot(
                data=df_plot,
                x="label",
                y="value",
                palette=palette,
            )

            full_name = f"{layer}_{orig}"
            plt.title(f"{full_name}\nFDR: {fdr_val:.3g}, higher in: {higher}")
            plt.xlabel("Label")
            plt.ylabel("Normalized value")
            plt.tight_layout()

            safe_name = full_name.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(boxplots_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

        print(f"📊 Saved BEST-model boxplots in: {boxplots_dir}")

    else:
        if not HAS_SCIPY:
            print("⚠️ SciPy not available → skipping p-values/FDR + boxplots.")
        elif n_classes != 2:
            print("⚠️ Boxplots with case vs control implemented only for binary labels.")

    # 11) Plot top 20 features by BEST-model importance
    topN = 20
    top_df = biomarker_best_df.head(topN)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top_df,
        x="importance_best_model",
        y="feature",
        hue="layer",
        dodge=False,
    )
    plt.title(f"Top {topN} features (BEST model importance)")
    plt.tight_layout()
    out_plot = os.path.join(plots_dir, "Top_features_supervised.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print("📊 Saved supervised ranking plot (BEST model) →", out_plot)

    # 12) ML summary JSON
    ml_summary = {
        "mode": "supervised",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "classes": [str(c) for c in unique_labels],
        "results_per_model": results,
        "best_model": best_model_name,
        "top_biomarkers_best_model_file": os.path.basename(top_biomarkers_best_path),
        "top_biomarkers_consensus_file": os.path.basename(
            top_biomarkers_consensus_path
        ),
        "performance_metrics_file": os.path.basename(perf_path),
        "top10_per_model_file": os.path.basename(top10_path),
        "roc_plot_file": "ROC_curves.png" if n_classes == 2 else None,
        "boxplots_dir": os.path.relpath(boxplots_dir, step4_dir),
    }
    ml_summary_path = os.path.join(step4_dir, "ml_summary.json")
    with open(ml_summary_path, "w", encoding="utf-8") as f:
        json.dump(ml_summary, f, indent=2)

    print("\n✅ STEP 4 complete (SUPERVISED branch).")

    return {
        "step_dir": step4_dir,
        "ml_summary": ml_summary_path,
        "mode": "supervised",
    }
