"""
STEP 5 — Extended Cross-Omics & Network Biology (Data-Driven)

Generalized version of the notebook Step 5.
"""

import os
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Try to import networkx for network plots (optional)
try:
    import networkx as nx  # type: ignore

    HAS_NX = True
except ImportError:
    HAS_NX = False


def split_feature_name(f: str):
    """Split 'layer__original' into (layer, original_feature)."""
    if "__" in f:
        layer, orig = f.split("__", 1)
    else:
        layer, orig = "unknown", f
    return layer, orig


def run_step5(base_dir: str) -> Dict[str, Optional[str]]:
    """
    Run Step 5 — Extended Cross-Omics & Network Biology.

    Parameters
    ----------
    base_dir : str
        Same base directory used for steps 1–4.

    Returns
    -------
    dict with key paths, e.g.
      - step_dir
      - evidence_table
      - summary
    """

    # ---------------- Paths ----------------
    step2_dir = os.path.join(base_dir, "step_2_preprocessing")
    step3_dir = os.path.join(base_dir, "step_3_integration")
    step4_dir = os.path.join(base_dir, "step_4_ml_biomarkers")
    step5_dir = os.path.join(base_dir, "step_5_crossomics")
    plots5_dir = os.path.join(step5_dir, "plots")

    os.makedirs(step5_dir, exist_ok=True)
    os.makedirs(plots5_dir, exist_ok=True)

    # ---------------- Load integration summary ----------------
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

    # ---------------- Load ML matrix & feature map ----------------
    X_path = os.path.join(step3_dir, integration_summary["ml_matrix_file"])
    featmap_path = os.path.join(step3_dir, integration_summary["feature_map_file"])

    X_ml = pd.read_parquet(X_path)
    X_ml.index = X_ml.index.astype(str)
    print(f"\n📐 ML core matrix: {X_ml.shape[0]} samples × {X_ml.shape[1]} features")

    feature_map = pd.read_csv(featmap_path)  # currently not used but kept for parity

    # ---------------- Load biomarker ranking (supervised or unsupervised) ----------------
    bm_super_path = os.path.join(step4_dir, "top_biomarkers_supervised.csv")
    bm_unsup_path = os.path.join(step4_dir, "top_biomarkers_unsupervised.csv")

    if os.path.exists(bm_super_path):
        bm = pd.read_csv(bm_super_path)
        bm_mode = "supervised"
        print(f"🔎 Using supervised biomarkers from: {bm_super_path}")
        if "importance_best_model" in bm.columns:
            bm["ml_importance_raw"] = bm["importance_best_model"].astype(float)
        elif "mean_importance" in bm.columns:
            bm["ml_importance_raw"] = bm["mean_importance"].astype(float)
        else:
            bm["ml_importance_raw"] = np.nan
    elif os.path.exists(bm_unsup_path):
        bm = pd.read_csv(bm_unsup_path)
        bm_mode = "unsupervised"
        print(f"🔎 Using unsupervised biomarkers from: {bm_unsup_path}")
        if "score" in bm.columns:
            bm["ml_importance_raw"] = bm["score"].astype(float)
        else:
            bm["ml_importance_raw"] = np.nan
    else:
        raise FileNotFoundError("No biomarker ranking file found in Step 4.")

    if "feature" not in bm.columns:
        raise ValueError("Biomarker file must contain a 'feature' column.")

    # Restrict to biomarkers that are in the ML matrix
    bm = bm[bm["feature"].isin(X_ml.columns)].copy()

    TOP_N_BM = 100
    bm = bm.head(TOP_N_BM).reset_index(drop=True)
    print(f"📌 Considering top {len(bm)} biomarkers for extended validation.")

    # ---------------- Helper: split "layer__original" ----------------
    bm["primary_layer"], bm["primary_original_feature"] = zip(
        *bm["feature"].map(split_feature_name)
    )

    # Normalize ML importance to [0,1]
    if bm["ml_importance_raw"].notna().any():
        mi = bm["ml_importance_raw"]
        mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-9)
        bm["ml_importance_norm"] = mi_norm
    else:
        bm["ml_importance_norm"] = np.nan

    # ---------------- Load normalized single-layer matrices from Step 2 ----------------
    norm_manifest_path = os.path.join(step2_dir, "normalization_summary.json")
    with open(norm_manifest_path, "r", encoding="utf-8") as f:
        norm_manifest = json.load(f)

    layer_mats = {}
    for lname, info in norm_manifest["layers"].items():
        fpath = os.path.join(step2_dir, info["file"])
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
            layer_mats[lname] = df
            print(f"[{lname}] matrix: {df.shape[0]} samples × {df.shape[1]} features")
        else:
            print(f"⚠️ Missing normalized file for layer '{lname}', skipping.")

    layers_available = list(layer_mats.keys())
    print("\n✅ Layers available for cross-omics check:", layers_available)

    # ---------------- Cross-omics presence & variance ----------------
    records = []

    for _, row in bm.iterrows():
        feat_full = row["feature"]
        prim_layer = row["primary_layer"]
        orig_feat = row["primary_original_feature"]

        presence = {}
        variance = {}

        for lname, mat in layer_mats.items():
            present = orig_feat in mat.columns
            presence[lname] = present
            variance[lname] = float(mat[orig_feat].var()) if present else None

        n_layers_present = sum(bool(v) for v in presence.values())

        if len(layer_mats) > 1:
            cross_score = (n_layers_present - 1) / max(1, len(layer_mats) - 1)
            cross_score = max(0.0, min(1.0, cross_score))
        else:
            cross_score = 0.0

        rec = {
            "feature_full": feat_full,
            "primary_layer": prim_layer,
            "primary_original_feature": orig_feat,
            "n_layers_present": int(n_layers_present),
            "cross_omics_support_score": float(cross_score),
        }

        for lname in layer_mats.keys():
            rec[f"present_in_{lname}"] = bool(presence[lname])
            rec[f"variance_in_{lname}"] = variance[lname]

        records.append(rec)

    val_df = pd.DataFrame(records)

    # ---------------- Data-driven correlation network among biomarkers ----------------
    print("\n🔗 Building data-driven correlation network among top biomarkers...")

    bm_features = bm["feature"].tolist()
    X_bm = X_ml[bm_features].copy()

    corr = X_bm.corr(method="pearson")
    abs_corr = np.abs(corr)

    CORR_THRESHOLD = 0.6
    print(f"Using |r| ≥ {CORR_THRESHOLD} as edge threshold for network.")

    degree = (abs_corr >= CORR_THRESHOLD).sum(axis=0) - 1
    degree = pd.Series(degree, index=bm_features, name="network_degree")

    deg_norm = (degree - degree.min()) / (degree.max() - degree.min() + 1e-9)
    deg_norm = deg_norm.reindex(bm["feature"])
    bm["network_degree"] = degree.reindex(bm["feature"]).fillna(0).values
    bm["network_degree_norm"] = deg_norm.fillna(0).values

    # ---------------- Data-driven modules via hierarchical clustering ----------------
    dist = 1.0 - abs_corr
    np.fill_diagonal(dist.values, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")

    MODULE_THRESHOLD = 0.7
    clusters = fcluster(Z, t=MODULE_THRESHOLD, criterion="distance")
    module_ids = pd.Series(clusters, index=bm_features, name="module_id")

    bm["module_id"] = module_ids.reindex(bm["feature"]).astype(int).values

    print(f"Identified {bm['module_id'].nunique()} modules among top biomarkers.")

    # ---------------- Combine all evidence into extended table ----------------
    extended_records = []

    for _, row in bm.iterrows():
        feat_full = row["feature"]
        prim_layer = row["primary_layer"]
        orig_feat = row["primary_original_feature"]

        v_row = val_df.loc[val_df["feature_full"] == feat_full].iloc[0]

        rec = {
            "feature_full": feat_full,
            "primary_layer": prim_layer,
            "primary_original_feature": orig_feat,
            "ml_importance_raw": float(row["ml_importance_raw"])
            if not np.isnan(row["ml_importance_raw"])
            else None,
            "ml_importance_norm": float(row["ml_importance_norm"])
            if not np.isnan(row["ml_importance_norm"])
            else None,
            "cross_omics_support_score": float(
                v_row["cross_omics_support_score"]
            ),
            "n_layers_present": int(v_row["n_layers_present"]),
            "network_degree": float(row["network_degree"]),
            "network_degree_norm": float(row["network_degree_norm"]),
            "module_id": int(row["module_id"]),
        }

        for lname in layer_mats.keys():
            rec[f"present_in_{lname}"] = bool(v_row[f"present_in_{lname}"])
            rec[f"variance_in_{lname}"] = v_row[f"variance_in_{lname}"]

        extended_records.append(rec)

    extended_df = pd.DataFrame(extended_records)

    # Final extended biology score = mean of normalized components
    components = []
    if extended_df["ml_importance_norm"].notna().any():
        components.append(extended_df["ml_importance_norm"].fillna(0.0))
    components.append(extended_df["cross_omics_support_score"].fillna(0.0))
    components.append(extended_df["network_degree_norm"].fillna(0.0))

    if len(components) > 0:
        stacked = np.vstack([c.values for c in components])
        final_score = stacked.mean(axis=0)
    else:
        final_score = np.zeros(len(extended_df))

    extended_df["extended_biology_score"] = final_score

    extended_df = extended_df.sort_values(
        "extended_biology_score", ascending=False
    ).reset_index(drop=True)

    out_csv = os.path.join(step5_dir, "extended_biomarker_evidence.csv")
    extended_df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved extended biomarker evidence table → {out_csv}")

    # ---------------- PLOT 1: Evidence heatmap ----------------
    TOP_HEAT = min(40, len(extended_df))
    heat_subset = extended_df.head(TOP_HEAT).copy()

    evidence_cols = []
    if "ml_importance_norm" in extended_df.columns:
        evidence_cols.append("ml_importance_norm")
    evidence_cols += [
        "cross_omics_support_score",
        "network_degree_norm",
    ]

    evidence_mat = heat_subset[evidence_cols].copy()
    evidence_mat.index = heat_subset["feature_full"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        evidence_mat,
        cmap="viridis",
        cbar_kws={"label": "Normalized evidence (0–1)"},
    )
    plt.title(f"Evidence integration for top {TOP_HEAT} biomarkers")
    plt.xlabel("Evidence type")
    plt.ylabel("Biomarker")
    plt.tight_layout()
    heat_evidence_path = os.path.join(
        plots5_dir, "Biomarker_evidence_heatmap.png"
    )
    plt.savefig(heat_evidence_path, dpi=300)
    plt.close()
    print("📊 Saved evidence integration heatmap →", heat_evidence_path)

    # ---------------- PLOT 2: Correlation heatmap among biomarkers ----------------
    TOP_CORR = min(20, len(bm_features))
    corr_subset = corr.iloc[:TOP_CORR, :TOP_CORR].copy()
    corr_subset.index = bm_features[:TOP_CORR]
    corr_subset.columns = bm_features[:TOP_CORR]

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_subset,
        cmap="coolwarm",
        center=0.0,
        square=True,
        xticklabels=False,
    )
    plt.title(f"Biomarker–biomarker correlation (top {TOP_CORR})")
    plt.ylabel("Biomarkers")
    plt.xlabel("Biomarkers")
    plt.tight_layout()
    corr_heat_path = os.path.join(
        plots5_dir, "Biomarker_correlation_heatmap.png"
    )
    plt.savefig(corr_heat_path, dpi=300)
    plt.close()
    print("📊 Saved biomarker correlation heatmap →", corr_heat_path)

    # ---------------- PLOT 3: Data-driven network graph ----------------
    net_path: Optional[str] = None
    if HAS_NX:
        print("\n🔧 Building network graph (networkx)...")
        G = nx.Graph()

        # Nodes: top 50 by extended biology score
        for _, row in extended_df.head(50).iterrows():
            G.add_node(
                row["feature_full"],
                layer=row["primary_layer"],
                score=row["extended_biology_score"],
            )

        # Edges for |corr| ≥ threshold
        for i, fi in enumerate(bm_features):
            if fi not in G:
                continue
            for j in range(i + 1, len(bm_features)):
                fj = bm_features[j]
                if fj not in G:
                    continue
                r = corr.loc[fi, fj]
                if np.abs(r) >= CORR_THRESHOLD:
                    G.add_edge(fi, fj, weight=float(r))

        pos = nx.spring_layout(G, k=0.4, seed=42)

        layers_unique = extended_df["primary_layer"].unique().tolist()
        palette = sns.color_palette("Set2", n_colors=len(layers_unique))
        layer_to_color = {L: palette[i] for i, L in enumerate(layers_unique)}

        node_colors = [layer_to_color[G.nodes[n]["layer"]] for n in G.nodes()]
        node_sizes = [300 + 1200 * G.nodes[n]["score"] for n in G.nodes()]

        plt.figure(figsize=(8, 7))
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0)
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9
        )
        top_labels = set(extended_df.head(15)["feature_full"])
        labels = {n: n for n in G.nodes() if n in top_labels}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

        for L, c in layer_to_color.items():
            plt.scatter([], [], c=[c], label=L)
        plt.legend(title="Primary layer", loc="best", fontsize=8)

        plt.title("Data-driven biomarker network (top features)")
        plt.axis("off")
        net_path = os.path.join(
            plots5_dir, "Biomarker_network_data_driven.png"
        )
        plt.tight_layout()
        plt.savefig(net_path, dpi=300)
        plt.close()
        print("📊 Saved biomarker network graph →", net_path)
    else:
        print("⚠️ networkx not installed; skipping network graph plot.")

    # ---------------- Optional: per-biomarker mini-panels ----------------
    TOP_PANEL = min(4, len(extended_df))
    print(
        f"\n🎨 Building per-biomarker mini-panels for top {TOP_PANEL} biomarkers..."
    )

    bm_panel_dir = os.path.join(plots5_dir, "biomarkers")
    os.makedirs(bm_panel_dir, exist_ok=True)

    for _, row in extended_df.head(TOP_PANEL).iterrows():
        feat_full = row["feature_full"]
        prim_layer = row["primary_layer"]
        orig_feat = row["primary_original_feature"]
        mod_id = row["module_id"]
        score = row["extended_biology_score"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            f"{orig_feat} ({feat_full})\n"
            f"Layer={prim_layer}, Module={mod_id}, Extended score={score:.2f}",
            fontsize=12,
        )

        # Panel 1: ML importance
        ax = axes[0, 0]
        if not np.isnan(
            row["ml_importance_norm"]
        ):
            ax.bar([0], [row["ml_importance_norm"]])
            ax.set_xticks([0])
            ax.set_xticklabels(["ML importance (norm)"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Norm. importance")
        else:
            ax.text(
                0.5,
                0.5,
                "No ML importance (unsupervised only).",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

        # Panel 2: cross-omics presence
        ax = axes[0, 1]
        pres_vals = []
        pres_labels = []
        for lname in layer_mats.keys():
            pres_labels.append(lname)
            pres_vals.append(int(row[f"present_in_{lname}"]))
        ax.bar(pres_labels, pres_vals)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Presence (0/1)")
        ax.set_title("Cross-omics presence")

        # Panel 3: distribution in primary layer
        ax = axes[1, 0]
        if (
            prim_layer in layer_mats
            and orig_feat in layer_mats[prim_layer].columns
        ):
            ax.hist(layer_mats[prim_layer][orig_feat], bins=30)
            ax.set_title(f"Distribution in {prim_layer}")
            ax.set_xlabel("Normalized value")
            ax.set_ylabel("Count")
        else:
            ax.text(
                0.5,
                0.5,
                f"{orig_feat} not in {prim_layer} matrix.",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

        # Panel 4: network degree
        ax = axes[1, 1]
        ax.bar(["degree"], [row["network_degree_norm"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Norm. degree")
        ax.set_title("Network centrality")

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        out_path = os.path.join(bm_panel_dir, f"{orig_feat.replace(' ', '_')}_panel.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("  ➜ Saved panel for", orig_feat, "→", out_path)

    # ---------------- Summary JSON ----------------
    summary5 = {
        "biomarker_mode": bm_mode,
        "n_biomarkers_considered": int(len(bm)),
        "layers_used": layers_available,
        "corr_threshold": CORR_THRESHOLD,
        "n_modules": int(extended_df["module_id"].nunique()),
        "evidence_table_file": os.path.basename(out_csv),
        "plots": {
            "evidence_heatmap": os.path.basename(heat_evidence_path),
            "correlation_heatmap": os.path.basename(corr_heat_path),
            "network_graph": os.path.basename(net_path)
            if net_path is not None
            else None,
            "per_biomarker_panels_dir": os.path.basename(bm_panel_dir),
        },
    }

    summary_file = os.path.join(step5_dir, "extended_crossomics_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary5, f, indent=2)

    print("\n✅ STEP 5 (Extended, data-driven biology) complete.")
    print("Summary:")
    print(json.dumps(summary5, indent=2))

    return {
        "step_dir": step5_dir,
        "evidence_table": out_csv,
        "summary": summary_file,
    }
