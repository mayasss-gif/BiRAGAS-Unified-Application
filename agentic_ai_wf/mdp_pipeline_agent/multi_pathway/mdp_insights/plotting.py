from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import InsightsConfig

# Reduce noisy pillow PNG chunk debug logs and layout warnings.
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.ticker").setLevel(logging.INFO)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")
#dynamic plotting

# ---- dynamic label helpers ----
def _estimate_fig_width_from_labels(labels, base: float = 8.0, per_char: float = 0.10, max_w: float = 16.0) -> float:
    """Increase width based on the longest label; caps at max_w."""
    mlen = max((len(str(s)) for s in labels), default=0)
    return min(max(base, base + per_char * mlen), max_w)

def _wrap_text(s: str, width: int = 32) -> str:
    try:
        return "\n".join(textwrap.wrap(str(s), width=width)) or str(s)
    except Exception:
        return str(s)

def _wrap_labels(labels, width: int = 32):
    return [ _wrap_text(str(s), width=width) for s in labels ]


# ---------- figure saving & placeholder ----------

def _save_fig(fig, fname: Path, cfg: InsightsConfig) -> Optional[Path]:
    try:
        fig.tight_layout()
        fig.savefig(fname.with_suffix(".png"), dpi=cfg.dpi, bbox_inches="tight")
        fig.savefig(fname.with_suffix(".svg"), bbox_inches="tight")
        return fname.with_suffix(".png")
    except Exception as e:
        logging.error(f"Failed saving figure {fname}: {e}")
        return None
    finally:
        plt.close(fig)
        

def _plot_message(title: str, message: str, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    try:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis('off')
        ax.set_title(title)
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"_plot_message failed: {e}")
        return None

# ---------- generic plots ----------

def bar_top(df: pd.DataFrame, x: str, y: str, title: str, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return _plot_message(title, "No data available", out, cfg)

        data = df.copy().head(cfg.top_k).iloc[::-1]
        vals = pd.to_numeric(data[y], errors="coerce").fillna(0.0)

        # dynamic width & label wrap
        fig_w = _estimate_fig_width_from_labels(data[x].astype(str))
        labels_wrapped = _wrap_labels(data[x].astype(str), width=32)

        fig_h = max(3.0, 0.42 * len(data))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        ax.barh(labels_wrapped, vals.values)

        # x padding so numbers don't hug the right border
        vmax = float(vals.max()) if len(vals) else 0.0
        ax.set_xlim(0, vmax * 1.12 + (0.02 * (1.0 if vmax == 0 else vmax)))

        ax.set_title(title)
        ax.set_xlabel(y)
        ax.set_ylabel(x)

        # smart value placement: inside when bar wide enough else outside
        # threshold is 8% of (max - min)
        span = max(vmax, 1e-9)
        thresh = 0.08 * span
        for i, v in enumerate(vals.tolist()):
            if v >= thresh:
                ax.text(v - 0.01 * span, i, f"{v:.2f}", va="center", ha="right", color="white", fontsize=9, clip_on=False)
            else:
                ax.text(v + 0.01 * span, i, f"{v:.2f}", va="center", ha="left", fontsize=9, clip_on=False)

        # nicer y tick label font
        ax.tick_params(axis="y", labelsize=9)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"bar_top failed: {e}")
        return _plot_message(title, "Error rendering plot", out, cfg)


def heatmap_matrix(mat: pd.DataFrame, title: str, out: Path, cfg: InsightsConfig,
                   xlabel: str, ylabel: str) -> Optional[Path]:
    try:
        if mat is None or mat.empty:
            return _plot_message(title, "No data available", out, cfg)
        fig, ax = plt.subplots(figsize=(10, max(3, 0.20*mat.shape[0])))
        im = ax.imshow(mat.values, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"heatmap_matrix failed: {e}")
        return _plot_message(title, "Error rendering heatmap", out, cfg)

# ---------- shared plots ----------

def shared_presence_heatmap(mat_any: pd.DataFrame, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    return heatmap_matrix(mat_any, "Shared Pathways (ANY presence)", out, cfg, xlabel="Disease", ylabel="Pathway")

def shared_upset_like(memberships: Dict[str, Set[str]], out: Path, cfg: InsightsConfig) -> Optional[Path]:
    try:
        title = "Top Shared Pathways (# diseases)"
        if not memberships:
            return _plot_message(title, "No shared pathways to display", out, cfg)
        items = sorted(memberships.items(), key=lambda kv: len(kv[1]), reverse=True)[:cfg.top_k]
        labels = [k for k,_ in items]
        sizes = [len(v) for _,v in items]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35*len(items))))
        ax.barh(list(reversed(labels)), list(reversed(sizes)))
        ax.set_title(title)
        ax.set_xlabel("# Diseases sharing pathway")
        ax.set_ylabel("Pathway")
        for i, v in enumerate(reversed(sizes)):
            ax.text(v, i, str(v), va="center", ha="left")
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"shared_upset_like failed: {e}")
        return _plot_message("Top Shared Pathways (# diseases)", "Error rendering chart", out, cfg)

def shared_entity_mix_bars(shared_tbl: pd.DataFrame, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    try:
        title = "Entity-Type Richness per Shared Pathway"
        if shared_tbl is None or shared_tbl.empty:
            return _plot_message(title, "No shared pathways", out, cfg)
        df = shared_tbl.copy().head(cfg.top_k)
        req = {"tf_n","epigenetic_n","metabolite_n","pathway"}
        if not req.issubset(df.columns):
            return _plot_message(title, "Missing columns", out, cfg)

        labels_wrapped = _wrap_labels(df["pathway"].astype(str).tolist(), width=32)
        dfp = df[["tf_n","epigenetic_n","metabolite_n"]].fillna(0).copy().iloc[::-1]
        labels_wrapped = list(reversed(labels_wrapped))  # keep alignment with iloc[::-1]

        fig_w = _estimate_fig_width_from_labels(labels_wrapped)
        fig_h = max(3.2, 0.50 * len(dfp))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        idx = np.arange(len(dfp))
        width = 0.28
        ax.barh(idx - width, dfp["tf_n"].values, height=width, label="TF")
        ax.barh(idx,            dfp["epigenetic_n"].values, height=width, label="Epigenetic")
        ax.barh(idx + width,    dfp["metabolite_n"].values, height=width, label="Metabolites")

        ax.set_yticks(idx)
        ax.set_yticklabels(labels_wrapped)
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.margins(x=0.02)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"shared_entity_mix_bars failed: {e}")
        return _plot_message("Entity-Type Richness per Shared Pathway", "Error rendering chart", out, cfg)

def shared_genes_count_bar(shared_tbl: pd.DataFrame, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    title = "Shared Gene Intersection Size per Pathway"
    if shared_tbl is None or shared_tbl.empty:
        return _plot_message(title, "No shared pathways", out, cfg)
    if "shared_genes_n" not in shared_tbl.columns or "pathway" not in shared_tbl.columns:
        return _plot_message(title, "Missing columns", out, cfg)
    df = shared_tbl[["pathway","shared_genes_n"]].copy().sort_values("shared_genes_n", ascending=False).head(cfg.top_k)
    return bar_top(df, x="pathway", y="shared_genes_n", title=title, out=out, cfg=cfg)

def disease_similarity_heatmap(mat_any: pd.DataFrame, out: Path, cfg: InsightsConfig) -> Optional[Path]:
    title = "Disease–Disease Similarity (Jaccard, ANY pathways)"
    if mat_any is None or mat_any.empty:
        return _plot_message(title, "No presence data", out, cfg)
    try:
        X = mat_any.to_numpy().astype(int)
        inter = X.T @ X
        col_sums = X.sum(axis=0, keepdims=True)
        union = col_sums + col_sums.T - inter
        union[union == 0] = 1
        sim = inter / union
        sim_df = pd.DataFrame(sim, index=mat_any.columns, columns=mat_any.columns)
        return heatmap_matrix(sim_df, title, out, cfg, xlabel="Disease", ylabel="Disease")
    except Exception as e:
        logging.error(f"disease_similarity_heatmap failed: {e}")
        return _plot_message(title, "Error computing similarity", out, cfg)

def pathway_cooccurrence_heatmap(mat_any: pd.DataFrame, shared_tbl: pd.DataFrame,
                                 out: Path, cfg: InsightsConfig) -> Optional[Path]:
    title = "Pathway–Pathway Co-occurrence (# diseases in common)"
    if mat_any is None or mat_any.empty or shared_tbl is None or shared_tbl.empty:
        return _plot_message(title, "No data for co-occurrence", out, cfg)
    try:
        top_pw = (shared_tbl.sort_values("n_diseases", ascending=False)["pathway"]
                          .head(min(cfg.top_k, 30)).tolist())
        sub = mat_any.loc[mat_any.index.intersection(top_pw)]
        if sub.empty:
            return _plot_message(title, "No overlap among selected pathways", out, cfg)
        X = sub.to_numpy().astype(int)
        co = X @ X.T
        co_df = pd.DataFrame(co, index=sub.index, columns=sub.index)
        return heatmap_matrix(co_df, title, out, cfg, xlabel="Pathway", ylabel="Pathway")
    except Exception as e:
        logging.error(f"pathway_cooccurrence_heatmap failed: {e}")
        return _plot_message(title, "Error computing co-occurrence", out, cfg)

# ---------- individual (kept from your base and hardened) ----------
def grouped_entity_mix_per_pathway(mix_df: pd.DataFrame, title: str, out: Path, cfg: InsightsConfig):
    try:
        if mix_df is None or mix_df.empty:
            return _plot_message(title, "No data available", out, cfg)
        cols = {"pathway","tf_n","epigenetic_n","metabolite_n"}
        if not cols.issubset(mix_df.columns):
            return _plot_message(title, "Missing columns", out, cfg)
        df = mix_df[["pathway","tf_n","epigenetic_n","metabolite_n"]].fillna(0).copy().head(cfg.top_k).iloc[::-1]

        labels_wrapped = _wrap_labels(df["pathway"].astype(str).tolist(), width=32)
        fig_w = _estimate_fig_width_from_labels(labels_wrapped)
        fig_h = max(3.2, 0.50 * len(df))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        idx = np.arange(len(df))
        width = 0.28
        ax.barh(idx - width, df["tf_n"].values, height=width, label="TF")
        ax.barh(idx,            df["epigenetic_n"].values, height=width, label="Epigenetic")
        ax.barh(idx + width,    df["metabolite_n"].values, height=width, label="Metabolites")

        ax.set_yticks(idx)
        ax.set_yticklabels(labels_wrapped)
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.margins(x=0.02)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"grouped_entity_mix_per_pathway failed: {e}")
        return _plot_message(title, "Error rendering chart", out, cfg)

def heatmap_entity_mix_per_pathway(mix_df: pd.DataFrame, title: str, out: Path, cfg: InsightsConfig):
    cols = {"pathway","tf_n","epigenetic_n","metabolite_n"}
    if mix_df is None or mix_df.empty or not cols.issubset(mix_df.columns):
        return _plot_message(title, "No data available", out, cfg)
    mat = (mix_df[["pathway","tf_n","epigenetic_n","metabolite_n"]]
           .set_index("pathway"))
    mat.columns = ["TF","Epigenetic","Metabolites"]
    return heatmap_matrix(mat, title, out, cfg, xlabel="Entity Type", ylabel="Pathway")

def bar_genes_per_pathway(genes_df: pd.DataFrame, title: str, out: Path, cfg: InsightsConfig):
    cols = {"pathway","genes_n"}
    if genes_df is None or genes_df.empty or not cols.issubset(genes_df.columns):
        return _plot_message(title, "No data available", out, cfg)
    df = genes_df[["pathway","genes_n"]].copy().sort_values("genes_n", ascending=False).head(cfg.top_k)

    # reuse bar_top behavior by renaming columns
    tmp = df.rename(columns={"pathway":"_x", "genes_n":"_y"})
    return bar_top(tmp, x="_x", y="_y", title=title, out=out, cfg=cfg)


def hist_metric_per_type(df: pd.DataFrame, value_col: str, title: str, out: Path, cfg: InsightsConfig, bins: int = 30):
    try:
        if df is None or df.empty or value_col not in df or "entity_type" not in df:
            return _plot_message(title, "No data available", out, cfg)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for et, sub in df.groupby("entity_type"):
            vals = pd.to_numeric(sub[value_col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.5, label=str(et))
        ax.set_title(title)
        ax.set_xlabel(value_col)
        ax.set_ylabel("Frequency")
        ax.legend(loc="best")
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"hist_metric_per_type failed: {e}")
        return _plot_message(title, "Error rendering histogram", out, cfg)

def box_metric_per_type(df: pd.DataFrame, value_col: str, title: str, out: Path, cfg: InsightsConfig):
    try:
        if df is None or df.empty or value_col not in df or "entity_type" not in df:
            return _plot_message(title, "No data available", out, cfg)
        order = ["tf","epigenetic","metabolites"]
        data = [pd.to_numeric(df.loc[df["entity_type"]==et, value_col], errors="coerce").dropna().values for et in order]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(data, labels=order, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel(value_col)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"box_metric_per_type failed: {e}")
        return _plot_message(title, "Error rendering boxplot", out, cfg)

def scatter_pss_vs_n(top_pw: pd.DataFrame, title: str, out: Path, cfg: InsightsConfig):
    try:
        req = {"pss","n_entities","pathway"}
        if top_pw is None or top_pw.empty or not req.issubset(top_pw.columns):
            return _plot_message(title, "No data available", out, cfg)
        df = top_pw.copy().head(cfg.top_k)
        x = pd.to_numeric(df["n_entities"], errors="coerce").fillna(0.0).values
        y = pd.to_numeric(df["pss"], errors="coerce").fillna(0.0).values
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(x, y)
        for _, row in df.iterrows():
            ax.annotate(str(row["pathway"])[:28], (float(row["n_entities"]), float(row["pss"])),
                        fontsize=8, xytext=(2,2), textcoords="offset points")
        ax.set_xlabel("n_entities")
        ax.set_ylabel("PSS")
        ax.set_title(title)
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"scatter_pss_vs_n failed: {e}")
        return _plot_message(title, "Error rendering scatter", out, cfg)

#new added functions
def shared_up_significance_heatmap(entities_up: pd.DataFrame, out: Path, cfg: InsightsConfig):
    """
    Pathway x Disease matrix of UP significance (sig = -log10(q/p)), with ANY borrowed to UP by config.
    Value = max sig per (pathway, disease) in UP view.
    """
    title = "Shared Directional Significance (UP; ANY→UP)"
    try:
        if entities_up is None or entities_up.empty:
            return _plot_message(title, "No UP (or ANY) data", out, cfg)
        df = (entities_up.groupby(["pathway","disease"])["sig"]
              .max().unstack("disease", fill_value=0.0))
        return heatmap_matrix(df, title, out, cfg, xlabel="Disease", ylabel="Pathway")
    except Exception as e:
        logging.error(f"shared_up_significance_heatmap failed: {e}")
        return _plot_message(title, "Error rendering heatmap", out, cfg)

def shared_pathway_leaderboard(top_tbl: pd.DataFrame, out: Path, cfg: InsightsConfig):
    title = "Pathway Intelligence Leaderboard (Top by PIS)"
    if top_tbl is None or top_tbl.empty:
        return _plot_message(title, "No ranked pathways", out, cfg)
    cols = {"pathway","PIS"}
    if not cols.issubset(top_tbl.columns):
        return _plot_message(title, "Missing PIS columns", out, cfg)
    df = top_tbl[["pathway","PIS"]].copy().sort_values("PIS", ascending=False).head(min(cfg.top_k, 30))
    return bar_top(df, x="pathway", y="PIS", title=title, out=out, cfg=cfg)

def shared_target_volcano(entities_any: pd.DataFrame, etype: str, out: Path, cfg: InsightsConfig):
    """
    x = median log10(OR), y = max sig, size = # diseases (scaled), one chart per entity type.
    """
    title = f"Target Volcano — {etype.upper()}"
    try:
        if entities_any is None or entities_any.empty:
            return _plot_message(title, "No data", out, cfg)
        sub = entities_any[entities_any["entity_type"].eq(etype)].copy()
        if sub.empty:
            return _plot_message(title, "No data for this type", out, cfg)
        agg = (sub.groupby("entity")
                 .agg(med_log10OR=("OR", lambda s: float(np.nanmedian(np.log10(np.clip(pd.to_numeric(s, errors="coerce").values, 1e-12, None))))),
                      max_sig=("sig", lambda s: float(np.nanmax(pd.to_numeric(s, errors="coerce")))),
                      n_diseases=("disease","nunique"))
                 .reset_index())
        if agg.empty:
            return _plot_message(title, "No aggregate rows", out, cfg)

        sz = agg["n_diseases"].fillna(0).astype(float)
        sz = 50.0 + 10.0 * (sz / max(1.0, sz.max()))

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.scatter(agg["med_log10OR"].values, agg["max_sig"].values, s=sz, alpha=0.75)
        ax.set_xlabel("median log10(OR)")
        ax.set_ylabel("max sig (−log10 q/p)")
        ax.set_title(title)
        # light labeling for top points
        top = agg.sort_values(["max_sig","n_diseases","med_log10OR"], ascending=[False, False, False]).head(10)
        for _, r in top.iterrows():
            ax.annotate(str(r["entity"])[:28], (r["med_log10OR"], r["max_sig"]), fontsize=8, xytext=(3,3), textcoords="offset points")
        return _save_fig(fig, out, cfg)
    except Exception as e:
        logging.error(f"shared_target_volcano failed: {e}")
        return _plot_message(title, "Error rendering volcano", out, cfg)
