"""
ACE Causality Figures Module

Generates publication-grade HTML figures from ACE analysis results.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network


# ---------------------------
# IO helpers
# ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns {missing}. Found columns: {df.columns.tolist()}")


def write_html(fig: go.Figure, outpath: str, title: str | None = None) -> None:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=25, t=70 if title else 45, b=45),
        title=title
    )
    fig.write_html(outpath, include_plotlyjs="cdn", full_html=True)


# ---------------------------
# Load and normalize inputs
# ---------------------------

def load_ace(path: Path) -> pd.DataFrame:
    ace = read_csv(str(path))

    # accept either "gene" or "Gene"
    if "gene" not in ace.columns and "Gene" in ace.columns:
        ace = ace.rename(columns={"Gene": "gene"})

    require_cols(ace, ["gene", "ACE", "CI_low", "CI_high", "Stability", "Verdict"], "CausalEffects_ACE.csv")
    return ace


def load_ranked(path: Path) -> pd.DataFrame:
    ranked = read_csv(str(path))
    if "gene" not in ranked.columns and "Gene" in ranked.columns:
        ranked = ranked.rename(columns={"Gene": "gene"})
    require_cols(ranked, ["gene"], "CausalDrivers_Ranked.csv")

    # Rank is helpful but optional
    if "Rank" not in ranked.columns:
        ranked["Rank"] = np.arange(1, len(ranked) + 1)

    return ranked


def load_program_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = read_csv(str(path))
    # expect ModelID present
    if "ModelID" not in df.columns:
        # tolerate "model_id" etc
        for alt in ["model_id", "DepMap_ID", "ACH_ID"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ModelID"})
                break
    if "ModelID" not in df.columns:
        return None
    return df


def load_edges(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    edges = read_csv(str(path))
    require_cols(edges, ["source", "target"], "CausalNetwork_Edges.csv")
    # weight optional, edge_type optional
    return edges


def load_effects_matrix_from_tidy(tidy_path: Optional[Path]) -> pd.DataFrame | None:
    """
    Builds ModelID x Gene matrix from tidy file.
    Needed for program->viability proxy computation.
    """
    if not tidy_path or not tidy_path.exists():
        return None
    tidy = read_csv(str(tidy_path))

    # normalize expected columns
    rename = {}
    if "Gene" in tidy.columns and "Gene" not in rename:
        rename["Gene"] = "Gene"
    if "ModelID" not in tidy.columns:
        for alt in ["DepMap_ID", "ACH_ID", "model_id"]:
            if alt in tidy.columns:
                rename[alt] = "ModelID"
                break
    tidy = tidy.rename(columns=rename)

    required = {"ModelID", "Gene", "ChronosGeneEffect"}
    if not required.issubset(set(tidy.columns)):
        return None

    mat = tidy.pivot_table(
        index="ModelID",
        columns="Gene",
        values="ChronosGeneEffect",
        aggfunc="mean"
    )
    return mat


# ---------------------------
# Figure 01: Forest plot
# ---------------------------

def fig_top_drivers_forest(ace: pd.DataFrame, ranked: pd.DataFrame, outpath: Path, top_n: int) -> None:
    # Use ACE as truth; bring only Rank from ranked (avoid ACE collisions)
    rcols = ["gene", "Rank"]
    if "DepMapConfidencePct" in ranked.columns:
        rcols.append("DepMapConfidencePct")
    if "q_empirical" in ranked.columns:
        rcols.append("q_empirical")

    df = ace.merge(ranked[rcols].drop_duplicates("gene"), on="gene", how="left")
    df["Rank"] = df["Rank"].fillna(1e9)

    df = df.sort_values("Rank").head(top_n).copy()
    df = df.dropna(subset=["ACE", "CI_low", "CI_high"]).copy()

    df["label"] = df["gene"].astype(str)
    df = df.sort_values("ACE", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ACE"],
        y=df["label"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["CI_high"] - df["ACE"]).clip(lower=0),
            arrayminus=(df["ACE"] - df["CI_low"]).clip(lower=0),
            thickness=1.5
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ACE=%{x:.4f}<br>"
            "CI=[%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>"
            "Stability=%{customdata[2]:.3f}<br>"
            "Verdict=%{customdata[3]}<extra></extra>"
        ),
        customdata=np.c_[df["CI_low"], df["CI_high"], df["Stability"], df["Verdict"]],
    ))

    fig.update_layout(
        xaxis_title="ACE (Chronos effect; negative = decreases viability)",
        yaxis_title="Gene",
        height=max(700, 18 * len(df)),
    )
    write_html(fig, str(outpath), title=f"Top {top_n} causal drivers (ACE ± 95% CI)")


# ---------------------------
# Figure 02: ACE vs Stability
# ---------------------------

def fig_ace_vs_stability(ace: pd.DataFrame, outpath: Path) -> None:
    df = ace.copy()
    fig = px.scatter(
        df,
        x="ACE",
        y="Stability",
        hover_name="gene",
        color="Verdict",
        opacity=0.85
    )
    fig.update_layout(
        xaxis_title="ACE",
        yaxis_title="Bootstrap sign-stability",
        height=650
    )
    write_html(fig, str(outpath), title="ACE vs Stability")


# ---------------------------
# Figure 03: Volcano-like
# ---------------------------

def fig_volcano_like(ace: pd.DataFrame, ranked: pd.DataFrame, outpath: Path) -> None:
    """
    Volcano-style plot:
      x = ACE (from ACE table ONLY)
      y = -log10(q or p) if available; else instability proxy
    """
    df = ace.copy()

    # attach significance columns only (no collisions)
    sig_cols = []
    for c in ["q_empirical", "p_empirical"]:
        if c in ranked.columns and ranked[c].notna().any():
            sig_cols.append(c)

    if sig_cols:
        df = df.merge(
            ranked[["gene"] + sig_cols].drop_duplicates("gene"),
            on="gene",
            how="left"
        )

    if "q_empirical" in df.columns and df["q_empirical"].notna().any():
        sig = df["q_empirical"].astype(float)
        ylab = "-log10(q_empirical)"
        df["neglog"] = -np.log10(sig.replace(0, np.nan).fillna(1.0).clip(lower=1e-300))
    elif "p_empirical" in df.columns and df["p_empirical"].notna().any():
        sig = df["p_empirical"].astype(float)
        ylab = "-log10(p_empirical)"
        df["neglog"] = -np.log10(sig.replace(0, np.nan).fillna(1.0).clip(lower=1e-300))
    else:
        # fallback proxy: instability = 1 - stability
        df["neglog"] = 1.0 - df["Stability"].fillna(0.0)
        ylab = "1 − Stability"

    fig = px.scatter(
        df,
        x="ACE",
        y="neglog",
        hover_name="gene",
        color="Verdict",
        opacity=0.85
    )
    fig.update_layout(
        xaxis_title="ACE",
        yaxis_title=ylab,
        height=650
    )
    write_html(fig, str(outpath), title="Volcano-style view: ACE vs significance/proxy")


# ---------------------------
# Figure 04: Program heatmap
# ---------------------------

def fig_program_heatmap(program_scores: pd.DataFrame, outpath: Path, max_models: int = 200) -> None:
    df = program_scores.copy()
    if "ModelID" in df.columns:
        df = df.set_index("ModelID")

    # keep only Program_ columns
    prog_cols = [c for c in df.columns if str(c).startswith("Program_")]
    if not prog_cols:
        return

    df = df[prog_cols].copy()

    if df.shape[0] > max_models:
        df = df.sample(max_models, random_state=7)

    # z-score per program
    Z = (df - df.mean(axis=0)) / (df.std(axis=0).replace(0, np.nan))
    Z = Z.fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=Z.to_numpy(),
        x=Z.columns.tolist(),
        y=Z.index.astype(str).tolist(),
        colorbar=dict(title="z-score"),
        hovertemplate="Model=%{y}<br>Program=%{x}<br>z=%{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title="Programs",
        yaxis_title="Models (subset)",
        height=720
    )
    write_html(fig, str(outpath), title="Program activity heatmap (z-scored)")


# ---------------------------
# Network: Interpretable hierarchical graph
# ---------------------------

def build_program_membership_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Membership edges are those with target like Program_*
    Returns gene, program, weight
    """
    df = edges.copy()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    df = df[df["target"].str.startswith("Program_")].copy()
    if df.empty:
        return df

    if "weight" not in df.columns:
        df["weight"] = 1.0
    else:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)

    return df.rename(columns={"source": "gene", "target": "program"})[["gene", "program", "weight"]]


def compute_program_viability_edges(program_scores: pd.DataFrame, effects_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Defines ViabilityProxy(model) = mean ChronosGeneEffect across genes (model-level mean)
    Then connects each Program_* to Viability with corr(program_score, ViabilityProxy)
    """
    if program_scores is None or effects_matrix is None:
        return pd.DataFrame()

    P = program_scores.copy()
    if "ModelID" in P.columns:
        P = P.set_index("ModelID")

    prog_cols = [c for c in P.columns if str(c).startswith("Program_")]
    if not prog_cols:
        return pd.DataFrame()

    P = P[prog_cols].copy()

    common = P.index.intersection(effects_matrix.index)
    if len(common) < 10:
        return pd.DataFrame()

    P = P.loc[common]
    X = effects_matrix.loc[common]

    viability_proxy = X.mean(axis=1, skipna=True)

    out = []
    for prog in prog_cols:
        r = P[prog].corr(viability_proxy, method="pearson")
        if pd.isna(r):
            continue
        out.append({"source": prog, "target": "Viability", "weight": float(r), "edge_type": "program_to_viability_corr"})
    return pd.DataFrame(out)


def make_network_html_interpretable(
    ace: pd.DataFrame,
    edges: pd.DataFrame,
    program_scores: pd.DataFrame | None,
    effects_matrix: pd.DataFrame | None,
    outpath: Path,
    top_gene_to_viability: int = 30,
    max_gene_program_edges: int = 600
) -> None:
    ace2 = ace.copy()
    ace2["absACE"] = ace2["ACE"].abs()

    # Top gene->viability edges (true ACE)
    top_genes = ace2.sort_values(["Verdict", "absACE"], ascending=[True, False]).head(top_gene_to_viability).copy()

    membership = build_program_membership_from_edges(edges)
    if membership is not None and not membership.empty:
        membership["absw"] = membership["weight"].abs()
        membership = membership.sort_values("absw", ascending=False).head(max_gene_program_edges).drop(columns=["absw"])

    prog2v = compute_program_viability_edges(program_scores, effects_matrix) if effects_matrix is not None else pd.DataFrame()

    net = Network(
        height="820px",
        width="100%",
        bgcolor="white",
        font_color="black",
        directed=True,
        cdn_resources="in_line"  # CRITICAL: prevents lib folder copy into CWD (Drive-safe)
    )
    net.toggle_physics(True)

    # Viability node
    net.add_node("Viability", label="Viability", shape="dot", size=36,
                 title="Viability (proxy)\nProgram→Viability edges are correlations with model-level mean Chronos effect.\nGene→Viability edges are ACE effects.")

    # Program nodes
    program_nodes = set()
    if membership is not None and not membership.empty:
        program_nodes.update(membership["program"].unique().tolist())
    if not prog2v.empty:
        program_nodes.update(prog2v["source"].unique().tolist())

    for prog in sorted(program_nodes):
        n_members = int((membership["program"] == prog).sum()) if (membership is not None and not membership.empty) else 0
        net.add_node(
            prog,
            label=prog,
            shape="dot",
            size=18 + min(22, n_members // 3),
            title=f"{prog}<br>Members shown: {n_members}"
        )

    # Gene nodes
    genes = set(top_genes["gene"].astype(str).tolist())
    if membership is not None and not membership.empty:
        genes.update(membership["gene"].astype(str).tolist())

    ace_map = ace2.set_index("gene", drop=False)

    for g in sorted(genes):
        if g in ace_map.index:
            r = ace_map.loc[g]
            title = (
                f"<b>{g}</b><br>"
                f"ACE: {float(r['ACE']):.4f}<br>"
                f"CI: [{float(r['CI_low']):.4f}, {float(r['CI_high']):.4f}]<br>"
                f"Stability: {float(r['Stability']):.3f}<br>"
                f"Verdict: {r['Verdict']}<br>"
                f"Direction: {r.get('Direction','')}"
            )
            size = 10 + min(18, float(abs(r["ACE"])) * 10.0)
        else:
            title = f"<b>{g}</b>"
            size = 10

        net.add_node(g, label=g, shape="dot", size=size, title=title)

    # Gene -> Program membership edges
    if membership is not None and not membership.empty:
        for _, r in membership.iterrows():
            g = str(r["gene"])
            p = str(r["program"])
            w = float(r["weight"])
            width = 1.0 + min(6.0, abs(w) * 4.0)
            net.add_edge(g, p, value=width, title=f"{g} → {p}<br>membership weight={w:.3f}")

    # Program -> Viability edges
    if not prog2v.empty:
        for _, r in prog2v.iterrows():
            p = str(r["source"])
            w = float(r["weight"])
            width = 2.0 + min(8.0, abs(w) * 6.0)
            net.add_edge(p, "Viability", value=width, title=f"{p} → Viability<br>corr={w:.3f}")

    # Top Gene -> Viability edges (ACE)
    for _, r in top_genes.iterrows():
        g = str(r["gene"])
        w = float(r["ACE"])
        width = 2.0 + min(10.0, abs(w) * 10.0)
        net.add_edge(g, "Viability", value=width, title=f"{g} → Viability<br>ACE={w:.4f}<br>Stability={float(r['Stability']):.3f}")

    # Layout options
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 170,
          "springConstant": 0.02,
          "damping": 0.09,
          "avoidOverlap": 0.65
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    net.write_html(str(outpath), open_browser=False)


# ---------------------------
# Index page
# ---------------------------

def write_index(outdir: Path, files: list[tuple[str, str]], note: str) -> None:
    items = "\n".join([f'<li><a href="{fn}">{label}</a></li>' for fn, label in files])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DepMap Causality Figures</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 6px; }}
    .note {{ color: #333; max-width: 980px; line-height: 1.45; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>DepMap Causality Figures</h1>
  <div class="note">
    <p><b>Interpretation note (network):</b> This is meant to be interpretable:
    <b>Genes → Programs → Viability</b>. Programs are perturbation similarity clusters. Program→Viability edges are correlations
    to a viability proxy (model-level mean Chronos effect) <i>only if you provide the tidy input</i>. Gene→Viability edges are
    direct ACE effects from <code>CausalEffects_ACE.csv</code>.</p>
    <p>{note}</p>
  </div>
  <ol>
    {items}
  </ol>
</body>
</html>
"""
    with open(outdir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------
# Main Function
# ---------------------------

def generate_causality_figures(
    causality_dir: Path,
    output_dir: Path,
    tidy_path: Optional[Path] = None,
    top_n: int = 50,
    top_gene_viability: int = 30,
    max_gene_program_edges: int = 600,
    logger=None
) -> Path:
    """
    Generate all ACE causality figures.
    
    Args:
        causality_dir: Directory containing ACE CSV outputs
        output_dir: Output directory for HTML figures
        tidy_path: Optional path to tidy dependencies file (for Program→Viability edges)
        top_n: Top N genes for forest plot
        top_gene_viability: Top genes to connect directly to Viability in network
        max_gene_program_edges: Max gene->program edges shown in network
        logger: Logger instance for progress tracking
        
    Returns:
        Path to index.html
    """
    if logger:
        logger.info("Generating ACE causality figures")
        
    output_dir.mkdir(parents=True, exist_ok=True)

    # Required
    p_ace = causality_dir / "CausalEffects_ACE.csv"
    p_rank = causality_dir / "CausalDrivers_Ranked.csv"
    
    if logger:
        logger.info(f"Loading ACE data from: {p_ace}")
    ace = load_ace(p_ace)
    
    if logger:
        logger.info(f"Loading ranked drivers from: {p_rank}")
    ranked = load_ranked(p_rank)

    # Optional
    p_prog = causality_dir / "ProgramScores_ModelMatrix.csv"
    program_scores = load_program_scores(p_prog)

    p_edges = causality_dir / "CausalNetwork_Edges.csv"
    edges = load_edges(p_edges)

    # Optional tidy effects matrix (for Program->Viability)
    effects_matrix = load_effects_matrix_from_tidy(tidy_path) if tidy_path else None

    out_files: list[tuple[str, str]] = []

    # 01 - Forest plot
    if logger:
        logger.info("Creating forest plot")
    f1 = output_dir / "01_top_drivers_forest.html"
    fig_top_drivers_forest(ace, ranked, f1, top_n=top_n)
    out_files.append(("01_top_drivers_forest.html", f"Forest plot – Top {top_n} causal drivers (ACE ± CI)"))

    # 02 - ACE vs Stability
    if logger:
        logger.info("Creating ACE vs Stability scatter plot")
    f2 = output_dir / "02_ace_vs_stability.html"
    fig_ace_vs_stability(ace, f2)
    out_files.append(("02_ace_vs_stability.html", "Scatter – ACE vs Stability"))

    # 03 - Volcano plot
    if logger:
        logger.info("Creating volcano-style plot")
    f3 = output_dir / "03_volcano_like.html"
    fig_volcano_like(ace, ranked, f3)
    out_files.append(("03_volcano_like.html", "Volcano-style – ACE vs significance/proxy"))

    # 04 - Program heatmap (optional)
    if program_scores is not None:
        prog_cols = [c for c in program_scores.columns if str(c).startswith("Program_")]
        if len(prog_cols) >= 2:
            if logger:
                logger.info("Creating program heatmap")
            f4 = output_dir / "04_program_heatmap.html"
            fig_program_heatmap(program_scores, f4)
            out_files.append(("04_program_heatmap.html", "Heatmap – Program scores (z-scored)"))

    # 05 - Network (optional)
    if edges is not None:
        if logger:
            logger.info("Creating interactive network")
        f5 = output_dir / "05_network_interpretable.html"
        make_network_html_interpretable(
            ace=ace,
            edges=edges,
            program_scores=program_scores,
            effects_matrix=effects_matrix,
            outpath=f5,
            top_gene_to_viability=top_gene_viability,
            max_gene_program_edges=max_gene_program_edges
        )
        out_files.append(("05_network_interpretable.html", "Interactive network – Genes → Programs → Viability"))

    note = ""
    if effects_matrix is None:
        note = (
            "Program→Viability edges are skipped because tidy input was not provided. "
            "To enable them, provide the tidy dependencies file path."
        )
    else:
        note = "Program→Viability edges are enabled (tidy input detected)."

    if logger:
        logger.info("Creating index page")
    write_index(output_dir, out_files, note)

    if logger:
        logger.info(f"Figures saved to: {output_dir / 'index.html'}")

    return output_dir / "index.html"
