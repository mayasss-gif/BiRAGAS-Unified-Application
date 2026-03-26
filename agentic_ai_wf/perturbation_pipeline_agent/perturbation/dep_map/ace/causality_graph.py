"""
ACE Graph Visualization Module

Generates signed Gene → Viability causal graphs with HTML legend overlay.
"""

import os
from pathlib import Path

import pandas as pd
from pyvis.network import Network


# -------------------------------------------------
# HTML Legend
# -------------------------------------------------

LEGEND_HTML = """
<div id="legend-box">
  <h3>Legend</h3>

  <div class="legend-item">
    <span class="dot gene"></span> Gene
  </div>

  <div class="legend-item">
    <span class="dot viability"></span> Viability (phenotype)
  </div>

  <hr>

  <div class="legend-item">
    <span class="line red thick"></span>
    Strong negative ACE (essential gene)
  </div>

  <div class="legend-item">
    <span class="line red thin"></span>
    Weak negative ACE
  </div>

  <div class="legend-item">
    <span class="line green medium"></span>
    Positive ACE (increases fitness)
  </div>

  <div class="legend-note">
    Line thickness ∝ |ACE|
  </div>
</div>

<style>
#legend-box {
  position: fixed;
  top: 20px;
  right: 20px;
  background: white;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 12px 14px;
  font-family: Arial, sans-serif;
  font-size: 13px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  z-index: 9999;
}

#legend-box h3 {
  margin: 0 0 8px 0;
  font-size: 15px;
}

.legend-item {
  display: flex;
  align-items: center;
  margin: 4px 0;
}

.dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
}

.dot.gene {
  background: #97C2FC;
}

.dot.viability {
  background: #B39DDB;
}

.line {
  width: 30px;
  height: 0;
  border-top-style: solid;
  margin-right: 8px;
}

.line.red {
  border-top-color: red;
}

.line.green {
  border-top-color: green;
}

.line.thick {
  border-top-width: 6px;
}

.line.medium {
  border-top-width: 4px;
}

.line.thin {
  border-top-width: 2px;
}

.legend-note {
  margin-top: 6px;
  font-size: 11px;
  color: #444;
}
</style>
"""


def inject_legend(html_path: str):
    """Inject the legend HTML before </body> tag."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    if "legend-box" in html:
        return  # already injected

    html = html.replace("</body>", LEGEND_HTML + "\n</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# -------------------------------------------------
# IO
# -------------------------------------------------

def load_ace(causality_dir: Path) -> pd.DataFrame:
    """Load ACE results from causality directory."""
    path = causality_dir / "CausalEffects_ACE.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


# -------------------------------------------------
# Graph Generation
# -------------------------------------------------

def signed_gene_viability_graph(
    ace: pd.DataFrame,
    outpath: Path,
    top_n: int = 40
):
    """
    Generate signed Gene → Viability causal graph.
    
    Args:
        ace: DataFrame with ACE results
        outpath: Output path for HTML file
        top_n: Number of top genes to display (by absolute ACE)
    """
    df = ace.copy()
    df["absACE"] = df["ACE"].abs()
    df = df.sort_values("absACE", ascending=False).head(top_n)

    net = Network(
        height="900px",
        width="100%",
        bgcolor="white",
        font_color="black",
        directed=True,
        cdn_resources="in_line"
    )

    net.add_node(
        "Viability",
        label="Viability",
        shape="dot",
        size=44,
        color="#B39DDB",
        title="Cell viability / fitness outcome"
    )

    for _, r in df.iterrows():
        gene = str(r["gene"])
        ace_val = float(r["ACE"])
        stability = float(r["Stability"])

        width = 2.0 + min(10.0, abs(ace_val) * 4.0)
        size = 10.0 + min(20.0, abs(ace_val) * 6.0)
        opacity = 0.35 + 0.65 * stability

        rgba = (
            f"rgba(255,0,0,{opacity:.2f})"
            if ace_val < 0
            else f"rgba(0,150,0,{opacity:.2f})"
        )

        net.add_node(
            gene,
            label=gene,
            shape="dot",
            size=size,
            color="#97C2FC",
            title=(
                f"<b>{gene}</b><br>"
                f"ACE: {ace_val:.4f}<br>"
                f"Stability: {stability:.3f}"
            )
        )

        net.add_edge(
            gene,
            "Viability",
            width=width,
            color=rgba,
            title=f"{gene} → Viability<br>ACE={ace_val:.4f}"
        )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -32000,
          "centralGravity": 0.35,
          "springLength": 160,
          "springConstant": 0.02,
          "damping": 0.09,
          "avoidOverlap": 0.7
        }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }
      }
    }
    """)

    net.write_html(str(outpath), open_browser=False)
    inject_legend(str(outpath))


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def generate_ace_graph(
    causality_dir: Path,
    output_dir: Path,
    top_n: int = 40,
    logger=None
) -> Path:
    """
    Generate signed ACE graph visualization.
    
    Args:
        causality_dir: Directory containing CausalEffects_ACE.csv
        output_dir: Output directory for graph HTML
        top_n: Number of top genes to display
        logger: Logger instance for progress tracking
        
    Returns:
        Path to generated HTML file
    """
    if logger:
        logger.info(f"Loading ACE data from: {causality_dir}")
        
    ace = load_ace(causality_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "signed_gene_viability_alignment.html"
    
    if logger:
        logger.info(f"Generating signed ACE graph with top {top_n} genes")
        
    signed_gene_viability_graph(ace, outpath, top_n=top_n)
    
    if logger:
        logger.info(f"Graph saved to: {outpath}")
        
    return outpath
