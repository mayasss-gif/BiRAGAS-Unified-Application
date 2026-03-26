from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _relpath(fp: Path, root: Path) -> str:
    try:
        return fp.relative_to(root).as_posix() if fp.is_absolute() else str(fp)
    except Exception:
        return str(fp)


def _maybe_embed(figures: Dict[str, List[Path]],
                 section_key: str,
                 out_root: Path,
                 prefix: str,
                 lines: list[str]) -> None:
    """
    Append image markdown lines for figures whose filename starts with `prefix`.
    No-op if key/prefix not present.
    """
    try:
        for fp in figures.get(section_key, []):
            name = fp.name
            if name.startswith(prefix):
                rel = _relpath(fp, out_root)
                # Use the filename (without extension) as alt text fallback
                alt = prefix.replace("_", " ")
                lines.append(f"![{alt}]({rel})\n")
    except Exception as e:
        logging.debug(f"_maybe_embed failed for prefix={prefix}: {e}")


def _maybe_link_table(out_root: Path, relpath: str, label: str) -> str:
    p = out_root / relpath
    if p.exists():
        return f"- [{label}]({relpath})"
    return ""


def build_report(out_root: Path,
                 shared_tbl: pd.DataFrame,
                 indiv: Dict[str, Dict[str, pd.DataFrame]],
                 figures: Dict[str, List[Path]]) -> None:
    try:
        report = out_root / "summary.md"
        lines: list[str] = []

        # Header
        lines.append("# MDP Insights Summary\n")
        lines.append(
            "This report summarizes individual disease insights and shared pathway/entity "
            "findings derived from MDP JSONs.\n"
        )

        # ---------------- Shared Highlights ----------------
        lines.append("## Shared Highlights\n")
        if shared_tbl is not None and not shared_tbl.empty:
            top = shared_tbl.head(10)
            lines.append("Top shared pathways (by # diseases):\n")
            for _, r in top.iterrows():
                lines.append(
                    f"- **{r['pathway']}** — diseases: {r['diseases']} (n={r['n_diseases']}); "
                    f"TFs: {r['tf_n']}, EPI: {r['epigenetic_n']}, MET: {r['metabolite_n']}; "
                    f"shared genes n={r['shared_genes_n']}"
                )
            lines.append("")
        else:
            lines.append("_No shared rows._\n")

        # Key tables quick-links (only if they exist)
        table_links = [
            _maybe_link_table(out_root, "tables/top200_pathways_cross_disease.csv", "Top-200 Pathways (Cross-Disease)"),
            _maybe_link_table(out_root, "tables/top200_targets_tf.csv", "Top-200 Targets — TF"),
            _maybe_link_table(out_root, "tables/top200_targets_epigenetic.csv", "Top-200 Targets — Epigenetic"),
            _maybe_link_table(out_root, "tables/top200_targets_metabolites.csv", "Top-200 Targets — Metabolites"),
            _maybe_link_table(out_root, "tables/disease_interconnection.csv", "Disease Interconnection (Jaccard)"),
            _maybe_link_table(out_root, "tables/directional_concordance_by_pathway.csv", "Directional Concordance by Pathway"),
        ]
        table_links = [t for t in table_links if t]
        if table_links:
            lines.append("**Key Tables**")
            lines.extend(table_links)
            lines.append("")

        # Embed shared figures: first the legacy ones (if any)
        if "shared" in figures:
            # Legacy/Existing
            _maybe_embed(figures, "shared", out_root, "shared_presence_heatmap", lines)
            _maybe_embed(figures, "shared", out_root, "shared_upset_like", lines)
            _maybe_embed(figures, "shared", out_root, "shared_entity_mix_bars", lines)
            _maybe_embed(figures, "shared", out_root, "shared_genes_count", lines)
            _maybe_embed(figures, "shared", out_root, "shared_disease_similarity", lines)
            _maybe_embed(figures, "shared", out_root, "shared_pathway_cooccurrence", lines)

            # NEW shared visuals
            _maybe_embed(figures, "shared", out_root, "shared_up_significance_heatmap", lines)
            _maybe_embed(figures, "shared", out_root, "shared_pathway_leaderboard", lines)
            _maybe_embed(figures, "shared", out_root, "shared_target_volcano_tf", lines)
            _maybe_embed(figures, "shared", out_root, "shared_target_volcano_epigenetic", lines)
            _maybe_embed(figures, "shared", out_root, "shared_target_volcano_metabolites", lines)

        # ---------------- Individual Disease Highlights ----------------
        lines.append("\n## Individual Disease Highlights\n")
        for disease, parts in indiv.items():
            lines.append(f"### {disease}\n")
            tp = parts.get("top_pathways", pd.DataFrame()).head(5)
            if not tp.empty:
                for _, r in tp.iterrows():
                    pss = float(r.get("pss", 0.0)) if pd.notna(r.get("pss", None)) else 0.0
                    nent = int(r.get("n_entities", 0)) if pd.notna(r.get("n_entities", None)) else 0
                    lines.append(f"- {r['pathway']} (PSS={pss:.3f}, n_entities={nent})")
            else:
                lines.append("- No top pathways available.")
            lines.append("")

            if disease in figures:
                for fp in figures[disease]:
                    rel = _relpath(fp, out_root)
                    lines.append(f"![{disease}]({rel})\n")

        # Write file
        with report.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logging.info(f"Report written: {report}")

    except Exception as e:
        logging.error(f"build_report failed: {e}")
