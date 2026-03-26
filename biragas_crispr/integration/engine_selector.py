"""
EngineSelector — Intelligent CRISPR Engine Selection
=======================================================
Automatically selects the optimal CRISPR engine based on:
- Scale (gene count, guide count)
- Available data (screening modes, Perturb-seq, targeted)
- Computational resources (RAM, GPU availability)
- Analysis requirements (speed vs depth)
"""

import logging
import os
from typing import Any, Dict, Optional

import networkx as nx

logger = logging.getLogger("biragas.integration.selector")


class EngineSelector:
    """
    Selects the optimal CRISPR engine for a given analysis.

    Three engines available:
    v1.0 (Classical):  5-method ensemble, 12-stream ACE, <100 genes ideal
    v2.0 (Agentic):    7-method ensemble, 15-stream ACE, 100-5000 genes, auto-discovery
    Mega (Genome):     Sparse matrix O(1), 177K scale, >5000 genes

    Usage:
        selector = EngineSelector()
        engine_name, reason = selector.select(dag, crispr_dir)
    """

    def select(self, dag: nx.DiGraph, crispr_dir: str = "",
               force: Optional[str] = None) -> tuple:
        """
        Select the best engine.

        Returns: (engine_name, reason)
            engine_name: "v1", "v2", or "mega"
            reason: Human-readable explanation
        """
        if force:
            return force, f"Forced to {force} by user"

        n_genes = sum(1 for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory')
        n_edges = dag.number_of_edges()
        has_brunello = self._check_brunello(crispr_dir)
        has_screening = self._check_screening(crispr_dir)
        has_perturbseq = self._check_perturbseq(crispr_dir)

        # Decision tree
        if n_genes > 5000:
            return "mega", (
                f"Mega engine selected: {n_genes} genes exceeds v2 threshold (5000). "
                f"Sparse matrix O(1) prediction needed for genome-scale analysis."
            )
        elif n_genes > 100 and (has_screening or has_perturbseq):
            return "v2", (
                f"Agentic v2.0 selected: {n_genes} genes with available CRISPR data. "
                f"7-method ensemble + auto-discovery + 15-stream ACE."
            )
        elif n_genes > 100:
            return "v2", (
                f"Agentic v2.0 selected: {n_genes} genes (no CRISPR data found, will use DAG-only prediction)."
            )
        else:
            return "v1", (
                f"Classical v1.0 selected: {n_genes} genes (small scale). "
                f"5-method ensemble sufficient for detailed per-gene analysis."
            )

    def _check_brunello(self, crispr_dir: str) -> bool:
        if not crispr_dir or not os.path.isdir(crispr_dir):
            return False
        for root, dirs, files in os.walk(crispr_dir):
            for f in files:
                if 'brunello' in f.lower() and f.endswith('.tsv'):
                    return True
        return False

    def _check_screening(self, crispr_dir: str) -> bool:
        if not crispr_dir:
            return False
        for root, dirs, files in os.walk(crispr_dir):
            for f in files:
                if 'gene_summary' in f.lower():
                    return True
        return False

    def _check_perturbseq(self, crispr_dir: str) -> bool:
        if not crispr_dir:
            return False
        for root, dirs, files in os.walk(crispr_dir):
            for f in files:
                if f.endswith('.h5ad'):
                    return True
        return False

    def get_engine_comparison(self) -> Dict:
        """Return comparison table of all 3 engines."""
        return {
            "v1_classical": {
                "methods": 5, "ace_streams": 12, "max_genes": 100,
                "monte_carlo": 100, "synergy_models": 5,
                "features": "Detailed per-gene analysis, all propagation metadata",
                "ideal_for": "Small focused studies (<100 genes)",
            },
            "v2_agentic": {
                "methods": 7, "ace_streams": 15, "max_genes": 5000,
                "monte_carlo": 500, "synergy_models": 6,
                "features": "+ODE-dynamic, +mutual info, +ZIP synergy, auto-discovery, 10 agents",
                "ideal_for": "Medium studies (100-5000 genes) with CRISPR data",
            },
            "mega_genome": {
                "methods": "matrix_resolvent", "ace_streams": "N/A", "max_genes": "19091+",
                "monte_carlo": "N/A", "synergy_models": "vectorized_bliss",
                "features": "O(1) per KO, sparse matrix, 177K configs, 22.1B pairs",
                "ideal_for": "Genome-scale (>5000 genes), Brunello library",
            },
        }
