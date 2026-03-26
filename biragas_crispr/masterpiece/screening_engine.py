"""
ScreeningEngine — MAGeCK + BAGEL2 + DrugZ Screening Analysis
================================================================
Wraps the CRISPR screening pipeline outputs into a unified analysis engine.
Loads results from all 6 screening modes and produces unified gene scoring.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biragas.masterpiece.screening")

# Import from existing screening agent
_CRISPR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CRISPR_ROOT)
sys.path.insert(0, os.path.join(_CRISPR_ROOT, "CRISPR_MultiKnockout"))


class ScreeningEngine:
    """
    Unified CRISPR screening analysis integrating MAGeCK + BAGEL2 + DrugZ.

    Wraps the ScreeningAgent from crispr_agentic_engine with
    additional analysis from CRISPR_PHASE1 ScreeningToPhase1 converter.

    Usage:
        engine = ScreeningEngine()
        engine.auto_load("/path/to/CRISPR/")
        results = engine.get_results()
        top_drivers = engine.get_top_drivers(50)
    """

    def __init__(self):
        self._screening_agent = None
        self._converter = None
        self._loaded = False

    def auto_load(self, crispr_dir: str) -> Dict:
        """Auto-discover and load all screening data."""
        status = {'loaded': [], 'failed': []}

        try:
            from crispr_agentic_engine.data_discovery import DataDiscoveryAgent
            from crispr_agentic_engine.screening_agent import ScreeningAgent

            discovery = DataDiscoveryAgent().discover(crispr_dir)
            self._screening_agent = ScreeningAgent()
            load_status = self._screening_agent.load_from_discovery(discovery)
            status['loaded'].append(f"ScreeningAgent: {self._screening_agent.get_summary()['total_genes']} genes")
            self._loaded = True
        except Exception as e:
            status['failed'].append(f"ScreeningAgent: {e}")

        try:
            from CRISPR_PHASE1.screening_converter import ScreeningToPhase1
            self._converter = ScreeningToPhase1()
            # Auto-find RRA/MLE files
            for root, dirs, files in os.walk(crispr_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if 'gene_summary' in f.lower() and 'rra' in root.lower():
                        self._converter.load_rra(fp)
                        status['loaded'].append(f"RRA: {f}")
                    elif 'gene_summary' in f.lower() and 'mle' in root.lower():
                        self._converter.load_mle(fp)
                        status['loaded'].append(f"MLE: {f}")
                    elif f == 'essential_genes.txt':
                        self._converter.load_essential_genes(fp)
                    elif f == 'nonessential_genes.txt':
                        self._converter.load_nonessential_genes(fp)

            self._converter.compute_ace_scores()
            status['loaded'].append(f"Converter: {self._converter.get_summary()['total_genes']} genes")
        except Exception as e:
            status['failed'].append(f"Converter: {e}")

        logger.info(f"ScreeningEngine: {len(status['loaded'])} loaded, {len(status['failed'])} failed")
        return status

    def get_results(self) -> Dict:
        """Get all screening results."""
        results = {}
        if self._screening_agent:
            results['screening'] = self._screening_agent.get_summary()
        if self._converter:
            results['converter'] = self._converter.get_summary()
        return results

    def get_top_drivers(self, n: int = 50) -> List:
        """Get top causal drivers."""
        if self._screening_agent:
            return self._screening_agent.get_top_drivers(n)
        return []

    def get_safe_drivers(self) -> List:
        """Get non-essential drivers (ideal drug targets)."""
        if self._screening_agent:
            return self._screening_agent.get_safe_drivers()
        return []

    def export_phase1_files(self, output_dir: str):
        """Export BiRAGAS Phase 1 compatible files."""
        if self._converter:
            self._converter.export_phase1_files(output_dir)
        elif self._screening_agent:
            self._screening_agent.export_biragas_csv(output_dir)

    def is_loaded(self) -> bool:
        return self._loaded
