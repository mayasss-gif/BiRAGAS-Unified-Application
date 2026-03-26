"""
ResultMerger — Combines Outputs from All Three Systems
=========================================================
Merges results from v1.0, v2.0, Mega engines + BiRAGAS phases
into a unified ranked target list and comprehensive report.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biragas.integration.merger")


class ResultMerger:
    """
    Merges knockout predictions from multiple engines into unified output.

    Handles:
    - v1.0 KnockoutResult (5-method ensemble_score)
    - v2.0 KnockoutResult (7-method ensemble)
    - Mega MegaKnockoutResult (influence_score)
    - BiRAGAS Phase 5 target_ranking

    Output: Unified ranked list with cross-engine consensus.
    """

    def merge_knockout_results(self, results_by_engine: Dict[str, List]) -> List[Dict]:
        """
        Merge knockout results from multiple engines.

        Args:
            results_by_engine: {"v1": [results], "v2": [results], "mega": [results]}

        Returns:
            Unified list sorted by consensus score.
        """
        gene_scores = {}

        for engine, results in results_by_engine.items():
            for r in results:
                gene = r.gene if hasattr(r, 'gene') else r.get('gene', '')
                if not gene:
                    continue

                if gene not in gene_scores:
                    gene_scores[gene] = {
                        'gene': gene, 'engines': [],
                        'scores': [], 'effects': [], 'directions': [],
                        'essentiality': '', 'alignment': '',
                    }

                entry = gene_scores[gene]
                entry['engines'].append(engine)

                if hasattr(r, 'ensemble'):
                    entry['scores'].append(r.ensemble)
                elif hasattr(r, 'ensemble_score'):
                    entry['scores'].append(r.ensemble_score)
                elif isinstance(r, dict):
                    entry['scores'].append(r.get('ensemble', r.get('ensemble_score', 0)))

                if hasattr(r, 'disease_effect'):
                    entry['effects'].append(r.disease_effect)
                if hasattr(r, 'direction'):
                    entry['directions'].append(r.direction)
                if hasattr(r, 'essentiality') and r.essentiality:
                    entry['essentiality'] = r.essentiality
                elif hasattr(r, 'essentiality_class') and r.essentiality_class:
                    entry['essentiality'] = r.essentiality_class
                if hasattr(r, 'alignment') and r.alignment:
                    entry['alignment'] = r.alignment
                elif hasattr(r, 'therapeutic_alignment') and r.therapeutic_alignment:
                    entry['alignment'] = r.therapeutic_alignment

        # Compute consensus score
        unified = []
        for gene, data in gene_scores.items():
            import numpy as np
            scores = data['scores']
            consensus = float(np.mean(scores)) if scores else 0.0
            n_engines = len(set(data['engines']))
            # Bonus for cross-engine agreement
            cross_engine_bonus = 0.1 * (n_engines - 1)
            final_score = consensus + cross_engine_bonus

            unified.append({
                'gene': gene,
                'consensus_score': round(final_score, 6),
                'mean_score': round(consensus, 6),
                'n_engines': n_engines,
                'engines': list(set(data['engines'])),
                'mean_effect': round(float(np.mean(data['effects'])), 6) if data['effects'] else 0.0,
                'direction': max(set(data['directions']), key=data['directions'].count) if data['directions'] else 'unknown',
                'essentiality': data['essentiality'],
                'alignment': data['alignment'],
            })

        unified.sort(key=lambda x: -x['consensus_score'])
        for i, u in enumerate(unified):
            u['rank'] = i + 1

        return unified

    def merge_with_biragas(self, unified_ko: List[Dict], biragas_ranking: Dict) -> List[Dict]:
        """
        Merge unified knockout results with BiRAGAS Phase 5 target ranking.
        """
        biragas_targets = {}
        for t in biragas_ranking.get('ranked_targets', []):
            gene = t.get('gene', '')
            if gene:
                biragas_targets[gene] = t.get('composite_score', 0)

        for entry in unified_ko:
            gene = entry['gene']
            biragas_score = biragas_targets.get(gene, 0)
            entry['biragas_score'] = round(biragas_score, 6)
            # Combined: 60% CRISPR consensus + 40% BiRAGAS ranking
            entry['integrated_score'] = round(
                0.6 * entry['consensus_score'] + 0.4 * biragas_score, 6
            )

        unified_ko.sort(key=lambda x: -x['integrated_score'])
        for i, u in enumerate(unified_ko):
            u['integrated_rank'] = i + 1

        return unified_ko

    def export_unified_csv(self, results: List[Dict], filepath: str):
        """Export unified results to CSV."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'gene', 'consensus_score', 'integrated_score', 'biragas_score',
                        'n_engines', 'engines', 'mean_effect', 'direction',
                        'essentiality', 'alignment'])
            for r in results:
                w.writerow([
                    r.get('integrated_rank', r.get('rank')),
                    r['gene'], r['consensus_score'],
                    r.get('integrated_score', ''), r.get('biragas_score', ''),
                    r['n_engines'], '+'.join(r['engines']),
                    r['mean_effect'], r['direction'],
                    r['essentiality'], r['alignment'],
                ])
        logger.info(f"Exported {len(results)} unified results to {filepath}")

    def export_summary_json(self, results: List[Dict], filepath: str):
        """Export summary to JSON."""
        summary = {
            "total_genes": len(results),
            "top_10": results[:10],
            "engines_used": list(set(e for r in results for e in r.get('engines', []))),
            "n_multi_engine": sum(1 for r in results if r.get('n_engines', 0) > 1),
        }
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
