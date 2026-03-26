"""
Phase 5: PHARMA INTERVENTION — Module 2
DruggabilityScorer (INTENT I_03 Module 2)
==========================================
Assesses the druggability of causal driver genes.

Druggability Dimensions:
  1. Protein Structure (surface accessibility, binding pockets)
  2. Target Class (kinase, GPCR, nuclear receptor, etc.)
  3. Existing Ligands (approved drugs, clinical candidates, tool compounds)
  4. Genetic Tractability (loss-of-function tolerance)
  5. Expression Accessibility (tissue-specific expression)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

DRUGGABLE_FAMILIES = {
    'kinase': 0.9, 'gpcr': 0.9, 'nuclear_receptor': 0.85,
    'ion_channel': 0.8, 'protease': 0.75, 'phosphatase': 0.7,
    'epigenetic': 0.7, 'transporter': 0.65, 'enzyme': 0.6,
    'transcription_factor': 0.3, 'scaffold': 0.2, 'unknown': 0.4,
}


@dataclass
class DruggabilityConfig:
    w_target_class: float = 0.30
    w_existing_ligands: float = 0.25
    w_genetic_tractability: float = 0.20
    w_expression: float = 0.15
    w_structure: float = 0.10
    druggable_threshold: float = 0.50


class DruggabilityScorer:
    """Scores gene druggability for therapeutic targeting."""

    def __init__(self, config: Optional[DruggabilityConfig] = None):
        self.config = config or DruggabilityConfig()

    def score_all(self, dag: nx.DiGraph) -> Dict[str, Dict]:
        """Score druggability for all regulatory genes in the DAG."""
        results = {}
        genes = [(n, d) for n, d in dag.nodes(data=True)
                 if d.get('layer') == 'regulatory']

        for gene, data in genes:
            result = self.score_gene(gene, data)
            results[gene] = result
            dag.nodes[gene]['druggability_score'] = result['score']
            dag.nodes[gene]['druggability_class'] = result['druggability_class']

        return results

    def score_gene(self, gene: str, data: Dict) -> Dict:
        """Score a single gene's druggability."""
        cfg = self.config

        target_class = self._score_target_class(data)
        ligands = self._score_existing_ligands(data)
        tractability = self._score_genetic_tractability(data)
        expression = self._score_expression(data)
        structure = self._score_structure(data)

        composite = (cfg.w_target_class * target_class +
                     cfg.w_existing_ligands * ligands +
                     cfg.w_genetic_tractability * tractability +
                     cfg.w_expression * expression +
                     cfg.w_structure * structure)

        if composite >= 0.7:
            drug_class = 'Highly_Druggable'
        elif composite >= cfg.druggable_threshold:
            drug_class = 'Druggable'
        elif composite >= 0.3:
            drug_class = 'Challenging'
        else:
            drug_class = 'Undruggable'

        return {
            'gene': gene,
            'score': round(composite, 4),
            'druggability_class': drug_class,
            'dimensions': {
                'target_class': round(target_class, 4),
                'existing_ligands': round(ligands, 4),
                'genetic_tractability': round(tractability, 4),
                'expression': round(expression, 4),
                'structure': round(structure, 4),
            },
            'protein_family': data.get('protein_family', 'unknown'),
        }

    def _score_target_class(self, data: Dict) -> float:
        """Score based on protein family druggability."""
        family = data.get('protein_family', 'unknown').lower()
        return DRUGGABLE_FAMILIES.get(family, 0.4)

    def _score_existing_ligands(self, data: Dict) -> float:
        """Score based on known ligands and drugs."""
        score = 0.0
        if data.get('has_approved_drug'):
            score += 0.5
        if data.get('in_clinical_trial'):
            score += 0.25
        if data.get('has_chemical_probe'):
            score += 0.15
        n_ligands = data.get('n_known_ligands', 0)
        score += min(0.1, n_ligands * 0.02)
        return min(1.0, score)

    def _score_genetic_tractability(self, data: Dict) -> float:
        """Score based on loss-of-function tolerance."""
        ess = data.get('essentiality_tag', 'Unknown')
        if ess == 'Tumor-Selective Dependency':
            return 0.9
        elif ess == 'Non-Essential':
            return 0.7
        elif ess == 'Core Essential':
            return 0.3
        return 0.5

    def _score_expression(self, data: Dict) -> float:
        """Score based on expression accessibility."""
        score = 0.5
        if data.get('disease_tissue_expressed', False):
            score += 0.3
        if data.get('cell_surface', False):
            score += 0.2
        return min(1.0, score)

    def _score_structure(self, data: Dict) -> float:
        """Score based on structural information availability."""
        score = 0.3
        if data.get('has_crystal_structure'):
            score += 0.4
        if data.get('has_binding_pocket'):
            score += 0.3
        return min(1.0, score)
