"""
Phase 2: NETWORK CAUSAL IMPORTANCE — Module 1
CentralityCalculator (INTENT I_01 Module 2)
=============================================
Computes causal centrality metrics and 3-tier gene classification.

Based on production CentralityCalculator architecture:
  1. Topological Dominance & Apex Scores (weighted in/out degrees)
  2. Probabilistic Betweenness Centrality (distance = 1 - confidence)
  3. Reverse Personalized PageRank (disease proximity)
  4. Systemic Pleiotropy (super-node traversal)
  5. Strict 3-Tier Arbitration (Master Regulator / Secondary Driver / Downstream Effector)

Causal Importance = (apex * causal_strength) + betweenness + (ppr * 10)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CentralityConfig:
    ppr_alpha: float = 0.85
    tier1_apex_threshold: float = 0.60
    tier1_pleiotropy_min: int = 1
    tier3_apex_threshold: float = 0.30
    tier3_ppr_threshold: float = 0.001
    betweenness_percentile: float = 85.0
    target_node: str = "Disease_Activity"


class CentralityCalculator:
    """Computes causal centrality and classifies genes into 3 tiers."""

    def __init__(self, config: Optional[CentralityConfig] = None):
        self.config = config or CentralityConfig()
        self.metrics_report = {
            'meta': {'module': 'CentralityCalculator', 'total_scored_genes': 0,
                     'total_scored_programs': 0},
            'hub_classifications': {
                'Tier_1_Master_Regulators': [],
                'Tier_2_Secondary_Drivers': [],
                'Tier_3_Downstream_Effectors': [],
            },
            'gene_scores': {},
            'program_scores': {},
        }

    def run_pipeline(self, dag: nx.DiGraph) -> tuple:
        """Run the full centrality pipeline. Returns (enriched_dag, metrics_report)."""
        self._preprocess_edges(dag)

        logger.info("Stage 1: Calculating Topological Dominance & Apex Scores...")
        self._calculate_degrees(dag)

        logger.info("Stage 2: Calculating Probabilistic Betweenness Centrality...")
        self._calculate_betweenness(dag)

        logger.info("Stage 3: Running Reverse Personalized PageRank...")
        self._calculate_reverse_ppr(dag)

        logger.info("Stage 4: Traversing Super-Nodes for Systemic Pleiotropy...")
        self._calculate_pleiotropy(dag)

        logger.info("Stage 5: Arbitrating Strict 3-Tier Classifications...")
        self._arbitrate_tiers(dag)

        return dag, self.metrics_report

    def _preprocess_edges(self, dag: nx.DiGraph) -> None:
        """Convert biological confidence into mathematical costs and capacities."""
        for u, v, d in dag.edges(data=True):
            conf = d.get('confidence_score', 0.1)
            weight = d.get('weight', 0.1)
            d['distance'] = max(0.01, 1.0 - conf)
            d['capacity'] = conf * weight

    def _calculate_degrees(self, dag: nx.DiGraph) -> None:
        """Weighted in/out degrees, causal strength, and apex scores."""
        for node in dag.nodes():
            in_edges = dag.in_edges(node, data=True)
            out_edges = dag.out_edges(node, data=True)

            w_in = sum(d.get('capacity', 0) for _, _, d in in_edges)
            w_out = sum(d.get('capacity', 0) for _, _, d in out_edges)

            out_weights = [d.get('weight', 0) for _, _, d in dag.out_edges(node, data=True)]
            causal_strength = float(np.mean(out_weights)) if out_weights else 0.0

            apex = w_out / (w_in + w_out + 1e-9)

            nx.set_node_attributes(dag, {node: {
                'w_in': round(w_in, 3),
                'w_out': round(w_out, 3),
                'apex_score': round(apex, 3),
                'causal_strength': round(causal_strength, 3),
            }})

    def _calculate_betweenness(self, dag: nx.DiGraph) -> None:
        """Probabilistic betweenness centrality using distance weights."""
        # OPTIMIZATION: Approximate for large graphs (>500 nodes)
        if dag.number_of_nodes() > 500:
            betweenness = nx.betweenness_centrality(dag, weight='distance', normalized=True, k=min(100, dag.number_of_nodes()))
        else:
            betweenness = nx.betweenness_centrality(dag, weight='distance', normalized=True)
        nx.set_node_attributes(dag, {n: {'betweenness': round(b, 4)}
                                     for n, b in betweenness.items()})

    def _calculate_reverse_ppr(self, dag: nx.DiGraph) -> None:
        """Reverse Personalized PageRank from disease node."""
        target = self.config.target_node
        if target not in dag.nodes():
            logger.warning(f"Target node '{target}' not found. Skipping PPR.")
            nx.set_node_attributes(dag, 0.0, 'disease_proximity')
            return

        G_rev = dag.reverse(copy=True)
        try:
            ppr = nx.pagerank(G_rev, alpha=self.config.ppr_alpha,
                              personalization={target: 1.0}, weight='capacity')
            nx.set_node_attributes(dag, {n: {'disease_proximity': round(p, 5)}
                                         for n, p in ppr.items()})
        except Exception as e:
            logger.error(f"PPR calculation failed: {e}")
            nx.set_node_attributes(dag, 0.0, 'disease_proximity')

    def _calculate_pleiotropy(self, dag: nx.DiGraph) -> None:
        """Count distinct pathway classes each regulatory gene connects to."""
        for node, d in dag.nodes(data=True):
            if d.get('layer') != 'regulatory':
                continue
            unique_classes = set()
            for succ in dag.successors(node):
                succ_data = dag.nodes[succ]
                if succ_data.get('layer') == 'program':
                    main_class = succ_data.get('main_class', 'Unclassified')
                    if main_class != 'Unclassified':
                        unique_classes.add(main_class)
            dag.nodes[node]['pleiotropic_reach'] = len(unique_classes)

    def _arbitrate_tiers(self, dag: nx.DiGraph) -> None:
        """Strict 3-tier classification of regulatory genes."""
        cfg = self.config
        gene_count = 0
        program_count = 0

        all_betweenness = [d.get('betweenness', 0) for _, d in dag.nodes(data=True)
                           if d.get('layer') == 'regulatory']
        b_85th = np.percentile(all_betweenness, cfg.betweenness_percentile) if all_betweenness else 0.0

        for node, d in dag.nodes(data=True):
            layer = d.get('layer')

            if layer == 'program':
                program_count += 1
                self.metrics_report['program_scores'][node] = {
                    'main_class': d.get('main_class'),
                    'metrics': {
                        'weighted_in_degree': d.get('w_in'),
                        'betweenness': d.get('betweenness'),
                        'disease_proximity': d.get('disease_proximity'),
                    },
                }
                continue

            if layer == 'regulatory':
                gene_count += 1
                apex = d.get('apex_score', 0)
                betweenness = d.get('betweenness', 0)
                pleiotropy = d.get('pleiotropic_reach', 0)
                ppr = d.get('disease_proximity', 0)
                causal_strength = d.get('causal_strength', 0)

                is_prior = d.get('causal_tier') in ('Validated Driver',)
                has_direct_edge = dag.has_edge(node, cfg.target_node)

                causal_importance = (apex * causal_strength) + betweenness + (ppr * 10)
                dag.nodes[node]['causal_importance'] = round(causal_importance, 4)

                tier = 'Tier_2_Secondary_Driver'

                if apex >= cfg.tier1_apex_threshold and pleiotropy >= cfg.tier1_pleiotropy_min and is_prior:
                    tier = 'Tier_1_Master_Regulator'
                elif ((apex <= cfg.tier3_apex_threshold or has_direct_edge or ppr > cfg.tier3_ppr_threshold)
                      and betweenness < b_85th):
                    tier = 'Tier_3_Downstream_Effector'

                dag.nodes[node]['network_tier'] = tier

                self.metrics_report['hub_classifications'][tier + 's'].append(node)
                self.metrics_report['gene_scores'][node] = {
                    'assigned_network_tier': tier,
                    'metrics': {
                        'weighted_out_degree': d.get('w_out'),
                        'weighted_in_degree': d.get('w_in'),
                        'apex_cascade_position': apex,
                        'probabilistic_betweenness': betweenness,
                        'pleiotropic_reach_classes': pleiotropy,
                        'causal_strength': causal_strength,
                        'causal_importance_score': round(causal_importance, 4),
                    },
                }

        for tier_key in self.metrics_report['hub_classifications']:
            self.metrics_report['hub_classifications'][tier_key].sort(
                key=lambda x: dag.nodes[x].get('causal_importance', 0), reverse=True
            )

        self.metrics_report['meta']['total_scored_genes'] = gene_count
        self.metrics_report['meta']['total_scored_programs'] = program_count
