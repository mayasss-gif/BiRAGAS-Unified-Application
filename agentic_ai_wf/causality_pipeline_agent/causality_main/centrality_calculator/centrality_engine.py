import networkx as nx
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[CentralityEngine] %(message)s')

class CentralityCalculator:
    def __init__(self, dag: nx.DiGraph, disease_name: str):
        self.G = dag.copy()
        self.disease_name = disease_name
        self.target_node = f"{disease_name.replace(' ', '_')}_Disease_Activity"

        self.metrics_report = {
            "meta": {
                "module": "CentralityCalculator",
                "disease": disease_name,
                "total_scored_genes": 0,
                "total_scored_programs": 0
            },
            "hub_classifications": {
                "Tier_1_Master_Regulators": [],
                "Tier_2_Secondary_Drivers": [],
                "Tier_3_Downstream_Effectors": []
            },
            "gene_scores": {},
            "program_scores": {}
        }

        self._preprocess_edges()

    def _preprocess_edges(self):
        """Converts biological confidence into mathematical costs and capacities."""
        for u, v, d in self.G.edges(data=True):
            conf = d.get('confidence_score', 0.1)
            weight = d.get('weight', 0.1)
            d['distance'] = max(0.01, 1.0 - conf)
            d['capacity'] = conf * weight

    def run_pipeline(self):
        logging.info("Calculating Topological Dominance & Apex Scores...")
        self._calculate_degrees()

        logging.info("Calculating Probabilistic Betweenness Centrality...")
        self._calculate_betweenness()

        logging.info("Running Reverse Personalized PageRank (Disease Proximity)...")
        self._calculate_reverse_ppr()

        logging.info("Traversing Super-Nodes for Systemic Pleiotropy...")
        self._calculate_pleiotropy()

        logging.info("Arbitrating Strict 3-Tier Classifications...")
        self._arbitrate_tiers()

        return self.G, self.metrics_report

    def _calculate_degrees(self):
        for node in self.G.nodes():
            in_edges = self.G.in_edges(node, data=True)
            out_edges = self.G.out_edges(node, data=True)

            w_in = sum([d.get('capacity', 0) for u, v, d in in_edges])
            w_out = sum([d.get('capacity', 0) for u, v, d in out_edges])

            out_weights = [d.get('weight', 0) for u, v, d in out_edges]
            causal_strength = float(np.mean(out_weights)) if out_weights else 0.0

            apex = w_out / (w_in + w_out + 1e-9)

            nx.set_node_attributes(self.G, {node: {
                'w_in': round(w_in, 3),
                'w_out': round(w_out, 3),
                'apex_score': round(apex, 3),
                'causal_strength': round(causal_strength, 3)
            }})

    def _calculate_betweenness(self):
        betweenness = nx.betweenness_centrality(self.G, weight='distance', normalized=True)
        nx.set_node_attributes(self.G, {n: {'betweenness': round(b, 4)} for n, b in betweenness.items()})

    def _calculate_reverse_ppr(self):
        if self.target_node not in self.G.nodes():
            logging.warning(f"Target node '{self.target_node}' not found. Skipping PPR.")
            nx.set_node_attributes(self.G, 0.0, 'disease_proximity')
            return

        G_rev = self.G.reverse(copy=True)
        try:
            ppr = nx.pagerank(G_rev, alpha=0.85, personalization={self.target_node: 1.0}, weight='capacity')
            nx.set_node_attributes(self.G, {n: {'disease_proximity': round(p, 5)} for n, p in ppr.items()})
        except Exception as e:
            logging.error(f"PPR Calculation failed: {e}")
            nx.set_node_attributes(self.G, 0.0, 'disease_proximity')

    def _calculate_pleiotropy(self):
        for node, d in self.G.nodes(data=True):
            if d.get('layer') != 'regulatory':
                continue
            unique_classes = set()
            for succ in self.G.successors(node):
                succ_data = self.G.nodes[succ]
                if succ_data.get('layer') == 'program':
                    main_class = succ_data.get('main_class', 'Unclassified')
                    if main_class != 'Unclassified':
                        unique_classes.add(main_class)
            self.G.nodes[node]['pleiotropic_reach'] = len(unique_classes)

    def _arbitrate_tiers(self):
        gene_count = 0
        program_count = 0

        all_betweenness = [d.get('betweenness', 0) for n, d in self.G.nodes(data=True) if d.get('layer') == 'regulatory']
        b_85th_percentile = np.percentile(all_betweenness, 85) if all_betweenness else 0.0

        for node, d in self.G.nodes(data=True):
            layer = d.get('layer')

            if layer == 'program':
                program_count += 1
                self.metrics_report['program_scores'][node] = {
                    "main_class": d.get('main_class'),
                    "metrics": {
                        "weighted_in_degree": d.get('w_in'),
                        "betweenness": d.get('betweenness'),
                        "disease_proximity": d.get('disease_proximity')
                    }
                }
                continue

            if layer == 'regulatory':
                gene_count += 1
                apex = d.get('apex_score', 0)
                betweenness = d.get('betweenness', 0)
                pleiotropy = d.get('pleiotropic_reach', 0)
                ppr = d.get('disease_proximity', 0)
                causal_strength = d.get('causal_strength', 0)

                is_prior = d.get('causal_tier') in ["Validated Driver"]
                has_direct_edge = self.G.has_edge(node, self.target_node)

                causal_importance = (apex * causal_strength) + betweenness + (ppr * 10)
                self.G.nodes[node]['causal_importance'] = round(causal_importance, 4)

                tier = "Tier_2_Secondary_Driver"

                if apex >= 0.60 and pleiotropy >= 1 and is_prior:
                    tier = "Tier_1_Master_Regulator"
                elif (apex <= 0.30 or has_direct_edge or ppr > 0.001) and (betweenness < b_85th_percentile):
                    tier = "Tier_3_Downstream_Effector"
                else:
                    tier = "Tier_2_Secondary_Driver"

                self.G.nodes[node]['network_tier'] = tier
                self.metrics_report['hub_classifications'][tier + "s"].append(node)

                self.metrics_report['gene_scores'][node] = {
                    "assigned_network_tier": tier,
                    "metrics": {
                        "weighted_out_degree": d.get('w_out'),
                        "weighted_in_degree": d.get('w_in'),
                        "apex_cascade_position": apex,
                        "probabilistic_betweenness": betweenness,
                        "pleiotropic_reach_classes": pleiotropy,
                        "causal_strength": causal_strength,
                        "causal_importance_score": round(causal_importance, 4),
                    },
                }

        for tier_key in self.metrics_report['hub_classifications']:
            self.metrics_report['hub_classifications'][tier_key].sort(
                key=lambda x: self.G.nodes[x]['causal_importance'], reverse=True
            )

        self.metrics_report['meta']['total_scored_genes'] = gene_count
        self.metrics_report['meta']['total_scored_programs'] = program_count
