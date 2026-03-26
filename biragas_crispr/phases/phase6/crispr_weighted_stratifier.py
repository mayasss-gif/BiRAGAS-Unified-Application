"""
CRISPRWeightedStratifier — CRISPR-Weighted Patient Stratification
====================================================================
Gap: CohortStratifier uses binary Jaccard (edge present/absent).
CRISPR-validated edges and purely computational edges contribute equally.

Fix: Weighted Jaccard where each edge is weighted by CRISPR evidence:
    w(edge) = 1.0 + crispr_bonus
    crispr_bonus = 0.5 if both endpoints have ACE ≤ -0.1
                 = 0.3 if one endpoint has ACE ≤ -0.1
                 = 0.0 otherwise

Weighted Jaccard: d(A,B) = 1 - Σ min(wA, wB) / Σ max(wA, wB)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger("biragas.crispr_phase6.stratifier")


class CRISPRWeightedStratifier:
    """
    CRISPR-weighted patient stratification using weighted Jaccard distance.

    Patients sharing CRISPR-validated edges cluster more tightly than
    patients sharing only computational edges.

    Usage:
        stratifier = CRISPRWeightedStratifier()
        results = stratifier.stratify(patient_dags, consensus_dag)
    """

    def __init__(self, n_clusters: int = 3, ace_threshold: float = -0.1,
                 both_ace_bonus: float = 0.5, one_ace_bonus: float = 0.3):
        self.n_clusters = n_clusters
        self.ace_threshold = ace_threshold
        self.both_ace_bonus = both_ace_bonus
        self.one_ace_bonus = one_ace_bonus

    def stratify(self, patient_dags: Dict[str, nx.DiGraph],
                 consensus_dag: Optional[nx.DiGraph] = None) -> Dict:
        """
        Stratify patients using CRISPR-weighted Jaccard distance.
        """
        if isinstance(patient_dags, list):
            patient_dags = {f"p_{i}": d for i, d in enumerate(patient_dags)}

        patient_ids = list(patient_dags.keys())
        n = len(patient_ids)

        if n < 2:
            return {'error': 'Need at least 2 patient DAGs', 'subgroups': {}}

        # Extract CRISPR-weighted edge features
        features = {}
        for pid, dag in patient_dags.items():
            edge_weights = {}
            for u, v in dag.edges():
                u_ace = dag.nodes[u].get('perturbation_ace', 0)
                v_ace = dag.nodes[v].get('perturbation_ace', 0)
                u_has = isinstance(u_ace, (int, float)) and u_ace <= self.ace_threshold
                v_has = isinstance(v_ace, (int, float)) and v_ace <= self.ace_threshold

                w = 1.0  # Base weight
                if u_has and v_has:
                    w += self.both_ace_bonus
                elif u_has or v_has:
                    w += self.one_ace_bonus

                edge_weights[f"{u}->{v}"] = w
            features[pid] = edge_weights

        # Compute CRISPR-weighted Jaccard distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                fi = features[patient_ids[i]]
                fj = features[patient_ids[j]]
                all_edges = set(fi.keys()) | set(fj.keys())

                if not all_edges:
                    dist_matrix[i, j] = dist_matrix[j, i] = 1.0
                    continue

                min_sum = sum(min(fi.get(e, 0), fj.get(e, 0)) for e in all_edges)
                max_sum = sum(max(fi.get(e, 0), fj.get(e, 0)) for e in all_edges)

                jaccard = min_sum / max_sum if max_sum > 0 else 0
                dist_matrix[i, j] = dist_matrix[j, i] = 1.0 - jaccard

        # K-medoids clustering
        k = min(self.n_clusters, n // 2)
        k = max(2, k)
        labels = self._kmedoids(dist_matrix, k)

        # Build subgroups
        subgroups = {}
        for cluster_id in range(k):
            members = [patient_ids[i] for i in range(n) if labels[i] == cluster_id]
            if not members:
                continue

            # Characterize subgroup
            member_dags = {pid: patient_dags[pid] for pid in members}
            profile = self._characterize(member_dags)

            subgroups[f"subgroup_{cluster_id}"] = {
                'patient_ids': members,
                'size': len(members),
                'profile': profile,
            }

        # Compare with standard (unweighted) to show CRISPR impact
        report = {
            'n_patients': n,
            'n_subgroups': len(subgroups),
            'subgroups': subgroups,
            'method': 'crispr_weighted_jaccard',
            'ace_threshold': self.ace_threshold,
            'weights_used': {'both_ace': self.both_ace_bonus, 'one_ace': self.one_ace_bonus},
        }

        logger.info(f"CRISPRWeightedStratifier: {n} patients → {len(subgroups)} subgroups")
        return report

    def _kmedoids(self, dist, k, max_iter=50):
        """K-medoids clustering on distance matrix."""
        n = dist.shape[0]
        rng = np.random.RandomState(42)
        medoids = rng.choice(n, k, replace=False)
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign
            for i in range(n):
                labels[i] = min(range(k), key=lambda m: dist[i, medoids[m]])
            # Update medoids
            changed = False
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                costs = [sum(dist[m, j] for j in members) for m in members]
                new_medoid = members[np.argmin(costs)]
                if new_medoid != medoids[c]:
                    medoids[c] = new_medoid
                    changed = True
            if not changed:
                break
        return labels

    def _characterize(self, member_dags: Dict) -> Dict:
        """Characterize a subgroup by dominant genes and CRISPR drivers."""
        gene_freq = {}
        crispr_drivers = {}

        for pid, dag in member_dags.items():
            for n in dag.nodes():
                if dag.nodes[n].get('layer') == 'regulatory':
                    gene_freq[n] = gene_freq.get(n, 0) + 1
                    ace = dag.nodes[n].get('perturbation_ace', 0)
                    if isinstance(ace, (int, float)) and ace <= self.ace_threshold:
                        crispr_drivers[n] = crispr_drivers.get(n, 0) + 1

        n_members = len(member_dags)
        dominant = {g: c / n_members for g, c in gene_freq.items() if c / n_members >= 0.5}
        dominant_crispr = {g: c / n_members for g, c in crispr_drivers.items() if c / n_members >= 0.5}

        return {
            'n_genes': len(gene_freq),
            'dominant_genes': dict(sorted(dominant.items(), key=lambda x: -x[1])[:10]),
            'crispr_drivers': dict(sorted(dominant_crispr.items(), key=lambda x: -x[1])[:10]),
            'n_crispr_drivers': len(crispr_drivers),
        }
