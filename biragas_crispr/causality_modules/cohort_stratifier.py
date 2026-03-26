"""
Phase 6: COMPARATIVE EVOLUTION — Module 1
CohortStratifier (INTENT I_04 Module 1)
=========================================
Stratifies patient cohorts by molecular causal architecture.

Stratification uses patient-level DAGs built in Phase 1:
  1. Feature extraction from each patient DAG (active genes, edges, pathways)
  2. Distance computation (Jaccard on edge sets, spectral distance)
  3. Hierarchical/k-medoids clustering
  4. Subgroup characterization (dominant drivers, pathway profiles)

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StratifierConfig:
    n_clusters: int = 3
    min_cluster_size: int = 5
    distance_metric: str = 'jaccard'
    feature_type: str = 'edges'  # 'edges', 'nodes', 'combined'
    linkage_method: str = 'ward'


class CohortStratifier:
    """Stratifies patients into molecular subtypes based on causal DAG architecture."""

    def __init__(self, config: Optional[StratifierConfig] = None):
        self.config = config or StratifierConfig()

    def stratify(self, patient_dags: Dict[str, nx.DiGraph],
                 consensus_dag: Optional[nx.DiGraph] = None) -> Dict:
        """Stratify patients based on their individual causal DAGs."""
        if len(patient_dags) < self.config.min_cluster_size:
            return {'error': 'Too few patients for stratification',
                    'n_patients': len(patient_dags)}

        features = self._extract_features(patient_dags)
        distance_matrix = self._compute_distances(features)
        labels = self._cluster(distance_matrix, len(patient_dags))

        patient_ids = list(patient_dags.keys())
        subgroups = {}
        for i, pid in enumerate(patient_ids):
            cluster = int(labels[i])
            if cluster not in subgroups:
                subgroups[cluster] = []
            subgroups[cluster].append(pid)

        profiles = {}
        for cluster_id, members in subgroups.items():
            member_dags = {pid: patient_dags[pid] for pid in members}
            profiles[cluster_id] = self._characterize_subgroup(member_dags, consensus_dag)

        return {
            'n_patients': len(patient_dags),
            'n_subgroups': len(subgroups),
            'subgroups': {k: {'members': v, 'size': len(v)} for k, v in subgroups.items()},
            'profiles': profiles,
            'patient_assignments': {pid: int(labels[i]) for i, pid in enumerate(patient_ids)},
        }

    def _extract_features(self, patient_dags: Dict[str, nx.DiGraph]) -> Dict[str, Set]:
        """Extract feature sets from patient DAGs."""
        features = {}
        for pid, dag in patient_dags.items():
            if self.config.feature_type == 'nodes':
                features[pid] = set(dag.nodes())
            elif self.config.feature_type == 'combined':
                features[pid] = set(dag.nodes()) | {f"{u}->{v}" for u, v in dag.edges()}
            else:
                features[pid] = {f"{u}->{v}" for u, v in dag.edges()}
        return features

    def _compute_distances(self, features: Dict[str, Set]) -> np.ndarray:
        """Compute pairwise distance matrix."""
        patients = list(features.keys())
        n = len(patients)
        dist = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                fi = features[patients[i]]
                fj = features[patients[j]]
                union = fi | fj
                if len(union) == 0:
                    d = 1.0
                else:
                    d = 1.0 - len(fi & fj) / len(union)
                dist[i, j] = d
                dist[j, i] = d

        return dist

    def _cluster(self, dist_matrix: np.ndarray, n_patients: int) -> np.ndarray:
        """Simple k-medoids clustering on distance matrix."""
        k = min(self.config.n_clusters, n_patients // self.config.min_cluster_size)
        k = max(2, k)

        np.random.seed(42)
        medoids = np.random.choice(n_patients, size=k, replace=False)
        labels = np.zeros(n_patients, dtype=int)

        for _ in range(50):
            for i in range(n_patients):
                dists_to_medoids = [dist_matrix[i, m] for m in medoids]
                labels[i] = int(np.argmin(dists_to_medoids))

            new_medoids = []
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    new_medoids.append(medoids[c])
                    continue
                sub_dist = dist_matrix[np.ix_(members, members)]
                total_dists = sub_dist.sum(axis=1)
                best = members[np.argmin(total_dists)]
                new_medoids.append(best)

            new_medoids = np.array(new_medoids)
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        return labels

    def _characterize_subgroup(self, member_dags: Dict[str, nx.DiGraph],
                                consensus_dag: Optional[nx.DiGraph]) -> Dict:
        """Characterize a patient subgroup by its common causal features."""
        all_genes = {}
        all_edges = {}
        all_pathways = {}

        for pid, dag in member_dags.items():
            for node, data in dag.nodes(data=True):
                if data.get('layer') == 'regulatory':
                    all_genes[node] = all_genes.get(node, 0) + 1
                elif data.get('layer') == 'program':
                    all_pathways[node] = all_pathways.get(node, 0) + 1
            for u, v in dag.edges():
                key = f"{u}->{v}"
                all_edges[key] = all_edges.get(key, 0) + 1

        n = len(member_dags)
        threshold = 0.5

        dominant_genes = sorted(
            [(g, c / n) for g, c in all_genes.items() if c / n >= threshold],
            key=lambda x: x[1], reverse=True
        )
        dominant_pathways = sorted(
            [(p, c / n) for p, c in all_pathways.items() if c / n >= threshold],
            key=lambda x: x[1], reverse=True
        )

        return {
            'n_members': n,
            'dominant_genes': [{'gene': g, 'frequency': round(f, 3)} for g, f in dominant_genes[:10]],
            'dominant_pathways': [{'pathway': p, 'frequency': round(f, 3)} for p, f in dominant_pathways[:10]],
            'mean_edges': round(np.mean([dag.number_of_edges() for dag in member_dags.values()]), 1),
            'mean_nodes': round(np.mean([dag.number_of_nodes() for dag in member_dags.values()]), 1),
        }
