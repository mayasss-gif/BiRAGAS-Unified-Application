"""
QualityAgent — Input Validation + Self-Correction
===================================================
Validates input data quality before engine execution.
Fixes DAG integrity issues automatically.
"""
import logging
from typing import Any, Dict, List, Tuple
import networkx as nx

logger = logging.getLogger("biragas.crispr.quality")

class QualityAgent:
    """Input validation and DAG self-correction."""

    def validate_dag(self, dag: nx.DiGraph) -> Tuple[bool, List[str]]:
        issues = []
        if not nx.is_directed_acyclic_graph(dag):
            issues.append("CRITICAL: Graph contains cycles")
        if dag.number_of_nodes() == 0:
            issues.append("CRITICAL: Empty graph")
        if dag.number_of_edges() == 0 and dag.number_of_nodes() > 1:
            issues.append("CRITICAL: No edges")
        trait_nodes = [n for n in dag.nodes() if dag.nodes[n].get('layer') == 'trait']
        if not trait_nodes:
            issues.append("WARNING: No disease (trait) node")
        reg_nodes = [n for n in dag.nodes() if dag.nodes[n].get('layer') == 'regulatory']
        if len(reg_nodes) == 0:
            issues.append("CRITICAL: No regulatory (gene) nodes")
        low_conf = sum(1 for _,_,d in dag.edges(data=True) if isinstance(d.get('confidence'), (int,float)) and d['confidence'] < 0.3)
        if low_conf > 0:
            issues.append(f"INFO: {low_conf} edges below confidence 0.3")
        return not any(i.startswith("CRITICAL") for i in issues), issues

    def auto_fix(self, dag: nx.DiGraph) -> Tuple[nx.DiGraph, List[str]]:
        fixes = []
        # Fix cycles
        attempts = 0
        while not nx.is_directed_acyclic_graph(dag) and attempts < 50:
            try:
                cycle = list(nx.find_cycle(dag))
                weakest = min(cycle, key=lambda e: dag.edges[e[0], e[1]].get('confidence', 0.5) if isinstance(dag.edges[e[0], e[1]].get('confidence'), (int,float)) else 0.5)
                dag.remove_edge(weakest[0], weakest[1])
                fixes.append(f"Removed cycle edge: {weakest[0]}→{weakest[1]}")
            except nx.NetworkXNoCycle:
                break
            attempts += 1
        # Fix orphans
        orphans = [n for n in dag.nodes() if dag.degree(n) == 0 and dag.nodes[n].get('layer') != 'trait']
        for o in orphans:
            dag.remove_node(o)
            fixes.append(f"Removed orphan: {o}")
        # Fix non-numeric confidence
        for u, v, d in dag.edges(data=True):
            c = d.get('confidence')
            if c is not None and not isinstance(c, (int, float)):
                try: dag.edges[u,v]['confidence'] = float(c)
                except: dag.edges[u,v]['confidence'] = 0.5; fixes.append(f"Fixed confidence: {u}→{v}")
        return dag, fixes

    def validate_screening(self, screening_results: Dict) -> Tuple[bool, List[str]]:
        issues = []
        if not screening_results:
            issues.append("CRITICAL: No screening results")
            return False, issues
        n = len(screening_results)
        if n < 100:
            issues.append(f"WARNING: Only {n} genes — expected 1000+")
        with_ace = sum(1 for g in screening_results.values() if hasattr(g, 'ace_score') and g.ace_score != 0)
        if with_ace == 0:
            issues.append("CRITICAL: No genes have ACE scores")
        return not any(i.startswith("CRITICAL") for i in issues), issues
