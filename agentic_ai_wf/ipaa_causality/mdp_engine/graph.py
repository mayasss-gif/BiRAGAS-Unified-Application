# mdp_engine/graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .exceptions import ValidationError


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    interaction: str = "activation"  # activation/inhibition/other


class SimpleDiGraph:
    """
    Minimal directed graph for topology methods (PageRank weighting etc.).
    """

    def __init__(self) -> None:
        self._nodes: Set[str] = set()
        self._out: Dict[str, List[Edge]] = {}
        self._in: Dict[str, List[Edge]] = {}

    def add_node(self, node: str) -> None:
        node = str(node).strip()
        if not node:
            raise ValidationError("node cannot be empty")
        if node in self._nodes:
            return
        self._nodes.add(node)
        self._out.setdefault(node, [])
        self._in.setdefault(node, [])

    def add_edge(self, src: str, dst: str, interaction: str = "activation") -> None:
        src = str(src).strip()
        dst = str(dst).strip()
        if not src or not dst:
            raise ValidationError("src/dst cannot be empty")

        self.add_node(src)
        self.add_node(dst)
        e = Edge(src=src, dst=dst, interaction=str(interaction or "activation"))
        self._out[src].append(e)
        self._in[dst].append(e)

    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def successors(self, node: str) -> List[str]:
        node = str(node)
        return [e.dst for e in self._out.get(node, [])]

    def predecessors(self, node: str) -> List[Edge]:
        node = str(node)
        return list(self._in.get(node, []))

    def out_degree(self, node: str) -> int:
        node = str(node)
        return len(self._out.get(node, []))

    def subgraph(self, nodes: Set[str]) -> "SimpleDiGraph":
        sg = SimpleDiGraph()
        keep = set(str(x) for x in nodes)
        for n in keep:
            if n in self._nodes:
                sg.add_node(n)
        for src in keep:
            for e in self._out.get(src, []):
                if e.dst in keep:
                    sg.add_edge(e.src, e.dst, e.interaction)
        return sg
