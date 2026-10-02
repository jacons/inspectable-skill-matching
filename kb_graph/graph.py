import math
import pickle
from pathlib import Path

import networkx as nx

from .loaders import load_graph_file


class KBGraph:
    """Undirected graph of a knowledge base: occupation and skill nodes, taxonomy edges and
    occupation-skill edges. README.md describes the input files."""

    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self._subgraphs = {}
        self._label_to_uris = {}
        for node, attrs in graph.nodes(data=True):
            self._label_to_uris.setdefault(attrs["label"], []).append(node)

    @classmethod
    def from_processed(cls, processed_dir: Path) -> "KBGraph":
        """Build the graph from the CSV files in `processed_dir`."""
        nodes, edges = load_graph_file(processed_dir)
        graph = nx.Graph()
        graph.add_nodes_from(nodes.items())
        graph.add_edges_from(edges)
        return cls(graph)

    @classmethod
    def load(cls, path: Path) -> "KBGraph":
        with open(path, "rb") as fh:
            return cls(nx.node_link_graph(pickle.load(fh)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(nx.node_link_data(self._graph), fh)

    def node_info(self, uri: str) -> dict:
        return self._graph.nodes[uri]

    def get_label(self, uri: str) -> str:
        return self._graph.nodes[uri]["label"]

    def get_level(self, uri: str) -> int:
        return self._graph.nodes[uri]["level"]

    def get_uri(self, label, reduce="Top"):
        """Id of the node with this label (a generator of ids for an iterable of labels).
        When several nodes share the label, "Top" picks the shallowest, "Down" the deepest."""
        if not isinstance(label, str):
            return (self.get_uri(l, reduce) for l in label)
        pick = {"Top": min, "Down": max}[reduce]
        return pick(self._label_to_uris[label], key=self.get_level)

    def nodes(self, properties: dict | None = None):
        """(id, attributes) of the nodes whose attributes match every (key, value) in `properties`."""
        properties = properties or {}
        return ((n, a) for n, a in self._graph.nodes(data=True)
                if all(a.get(k) == v for k, v in properties.items()))

    def neighbors(self, uri: str, properties: dict | None = None):
        """Ids of the neighbours of `uri` whose attributes match every (key, value) in `properties`."""
        properties = properties or {}
        return (n for n in self._graph.neighbors(uri)
                if all(self._graph.nodes[n].get(k) == v for k, v in properties.items()))

    def path(self, uri_a: str, uri_b: str, labeled: bool = False, intermediate_properties: dict | None = None) -> list:
        """Shortest path from uri_a to uri_b; with `intermediate_properties`, every node between
        the two ends must match them."""
        graph = self._graph
        if intermediate_properties is not None:
            key = frozenset(intermediate_properties.items())
            if key not in self._subgraphs:
                allowed = {n for n, _ in self.nodes(intermediate_properties)}
                self._subgraphs[key] = (allowed, graph.subgraph(allowed))
            allowed, graph = self._subgraphs[key]
            if uri_a not in allowed or uri_b not in allowed:
                graph = self._graph.subgraph(allowed | {uri_a, uri_b})
        path = nx.shortest_path(graph, uri_a, uri_b)
        return [self.get_label(n) for n in path] if labeled else path

    def similarity(self, uri_a: str, uri_b: str, how: str = "hop", intermediate_properties: dict | None = None) -> float:
        """how="hop": edges on the shortest path (0 = same node, inf = no path; lower = closer).
        how="neighbor": Jaccard overlap of the neighbours of the other type (the occupations of
        two skills, the skills of two occupations), in [0, 1]; higher = closer."""
        if how == "neighbor":
            other = "occupation" if self.node_info(uri_a)["type"] == "skill" else "skill"
            n_a = set(self.neighbors(uri_a, {"type": other}))
            n_b = set(self.neighbors(uri_b, {"type": other}))
            return len(n_a & n_b) / len(n_a | n_b) if n_a else 0.0
        if how != "hop":
            raise ValueError(f"how must be 'hop' or 'neighbor', not {how!r}")
        try:
            return float(len(self.path(uri_a, uri_b, intermediate_properties=intermediate_properties)) - 1)
        except nx.NetworkXNoPath:
            return math.inf

    def is_language(self, uri: str | None = None, label: str | None = None) -> bool:
        """True for a language skill (skill_type == "language")."""
        if label is not None:
            uri = self.get_uri(label)
        return self.node_info(uri).get("skill_type") == "language"

    def generalize_lang(self, uri: str | None = None, label: str | None = None) -> str:
        """Label of the topmost language ancestor of a language skill."""
        if label is not None:
            uri = self.get_uri(label)
        while parent := next(self.neighbors(uri, {"skill_type": "language", "level": self.get_level(uri) - 1}), None):
            uri = parent
        return self.get_label(uri)

    def ancestor_at_level(self, uri: str, level: int, labeled: bool = False) -> str:
        """Ancestor of `uri` at `level` in its own hierarchy, climbing one level at a time (a tree)."""
        while self.get_level(uri) > level:
            uri = next(self.neighbors(uri, {"type": self.node_info(uri)["type"], "level": self.get_level(uri) - 1}))
        return self.get_label(uri) if labeled else uri
