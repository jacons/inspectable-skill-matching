from ast import literal_eval
from pathlib import Path

import networkx as nx
import pandas as pd

_ROOT = "\x00root"


def taxonomy_levels(tax: pd.DataFrame) -> tuple[dict[str, int], dict[str, bool]]:
    """(level, leaf) of every node of a taxonomy given as conceptUri -> broaderUri rows.

    level is 1 at a root and then the shortest distance from a root; leaf means no narrower node.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(zip(tax["broaderUri"], tax["conceptUri"]))
    leaf = {node: out == 0 for node, out in graph.out_degree()}
    roots = [node for node, incoming in graph.in_degree() if incoming == 0]
    graph.add_edges_from((_ROOT, root) for root in roots)
    level = nx.single_source_shortest_path_length(graph, _ROOT)
    del level[_ROOT]
    return level, leaf


def _nodes(csv: Path, index: str, type_: str) -> pd.Series:
    df = pd.read_csv(csv, converters={"altLabels": literal_eval}, index_col=index)
    return df.apply(lambda row: {"type": type_, **row.to_dict()}, axis=1)


def load_graph_file(processed_dir: Path) -> tuple[pd.Series, list[tuple]]:
    """Nodes (attributes indexed by id) and edges (id_a, id_b, attributes) read from `processed_dir`."""
    nodes = pd.concat([_nodes(processed_dir / "occupations.csv", "occ_id", "occupation"),
                       _nodes(processed_dir / "skills.csv", "skill_id", "skill")])
    edges = []
    for name in ("occupation_tax.csv", "skill_tax.csv"):
        tax = pd.read_csv(processed_dir / name)
        edges += [(a, b, {}) for a, b in zip(tax["conceptUri"], tax["broaderUri"])]
    rel = pd.read_csv(processed_dir / "occ_skill_rel.csv")
    edges += [(a, b, {"relationType": t})
              for a, b, t in zip(rel["occupationUri"], rel["skillUri"], rel["relationType"])]
    if not nodes.index.is_unique:
        raise ValueError(f"node ids used twice (occupation and skill ids must differ): {sorted(set(nodes.index[nodes.index.duplicated()]))[:5]}")
    if len({frozenset((a, b)) for a, b, _ in edges}) != len(edges):
        raise ValueError("an edge is listed twice")
    missing = {n for a, b, _ in edges for n in (a, b)} - set(nodes.index)
    if missing:
        raise ValueError(f"{len(missing)} edge endpoints are not in the node files, e.g. {sorted(missing)[:5]}")
    return nodes, edges
