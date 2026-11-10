"""Fast ego-graph extraction.

The original BAED code built one huge networkx graph (``dgl.to_networkx``) and then ran
``nx.single_source_shortest_path_length`` + ``G.subgraph(...)`` once per center node. On
DGraphFin (3.7M nodes) that is the bottleneck of the whole preprocessing step.

Here the graph is kept as a CSR array and the k-hop extraction runs in Rust
(``baed_graph`` crate, PyO3 + rayon, all cores). A pure-Python fallback with identical
output is used when the extension is not installed.

Semantics are the same as the original code:

* BFS follows **out**-edges of the directed graph (networkx ``MultiDiGraph`` behaviour).
  With ``bidirectional=True`` it follows edges in both directions instead (in- and
  out-neighbours, i.e. the k-hop neighbourhood of the undirected graph).
* The ego-graph keeps every edge between kept nodes, merged into an undirected
  ``nx.Graph`` (self-loops kept).
* Node attributes ``feature`` (float tensor [F]) and ``label`` (0-d int tensor), and the
  graph attribute ``sublabel`` (int, label of the center node).

Only the *iteration order* of nodes/edges inside each ``nx.Graph`` can differ from the
original (nodes are sorted by id here, the original followed Python ``set`` order). This
is a node permutation and does not change the graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import torch
from loguru import logger

try:
    from baed_graph import ego_subgraphs as _rust_ego_subgraphs

    HAS_RUST = True
except ImportError:  # pragma: no cover - depends on the build environment
    _rust_ego_subgraphs = None
    HAS_RUST = False


def to_csr(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-edge CSR (``indptr``, ``indices``) of a directed edge list."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    order = np.argsort(src, kind="stable")
    indices = dst[order]
    counts = np.bincount(src, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.ascontiguousarray(indices)


def _py_ego_subgraphs(indptr, indices, centers, hops, max_nodes=None):
    """Reference implementation, same output layout as the Rust kernel."""
    kept, node_ptr, edge_ptr, nodes_all, edges_all = [], [0], [0], [], []
    for c in centers:
        seen = {int(c)}
        frontier = [int(c)]
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                nxt.update(indices[indptr[u] : indptr[u + 1]].tolist())
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = list(nxt)
        if max_nodes is not None and len(seen) > max_nodes:
            kept.append(False)
            continue
        nodes = sorted(seen)
        edges = set()
        for u in nodes:
            for v in indices[indptr[u] : indptr[u + 1]].tolist():
                if v in seen:
                    edges.add((u, v) if u <= v else (v, u))
        kept.append(True)
        nodes_all.extend(nodes)
        node_ptr.append(len(nodes_all))
        for e in sorted(edges):
            edges_all.extend(e)
        edge_ptr.append(len(edges_all) // 2)
    return (
        np.asarray(kept, dtype=bool),
        np.asarray(node_ptr, dtype=np.int64),
        np.asarray(nodes_all, dtype=np.int64),
        np.asarray(edge_ptr, dtype=np.int64),
        np.asarray(edges_all, dtype=np.int64),
    )


def ego_subgraphs(indptr, indices, centers, hops: int, max_nodes: int | None = None, backend: str = "auto"):
    """k-hop ego-graphs of ``centers``. Returns ``(kept, node_ptr, nodes, edge_ptr, edges)``."""
    centers = np.ascontiguousarray(centers, dtype=np.int64)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    use_rust = backend == "rust" or (backend == "auto" and HAS_RUST)
    if use_rust:
        if not HAS_RUST:
            raise ImportError("baed_graph Rust extension is not installed")
        return _rust_ego_subgraphs(indptr, indices, centers, hops, max_nodes)
    return _py_ego_subgraphs(indptr, indices, centers, hops, max_nodes)


def build_ego_graphs(
    src,
    dst,
    num_nodes: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    centers: Sequence[int],
    hops: int,
    max_nodes: int | None = None,
    backend: str = "auto",
    center_labels: torch.Tensor | None = None,
    bidirectional: bool = False,
) -> list[nx.Graph]:
    """Extract ego-graphs around ``centers`` and return them as attributed ``nx.Graph`` objects.

    ``labels`` is stored per node (``label`` attribute); ``center_labels`` (default ``labels``)
    gives ``graph['sublabel']``. They differ for tfinance, whose node labels are one-hot.
    ``bidirectional``: BFS over in- and out-edges (the original code follows out-edges only).
    """
    if center_labels is None:
        center_labels = labels
    src, dst = np.asarray(src), np.asarray(dst)
    if bidirectional:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    indptr, indices = to_csr(src, dst, num_nodes)
    centers = np.asarray(list(centers), dtype=np.int64)
    logger.info(
        f"ego-graphs: {len(centers)} centers, {hops} hop(s), {'both directions' if bidirectional else 'out-edges'}, backend="
        f"{'rust' if (backend != 'python' and HAS_RUST) else 'python'}"
    )
    kept, node_ptr, nodes, edge_ptr, edges = ego_subgraphs(indptr, indices, centers, hops, max_nodes, backend)
    edges = edges.reshape(-1, 2)
    kept_centers = centers[kept]
    if (~kept).any():
        logger.info(f"skipped {(~kept).sum()} ego-graphs larger than {max_nodes} nodes")

    graphs = []
    for j, c in enumerate(kept_centers):
        nd = nodes[node_ptr[j] : node_ptr[j + 1]]
        ed = edges[edge_ptr[j] : edge_ptr[j + 1]]
        idx = torch.from_numpy(nd)
        feat = features[idx]
        lab = labels[idx]
        g = nx.Graph()
        g.add_nodes_from((int(n), {"feature": feat[i], "label": lab[i]}) for i, n in enumerate(nd.tolist()))
        # dgl.to_networkx stores an edge attribute ``id``. Downstream code relies on it
        # existing (``del pyg_data.id`` after from_networkx), never on its value.
        g.add_edges_from((u, v, {"id": k}) for k, (u, v) in enumerate(ed.tolist()))
        g.graph["sublabel"] = int(center_labels[int(c)].item())
        graphs.append(g)
    return graphs
