"""Rust ego-graph kernel == Python fallback == original networkx procedure."""

import dgl
import networkx as nx
import numpy as np
import pytest
import torch

from baed import graph_ops


def random_graph(n=300, m=900, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    # a few self-loops and duplicates, like real data
    src[:5] = dst[:5]
    src[5:10], dst[5:10] = src[10:15], dst[10:15]
    return src.astype(np.int64), dst.astype(np.int64), n


def reference(src, dst, n, centers, hops, max_nodes=None):
    """The original BAED procedure (createSubGraphDataset/createDataset.py)."""
    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=n)
    g.ndata["feature"] = torch.randn(n, 4)
    g.ndata["label"] = torch.randint(0, 2, (n,))
    G = dgl.to_networkx(g, node_attrs=["feature", "label"])
    out = []
    for c in centers:
        nodes = set(nx.single_source_shortest_path_length(G, c, cutoff=hops).keys())
        if max_nodes is not None and len(nodes) > max_nodes:
            continue
        out.append(nx.Graph(G.subgraph(nodes).copy()))
    return out, g


@pytest.mark.parametrize("hops", [1, 2])
def test_python_backend_matches_networkx(hops):
    src, dst, n = random_graph()
    centers = list(range(0, n, 7))
    ref, g = reference(src, dst, n, centers, hops)
    got = graph_ops.build_ego_graphs(src, dst, n, g.ndata["feature"], g.ndata["label"], centers, hops, backend="python")
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert set(a.nodes) == set(b.nodes)
        assert {frozenset(e) for e in a.edges} == {frozenset(e) for e in b.edges}
        for v in a.nodes:
            assert torch.equal(a.nodes[v]["feature"], b.nodes[v]["feature"])
            assert int(a.nodes[v]["label"]) == int(b.nodes[v]["label"])
        assert all("id" in d for _, _, d in a.edges(data=True))


def test_max_nodes_filter_matches_networkx():
    src, dst, n = random_graph()
    centers = list(range(n))
    ref, g = reference(src, dst, n, centers, hops=1, max_nodes=4)
    got = graph_ops.build_ego_graphs(
        src, dst, n, g.ndata["feature"], g.ndata["label"], centers, 1, max_nodes=4, backend="python"
    )
    assert [sorted(x.nodes) for x in got] == [sorted(x.nodes) for x in ref]


@pytest.mark.skipif(not graph_ops.HAS_RUST, reason="baed_graph extension not built")
@pytest.mark.parametrize("hops,max_nodes", [(1, None), (2, None), (2, 15), (3, None)])
def test_rust_matches_python(hops, max_nodes):
    src, dst, n = random_graph(n=2000, m=8000, seed=1)
    indptr, indices = graph_ops.to_csr(src, dst, n)
    centers = np.arange(n)
    a = graph_ops.ego_subgraphs(indptr, indices, centers, hops, max_nodes, backend="rust")
    b = graph_ops.ego_subgraphs(indptr, indices, centers, hops, max_nodes, backend="python")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.skipif(not graph_ops.HAS_RUST, reason="baed_graph extension not built")
def test_rust_rejects_bad_input():
    with pytest.raises(ValueError):
        graph_ops.ego_subgraphs(np.array([0, 1]), np.array([5]), np.array([0]), 1, backend="rust")


def test_center_labels_used_for_sublabel():
    src, dst, n = np.array([0, 1]), np.array([1, 2]), 3
    onehot = torch.tensor([[0, 1], [1, 0], [1, 0]])
    gs = graph_ops.build_ego_graphs(
        src, dst, n, torch.zeros(3, 2), onehot, [0], 1, backend="python", center_labels=onehot.argmax(1)
    )
    assert gs[0].graph["sublabel"] == 1
    assert torch.equal(gs[0].nodes[0]["label"], torch.tensor([0, 1]))
