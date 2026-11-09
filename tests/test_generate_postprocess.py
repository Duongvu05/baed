"""Rust post-processing of generated graphs == original pyg -> networkx -> pyg path."""

import networkx as nx
import pytest
import torch
import torch_geometric as pyg
from torch_geometric.data import Batch, Data

from baed import graph_ops
from baed.pipelines.generate import postprocess_batch
from baed.utils import processGenerateSubgraph

F = 4


def original(batch):
    """Copy of GraphDataGenerator.generate_data (after sampling) + processGenerateSubgraph."""
    out = []
    for d in batch.to_data_list():
        keys = [k for k in d.keys() if k.startswith("feature")] + ["label"]  # noqa: SIM118 (Data iterates pairs)
        g = pyg.utils.to_networkx(d, to_undirected=True)
        for i, node in enumerate(g.nodes()):
            for k in keys:
                g.nodes[node][k] = d[k][i]
        g = g.subgraph(max(nx.connected_components(g), key=len))
        out.append(g)
    return processGenerateSubgraph(out, F)


def random_batch(seed, b=12):
    gen = torch.Generator().manual_seed(seed)
    ds = []
    for _ in range(b):
        n = int(torch.randint(1, 9, (1,), generator=gen))
        m = int(torch.randint(0, 2 * n, (1,), generator=gen))
        ei = torch.randint(0, n, (2, m), generator=gen)
        ei = pyg.utils.to_undirected(ei, num_nodes=n)
        d = Data(edge_index=ei, num_nodes=n, label=torch.ones(n, dtype=torch.long))
        for i in range(F):
            d[f"feature{i}"] = torch.randn(n, generator=gen)
        ds.append(d)
    return Batch.from_data_list(ds)


def canon(feat, ei):
    """Order-independent view: sorted feature rows and edge set in feature space."""
    rows = sorted(map(tuple, feat.tolist()))
    edges = sorted((tuple(feat[u].tolist()), tuple(feat[v].tolist())) for u, v in ei.t().tolist())
    return rows, edges


@pytest.mark.skipif(not graph_ops.HAS_RUST, reason="baed_graph extension not built")
@pytest.mark.parametrize("seed", range(5))
def test_fast_equals_original(seed):
    batch = random_batch(seed)
    of, oe = original(batch)
    ff, fe = postprocess_batch(batch, F)
    assert len(of) == len(ff)
    for a, b, c, d in zip(of, oe, ff, fe):
        assert canon(a, b) == canon(c, d)
