"""dgl.batch + per-graph max readout == running each ego-graph alone (all four detectors)."""

import dgl
import pytest
import torch

from baed.models.gnn import BWGNN, GCN, BernNet, GraphSAGE


def graphs(n=6, f=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(n):
        m = 2 + k % 4
        src = torch.arange(m - 1)
        dst = src + 1
        gr = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=m)
        gr.ndata["feature"] = torch.randn(m, f, generator=g)
        out.append(gr)
    return out


@pytest.mark.parametrize("cls", [GCN, BWGNN, GraphSAGE, BernNet])
def test_batched_equals_single(cls):
    torch.manual_seed(0)
    kw = {"activation": "Sigmoid"} if cls in (BWGNN, BernNet) else {}
    model = cls(in_feats=5, **kw).eval()
    gs = graphs()
    with torch.no_grad():
        single = torch.stack([model(g) for g in gs])
        batched = model(dgl.batch(gs))
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)
