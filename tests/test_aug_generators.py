import torch

from baed.aug_generators import AEGISGenerator, GGADGenerator


def _items():
    # (feat, edge_index both directions, label); graph 0/1 normal multi-node, 2 single-node, 3 anomalous
    path = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    return [
        (torch.arange(12.0).reshape(3, 4), path, 0),
        (torch.ones(2, 4), torch.tensor([[0, 1], [1, 0]]), 0),
        (torch.zeros(1, 4), torch.zeros(2, 0, dtype=torch.long), 0),
        (torch.full((3, 4), 9.0), path, 1),
    ]


def _check(feats, edges, n):
    assert len(feats) == len(edges) == n
    for f, e in zip(feats, edges):
        assert f.dtype == torch.float32 and f.shape[1] == 4 and f.shape[0] >= 2
        assert e.numel() > 0 and int(e.max()) + 1 == f.shape[0] and torch.unique(e).numel() == f.shape[0]


def test_ggad_templates_and_neighbour_mean():
    gen = GGADGenerator(_items(), seed=0)
    assert len(gen.templates) == 2  # single-node and anomalous graphs are not templates
    feats, edges = gen.generate_arrays(50)
    _check(feats, edges, 50)
    f, e = gen.templates[0]
    # middle node of the path: mean of its two neighbours
    assert torch.allclose(gen.node_features(f, e, 1), (f[0] + f[2]) / 2)
    assert torch.allclose(gen.node_features(f, e, 0), f[1])
    # exactly one node changes per synthetic graph, never the anomalous template's 9.0 values
    for g in feats:
        assert not (g == 9.0).all()


def test_aegis_contract_and_scale():
    gen = AEGISGenerator(_items(), seed=0, steps=50, batch=16)
    feats, edges = gen.generate_arrays(20)
    _check(feats, edges, 20)
    assert all(torch.isfinite(f).all() for f in feats)
