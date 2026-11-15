"""build_ego_lean(return_centers=True) must return the center of every kept item, also when max_nodes drops some."""
import numpy as np
import torch

from baed.graph_ops import build_ego_lean


def test_kept_centers_match_items():
    rng = np.random.default_rng(0)
    n = 60
    src = rng.integers(0, n, 400)
    dst = rng.integers(0, n, 400)
    hub = np.arange(1, 40)  # node 0 and node 50 get large ego-graphs, dropped by max_nodes
    src = np.concatenate([src, np.zeros(39, dtype=int), np.full(39, 50)])
    dst = np.concatenate([dst, hub, hub])
    feats = torch.arange(n, dtype=torch.float32).unsqueeze(1)  # feature = node id
    centers = list(range(n))
    labels = torch.zeros(n, dtype=torch.long)
    items, kept = build_ego_lean(src, dst, n, feats, centers, 1, labels, max_nodes=30, backend="python",
                                 return_centers=True)
    assert len(items) == len(kept) < n
    assert 0 not in kept and 50 not in kept
    for (f, _, _), c in zip(items, kept):
        assert int(f[0, 0]) == int(c)  # node 0 of every item is its center
