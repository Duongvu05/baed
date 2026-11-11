"""Degree-guided diffusion with anomaly guidance (arch TGNN_degree_embedding_guided)."""

from argparse import Namespace

import networkx as nx
import torch

from baed.datasets.data_utils import EmpiricalEmptyGraphGenerator, collate_fn, preprocess
from baed.model import get_model

F, EMB = 4, 256


class FakeGAE(torch.nn.Module):
    """Stands in for the pre-trained GAE: node features -> 256-d node embeddings."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(F, EMB)

    def encode(self, x, edge_index):
        return self.lin(x)


def ego_graphs(seed=0, n_graphs=24):
    rng = torch.Generator().manual_seed(seed)
    out = []
    for k in range(n_graphs):
        n = [1, 2, 3, 4, 5][k % 5]
        g = nx.path_graph(n) if k % 2 else nx.star_graph(n - 1) if n > 1 else nx.empty_graph(1)
        for v in g.nodes:
            g.nodes[v]["feature"] = torch.randn(F, generator=rng)
            g.nodes[v]["label"] = torch.tensor(1)
        for e, (u, v) in enumerate(g.edges):
            g.edges[u, v]["id"] = e
        out.append(g)
    return out


def build(gae, graphs):
    pyg = [preprocess(g, gae, device="cpu", degree=True, augmented_features=[]) for g in graphs]
    sampler = EmpiricalEmptyGraphGenerator(pyg, gae, device="cpu", degree=True,
                                           augment_features=["label", "feature"] + [f"feature{i}" for i in range(F)])
    args = Namespace(
        dataset="elliptic", arch="TGNN_degree_embedding_guided", parametrization="xt_prescribed_st",
        loss_type="vb_ce_xt_prescribred_st", degree=True, max_degree=4, num_node_classes=2, num_edge_classes=2,
        diffusion_dim=16, diffusion_steps=8, num_heads=[2, 1], dp_rate=0.0, norm="None",
        augmented_feature_dict={}, has_node_feature=False, final_prob_node=[1.0, 0.0],
        final_prob_edge=[1.0, 0.0], sample_time_method="uniform", noise_schedule="linear", device="cpu",
    )
    return pyg, get_model(args, initial_graph_sampler=sampler)


def test_preprocess_degree_covers_isolated_nodes():
    gae = FakeGAE()
    single = preprocess(ego_graphs()[0], gae, device="cpu", degree=True)  # one node, no edge
    assert single.degree.tolist() == [0]
    batch = collate_fn([preprocess(g, gae, device="cpu", degree=True) for g in ego_graphs()[:6]])
    assert batch.degree.shape[0] == batch.num_nodes


def test_train_loss_and_guided_sampling():
    torch.manual_seed(0)
    gae = FakeGAE()
    data, model = build(gae, ego_graphs())
    batch = collate_fn(data[:8])
    model.train()
    loss = -model.log_prob(batch).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert model._denoise_fn.guide[0].weight.grad.abs().sum() > 0  # guidance reaches the loss

    model.eval()
    with torch.no_grad():
        out = model.sample(16, torch.randn(EMB))
    assert out.num_graphs == 16
    # every generated edge stays inside its graph, upper-triangular full_edge_index
    assert (out.batch[out.edge_index[0]] == out.batch[out.edge_index[1]]).all()


def test_guidance_changes_edge_logits():
    torch.manual_seed(0)
    gae = FakeGAE()
    data, model = build(gae, ego_graphs())
    model.eval()
    b = collate_fn(data[:8])
    t = torch.zeros(b.num_graphs, dtype=torch.long)
    t_node = t.repeat_interleave(b.nodes_per_graph)
    t_edge = t.repeat_interleave(b.edges_per_graph)
    model._q_sample_and_set_xtmin1_xt_given_x0(b, t_node, t_edge)
    b.active_node_indices = torch.arange(b.num_nodes)
    b.active_edge_indices = torch.arange(b.full_edge_index.size(1))
    net = model._denoise_fn
    with torch.no_grad():
        _, a = net(b, t_node, t_edge, torch.zeros(EMB))
        _, c = net(b, t_node, t_edge, torch.ones(EMB) * 5)
        _, own = net(b, t_node, t_edge)  # training path: each graph's own pooled embedding
    assert a.shape == c.shape == own.shape == (b.full_edge_index.size(1), 2)
    assert not torch.allclose(a, c)
