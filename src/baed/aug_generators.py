"""Negative-sample generators used as baselines next to BAED (paper Table 1/2: AEGIS, GGAD).

Both original methods are node-level and generate pseudo anomalies in an embedding space
inside their own detector. The BAED paper does not say how they were plugged into the
ego-graph pipeline, so this module uses one shared, explicit adaptation:

* a synthetic sample is a *template* ego-graph drawn from the normal training ego-graphs
  (at least two nodes, so it survives the single-node filter of the released setting);
* one node ``v`` of the template gets synthetic features, the other nodes and the edges
  are kept, and the graph is labelled anomalous;
* the detector consumes these graphs exactly like BAED's (``generate_arrays``), i.e. each
  training batch is topped up with synthetic anomalies until it is balanced.

GGAD (Qiao et al., NeurIPS 2024): outliers are built from a normal node's neighbourhood
("egocentric closeness") plus a small Gaussian perturbation. Here, in feature space:
``x_v <- mean(x_u for u in N(v)) + N(mean, std)``. The official code uses mean = std = 0 on
the large datasets, which is the default. GGAD's learned refinement (affinity / closeness
losses on the embedding) is not reproduced.

AEGIS (Ding et al., IJCAI 2021): a generator maps Gaussian noise to pseudo anomalies and
is trained adversarially against a discriminator on normal nodes. Here a small MLP GAN
is fit on the (standardised) features of the nodes of normal training ego-graphs and
``x_v <- G(z)``. AEGIS's GDN encoder is not reproduced; the GAN works on raw features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _templates(train_items):
    """Normal training ego-graphs with >= 2 nodes and at least one edge: (feat, edge_index)."""
    out = [(f.float(), e) for f, e, y in train_items if int(y) == 0 and f.shape[0] >= 2 and e.numel() > 0]
    if not out:
        raise ValueError("no normal multi-node ego-graph in the training set")
    return out


class _TemplateGenerator:
    def __init__(self, train_items, seed=0):
        self.templates = _templates(train_items)
        self.rng = np.random.default_rng(seed)

    def node_features(self, feat, edge_index, v):  # pragma: no cover - abstract
        raise NotImplementedError

    def generate_arrays(self, num_samples, feature_num=None, embedding=None):
        """Same contract as ``GraphDataGenerator.generate_arrays``: lists of features and edge indices."""
        feats, edges = [], []
        for k in self.rng.integers(0, len(self.templates), num_samples):
            f, e = self.templates[k]
            v = int(self.rng.integers(0, f.shape[0]))
            f = f.clone()
            f[v] = self.node_features(f, e, v)
            feats.append(f)
            edges.append(e)
        return feats, edges


class GGADGenerator(_TemplateGenerator):
    def __init__(self, train_items, seed=0, noise_mean=0.0, noise_std=0.0):
        super().__init__(train_items, seed)
        self.noise_mean, self.noise_std = noise_mean, noise_std
        self.g = torch.Generator().manual_seed(seed)

    def node_features(self, feat, edge_index, v):
        nb = edge_index[1][(edge_index[0] == v) & (edge_index[1] != v)]
        x = feat[nb].mean(0) if nb.numel() else feat[v]
        if self.noise_std > 0 or self.noise_mean != 0:
            x = x + torch.randn(x.shape, generator=self.g) * self.noise_std + self.noise_mean
        return x


class AEGISGenerator(_TemplateGenerator):
    def __init__(self, train_items, seed=0, z_dim=64, hidden=128, steps=2000, batch=512, lr=1e-3, device="cpu"):
        super().__init__(train_items, seed)
        torch.manual_seed(seed)
        x = torch.cat([f for f, _ in self.templates])
        self.mu, self.sd = x.mean(0), x.std(0).clamp_min(1e-6)
        xs = ((x - self.mu) / self.sd).to(device)
        d = x.shape[1]
        self.z_dim, self.device = z_dim, device
        self.G = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, d)).to(device)
        D = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1)).to(device)
        og, od = torch.optim.Adam(self.G.parameters(), lr), torch.optim.Adam(D.parameters(), lr)
        bce = nn.BCEWithLogitsLoss()
        ones, zeros = torch.ones(batch, 1, device=device), torch.zeros(batch, 1, device=device)
        for _ in range(steps):
            real = xs[torch.randint(0, len(xs), (batch,), device=device)]
            fake = self.G(torch.randn(batch, z_dim, device=device))
            od.zero_grad()
            (bce(D(real), ones) + bce(D(fake.detach()), zeros)).backward()
            od.step()
            og.zero_grad()
            bce(D(fake), ones).backward()
            og.step()
        self.G.eval()
        self._buf = torch.empty(0, d)

    @torch.no_grad()
    def node_features(self, feat, edge_index, v):
        if len(self._buf) == 0:  # sample in chunks: one G call per 4096 synthetic nodes
            z = torch.randn(4096, self.z_dim, device=self.device)
            self._buf = self.G(z).cpu() * self.sd + self.mu
        x, self._buf = self._buf[0], self._buf[1:]
        return x


def make_generator(name, train_items, seed=0, device="cpu"):
    if name == "ggad":
        return GGADGenerator(train_items, seed)
    if name == "aegis":
        return AEGISGenerator(train_items, seed, device=device)
    raise ValueError(name)
