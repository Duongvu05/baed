"""Lean ego-graph tensors for the batched detector path.

``utils.preprocessDataset`` converts every ego-graph with ``from_networkx`` and then
adds 166 per-feature attributes, an O(n^2) ``full_edge_index`` and a watchdog thread
per graph. The detector only reads ``feature``, ``edge_index`` and ``sublabel``, so
here we extract exactly those (same node order as ``from_networkx``: ``G.nodes()``)
and cache them on disk.

Filtering is unchanged by default: ``preprocess`` returns ``None`` for single-node
ego-graphs (empty ``full_edge_attr``), so they are dropped unless ``keep_isolated``.
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import torch
from loguru import logger


def convert(g: nx.Graph) -> tuple[torch.Tensor, torch.Tensor, int]:
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    feat = torch.stack([torch.as_tensor(g.nodes[n]["feature"]) for n in nodes])
    src, dst = [], []
    for u, v in g.edges():
        a, b = idx[u], idx[v]
        src.append(a)
        dst.append(b)
        if a != b:  # from_networkx: undirected edge -> both directions, self-loop once
            src.append(b)
            dst.append(a)
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return feat, ei, int(g.graph["sublabel"])


def load_lean(pkl_path: str | os.PathLike, load_nx, cache_dir: str | os.PathLike):
    """Convert all graphs (cached by pkl path + mtime). Returns a list of (feat, edge_index, label).

    ``load_nx`` is called only on a cache miss (the networkx pickle is large and slow to load).
    """
    pkl_path = Path(pkl_path)
    direct = Path(cache_dir) / f"{pkl_path.stem}.lean.pt"  # written by baed-build-subgraphs --lean-only (no pickle)
    if not pkl_path.exists() and direct.exists():
        logger.info(f"lean file (built without networkx): {direct}")
        return torch.load(direct)
    cache = Path(cache_dir) / f"{pkl_path.stem}.lean.{int(pkl_path.stat().st_mtime)}.pt"
    if cache.exists():
        logger.info(f"lean cache hit: {cache}")
        return torch.load(cache)
    items = [convert(g) for g in load_nx()]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, cache)
    logger.info(f"lean cache written: {cache} ({len(items)} graphs)")
    return items


def keep(item, keep_isolated: bool) -> bool:
    """Same rule as utils.preprocess (drop single-node graphs) unless keep_isolated."""
    return keep_isolated or item[0].shape[0] > 1


class LeanBatch:
    __slots__ = ("subFeature", "sub_edge_index", "sub_label")

    def __init__(self, items):
        self.subFeature = [it[0] for it in items]
        self.sub_edge_index = [it[1] for it in items]
        self.sub_label = torch.tensor([it[2] for it in items], dtype=torch.long)


def lean_collate(items):
    return LeanBatch(items) if items else None
