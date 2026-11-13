"""Step 01 - build the ego-graph datasets.

For every labelled center node (label 1 only, or labels 0 and 1) extract its k-hop
neighbourhood and save the list of ``nx.Graph`` objects (each with ``graph['sublabel']``)
to ``data/graphs/<dataset>_init_<1|0_1>_dataset.pkl``.

Hops per dataset (as in the original code): dgraph/elliptic 2, tfinance/photo/reddit/Amazon 1.
tfinance additionally drops ego-graphs with more than 1200 nodes.

Compared with ``createSubGraphDataset/createDataset.py`` the output is the same graphs,
but the extraction runs in Rust on a CSR array (``baed.graph_ops``) instead of networkx,
and Elliptic uses a sparse loader instead of a 330 GB dense matrix.
"""

from __future__ import annotations

import argparse
import pickle as pkl
import time
import warnings

import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from dgl.data.utils import load_graphs
from loguru import logger

from baed import paths
from baed.datasets.elliptic import iter_timesteps
from baed.graph_ops import build_ego_graphs

warnings.filterwarnings("ignore")

DATASETS = ["dgraph", "elliptic", "tfinance", "photo", "reddit", "Amazon"]
LABEL_TYPES = ["label_1", "label_0_1"]


def preprocess_features(features):
    """Row-normalize a feature matrix and return it dense."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def adj_to_dgl_graph(adj):
    """SciPy sparse adjacency -> DGL graph (via networkx, as in the original code)."""
    nx_graph = nx.from_scipy_sparse_array(adj)
    return dgl.from_networkx(nx_graph)


# ego-graph radius of the released code for the datasets loaded as one static graph
FULL_GRAPH_HOPS = {"dgraph": 2, "tfinance": 1, "photo": 1, "reddit": 1, "Amazon": 1}


def load_full_graph(dataset: str):
    """The whole graph of a non-temporal dataset: DGL graph with ndata 'feature' (float) and 'label' (as
    stored), plus integer node labels (T-Finance stores one-hot labels). Self-loops are removed (the .mat
    adjacencies carry one on every node; Elliptic has none, and with a self-loop both nodes of a 2-node
    ego-graph get the same embedding, so the deviation readout gives 0). Shared by the ego-graph builder and
    the GAE pre-training (baed-gae-dataset --whole), so the guidance encoder sees the ego-graph features."""
    graph, labels = _load_full_graph(dataset)
    n_loops = int((graph.edges()[0] == graph.edges()[1]).sum())
    if n_loops:
        graph = dgl.remove_self_loop(graph)
        logger.info(f"{dataset}: removed {n_loops} self-loops")
    return graph, labels


def _load_full_graph(dataset: str):
    if dataset == "dgraph":
        item = np.load(paths.graph_file("dgraphfin.npz"))
        edge_index = item["edge_index"]
        graph = dgl.graph(
            (torch.tensor(edge_index[:, 0], dtype=torch.int64), torch.tensor(edge_index[:, 1], dtype=torch.int64)),
            num_nodes=len(item["y"]),
        )
        graph.ndata["feature"] = torch.tensor(item["x"], dtype=torch.float32)
        graph.ndata["label"] = torch.tensor(item["y"], dtype=torch.int64)
        return graph, graph.ndata["label"]
    if dataset == "tfinance":
        graph = load_graphs(str(paths.graph_file("tfinance")))[0][0]
        graph.ndata["feature"] = graph.ndata["feature"].to(torch.float32)
        return graph, graph.ndata["label"].argmax(1)  # one-hot -> int (node attribute stays one-hot)
    if dataset in ["photo", "reddit", "Amazon"]:
        data = sio.loadmat(str(paths.graph_file(f"{dataset}.mat")))
        label = data.get("Label", data.get("gnd"))
        attr = data.get("Attributes", data.get("X"))
        network = data.get("Network", data.get("A"))
        adj = sp.csr_matrix(network)
        features = preprocess_features(sp.lil_matrix(attr))
        graph = adj_to_dgl_graph(adj)
        graph.ndata["feature"] = torch.tensor(features, dtype=torch.float32)
        graph.ndata["label"] = torch.tensor(np.squeeze(np.array(label)), dtype=torch.int64)
        return graph, graph.ndata["label"]
    raise ValueError(f"unsupported dataset: {dataset}")


def labelled_subgraph(graph, labels):
    """Drop the nodes whose label is not 0 or 1 (DGraph background nodes) and every edge touching them,
    the same rule as the labelled-only Elliptic ego-graphs. Node order is kept, so the centers keep
    their relative order (and the sequential split its meaning)."""
    keep = torch.nonzero((labels == 0) | (labels == 1), as_tuple=True)[0]
    sub = dgl.node_subgraph(graph, keep)
    logger.info(f"labelled subgraph: {sub.num_nodes()} of {graph.num_nodes()} nodes, {sub.num_edges()} of "
                f"{graph.num_edges()} edges")
    return sub, labels[keep]


def select_centers(labels: torch.Tensor, label_type: str) -> list[int]:
    if label_type == "label_1":
        mask = labels == 1
    elif label_type == "label_0_1":
        mask = (labels == 0) | (labels == 1)
    else:
        raise ValueError("label_type must be 'label_1' or 'label_0_1'")
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def output_file(dataset: str, label_type: str, tag: str = ""):
    suffix = "1" if label_type == "label_1" else "0_1"
    return paths.ensure(paths.graph_file(f"{dataset}{tag}_init_{suffix}_dataset.pkl"))


def ego_from_dgl(graph, labels, label_type, hops, max_nodes=None, backend="auto", bidirectional=False):
    """``labels``: integer label per node, used to pick centers and for ``sublabel``.
    Node attributes are taken from ``graph.ndata['feature']`` / ``graph.ndata['label']``."""
    src, dst = graph.edges()
    centers = select_centers(labels, label_type)
    return build_ego_graphs(
        src.numpy(),
        dst.numpy(),
        graph.num_nodes(),
        graph.ndata["feature"],
        graph.ndata["label"],
        centers,
        hops=hops,
        max_nodes=max_nodes,
        backend=backend,
        center_labels=labels,
        bidirectional=bidirectional,
    )


def build(dataset: str, label_type: str, backend: str = "auto", elliptic_full: bool = False,
          bidirectional: bool = False, drop_unlabelled: bool = False) -> list[nx.Graph]:
    logger.info(f"dataset={dataset} label_type={label_type}")

    if dataset in FULL_GRAPH_HOPS:
        graph, labels = load_full_graph(dataset)
        if drop_unlabelled:
            graph, labels = labelled_subgraph(graph, labels)
        return ego_from_dgl(graph, labels, label_type, hops=FULL_GRAPH_HOPS[dataset],
                            max_nodes=1200 if dataset == "tfinance" else None, backend=backend,
                            bidirectional=bidirectional)

    if dataset == "elliptic":
        nx_graphs = []
        # elliptic_full: use all 49 timesteps (the original loop range(1, 49) with ts + 1 skips
        # timestep 1) and give the DGL graph all labelled nodes, so no timestep is dropped.
        start = 0 if elliptic_full else 1
        for step in iter_timesteps(str(paths.graph_file("elliptic_bitcoin_dataset")), start, 49, node_order="set"):
            if len(step.tx_ids) == 0:
                continue
            num_nodes = len(step.tx_ids) if elliptic_full else None
            graph = dgl.graph((torch.from_numpy(step.src), torch.from_numpy(step.dst)), num_nodes=num_nodes)
            node_features = torch.tensor(step.features.values, dtype=torch.float32)
            node_labels = torch.tensor(
                step.classes["class"].apply(lambda x: 1 if x == "1" else 0).values, dtype=torch.int64
            )
            # Original behaviour: a timestep is used only if the DGL graph built from the
            # edge list covers every labelled node (i.e. the highest-index node has an edge).
            if len(node_features) != graph.num_nodes():
                logger.debug(
                    f"elliptic ts={step.ts}: skipped ({graph.num_nodes()} graph nodes vs {len(node_features)})"
                )
                continue
            graph.ndata["feature"] = node_features
            graph.ndata["label"] = node_labels
            nx_graphs += ego_from_dgl(graph, node_labels, label_type, hops=2, backend=backend, bidirectional=bidirectional)
        return nx_graphs

    raise ValueError(f"unsupported dataset: {dataset}")


def initOriginDataset(dataset: str, label_type: str, backend: str = "auto", elliptic_full: bool = False,
                      bidirectional: bool = False, drop_unlabelled: bool = False, node_time: bool = False) -> None:
    """Build and save one ego-graph dataset (name kept from the original code)."""
    t0 = time.time()
    nx_graphs = build(dataset, label_type, backend=backend, elliptic_full=elliptic_full, bidirectional=bidirectional,
                      drop_unlabelled=drop_unlabelled)
    if node_time:
        # order the ego-graphs by the appearance time of their center (stable), so the sequential 70/15/15 split
        # of the detector is a temporal split, as on Elliptic
        t = np.load(paths.graph_file(f"{dataset}_node_time.npy"))
        nx_graphs = [g for _, g in sorted(((t[next(iter(g.nodes))], g) for g in nx_graphs), key=lambda p: p[0])]
        logger.info(f"ego-graphs ordered by center time ({dataset}_node_time.npy)")
    tag = ("_full" if elliptic_full else "") + ("_lab" if drop_unlabelled else "") + ("_t" if node_time else "") + ("_bi" if bidirectional else "")
    out = output_file(dataset, label_type, tag=tag)
    with open(out, "wb") as f:
        pkl.dump(nx_graphs, f)
    logger.info(f"saved {len(nx_graphs)} ego-graphs to {out} ({time.time() - t0:.1f}s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 01 - build k-hop ego-graph datasets.")
    parser.add_argument("--dataset", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--label-type", nargs="+", default=LABEL_TYPES, choices=LABEL_TYPES)
    parser.add_argument("--backend", default="auto", choices=["auto", "rust", "python"])
    parser.add_argument("--elliptic-full", action="store_true",
                        help="Elliptic: keep all 49 timesteps (fixes two quirks of the original code); "
                             "writes elliptic_full_init_*_dataset.pkl")
    parser.add_argument("--drop-unlabelled", action="store_true",
                        help="non-temporal datasets: remove the nodes labelled neither 0 nor 1 (DGraph background) "
                             "before extracting ego-graphs, as for Elliptic; adds _lab to the file name")
    parser.add_argument("--node-time", action="store_true",
                        help="order the ego-graphs by data/graphs/<dataset>_node_time.npy (scripts/make_node_time.py), "
                             "so the sequential split is temporal; adds _t to the file name")
    parser.add_argument("--bidirectional", action="store_true",
                        help="ego-graphs over in- and out-edges (original: out-edges only); adds _bi to the file name")
    args = parser.parse_args(argv)
    for dataset in args.dataset:
        for label_type in args.label_type:
            initOriginDataset(dataset, label_type, backend=args.backend,
                              elliptic_full=args.elliptic_full and dataset == "elliptic",
                              bidirectional=args.bidirectional,
                              drop_unlabelled=args.drop_unlabelled and dataset != "elliptic",
                              node_time=args.node_time and dataset != "elliptic")


if __name__ == "__main__":
    main()
