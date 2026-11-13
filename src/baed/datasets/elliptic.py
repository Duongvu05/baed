"""Sparse loader for the Elliptic Bitcoin dataset.

The released code built, for every one of the 48 timesteps, a dense pandas DataFrame of
shape (203769, 203769) (~330 GB of float64) and filled it edge by edge with ``.loc``.
That cannot run on any real machine. This loader produces the *same* per-timestep graphs
(same node indexing, same edge set, same row-major edge order as ``np.nonzero`` on the
dense matrix) from the edge list directly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EllipticTimestep:
    ts: int
    tx_ids: list  # labelled transactions of this timestep, position = local node id
    src: np.ndarray  # local ids, row-major sorted, deduplicated (== np.nonzero(dense_adj))
    dst: np.ndarray
    features: pd.DataFrame  # rows aligned with tx_ids
    classes: pd.DataFrame  # rows aligned with tx_ids, column 'class' in {'1', '2'}


def read_elliptic(data_dir: str):
    classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"), index_col="txId")
    edgelist = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
    features = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None, index_col=0)
    classes["class"] = classes["class"].astype(str)
    return classes, edgelist, features


def iter_timesteps(data_dir: str, start_ts: int, end_ts: int, node_order: str = "set",
                   labelled_only: bool = True) -> Iterator[EllipticTimestep]:
    """Yield the labelled sub-graph of each timestep in ``[start_ts, end_ts)``.

    ``node_order`` reproduces the two variants of the original code:
      * ``"set"``  - ``createSubGraphDataset`` iterated ``set(features_ts.index)``
      * ``"file"`` - ``gae/saveGAEdataset`` iterated ``list(features_ts.index)``

    ``labelled_only=False`` keeps the unlabelled ("unknown") transactions and every edge touching them,
    as in the paper's ego-graph definition (all nodes reachable within K hops). The labelled transactions
    come first, in the same order as with ``labelled_only=True`` (same centers in the same order, so the
    sequential split is unchanged), followed by the unlabelled ones in file order. ``classes`` then
    contains 'unknown' rows.
    """
    classes, edgelist, features = read_elliptic(data_dir)
    labelled_tx = set(classes.index[classes["class"] != "unknown"])
    for ts in range(start_ts, end_ts):
        features_ts = features[features[1] == ts + 1]
        tx_ts = set(features_ts.index) if node_order == "set" else list(features_ts.index)
        tx_ids = [tx for tx in tx_ts if tx in labelled_tx]
        if not labelled_only:
            tx_ids += [tx for tx in features_ts.index if tx not in labelled_tx]
        pos = pd.Series(np.arange(len(tx_ids), dtype=np.int64), index=pd.Index(tx_ids))
        e = edgelist[edgelist["txId1"].isin(pos.index) & edgelist["txId2"].isin(pos.index)]
        pairs = np.stack([pos.loc[e["txId1"]].to_numpy(), pos.loc[e["txId2"]].to_numpy()], axis=1)
        pairs = np.unique(pairs.reshape(-1, 2), axis=0)
        yield EllipticTimestep(
            ts=ts,
            tx_ids=tx_ids,
            src=pairs[:, 0].astype(np.int64),
            dst=pairs[:, 1].astype(np.int64),
            features=features.loc[tx_ids],
            classes=classes.loc[tx_ids],
        )
