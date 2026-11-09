"""Sparse Elliptic loader == the original dense-DataFrame construction (tiny synthetic data)."""

import os

import numpy as np
import pandas as pd
import pytest

from baed.datasets.elliptic import iter_timesteps


def write_fake(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    tx = rng.choice(np.arange(1000, 5000), n, replace=False)
    ts = rng.integers(1, 4, n)  # timesteps 1..3
    cls = rng.choice(["1", "2", "unknown"], n)
    feats = np.c_[tx, ts, rng.normal(size=(n, 3))]
    pd.DataFrame({"txId": tx, "class": cls}).to_csv(tmp_path / "elliptic_txs_classes.csv", index=False)
    e = rng.integers(0, n, (150, 2))
    pd.DataFrame({"txId1": tx[e[:, 0]], "txId2": tx[e[:, 1]]}).to_csv(
        tmp_path / "elliptic_txs_edgelist.csv", index=False
    )
    pd.DataFrame(feats).to_csv(tmp_path / "elliptic_txs_features.csv", index=False, header=False)


def dense_original(data_dir, start_ts, end_ts, node_order):
    """Copy of the released code (createSubGraphDataset / saveGAEdataset)."""
    classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"), index_col="txId")
    edgelist = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"), index_col="txId1")
    features = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None, index_col=0)
    labelled_tx = set(classes[classes["class"] != "unknown"].index)
    total_tx = list(classes.index)
    out = []
    for ts in range(start_ts, end_ts):
        features_ts = features[features[1] == ts + 1]
        tx_ts = set(features_ts.index) if node_order == "set" else list(features_ts.index)
        labelled_tx_ts = [tx for tx in tx_ts if tx in labelled_tx]
        adj_mat = pd.DataFrame(np.zeros((len(total_tx), len(total_tx))), index=total_tx, columns=total_tx)
        el = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
        for i in range(el.shape[0]):
            src, dst = el.index[i], el.iloc[i]["txId2"]
            if dst in adj_mat.columns:
                adj_mat.loc[src, dst] = 1
        adj = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
        out.append((labelled_tx_ts, np.nonzero(adj.values), features.loc[labelled_tx_ts]))
    return out


@pytest.mark.parametrize("node_order", ["set", "file"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_equals_dense(tmp_path, node_order, seed):
    write_fake(tmp_path, seed)
    ref = dense_original(str(tmp_path), 0, 3, node_order)
    got = list(iter_timesteps(str(tmp_path), 0, 3, node_order=node_order))
    assert len(ref) == len(got)
    for (tx, (s, d), f), step in zip(ref, got):
        assert list(tx) == list(step.tx_ids)
        np.testing.assert_array_equal(s, step.src)
        np.testing.assert_array_equal(d, step.dst)
        pd.testing.assert_frame_equal(f, step.features)
