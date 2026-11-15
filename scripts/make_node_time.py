"""Per-node appearance time for datasets whose GAD release dropped the timestamps.

    uv run python scripts/make_node_time.py reddit <path to JODIE reddit.csv>
    uv run python scripts/make_node_time.py dgraph
    uv run python scripts/make_node_time.py tfinance

Reddit: data/graphs/reddit.mat is the JODIE Reddit network (snap.stanford.edu/jodie/reddit.csv): node u is
user u (0..9,999) and node 10,000 + i is subreddit i (0..983); its 78,516 edges and 366 anomalies (banned
users) are exactly JODIE's unique user-subreddit pairs and banned users. A node's time is its first interaction,
the analogue of the Elliptic timestep. Writes data/graphs/reddit_node_time.npy (float, seconds).

DGraph-Fin: dgraphfin.npz keeps edge_timestamp (days 1..821) but has no node time, and its node ids do not follow
time (Spearman 0.15 with the first edge day on labelled nodes). A node's time is the day of its first edge (in or
out). Every node labelled 0/1 has at least one edge. Writes data/graphs/dgraph_node_time.npy (float, days).

T-Finance: no timestamp is released. BWGNN (Tang et al. 2022) describes the 10 features as "related to registration
days, logging activities and interaction frequency" without naming the columns. Column 0 is an integer 11..5000 that
behaves as the account age in days at the snapshot: the last node ids are the newest accounts (11-35 days), node id
follows it (Spearman -0.61) and the fraud rate rises from 0.4% (oldest tenth) to about 10% (newest tenth). A node's
time is its registration day relative to the snapshot, -column 0 (oldest first; ties keep node-id order).
Writes data/graphs/tfinance_node_time.npy (float, days).
"""
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

from baed import paths


def dgraph():
    d = np.load(paths.graph_file("dgraphfin.npz"))
    e, ts = d["edge_index"], d["edge_timestamp"].astype(float)
    t = np.full(len(d["y"]), np.inf)
    np.minimum.at(t, e[:, 0], ts)
    np.minimum.at(t, e[:, 1], ts)
    lab = (d["y"] == 0) | (d["y"] == 1)
    assert np.isfinite(t[lab]).all(), "a labelled node has no edge"
    out = paths.graph_file("dgraph_node_time.npy")
    np.save(out, t)
    print(f"{out}: {len(t)} nodes, labelled first edge day {t[lab].min():.0f}..{t[lab].max():.0f}, "
          f"{int((~np.isfinite(t)).sum())} nodes without edges (inf)")


def tfinance():
    from dgl.data.utils import load_graphs

    x = load_graphs(str(paths.graph_file("tfinance")))[0][0].ndata["feature"].numpy()
    age = x[:, 0]
    assert np.allclose(age, np.round(age)) and age.min() > 0, "column 0 is not an integer number of days"
    t = -age.astype(float)
    out = paths.graph_file("tfinance_node_time.npy")
    np.save(out, t)
    print(f"{out}: {len(t)} nodes, registration day {t.min():.0f}..{t.max():.0f} (days before the snapshot)")


def main():
    if sys.argv[1] == "dgraph":
        return dgraph()
    if sys.argv[1] == "tfinance":
        return tfinance()
    dataset, csv = sys.argv[1], sys.argv[2]
    assert dataset == "reddit", "only reddit, dgraph and tfinance are implemented"
    df = pd.read_csv(csv, usecols=[0, 1, 2], header=None, skiprows=1, names=["user", "item", "ts"])
    n_users = int(df.user.max()) + 1
    n = int(sio.loadmat(str(paths.graph_file("reddit.mat")))["Network"].shape[0])
    assert n_users + int(df.item.max()) + 1 == n, "node count does not match reddit.mat"
    t = np.full(n, np.inf)
    first_u = df.groupby("user").ts.min()
    first_i = df.groupby("item").ts.min()
    t[first_u.index.values] = first_u.values
    t[n_users + first_i.index.values] = first_i.values
    assert np.isfinite(t).all()
    out = paths.graph_file(f"{dataset}_node_time.npy")
    np.save(out, t)
    print(f"{out}: {n} nodes, first interaction {t.min():.0f}..{t.max():.0f} s")


if __name__ == "__main__":
    main()
