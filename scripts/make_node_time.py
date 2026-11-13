"""Per-node appearance time for datasets whose GAD release dropped the timestamps.

    uv run python scripts/make_node_time.py reddit <path to JODIE reddit.csv>

Reddit: data/graphs/reddit.mat is the JODIE Reddit network (snap.stanford.edu/jodie/reddit.csv): node u is
user u (0..9,999) and node 10,000 + i is subreddit i (0..983); its 78,516 edges and 366 anomalies (banned
users) are exactly JODIE's unique user-subreddit pairs and banned users. A node's time is its first interaction,
the analogue of the Elliptic timestep. Writes data/graphs/reddit_node_time.npy (float, seconds).
"""
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

from baed import paths


def main():
    dataset, csv = sys.argv[1], sys.argv[2]
    assert dataset == "reddit", "only reddit is implemented"
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
