"""Convert a GADBench DGL graph file to the .mat layout read by BAED (Network / Attributes / Label)."""

import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from dgl.data.utils import load_graphs


def main(src: str, dst: str) -> None:
    g = load_graphs(src)[0][0]
    n = g.num_nodes()
    u, v = g.edges()
    adj = sp.csr_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(n, n))
    adj = ((adj + adj.T) > 0).astype(np.float64)  # undirected, like the original .mat files
    feat = g.ndata["feature"].numpy().astype(np.float64)
    label = g.ndata["label"].numpy().astype(np.int64).reshape(-1, 1)
    sio.savemat(dst, {"Network": adj, "Attributes": sp.csr_matrix(feat), "Label": label})
    print(f"{dst}: {n} nodes, {adj.nnz} directed edges, {feat.shape[1]} features, {int(label.sum())} anomalies")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
