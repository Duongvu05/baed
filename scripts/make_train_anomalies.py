"""Anomalous ego-graphs of the TRAIN split only, for training the diffusion model without leakage.

    uv run python scripts/make_train_anomalies.py [--src-tag _full] [--out-tag _fulltrain]

The released code trains the diffusion model on every anomalous ego-graph
(<dataset>_init_1_dataset.pkl), including those of test nodes. The paper says test
ego-graphs are excluded from training, so here we keep the anomalies of the first 70%
(the detector's sequential train split) of <dataset><src-tag>_init_0_1_dataset.pkl.
"""

import argparse
import pickle

from baed import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--src-tag", default="_full")
    ap.add_argument("--out-tag", default="_fulltrain")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    a = ap.parse_args()
    with open(paths.graph_file(f"{a.dataset}{a.src_tag}_init_0_1_dataset.pkl"), "rb") as f:
        gs = pickle.load(f)
    train = gs[: int(a.train_ratio * len(gs))]
    anom = [g for g in train if g.graph["sublabel"] == 1]
    out = paths.ensure(paths.graph_file(f"{a.dataset}{a.out_tag}_init_1_dataset.pkl"))
    with open(out, "wb") as f:
        pickle.dump(anom, f)
    print(f"{out}: {len(anom)} anomalous ego-graphs from the first {len(train)} of {len(gs)}")


if __name__ == "__main__":
    main()
