"""Describe the train / val / test ego-graphs of a dataset (uses the lean cache).

    uv run python scripts/inspect_split.py [--dataset elliptic] [--split sequential|random] [--seed 15]

For Elliptic the first node feature is the timestep, so the time span of each split
can be read directly. Leakage check: an ego-graph is fingerprinted by the multiset of
its node feature rows; overlap = test graphs whose fingerprint also occurs in train,
and test nodes (feature rows) that also appear inside some train ego-graph.
"""

import argparse
import collections
import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

from baed import paths
from baed.datasets.lean import load_lean


def fp_rows(feat):
    return [hashlib.md5(r.tobytes()).hexdigest() for r in feat.numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--split", default="sequential", choices=["sequential", "random"])
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()
    pkl = paths.graph_file(f"{a.dataset}_init_0_1_dataset.pkl")
    items = load_lean(pkl, lambda: (_ for _ in ()).throw(RuntimeError("build the lean cache first")), paths.root() / "data" / "cache")
    n = len(items)
    y = np.array([it[2] for it in items])
    if a.split == "random":
        idx = np.arange(n)
        tr, rest = train_test_split(idx, train_size=0.7, stratify=y, random_state=a.seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=a.seed)
    else:
        t, v = int(0.7 * n), int(0.15 * n)
        tr, va, te = np.arange(t), np.arange(t, t + v), np.arange(t + v, t + 2 * v)
    splits = {"train": tr, "val": va, "test": te}

    print(f"dataset={a.dataset} split={a.split} total={n} anomalies={y.sum()} ({y.mean():.2%})")
    for name, ix in splits.items():
        sizes = np.array([items[i][0].shape[0] for i in ix])
        ts = np.array([int(items[i][0][0, 0]) for i in ix]) if a.dataset == "elliptic" else None
        ev = sizes > 1
        line = (f"{name:5s} graphs={len(ix):6d} anom={y[ix].sum():5d} ({y[ix].mean():6.2%}) | evaluated(n>1)={ev.sum():6d} "
                f"anom={y[ix][ev].sum():5d} ({y[ix][ev].mean():6.2%}) | nodes mean={sizes.mean():.2f} max={sizes.max()}")
        if ts is not None:
            line += f" | timesteps {ts.min()}..{ts.max()}"
        print(line)

    if a.dataset == "elliptic":
        print("\nper-timestep (test):")
        c = collections.Counter()
        ca = collections.Counter()
        for i in te:
            t = int(items[i][0][0, 0])
            c[t] += 1
            ca[t] += int(y[i])
        print("  " + "  ".join(f"t{t}:{ca[t]}/{c[t]}" for t in sorted(c)))

    # leakage
    def graph_fp(i):
        return hashlib.md5("".join(sorted(fp_rows(items[i][0]))).encode()).hexdigest()

    train_graphs = {graph_fp(i) for i in tr}
    train_nodes = set()
    for i in tr:
        train_nodes.update(fp_rows(items[i][0]))
    dup = sum(graph_fp(i) in train_graphs for i in te)
    test_nodes = [r for i in te for r in fp_rows(items[i][0])]
    shared = sum(r in train_nodes for r in test_nodes)
    centers_like = sum(fp_rows(items[i][0])[0] in train_nodes for i in te)
    print(f"\nleakage: test ego-graphs identical to a train ego-graph: {dup}/{len(te)} ({dup / len(te):.1%})")
    print(f"         test node rows also inside train ego-graphs: {shared}/{len(test_nodes)} ({shared / max(len(test_nodes), 1):.1%})")
    print(f"         test graphs whose first node appears in train: {centers_like}/{len(te)} ({centers_like / len(te):.1%})")


if __name__ == "__main__":
    main()
