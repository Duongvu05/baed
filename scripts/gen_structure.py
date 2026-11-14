"""Structure of real vs generated fraud ego-graphs: is the generated center (node 0) where the real center is?

  python scripts/gen_structure.py --dataset reddit --data-tag _t --node-feats 64 \
      --encoder reddit_gin_relu_256_whole.pt --run reddit_t_train_gin_et_b32_lr1e4 --run reddit_t_train_gin_et_b32_lr1e4_cflag

Part 1, ego-graphs as the detector sees them (real ones, and center-fill samples cut at 1 and 2 hops around node 0):
nodes, edges, center degree, share of nodes adjacent to the center, share of edges not touching the center, farthest
node from the center. Part 2, the full generated graphs before the cut: share of perfect stars and share of graphs
whose highest-degree node is node 0.
"""
import argparse
import random

import numpy as np
import torch

from baed import paths
from baed.datasets.lean import keep, load_lean
from baed.pipelines import detect as D
from baed.pipelines.generate import GenerateDataArgs, GraphDataGenerator


def ego_stats(name, data):
    n, m, cdeg, cov, noncen, ecc = [], [], [], [], [], []
    for f, e in data:
        N = f.shape[0]
        e = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
        und = {(min(a, b), max(a, b)) for a, b in zip(e[0], e[1]) if a != b}
        n.append(N); m.append(len(und))
        nb = {b if a == 0 else a for a, b in und if 0 in (a, b)}
        cdeg.append(len(nb)); cov.append(len(nb) / max(N - 1, 1))
        noncen.append(sum(1 for a, b in und if 0 not in (a, b)) / max(len(und), 1))
        adj = {}
        for a, b in und:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        dist, fr = {0: 0}, [0]
        while fr:
            nxt = []
            for u in fr:
                for w in adj.get(u, ()):
                    if w not in dist:
                        dist[w] = dist[u] + 1; nxt.append(w)
            fr = nxt
        ecc.append(max(dist.values()))
    print(f"  {name:34s} nodes {np.mean(n):6.1f} | edges {np.mean(m):6.1f} | center degree {np.mean(cdeg):5.1f} | "
          f"nodes adjacent to center {np.mean(cov):.2f} | edges not touching center {np.mean(noncen):.2f} | "
          f"farthest node {np.mean(ecc):.2f} hops")


def full_stats(gen, emb, n):
    tmpl = gen.draw_templates(n)
    init = gen.initial_graph_sampler._generate_empty_data([k for k, _ in tmpl], [ft for _, ft in tmpl])
    with torch.no_grad():
        b = gen.model.sample(n, emb.reshape(1, -1).expand(n, -1), init_graph=init)
    ptr = b.ptr.cpu().numpy()
    ei = b.edge_index.cpu().numpy() if b.edge_index.numel() else np.zeros((2, 0), np.int64)
    gid = np.searchsorted(ptr, ei[0], side="right") - 1
    rows = []
    for g in range(n):
        s, t = ptr[g], ptr[g + 1]; N = t - s
        sel = gid == g
        und = {(min(a, c), max(a, c)) for a, c in zip(ei[0, sel] - s, ei[1, sel] - s) if a != c}
        deg = np.zeros(N, int)
        for a, c in und:
            deg[a] += 1; deg[c] += 1
        rows.append((N, len(und) == N - 1 and deg.max() == N - 1, deg.argmax() == 0, deg[0] / max(N - 1, 1)))
    r = np.array(rows, dtype=float)
    for name, sel in (("all", np.ones(len(r), bool)), (">= 4 nodes", r[:, 0] >= 4)):
        x = r[sel]
        print(f"  full generated graphs [{name}, {len(x)}]: perfect star {x[:, 1].mean():.2f} | hub is node 0 "
              f"{x[:, 2].mean():.2f} | node-0 degree share {x[:, 3].mean():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-tag", default="")
    p.add_argument("--node-feats", type=int, required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--run", action="append", required=True, help="diffusion run name (repeat to compare)")
    p.add_argument("--checkpoint", type=int, default=300)
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()

    torch.manual_seed(0); random.seed(0)
    D.device = torch.device("cuda:0")
    pkl = str(paths.graph_file(f"{a.dataset}{a.data_tag}_init_0_1_dataset.pkl"))
    items = load_lean(pkl, lambda: D.load_dataset(pkl), paths.root() / "data" / "cache")
    train, _, test = D.split_dataset(items)
    train = [it for it in train if keep(it, False)]
    tr_f = [(it[0], it[1]) for it in train if it[2] == 1]
    tr_n = [(it[0], it[1]) for it in train if it[2] == 0]
    te_f = [(it[0], it[1]) for it in test if keep(it, False) and it[2] == 1]
    gae = torch.load(str(paths.gae_weight(a.encoder)), map_location=D.device)
    with torch.no_grad():
        emb = D._guidance(gae, [x[0] for x in tr_f], [x[1] for x in tr_f], torch.ones(len(tr_f), dtype=torch.long), None)
    print("real ego-graphs")
    random.seed(1)
    ego_stats("real normal (train, 500)", random.sample(tr_n, min(500, len(tr_n))))
    ego_stats(f"real fraud (train, {len(tr_f)})", tr_f)
    ego_stats(f"real fraud (test, {len(te_f)})", te_f)
    for run in a.run:
        print(f"diffusion run {run}")
        gen = GraphDataGenerator(GenerateDataArgs(run_name=run, dataset=a.dataset, num_samples=64, seed=15,
                                                  checkpoint=a.checkpoint))
        print(f"  center_flag = {getattr(gen.model._denoise_fn, 'center_flag', False)}")
        for hops in (2, 1):
            D.GEN_HOPS, D.GEN_MAX_TRIES = hops, 50
            D.STATS.update(gen_rounds=0, resampled=0, template_replaced=0)
            gf, ge = D._center_fill(gen, a.n, a.node_feats, emb)
            ego_stats(f"generated, {hops}-hop cut ({a.n})", list(zip(gf, ge)))
        full_stats(gen, emb, a.n)


if __name__ == "__main__":
    main()
