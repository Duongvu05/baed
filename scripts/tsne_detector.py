"""t-SNE of detector embeddings h = H[center] - mean(neighbours) (the MLP input of the deviation readout, paper Eq. 6).

Points: normal train graphs, known anomalies (train fraud), synthetic fraud from the diffusion model (center fill,
guidance = mean guidance embedding of the train fraud, exactly as in training), test fraud. One panel per detector
checkpoint. Also prints, in the embedding space itself (not t-SNE), the share of normal / train fraud / generated
among the 10 nearest neighbours of each test fraud graph.

Example (Reddit, time order):
  python scripts/tsne_detector.py --dataset reddit --data-tag _t --node-feats 64 \
      --run-name reddit_t_train_gin_et_b32_lr1e4 --encoder reddit_gin_relu_256_whole.pt \
      --panel "GCN base=outputs/detect_weights/reddit_GCN_None_GCN_None_s15_b_devcatnb_t.pth" \
      --panel "GCN + BAED (lr 1e-3)=outputs/detect_weights/reddit_GCN_BAED_GCN_BAED_s15_b_paug_cfill_devcatnb_t.pth" \
      --out reports/figures/tsne_gcn_baed_reddit_t
"""
import argparse
import os
import random

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from baed import paths  # noqa: E402
from baed.datasets.lean import keep, load_lean  # noqa: E402
from baed.pipelines import detect as D  # noqa: E402
from baed.pipelines.generate import GenerateDataArgs, GraphDataGenerator  # noqa: E402

STYLE = {  # same colours as the Elliptic figure
    "Normal": dict(c="#1f77b4", marker="o", s=10),
    "Known anomalies (train fraud)": dict(c="#d62728", marker="o", s=12),
    "Generated (ours)": dict(c="#ffb000", marker="o", s=12),
    "Test fraud": dict(c="#2ca02c", marker="^", s=18),
}


@torch.no_grad()
def embed(det, data):
    cap = []
    h = det.dev.mlp.register_forward_pre_hook(lambda m, inp: cap.append(inp[0].detach().cpu()))
    for i in range(0, len(data), 512):
        b = data[i:i + 512]
        bg, _, _ = D._batch([x[0].float() for x in b], [x[1] for x in b], torch.tensor([x[2] for x in b]))
        det(bg)
    h.remove()
    return torch.cat(cap).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-tag", default="")
    p.add_argument("--node-feats", type=int, required=True)
    p.add_argument("--run-name", required=True, help="diffusion run used for the synthetic fraud")
    p.add_argument("--checkpoint", type=int, default=300)
    p.add_argument("--encoder", required=True, help="guidance encoder in weights/gae/")
    p.add_argument("--readout", default="deviation_cat_nb")
    p.add_argument("--self-loop", action="store_true")
    p.add_argument("--panel", action="append", required=True, help='"title=path/to/detector.pth"')
    p.add_argument("--n-known", type=int, default=300)
    p.add_argument("--n-gen", type=int, default=300)
    p.add_argument("--n-normal", type=int, default=500)
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True, help="output path without extension (.png and .pdf are written)")
    a = p.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    D.device = torch.device("cuda:0"); D.AGG_MODE = "orig"
    pkl = str(paths.graph_file(f"{a.dataset}{a.data_tag}_init_0_1_dataset.pkl"))
    items = load_lean(pkl, lambda: D.load_dataset(pkl), paths.root() / "data" / "cache")
    train, _, test = D.split_dataset(items)
    train = [it for it in train if keep(it, False)]; test = [it for it in test if keep(it, False)]
    tr_f = [it for it in train if it[2] == 1]; tr_n = [it for it in train if it[2] == 0]
    te_f = [it for it in test if it[2] == 1]
    known = random.sample(tr_f, min(a.n_known, len(tr_f)))
    normal = random.sample(tr_n, min(a.n_normal, len(tr_n)))

    gen = GraphDataGenerator(GenerateDataArgs(run_name=a.run_name, dataset=a.dataset, num_samples=64, seed=15,
                                              checkpoint=a.checkpoint))
    gae = torch.load(str(paths.gae_weight(a.encoder)), map_location=D.device)
    with torch.no_grad():
        emb = D._guidance(gae, [x[0] for x in tr_f], [x[1] for x in tr_f], torch.ones(len(tr_f), dtype=torch.long), None)
    D.GEN_HOPS, D.GEN_MAX_TRIES = 2, 50
    D.STATS.update(gen_rounds=0, resampled=0, template_replaced=0)
    gf, ge = D._center_fill(gen, a.n_gen, a.node_feats, emb)
    synthetic = [(f, e, 1) for f, e in zip(gf, ge)]

    groups = [("Normal", normal), ("Known anomalies (train fraud)", known), ("Generated (ours)", synthetic),
              ("Test fraud", te_f)]
    size = {name: np.array([x[0].shape[0] for x in data]) for name, data in groups}
    print("mean nodes per graph:", {k: round(float(v.mean()), 1) for k, v in size.items()})

    panels = [s.split("=", 1) for s in a.panel]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False)
    for ax, (title, path) in zip(axes[0], panels):
        det = D.build_detector("GCN", a.node_feats, a.readout, self_loop=a.self_loop).to(D.device)
        det.load_state_dict(torch.load(path, map_location=D.device)); det.eval()
        Z = [embed(det, data) for _, data in groups]
        Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(np.concatenate(Z))
        o = 0
        for (name, data), z in zip(groups, Z):
            y = Y[o:o + len(z)]; o += len(z)
            ax.scatter(y[:, 0], y[:, 1], edgecolors="black", linewidths=0.3, alpha=0.85,
                       label=f"{name} ({len(z)})", **STYLE[name])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best", frameon=True)
        ref = np.concatenate(Z[:3]); lab = np.repeat([0, 1, 2], [len(z) for z in Z[:3]])
        idx = NearestNeighbors(n_neighbors=10).fit(ref).kneighbors(Z[3])[1]
        share = [(lab[idx] == k).mean() for k in range(3)]
        print(f"{title}: 10-NN of test fraud -> normal {share[0]:.2f}, train fraud {share[1]:.2f}, "
              f"generated {share[2]:.2f}")
    fig.suptitle(a.title or f"{a.dataset}{a.data_tag}: t-SNE of detector embeddings "
                 "$h = H[v_i] - \\mathrm{mean}(H[\\mathcal{N}(v_i)])$", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".png", dpi=200); fig.savefig(a.out + ".pdf")
    print("saved", a.out + ".png", "| center-fill resamples", D.STATS["resampled"])


if __name__ == "__main__":
    main()
