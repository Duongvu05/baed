# BAED

Refactored implementation of **BAED - Balanced Anomaly-guided Ego-graph Diffusion Model for
Inductive Graph Anomaly Detection** (KDD 2026, [arXiv:2602.05232](https://arxiv.org/abs/2602.05232)).

Upstream code: [OaxKnud/BAED](https://github.com/OaxKnud/BAED) (commit `aa55a89`). This repo
reorganises it with the [cookiecutter-uv-python](https://github.com/Duongvu05/cookiecutter-uv-python)
layout, a modern CUDA 12.1 stack managed by `uv`, and a Rust kernel for the preprocessing hot loop.

## Layout

```text
baed/
├── src/baed/
│   ├── pipelines/            # one module per step, each exposes main() + a console script
│   │   ├── build_subgraphs.py    01  (was createSubGraphDataset/createDataset.py)
│   │   ├── gae_dataset.py        02a (was gae/saveGAEdataset.py)
│   │   ├── gae_train.py          02b (was gae/trainGAE.py)
│   │   ├── train.py              03  (was train.py)
│   │   ├── generate.py           04  (was generateData.py)
│   │   └── detect.py             05  (was trainSubgraphModel.py)
│   ├── graph_ops.py          # ego-graph extraction (Rust kernel + Python fallback)
│   ├── paths.py              # every data / weight / output path in one place
│   ├── datasets/ diffusion/ layers/ models/ gae/ eval_utils/   # model code, as released
│   └── model.py experiment.py utils.py
├── rust/baed-graph/          # PyO3 + rayon crate, built by maturin during `uv sync`
├── scripts/                  # build_dgl_wheel.py, download_data.sh, run_elliptic.sh, dgl_to_mat.py
├── tests/                    # equivalence tests vs. the original procedures
└── pyproject.toml
```

Data and outputs live under `$BAED_ROOT` (default: current directory):
`data/graphs/` (raw + ego-graph datasets), `data/gae/`, `data/generated/`, `weights/gae/`,
`outputs/runs/` (was `wandb/`).

## Install

Requires Linux x86_64, CUDA 12.1 driver, [uv](https://github.com/astral-sh/uv) and a Rust toolchain (`cargo`).

```bash
python3 scripts/build_dgl_wheel.py   # DGL 2.4.0+cu121 wheel from the official dglteam conda build
uv sync                              # torch 2.3.1+cu121, PyG 2.5.3, DGL, Rust kernel, ...
uv run python -m baed.main           # prints the steps and whether the Rust kernel is active
```

DGL no longer ships CUDA wheels on PyPI and `data.dgl.ai` is unreachable from our machines, so
`build_dgl_wheel.py` repacks the official conda package into `wheels/`.

## Run

```bash
bash scripts/download_data.sh elliptic     # also: dgraph tfinance reddit
bash scripts/run_elliptic.sh               # steps 01 -> 03, paper hyper-parameters (300 epochs)
uv run baed-generate --run-name <run> --dataset elliptic --checkpoint 300
uv run baed-detect --dataset elliptic --model GraphSAGE --gae-weight elliptic_gae_relu_256.pt --node-feats 166 --epochs 50
```

Every step has `--help`. Defaults reproduce the values hard-coded in the original scripts.

## Unified +BAED setting (Elliptic)

`MODEL=GCN SEEDS="15" bash scripts/run_baed_unified.sh` runs base and +BAED with the setting below.
Results go to `outputs/results/elliptic/unified/`.

| Part | Choice | Flag |
|---|---|---|
| Split | ego-graphs in time order, sequential 70 / 15 / 15 (as released) | `--split sequential` |
| Ego-graphs | out-edges, labelled nodes only, single-node ego-graphs dropped | default |
| Node features | timestep and the 93 local features, 1-hop aggregates recomputed on each ego-graph | `--agg-features recompute` |
| Generator | GIN-guided ego-graph diffusion, run `elliptic_fulltrain_gin_et_b32` | `--run-name`, `--checkpoint 300` |
| Synthetic node features | copied from the template ego-graph, no Gaussian noise | `--gen-noise 0` |
| Guidance weights | paper Eq. 17 and 18 | `--curriculum sigmoid` |
| Sampling order | per batch, as in Algorithm 3 | `--gen-chunk 1` (default) |

Why these choices differ from the paper text.

- **No Gaussian noise.** The paper adds N(0, 1) to the copied features. Real train fraud nodes lie
  about 0.09 apart (median nearest-neighbour distance), and the noise moves a synthetic node about 13
  away from every real node. 154 of the 166 feature dimensions have a fraud std below 1. The detector
  then learns to flag noisy points rather than fraud. Removing the noise raised val macro F1 of
  GCN +BAED from 0.666 to 0.727 (seed 15).
- **Recomputed aggregates.** Columns 94 to 165 of Elliptic are anonymised 1-hop statistics computed on
  the full graph. For a generated ego-graph they describe the neighbours of the copied template, not the
  generated edges. `--agg-features recompute` drops them and adds max, min, mean and std of the 93 local
  features over the ego-graph's own 1-hop neighbours (undirected), plus log(1 + degree). The same
  function runs on real and generated ego-graphs, so message passing and the aggregate features use
  the same edge set. Node features become 467-dimensional. The guidance encoder and the diffusion
  still see the 166 released features.
- **`--gen-chunk` stays at 1.** Algorithm 3 computes the loss of the current batch, builds the guidance,
  samples, and updates the detector before it moves to the next batch. K = 1 follows this order. With
  K > 1 the synthetic graphs of K batches are sampled in one call, which is 5 to 8 times faster (sampling is
  launch-bound, 92 graphs take as long as 400), but the guidance of all K batches is then computed before
  the detector is updated on any of them. K > 1 is kept only for quick checks. A K = 1 run of GCN takes
  about 75 min on an RTX 3060.

GCN, seed 15, test set of 4350 ego-graphs with 116 frauds. F1 is the fraud class F1.

| GCN | AUROC | AUPRC | F1 | Macro F1 | Val AUROC | Val Macro F1 |
|---|---|---|---|---|---|---|
| Base | 0.689 | 0.058 | 0.060 | 0.521 | 0.919 | 0.758 |
| +BAED, N(0, 1) noise | 0.653 | 0.040 | 0.009 | 0.492 | 0.897 | 0.666 |
| +BAED, no noise (K = 1) | 0.756 | 0.061 | 0.039 | 0.511 | 0.904 | 0.676 |
| +BAED, no noise (K = 16, approximate) | 0.700 | 0.051 | 0.012 | 0.497 | 0.908 | 0.727 |

The two no-noise runs differ by about 0.05 in AUROC and in val macro F1. K = 16 changes the order in which
random numbers are drawn, so this gap is mostly run-to-run variance. One seed is not enough to rank
methods that are closer than that.

## What changed vs. upstream

| Area | Upstream | Here |
|---|---|---|
| Imports | top-level (`from diffusion...`, `sys.path` hacks) | package `baed.*` |
| Paths | relative to the cwd (`../graphs`, `./wandb`), `/home/chao/AGGS/...` | `baed.paths` |
| Scripts | config hard-coded at module level | `main()` + argparse, console scripts |
| Env | Python 3.7, torch 1.8.1, dgl-cu101 0.6.1 | Python 3.11, torch 2.3.1+cu121, PyG 2.5.3, DGL 2.4.0+cu121 |
| Ego-graphs | networkx BFS per center | Rust CSR BFS, parallel over all cores |
| Elliptic loader | dense 203769 x 203769 DataFrame per timestep (~330 GB) | sparse edge list, same output |

### Hyper-parameters vs. the paper (Appendix D, Sec. 5.1.4)

| Setting | Paper | Upstream README / code | Here |
|---|---|---|---|
| Diffusion epochs | **300** | 2000 | 300 (`run_elliptic.sh`) |
| Diffusion dim / steps | 64 / 128 | 64 / 128 | same |
| LR / optimizer | 1e-4 / Adam | 1e-4 / Adam, constant (no milestones) | same |
| Noise schedule / dropout | linear / 0.1 | linear / 0.1 | same |
| Attention heads, degree | 8 heads, node degree | `--num_heads 8 8 8 8 1 --degree` | same |
| Batch size | not stated | 8 | 8 |
| Parametrization / loss | not stated | `xt_prescribed_st` / `vb_ce_xt_prescribred_st` (rejected by code) | `xt` / `vb_ce_xt` |
| Downstream epochs | 50 | 10 (`trainSubgraphModel.py`) | pass `--epochs 50` |
| Downstream LR | tuned in 1e-3 ... 1e-6 | 1e-4 | `--lr` |
| Split | 70 / 15 / 15 | 70 / 15 / 15 (sequential, no shuffle) | same |
| Seeds | 15, 42, 63, 87, 94 | 0 | `baed-train --seed` (detect has no seed option yet) |

LR is constant and early stopping is disabled in `diffusion/base.py`, so the epoch-300 checkpoint
of a longer run equals a 300-epoch run.

### Fixes needed to make the released code run

- `gae/saveGAEdataset.py` never wrote the file that `trainGAE.py` reads. `baed-gae-dataset`
  now applies `train_test_split_edges` and saves `data/gae/<dataset>_gae.pt`.
- `trainSubgraphModel.py` imported `generateData`, which ran the whole generation at import time.
- `README` calls `evaluate.py` for synthesis. That file does not exist. It is `generateData.py` (now `baed-generate`).
- `nx.from_scipy_sparse_matrix` (removed in networkx 3) -> `from_scipy_sparse_array`.
- The README training command uses `--parametrization xt_prescribed_st --loss_type vb_ce_xt_prescribred_st`
  with `--arch TGNN_embedding_guided`, which `get_model()` rejects (assertion). Those options belong to
  the degree-guided EDGE branch. `scripts/run_elliptic.sh` uses the consistent pair `xt` / `vb_ce_xt`.
- `baed-gae-dataset` drops `train_neg_adj_mask` (a dense N x N mask, 2 GB on Elliptic) that nothing reads.

### Behaviour kept as released (worth knowing)

- `gae_train`: `best_model = model` keeps a reference, so the *last* epoch is saved, not the best.
- `gae_dataset` (elliptic): per-timestep edge indices are concatenated without node offsets.
- Elliptic ego-graphs: a timestep is skipped when its highest-index labelled node has no edge
  (the DGL graph then has fewer nodes than the feature matrix).
- `datasets/data.py`: `eval_nx_graphs = nx_graphs[:int(0.2*l)]` overlaps the training split.

### Equivalence of the Rust path

`tests/test_graph_ops.py` checks that the Rust kernel, the Python fallback and the original
`dgl.to_networkx` + `nx.single_source_shortest_path_length` procedure give the same node sets,
edge sets and node attributes. Only the node iteration order inside each `nx.Graph` differs
(sorted ids instead of Python `set` order), i.e. a node permutation.
`tests/test_elliptic.py` checks the sparse Elliptic loader against the original dense code.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
cd rust/baed-graph && cargo test --release
```
