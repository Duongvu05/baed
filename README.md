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
bash scripts/run_elliptic.sh               # steps 01 -> 03 with the authors' hyper-parameters
uv run baed-generate --run-name <run> --dataset elliptic --checkpoint 2000
uv run baed-detect --dataset elliptic --model GraphSAGE --gae-weight elliptic_gae_relu_256.pt --node-feats 166
```

Every step has `--help`. Defaults reproduce the values hard-coded in the original scripts.

## What changed vs. upstream

| Area | Upstream | Here |
|---|---|---|
| Imports | top-level (`from diffusion...`, `sys.path` hacks) | package `baed.*` |
| Paths | relative to the cwd (`../graphs`, `./wandb`), `/home/chao/AGGS/...` | `baed.paths` |
| Scripts | config hard-coded at module level | `main()` + argparse, console scripts |
| Env | Python 3.7, torch 1.8.1, dgl-cu101 0.6.1 | Python 3.11, torch 2.3.1+cu121, PyG 2.5.3, DGL 2.4.0+cu121 |
| Ego-graphs | networkx BFS per center | Rust CSR BFS, parallel over all cores |
| Elliptic loader | dense 203769 x 203769 DataFrame per timestep (~330 GB) | sparse edge list, same output |

### Fixes needed to make the released code run

- `gae/saveGAEdataset.py` never wrote the file that `trainGAE.py` reads. `baed-gae-dataset`
  now applies `train_test_split_edges` and saves `data/gae/<dataset>_gae.pt`.
- `trainSubgraphModel.py` imported `generateData`, which ran the whole generation at import time.
- `README` calls `evaluate.py` for synthesis. That file does not exist. It is `generateData.py` (now `baed-generate`).
- `nx.from_scipy_sparse_matrix` (removed in networkx 3) -> `from_scipy_sparse_array`.

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
